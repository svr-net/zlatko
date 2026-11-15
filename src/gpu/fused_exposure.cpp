#include "ccr/gpu/fused_exposure.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "ccr/core/matrix.hpp"
#include "ccr/core/regression.hpp"
#include "ccr/instruments/asset_forward.hpp"
#include "ccr/instruments/bermudan_swaption.hpp"
#include "ccr/instruments/european_option.hpp"
#include "ccr/instruments/interest_rate_swap.hpp"

namespace ccr::gpu {

namespace {

std::uint32_t floatBits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

std::size_t assetIndex(const ScenarioGenerator& gen, const std::string& name) {
  for (std::size_t a = 0; a < gen.assets().size(); ++a)
    if (gen.assets()[a]->name() == name) return a;
  throw std::invalid_argument("unknown asset " + name);
}

using TermRecord = std::vector<double>;  // [kind, slot, c, A, B, p1, p2, p3, trade, bermudan, extra]

// Merges the valuation terms of one date in double precision, so the kernel evaluates one
// exponential per distinct cash-flow date instead of one per cash flow, and large coupons that
// cancel (a period's end and the next period's start, pay and receive legs) cancel exactly
// here rather than in 32-bit floats on the GPU:
//  - bonds with the same (A, B), i.e. the same maturity, are summed and stored as (c A) e^{-B x};
//  - a bond on the payment date of a running floating coupon is folded into that coupon's
//    fixed part p1, so the kernel prices (c fix + p1) A e^{-B x} with one exponential;
//  - forwards on the same asset are summed as one c S term;
//  - options stay as they are, AMC terms keep their order (exercise values, then policies, then
//    continuation and exercised tails), and fixings come last: they only ever feed later dates,
//    because a period is priced from its fixing strictly after the fixing date.
std::vector<TermRecord> mergeTerms(const std::vector<TermRecord>& in) {
  std::vector<TermRecord> floating, bonds, assets, options, fixings, exerciseBonds, amcTerms;
  auto sameBond = [](const TermRecord& a, const TermRecord& b) { return a[3] == b[3] && a[4] == b[4]; };
  for (const auto& r : in) {
    switch (static_cast<int>(r[0])) {
      case kBond: {
        auto it = std::find_if(bonds.begin(), bonds.end(), [&](const TermRecord& b) { return sameBond(b, r); });
        if (it == bonds.end()) bonds.push_back(r);
        else (*it)[2] += r[2];
        break;
      }
      case kFixedFloat: {
        auto it = std::find_if(floating.begin(), floating.end(),
                               [&](const TermRecord& f) { return f[1] == r[1] && sameBond(f, r); });
        if (it == floating.end()) floating.push_back(r);
        else (*it)[2] += r[2], (*it)[5] += r[5];
        break;
      }
      case kAsset: {
        auto it = std::find_if(assets.begin(), assets.end(), [&](const TermRecord& a) { return a[1] == r[1]; });
        const double cA = r[2] * r[3];
        if (it == assets.end()) assets.push_back({r[0], r[1], cA, 1.0, 0.0, 0.0, 0.0, 0.0});
        else (*it)[2] += cA;
        break;
      }
      case kSetFixing: fixings.push_back(r); break;
      case kOption: options.push_back(r); break;
      case kBermExercise: {  // the exercise value of one Bermudan: bonds of one maturity merged
        auto it = std::find_if(exerciseBonds.begin(), exerciseBonds.end(),
                               [&](const TermRecord& b) { return b[9] == r[9] && sameBond(b, r); });
        if (it == exerciseBonds.end()) exerciseBonds.push_back(r);
        else (*it)[2] += r[2];
        break;
      }
      default: amcTerms.push_back(r);  // policy, continuation and exercised-tail terms, in order
    }
  }
  std::vector<TermRecord> out;
  for (auto& f : floating) {
    auto it = std::find_if(bonds.begin(), bonds.end(), [&](const TermRecord& b) { return sameBond(b, f); });
    if (it != bonds.end()) f[5] += (*it)[2], bonds.erase(it);
    out.push_back(f);
  }
  for (auto& b : bonds) {
    if (b[2] == 0.0) continue;  // e.g. offsetting trades
    out.push_back({b[0], 0.0, b[2] * b[3], 1.0, b[4], 0.0, 0.0, 0.0});
  }
  for (auto& a : assets)
    if (a[2] != 0.0) out.push_back(a);
  out.insert(out.end(), options.begin(), options.end());
  // AMC: exercise values first, then the policies that read them, then continuation and tails.
  out.insert(out.end(), exerciseBonds.begin(), exerciseBonds.end());
  out.insert(out.end(), amcTerms.begin(), amcTerms.end());
  out.insert(out.end(), fixings.begin(), fixings.end());
  return out;
}

// Longstaff-Schwartz for the Bermudans of a plan, exactly as BermudanSwaption does it, on the
// paths of the kernel itself: a CPU pass records the state, the deflator and the exercise values,
// the regressions are fitted in double precision backwards over the exercise dates, and their
// standardisation and coefficients go into the policy and continuation terms.
void fitAmc(FusedPlan& plan) {
  const std::size_t nT = plan.numDates(), nP = plan.numPaths;
  for (auto& a : plan.amc) a.decideTerm.assign(a.exerciseDates.size(), std::numeric_limits<std::size_t>::max());
  for (std::size_t j = 0; j < nT; ++j)
    for (std::size_t t = plan.indices[j]; t < plan.indices[j + 1]; ++t) {
      const auto code = static_cast<std::uint32_t>(plan.terms[t * kTermStride]);
      const std::uint32_t kind = code & 15u, b = (code >> 8u) & 15u, extra = code >> 12u;
      if (kind == kBermDecide) plan.amc.at(b).decideTerm.at(extra) = t;
      if (kind == kBermContinue) plan.amc.at(b).continueTerm.at(j) = t;
    }
  const AmcStates st = runAmcStates(plan);
  auto write = [&](std::size_t term, const PolynomialRegression& reg) {
    float* r = &plan.terms[term * kTermStride];
    r[2] = static_cast<float>(reg.center());
    r[3] = static_cast<float>(1.0 / reg.scale());
    for (std::size_t d = 0; d < 4; ++d) r[4 + d] = d < reg.coefficients().size() ? static_cast<float>(reg.coefficients()[d]) : 0.0f;
  };
  std::size_t row = 0;
  std::vector<double> xs(nP), target(nP);
  std::vector<bool> mask(nP);
  for (const auto& a : plan.amc) {
    const std::size_t nE = a.exerciseDates.size();
    PolynomialRegression reg(a.degree);
    std::vector<double> realised(nP, 0.0);
    std::vector<std::vector<double>> strategy(nE + 1, std::vector<double>(nP, 0.0));
    for (std::size_t k = nE; k-- > 0;) {
      const std::size_t j = a.exerciseDates[k];
      const float* E = &st.exercise[(row + k) * nP];
      for (std::size_t p = 0; p < nP; ++p) {
        xs[p] = st.x[j * nP + p];
        target[p] = realised[p] / st.deflator[j * nP + p];
        mask[p] = E[p] > 0.0f;
      }
      reg.fit(xs, target, mask);
      write(a.decideTerm.at(k), reg);
      for (std::size_t p = 0; p < nP; ++p)
        if (E[p] > 0.0f && E[p] > reg.predict(xs[p])) realised[p] = st.deflator[j * nP + p] * E[p];
      strategy[k] = realised;
    }
    for (std::size_t j = 0; j < nT; ++j) {
      if (a.continueTerm[j] == std::numeric_limits<std::size_t>::max()) continue;
      const auto& s = strategy[a.nextExercise[j]];
      for (std::size_t p = 0; p < nP; ++p) {
        xs[p] = st.x[j * nP + p];
        target[p] = s[p] / st.deflator[j * nP + p];
      }
      reg.fit(xs, target);
      write(a.continueTerm[j], reg);
    }
    row += nE;
  }
}

}  // namespace

std::string limitation(const ExposureEngine& engine, const NettingSet& nettingSet, bool perTrade) {
  const ScenarioGenerator& gen = engine.generator();
  if (perTrade && nettingSet.trades.size() > kMaxTradeGroups)
    return "per-trade statistics on the GPU support at most 16 trades";
  std::size_t bermudans = 0;
  for (const auto& t : nettingSet.trades) {
    const auto* bs = dynamic_cast<const BermudanSwaption*>(t.get());
    if (!bs && !dynamic_cast<const InterestRateSwap*>(t.get()) && !dynamic_cast<const AssetForward*>(t.get()) &&
        !dynamic_cast<const EuropeanOption*>(t.get()))
      return t->id() + " is not a trade type the fused kernels price";
    if (bs) {
      if (++bermudans > kMaxBermudans) return "the fused kernels price at most 4 Bermudan swaptions";
      if (bs->exerciseTimes().size() > kMaxExercises) return t->id() + " has more than 64 exercise dates";
      if (bs->regressionDegree() > kMaxAmcDegree) return t->id() + ": the fused kernels regress with degree at most 3";
    }
  }
  if (gen.assets().size() > kMaxAssets) return "the GPU kernel supports at most 4 FX/equity assets";
  if (gen.credits().size() > kMaxCredits) return "the GPU kernel supports at most 2 credit curves";
  for (const auto& c : gen.credits())
    if (c->substeps() != gen.credits().front()->substeps()) return "the GPU kernel needs the same CIR sub-steps for all credits";
  std::size_t swaps = 0;
  for (const auto& t : nettingSet.trades)
    swaps += dynamic_cast<const InterestRateSwap*>(t.get()) || dynamic_cast<const BermudanSwaption*>(t.get()) ? 1 : 0;
  if (swaps > kMaxSwapSlots) return "the GPU kernel supports at most 32 swaps";
  std::size_t normals = gen.numFactors() + 1;
  for (const auto& c : gen.credits()) normals += c->extraNormals();
  if (normals > kMaxNormals) return "too many normals per step for the GPU kernel";
  return {};
}

FusedPlan compile(const ExposureEngine& engine, const NettingSet& nettingSet, double pfeQuantile,
                  std::size_t counterparty) {
  FusedOptions o;
  o.pfeQuantile = pfeQuantile;
  o.counterparty = counterparty;
  return compile(engine, nettingSet, o);
}

FusedPlan compile(const ExposureEngine& engine, const NettingSet& nettingSet, const FusedOptions& options) {
  const double pfeQuantile = options.pfeQuantile;
  const std::size_t counterparty = options.counterparty;
  if (options.perTrade && nettingSet.trades.size() > kMaxTradeGroups)
    throw std::invalid_argument("per-trade statistics on the GPU support at most 16 trades");
  if (options.imQuantile < 0.0 || options.imQuantile > 1.0) throw std::invalid_argument("imQuantile must be in [0, 1]");
  const ScenarioGenerator& gen = engine.generator();
  const auto& assets = gen.assets();
  const auto& credits = gen.credits();
  if (assets.size() > kMaxAssets) throw std::invalid_argument("GPU kernel supports at most 4 assets");
  if (credits.size() > kMaxCredits) throw std::invalid_argument("GPU kernel supports at most 2 credits");
  for (const auto& c : credits)
    if (c->substeps() != credits.front()->substeps())
      throw std::invalid_argument("GPU kernel needs the same CIR substeps for all credits");
  if (!credits.empty() && counterparty >= credits.size()) throw std::invalid_argument("counterparty index out of range");

  const TimeGrid reporting = engine.reportingGrid(nettingSet);
  const TimeGrid grid = engine.simulationGrid(nettingSet);
  const std::size_t nT = grid.size();
  const HullWhite1F& hw = *gen.rateModel();
  const double a = hw.meanReversion(), sigma = hw.volatility();

  FusedPlan plan;
  plan.numPaths = engine.config().numPaths;
  plan.times = grid.times();
  plan.numAssets = assets.size();
  plan.numCredits = credits.size();
  plan.counterparty = counterparty;
  plan.substeps = credits.empty() ? 1 : credits.front()->substeps();
  plan.seed = static_cast<std::uint32_t>(engine.config().seed % 4294967296ull);
  plan.antithetic = engine.config().antithetic;
  plan.hasCsa = nettingSet.collateral.has_value();
  plan.pfeQuantile = pfeQuantile;
  plan.imQuantile = options.imQuantile;
  plan.numTradeGroups = options.perTrade ? nettingSet.trades.size() : 0;
  plan.statsPerDate = (kStatsPerDate + kTradeStats * plan.numTradeGroups + 3) / 4 * 4;

  // Per-step deterministic coefficients (the exact transitions of ScenarioGenerator).
  plan.steps.assign(nT * kStepStride, 0.0f);
  for (std::size_t j = 1; j < nT; ++j) {
    const double t0 = grid[j - 1], t1 = grid[j], dt = t1 - t0, e = std::exp(-a * dt), s2 = sigma * sigma;
    const double varX = s2 * (1 - e * e) / (2 * a);
    const double varI = s2 / (a * a) * (dt - 2 * (1 - e) / a + (1 - e * e) / (2 * a));
    const double cov = s2 / (2 * a * a) * (1 - e) * (1 - e);
    const double sdX = std::sqrt(varX);
    const double c1 = sdX > 0 ? cov / sdX : 0.0;
    const double c2 = std::sqrt(std::max(varI - c1 * c1, 0.0));
    const double g2 = dt - 2 / a * (std::exp(-a * t0) - std::exp(-a * t1)) + 0.5 / a * (std::exp(-2 * a * t0) - std::exp(-2 * a * t1));
    const double intPhi = std::log(hw.curve().discount(t0) / hw.curve().discount(t1)) + s2 / (2 * a * a) * g2;
    float* row = &plan.steps[j * kStepStride];
    row[0] = static_cast<float>(e);
    row[1] = static_cast<float>(sdX);
    row[2] = static_cast<float>(c1);
    row[3] = static_cast<float>(c2);
    row[4] = static_cast<float>(intPhi);
    row[5] = static_cast<float>((1 - e) / a);
    row[6] = static_cast<float>(dt);
    for (std::size_t k = 0; k < assets.size(); ++k) {
      const auto& m = *assets[k];
      row[8 + k] = static_cast<float>(std::log(m.carryCurve().discount(t0) / m.carryCurve().discount(t1)));
      row[12 + k] = static_cast<float>(m.volatility() * std::sqrt(dt));
      row[16 + k] = static_cast<float>(0.5 * m.volatility() * m.volatility() * dt);
    }
    for (std::size_t c = 0; c < credits.size(); ++c) row[20 + c] = static_cast<float>(credits[c]->integratedShift(t1));
  }

  // Valuation terms per date: [kind, slot, c, A, B, p1, p2, p3] and the index of the trade.
  std::vector<std::vector<TermRecord>> termsAt(nT);
  std::size_t tradeIndex = 0;
  auto addTerm = [&](std::size_t j, TermKind kind, double slot, double c, double A, double B, double p1 = 0,
                     double p2 = 0, double p3 = 0) {
    termsAt[j].push_back({double(kind), slot, c, A, B, p1, p2, p3, double(tradeIndex), 0.0, 0.0});
  };
  auto addAmcTerm = [&](std::size_t j, TermKind kind, std::size_t bermudan, std::size_t extra, double slot, double c,
                        double A, double B, double p1 = 0) {
    termsAt[j].push_back({double(kind), slot, c, A, B, p1, 0.0, 0.0, double(tradeIndex), double(bermudan), double(extra)});
  };
  std::size_t bermudans = 0;
  auto bondA = [&](double t, double T) { return hw.zeroBond(t, T, 0.0); };
  auto bondB = [&](double t, double T) { return T <= t ? 0.0 : hw.B(t, T); };
  std::size_t slots = 0;
  for (; tradeIndex < nettingSet.trades.size(); ++tradeIndex) {
    const auto& trade = nettingSet.trades[tradeIndex];
    if (const auto* sw = dynamic_cast<const InterestRateSwap*>(trade.get())) {
      const double sign = sw->direction() == InterestRateSwap::Direction::PayFixed ? 1.0 : -1.0;
      const double n = sw->notional(), k = sw->fixedRate();
      const auto& sch = sw->schedule();
      if (slots >= kMaxSwapSlots) throw std::invalid_argument("too many swaps for the GPU kernel");
      const double slot = static_cast<double>(slots++);
      for (std::size_t i = 1; i < sch.size(); ++i) {
        const double ts = sch[i - 1], te = sch[i], tau = te - ts;
        // Record the fixing on the path at the last grid date on or before ts (as the C++ swap does).
        // Appended after the previous period's terms at that date, which still use the old fixing.
        const std::size_t kFix = grid.indexAtOrBefore(ts);
        // P(t, ts) / P(t, te) = (A_s / A_e) e^{-(B_s - B_e) x}: one exponential, ratio taken in double.
        addTerm(kFix, kSetFixing, slot, tau, bondA(grid[kFix], ts) / bondA(grid[kFix], te),
                bondB(grid[kFix], ts) - bondB(grid[kFix], te));
        for (std::size_t j = 0; j < nT; ++j) {
          const double t = grid[j];
          if (te <= t + 1e-10) continue;
          if (ts >= t - 1e-10) {
            addTerm(j, kBond, 0, sign * n, bondA(t, ts), bondB(t, ts));
            addTerm(j, kBond, 0, -sign * n * (1 + k * tau), bondA(t, te), bondB(t, te));
          } else {
            addTerm(j, kFixedFloat, slot, sign * n * tau, bondA(t, te), bondB(t, te));
            addTerm(j, kBond, 0, -sign * n * k * tau, bondA(t, te), bondB(t, te));
          }
        }
      }
    } else if (const auto* fw = dynamic_cast<const AssetForward*>(trade.get())) {
      const std::size_t ai = assetIndex(gen, fw->asset());
      const auto& m = *assets[ai];
      const double T = fw->maturity();
      for (std::size_t j = 0; j < nT; ++j) {
        const double tj = grid[j];
        if (tj >= T - 1e-10) continue;
        addTerm(j, kAsset, double(ai), fw->notional(), m.carryDiscount(tj, T), 0.0);
        addTerm(j, kBond, 0, -fw->notional() * fw->strike(), bondA(tj, T), bondB(tj, T));
      }
    } else if (const auto* op = dynamic_cast<const EuropeanOption*>(trade.get())) {
      const std::size_t ai = assetIndex(gen, op->asset());
      const auto& m = *assets[ai];
      const double T = op->expiry();
      for (std::size_t j = 0; j < nT; ++j) {
        const double tj = grid[j];
        if (tj >= T - 1e-10) continue;
        const double sd = m.volatility() * std::sqrt(T - tj);
        addTerm(j, kOption, double(ai), op->notional(), bondA(tj, T), bondB(tj, T), m.carryDiscount(tj, T), op->strike(),
                op->type() == OptionType::Call ? sd : -sd);
      }
    } else if (const auto* bs = dynamic_cast<const BermudanSwaption*>(trade.get())) {
      if (bermudans >= kMaxBermudans) throw std::invalid_argument("the GPU kernel prices at most 4 Bermudan swaptions");
      if (bs->exerciseTimes().size() > kMaxExercises) throw std::invalid_argument("too many exercise dates for the GPU kernel");
      if (bs->regressionDegree() > kMaxAmcDegree) throw std::invalid_argument("AMC regression degree above 3 on the GPU");
      const std::size_t b = bermudans++;
      const InterestRateSwap& u = bs->underlying();
      const double sign = u.direction() == InterestRateSwap::Direction::PayFixed ? 1.0 : -1.0;
      const double n = u.notional(), k = u.fixedRate(), pos = bs->position();
      const bool physical = bs->settlement() == Settlement::Physical;
      const auto& sch = u.schedule();
      const auto& ex = bs->exerciseTimes();
      const std::size_t nE = ex.size();
      AmcTerms info;
      info.degree = bs->regressionDegree();
      for (double e : ex) {
        const auto idx = grid.find(e, 1e-8);
        if (!idx) throw std::invalid_argument("Bermudan exercise date not on the simulation grid");
        info.exerciseDates.push_back(*idx);
      }
      // Period i belongs to the tail entered at exercise k iff e_k <= ts_i: the last such k.
      std::vector<long> lastExercise(sch.size() - 1, -1);
      for (std::size_t i = 1; i < sch.size(); ++i)
        for (std::size_t e = 0; e < nE; ++e)
          if (ex[e] <= sch[i - 1] + 1e-10) lastExercise[i - 1] = static_cast<long>(e);
      if (slots >= kMaxSwapSlots) throw std::invalid_argument("too many swaps for the GPU kernel");
      const double slot = static_cast<double>(slots++);
      if (physical)
        for (std::size_t i = 1; i < sch.size(); ++i) {
          if (lastExercise[i - 1] < 0) continue;
          const double ts = sch[i - 1], te = sch[i];
          const std::size_t kFix = grid.indexAtOrBefore(ts);
          addTerm(kFix, kSetFixing, slot, te - ts, bondA(grid[kFix], ts) / bondA(grid[kFix], te),
                  bondB(grid[kFix], ts) - bondB(grid[kFix], te));
        }
      info.continueTerm.assign(nT, std::numeric_limits<std::size_t>::max());
      for (std::size_t j = 0; j < nT; ++j) {
        const double t = grid[j];
        std::size_t next = 0;
        while (next < nE && ex[next] <= t + 1e-10) ++next;
        info.nextExercise.push_back(next);
        // At an exercise date: the value of the tail swap entered (no position), then the policy.
        for (std::size_t e = 0; e < nE; ++e) {
          if (info.exerciseDates[e] != j) continue;
          for (std::size_t i = 1; i < sch.size(); ++i) {
            const double ts = sch[i - 1], te = sch[i], tau = te - ts;
            if (ts < ex[e] - 1e-10) continue;
            addAmcTerm(j, kBermExercise, b, e, 0.0, sign * n, bondA(t, ts), bondB(t, ts));
            addAmcTerm(j, kBermExercise, b, e, 0.0, -sign * n * (1 + k * tau), bondA(t, te), bondB(t, te));
          }
          addAmcTerm(j, kBermDecide, b, e, 0.0, 0.0, 0.0, 0.0);  // coefficients fitted below
        }
        // Not exercised yet and an exercise still ahead: the regressed continuation value.
        if (next < nE) addAmcTerm(j, kBermContinue, b, 0, pos, 0.0, 0.0, 0.0);
        // Exercised (physical): the tail swap, period by period, as long as the period is alive.
        if (physical)
          for (std::size_t i = 1; i < sch.size(); ++i) {
            if (lastExercise[i - 1] < 0) continue;
            const double ts = sch[i - 1], te = sch[i], tau = te - ts;
            if (te <= t + 1e-10) continue;
            const auto kMax = static_cast<std::size_t>(lastExercise[i - 1]);
            if (ts >= t - 1e-10) {
              addAmcTerm(j, kBermBond, b, kMax, 0.0, pos * sign * n, bondA(t, ts), bondB(t, ts));
              addAmcTerm(j, kBermBond, b, kMax, 0.0, -pos * sign * n * (1 + k * tau), bondA(t, te), bondB(t, te));
            } else {
              addAmcTerm(j, kBermFloat, b, kMax, slot, pos * sign * n * tau, bondA(t, te), bondB(t, te), -pos * sign * n * k * tau);
            }
          }
      }
      plan.amc.push_back(std::move(info));
    } else {
      plan.unsupported.push_back(trade->id() + " is not a trade type the fused kernels price");
    }
  }

  // Index table: termStart[nT+1] | callIndex[nT] | reportingSlot[nT] (1 + rank among the reporting dates, 0 if not one).
  const std::size_t offCall = nT + 1, offReporting = offCall + nT;
  plan.indices.assign(offReporting + nT, 0u);
  for (std::size_t j = 0; j < nT; ++j) {
    plan.indices[j] = static_cast<std::uint32_t>(plan.terms.size() / kTermStride);
    // The first word packs kind, trade, bermudan and extra (see TermKind).
    auto emit = [&](const std::vector<TermRecord>& recs, std::size_t group) {
      for (const auto& rec : mergeTerms(recs)) {
        const double bermudan = rec.size() > 9 ? rec[9] : 0.0, extra = rec.size() > 10 ? rec[10] : 0.0;
        plan.terms.push_back(static_cast<float>(rec[0] + 16.0 * static_cast<double>(group) + 256.0 * bermudan + 4096.0 * extra));
        for (std::size_t k = 1; k < kTermStride; ++k) plan.terms.push_back(static_cast<float>(rec[k]));
      }
    };
    if (!plan.perTrade()) {
      emit(termsAt[j], 0);
    } else {
      // Terms are merged within each trade only, so every term belongs to one trade.
      for (std::size_t g = 0; g < plan.numTradeGroups; ++g) {
        std::vector<TermRecord> group;
        for (const auto& rec : termsAt[j])
          if (static_cast<std::size_t>(rec[8]) == g) group.push_back(rec);
        emit(group, g);
      }
    }
  }
  plan.indices[nT] = static_cast<std::uint32_t>(plan.terms.size() / kTermStride);
  const double mpr = plan.hasCsa ? nettingSet.collateral->marginPeriodOfRisk : 0.0;
  for (std::size_t j = 0; j < nT; ++j) {
    plan.indices[offCall + j] = static_cast<std::uint32_t>(grid.indexAtOrBefore(std::max(grid[j] - mpr, 0.0)));
    if (reporting.find(grid[j]).has_value()) {
      plan.reporting.push_back(j);
      plan.indices[offReporting + j] = static_cast<std::uint32_t>(plan.reporting.size());
    }
  }

  plan.numCorrelated = gen.numFactors();
  plan.numNormals = plan.numCorrelated + 1;
  for (const auto& c : credits) plan.numNormals += c->extraNormals();
  if (plan.numNormals > kMaxNormals) throw std::invalid_argument("too many normals per step for the GPU kernel");
  const Matrix l = gen.correlation().rows() ? cholesky(gen.correlation()) : Matrix::identity(plan.numCorrelated);
  plan.params.assign(kParamCount, 0.0f);
  for (std::size_t c = 0; c < credits.size(); ++c) {
    plan.params[4 * c + 0] = static_cast<float>(credits[c]->kappa());
    plan.params[4 * c + 1] = static_cast<float>(credits[c]->theta());
    plan.params[4 * c + 2] = static_cast<float>(credits[c]->xi());
    plan.params[4 * c + 3] = static_cast<float>(credits[c]->y0());
  }
  if (plan.hasCsa) {
    const CollateralAgreement& csa = *nettingSet.collateral;
    plan.params[8] = static_cast<float>(csa.thresholdCounterparty);
    plan.params[9] = static_cast<float>(std::isfinite(csa.thresholdOwn) ? csa.thresholdOwn : -1.0);
    plan.params[10] = static_cast<float>(csa.minimumTransferAmount);
    plan.params[11] = static_cast<float>(csa.independentAmount);
  }
  for (std::size_t k = 0; k < assets.size(); ++k) plan.params[12 + k] = static_cast<float>(assets[k]->spot());
  for (std::size_t i = 0; i < plan.numCorrelated; ++i)
    for (std::size_t k = 0; k <= i; ++k) plan.params[16 + i * kCholeskyStride + k] = static_cast<float>(l(i, k));

  for (double t : plan.times)
    plan.marketSurvival.push_back(credits.empty() ? 1.0 : credits[counterparty]->marketCurve().survival(t));

  plan.header = {static_cast<std::uint32_t>(plan.numPaths), static_cast<std::uint32_t>(nT),
                 static_cast<std::uint32_t>(plan.numAssets), static_cast<std::uint32_t>(plan.numCredits),
                 static_cast<std::uint32_t>(plan.substeps), static_cast<std::uint32_t>(plan.numNormals), plan.seed,
                 plan.antithetic ? 1u : 0u, plan.hasCsa ? 1u : 0u, static_cast<std::uint32_t>(counterparty),
                 static_cast<std::uint32_t>(kStepStride / 4), static_cast<std::uint32_t>(plan.numCorrelated),
                 static_cast<std::uint32_t>(offCall), static_cast<std::uint32_t>(offReporting),
                 floatBits(static_cast<float>(pfeQuantile)), static_cast<std::uint32_t>(plan.numWorkgroups()),
                 static_cast<std::uint32_t>(plan.statsPerDate), static_cast<std::uint32_t>(plan.reporting.size()),
                 static_cast<std::uint32_t>(plan.numTradeGroups), plan.hasInitialMargin() ? 1u : 0u,
                 floatBits(static_cast<float>(plan.imQuantile)), 0u, 0u, 0u};
  if (!plan.amc.empty()) fitAmc(plan);
  return plan;
}

FusedResult summarise(const FusedPlan& plan, const FusedOutput& output) {
  const std::size_t nT = plan.numDates();
  const std::size_t nS = plan.statsPerDate, nR = plan.reporting.size();
  if (output.sums.size() != nT * nS || output.pfe.size() != plan.quantileRows())
    throw std::invalid_argument("gpu::summarise: read-back does not match the plan");
  FusedResult r;
  ExposureProfile& p = r.profile;
  p.pfeQuantile = plan.pfeQuantile;
  auto stat = [&](std::size_t j, std::size_t f) { return static_cast<double>(output.sums[j * nS + f]); };
  r.trades.resize(plan.numTradeGroups);
  for (std::size_t k = 0; k < plan.reporting.size(); ++k) {
    const std::size_t j = plan.reporting[k];
    p.times.push_back(plan.times[j]);
    p.expectedValue.push_back(stat(j, 0));
    p.expectedExposure.push_back(stat(j, 1));
    p.expectedNegativeExposure.push_back(stat(j, 2));
    p.discountedExpectedExposure.push_back(stat(j, 3));
    p.discountedExpectedNegativeExposure.push_back(stat(j, 4));
    p.potentialFutureExposure.push_back(output.pfe[k]);
    r.cvaIncrements.push_back(stat(j, 5));
    r.meanSurvival.push_back(stat(j, 6));
    r.marketSurvival.push_back(plan.marketSurvival[j]);
    r.meanDeflator.push_back(stat(j, 7));
    if (plan.hasInitialMargin()) r.valueChangeQuantile.push_back(k ? output.pfe[nR + k] : 0.0);
    for (std::size_t i = 0; i < plan.numTradeGroups; ++i) {
      const std::size_t o = kStatsPerDate + kTradeStats * i;
      auto& t = r.trades[i];
      t.marginalDiscounted.push_back(stat(j, o));
      t.marginal.push_back(stat(j, o + 1));
      t.incrementalDiscounted.push_back(stat(j, o + 2));
      t.standaloneDiscounted.push_back(stat(j, o + 3));
      t.standalone.push_back(stat(j, o + 4));
    }
  }
  for (std::size_t k = 0; k < r.cvaIncrements.size(); ++k) {
    const double pd = k ? r.meanSurvival[k - 1] - r.meanSurvival[k] : 0.0;
    r.conditionalDiscountedEe.push_back(k && pd > 0 ? r.cvaIncrements[k] / pd : std::numeric_limits<double>::quiet_NaN());
  }
  if (plan.numCredits == 0) r.cvaIncrements.assign(r.cvaIncrements.size(), std::numeric_limits<double>::quiet_NaN());
  return r;
}

double FusedResult::pathwiseCva(double recovery) const {
  double sum = 0.0;
  for (double v : cvaIncrements) sum += v;
  return (1.0 - recovery) * sum;
}

}  // namespace ccr::gpu
