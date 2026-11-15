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
  FusedOptions effective = options;
  if (options.gross || options.tradeViews) effective.perTrade = true;
  if (effective.perTrade && nettingSet.trades.size() > kMaxTradeGroups)
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
  TimeGrid grid = engine.simulationGrid(nettingSet);
  // Nested initial margin: the inner paths run from every reporting date t to t + MPR, so t + MPR
  // joins the simulation grid (its terms, step coefficients and fixings are then in the plan).
  if (options.imInnerPaths > 0) {
    if (options.imInnerPaths > kMaxInnerPaths || options.imInnerPaths < 2)
      throw std::invalid_argument("nested initial margin: 2 to 1024 inner paths");
    const double rank = options.imQuantile * static_cast<double>(options.imInnerPaths - 1);
    if (static_cast<double>(options.imInnerPaths - 1) - std::floor(rank) >= static_cast<double>(kInnerTop - 1))
      throw std::invalid_argument("nested initial margin: the quantile needs more inner paths (keeps the 16 largest)");
    std::vector<double> ahead;
    for (double t : reporting.times())
      if (t + options.imMarginPeriod <= grid.horizon() + 1e-10) ahead.push_back(t + options.imMarginPeriod);
    grid = grid.merged(ahead);
  }
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
  plan.numTradeGroups = effective.perTrade ? nettingSet.trades.size() : 0;
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

  // ---- Rows and row-statistics requests of the requested views.
  std::vector<std::size_t> reportingDates;
  for (std::size_t j = 0; j < nT; ++j)
    if (reporting.find(grid[j]).has_value()) reportingDates.push_back(j);
  const auto nR = static_cast<std::uint32_t>(reportingDates.size());
  const std::size_t nTr = plan.numTradeGroups;
  const std::size_t S = std::min(options.samples, std::min<std::size_t>(kMaxSamples, plan.numPaths));
  RowLayout& L = plan.rows;
  ViewOutputs& V = plan.views;
  std::uint32_t nextRow = 0;
  auto block = [&](std::uint32_t rows) { const std::uint32_t r = nextRow; nextRow += rows; return r; };
  L.exposure = block(nR);
  if (plan.hasInitialMargin()) L.valueChange = block(nR);
  const bool histogram = options.histogramTime >= 0.0;
  if (S > 0 || histogram) L.exposureValue = block(nR);
  if (S > 0) L.netted = block(nR);
  if (plan.hasCsa && (S > 0 || options.collateralBands)) L.collateral = block(nR);
  if (options.gross) L.gross = block(nR);
  if (options.tradeViews) L.trades = block(static_cast<std::uint32_t>(nTr * nR));
  if (options.exerciseViews && !plan.amc.empty()) L.exercise = block(static_cast<std::uint32_t>(plan.amc.size()));
  // Risk-factor probes: terms that write the state into a row on a date.
  auto probe = [&](std::size_t j, std::uint32_t row, int what, std::size_t index, double offset = 0.0, double A = 0.0, double B = 0.0) {
    termsAt[j].push_back({double(kProbe), double(what + 16 * static_cast<int>(index)), offset, A, B, double(row), 0.0, 0.0, 0.0, 0.0, 0.0});
  };
  plan.imInnerPaths = options.imInnerPaths;
  if (options.imInnerPaths > 0) L.conditionalIm = block(nR);
  std::uint32_t deflatorRows = kNoRow;
  if (options.gross || options.tradeViews || options.factorViews || options.imInnerPaths > 0) {
    deflatorRows = block(nR);
    for (std::uint32_t r = 0; r < nR; ++r) probe(reportingDates[r], deflatorRows + r, 1, 0);
  }
  struct FactorRows { std::uint32_t x = kNoRow, bond = kNoRow; std::vector<std::uint32_t> asset, deflatedAsset, intensity, survival, increments; } F;
  if (options.factorViews) {
    F.x = block(nR);
    F.bond = block(nR);
    for (std::uint32_t r = 0; r < nR; ++r) {
      const std::size_t j = reportingDates[r];
      const double T = std::max(options.bondMaturity, grid[j]);
      probe(j, F.x + r, 0, 0);
      probe(j, F.bond + r, 7, 0, 0.0, bondA(grid[j], T), bondB(grid[j], T));
    }
    for (std::size_t ai = 0; ai < assets.size(); ++ai) {
      F.asset.push_back(block(nR));
      F.deflatedAsset.push_back(block(nR));
      for (std::uint32_t r = 0; r < nR; ++r) {
        probe(reportingDates[r], F.asset.back() + r, 2, ai);
        probe(reportingDates[r], F.deflatedAsset.back() + r, 6, ai);
      }
    }
    for (std::size_t c = 0; c < credits.size(); ++c) {
      F.intensity.push_back(block(nR));
      F.survival.push_back(block(nR));
      for (std::uint32_t r = 0; r < nR; ++r) {
        probe(reportingDates[r], F.intensity.back() + r, 4, c);
        probe(reportingDates[r], F.survival.back() + r, 5, c);
      }
    }
    if (nT > 1) {  // first-step increments: x(t1) - 0, ln S(t1) / S0, y(t1) - y0
      F.increments.push_back(block(1));
      probe(1, F.increments.back(), 0, 0);
      for (std::size_t ai = 0; ai < assets.size(); ++ai) F.increments.push_back(block(1)), probe(1, F.increments.back(), 3, ai);
      for (std::size_t c = 0; c < credits.size(); ++c)
        F.increments.push_back(block(1)), probe(1, F.increments.back(), 4, c, -credits[c]->y0());
    }
  }
  L.total = nextRow;

  auto request = [&](RowOp op, std::uint32_t row, std::uint32_t count = 1, double level = 0.0, std::uint32_t row2 = 0,
                     double lo = 0.0, double hi = 0.0) {
    const std::size_t out = plan.numOutputs;
    for (double v : {double(op), double(row), double(row2), double(count), level, lo, hi, double(out)})
      plan.requests.push_back(static_cast<float>(v));
    plan.numOutputs += op == kRowHistogram ? count + 2 : count;
    return out;
  };
  auto perDate = [&](RowOp op, std::uint32_t base, std::uint32_t count = 1, double level = 0.0, std::uint32_t row2Base = 0) {
    std::size_t first = ViewOutputs::none;
    for (std::uint32_t r = 0; r < nR; ++r) {
      const std::size_t o = request(op, base + r, count, level, row2Base == 0 ? 0 : row2Base + r);
      if (r == 0) first = o;
    }
    return first;
  };
  auto profileOf = [&](std::uint32_t base) {
    ProfileOutputs o;
    o.mean = perDate(kRowMean, base);
    o.positive = perDate(kRowMeanPos, base);
    o.negative = perDate(kRowMeanNeg, base);
    o.pfe = perDate(kRowQuantilePos, base, 1, pfeQuantile);
    o.positiveDiscounted = perDate(kRowMeanPosTimes, base, 1, 0.0, deflatorRows);
    o.negativeDiscounted = perDate(kRowMeanNegTimes, base, 1, 0.0, deflatorRows);
    return o;
  };
  V.pfe = perDate(kRowQuantile, L.exposure, 1, pfeQuantile);
  if (plan.hasInitialMargin()) V.valueChange = perDate(kRowQuantile, L.valueChange, 1, options.imQuantile);
  V.samples = S;
  if (S > 0) {
    V.exposureSamples = perDate(kRowSamples, L.exposureValue, static_cast<std::uint32_t>(S));
    V.nettedSamples = perDate(kRowSamples, L.netted, static_cast<std::uint32_t>(S));
    if (L.collateral != kNoRow) V.collateralSamples = perDate(kRowSamples, L.collateral, static_cast<std::uint32_t>(S));
  }
  if (L.conditionalIm != kNoRow) {
    V.conditionalIm = perDate(kRowMeanProduct, L.conditionalIm, 1, 0.0, deflatorRows);
    V.conditionalImMean = perDate(kRowMean, L.conditionalIm);
    for (double q : {0.05, 0.5, 0.95}) V.conditionalImBands.push_back(perDate(kRowQuantile, L.conditionalIm, 1, q));
  }
  if (options.collateralBands && L.collateral != kNoRow)
    for (double q : options.bandLevels) V.collateralBands.push_back(perDate(kRowQuantile, L.collateral, 1, q));
  if (options.gross) {
    V.gross = true;
    V.grossProfile = profileOf(L.gross);
  }
  if (histogram) {
    std::size_t rank = 0;
    for (std::size_t r = 0; r < nR; ++r)
      if (grid[reportingDates[r]] <= options.histogramTime + 1e-10) rank = r;
    V.histogramRank = rank;
    V.histogramBins = std::max<std::size_t>(1, std::min<std::size_t>(options.histogramBins, kPfeBins));
    V.histogram = request(kRowHistogram, L.exposureValue + static_cast<std::uint32_t>(rank), static_cast<std::uint32_t>(V.histogramBins));
  }
  if (options.tradeViews)
    for (std::size_t i = 0; i < nTr; ++i) {
      const auto base = L.trades + static_cast<std::uint32_t>(i * nR);
      ViewOutputs::Trade t;
      t.profile = profileOf(base);
      if (S > 0) t.samples = perDate(kRowSamples, base, static_cast<std::uint32_t>(S));
      for (double q : options.bandLevels) t.bands.push_back(perDate(kRowQuantile, base, 1, q));
      t.meanDeflated = perDate(kRowMeanProduct, base, 1, 0.0, deflatorRows);
      V.trades.push_back(std::move(t));
    }
  if (L.exercise != kNoRow)
    for (std::size_t b = 0; b < plan.amc.size(); ++b) {
      const auto row = L.exercise + static_cast<std::uint32_t>(b);
      const double nE = static_cast<double>(plan.amc[b].exerciseDates.size());
      V.exerciseHistogram.push_back(request(kRowHistogram, row, static_cast<std::uint32_t>(nE + 1), 0.0, 0, -1.5, nE - 0.5));
      V.exerciseSamples.push_back(S > 0 ? request(kRowSamples, row, static_cast<std::uint32_t>(S)) : ViewOutputs::none);
    }
  if (options.factorViews) {
    V.factors = true;
    if (S > 0) V.rateSamples = perDate(kRowSamples, F.x, static_cast<std::uint32_t>(S));
    for (double q : options.factorBandLevels) V.rateBands.push_back(perDate(kRowQuantile, F.x, 1, q));
    V.rateMean = perDate(kRowMean, F.x);
    V.deflatorMean = perDate(kRowMean, deflatorRows);
    V.bondMean = perDate(kRowMean, F.bond);
    for (std::size_t ai = 0; ai < assets.size(); ++ai) {
      ViewOutputs::Asset o;
      if (S > 0) o.samples = perDate(kRowSamples, F.asset[ai], static_cast<std::uint32_t>(S));
      for (double q : options.factorBandLevels) o.bands.push_back(perDate(kRowQuantile, F.asset[ai], 1, q));
      o.mean = perDate(kRowMean, F.asset[ai]);
      o.deflatedMean = perDate(kRowMean, F.deflatedAsset[ai]);
      V.assets.push_back(std::move(o));
    }
    for (std::size_t c = 0; c < credits.size(); ++c) {
      ViewOutputs::Credit o;
      if (S > 0) {
        o.intensitySamples = perDate(kRowSamples, F.intensity[c], static_cast<std::uint32_t>(S));
        o.survivalSamples = perDate(kRowSamples, F.survival[c], static_cast<std::uint32_t>(S));
      }
      for (double q : options.factorBandLevels) o.intensityBands.push_back(perDate(kRowQuantile, F.intensity[c], 1, q));
      o.survivalMean = perDate(kRowMean, F.survival[c]);
      V.credits.push_back(std::move(o));
    }
    V.increments = F.increments.size();
    if (!F.increments.empty()) {
      V.incrementMeans = plan.numOutputs;
      for (auto row : F.increments) request(kRowMean, row);
      V.incrementProducts = plan.numOutputs;
      for (auto ra : F.increments)
        for (auto rb : F.increments) request(kRowMeanProduct, ra, 1, 0.0, rb);
    }
  }

  // Index table: termStart[nT+1] | callIndex[nT] | reportingSlot[nT] (1 + rank among the reporting dates, 0 if not one).
  const std::size_t offCall = nT + 1, offReporting = offCall + nT, offImTarget = offReporting + nT;
  plan.indices.assign(offImTarget + nT, kNoRow);
  for (std::size_t j = 0; j < offImTarget; ++j) plan.indices[j] = 0u;
  if (options.imInnerPaths > 0)
    for (std::size_t r = 0; r < reportingDates.size(); ++r)
      if (const auto target = grid.find(grid[reportingDates[r]] + options.imMarginPeriod, 1e-8))
        plan.indices[offImTarget + reportingDates[r]] = static_cast<std::uint32_t>(*target);
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
    // Securities collateral: its type, asset and haircut; a bond's A and B go in the step table.
    const std::size_t o = 16 + kCholeskyStride * kCholeskyStride;
    plan.params[o] = static_cast<float>(static_cast<int>(csa.type));
    plan.params[o + 1] = static_cast<float>(csa.assetIndex);
    plan.params[o + 2] = static_cast<float>(csa.type == CollateralAgreement::Type::Cash ? 0.0 : csa.haircut);
    plan.securitiesCollateral = csa.type != CollateralAgreement::Type::Cash;
    if (csa.type == CollateralAgreement::Type::Asset && csa.assetIndex >= assets.size())
      throw std::invalid_argument("collateral asset index out of range");
    if (csa.type == CollateralAgreement::Type::Bond) {
      if (csa.bondMaturity < grid.horizon() - 1e-10) throw std::invalid_argument("the collateral bond must mature after the horizon");
      for (std::size_t j = 0; j < nT; ++j) {
        plan.steps[j * kStepStride + 22] = static_cast<float>(bondA(grid[j], csa.bondMaturity));
        plan.steps[j * kStepStride + 23] = static_cast<float>(bondB(grid[j], csa.bondMaturity));
      }
    }
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
                 static_cast<std::uint32_t>(plan.numTradeGroups), plan.rows.valueChange, plan.rows.exposureValue,
                 plan.rows.netted, plan.rows.collateral, plan.rows.gross, plan.rows.trades, plan.rows.exercise,
                 static_cast<std::uint32_t>(plan.numRequests()), static_cast<std::uint32_t>(plan.amc.size()),
                 plan.rows.exposure, plan.rows.conditionalIm, static_cast<std::uint32_t>(plan.imInnerPaths),
                 floatBits(static_cast<float>(options.imQuantile))};
  if (!plan.amc.empty()) fitAmc(plan);
  return plan;
}

namespace {

// The path views of a plan from the outputs of the row-statistics kernel.
void addViews(const FusedPlan& plan, const std::vector<float>& out, FusedResult& r) {
  const ViewOutputs& V = plan.views;
  const std::size_t nR = plan.reporting.size(), S = V.samples;
  const auto& times = r.profile.times;
  auto series = [&](std::size_t o) {
    std::vector<double> v(nR);
    for (std::size_t k = 0; k < nR; ++k) v[k] = out[o + k];
    return v;
  };
  auto samples = [&](std::size_t o) {  // outputs [date][path] -> [path][date]
    std::vector<std::vector<double>> v(S, std::vector<double>(nR));
    for (std::size_t k = 0; k < nR; ++k)
      for (std::size_t q = 0; q < S; ++q) v[q][k] = out[o + k * S + q];
    return v;
  };
  auto bands = [&](const std::vector<std::size_t>& offsets) {
    std::vector<std::vector<double>> v;
    for (std::size_t o : offsets) v.push_back(series(o));
    return v;
  };
  auto profile = [&](const ProfileOutputs& o) {
    ExposureProfile p;
    p.pfeQuantile = plan.pfeQuantile;
    p.times = times;
    p.expectedValue = series(o.mean);
    p.expectedExposure = series(o.positive);
    p.expectedNegativeExposure = series(o.negative);
    p.potentialFutureExposure = series(o.pfe);
    p.discountedExpectedExposure = series(o.positiveDiscounted);
    p.discountedExpectedNegativeExposure = series(o.negativeDiscounted);
    return p;
  };
  constexpr std::size_t none = ViewOutputs::none;
  if (V.conditionalIm != none) {
    r.conditionalImDiscounted = series(V.conditionalIm);
    r.conditionalImMean = series(V.conditionalImMean);
    r.conditionalImBands = bands(V.conditionalImBands);
  }
  if (V.exposureSamples != none) r.exposureSamples = samples(V.exposureSamples);
  if (V.nettedSamples != none) r.nettedSamples = samples(V.nettedSamples);
  if (V.collateralSamples != none) r.collateralSamples = samples(V.collateralSamples);
  r.collateralBands = bands(V.collateralBands);
  if (V.gross) r.grossProfile = profile(V.grossProfile);
  if (V.histogram != none) {
    const std::size_t nB = V.histogramBins;
    const double lo = out[V.histogram + nB], hi = out[V.histogram + nB + 1];
    r.histogramTime = times.at(V.histogramRank);
    for (std::size_t b = 0; b <= nB; ++b) r.histogramEdges.push_back(lo + (hi - lo) * static_cast<double>(b) / static_cast<double>(nB));
    for (std::size_t b = 0; b < nB; ++b) r.histogramCounts.push_back(out[V.histogram + b] / static_cast<double>(plan.numPaths));
  }
  for (const auto& t : V.trades) {
    FusedResult::TradeView v;
    v.profile = profile(t.profile);
    if (t.samples != none) v.samples = samples(t.samples);
    v.bands = bands(t.bands);
    v.meanDeflated = series(t.meanDeflated);
    r.tradeViews.push_back(std::move(v));
  }
  for (std::size_t b = 0; b < V.exerciseHistogram.size(); ++b) {
    FusedResult::ExerciseView e;
    const std::size_t nE = plan.amc[b].exerciseDates.size();
    for (std::size_t k = 1; k <= nE; ++k) e.distribution.push_back(out[V.exerciseHistogram[b] + k] / static_cast<double>(plan.numPaths));
    e.distribution.push_back(out[V.exerciseHistogram[b]] / static_cast<double>(plan.numPaths));  // never exercised (-1)
    if (V.exerciseSamples[b] != none)
      for (std::size_t q = 0; q < S; ++q) e.sampleIndex.push_back(out[V.exerciseSamples[b] + q]);
    r.exercise.push_back(std::move(e));
  }
  if (V.factors) {
    FusedResult::FactorView f;
    if (V.rateSamples != none) f.rateSamples = samples(V.rateSamples);
    f.rateBands = bands(V.rateBands);
    f.rateMean = series(V.rateMean);
    f.deflatorMean = series(V.deflatorMean);
    f.bondMean = series(V.bondMean);
    for (const auto& a : V.assets) {
      FusedResult::FactorView::Asset o;
      if (a.samples != none) o.samples = samples(a.samples);
      o.bands = bands(a.bands);
      o.mean = series(a.mean);
      o.deflatedMean = series(a.deflatedMean);
      f.assets.push_back(std::move(o));
    }
    for (const auto& c : V.credits) {
      FusedResult::FactorView::Credit o;
      if (c.intensitySamples != none) o.intensitySamples = samples(c.intensitySamples), o.survivalSamples = samples(c.survivalSamples);
      o.intensityBands = bands(c.intensityBands);
      o.survivalMean = series(c.survivalMean);
      f.credits.push_back(std::move(o));
    }
    const std::size_t n = V.increments;
    f.incrementCorrelation = Matrix(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t k = 0; k < n; ++k) {
        auto m = [&](std::size_t a) { return static_cast<double>(out[V.incrementMeans + a]); };
        auto e = [&](std::size_t a, std::size_t b) { return static_cast<double>(out[V.incrementProducts + a * n + b]); };
        const double cov = e(i, k) - m(i) * m(k), vi = e(i, i) - m(i) * m(i), vk = e(k, k) - m(k) * m(k);
        f.incrementCorrelation(i, k) = vi > 0 && vk > 0 ? cov / std::sqrt(vi * vk) : (i == k ? 1.0 : 0.0);
      }
    r.factors = std::move(f);
  }
}

}  // namespace

FusedResult summarise(const FusedPlan& plan, const FusedOutput& output) {
  const std::size_t nT = plan.numDates();
  const std::size_t nS = plan.statsPerDate;
  if (output.sums.size() != nT * nS || output.values.size() < plan.numOutputs)
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
    p.potentialFutureExposure.push_back(output.values[plan.views.pfe + k]);
    r.cvaIncrements.push_back(stat(j, 5));
    r.meanSurvival.push_back(stat(j, 6));
    r.marketSurvival.push_back(plan.marketSurvival[j]);
    r.meanDeflator.push_back(stat(j, 7));
    if (plan.hasInitialMargin()) r.valueChangeQuantile.push_back(k ? output.values[plan.views.valueChange + k] : 0.0);
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
  addViews(plan, output.values, r);
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
