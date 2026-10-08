#include "ccr/gpu/fused_exposure.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "ccr/core/matrix.hpp"
#include "ccr/instruments/asset_forward.hpp"
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

}  // namespace

std::string limitation(const ExposureEngine& engine, const NettingSet& nettingSet) {
  const ScenarioGenerator& gen = engine.generator();
  for (const auto& t : nettingSet.trades)
    if (!dynamic_cast<const InterestRateSwap*>(t.get()) && !dynamic_cast<const AssetForward*>(t.get()) &&
        !dynamic_cast<const EuropeanOption*>(t.get()))
      return t->id() + " needs American Monte Carlo, which runs in WebAssembly only";
  if (gen.assets().size() > kMaxAssets) return "the GPU kernel supports at most 4 FX/equity assets";
  if (gen.credits().size() > kMaxCredits) return "the GPU kernel supports at most 2 credit curves";
  for (const auto& c : gen.credits())
    if (c->substeps() != gen.credits().front()->substeps()) return "the GPU kernel needs the same CIR sub-steps for all credits";
  std::size_t swaps = 0;
  for (const auto& t : nettingSet.trades) swaps += dynamic_cast<const InterestRateSwap*>(t.get()) ? 1 : 0;
  if (swaps > kMaxSwapSlots) return "the GPU kernel supports at most 32 swaps";
  std::size_t normals = gen.numFactors() + 1;
  for (const auto& c : gen.credits()) normals += c->extraNormals();
  if (normals > kMaxNormals) return "too many normals per step for the GPU kernel";
  return {};
}

FusedPlan compile(const ExposureEngine& engine, const NettingSet& nettingSet, double pfeQuantile,
                  std::size_t counterparty) {
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
      row[7 + 3 * k] = static_cast<float>(std::log(m.carryCurve().discount(t0) / m.carryCurve().discount(t1)));
      row[8 + 3 * k] = static_cast<float>(m.volatility() * std::sqrt(dt));
      row[9 + 3 * k] = static_cast<float>(0.5 * m.volatility() * m.volatility() * dt);
    }
    for (std::size_t c = 0; c < credits.size(); ++c)
      row[7 + 3 * kMaxAssets + c] = static_cast<float>(credits[c]->integratedShift(t1));
  }

  // Valuation terms per date: [kind, slot, c, A, B, p1, p2, p3].
  std::vector<std::vector<std::vector<double>>> termsAt(nT);
  auto addTerm = [&](std::size_t j, TermKind kind, double slot, double c, double A, double B, double p1 = 0,
                     double p2 = 0, double p3 = 0) { termsAt[j].push_back({double(kind), slot, c, A, B, p1, p2, p3}); };
  auto bondA = [&](double t, double T) { return hw.zeroBond(t, T, 0.0); };
  auto bondB = [&](double t, double T) { return T <= t ? 0.0 : hw.B(t, T); };
  std::size_t slots = 0;
  for (const auto& trade : nettingSet.trades) {
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
        addTerm(kFix, kSetFixing, slot, tau, bondA(grid[kFix], ts), bondB(grid[kFix], ts), bondA(grid[kFix], te),
                bondB(grid[kFix], te));
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
    } else {
      plan.unsupported.push_back(trade->id() + " needs American Monte Carlo, which runs in WebAssembly only");
    }
  }

  // Index table: termStart[nT+1] | callIndex[nT] | isReporting[nT].
  const std::size_t offCall = nT + 1, offReporting = offCall + nT;
  plan.indices.assign(offReporting + nT, 0u);
  for (std::size_t j = 0; j < nT; ++j) {
    plan.indices[j] = static_cast<std::uint32_t>(plan.terms.size() / kTermStride);
    for (const auto& rec : termsAt[j])
      for (double v : rec) plan.terms.push_back(static_cast<float>(v));
  }
  plan.indices[nT] = static_cast<std::uint32_t>(plan.terms.size() / kTermStride);
  const double mpr = plan.hasCsa ? nettingSet.collateral->marginPeriodOfRisk : 0.0;
  for (std::size_t j = 0; j < nT; ++j) {
    plan.indices[offCall + j] = static_cast<std::uint32_t>(grid.indexAtOrBefore(std::max(grid[j] - mpr, 0.0)));
    const bool isReporting = reporting.find(grid[j]).has_value();
    plan.indices[offReporting + j] = isReporting ? 1u : 0u;
    if (isReporting) plan.reporting.push_back(j);
  }

  plan.numCorrelated = gen.numFactors();
  plan.numNormals = plan.numCorrelated + 1;
  for (const auto& c : credits) plan.numNormals += c->extraNormals();
  if (plan.numNormals > kMaxNormals) throw std::invalid_argument("too many normals per step for the GPU kernel");
  const Matrix l = gen.correlation().rows() ? cholesky(gen.correlation()) : Matrix::identity(plan.numCorrelated);
  plan.params.assign(16 + kMaxFactors * kMaxFactors, 0.0f);
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
    for (std::size_t k = 0; k <= i; ++k) plan.params[16 + i * kMaxFactors + k] = static_cast<float>(l(i, k));

  for (double t : plan.times)
    plan.marketSurvival.push_back(credits.empty() ? 1.0 : credits[counterparty]->marketCurve().survival(t));

  plan.header = {static_cast<std::uint32_t>(plan.numPaths), static_cast<std::uint32_t>(nT),
                 static_cast<std::uint32_t>(plan.numAssets), static_cast<std::uint32_t>(plan.numCredits),
                 static_cast<std::uint32_t>(plan.substeps), static_cast<std::uint32_t>(plan.numNormals), plan.seed,
                 plan.antithetic ? 1u : 0u, plan.hasCsa ? 1u : 0u, static_cast<std::uint32_t>(counterparty),
                 static_cast<std::uint32_t>(kStepStride), static_cast<std::uint32_t>(plan.numCorrelated),
                 static_cast<std::uint32_t>(offCall), static_cast<std::uint32_t>(offReporting),
                 floatBits(static_cast<float>(pfeQuantile)), static_cast<std::uint32_t>(plan.numWorkgroups())};
  return plan;
}

FusedResult summarise(const FusedPlan& plan, const FusedOutput& output) {
  const std::size_t nT = plan.numDates();
  if (output.sums.size() != nT * kStatsPerDate || output.pfe.size() != nT)
    throw std::invalid_argument("gpu::summarise: read-back does not match the plan");
  FusedResult r;
  ExposureProfile& p = r.profile;
  p.pfeQuantile = plan.pfeQuantile;
  auto stat = [&](std::size_t j, std::size_t f) { return static_cast<double>(output.sums[j * kStatsPerDate + f]); };
  for (std::size_t j : plan.reporting) {
    p.times.push_back(plan.times[j]);
    p.expectedValue.push_back(stat(j, 0));
    p.expectedExposure.push_back(stat(j, 1));
    p.expectedNegativeExposure.push_back(stat(j, 2));
    p.discountedExpectedExposure.push_back(stat(j, 3));
    p.discountedExpectedNegativeExposure.push_back(stat(j, 4));
    p.potentialFutureExposure.push_back(output.pfe[j]);
    r.cvaIncrements.push_back(stat(j, 5));
    r.meanSurvival.push_back(stat(j, 6));
    r.marketSurvival.push_back(plan.marketSurvival[j]);
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
