#include <algorithm>
#include <cmath>
#include <memory>

#include "ccr/core/math.hpp"
#include "ccr/cva/cva.hpp"
#include "ccr/exposure/exposure_engine.hpp"
#include "ccr/gpu/fused_exposure.hpp"
#include "ccr/instruments/asset_forward.hpp"
#include "ccr/instruments/bermudan_swaption.hpp"
#include "ccr/instruments/european_option.hpp"
#include "ccr/instruments/interest_rate_swap.hpp"
#include "test_framework.hpp"

using namespace ccr;

namespace {

struct GpuSetup {
  ExposureEngine engine;
  NettingSet nettingSet;
  std::shared_ptr<const CreditCurve> market;
};

GpuSetup gpuSetup(std::size_t paths, bool withCsa) {
  const auto yc = std::make_shared<YieldCurve>(std::vector<double>{1.0, 10.0}, std::vector<double>{0.02, 0.03});
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto market = std::make_shared<CreditCurve>(CreditCurve::flat(0.02));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 0.01);
  const auto fx = std::make_shared<LognormalAsset>("FX", 1.1, 0.12, foreign);
  const auto cir = std::make_shared<CirIntensity>("C", market, 0.4, 0.02, 0.1, 0.02, 4);
  Matrix corr = Matrix::identity(3);
  corr(0, 1) = corr(1, 0) = 0.2;
  NettingSet ns;
  ns.id = "cpty";
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("pay", 1e6, 0.028, 0.0, 5.0, 2, InterestRateSwap::Direction::PayFixed)));
  ns.trades.push_back(std::make_shared<AssetForward>("fwd", "FX", 5e5, 1.12, 3.0));
  ns.trades.push_back(std::make_shared<EuropeanOption>("call", "FX", OptionType::Call, -4e5, 1.15, 2.0));
  if (withCsa) {
    CollateralAgreement csa;
    csa.thresholdCounterparty = 2e4;
    csa.thresholdOwn = 2e4;
    csa.minimumTransferAmount = 5e3;
    ns.collateral = csa;
  }
  return {ExposureEngine(ScenarioGenerator(hw, {fx}, {cir}, corr), {paths, 7, true}, TimeGrid::standardExposureGrid(5.0)),
          std::move(ns), market};
}

double maxRelDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double scale = 1e-12, diff = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    scale = std::max(scale, std::fabs(b[i]));
    diff = std::max(diff, std::fabs(a[i] - b[i]));
  }
  return diff / scale;
}

}  // namespace

TEST(gpu_plan_layout) {
  const auto s = gpuSetup(1000, true);
  const auto plan = gpu::compile(s.engine, s.nettingSet, 0.95, 0);
  const std::size_t nT = plan.numDates();
  CHECK(plan.unsupported.empty());
  CHECK(plan.header.size() == gpu::kHeaderWords);
  CHECK(plan.header[0] == 1000u && plan.header[1] == nT && plan.header[15] == plan.numWorkgroups());
  CHECK(plan.indices.size() == 3 * nT + 1);
  CHECK(plan.indices[nT] == plan.numTerms());
  CHECK(plan.steps.size() == nT * gpu::kStepStride);
  CHECK(plan.reporting.size() == s.engine.reportingGrid(s.nettingSet).size());
  CHECK(plan.numNormals == plan.numCorrelated + 1 + 3);  // 3 extra Brownian-bridge normals for 4 sub-steps
  CHECK(gpu::limitation(s.engine, s.nettingSet).empty());
  CHECK(gpu::fusedExposureKernel().find("fn main") != std::string::npos);
  CHECK(gpu::pfeQuantileKernel().find("3072") != std::string::npos);
  // Netted values are kept only for the CSA look-back, exposures only on the reporting dates.
  CHECK(plan.valsBytes() == 1000 * nT * 4);
  CHECK(plan.expoBytes() == 1000 * plan.reporting.size() * 4);
  CHECK(plan.pfeDispatch() == plan.reporting.size());
  CHECK(gpu::compile(gpuSetup(1000, false).engine, gpuSetup(1000, false).nettingSet).valsBytes() == 16);

  // A Bermudan needs AMC: reported, left out of the plan.
  NettingSet withBermudan = s.nettingSet;
  const auto underlying = InterestRateSwap::vanilla("u", 1e6, 0.03, 1.0, 4.0, 1, InterestRateSwap::Direction::ReceiveFixed);
  withBermudan.trades.push_back(std::make_shared<BermudanSwaption>("berm", underlying, std::vector<double>{1.0, 2.0, 3.0}));
  CHECK(!gpu::limitation(s.engine, withBermudan).empty());
  CHECK(gpu::compile(s.engine, withBermudan).unsupported.size() == 1);
}

TEST(gpu_fused_kernel_matches_exposure_engine) {
  // The kernels (run on the CPU in f32 with their own random numbers) against the
  // double-precision ExposureEngine: equal at t = 0, within Monte Carlo error after.
  for (const bool csa : {false, true}) {
    const auto s = gpuSetup(8000, csa);
    const auto plan = gpu::compile(s.engine, s.nettingSet, 0.95, 0);
    const auto fused = gpu::summarise(plan, gpu::runFusedReference(plan));
    const auto cpu = s.engine.run(s.nettingSet, 0.95).profile;
    CHECK(fused.profile.times.size() == cpu.times.size());
    CHECK_NEAR(fused.profile.expectedValue[0], cpu.expectedValue[0], 1e-6 * 1.9e6);
    // Two independent Monte Carlo estimates: integrated measures within 4 standard errors,
    // the worst of the ~100 dates within 6, and no systematic lean across dates.
    const double tol = 4.0 / std::sqrt(8000.0), worstDate = 6.0 / std::sqrt(8000.0);
    CHECK(maxRelDiff(fused.profile.expectedExposure, cpu.expectedExposure) < worstDate);
    CHECK(maxRelDiff(fused.profile.potentialFutureExposure, cpu.potentialFutureExposure) < worstDate);
    CHECK(maxRelDiff(fused.profile.discountedExpectedNegativeExposure, cpu.discountedExpectedNegativeExposure) < worstDate);
    CHECK_NEAR(fused.profile.effectiveExpectedPositiveExposure(1.0), cpu.effectiveExpectedPositiveExposure(1.0),
               tol * cpu.effectiveExpectedPositiveExposure(1.0));
    double lean = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < cpu.times.size(); ++i) {
      lean += fused.profile.expectedExposure[i] - cpu.expectedExposure[i];
      scale += cpu.expectedExposure[i];
    }
    CHECK(std::fabs(lean) < tol * scale);
    const double cvaFused = unilateralCva(fused.profile, *s.market, 0.4);
    const double cvaCpu = unilateralCva(cpu, *s.market, 0.4);
    CHECK_NEAR(cvaFused, cvaCpu, tol * cvaCpu);
    // CIR++ reproduces the market survival; pathwise CVA matches CVA under independence.
    CHECK(maxRelDiff(fused.meanSurvival, fused.marketSurvival) < 0.01);
    CHECK_NEAR(fused.pathwiseCva(0.4), cvaFused, 0.05 * cvaFused);
  }
}

TEST(gpu_fused_kernel_common_random_numbers) {
  // Same seed: a small spot bump moves the profile smoothly (common random numbers).
  const auto s = gpuSetup(2000, false);
  const auto& gen = s.engine.generator();
  auto bumped = [&](double h) {
    const auto fx = std::make_shared<LognormalAsset>(gen.assets()[0]->withSpot(gen.assets()[0]->spot() + h));
    const ExposureEngine e(gen.withAsset(0, fx), s.engine.config(), s.engine.baseGrid());
    const auto plan = gpu::compile(e, s.nettingSet);
    return unilateralCva(gpu::summarise(plan, gpu::runFusedReference(plan)).profile, *s.market, 0.4);
  };
  const double coarse = (bumped(0.01) - bumped(-0.01)) / 0.02;
  const double fine = (bumped(0.005) - bumped(-0.005)) / 0.01;
  CHECK_NEAR(fine, coarse, 0.05 * std::fabs(coarse));
}

TEST(gpu_fused_kernel_split_across_workers_is_exact) {
  // The CPU backend splits the workgroups across workers: any split must reproduce the
  // single-pass result bit for bit (same per-workgroup partials, same reduction order).
  const auto s = gpuSetup(1000, true);  // 16 workgroups, the last one partial
  const auto plan = gpu::compile(s.engine, s.nettingSet, 0.95, 0);
  const auto whole = gpu::runFusedReference(plan);
  const std::size_t numWG = plan.numWorkgroups();
  for (const std::size_t parts : {std::size_t(2), std::size_t(3), std::size_t(7), numWG + 3}) {
    std::vector<gpu::FusedSlice> slices;
    for (std::size_t k = 0; k < parts; ++k) slices.push_back(gpu::runFusedWorkgroups(plan, numWG * k / parts, numWG * (k + 1) / parts));
    const auto split = gpu::finishFused(plan, slices);
    CHECK(split.sums == whole.sums);
    CHECK(split.pfe == whole.pfe);
  }
  // Slices that leave a gap are rejected.
  CHECK_THROWS(gpu::finishFused(plan, {gpu::runFusedWorkgroups(plan, 0, numWG / 2)}));
}

TEST(gpu_terms_are_merged_per_cash_flow_date) {
  // Identical swaps share every cash-flow date: their terms merge into one set, so the plan
  // does not grow with the number of trades, and offsetting trades leave no bond terms at all.
  const auto s = gpuSetup(256, false);
  NettingSet tripled = s.nettingSet, offset = s.nettingSet;
  const auto pay = InterestRateSwap::vanilla("p2", 1e6, 0.028, 0.0, 5.0, 2, InterestRateSwap::Direction::PayFixed);
  const auto rec = InterestRateSwap::vanilla("r2", 1e6, 0.028, 0.0, 5.0, 2, InterestRateSwap::Direction::ReceiveFixed);
  tripled.trades.push_back(std::make_shared<InterestRateSwap>(pay));
  tripled.trades.push_back(std::make_shared<InterestRateSwap>(pay));
  offset.trades.push_back(std::make_shared<InterestRateSwap>(rec));
  const auto base = gpu::compile(s.engine, s.nettingSet), three = gpu::compile(s.engine, tripled),
             zero = gpu::compile(s.engine, offset);
  auto count = [](const gpu::FusedPlan& p, int kind) {
    std::size_t n = 0;
    for (std::size_t t = 0; t < p.numTerms(); ++t) n += static_cast<int>(p.terms[t * gpu::kTermStride]) == kind;
    return n;
  };
  CHECK(count(three, gpu::kBond) == count(base, gpu::kBond));
  CHECK(count(three, gpu::kFixedFloat) == 3 * count(base, gpu::kFixedFloat));  // one per swap (own fixing)
  CHECK(count(zero, gpu::kBond) < count(base, gpu::kBond));
  // The merged plan prices the same book: t = 0 values against the double-precision engine.
  for (const auto* ns : {&tripled, &offset}) {
    const auto plan = gpu::compile(s.engine, *ns);
    const double fused = gpu::summarise(plan, gpu::runFusedReference(plan)).profile.expectedValue[0];
    const double exact = s.engine.run(*ns, 0.95).profile.expectedValue[0];
    CHECK_NEAR(fused, exact, 2e-6 * 3e6);
  }
}

TEST(gpu_pfe_kernel_is_the_interpolated_empirical_quantile) {
  // The two-level histogram of the PFE kernel reproduces ccr::quantile (linear interpolation
  // between order statistics) of the very exposures the fused kernel produced, to f32 rounding.
  for (const bool csa : {false, true})
    for (const double level : {0.5, 0.95, 0.99}) {
      const auto s = gpuSetup(3000, csa);
      const auto plan = gpu::compile(s.engine, s.nettingSet, level, 0);
      const auto slice = gpu::runFusedWorkgroups(plan, 0, plan.numWorkgroups());
      const auto out = gpu::finishFused(plan, {slice});
      CHECK(out.pfe.size() == plan.reporting.size());
      const std::size_t nP = plan.numPaths;
      for (std::size_t r = 0; r < plan.reporting.size(); ++r) {
        const std::vector<double> e(slice.exposure.begin() + r * nP, slice.exposure.begin() + (r + 1) * nP);
        const double exact = quantile(e, level), mx = *std::max_element(e.begin(), e.end());
        CHECK_NEAR(out.pfe[r], exact, 1e-6 * std::max(mx, 1.0));
      }
    }
}
