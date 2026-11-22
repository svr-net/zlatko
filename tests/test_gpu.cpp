#include <algorithm>
#include <cmath>
#include <memory>

#include "ccr/core/math.hpp"
#include "ccr/cva/cva.hpp"
#include "ccr/exposure/allocation.hpp"
#include "ccr/xva/xva.hpp"
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
  CHECK(plan.indices.size() == 4 * nT + 1);  // + the date of t + MPR (nested initial margin)
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

  // A Bermudan is priced by AMC inside the kernel: supported, with its regression terms located.
  NettingSet withBermudan = s.nettingSet;
  const auto underlying = InterestRateSwap::vanilla("u", 1e6, 0.03, 1.0, 4.0, 1, InterestRateSwap::Direction::ReceiveFixed);
  withBermudan.trades.push_back(std::make_shared<BermudanSwaption>("berm", underlying, std::vector<double>{1.0, 2.0, 3.0}));
  CHECK(gpu::limitation(s.engine, withBermudan).empty());
  const auto bplan = gpu::compile(s.engine, withBermudan);
  CHECK(bplan.unsupported.empty() && bplan.amc.size() == 1 && bplan.amc[0].decideTerm.size() == 3);
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
    CHECK(split.values == whole.values);
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
      CHECK(out.values.size() >= plan.reporting.size());
      const std::size_t nP = plan.numPaths;
      for (std::size_t r = 0; r < plan.reporting.size(); ++r) {
        const std::vector<double> e(slice.exposure.begin() + r * nP, slice.exposure.begin() + (r + 1) * nP);
        const double exact = quantile(e, level), mx = *std::max_element(e.begin(), e.end());
        CHECK_NEAR(out.values[plan.views.pfe + r], exact, 1e-6 * std::max(mx, 1.0));
      }
    }
}

TEST(gpu_per_trade_statistics_and_initial_margin) {
  // Per-trade mode and the margin quantile against the double-precision engine's paths: the
  // Euler contributions add up to the netting-set EE*, and every statistic agrees with
  // marginalExposureContributions / incrementalExposure / initialMarginProfile within MC error.
  const auto s = gpuSetup(8000, false);
  gpu::FusedOptions o;
  o.perTrade = true;
  o.imQuantile = 0.99;
  const auto plan = gpu::compile(s.engine, s.nettingSet, o);
  CHECK(plan.perTrade() && plan.hasInitialMargin());
  CHECK(plan.statsPerDate == 24);  // 8 + 5 x 3 trades, padded to a multiple of 4
  CHECK(gpu::fusedExposureKernel(true).find("MAX_TRADES : u32 = 16u") != std::string::npos);
  const auto f = gpu::summarise(plan, gpu::runFusedReference(plan));
  CHECK(f.trades.size() == 3);
  // The default plan prices the same netting set: the profiles agree to f32 rounding at t = 0.
  const auto plain = gpu::summarise(gpu::compile(s.engine, s.nettingSet), gpu::runFusedReference(gpu::compile(s.engine, s.nettingSet)));
  CHECK_NEAR(f.profile.expectedValue[0], plain.profile.expectedValue[0], 1e-6 * 1e6);

  const ExposureResult cpu = s.engine.run(s.nettingSet, 0.95);
  const auto& idx = cpu.reportingIndices;
  const Matrix net = selectColumns(cpu.nettedValue, idx), deflator = selectColumns(cpu.scenarios.deflator, idx);
  std::vector<Matrix> values;
  for (const auto& m : cpu.tradeValues) values.push_back(selectColumns(m, idx));
  const auto marginal = marginalExposureContributions(values, net, &deflator);
  const double scale = *std::max_element(cpu.profile.discountedExpectedExposure.begin(), cpu.profile.discountedExpectedExposure.end());
  const double tol = 6.0 / std::sqrt(8000.0) * scale;
  for (std::size_t k = 0; k < f.profile.times.size(); ++k) {
    double sum = 0.0;
    for (const auto& t : f.trades) sum += t.marginalDiscounted[k];
    CHECK_NEAR(sum, f.profile.discountedExpectedExposure[k], 1e-5 * scale);  // Euler allocation adds up
    for (std::size_t i = 0; i < 3; ++i) {
      CHECK_NEAR(f.trades[i].marginalDiscounted[k], marginal[i][k], tol);
      Matrix without(net.rows(), net.cols(), 0.0);
      for (std::size_t m = 0; m < 3; ++m)
        if (m != i) without += values[m];
      CHECK_NEAR(f.trades[i].incrementalDiscounted[k], incrementalExposure(without, values[i], &deflator)[k], tol);
    }
  }
  const auto im = initialMarginProfile(net, TimeGrid(cpu.profile.times), deflator, 0.99, 10.0 / 250.0);
  CHECK(f.valueChangeQuantile.size() == im.size() && f.valueChangeQuantile[0] == 0.0);
  double imScale = *std::max_element(im.begin(), im.end()), worst = 0.0;
  for (std::size_t k = 1; k < im.size(); ++k) {
    const double dt = cpu.profile.times[k] - cpu.profile.times[k - 1];
    const double gpuIm = f.valueChangeQuantile[k] * std::sqrt(10.0 / 250.0 / dt) * f.meanDeflator[k];
    worst = std::max(worst, std::fabs(gpuIm - im[k]));
  }
  CHECK(worst < 0.1 * imScale);  // a 99% quantile of independent samples: a few % noise
  // At most 16 trades get their own statistics.
  NettingSet many = s.nettingSet;
  while (many.trades.size() <= gpu::kMaxTradeGroups) many.trades.push_back(s.nettingSet.trades[0]);
  CHECK(!gpu::limitation(s.engine, many, true).empty() && gpu::limitation(s.engine, many, false).empty());
  CHECK_THROWS(gpu::compile(s.engine, many, o));
}

TEST(gpu_cpu_kernels_are_bit_identical_on_any_number_of_threads) {
  // The CPU backend runs workgroups, statistics and quantile rows on the lock-free pool: one
  // thread and eight must give the same bits, per-trade and margin plans included.
  const auto s = gpuSetup(3000, true);
  gpu::FusedOptions o;
  o.perTrade = true;
  o.imQuantile = 0.99;
  for (const auto& plan : {gpu::compile(s.engine, s.nettingSet), gpu::compile(s.engine, s.nettingSet, o)}) {
    SpinPool one(0), eight(7);
    const auto a = gpu::runFusedReference(plan, one), b = gpu::runFusedReference(plan, eight);
    CHECK(a.sums == b.sums);
    CHECK(a.values == b.values);
  }
}

TEST(gpu_amc_bermudan_matches_exposure_engine) {
  // Bermudan swaptions by Longstaff-Schwartz inside the fused kernel (regressions fitted on a CPU
  // pass of the kernel) against BermudanSwaption on ExposureEngine: price and exposure profile
  // within Monte Carlo error, for physical and cash settlement, alone and in a netting set.
  const auto s = gpuSetup(8000, false);
  const auto u = InterestRateSwap::vanilla("u", 1e6, 0.03, 1.0, 4.0, 1, InterestRateSwap::Direction::ReceiveFixed);
  for (const Settlement settle : {Settlement::Physical, Settlement::Cash}) {
    NettingSet alone, mixed = s.nettingSet;
    const auto berm = std::make_shared<BermudanSwaption>("berm", u, std::vector<double>{1.0, 2.0, 3.0, 4.0}, settle, 1.0, 2);
    alone.trades.push_back(berm);
    mixed.trades.push_back(std::make_shared<BermudanSwaption>("short", u, std::vector<double>{2.0, 3.0}, settle, -0.5, 3));
    for (const auto* ns : {&alone, &mixed}) {
      const auto plan = gpu::compile(s.engine, *ns);
      CHECK(plan.unsupported.empty());
      const auto fused = gpu::summarise(plan, gpu::runFusedReference(plan)).profile;
      const auto cpu = s.engine.run(*ns, 0.95).profile;
      const double scale = std::max(1e3, *std::max_element(cpu.expectedExposure.begin(), cpu.expectedExposure.end()));
      CHECK_NEAR(fused.expectedValue[0], cpu.expectedValue[0], 0.04 * scale);
      CHECK(maxRelDiff(fused.expectedExposure, cpu.expectedExposure) < 0.06);
      CHECK(maxRelDiff(fused.discountedExpectedExposure, cpu.discountedExpectedExposure) < 0.06);
      // ENE is small next to EE for a long option: measured against the exposure scale.
      double worst = 0.0;
      for (std::size_t k = 0; k < cpu.times.size(); ++k)
        worst = std::max(worst, std::fabs(fused.expectedNegativeExposure[k] - cpu.expectedNegativeExposure[k]));
      CHECK(worst < 0.03 * scale);
    }
  }
}

TEST(gpu_path_views_match_exposure_engine) {
  // The path views computed by the row-statistics kernel against the double-precision engine:
  // sample paths are consistent, distributions and profiles agree within Monte Carlo error, and
  // the risk-factor martingale tests and correlations hold.
  const auto s = gpuSetup(8000, true);
  gpu::FusedOptions o;
  o.samples = 12;
  o.histogramTime = 2.0;
  o.gross = true;
  o.collateralBands = true;
  o.tradeViews = true;
  const auto plan = gpu::compile(s.engine, s.nettingSet, o);
  const auto f = gpu::summarise(plan, gpu::runFusedReference(plan));
  const ExposureResult cpu = s.engine.run(s.nettingSet, 0.95);
  const std::size_t nR = f.profile.times.size();
  CHECK(f.exposureSamples.size() == 12 && f.exposureSamples[0].size() == nR);
  for (std::size_t q = 0; q < 12; ++q)
    for (std::size_t k = 0; k < nR; ++k)
      CHECK_NEAR(f.exposureSamples[q][k], f.nettedSamples[q][k] - f.collateralSamples[q][k], 1e-3 * (1.0 + std::fabs(f.nettedSamples[q][k])));
  // Histogram: fractions of paths summing to one over increasing edges.
  double total = 0.0;
  for (double c : f.histogramCounts) total += c;
  CHECK(f.histogramCounts.size() == 40 && std::fabs(total - 1.0) < 1e-6 && f.histogramEdges.front() < f.histogramEdges.back());
  CHECK_NEAR(f.histogramTime, 2.0, 0.26);
  // Profiles against the engine.
  const auto& idx = cpu.reportingIndices;
  const TimeGrid rep(cpu.profile.times);
  const Matrix deflator = selectColumns(cpu.scenarios.deflator, idx);
  Matrix gross(cpu.nettedValue.rows(), idx.size(), 0.0);
  for (const auto& m : cpu.tradeValues) {
    const Matrix v = selectColumns(m, idx);
    for (std::size_t p = 0; p < v.rows(); ++p)
      for (std::size_t k = 0; k < v.cols(); ++k) gross(p, k) += std::max(v(p, k), 0.0);
  }
  const auto grossCpu = ExposureProfile::compute(gross, rep, deflator, 0.95);
  const double tol = 6.0 / std::sqrt(8000.0);
  CHECK(f.grossProfile && maxRelDiff(f.grossProfile->expectedExposure, grossCpu.expectedExposure) < tol);
  CHECK(maxRelDiff(f.grossProfile->potentialFutureExposure, grossCpu.potentialFutureExposure) < tol);
  CHECK(maxRelDiff(f.grossProfile->discountedExpectedExposure, grossCpu.discountedExpectedExposure) < tol);
  CHECK(f.tradeViews.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto tcpu = ExposureProfile::compute(selectColumns(cpu.tradeValues[i], idx), rep, deflator, 0.95);
    const auto& tv = f.tradeViews[i];
    CHECK_NEAR(tv.profile.expectedValue[0], tcpu.expectedValue[0], 1e-5 * 1e6);  // t = 0: deterministic
    double scale = 1.0;
    for (double x : tcpu.expectedExposure) scale = std::max(scale, x);
    for (double x : tcpu.expectedNegativeExposure) scale = std::max(scale, -x);
    for (std::size_t k = 0; k < nR; ++k) {
      CHECK_NEAR(tv.profile.expectedExposure[k], tcpu.expectedExposure[k], tol * scale);
      CHECK_NEAR(tv.profile.expectedNegativeExposure[k], tcpu.expectedNegativeExposure[k], tol * scale);
      CHECK(tv.bands[0][k] <= tv.bands[1][k] + 1e-6 && tv.bands[1][k] <= tv.bands[2][k] + 1e-6);
    }
  }
  const Matrix coll = selectColumns(cpu.collateral, idx);
  double collScale = 1.0, worst = 0.0;
  for (std::size_t k = 0; k < nR; ++k) {
    std::vector<double> c(coll.rows());
    for (std::size_t p = 0; p < coll.rows(); ++p) c[p] = coll(p, k);
    for (std::size_t b = 0; b < 3; ++b) {
      const double q = quantile(c, std::vector<double>{0.05, 0.5, 0.95}[b]);
      collScale = std::max(collScale, std::fabs(q));
      worst = std::max(worst, std::fabs(f.collateralBands[b][k] - q));
    }
  }
  CHECK(worst < 0.1 * collScale);

  // Risk factors: martingale tests, CIR++ fit and realised correlation.
  gpu::FusedOptions fo;
  fo.factorViews = true;
  fo.samples = 5;
  fo.bondMaturity = 8.0;
  const NettingSet empty;
  const auto fp = gpu::compile(s.engine, empty, fo);
  const auto fr = gpu::summarise(fp, gpu::runFusedReference(fp));
  CHECK(fr.factors.has_value());
  const auto& F = *fr.factors;
  const auto& gen = s.engine.generator();
  const auto& times = fr.profile.times;
  for (std::size_t k = 0; k < times.size(); ++k) {
    CHECK_NEAR(F.deflatorMean[k], gen.rateModel()->curve().discount(times[k]), 5e-3);
    CHECK_NEAR(F.bondMean[k], gen.rateModel()->curve().discount(8.0), 5e-3);
    CHECK_NEAR(F.assets[0].deflatedMean[k], gen.assets()[0]->spot() * gen.assets()[0]->carryCurve().discount(times[k]), 0.01);
    CHECK_NEAR(F.credits[0].survivalMean[k], s.market->survival(times[k]), 0.01);
  }
  CHECK(F.rateSamples.size() == 5 && F.rateBands.size() == 5 && F.incrementCorrelation.rows() == 3);
  CHECK_NEAR(F.incrementCorrelation(0, 1), 0.2, 0.05);
  CHECK_NEAR(F.incrementCorrelation(0, 2), 0.0, 0.05);
}

TEST(gpu_securities_collateral_matches_exposure_engine) {
  // Variation margin posted in a government bond or in the FX asset, with haircuts: units fixed
  // at the call date and revalued at close-out, in the fused kernels and in ExposureEngine.
  for (const auto type : {CollateralAgreement::Type::Bond, CollateralAgreement::Type::Asset}) {
    auto s = gpuSetup(8000, true);
    auto& csa = *s.nettingSet.collateral;
    csa.type = type;
    csa.haircut = type == CollateralAgreement::Type::Bond ? 0.02 : 0.15;
    csa.bondMaturity = 10.0;
    csa.assetIndex = 0;
    csa.marginPeriodOfRisk = 20.0 / 250.0;
    const auto plan = gpu::compile(s.engine, s.nettingSet);
    CHECK(plan.securitiesCollateral && plan.valsBytes() == 2 * plan.numPaths * plan.numDates() * 4);
    const auto fused = gpu::summarise(plan, gpu::runFusedReference(plan)).profile;
    const auto cpu = s.engine.run(s.nettingSet, 0.95).profile;
    CHECK_NEAR(fused.expectedValue[0], cpu.expectedValue[0], 1e-5 * 1.9e6);
    const double scale = *std::max_element(cpu.expectedExposure.begin(), cpu.expectedExposure.end());
    double worst = 0.0;
    for (std::size_t k = 0; k < cpu.times.size(); ++k) worst = std::max(worst, std::fabs(fused.expectedExposure[k] - cpu.expectedExposure[k]));
    CHECK(worst < 6.0 / std::sqrt(8000.0) * scale);
  }
  // Cash with a zero haircut is the old cash CSA exactly; a bond maturing before the horizon is refused.
  auto s = gpuSetup(1000, true);
  const auto cash = gpu::runFusedReference(gpu::compile(s.engine, s.nettingSet));
  s.nettingSet.collateral->type = CollateralAgreement::Type::Bond;
  s.nettingSet.collateral->bondMaturity = 2.0;
  CHECK_THROWS(gpu::compile(s.engine, s.nettingSet));
  CHECK_THROWS(s.engine.run(s.nettingSet, 0.95));
  CHECK(!cash.values.empty());
}

TEST(gpu_nested_initial_margin_matches_the_analytic_conditional_quantile) {
  // An FX forward with (almost) deterministic rates: over the margin period h the value moves by
  // N Q(t, T) S(t) (e^{sigma sqrt(h) Z + ...} - 1), so the conditional 99% quantile of |dV| is
  // 2.5758 N Q(t, T) S(t) sigma sqrt(h) to first order, and E[D IM(t)] = 2.5758 N sigma sqrt(h) S0 Q(0, T)
  // on every date before the maturity: the nested estimate must reproduce it.
  const auto yc = std::make_shared<YieldCurve>(YieldCurve::flat(0.02));
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 1e-5);
  const double sigma = 0.12, S0 = 1.1, N = 1e6, T = 3.0, h = 10.0 / 250.0;
  const auto fx = std::make_shared<LognormalAsset>("FX", S0, sigma, foreign);
  NettingSet ns;
  ns.trades.push_back(std::make_shared<AssetForward>("fwd", "FX", N, 1.1, T));
  const ExposureEngine engine(ScenarioGenerator(hw, {fx}, {}, Matrix::identity(2)), {2000, 7, true}, TimeGrid::standardExposureGrid(5.0));
  gpu::FusedOptions o;
  o.imQuantile = 0.99;
  o.imInnerPaths = 512;
  o.imMarginPeriod = h;
  const auto plan = gpu::compile(engine, ns, o);
  CHECK(plan.imInnerPaths == 512 && plan.rows.conditionalIm != gpu::kNoRow);
  const auto f = gpu::summarise(plan, gpu::runFusedReference(plan));
  const double exact = 2.5758 * N * sigma * std::sqrt(h) * S0 * foreign->discount(T);
  std::size_t checked = 0;
  for (std::size_t k = 0; k < f.profile.times.size(); ++k) {
    if (f.profile.times[k] + h > T) continue;
    CHECK_NEAR(f.conditionalImDiscounted[k], exact, 0.04 * exact);
    CHECK(f.conditionalImBands[0][k] <= f.conditionalImBands[1][k] && f.conditionalImBands[1][k] <= f.conditionalImBands[2][k]);
    ++checked;
  }
  CHECK(checked > 20);
  // Too few inner paths for the quantile (it keeps the 8 largest samples) is refused.
  o.imInnerPaths = 1000;
  o.imQuantile = 0.9;
  CHECK_THROWS(gpu::compile(engine, ns, o));
}

TEST(gpu_nested_initial_margin_audit_removes_the_small_sample_bias) {
  // The same FX forward with only 64 inner paths: the empirical 99% quantile of 64 samples is several
  // percent low on every path, and MVA (linear in IM) keeps that bias. The audit (chapter X11) reruns
  // the first 128 paths with 512 inner paths, the kernel's 64 extended, and scales by the ratio.
  const auto yc = std::make_shared<YieldCurve>(YieldCurve::flat(0.02));
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 1e-5);
  const double sigma = 0.12, S0 = 1.1, N = 1e6, T = 3.0, h = 10.0 / 250.0;
  const auto fx = std::make_shared<LognormalAsset>("FX", S0, sigma, foreign);
  NettingSet ns;
  ns.trades.push_back(std::make_shared<AssetForward>("fwd", "FX", N, 1.1, T));
  const ExposureEngine engine(ScenarioGenerator(hw, {fx}, {}, Matrix::identity(2)), {1000, 7, true}, TimeGrid::uniform(2.5, 10));
  gpu::FusedOptions o;
  o.imQuantile = 0.99;
  o.imInnerPaths = 64;
  o.imMarginPeriod = h;
  const auto plain = gpu::compile(engine, ns, o);
  CHECK(plain.imAuditPaths == 0 && gpu::auditConditionalIm(plain).empty());
  CHECK(gpu::summarise(plain, gpu::runFusedReference(plain)).conditionalImAuditRatio.empty());
  o.imAuditPaths = 128;
  o.imAuditInnerPaths = 512;
  const auto plan = gpu::compile(engine, ns, o);
  CHECK(plan.imAuditPaths == 128 && plan.imAuditInnerPaths == 512 && plan.header == plain.header);
  const auto f = gpu::summarise(plan, gpu::runFusedReference(plan));
  const double exact = 2.5758 * N * sigma * std::sqrt(h) * S0 * foreign->discount(T);
  double raw = 0.0, corrected = 0.0;
  std::size_t n = 0;
  for (std::size_t k = 0; k < f.profile.times.size(); ++k) {
    if (f.profile.times[k] + h > T) continue;
    CHECK(f.conditionalImAuditRatio[k] > 1.0);
    CHECK_NEAR(f.conditionalImDiscounted[k], f.conditionalImAuditRatio[k] * f.conditionalImDiscountedUncorrected[k], 1e-9 * exact);
    raw += f.conditionalImDiscountedUncorrected[k] / exact - 1.0;
    corrected += f.conditionalImDiscounted[k] / exact - 1.0;
    ++n;
  }
  CHECK(n >= 9);
  CHECK(raw / static_cast<double>(n) < -0.05);
  CHECK_NEAR(corrected / static_cast<double>(n), 0.0, 0.025);
  // The audit needs more inner paths than the kernel, within the kernel's limits.
  o.imAuditInnerPaths = 64;
  CHECK_THROWS(gpu::compile(engine, ns, o));
  o.imAuditInnerPaths = 2048;
  CHECK_THROWS(gpu::compile(engine, ns, o));
}
