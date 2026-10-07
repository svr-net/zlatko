#include <algorithm>
#include <cmath>
#include <memory>

#include "ccr/exposure/allocation.hpp"
#include "ccr/exposure/collateral.hpp"
#include "ccr/exposure/exposure_engine.hpp"
#include "ccr/instruments/asset_forward.hpp"
#include "ccr/instruments/interest_rate_swap.hpp"
#include "test_framework.hpp"

using namespace ccr;

TEST(exposure_profile_statistics) {
  // Two paths, three dates: values {+2, -1}, {+4, +1}, {-3, +6}.
  const TimeGrid grid({0.5, 1.0});
  Matrix v(2, 3);
  v(0, 0) = 2.0;  v(0, 1) = 4.0;  v(0, 2) = -3.0;
  v(1, 0) = -1.0; v(1, 1) = 1.0;  v(1, 2) = 6.0;
  const Matrix d(2, 3, 1.0);
  const auto profile = ExposureProfile::compute(v, grid, d, 1.0);
  CHECK_NEAR(profile.expectedExposure[0], 1.0, 1e-15);
  CHECK_NEAR(profile.expectedExposure[1], 2.5, 1e-15);
  CHECK_NEAR(profile.expectedExposure[2], 3.0, 1e-15);
  CHECK_NEAR(profile.expectedNegativeExposure[0], -0.5, 1e-15);
  CHECK_NEAR(profile.expectedNegativeExposure[2], -1.5, 1e-15);
  CHECK_NEAR(profile.potentialFutureExposure[1], 4.0, 1e-15);
  CHECK_NEAR(profile.maxPotentialFutureExposure(), 6.0, 1e-15);
  // EPE over one year: (2.5 * 0.5 + 3.0 * 0.5) / 1.0
  CHECK_NEAR(profile.expectedPositiveExposure(1.0), 2.75, 1e-15);

  // Effective EE never decreases.
  Matrix w(1, 3);
  w(0, 0) = 1.0; w(0, 1) = 3.0; w(0, 2) = 2.0;
  const auto p2 = ExposureProfile::compute(w, grid, Matrix(1, 3, 1.0));
  const auto eff = p2.effectiveExpectedExposure();
  CHECK_NEAR(eff[2], 3.0, 1e-15);
  CHECK_NEAR(p2.effectiveExpectedPositiveExposure(1.0), 3.0, 1e-15);
  CHECK(p2.effectiveExpectedPositiveExposure(1.0) >= p2.expectedPositiveExposure(1.0));
}

TEST(collateral_thresholds_mta_and_lag) {
  const TimeGrid grid({0.1, 0.2, 0.3});
  Matrix v(1, 4);
  v(0, 0) = 5.0; v(0, 1) = 12.0; v(0, 2) = 12.5; v(0, 3) = -20.0;

  CollateralAgreement csa;
  csa.thresholdCounterparty = 2.0;
  csa.thresholdOwn = 4.0;
  csa.minimumTransferAmount = 1.0;
  csa.marginPeriodOfRisk = 0.0;
  const Matrix c = simulateCollateral(csa, v, grid);
  CHECK_NEAR(c(0, 0), 3.0, 1e-15);    // 5 - 2
  CHECK_NEAR(c(0, 1), 10.0, 1e-15);   // 12 - 2
  CHECK_NEAR(c(0, 2), 10.0, 1e-15);   // change of 0.5 < MTA: no call
  CHECK_NEAR(c(0, 3), -16.0, 1e-15);  // we post 20 - 4

  // With a margin period of risk the collateral lags the value by one date.
  csa.marginPeriodOfRisk = 0.1;
  csa.minimumTransferAmount = 0.0;
  const Matrix lagged = simulateCollateral(csa, v, grid);
  CHECK_NEAR(lagged(0, 1), 3.0, 1e-12);
  CHECK_NEAR(lagged(0, 2), 10.0, 1e-12);
  CHECK_NEAR(lagged(0, 3), 10.5, 1e-12);

  // One-way CSA with an independent amount.
  CollateralAgreement oneWay;
  oneWay.independentAmount = 1.5;
  oneWay.marginPeriodOfRisk = 0.0;
  const Matrix c2 = simulateCollateral(oneWay, v, grid);
  CHECK_NEAR(c2(0, 3), 1.5, 1e-15);
  CHECK_NEAR(c2(0, 1), 13.5, 1e-15);
}

namespace {

ExposureEngine makeEngine(std::size_t paths = 4000) {
  const auto yc = std::make_shared<YieldCurve>(std::vector<double>{1.0, 10.0}, std::vector<double>{0.02, 0.03});
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 0.01);
  const auto fx = std::make_shared<LognormalAsset>("FX", 1.0, 0.12, foreign);
  return ExposureEngine(ScenarioGenerator(hw, {fx}), {paths, 17, true}, TimeGrid::standardExposureGrid(5.0));
}

}  // namespace

TEST(netting_and_collateral_reduce_exposure) {
  const auto engine = makeEngine();
  NettingSet ns;
  ns.id = "cpty";
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("pay", 1e6, 0.028, 0.0, 5.0, 2, InterestRateSwap::Direction::PayFixed)));
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("rec", 8e5, 0.026, 0.0, 4.0, 2, InterestRateSwap::Direction::ReceiveFixed)));
  ns.trades.push_back(std::make_shared<AssetForward>("fx", "FX", 1e6, 1.0, 3.0));

  const auto uncollateralised = engine.run(ns);
  const auto& s = uncollateralised.scenarios;
  const Matrix gross = grossPositiveValues(uncollateralised.tradeValues);
  const auto grossProfile = ExposureProfile::compute(gross, s.grid, s.deflator);
  for (std::size_t j = 0; j < s.numTimes(); ++j) {
    CHECK(uncollateralised.profile.expectedExposure[j] <= grossProfile.expectedExposure[j] + 1e-9);
    CHECK(uncollateralised.profile.potentialFutureExposure[j] + 1e-9 >= uncollateralised.profile.expectedExposure[j]);
  }

  ns.collateral = CollateralAgreement{};
  ns.collateral->thresholdOwn = 0.0;
  const auto collateralised = engine.run(ns);
  CHECK(collateralised.scenarios.numTimes() > s.numTimes());  // margin call dates added
  CHECK(collateralised.profile.times == uncollateralised.profile.times);  // reported on the same dates
  CHECK(collateralised.profile.effectiveExpectedPositiveExposure() <
        0.5 * uncollateralised.profile.effectiveExpectedPositiveExposure());
  // The PFE is much reduced, except on the first few weeks (the margin period of risk is a large
  // part of the horizon) and on trade cash-flow dates, where collateral posted before the payment
  // is still held by the counterparty over the margin period of risk.
  std::vector<double> events;
  for (const auto& trade : ns.trades)
    for (double t : trade->eventTimes()) events.push_back(t);
  for (std::size_t j = 1; j < s.numTimes(); ++j) {
    const double t = s.grid[j];
    if (t < 0.25 || std::any_of(events.begin(), events.end(), [t](double e) { return std::fabs(e - t) < 1e-9; }))
      continue;
    CHECK(collateralised.profile.potentialFutureExposure[j] <
          0.5 * uncollateralised.profile.potentialFutureExposure[j] + 1.0);
  }
}

TEST(marginal_contributions_add_up) {
  const auto engine = makeEngine(2000);
  NettingSet ns;
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("a", 1e6, 0.027, 0.0, 5.0, 2, InterestRateSwap::Direction::PayFixed)));
  ns.trades.push_back(std::make_shared<AssetForward>("b", "FX", -5e5, 1.0, 2.0));
  ns.trades.push_back(std::make_shared<AssetForward>("c", "FX", 3e5, 0.95, 4.0));
  const auto r = engine.run(ns);
  const auto& s = r.scenarios;

  const auto contributions = marginalExposureContributions(r.tradeValues, r.nettedValue, &s.deflator);
  for (std::size_t j = 0; j < s.numTimes(); ++j) {
    double sum = 0.0;
    for (const auto& c : contributions) sum += c[j];
    CHECK_NEAR(sum, r.profile.discountedExpectedExposure[j], 1e-6);
  }

  // Incremental exposure of the last trade = EE(all) - EE(all but last).
  const Matrix without = r.tradeValues[0] + r.tradeValues[1];
  const auto inc = incrementalExposure(without, r.tradeValues[2]);
  const auto before = ExposureProfile::compute(without, s.grid, s.deflator);
  for (std::size_t j = 0; j < s.numTimes(); ++j)
    CHECK_NEAR(inc[j], r.profile.expectedExposure[j] - before.expectedExposure[j], 1e-6);
}
