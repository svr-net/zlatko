#include <cmath>
#include <memory>

#include "ccr/cva/cva.hpp"
#include "ccr/exposure/exposure_engine.hpp"
#include "ccr/hedging/sensitivities.hpp"
#include "ccr/instruments/asset_forward.hpp"
#include "ccr/instruments/interest_rate_swap.hpp"
#include "test_framework.hpp"

using namespace ccr;

TEST(cds_hedge_neutralises_cva_spread_risk) {
  const YieldCurve yc({1.0, 10.0}, {0.02, 0.03});
  const std::vector<CdsQuote> quotes = {{1.0, 0.0080}, {3.0, 0.0100}, {5.0, 0.0130}};
  const double recovery = 0.4;

  // A humped discounted EE profile (typical of a swap).
  std::vector<double> times, ee;
  for (int i = 0; i <= 20; ++i) {
    const double t = 0.25 * i;
    times.push_back(t);
    ee.push_back(1e6 * 0.02 * std::sqrt(t) * (5.0 - t) / 5.0);
  }
  auto cvaOf = [&](const CreditCurve& curve) { return unilateralCva(times, ee, curve, recovery); };

  const auto cs01 = creditSpreadSensitivities(cvaOf, yc, quotes, recovery);
  // A parallel widening increases CVA. (A single short-end bucket can have the opposite sign:
  // re-bootstrapping lowers the hazard rate just after it, where this exposure is larger.)
  CHECK(cs01[0] + cs01[1] + cs01[2] > 0.0);
  CHECK(cs01[2] > 0.0);
  const Matrix jacobian = cdsHedgeJacobian(yc, quotes, recovery);
  const auto notionals = cdsHedgeNotionals(cs01, jacobian);
  for (double n : notionals) CHECK(std::isfinite(n));

  // P&L of CVA versus hedge for a non-parallel spread move.
  const double base = cvaOf(bootstrapCreditCurve(yc, quotes, recovery));
  auto moved = quotes;
  moved[0].spread += 0.0002;
  moved[1].spread += 0.0004;
  moved[2].spread += 0.0003;
  const CreditCurve baseCurve = bootstrapCreditCurve(yc, quotes, recovery);
  const CreditCurve movedCurve = bootstrapCreditCurve(yc, moved, recovery);
  const double cvaChange = cvaOf(movedCurve) - base;
  double hedgeChange = 0.0;
  for (std::size_t k = 0; k < quotes.size(); ++k) {
    const CreditDefaultSwap cds(quotes[k].maturity, quotes[k].spread, notionals[k], recovery);
    hedgeChange += cds.npv(yc, movedCurve) - cds.npv(yc, baseCurve);
  }
  CHECK(cvaChange > 0.01 * base);
  CHECK_NEAR(hedgeChange, cvaChange, 0.01 * std::fabs(cvaChange));
}

TEST(market_risk_cva_delta_with_common_random_numbers) {
  const auto yc = std::make_shared<YieldCurve>(YieldCurve::flat(0.02));
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 0.01);
  const ScenarioGenerator base(hw, {std::make_shared<LognormalAsset>("FX", 1.0, 0.12, foreign)});
  const CreditCurve cpty = CreditCurve::flat(0.02);

  NettingSet ns;
  ns.trades.push_back(std::make_shared<AssetForward>("fwd", "FX", 1e6, 1.0, 3.0));

  auto cvaForSpot = [&](double bump) {
    const auto asset = std::make_shared<LognormalAsset>(base.assets()[0]->withSpot(1.0 + bump));
    const ExposureEngine engine(base.withAsset(0, asset), {4000, 5, true}, TimeGrid::uniform(3.0, 12));
    return unilateralCva(engine.run(ns).profile, cpty, 0.4);
  };
  const double delta1 = centralDifference(cvaForSpot, 0.01);
  const double delta2 = centralDifference(cvaForSpot, 0.005);
  CHECK(delta1 > 0.0);  // long forward: exposure and CVA increase with the spot
  CHECK_NEAR(delta1, delta2, 0.05 * delta1);
}
