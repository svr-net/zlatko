#include <cmath>

#include "ccr/instruments/credit_default_swap.hpp"
#include "ccr/market/credit_curve.hpp"
#include "ccr/market/yield_curve.hpp"
#include "test_framework.hpp"

using namespace ccr;

TEST(yield_curve_interpolation) {
  const YieldCurve yc({1.0, 5.0}, {0.02, 0.04});
  CHECK_NEAR(yc.zeroRate(0.5), 0.02, 1e-15);
  CHECK_NEAR(yc.zeroRate(3.0), 0.03, 1e-15);
  CHECK_NEAR(yc.zeroRate(10.0), 0.04, 1e-15);
  CHECK_NEAR(yc.discount(3.0), std::exp(-0.09), 1e-15);
  // Instantaneous forward is consistent with the discount curve.
  const double h = 1e-6;
  const double numeric = -(std::log(yc.discount(3.0 + h)) - std::log(yc.discount(3.0 - h))) / (2 * h);
  CHECK_NEAR(yc.instantaneousForward(3.0), numeric, 1e-7);
}

TEST(credit_curve_survival) {
  const CreditCurve cc({1.0, 3.0}, {0.01, 0.03});
  CHECK_NEAR(cc.survival(0.5), std::exp(-0.005), 1e-15);
  CHECK_NEAR(cc.survival(2.0), std::exp(-0.01 - 0.03), 1e-15);
  CHECK_NEAR(cc.survival(5.0), std::exp(-0.01 - 0.06 - 0.06), 1e-15);
  CHECK_NEAR(cc.defaultProbability(1.0, 2.0), cc.survival(1.0) - cc.survival(2.0), 1e-15);
}

TEST(cds_credit_triangle) {
  // Flat hazard: par spread ~ (1 - R) * lambda (exact for continuous premium; quarterly
  // premium payments in arrears make it slightly higher).
  const YieldCurve yc = YieldCurve::flat(0.03);
  const CreditCurve cc = CreditCurve::flat(0.02);
  const CreditDefaultSwap cds(5.0, 0.0, 1.0, 0.4);
  CHECK_NEAR(cds.parSpread(yc, cc), 0.6 * 0.02, 1e-4);
}

TEST(cds_bootstrap_reprices_quotes) {
  const YieldCurve yc({1.0, 10.0}, {0.02, 0.035});
  const std::vector<CdsQuote> quotes = {{1.0, 0.0060}, {3.0, 0.0085}, {5.0, 0.0110}, {7.0, 0.0120}, {10.0, 0.0125}};
  const CreditCurve cc = bootstrapCreditCurve(yc, quotes, 0.4);
  for (const auto& q : quotes) {
    const CreditDefaultSwap cds(q.maturity, q.spread, 1.0, 0.4);
    CHECK_NEAR(cds.parSpread(yc, cc), q.spread, 1e-10);
    CHECK_NEAR(cds.npv(yc, cc), 0.0, 1e-10);
  }
}
