#include <cmath>
#include <memory>

#include "ccr/models/scenario_generator.hpp"
#include "test_framework.hpp"

using namespace ccr;

namespace {

std::shared_ptr<const YieldCurve> domesticCurve() {
  return std::make_shared<YieldCurve>(std::vector<double>{0.5, 2.0, 5.0, 10.0},
                                      std::vector<double>{0.020, 0.025, 0.030, 0.033});
}

}  // namespace

TEST(hull_white_reprices_initial_curve) {
  const auto curve = domesticCurve();
  const HullWhite1F hw(curve, 0.05, 0.01);
  CHECK_NEAR(hw.zeroBond(0.0, 7.0, 0.0), curve->discount(7.0), 1e-15);

  const auto model = std::make_shared<HullWhite1F>(curve, 0.05, 0.01);
  const ScenarioGenerator gen(model);
  const auto s = gen.generate(TimeGrid::uniform(10.0, 40), {20000, 7, true});
  for (std::size_t j : {std::size_t(4), std::size_t(20), std::size_t(40)}) {
    double meanD = 0.0, meanBond = 0.0;
    for (std::size_t p = 0; p < s.numPaths; ++p) {
      meanD += s.deflator(p, j);
      meanBond += s.deflator(p, j) * s.zeroBond(p, j, 12.0);
    }
    meanD /= s.numPaths;
    meanBond /= s.numPaths;
    CHECK_NEAR(meanD, curve->discount(s.grid[j]), 2e-3);
    CHECK_NEAR(meanBond, curve->discount(12.0), 2e-3);
  }
}

TEST(hull_white_exact_step_matches_fine_simulation) {
  // A single exact step and forty exact steps give the same distribution of D(0, T).
  const auto model = std::make_shared<HullWhite1F>(domesticCurve(), 0.1, 0.015);
  const ScenarioGenerator gen(model);
  const auto coarse = gen.generate(TimeGrid::uniform(5.0, 1), {40000, 1, true});
  const auto fine = gen.generate(TimeGrid::uniform(5.0, 40), {40000, 2, true});
  auto stats = [](const ScenarioSet& s) {
    double m = 0.0, m2 = 0.0;
    const std::size_t j = s.numTimes() - 1;
    for (std::size_t p = 0; p < s.numPaths; ++p) {
      m += s.deflator(p, j);
      m2 += s.deflator(p, j) * s.deflator(p, j);
    }
    m /= s.numPaths;
    return std::make_pair(m, std::sqrt(m2 / s.numPaths - m * m));
  };
  const auto a = stats(coarse);
  const auto b = stats(fine);
  CHECK_NEAR(a.first, b.first, 2e-3);
  CHECK_NEAR(a.second, b.second, 2e-3);
}

TEST(hull_white_jamshidian_matches_monte_carlo) {
  const auto curve = domesticCurve();
  const auto model = std::make_shared<HullWhite1F>(curve, 0.05, 0.012);
  const double expiry = 2.0;
  std::vector<double> payments, accruals;
  for (int i = 1; i <= 10; ++i) {
    payments.push_back(expiry + 0.5 * i);
    accruals.push_back(0.5);
  }
  const double strike = 0.031;
  const double payer = model->europeanSwaption(expiry, payments, accruals, strike, true);
  const double receiver = model->europeanSwaption(expiry, payments, accruals, strike, false);

  // Payer - receiver = forward-starting payer swap.
  double annuity = 0.0;
  for (std::size_t i = 0; i < payments.size(); ++i) annuity += accruals[i] * curve->discount(payments[i]);
  const double swap = curve->discount(expiry) - curve->discount(payments.back()) - strike * annuity;
  CHECK_NEAR(payer - receiver, swap, 1e-12);

  const ScenarioGenerator gen(model);
  const auto s = gen.generate(TimeGrid({expiry}), {40000, 11, true});
  double mc = 0.0;
  for (std::size_t p = 0; p < s.numPaths; ++p) {
    double value = 1.0 - s.zeroBond(p, 1, payments.back());
    for (std::size_t i = 0; i < payments.size(); ++i) value -= strike * accruals[i] * s.zeroBond(p, 1, payments[i]);
    mc += s.deflator(p, 1) * std::max(value, 0.0);
  }
  mc /= s.numPaths;
  CHECK_NEAR(mc, payer, 0.02 * payer);
}

TEST(fx_rate_is_martingale_with_stochastic_rates) {
  const auto curve = domesticCurve();
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto model = std::make_shared<HullWhite1F>(curve, 0.05, 0.01);
  const auto fx = std::make_shared<LognormalAsset>("EURUSD", 1.10, 0.12, foreign);
  Matrix corr = Matrix::identity(2);
  corr(0, 1) = corr(1, 0) = 0.4;
  const ScenarioGenerator gen(model, {fx}, {}, corr);
  const auto s = gen.generate(TimeGrid::uniform(5.0, 20), {20000, 3, true});
  const std::size_t a = s.assetIndex("EURUSD");
  for (std::size_t j : {std::size_t(4), std::size_t(20)}) {
    double m = 0.0;
    for (std::size_t p = 0; p < s.numPaths; ++p) m += s.deflator(p, j) * s.assetValues[a](p, j);
    m /= s.numPaths;
    CHECK_NEAR(m, 1.10 * foreign->discount(s.grid[j]), 5e-3);
  }
}

TEST(cir_plus_plus_fits_market_survival) {
  const auto market = std::make_shared<CreditCurve>(std::vector<double>{1.0, 3.0, 5.0, 10.0},
                                                    std::vector<double>{0.010, 0.015, 0.020, 0.022});
  const auto cir = std::make_shared<CirIntensity>("CPTY", market, 0.5, 0.012, 0.08, 0.008, 8);

  // With no volatility the analytic CIR survival is the deterministic integral.
  const CirIntensity deterministic("D", market, 0.5, 0.02, 0.0, 0.01);
  CHECK_NEAR(deterministic.cirSurvival(3.0),
             std::exp(-(0.02 * 3.0 + (0.01 - 0.02) * (1.0 - std::exp(-1.5)) / 0.5)), 1e-14);

  const auto model = std::make_shared<HullWhite1F>(domesticCurve(), 0.05, 0.01);
  const ScenarioGenerator gen(model, {}, {cir});
  const auto s = gen.generate(TimeGrid::uniform(10.0, 40), {20000, 5, true});
  const std::size_t c = s.creditIndex("CPTY");
  for (std::size_t j : {std::size_t(8), std::size_t(20), std::size_t(40)}) {
    double m = 0.0;
    for (std::size_t p = 0; p < s.numPaths; ++p) m += s.survival[c](p, j);
    m /= s.numPaths;
    CHECK_NEAR(m, market->survival(s.grid[j]), 3e-3);
  }
}

TEST(scenarios_are_reproducible_and_validated) {
  const auto model = std::make_shared<HullWhite1F>(domesticCurve(), 0.05, 0.01);
  const ScenarioGenerator gen(model);
  const auto a = gen.generate(TimeGrid::uniform(1.0, 4), {10, 99, false});
  const auto b = gen.generate(TimeGrid::uniform(1.0, 4), {10, 99, false});
  for (std::size_t p = 0; p < 10; ++p) CHECK(a.rateState(p, 4) == b.rateState(p, 4));

  Matrix badCorr = Matrix::identity(3);
  CHECK_THROWS(ScenarioGenerator(model, {}, {}, badCorr));
}
