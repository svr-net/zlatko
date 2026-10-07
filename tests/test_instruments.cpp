#include <cmath>
#include <memory>

#include "ccr/instruments/asset_forward.hpp"
#include "ccr/instruments/bermudan_swaption.hpp"
#include "ccr/instruments/european_option.hpp"
#include "ccr/instruments/interest_rate_swap.hpp"
#include "test_framework.hpp"

using namespace ccr;

namespace {

std::shared_ptr<const YieldCurve> curve() {
  return std::make_shared<YieldCurve>(std::vector<double>{1.0, 5.0, 10.0}, std::vector<double>{0.02, 0.03, 0.035});
}

double deflatedMean(const ScenarioSet& s, const Matrix& values, std::size_t j) {
  double m = 0.0;
  for (std::size_t p = 0; p < s.numPaths; ++p) m += s.deflator(p, j) * values(p, j);
  return m / static_cast<double>(s.numPaths);
}

}  // namespace

TEST(swap_value_at_inception_matches_curve) {
  const auto yc = curve();
  const auto model = std::make_shared<HullWhite1F>(yc, 0.05, 0.01);
  auto swap = InterestRateSwap::vanilla("swap", 1e6, 0.0, 0.0, 5.0, 2, InterestRateSwap::Direction::PayFixed);
  swap = swap.withFixedRate(swap.parRate(*yc));
  CHECK_NEAR(swap.npv(*yc), 0.0, 1e-8);

  const auto offMarket = swap.withFixedRate(0.02);
  const auto s = ScenarioGenerator(model).generate(TimeGrid::uniform(5.0, 10), {100, 1, true});
  const Matrix v = offMarket.valueCube(s);
  CHECK_NEAR(v(0, 0), offMarket.npv(*yc), 1e-6);
  CHECK(offMarket.npv(*yc) > 0.0);  // paying 2% below the ~2.9% par rate
  for (std::size_t p = 0; p < s.numPaths; ++p) CHECK_NEAR(v(p, 10), 0.0, 0.0);
}

TEST(swap_fixing_recovered_inside_accrual_period) {
  // No cash flow before t = 0.75, so E[D(0, t) V(t)] must equal V(0), including the
  // period fixed at 0.5 and paid at 1.0.
  const auto yc = curve();
  const auto model = std::make_shared<HullWhite1F>(yc, 0.05, 0.015);
  const InterestRateSwap swap("s", 1e6, 0.025, {0.5, 1.0, 1.5}, InterestRateSwap::Direction::ReceiveFixed);
  const auto s = ScenarioGenerator(model).generate(TimeGrid({0.25, 0.5, 0.75, 1.0}), {40000, 3, true});
  const Matrix v = swap.valueCube(s);
  const double v0 = swap.npv(*yc);
  CHECK_NEAR(v(0, 0), v0, 1e-6);
  CHECK_NEAR(deflatedMean(s, v, 1), v0, 30.0);
  CHECK_NEAR(deflatedMean(s, v, 2), v0, 30.0);
  CHECK_NEAR(deflatedMean(s, v, 3), v0, 30.0);
}

TEST(fx_forward_and_option) {
  const auto yc = curve();
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto model = std::make_shared<HullWhite1F>(yc, 0.05, 1e-8);
  const auto fx = std::make_shared<LognormalAsset>("FX", 1.2, 0.1, foreign);
  const ScenarioGenerator gen(model, {fx});
  const auto s = gen.generate(TimeGrid::uniform(2.0, 8), {20000, 4, true});

  const AssetForward fwd("fwd", "FX", 1e6, 1.25, 2.0);
  const Matrix v = fwd.valueCube(s);
  const double v0 = 1e6 * (1.2 * foreign->discount(2.0) - 1.25 * yc->discount(2.0));
  CHECK_NEAR(v(0, 0), v0, 1e-6);
  CHECK_NEAR(deflatedMean(s, v, 4), v0, 1500.0);

  const EuropeanOption call("call", "FX", OptionType::Call, 1e6, 1.25, 2.0);
  const Matrix c = call.valueCube(s);
  const double forward = 1.2 * foreign->discount(2.0) / yc->discount(2.0);
  const double analytic = 1e6 * blackFormula(OptionType::Call, forward, 1.25, 0.1 * std::sqrt(2.0), yc->discount(2.0));
  CHECK_NEAR(c(0, 0), analytic, 1e-6);
  // Martingale property of the option value along the scenarios.
  CHECK_NEAR(deflatedMean(s, c, 4), analytic, 0.01 * analytic);
  // Payoff realised at expiry equals the discounted expectation.
  double payoff = 0.0;
  for (std::size_t p = 0; p < s.numPaths; ++p) payoff += s.deflator(p, 8) * std::max(s.assetValues[0](p, 8) - 1.25, 0.0);
  CHECK_NEAR(1e6 * payoff / s.numPaths, analytic, 0.03 * analytic);
}

TEST(bermudan_with_single_exercise_is_european) {
  const auto yc = curve();
  const auto model = std::make_shared<HullWhite1F>(yc, 0.05, 0.012);
  const auto swap = InterestRateSwap::vanilla("u", 1.0, 0.032, 2.0, 5.0, 2, InterestRateSwap::Direction::PayFixed);
  const BermudanSwaption option("bs", swap, {2.0});
  const auto s = ScenarioGenerator(model).generate(TimeGrid::uniform(7.0, 28).merged(option.eventTimes()),
                                                   {40000, 21, true});
  const auto valuation = option.valueWithExercise(s);
  const double jamshidian = model->europeanSwaption(2.0, swap.paymentTimes(), swap.accruals(), 0.032, true);
  CHECK_NEAR(valuation.price, jamshidian, 0.02 * jamshidian);
  // AMC mark-to-market is a martingale before expiry.
  CHECK_NEAR(deflatedMean(s, valuation.value, s.grid.indexAtOrBefore(1.0)), jamshidian, 0.03 * jamshidian);
}

TEST(bermudan_exceeds_european_and_settlement_drives_exposure) {
  const auto yc = curve();
  const auto model = std::make_shared<HullWhite1F>(yc, 0.03, 0.012);
  const auto swap = InterestRateSwap::vanilla("u", 1.0, 0.03, 1.0, 9.0, 1, InterestRateSwap::Direction::ReceiveFixed);
  std::vector<double> exercises;
  for (int i = 1; i <= 9; ++i) exercises.push_back(static_cast<double>(i));
  const BermudanSwaption physical("p", swap, exercises, Settlement::Physical);
  const BermudanSwaption cash("c", swap, exercises, Settlement::Cash);
  const auto s = ScenarioGenerator(model).generate(TimeGrid::uniform(10.0, 40).merged(physical.eventTimes()),
                                                   {20000, 8, true});
  const auto vp = physical.valueWithExercise(s);
  const auto vc = cash.valueWithExercise(s);

  double bestEuropean = 0.0;
  for (double e : exercises) {
    const auto tail = swap.tail(e);
    bestEuropean = std::max(bestEuropean, model->europeanSwaption(e, tail.paymentTimes(), tail.accruals(), 0.03, false));
  }
  CHECK(vp.price > bestEuropean);
  CHECK_NEAR(vp.price, vc.price, 1e-12);  // same policy and price; only post-exercise values differ

  // After exercise the cash-settled option has no exposure, the physical one carries the swap.
  const std::size_t j = s.grid.indexAtOrBefore(6.5);
  bool sawExercised = false;
  for (std::size_t p = 0; p < s.numPaths; ++p) {
    const std::size_t k = vp.exerciseIndex[p];
    if (k != BermudanSwaption::npos && exercises[k] < 6.5) {
      sawExercised = true;
      CHECK_NEAR(vc.value(p, j), 0.0, 0.0);
    } else {
      CHECK(vc.value(p, j) >= 0.0);
      CHECK_NEAR(vc.value(p, j), vp.value(p, j), 1e-12);
    }
  }
  CHECK(sawExercised);
  // Option values are never negative before exercise.
  for (std::size_t p = 0; p < s.numPaths; ++p) CHECK(vp.value(p, 0) >= 0.0);
}
