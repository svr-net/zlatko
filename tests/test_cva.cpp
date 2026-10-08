#include <cmath>
#include <memory>

#include "ccr/cva/cva.hpp"
#include "ccr/exposure/exposure_engine.hpp"
#include "ccr/instruments/asset_forward.hpp"
#include "ccr/instruments/credit_default_swap.hpp"
#include "test_framework.hpp"

using namespace ccr;

TEST(cva_of_constant_exposure) {
  const CreditCurve cc({2.0, 5.0}, {0.01, 0.03});
  std::vector<double> times, ee;
  for (int i = 0; i <= 20; ++i) {
    times.push_back(0.25 * i);
    ee.push_back(100.0);
  }
  CHECK_NEAR(unilateralCva(times, ee, cc, 0.4), 0.6 * 100.0 * (1.0 - cc.survival(5.0)), 1e-10);
  const auto terms = cvaTermStructure(times, ee, cc, 0.4);
  CHECK_NEAR(terms[0], 0.0, 0.0);
  CHECK_NEAR(terms[1], 60.0 * cc.defaultProbability(0.0, 0.25), 1e-12);
}

TEST(bilateral_cva_limits) {
  ExposureProfile profile;
  for (int i = 0; i <= 10; ++i) {
    profile.times.push_back(0.5 * i);
    profile.discountedExpectedExposure.push_back(50.0);
    profile.discountedExpectedNegativeExposure.push_back(-30.0);
  }
  const CreditCurve cpty = CreditCurve::flat(0.02);
  const CreditCurve riskless = CreditCurve::flat(0.0);
  const auto noOwnRisk = bilateralCva(profile, cpty, 0.4, riskless, 0.4);
  CHECK_NEAR(noOwnRisk.cva, unilateralCva(profile, cpty, 0.4), 1e-12);
  CHECK_NEAR(noOwnRisk.dva, 0.0, 1e-15);

  const auto both = bilateralCva(profile, cpty, 0.4, CreditCurve::flat(0.01), 0.4);
  CHECK(both.cva < noOwnRisk.cva);  // we may default first
  CHECK(both.dva > 0.0);
  CHECK_NEAR(both.total(), both.cva - both.dva, 1e-15);
}

TEST(cva_running_spread) {
  const YieldCurve yc = YieldCurve::flat(0.0);
  const CreditCurve cc = CreditCurve::flat(0.0);
  // Zero rates and no default: annuity = maturity.
  CHECK_NEAR(cvaRunningSpread(5000.0, yc, cc, 5.0, 1e6), 0.001, 1e-12);
}

namespace {

struct WwrSetup {
  ScenarioSet scenarios;
  Matrix exposure;
  std::shared_ptr<const CreditCurve> market;
};

WwrSetup wrongWayScenarios(double correlation) {
  const auto yc = std::make_shared<YieldCurve>(YieldCurve::flat(0.02));
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto market = std::make_shared<CreditCurve>(CreditCurve::flat(0.03));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 0.005);
  const auto fx = std::make_shared<LognormalAsset>("FX", 1.0, 0.15, foreign);
  const auto cir = std::make_shared<CirIntensity>("C", market, 0.3, 0.03, 0.15, 0.03, 4);
  Matrix corr = Matrix::identity(3);
  corr(1, 2) = corr(2, 1) = correlation;
  const ScenarioGenerator gen(hw, {fx}, {cir}, corr);
  const AssetForward fwd("f", "FX", 1e6, 1.0, 5.0);
  auto s = gen.generate(TimeGrid::uniform(5.0, 20).merged(fwd.eventTimes()), {20000, 31, true});
  Matrix v = fwd.valueCube(s);
  return {std::move(s), std::move(v), market};
}

}  // namespace

TEST(pathwise_cva_and_wrong_way_risk) {
  const auto independent = wrongWayScenarios(0.0);
  const auto& s = independent.scenarios;
  const auto profile = ExposureProfile::compute(independent.exposure, s.grid, s.deflator);
  const double cvaStandard = unilateralCva(profile, *independent.market, 0.4);
  const double cvaPathwise = pathwiseCva(s, independent.exposure, 0, 0.4);
  CHECK_NEAR(cvaPathwise, cvaStandard, 0.05 * cvaStandard);

  // Long the asset forward: exposure grows when the asset rallies. Positive correlation between
  // the asset and the counterparty's intensity is wrong-way risk, negative is right-way risk.
  const auto wrong = wrongWayScenarios(0.8);
  const auto right = wrongWayScenarios(-0.8);
  const double cvaWrong = pathwiseCva(wrong.scenarios, wrong.exposure, 0, 0.4);
  const double cvaRight = pathwiseCva(right.scenarios, right.exposure, 0, 0.4);
  CHECK(cvaWrong > 1.1 * cvaPathwise);
  CHECK(cvaRight < 0.9 * cvaPathwise);
}
