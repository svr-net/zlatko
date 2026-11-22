// Unit tests of adaptive nested Monte Carlo (chapter X11): the generic estimator on a model with a
// known conditional value, and the netting-set exposure against the closed forms.
#include <cmath>
#include <memory>
#include <numeric>

#include "ccr/ccr.hpp"
#include "test_framework.hpp"

using namespace ccr;
using namespace ccr::nested;

namespace {

// A one-factor toy nest: X ~ N(0, 1), V(x) = x^2 - 0.5 + 0.3 x, Y = V(X) + 2 Z. A degree-1 proxy cannot
// represent V, so the regression alone is biased; the inner simulation adds 2 Z to V(x) exactly.
double toyValue(double x) { return x * x - 0.5 + 0.3 * x; }

OuterSample toySample(std::size_t n, std::uint64_t seed) {
  OuterSample s;
  s.features = Matrix(n, 1);
  CounterNormals rng(seed);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = rng.next();
    s.features(i, 0) = x;
    s.payoff.push_back(toyValue(x) + 2.0 * rng.next());
  }
  return s;
}

InnerSimulator toyInner(const OuterSample& s) {
  return [&s](std::size_t i, std::uint64_t seed, double* out, std::size_t n) {
    CounterNormals rng(seed);
    for (std::size_t k = 0; k < n; ++k) out[k] = toyValue(s.features(i, 0)) + 2.0 * rng.next();
  };
}

std::vector<double> exactOn(const OuterSample& s, const std::vector<std::size_t>& rows) {
  std::vector<double> v;
  for (std::size_t i : rows) v.push_back(toyValue(s.features(i, 0)));
  return v;
}

}  // namespace

TEST(nested_counter_normals_are_standard_and_reproducible) {
  CounterNormals a(7), b(7), c(8);
  double sum = 0.0, sq = 0.0;
  const int n = 200000;
  for (int i = 0; i < n; ++i) {
    const double z = a.next();
    CHECK(z == b.next());
    sum += z;
    sq += z * z;
  }
  CHECK_NEAR(sum / n, 0.0, 0.01);
  CHECK_NEAR(sq / n, 1.0, 0.01);
  CHECK(CounterNormals(7).next() != c.next());
  CHECK(deriveSeed(1, 2, 3) != deriveSeed(1, 3, 2) && deriveSeed(1, 2, 3) == deriveSeed(1, 2, 3));
}

TEST(nested_hermite_proxy_recovers_a_polynomial_with_its_estimation_variance) {
  // y = 1 + 2 x1 - x1 x2 + 0.5 x2^3 exactly (no noise): degree 3 fits it and the HC0 variance is ~0.
  const std::size_t n = 400;
  Matrix f(n, 3);
  std::vector<double> y(n);
  CounterNormals rng(3);
  std::vector<std::size_t> rows(n);
  std::iota(rows.begin(), rows.end(), std::size_t{0});
  for (std::size_t i = 0; i < n; ++i) {
    const double x1 = rng.next(), x2 = 2.0 + rng.next();
    f(i, 0) = x1;
    f(i, 1) = x2;
    f(i, 2) = 5.0;  // no spread: dropped
    y[i] = 1.0 + 2.0 * x1 - x1 * x2 + 0.5 * x2 * x2 * x2;
  }
  HermiteProxy p(3);
  p.fit(f, y, rows);
  CHECK(p.degree() == 3 && p.numBasis() == 10);  // two active features
  const double x[3] = {0.3, 1.7, 5.0};
  double v = 1.0;
  CHECK_NEAR(p.predict(x, v), 1.0 + 0.6 - 0.51 + 0.5 * 1.7 * 1.7 * 1.7, 1e-6);
  CHECK(v < 1e-12);
  // With noise the variance is positive and shrinks with the sample; few rows lower the degree.
  for (std::size_t i = 0; i < n; ++i) y[i] += rng.next();
  p.fit(f, y, rows);
  double v400 = 0.0;
  p.predict(x, v400);
  const std::vector<std::size_t> head(rows.begin(), rows.begin() + 100);
  p.fit(f, y, head);
  double v100 = 0.0;
  p.predict(x, v100);
  CHECK(v400 > 0.0 && v100 > 2.0 * v400);
  const std::vector<std::size_t> few(rows.begin(), rows.begin() + 12);
  p.fit(f, y, few);
  CHECK(p.degree() == 2 && p.numBasis() == 6);  // 12 rows: at most 6 basis functions
}

TEST(nested_scenario_estimates_keep_selection_and_value_samples_apart) {
  // One scenario, proxy 10 with variance 4, outer payoff 13, residual variance 9.
  ScenarioEstimates s({13.0}, {1.0}, {10.0}, {4.0}, {0}, {9.0}, 2, 1000);
  CHECK(s.innerPaths(0) == 0);
  CHECK_NEAR(s.selectionEstimate(0), 10.0, 0.0);  // no fold-1 samples: the proxy alone
  CHECK_NEAR(s.valueEstimate(0), 13.0, 0.0);      // the outer payoff
  CHECK_NEAR(s.estimate(0), 10.0 + 4.0 / 13.0 * 3.0, 1e-12);  // precision blend of 10 (var 4) and 13 (var 9)
  const double first[4] = {16.0, 18.0, 22.0, 20.0};
  s.addInnerBatch(0, first, 4);  // 16, 18 -> fold 0 (selection); 22, 20 -> fold 1 (values)
  CHECK(s.innerPaths(0) == 4 && s.batches(0) == 1);
  CHECK_NEAR(s.valueEstimate(0), 21.0, 1e-12);
  // Selection blends the proxy with fold 0 (the outer payoff, 16 and 18: mean 47 / 3), never with fold 1.
  const double w = s.withinVariance(0);
  CHECK_NEAR(s.selectionEstimate(0), 10.0 + 4.0 / (4.0 + w / 3.0) * (47.0 / 3.0 - 10.0), 1e-12);
  const double second[2] = {30.0, 32.0};
  s.addInnerBatch(0, second, 2);  // 30 -> fold 0, 32 -> fold 1
  CHECK_NEAR(s.valueEstimate(0), (22.0 + 20.0 + 32.0) / 3.0, 1e-12);
  CHECK_NEAR(s.valueVariance(0), s.withinVariance(0) / 3.0, 1e-12);
  // Pooled sample: 13, 16, 18, 22, 20, 30, 32: mean 151 / 7, sample variance as the textbook formula.
  const std::vector<double> all{13, 16, 18, 22, 20, 30, 32};
  const double m = 151.0 / 7.0;
  double ss = 0.0;
  for (double a : all) ss += (a - m) * (a - m);
  CHECK_NEAR(s.withinVariance(0), ss / 6.0, 1e-9);
  CHECK(s.refinementGain(0) > 0.0 && s.refinementGain(0) < 1.0);
}

TEST(nested_adaptive_estimator_corrects_a_misspecified_proxy) {
  const OuterSample sample = toySample(4000, 11);
  const InnerSimulator inner = toyInner(sample);
  const PositivePart ee(0.01, 0.0, "EE");
  const Quantile q95(0.95, false, 2.0, "q95");
  AdaptiveOptions o;
  o.hermiteDegree = 1;  // V is quadratic: the proxy is wrong in both tails
  o.innerBudget = 200000;
  const AdaptiveResult r = estimateAdaptive(sample, inner, {&ee, &q95}, o);
  const std::vector<double> exact = exactOn(sample, r.evaluation);
  const double eeExact = ee.plugIn(exact, {}), qExact = q95.plugIn(exact, {});
  CHECK(r.evaluation.size() == 2000 && r.training.size() == 2000);
  CHECK(r.innerPaths <= o.innerBudget && r.auditInnerPaths > 0 && r.refinedScenarios > 0);
  // The proxy alone misses the quantile by far; the adaptive nest gets within 2%.
  CHECK(std::fabs(r.regression[1] - qExact) > 0.15 * qExact);
  CHECK_NEAR(r.estimate[1], qExact, 0.02 * qExact);
  CHECK_NEAR(r.estimate[0], eeExact, 0.02 * eeExact);
  // The audit found the bias where the linear proxy misses the parabola: in the tails.
  CHECK(r.bins.front().biasRms > 0.3 && r.bins.back().biasRms > 0.3);
  // A classical nest with the same inner paths on the same scenarios has its plug-in bias.
  const UniformResult u = estimateUniform(sample, inner, r.evaluation, r.innerPaths / r.evaluation.size(), {&ee, &q95}, 5);
  CHECK(u.innerPaths <= r.innerPaths);
  CHECK(std::fabs(r.estimate[1] - qExact) < std::fabs(u.estimate[1] - qExact));
}

TEST(nested_adaptive_estimator_is_reproducible_on_any_number_of_threads) {
  const OuterSample sample = toySample(1200, 5);
  const InnerSimulator inner = toyInner(sample);
  const PositivePart ee;
  const Quantile q(0.9);
  AdaptiveOptions o;
  o.innerBudget = 40000;
  SpinPool serial(0), parallel(3);
  const AdaptiveResult a = estimateAdaptive(sample, inner, {&ee, &q}, o, serial);
  const AdaptiveResult b = estimateAdaptive(sample, inner, {&ee, &q}, o, parallel);
  CHECK(a.estimate == b.estimate && a.innerPaths == b.innerPaths && a.value == b.value && a.inner == b.inner);
  o.seed += 1;
  const AdaptiveResult c = estimateAdaptive(sample, inner, {&ee, &q}, o, parallel);
  CHECK(c.estimate != a.estimate);
  // Settings are validated.
  o.trainingFraction = 1.0;
  CHECK_THROWS(estimateAdaptive(sample, inner, {&ee}, o));
}

TEST(nested_exposure_matches_the_closed_forms) {
  // A payer swap and an FX forward on correlated Hull-White and FX: both have exact values on the
  // scenarios, so EE* and PFE of the nest are compared with the closed form on the same scenarios.
  const auto yc = std::make_shared<YieldCurve>(YieldCurve::flat(0.03));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 0.01);
  const auto fx = std::make_shared<LognormalAsset>("FX", 1.1, 0.1, std::make_shared<YieldCurve>(YieldCurve::flat(0.01)));
  Matrix corr = Matrix::identity(2);
  corr(0, 1) = corr(1, 0) = 0.3;
  const ScenarioGenerator gen(hw, {fx}, {}, corr);
  NettingSet ns;
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("IRS", 1e6, 0.03, 0.0, 4.0, 2, InterestRateSwap::Direction::PayFixed)));
  ns.trades.push_back(std::make_shared<AssetForward>("fwd", "FX", 1e6, 1.1, 2.0));
  NestedExposureOptions o;
  o.adaptive.innerBudget = 60000;
  const NestedExposureResult r = nestedExposure(gen, ns, TimeGrid({0.0, 0.75, 1.5, 3.25, 4.0}), {1500, 3, true}, o);
  CHECK(r.points.size() == 5 && r.unsupported.empty() && r.outerPaths == 1500);
  // t = 0: every path in the same state, plain Monte Carlo on the outer payoffs.
  CHECK(r.points[0].innerPaths == 0 && r.points[0].evaluated == 1500);
  CHECK_NEAR(r.points[0].pfeAdaptive, r.points[0].pfeExact, 0.1 * r.points[1].pfeExact + 1e3);
  for (std::size_t k = 1; k < 4; ++k) {
    const auto& p = r.points[k];
    CHECK(p.live && p.innerPaths > 0 && p.innerPaths <= o.adaptive.innerBudget);
    CHECK_NEAR(p.eeAdaptive, p.eeExact, 0.04 * p.eeExact);
    CHECK_NEAR(p.pfeAdaptive, p.pfeExact, 0.04 * p.pfeExact);
    CHECK(p.uniformInnerPerScenario >= 2);
  }
  // At maturity nothing is left.
  CHECK(!r.points[4].live && r.points[4].eeExact == 0.0 && r.points[4].eeAdaptive == 0.0);
  CHECK(r.adaptive.times.size() == 5 && r.adaptive.discountedExpectedExposure[2] == r.points[2].eeAdaptive);
}

TEST(nested_exposure_lists_unsupported_trades_and_refuses_an_empty_netting_set) {
  class Opaque : public Trade {
   public:
    Opaque() : Trade("opaque") {}
    double maturity() const override { return 1.0; }
    Matrix valueCube(const ScenarioSet& s) const override { return Matrix(s.numPaths, s.numTimes()); }
  };
  const auto hw = std::make_shared<HullWhite1F>(std::make_shared<YieldCurve>(YieldCurve::flat(0.02)), 0.05, 0.01);
  const ScenarioGenerator gen(hw);
  NettingSet ns;
  ns.trades.push_back(std::make_shared<Opaque>());
  NestedExposureOptions o;
  CHECK_THROWS(nestedExposure(gen, ns, TimeGrid({0.0, 0.5}), {200, 1, true}, o));
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("IRS", 1e6, 0.02, 0.0, 1.0, 2, InterestRateSwap::Direction::ReceiveFixed)));
  o.adaptive.innerBudget = 5000;
  const NestedExposureResult r = nestedExposure(gen, ns, TimeGrid({0.0, 0.25}), {400, 1, true}, o);
  CHECK(r.unsupported.size() == 1 && r.unsupported[0] == "opaque");
}
