#include <cmath>
#include <vector>

#include "ccr/ai/surrogate.hpp"
#include "test_framework.hpp"

using namespace ccr;

namespace {

// A smooth five-input, six-output function standing in for the XVA engine.
std::vector<double> target(const std::vector<double>& x) {
  return {1 + 2 * x[0] + 0.5 * x[0] * x[1], 0.3 * x[1], 1 + x[2] * x[2], 0.2 * x[3], 0.5 + x[4], 0.1 * x[0] * x[4] + 2};
}

ai::Rows grid(std::size_t n, double scale) {
  ai::Rows x;
  std::uint32_t s = 12345u;
  auto u = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) / 16777216.0; };
  for (std::size_t i = 0; i < n; ++i) x.push_back({scale * u(), scale * u(), scale * u(), scale * u(), scale * u()});
  return x;
}

}  // namespace

TEST(ai_ridge_recovers_a_quadratic_exactly) {
  const auto X = grid(200, 1.0);
  ai::Rows Y;
  for (const auto& x : X) Y.push_back(target(x));
  const auto quad = ai::fitRidge(X, Y, 2, 1e-12);
  const auto lin = ai::fitRidge(X, Y, 1, 1e-12);
  CHECK(quad.parameters() == 6 * 21 && lin.parameters() == 6 * 6);
  for (const auto& x : grid(20, 1.0)) {
    const auto p = quad.predict(x), t = target(x);
    for (std::size_t k = 0; k < 6; ++k) CHECK_NEAR(p[k], t[k], 1e-6);
  }
}

TEST(ai_mlp_learns_and_is_identical_on_any_number_of_threads) {
  const auto X = grid(120, 1.0);
  ai::Rows Y;
  for (const auto& x : X) Y.push_back(target(x));
  ai::MlpOptions o;
  o.epochs = 600;
  o.seed = 7;
  SpinPool one(0), eight(7);
  const auto a = ai::fitMlp(X, Y, o, one), b = ai::fitMlp(X, Y, o, eight);
  CHECK(a.W[1] == b.W[1] && a.B[2] == b.B[2]);  // fixed chunks: the same sums in the same order
  double mse = 0.0, var = 0.0;
  for (const auto& x : grid(40, 1.0)) {
    const auto p = a.predict(x), t = target(x);
    for (std::size_t k = 0; k < 6; ++k) mse += (p[k] - t[k]) * (p[k] - t[k]), var += t[k] * t[k];
  }
  CHECK(mse < 0.01 * var);
  CHECK(a.parameters() == 5 * 16 + 16 + 16 * 16 + 16 + 16 * 6 + 6);
}

TEST(ai_backtest_scores_models_and_baselines) {
  ai::Dataset d;
  d.trainX = grid(150, 1.0);
  d.testX = grid(60, 1.3);  // partly outside the training box
  for (const auto& x : d.trainX) d.trainY.push_back(target(x));
  for (const auto& x : d.testX) d.testY.push_back(target(x));
  d.low.assign(5, 0.0);
  d.high.assign(5, 1.0);
  d.grossNotional = 1e3;
  ai::BacktestOptions o;
  o.mlpEpochs = 300;
  o.mlpRefitEpochs = 50;
  const auto bt = ai::runBacktest(d, o);
  CHECK(bt.days == 60 && bt.outsideDays > 0 && bt.models.size() == 3);
  const auto& quad = bt.models[1];
  CHECK(quad.key == "poly2" && quad.fittedOnce.total.maeBp < 1e-3);  // the target is quadratic
  CHECK(bt.models[0].fittedOnce.total.mae > quad.fittedOnce.total.mae);
  CHECK(quad.fittedOnce.errorTotal.size() == 60 && quad.fittedOnce.pnl.effectiveness > 0.999);
  CHECK(bt.lag1.total.mae > 0.0 && bt.weekly.inRange.has_value());
}
