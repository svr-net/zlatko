#include "ccr/core/math.hpp"
#include "ccr/core/matrix.hpp"
#include "ccr/core/regression.hpp"
#include "ccr/core/time_grid.hpp"
#include "test_framework.hpp"

using namespace ccr;

TEST(cholesky_reproduces_matrix) {
  Matrix a(3, 3);
  const double values[3][3] = {{1.0, 0.5, 0.2}, {0.5, 1.0, -0.3}, {0.2, -0.3, 1.0}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = values[i][j];
  const Matrix l = cholesky(a);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) sum += l(i, k) * l(j, k);
      CHECK_NEAR(sum, a(i, j), 1e-14);
    }
  Matrix bad = Matrix::identity(2);
  bad(0, 1) = bad(1, 0) = 1.5;
  CHECK_THROWS(cholesky(bad));
}

TEST(linear_solver) {
  Matrix a(3, 3);
  const double values[3][3] = {{0.0, 2.0, 1.0}, {1.0, 1.0, 0.0}, {3.0, 0.0, 1.0}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = values[i][j];
  const auto x = solveLinearSystem(a, {7.0, 3.0, 6.0});
  CHECK_NEAR(x[0], 1.0, 1e-12);
  CHECK_NEAR(x[1], 2.0, 1e-12);
  CHECK_NEAR(x[2], 3.0, 1e-12);
}

TEST(polynomial_regression_recovers_polynomial) {
  std::vector<double> x, y;
  for (int i = 0; i < 50; ++i) {
    const double v = -2.0 + 0.08 * i;
    x.push_back(v);
    y.push_back(1.0 - 2.0 * v + 0.5 * v * v * v);
  }
  PolynomialRegression reg(3);
  reg.fit(x, y);
  CHECK(reg.effectiveDegree() == 3);
  CHECK_NEAR(reg.predict(0.7), 1.0 - 1.4 + 0.5 * 0.343, 1e-8);

  // Degenerate regressor falls back to the sample mean.
  PolynomialRegression flat(3);
  flat.fit(std::vector<double>(10, 0.0), std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  CHECK(flat.effectiveDegree() == 0);
  CHECK_NEAR(flat.predict(123.0), 5.5, 1e-12);
}

TEST(brent_and_normal) {
  const double root = solveBrent([](double x) { return x * x * x - 2.0; }, 0.0, 2.0);
  CHECK_NEAR(root, std::cbrt(2.0), 1e-12);
  CHECK_NEAR(normalCdf(0.0), 0.5, 1e-15);
  CHECK_NEAR(normalCdf(1.959963984540054), 0.975, 1e-12);
  CHECK_NEAR(quantile({1, 2, 3, 4, 5}, 0.5), 3.0, 1e-15);
  CHECK_NEAR(quantile({1, 2, 3, 4, 5}, 0.9), 4.6, 1e-12);
  // Put-call parity of the Black formula.
  const double c = blackFormula(OptionType::Call, 105.0, 100.0, 0.2, 0.95);
  const double p = blackFormula(OptionType::Put, 105.0, 100.0, 0.2, 0.95);
  CHECK_NEAR(c - p, 0.95 * 5.0, 1e-12);
}

TEST(time_grid_construction) {
  const TimeGrid g({0.5, 1.0, 0.25, 1.0, -1.0});
  CHECK(g.size() == 4);
  CHECK_NEAR(g[0], 0.0, 0.0);
  CHECK_NEAR(g.horizon(), 1.0, 0.0);
  CHECK(g.indexAtOrBefore(0.3) == 1);
  CHECK(g.indexAtOrBefore(0.5) == 2);
  CHECK(g.find(0.25).has_value());
  CHECK(!g.find(0.3).has_value());

  const TimeGrid lagged = g.withLaggedTimes(0.1);
  CHECK(lagged.find(0.15).has_value());
  CHECK(lagged.find(0.9).has_value());

  const TimeGrid std5 = TimeGrid::standardExposureGrid(5.0);
  CHECK_NEAR(std5.horizon(), 5.0, 1e-12);
  CHECK(std5.find(1.0).has_value());
}
