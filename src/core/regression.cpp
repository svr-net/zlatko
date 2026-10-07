#include "ccr/core/regression.hpp"

#include <cmath>
#include <stdexcept>

namespace ccr {

std::vector<double> leastSquares(const Matrix& design, const std::vector<double>& y, double ridge) {
  const std::size_t n = design.rows();
  const std::size_t k = design.cols();
  if (y.size() != n) throw std::invalid_argument("leastSquares: dimension mismatch");
  if (n < k) throw std::invalid_argument("leastSquares: fewer observations than regressors");
  Matrix xtx(k, k, 0.0);
  std::vector<double> xty(k, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double* row = design.row(i);
    for (std::size_t a = 0; a < k; ++a) {
      xty[a] += row[a] * y[i];
      for (std::size_t b = 0; b <= a; ++b) xtx(a, b) += row[a] * row[b];
    }
  }
  for (std::size_t a = 0; a < k; ++a) {
    for (std::size_t b = 0; b < a; ++b) xtx(b, a) = xtx(a, b);
    xtx(a, a) += ridge;
  }
  return solveLinearSystem(xtx, xty);
}

void PolynomialRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
  fit(x, y, std::vector<bool>(x.size(), true));
}

void PolynomialRegression::fit(const std::vector<double>& x, const std::vector<double>& y,
                               const std::vector<bool>& mask) {
  if (x.size() != y.size() || x.size() != mask.size())
    throw std::invalid_argument("PolynomialRegression::fit: size mismatch");

  std::size_t n = 0;
  double sx = 0.0, sxx = 0.0, sy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!mask[i]) continue;
    ++n;
    sx += x[i];
    sxx += x[i] * x[i];
    sy += y[i];
  }
  if (n == 0) {
    center_ = 0.0;
    scale_ = 1.0;
    coefficients_ = {0.0};
    return;
  }
  center_ = sx / static_cast<double>(n);
  const double variance = std::max(sxx / static_cast<double>(n) - center_ * center_, 0.0);
  scale_ = std::sqrt(variance);

  // Degenerate regressor (e.g. all paths start from the same state at t = 0):
  // the conditional expectation is the plain sample mean.
  std::size_t degree = degree_;
  if (scale_ < 1e-12 * std::max(1.0, std::fabs(center_))) degree = 0;
  if (n < 2 * (degree + 1)) degree = n >= 4 ? (n / 2) - 1 : 0;
  if (degree == 0) {
    scale_ = 1.0;
    coefficients_ = {sy / static_cast<double>(n)};
    return;
  }

  Matrix design(n, degree + 1);
  std::vector<double> target(n);
  std::size_t row = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!mask[i]) continue;
    const double z = (x[i] - center_) / scale_;
    double power = 1.0;
    for (std::size_t d = 0; d <= degree; ++d) {
      design(row, d) = power;
      power *= z;
    }
    target[row] = y[i];
    ++row;
  }
  coefficients_ = leastSquares(design, target, 1e-12 * static_cast<double>(n));
}

double PolynomialRegression::predict(double x) const {
  const double z = (x - center_) / scale_;
  double result = 0.0;
  for (std::size_t d = coefficients_.size(); d-- > 0;) result = result * z + coefficients_[d];
  return result;
}

std::vector<double> PolynomialRegression::predict(const std::vector<double>& x) const {
  std::vector<double> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = predict(x[i]);
  return out;
}

}  // namespace ccr
