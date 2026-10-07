#pragma once

#include <cstddef>
#include <vector>

#include "ccr/core/matrix.hpp"

namespace ccr {

/// Ordinary least squares: coefficients beta minimising |X beta - y|^2,
/// solved through the normal equations with an optional ridge term.
std::vector<double> leastSquares(const Matrix& design, const std::vector<double>& y, double ridge = 0.0);

/// One-dimensional polynomial regression used for the conditional expectations
/// of American Monte Carlo (Longstaff-Schwartz). The regressor is standardised
/// before building the monomial basis to keep the normal equations well conditioned.
class PolynomialRegression {
 public:
  explicit PolynomialRegression(std::size_t degree = 2) : degree_(degree) {}

  /// Fits on all samples.
  void fit(const std::vector<double>& x, const std::vector<double>& y);
  /// Fits on the samples with mask[i] == true.
  void fit(const std::vector<double>& x, const std::vector<double>& y, const std::vector<bool>& mask);

  double predict(double x) const;
  std::vector<double> predict(const std::vector<double>& x) const;

  std::size_t degree() const { return degree_; }
  /// Degree actually used by the last fit (reduced for degenerate samples).
  std::size_t effectiveDegree() const { return coefficients_.empty() ? 0 : coefficients_.size() - 1; }
  const std::vector<double>& coefficients() const { return coefficients_; }

 private:
  std::size_t degree_;
  double center_ = 0.0;
  double scale_ = 1.0;
  std::vector<double> coefficients_;
};

}  // namespace ccr
