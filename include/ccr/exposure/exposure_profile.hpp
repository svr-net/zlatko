#pragma once

#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/core/time_grid.hpp"

namespace ccr {

/// Exposure statistics of a netting set across the simulation grid.
///
/// With V the (collateralised) netting-set value:
///   EE(t)  = E[max(V(t), 0)]                 expected (positive) exposure
///   ENE(t) = E[min(V(t), 0)]                 expected negative exposure (<= 0)
///   PFE(t) = q-quantile of max(V(t), 0)      potential future exposure
///   EE*(t) = E[D(0, t) max(V(t), 0)]         discounted EE, the CVA integrand
/// and the regulatory measures EPE, effective EE and effective EPE.
struct ExposureProfile {
  std::vector<double> times;
  std::vector<double> expectedValue;
  std::vector<double> expectedExposure;
  std::vector<double> expectedNegativeExposure;
  std::vector<double> potentialFutureExposure;
  std::vector<double> discountedExpectedExposure;
  std::vector<double> discountedExpectedNegativeExposure;
  double pfeQuantile = 0.95;

  static ExposureProfile compute(const Matrix& values, const TimeGrid& grid, const Matrix& deflator,
                                 double pfeQuantile = 0.95);

  /// Time-averaged EE over [0, min(horizon, last date)] (right-point rule on the grid).
  double expectedPositiveExposure(double horizon) const;
  /// Effective EE: running maximum of EE (non-decreasing).
  std::vector<double> effectiveExpectedExposure() const;
  /// Effective EPE: time average of effective EE over the first `horizon` years (Basel: one year).
  double effectiveExpectedPositiveExposure(double horizon = 1.0) const;
  /// Peak PFE over the grid.
  double maxPotentialFutureExposure() const;

 private:
  static double timeAverage(const std::vector<double>& times, const std::vector<double>& values, double horizon);
};

}  // namespace ccr
