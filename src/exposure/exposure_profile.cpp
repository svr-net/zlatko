#include "ccr/exposure/exposure_profile.hpp"

#include <algorithm>
#include <stdexcept>

#include "ccr/core/math.hpp"

namespace ccr {

ExposureProfile ExposureProfile::compute(const Matrix& values, const TimeGrid& grid, const Matrix& deflator,
                                         double pfeQuantile) {
  if (values.cols() != grid.size() || deflator.cols() != grid.size() || deflator.rows() != values.rows())
    throw std::invalid_argument("ExposureProfile::compute: dimension mismatch");
  const std::size_t nP = values.rows();
  const std::size_t nT = grid.size();
  const double n = static_cast<double>(nP);

  ExposureProfile profile;
  profile.pfeQuantile = pfeQuantile;
  profile.times = grid.times();
  profile.expectedValue.resize(nT);
  profile.expectedExposure.resize(nT);
  profile.expectedNegativeExposure.resize(nT);
  profile.potentialFutureExposure.resize(nT);
  profile.discountedExpectedExposure.resize(nT);
  profile.discountedExpectedNegativeExposure.resize(nT);

  std::vector<double> positive(nP);
  for (std::size_t j = 0; j < nT; ++j) {
    double sum = 0.0, ee = 0.0, ene = 0.0, dee = 0.0, dene = 0.0;
    for (std::size_t p = 0; p < nP; ++p) {
      const double v = values(p, j);
      const double d = deflator(p, j);
      sum += v;
      positive[p] = std::max(v, 0.0);
      ee += positive[p];
      ene += std::min(v, 0.0);
      dee += d * positive[p];
      dene += d * std::min(v, 0.0);
    }
    profile.expectedValue[j] = sum / n;
    profile.expectedExposure[j] = ee / n;
    profile.expectedNegativeExposure[j] = ene / n;
    profile.discountedExpectedExposure[j] = dee / n;
    profile.discountedExpectedNegativeExposure[j] = dene / n;
    profile.potentialFutureExposure[j] = quantile(positive, pfeQuantile);
  }
  return profile;
}

double ExposureProfile::timeAverage(const std::vector<double>& times, const std::vector<double>& values,
                                    double horizon) {
  double integral = 0.0;
  double end = 0.0;
  for (std::size_t j = 1; j < times.size() && times[j] <= horizon + 1e-10; ++j) {
    integral += values[j] * (times[j] - times[j - 1]);
    end = times[j];
  }
  return end > 0.0 ? integral / end : (values.empty() ? 0.0 : values.front());
}

double ExposureProfile::expectedPositiveExposure(double horizon) const {
  return timeAverage(times, expectedExposure, horizon);
}

std::vector<double> ExposureProfile::effectiveExpectedExposure() const {
  std::vector<double> eff(expectedExposure.size());
  double running = 0.0;
  for (std::size_t j = 0; j < eff.size(); ++j) {
    running = std::max(running, expectedExposure[j]);
    eff[j] = running;
  }
  return eff;
}

double ExposureProfile::effectiveExpectedPositiveExposure(double horizon) const {
  return timeAverage(times, effectiveExpectedExposure(), horizon);
}

double ExposureProfile::maxPotentialFutureExposure() const {
  return potentialFutureExposure.empty()
             ? 0.0
             : *std::max_element(potentialFutureExposure.begin(), potentialFutureExposure.end());
}

}  // namespace ccr
