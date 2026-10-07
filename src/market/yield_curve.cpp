#include "ccr/market/yield_curve.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ccr {

YieldCurve::YieldCurve(std::vector<double> times, std::vector<double> zeroRates)
    : times_(std::move(times)), rates_(std::move(zeroRates)) {
  if (times_.empty() || times_.size() != rates_.size())
    throw std::invalid_argument("YieldCurve: times and rates must be non-empty and of equal size");
  for (std::size_t i = 1; i < times_.size(); ++i)
    if (times_[i] <= times_[i - 1]) throw std::invalid_argument("YieldCurve: times must be strictly increasing");
}

YieldCurve YieldCurve::flat(double rate) { return YieldCurve({1.0}, {rate}); }

double YieldCurve::zeroRate(double t) const {
  if (t <= times_.front()) return rates_.front();
  if (t >= times_.back()) return rates_.back();
  const auto it = std::upper_bound(times_.begin(), times_.end(), t);
  const std::size_t i = static_cast<std::size_t>(it - times_.begin());
  const double w = (t - times_[i - 1]) / (times_[i] - times_[i - 1]);
  return rates_[i - 1] + w * (rates_[i] - rates_[i - 1]);
}

double YieldCurve::discount(double t) const { return std::exp(-zeroRate(t) * t); }

double YieldCurve::instantaneousForward(double t) const {
  // d/dt [r(t) t] = r(t) + t r'(t), with r' the slope of the linear segment.
  double slope = 0.0;
  if (t >= times_.front() && t < times_.back()) {
    const auto it = std::upper_bound(times_.begin(), times_.end(), t);
    const std::size_t i = static_cast<std::size_t>(it - times_.begin());
    slope = (rates_[i] - rates_[i - 1]) / (times_[i] - times_[i - 1]);
  }
  return zeroRate(t) + t * slope;
}

double YieldCurve::forwardRate(double t1, double t2) const {
  if (t2 <= t1) throw std::invalid_argument("YieldCurve::forwardRate: t2 must exceed t1");
  return (discount(t1) / discount(t2) - 1.0) / (t2 - t1);
}

YieldCurve YieldCurve::shifted(double shift) const {
  std::vector<double> r = rates_;
  for (double& x : r) x += shift;
  return YieldCurve(times_, r);
}

}  // namespace ccr
