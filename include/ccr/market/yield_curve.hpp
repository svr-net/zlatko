#pragma once

#include <vector>

namespace ccr {

/// Zero-coupon yield curve: continuously compounded zero rates, linearly
/// interpolated in time with flat extrapolation at both ends.
class YieldCurve {
 public:
  YieldCurve(std::vector<double> times, std::vector<double> zeroRates);
  static YieldCurve flat(double rate);

  double zeroRate(double t) const;
  /// P(0, t).
  double discount(double t) const;
  /// f(0, t) = -d ln P(0, t) / dt.
  double instantaneousForward(double t) const;
  /// Simply compounded forward rate for [t1, t2].
  double forwardRate(double t1, double t2) const;

  /// Curve with all zero rates shifted by `shift` (parallel shift).
  YieldCurve shifted(double shift) const;

  const std::vector<double>& times() const { return times_; }
  const std::vector<double>& zeroRates() const { return rates_; }

 private:
  std::vector<double> times_;
  std::vector<double> rates_;
};

}  // namespace ccr
