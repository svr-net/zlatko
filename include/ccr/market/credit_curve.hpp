#pragma once

#include <vector>

namespace ccr {

/// Survival curve with piecewise-constant hazard rates.
///
/// hazards[i] applies on (pillars[i-1], pillars[i]] (pillars[-1] = 0); the last
/// hazard rate is extrapolated flat beyond the last pillar.
class CreditCurve {
 public:
  CreditCurve(std::vector<double> pillars, std::vector<double> hazards);
  static CreditCurve flat(double hazard);

  /// Q(tau > t).
  double survival(double t) const;
  double hazard(double t) const;
  /// Q(t1 < tau <= t2).
  double defaultProbability(double t1, double t2) const { return survival(t1) - survival(t2); }

  const std::vector<double>& pillars() const { return pillars_; }
  const std::vector<double>& hazards() const { return hazards_; }

 private:
  std::vector<double> pillars_;
  std::vector<double> hazards_;
};

}  // namespace ccr
