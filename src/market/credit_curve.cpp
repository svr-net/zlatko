#include "ccr/market/credit_curve.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ccr {

CreditCurve::CreditCurve(std::vector<double> pillars, std::vector<double> hazards)
    : pillars_(std::move(pillars)), hazards_(std::move(hazards)) {
  if (pillars_.empty() || pillars_.size() != hazards_.size())
    throw std::invalid_argument("CreditCurve: pillars and hazards must be non-empty and of equal size");
  for (std::size_t i = 0; i < pillars_.size(); ++i) {
    if (pillars_[i] <= (i == 0 ? 0.0 : pillars_[i - 1]))
      throw std::invalid_argument("CreditCurve: pillars must be positive and strictly increasing");
    if (hazards_[i] < 0.0) throw std::invalid_argument("CreditCurve: hazard rates must be non-negative");
  }
}

CreditCurve CreditCurve::flat(double hazard) { return CreditCurve({1.0}, {hazard}); }

double CreditCurve::survival(double t) const {
  if (t <= 0.0) return 1.0;
  double integral = 0.0;
  double previous = 0.0;
  for (std::size_t i = 0; i < pillars_.size(); ++i) {
    const bool last = i + 1 == pillars_.size();
    const double end = last ? t : std::min(t, pillars_[i]);
    integral += hazards_[i] * (end - previous);
    if (end >= t) break;
    previous = end;
  }
  return std::exp(-integral);
}

double CreditCurve::hazard(double t) const {
  for (std::size_t i = 0; i < pillars_.size(); ++i)
    if (t <= pillars_[i]) return hazards_[i];
  return hazards_.back();
}

}  // namespace ccr
