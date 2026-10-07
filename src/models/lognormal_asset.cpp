#include "ccr/models/lognormal_asset.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ccr {

LognormalAsset::LognormalAsset(std::string name, double spot, double volatility,
                               std::shared_ptr<const YieldCurve> carryCurve)
    : name_(std::move(name)), spot_(spot), volatility_(volatility), carry_(std::move(carryCurve)) {
  if (spot_ <= 0.0) throw std::invalid_argument("LognormalAsset: spot must be positive");
  if (volatility_ < 0.0) throw std::invalid_argument("LognormalAsset: volatility must be non-negative");
  if (!carry_) throw std::invalid_argument("LognormalAsset: null carry curve");
}

double LognormalAsset::evolve(double t0, double t1, double s0, double integratedDomesticRate, double z) const {
  const double dt = t1 - t0;
  const double integratedCarry = std::log(carry_->discount(t0) / carry_->discount(t1));
  const double drift = integratedDomesticRate - integratedCarry - 0.5 * volatility_ * volatility_ * dt;
  return s0 * std::exp(drift + volatility_ * std::sqrt(dt) * z);
}

}  // namespace ccr
