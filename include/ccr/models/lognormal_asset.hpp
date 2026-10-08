#pragma once

#include <memory>
#include <string>

#include "ccr/market/yield_curve.hpp"

namespace ccr {

/// Log-normal risk factor for FX rates and equities under the domestic risk-neutral measure:
///
///   dS / S = (r_d(t) - q(t)) dt + sigma dW,
///
/// where r_d is the simulated domestic short rate and q is read from `carryCurve`
/// (the foreign interest-rate curve for FX, the dividend/repo curve for equities).
/// S(t) D(0, t) / Q(0, t) is an exact martingale, whatever the correlation with rates.
class LognormalAsset {
 public:
  LognormalAsset(std::string name, double spot, double volatility, std::shared_ptr<const YieldCurve> carryCurve);

  const std::string& name() const { return name_; }
  double spot() const { return spot_; }
  double volatility() const { return volatility_; }
  const YieldCurve& carryCurve() const { return *carry_; }

  /// Carry discount factor Q(t, T) = Q(0, T) / Q(0, t) (foreign discount factor for FX).
  double carryDiscount(double t, double T) const { return carry_->discount(T) / carry_->discount(t); }

  /// Exact step from s0 at t0 to t1 given the integrated domestic short rate over the step.
  double evolve(double t0, double t1, double s0, double integratedDomesticRate, double z) const;

  /// Copy with a different spot (used for bump-and-revalue sensitivities).
  LognormalAsset withSpot(double spot) const { return LognormalAsset(name_, spot, volatility_, carry_); }
  LognormalAsset withVolatility(double vol) const { return LognormalAsset(name_, spot_, vol, carry_); }

 private:
  std::string name_;
  double spot_;
  double volatility_;
  std::shared_ptr<const YieldCurve> carry_;
};

}  // namespace ccr
