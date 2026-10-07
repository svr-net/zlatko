#pragma once

#include <memory>
#include <vector>

#include "ccr/market/yield_curve.hpp"

namespace ccr {

/// One-factor Hull-White short-rate model in its shifted (G1++) form:
///
///   r(t) = x(t) + phi(t),   dx = -a x dt + sigma dW,   x(0) = 0,
///
/// with phi fitted to the initial discount curve so that zero-coupon bond
/// prices are reproduced exactly. The model drives the domestic interest
/// rates and the numeraire (bank account) of the simulation.
class HullWhite1F {
 public:
  HullWhite1F(std::shared_ptr<const YieldCurve> curve, double meanReversion, double volatility);

  double meanReversion() const { return a_; }
  double volatility() const { return sigma_; }
  const YieldCurve& curve() const { return *curve_; }
  std::shared_ptr<const YieldCurve> curvePtr() const { return curve_; }

  /// B(t, T) = (1 - exp(-a (T - t))) / a.
  double B(double t, double T) const;
  /// Zero-coupon bond P(t, T) given the state x(t).
  double zeroBond(double t, double T, double x) const;
  /// Short rate r(t) = x + phi(t).
  double shortRate(double t, double x) const;
  /// phi(t) = f(0, t) + sigma^2 / (2 a^2) (1 - exp(-a t))^2.
  double phi(double t) const;

  /// Exact joint Gaussian transition of x and of the integrated short rate over [t0, t1].
  /// z1 drives x (and may be correlated with other risk factors), z2 is an independent
  /// normal for the part of the integral not explained by x(t1).
  void evolve(double t0, double t1, double x0, double z1, double z2, double& x1, double& integratedRate) const;

  /// Closed-form zero-coupon bond options at time 0: option expiring at `expiry` on P(expiry, maturity).
  double zeroBondCall(double expiry, double maturity, double strike) const;
  double zeroBondPut(double expiry, double maturity, double strike) const;

  /// European swaption price at time 0 by Jamshidian's decomposition. The underlying swap starts
  /// at `expiry` and pays fixed `fixedRate * accruals[i]` at `paymentTimes[i]`; the floating leg
  /// is valued as 1 - P(expiry, last payment). Payer = right to pay fixed.
  double europeanSwaption(double expiry, const std::vector<double>& paymentTimes, const std::vector<double>& accruals,
                          double fixedRate, bool payer) const;

 private:
  double V(double t, double T) const;
  double bondOptionStdDev(double expiry, double maturity) const;

  std::shared_ptr<const YieldCurve> curve_;
  double a_;
  double sigma_;
};

}  // namespace ccr
