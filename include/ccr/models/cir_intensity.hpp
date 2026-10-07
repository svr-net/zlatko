#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "ccr/market/credit_curve.hpp"

namespace ccr {

/// Stochastic default intensity of CIR++ type (Brigo-Mercurio):
///
///   lambda(t) = y(t) + psi(t),   dy = kappa (theta - y) dt + xi sqrt(y) dW,
///
/// with the deterministic shift psi chosen so that E[exp(-int lambda)] reproduces
/// the market survival curve exactly. Correlating dW with the market risk factors
/// that drive the exposure is the standard way to model wrong-way risk.
///
/// y is discretised with a full-truncation Euler scheme on `substeps` sub-steps per
/// grid step; the sub-step increments are built by a Brownian bridge so that the
/// increment over the whole grid step is exactly the (correlated) driver.
class CirIntensity {
 public:
  CirIntensity(std::string name, std::shared_ptr<const CreditCurve> marketCurve, double kappa, double theta,
               double xi, double y0, std::size_t substeps = 8);

  const std::string& name() const { return name_; }
  const CreditCurve& marketCurve() const { return *market_; }
  double kappa() const { return kappa_; }
  double theta() const { return theta_; }
  double xi() const { return xi_; }
  double y0() const { return y0_; }
  std::size_t substeps() const { return substeps_; }
  /// Number of independent normals needed per grid step on top of the correlated driver.
  std::size_t extraNormals() const { return substeps_ - 1; }

  /// Analytic CIR survival E[exp(-int_0^t y ds)].
  double cirSurvival(double t) const;
  /// Integrated shift int_0^t psi(s) ds = ln(S_CIR(t) / S_market(t)).
  double integratedShift(double t) const;
  /// Pathwise survival exp(-int_0^t lambda) given the integrated CIR component.
  double survival(double t, double integratedY) const;

  /// Advances y over a step of length dt. `z` is the correlated driver of the whole step,
  /// `extra` points to extraNormals() independent normals. Accumulates int y into integratedY.
  void evolve(double dt, double z, const double* extra, double& y, double& integratedY) const;

 private:
  std::string name_;
  std::shared_ptr<const CreditCurve> market_;
  double kappa_, theta_, xi_, y0_;
  std::size_t substeps_;
};

}  // namespace ccr
