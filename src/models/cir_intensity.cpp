#include "ccr/models/cir_intensity.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ccr {

CirIntensity::CirIntensity(std::string name, std::shared_ptr<const CreditCurve> marketCurve, double kappa,
                           double theta, double xi, double y0, std::size_t substeps)
    : name_(std::move(name)),
      market_(std::move(marketCurve)),
      kappa_(kappa),
      theta_(theta),
      xi_(xi),
      y0_(y0),
      substeps_(substeps) {
  if (!market_) throw std::invalid_argument("CirIntensity: null market curve");
  if (kappa_ <= 0.0 || theta_ < 0.0 || xi_ < 0.0 || y0_ < 0.0)
    throw std::invalid_argument("CirIntensity: invalid parameters");
  if (substeps_ == 0) throw std::invalid_argument("CirIntensity: substeps must be positive");
}

double CirIntensity::cirSurvival(double t) const {
  if (t <= 0.0) return 1.0;
  if (xi_ == 0.0) {
    // Deterministic mean-reverting intensity.
    const double integral = theta_ * t + (y0_ - theta_) * (1.0 - std::exp(-kappa_ * t)) / kappa_;
    return std::exp(-integral);
  }
  const double h = std::sqrt(kappa_ * kappa_ + 2.0 * xi_ * xi_);
  const double eh = std::exp(h * t) - 1.0;
  const double denom = 2.0 * h + (kappa_ + h) * eh;
  const double logA =
      (2.0 * kappa_ * theta_ / (xi_ * xi_)) * std::log(2.0 * h * std::exp(0.5 * (kappa_ + h) * t) / denom);
  const double b = 2.0 * eh / denom;
  return std::exp(logA - b * y0_);
}

double CirIntensity::integratedShift(double t) const {
  return std::log(cirSurvival(t) / market_->survival(t));
}

double CirIntensity::survival(double t, double integratedY) const {
  return std::exp(-integratedY - integratedShift(t));
}

void CirIntensity::evolve(double dt, double z, const double* extra, double& y, double& integratedY) const {
  const double h = dt / static_cast<double>(substeps_);
  double remaining = std::sqrt(dt) * z;  // Brownian increment still to be distributed
  double remainingTime = dt;
  for (std::size_t s = 0; s < substeps_; ++s) {
    double dW = remaining;
    if (s + 1 < substeps_) {
      // Brownian bridge: W(h) | W(T_r) = R  ~  N(h/T_r * R, h (T_r - h) / T_r)
      dW = h / remainingTime * remaining + std::sqrt(h * (remainingTime - h) / remainingTime) * extra[s];
    }
    remaining -= dW;
    remainingTime -= h;
    const double yPlus = std::max(y, 0.0);
    const double yNext = y + kappa_ * (theta_ - yPlus) * h + xi_ * std::sqrt(yPlus) * dW;
    integratedY += 0.5 * (yPlus + std::max(yNext, 0.0)) * h;
    y = yNext;
  }
}

}  // namespace ccr
