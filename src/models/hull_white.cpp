#include "ccr/models/hull_white.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "ccr/core/math.hpp"

namespace ccr {

HullWhite1F::HullWhite1F(std::shared_ptr<const YieldCurve> curve, double meanReversion, double volatility)
    : curve_(std::move(curve)), a_(meanReversion), sigma_(volatility) {
  if (!curve_) throw std::invalid_argument("HullWhite1F: null curve");
  if (a_ <= 1e-8) throw std::invalid_argument("HullWhite1F: mean reversion must be positive");
  if (sigma_ < 0.0) throw std::invalid_argument("HullWhite1F: volatility must be non-negative");
}

double HullWhite1F::B(double t, double T) const { return (1.0 - std::exp(-a_ * (T - t))) / a_; }

double HullWhite1F::V(double t, double T) const {
  const double tau = T - t;
  const double e1 = std::exp(-a_ * tau);
  return sigma_ * sigma_ / (a_ * a_) * (tau + 2.0 / a_ * e1 - 0.5 / a_ * e1 * e1 - 1.5 / a_);
}

double HullWhite1F::zeroBond(double t, double T, double x) const {
  if (T <= t) return 1.0;
  const double ratio = curve_->discount(T) / curve_->discount(t);
  return ratio * std::exp(0.5 * (V(t, T) - V(0.0, T) + V(0.0, t)) - B(t, T) * x);
}

double HullWhite1F::phi(double t) const {
  const double g = 1.0 - std::exp(-a_ * t);
  return curve_->instantaneousForward(t) + sigma_ * sigma_ / (2.0 * a_ * a_) * g * g;
}

double HullWhite1F::shortRate(double t, double x) const { return x + phi(t); }

void HullWhite1F::evolve(double t0, double t1, double x0, double z1, double z2, double& x1,
                         double& integratedRate) const {
  const double dt = t1 - t0;
  if (dt <= 0.0) throw std::invalid_argument("HullWhite1F::evolve: t1 must exceed t0");
  const double e = std::exp(-a_ * dt);
  const double s2 = sigma_ * sigma_;

  const double varX = s2 * (1.0 - e * e) / (2.0 * a_);
  const double varI = s2 / (a_ * a_) * (dt - 2.0 * (1.0 - e) / a_ + (1.0 - e * e) / (2.0 * a_));
  const double cov = s2 / (2.0 * a_ * a_) * (1.0 - e) * (1.0 - e);

  const double sdX = std::sqrt(varX);
  x1 = x0 * e + sdX * z1;

  double c1 = 0.0, c2 = std::sqrt(std::max(varI, 0.0));
  if (sdX > 0.0) {
    c1 = cov / sdX;
    c2 = std::sqrt(std::max(varI - c1 * c1, 0.0));
  }
  const double integratedX = x0 * (1.0 - e) / a_ + c1 * z1 + c2 * z2;

  // Integral of the deterministic shift phi over [t0, t1].
  const double integralG2 = dt - 2.0 / a_ * (std::exp(-a_ * t0) - std::exp(-a_ * t1)) +
                            0.5 / a_ * (std::exp(-2.0 * a_ * t0) - std::exp(-2.0 * a_ * t1));
  const double integratedPhi =
      std::log(curve_->discount(t0) / curve_->discount(t1)) + s2 / (2.0 * a_ * a_) * integralG2;

  integratedRate = integratedX + integratedPhi;
}

double HullWhite1F::bondOptionStdDev(double expiry, double maturity) const {
  return sigma_ * std::sqrt((1.0 - std::exp(-2.0 * a_ * expiry)) / (2.0 * a_)) * B(expiry, maturity);
}

double HullWhite1F::zeroBondCall(double expiry, double maturity, double strike) const {
  const double pT = curve_->discount(expiry);
  const double pS = curve_->discount(maturity);
  const double sp = bondOptionStdDev(expiry, maturity);
  if (sp <= 0.0) return std::max(pS - strike * pT, 0.0);
  const double h = std::log(pS / (pT * strike)) / sp + 0.5 * sp;
  return pS * normalCdf(h) - strike * pT * normalCdf(h - sp);
}

double HullWhite1F::zeroBondPut(double expiry, double maturity, double strike) const {
  const double pT = curve_->discount(expiry);
  const double pS = curve_->discount(maturity);
  const double sp = bondOptionStdDev(expiry, maturity);
  if (sp <= 0.0) return std::max(strike * pT - pS, 0.0);
  const double h = std::log(pS / (pT * strike)) / sp + 0.5 * sp;
  return strike * pT * normalCdf(-h + sp) - pS * normalCdf(-h);
}

double HullWhite1F::europeanSwaption(double expiry, const std::vector<double>& paymentTimes,
                                     const std::vector<double>& accruals, double fixedRate, bool payer) const {
  if (paymentTimes.empty() || paymentTimes.size() != accruals.size())
    throw std::invalid_argument("HullWhite1F::europeanSwaption: invalid schedule");
  std::vector<double> coupons(paymentTimes.size());
  for (std::size_t i = 0; i < coupons.size(); ++i) coupons[i] = fixedRate * accruals[i];
  coupons.back() += 1.0;

  // x* such that the coupon bond is worth par at expiry.
  auto couponBond = [&](double x) {
    double sum = 0.0;
    for (std::size_t i = 0; i < coupons.size(); ++i) sum += coupons[i] * zeroBond(expiry, paymentTimes[i], x);
    return sum - 1.0;
  };
  double lo = -0.5, hi = 0.5;
  for (int i = 0; i < 60 && couponBond(lo) * couponBond(hi) > 0.0; ++i) {
    lo *= 2.0;
    hi *= 2.0;
  }
  const double xStar = solveBrent(couponBond, lo, hi, 1e-14);

  double price = 0.0;
  for (std::size_t i = 0; i < coupons.size(); ++i) {
    const double strike = zeroBond(expiry, paymentTimes[i], xStar);
    price += coupons[i] * (payer ? zeroBondPut(expiry, paymentTimes[i], strike)
                                 : zeroBondCall(expiry, paymentTimes[i], strike));
  }
  return price;
}

}  // namespace ccr
