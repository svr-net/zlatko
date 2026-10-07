#include "ccr/instruments/credit_default_swap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ccr/core/math.hpp"

namespace ccr {

namespace {

std::vector<double> premiumSchedule(double maturity, int frequency) {
  std::vector<double> times;
  const double step = 1.0 / frequency;
  for (double t = maturity; t > 1e-9; t -= step) times.push_back(t);
  std::reverse(times.begin(), times.end());
  return times;
}

constexpr int kProtectionSubsteps = 12;

}  // namespace

CreditDefaultSwap::CreditDefaultSwap(double maturity, double spread, double notional, double recovery,
                                     int paymentsPerYear, bool protectionBuyer)
    : maturity_(maturity),
      spread_(spread),
      notional_(notional),
      recovery_(recovery),
      frequency_(paymentsPerYear),
      buyer_(protectionBuyer) {
  if (maturity_ <= 0.0 || frequency_ <= 0) throw std::invalid_argument("CreditDefaultSwap: invalid arguments");
  if (recovery_ < 0.0 || recovery_ >= 1.0) throw std::invalid_argument("CreditDefaultSwap: recovery in [0, 1)");
}

std::vector<double> CreditDefaultSwap::paymentTimes() const { return premiumSchedule(maturity_, frequency_); }

double CreditDefaultSwap::riskyAnnuity(const YieldCurve& yc, const CreditCurve& cc) const {
  double annuity = 0.0;
  double previous = 0.0;
  for (double t : paymentTimes()) {
    const double tau = t - previous;
    annuity += tau * yc.discount(t) * cc.survival(t);
    // Accrued premium paid on default, assumed on average half-way through the period.
    annuity += 0.5 * tau * yc.discount(0.5 * (previous + t)) * cc.defaultProbability(previous, t);
    previous = t;
  }
  return annuity;
}

double CreditDefaultSwap::protectionLeg(const YieldCurve& yc, const CreditCurve& cc) const {
  double leg = 0.0;
  double previous = 0.0;
  for (double t : paymentTimes()) {
    const double h = (t - previous) / kProtectionSubsteps;
    for (int s = 0; s < kProtectionSubsteps; ++s) {
      const double a = previous + s * h;
      const double b = a + h;
      leg += yc.discount(0.5 * (a + b)) * cc.defaultProbability(a, b);
    }
    previous = t;
  }
  return (1.0 - recovery_) * leg;
}

double CreditDefaultSwap::parSpread(const YieldCurve& yc, const CreditCurve& cc) const {
  return protectionLeg(yc, cc) / riskyAnnuity(yc, cc);
}

double CreditDefaultSwap::npv(const YieldCurve& yc, const CreditCurve& cc) const {
  const double value = protectionLeg(yc, cc) - spread_ * riskyAnnuity(yc, cc);
  return (buyer_ ? 1.0 : -1.0) * notional_ * value;
}

CreditCurve bootstrapCreditCurve(const YieldCurve& yc, std::vector<CdsQuote> quotes, double recovery,
                                 int paymentsPerYear) {
  if (quotes.empty()) throw std::invalid_argument("bootstrapCreditCurve: no quotes");
  std::sort(quotes.begin(), quotes.end(), [](const CdsQuote& a, const CdsQuote& b) { return a.maturity < b.maturity; });
  std::vector<double> pillars, hazards;
  for (const CdsQuote& q : quotes) {
    pillars.push_back(q.maturity);
    hazards.push_back(0.0);
    const CreditDefaultSwap cds(q.maturity, q.spread, 1.0, recovery, paymentsPerYear);
    auto objective = [&](double h) {
      hazards.back() = h;
      const CreditCurve curve(pillars, hazards);
      return cds.parSpread(yc, curve) - q.spread;
    };
    double hi = 1.0;
    while (objective(hi) < 0.0 && hi < 100.0) hi *= 2.0;
    hazards.back() = solveBrent(objective, 0.0, hi, 1e-14);
  }
  return CreditCurve(pillars, hazards);
}

double riskyAnnuity(const YieldCurve& yc, const CreditCurve& cc, double maturity, int paymentsPerYear) {
  double annuity = 0.0;
  double previous = 0.0;
  for (double t : premiumSchedule(maturity, paymentsPerYear)) {
    annuity += (t - previous) * yc.discount(t) * cc.survival(t);
    previous = t;
  }
  return annuity;
}

}  // namespace ccr
