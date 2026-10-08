#include "ccr/cva/cva.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ccr/instruments/credit_default_swap.hpp"

namespace ccr {

std::vector<double> cvaTermStructure(const std::vector<double>& times, const std::vector<double>& discountedEe,
                                     const CreditCurve& counterparty, double recovery) {
  if (times.size() != discountedEe.size()) throw std::invalid_argument("cvaTermStructure: size mismatch");
  std::vector<double> terms(times.size(), 0.0);
  for (std::size_t i = 1; i < times.size(); ++i) {
    const double ee = 0.5 * (discountedEe[i - 1] + discountedEe[i]);
    terms[i] = (1.0 - recovery) * ee * counterparty.defaultProbability(times[i - 1], times[i]);
  }
  return terms;
}

double unilateralCva(const std::vector<double>& times, const std::vector<double>& discountedEe,
                     const CreditCurve& counterparty, double recovery) {
  const auto terms = cvaTermStructure(times, discountedEe, counterparty, recovery);
  double cva = 0.0;
  for (double t : terms) cva += t;
  return cva;
}

double unilateralCva(const ExposureProfile& profile, const CreditCurve& counterparty, double recovery) {
  return unilateralCva(profile.times, profile.discountedExpectedExposure, counterparty, recovery);
}

BilateralCva bilateralCva(const ExposureProfile& profile, const CreditCurve& counterparty, double recoveryCounterparty,
                          const CreditCurve& own, double recoveryOwn) {
  BilateralCva result;
  const auto& t = profile.times;
  const auto& epe = profile.discountedExpectedExposure;
  const auto& ene = profile.discountedExpectedNegativeExposure;
  for (std::size_t i = 1; i < t.size(); ++i) {
    const double mid = 0.5 * (t[i - 1] + t[i]);
    result.cva += (1.0 - recoveryCounterparty) * 0.5 * (epe[i - 1] + epe[i]) * own.survival(mid) *
                  counterparty.defaultProbability(t[i - 1], t[i]);
    result.dva += (1.0 - recoveryOwn) * 0.5 * (-ene[i - 1] - ene[i]) * counterparty.survival(mid) *
                  own.defaultProbability(t[i - 1], t[i]);
  }
  return result;
}

double cvaRunningSpread(double cva, const YieldCurve& yc, const CreditCurve& counterparty, double maturity,
                        double notional, int paymentsPerYear) {
  const double annuity = riskyAnnuity(yc, counterparty, maturity, paymentsPerYear);
  if (annuity <= 0.0 || notional == 0.0) throw std::invalid_argument("cvaRunningSpread: degenerate annuity");
  return cva / (std::fabs(notional) * annuity);
}

double pathwiseCva(const ScenarioSet& s, const Matrix& exposureValue, std::size_t creditIndex, double recovery,
                   const std::vector<std::size_t>& dates) {
  if (creditIndex >= s.survival.size()) throw std::out_of_range("pathwiseCva: credit index out of range");
  if (exposureValue.rows() != s.numPaths || exposureValue.cols() != s.numTimes())
    throw std::invalid_argument("pathwiseCva: exposure does not match the scenarios");
  std::vector<std::size_t> idx = dates;
  if (idx.empty())
    for (std::size_t j = 0; j < s.numTimes(); ++j) idx.push_back(j);
  const Matrix& q = s.survival[creditIndex];
  double total = 0.0;
  for (std::size_t p = 0; p < s.numPaths; ++p) {
    double previous = s.deflator(p, idx[0]) * std::max(exposureValue(p, idx[0]), 0.0);
    for (std::size_t k = 1; k < idx.size(); ++k) {
      const std::size_t i0 = idx[k - 1], i1 = idx[k];
      const double current = s.deflator(p, i1) * std::max(exposureValue(p, i1), 0.0);
      total += 0.5 * (previous + current) * (q(p, i0) - q(p, i1));
      previous = current;
    }
  }
  return (1.0 - recovery) * total / static_cast<double>(s.numPaths);
}

}  // namespace ccr
