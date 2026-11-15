#include "ccr/xva/xva.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "ccr/core/math.hpp"

namespace ccr {

namespace {

void checkSizes(const std::vector<double>& times, const std::vector<double>& values, const char* who) {
  if (times.size() != values.size()) throw std::invalid_argument(std::string(who) + ": size mismatch");
}

/// int f(t) Q_c(t) Q_own(t) dt with the trapezoidal rule on f and survival at the mid-points.
double survivalWeightedIntegral(const std::vector<double>& times, const std::vector<double>& f,
                                const CreditCurve& counterparty, const CreditCurve& own) {
  double total = 0.0;
  for (std::size_t i = 1; i < times.size(); ++i) {
    const double mid = 0.5 * (times[i - 1] + times[i]);
    total += 0.5 * (f[i - 1] + f[i]) * (times[i] - times[i - 1]) * counterparty.survival(mid) * own.survival(mid);
  }
  return total;
}

}  // namespace

ExposureXva exposureXva(const std::vector<double>& times, const std::vector<double>& discountedEe,
                        const CreditCurve& counterparty, double recoveryCounterparty, const CreditCurve& own,
                        const XvaParameters& p) {
  checkSizes(times, discountedEe, "exposureXva");
  ExposureXva r;
  for (std::size_t i = 1; i < times.size(); ++i)
    r.cva += (1.0 - recoveryCounterparty) * 0.5 * (discountedEe[i - 1] + discountedEe[i]) *
             own.survival(0.5 * (times[i - 1] + times[i])) * counterparty.defaultProbability(times[i - 1], times[i]);
  const double weighted = survivalWeightedIntegral(times, discountedEe, counterparty, own);
  r.fca = p.fundingSpreadBorrow * weighted;
  r.kva = p.hurdleRate * p.capitalRatio * p.riskWeight * p.alpha * weighted;
  return r;
}

double fundingBenefit(const std::vector<double>& times, const std::vector<double>& discountedEne,
                      const CreditCurve& counterparty, const CreditCurve& own, double fundingSpreadLend) {
  checkSizes(times, discountedEne, "fundingBenefit");
  std::vector<double> magnitude(discountedEne.size());
  for (std::size_t i = 0; i < magnitude.size(); ++i) magnitude[i] = -discountedEne[i];
  return fundingSpreadLend * survivalWeightedIntegral(times, magnitude, counterparty, own);
}

double marginValuationAdjustment(const std::vector<double>& times, const std::vector<double>& discountedInitialMargin,
                                 const CreditCurve& counterparty, const CreditCurve& own, double marginFundingSpread) {
  checkSizes(times, discountedInitialMargin, "marginValuationAdjustment");
  return marginFundingSpread * survivalWeightedIntegral(times, discountedInitialMargin, counterparty, own);
}

std::vector<double> initialMarginProfile(const Matrix& values, const TimeGrid& grid, const Matrix& deflator,
                                         double quantileLevel, double marginPeriodOfRisk) {
  if (values.cols() != grid.size() || deflator.rows() != values.rows() || deflator.cols() != values.cols())
    throw std::invalid_argument("initialMarginProfile: values, deflator and grid do not match");
  const std::size_t paths = values.rows();
  std::vector<double> margin(grid.size(), 0.0);
  for (std::size_t k = 1; k < grid.size(); ++k) {
    std::vector<double> change(paths);
    double meanDeflator = 0.0;
    for (std::size_t p = 0; p < paths; ++p) {
      change[p] = std::fabs(values(p, k) - values(p, k - 1));
      meanDeflator += deflator(p, k);
    }
    const double scale = std::sqrt(marginPeriodOfRisk / (grid[k] - grid[k - 1]));
    margin[k] = quantile(std::move(change), quantileLevel) * scale * meanDeflator / static_cast<double>(paths);
  }
  return margin;
}

Xva xva(const ExposureProfile& profile, const std::vector<double>& discountedInitialMargin,
        const CreditCurve& counterparty, double recoveryCounterparty, const CreditCurve& own, double recoveryOwn,
        const XvaParameters& p) {
  const auto& t = profile.times;
  const ExposureXva e =
      exposureXva(t, profile.discountedExpectedExposure, counterparty, recoveryCounterparty, own, p);
  Xva r;
  r.cva = e.cva;
  r.fca = e.fca;
  r.kva = e.kva;
  for (std::size_t i = 1; i < t.size(); ++i)
    r.dva += (1.0 - recoveryOwn) * 0.5 * (-profile.discountedExpectedNegativeExposure[i - 1] -
                                          profile.discountedExpectedNegativeExposure[i]) *
             counterparty.survival(0.5 * (t[i - 1] + t[i])) * own.defaultProbability(t[i - 1], t[i]);
  r.fba = fundingBenefit(t, profile.discountedExpectedNegativeExposure, counterparty, own, p.fundingSpreadLend);
  if (!discountedInitialMargin.empty())
    r.mva = marginValuationAdjustment(t, discountedInitialMargin, counterparty, own, p.marginFundingSpread);
  return r;
}

std::vector<ExposureXva> allocateExposureXva(const std::vector<double>& times,
                                             const std::vector<std::vector<double>>& discountedContributions,
                                             const CreditCurve& counterparty, double recoveryCounterparty,
                                             const CreditCurve& own, const XvaParameters& p) {
  std::vector<ExposureXva> result;
  result.reserve(discountedContributions.size());
  for (const auto& c : discountedContributions)
    result.push_back(exposureXva(times, c, counterparty, recoveryCounterparty, own, p));
  return result;
}

}  // namespace ccr
