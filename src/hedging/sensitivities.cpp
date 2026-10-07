#include "ccr/hedging/sensitivities.hpp"

#include <stdexcept>

namespace ccr {

double centralDifference(const std::function<double(double)>& f, double h) {
  if (h <= 0.0) throw std::invalid_argument("centralDifference: bump must be positive");
  return (f(h) - f(-h)) / (2.0 * h);
}

namespace {

std::vector<CdsQuote> bumped(const std::vector<CdsQuote>& quotes, std::size_t i, double bump) {
  auto q = quotes;
  q[i].spread += bump;
  return q;
}

}  // namespace

std::vector<double> creditSpreadSensitivities(const std::function<double(const CreditCurve&)>& valueOfCurve,
                                              const YieldCurve& yc, const std::vector<CdsQuote>& quotes,
                                              double recovery, double bump) {
  std::vector<double> sensitivities(quotes.size());
  for (std::size_t i = 0; i < quotes.size(); ++i) {
    const double up = valueOfCurve(bootstrapCreditCurve(yc, bumped(quotes, i, bump), recovery));
    const double down = valueOfCurve(bootstrapCreditCurve(yc, bumped(quotes, i, -bump), recovery));
    sensitivities[i] = 0.5 * (up - down);
  }
  return sensitivities;
}

Matrix cdsHedgeJacobian(const YieldCurve& yc, const std::vector<CdsQuote>& quotes, double recovery, double bump) {
  const std::size_t n = quotes.size();
  Matrix jacobian(n, n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    const CreditDefaultSwap cds(quotes[k].maturity, quotes[k].spread, 1.0, recovery);
    auto value = [&](const CreditCurve& curve) { return cds.npv(yc, curve); };
    const auto row = creditSpreadSensitivities(value, yc, quotes, recovery, bump);
    for (std::size_t i = 0; i < n; ++i) jacobian(k, i) = row[i];
  }
  return jacobian;
}

std::vector<double> cdsHedgeNotionals(const std::vector<double>& cvaSensitivities, const Matrix& jacobian) {
  if (jacobian.rows() != cvaSensitivities.size() || jacobian.cols() != cvaSensitivities.size())
    throw std::invalid_argument("cdsHedgeNotionals: dimension mismatch");
  return solveLinearSystem(transpose(jacobian), cvaSensitivities);
}

}  // namespace ccr
