#include "ccr/exposure/collateral.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ccr {

Matrix simulateCollateral(const CollateralAgreement& csa, const Matrix& portfolioValue, const TimeGrid& grid,
                          const Matrix* unitPrice) {
  if (portfolioValue.cols() != grid.size())
    throw std::invalid_argument("simulateCollateral: value matrix does not match the grid");
  if (csa.thresholdCounterparty < 0.0 || csa.thresholdOwn < 0.0 || csa.minimumTransferAmount < 0.0 ||
      csa.marginPeriodOfRisk < 0.0 || csa.haircut < 0.0 || csa.haircut >= 1.0)
    throw std::invalid_argument("simulateCollateral: invalid collateral agreement");
  const bool securities = csa.type != CollateralAgreement::Type::Cash;
  if (securities && (!unitPrice || unitPrice->rows() != portfolioValue.rows() || unitPrice->cols() != grid.size()))
    throw std::invalid_argument("simulateCollateral: securities collateral needs the unit price on every path and date");

  const std::size_t nP = portfolioValue.rows();
  const std::size_t nT = grid.size();
  std::vector<std::size_t> callIndex(nT);
  for (std::size_t j = 0; j < nT; ++j) callIndex[j] = grid.indexAtOrBefore(std::max(grid[j] - csa.marginPeriodOfRisk, 0.0));

  auto required = [&csa](double value) {
    double c = 0.0;
    if (value > csa.thresholdCounterparty) c += value - csa.thresholdCounterparty;
    if (std::isfinite(csa.thresholdOwn) && -value > csa.thresholdOwn) c -= -value - csa.thresholdOwn;
    return c;
  };

  Matrix collateral(nP, nT, 0.0);
  const double keep = securities ? 1.0 - csa.haircut : 1.0;
  auto price = [&](std::size_t p, std::size_t j) { return securities ? (*unitPrice)(p, j) : 1.0; };
  for (std::size_t p = 0; p < nP; ++p) {
    // Units held: the collateral value after haircut at the call date is what the CSA requires.
    double units = required(portfolioValue(p, callIndex[0])) / (keep * price(p, callIndex[0]));
    for (std::size_t j = 0; j < nT; ++j) {
      const std::size_t c = callIndex[j];
      const double target = required(portfolioValue(p, c));
      if (std::fabs(target - units * keep * price(p, c)) >= csa.minimumTransferAmount) units = target / (keep * price(p, c));
      collateral(p, j) = units * price(p, j) + csa.independentAmount;
    }
  }
  return collateral;
}

}  // namespace ccr
