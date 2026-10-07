#include "ccr/exposure/collateral.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ccr {

Matrix simulateCollateral(const CollateralAgreement& csa, const Matrix& portfolioValue, const TimeGrid& grid) {
  if (portfolioValue.cols() != grid.size())
    throw std::invalid_argument("simulateCollateral: value matrix does not match the grid");
  if (csa.thresholdCounterparty < 0.0 || csa.thresholdOwn < 0.0 || csa.minimumTransferAmount < 0.0 ||
      csa.marginPeriodOfRisk < 0.0)
    throw std::invalid_argument("simulateCollateral: invalid collateral agreement");

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
  for (std::size_t p = 0; p < nP; ++p) {
    double held = required(portfolioValue(p, callIndex[0]));
    for (std::size_t j = 0; j < nT; ++j) {
      const double target = required(portfolioValue(p, callIndex[j]));
      if (std::fabs(target - held) >= csa.minimumTransferAmount) held = target;
      collateral(p, j) = held + csa.independentAmount;
    }
  }
  return collateral;
}

}  // namespace ccr
