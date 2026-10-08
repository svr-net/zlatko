#include "ccr/exposure/allocation.hpp"

#include <algorithm>
#include <stdexcept>

namespace ccr {

std::vector<std::vector<double>> marginalExposureContributions(const std::vector<Matrix>& tradeValues,
                                                               const Matrix& nettedValue, const Matrix* deflator) {
  const std::size_t nP = nettedValue.rows();
  const std::size_t nT = nettedValue.cols();
  for (const Matrix& m : tradeValues)
    if (m.rows() != nP || m.cols() != nT) throw std::invalid_argument("marginalExposureContributions: size mismatch");
  if (deflator && (deflator->rows() != nP || deflator->cols() != nT))
    throw std::invalid_argument("marginalExposureContributions: deflator size mismatch");

  std::vector<std::vector<double>> contributions(tradeValues.size(), std::vector<double>(nT, 0.0));
  for (std::size_t j = 0; j < nT; ++j) {
    for (std::size_t p = 0; p < nP; ++p) {
      if (nettedValue(p, j) <= 0.0) continue;
      const double w = deflator ? (*deflator)(p, j) : 1.0;
      for (std::size_t i = 0; i < tradeValues.size(); ++i) contributions[i][j] += w * tradeValues[i](p, j);
    }
    for (auto& c : contributions) c[j] /= static_cast<double>(nP);
  }
  return contributions;
}

std::vector<double> incrementalExposure(const Matrix& nettedValue, const Matrix& newTrade, const Matrix* deflator) {
  const std::size_t nP = nettedValue.rows();
  const std::size_t nT = nettedValue.cols();
  if (newTrade.rows() != nP || newTrade.cols() != nT)
    throw std::invalid_argument("incrementalExposure: size mismatch");
  std::vector<double> delta(nT, 0.0);
  for (std::size_t j = 0; j < nT; ++j) {
    double sum = 0.0;
    for (std::size_t p = 0; p < nP; ++p) {
      const double w = deflator ? (*deflator)(p, j) : 1.0;
      const double before = nettedValue(p, j);
      sum += w * (std::max(before + newTrade(p, j), 0.0) - std::max(before, 0.0));
    }
    delta[j] = sum / static_cast<double>(nP);
  }
  return delta;
}

}  // namespace ccr
