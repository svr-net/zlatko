#include "ccr/instruments/european_option.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ccr {

EuropeanOption::EuropeanOption(std::string id, std::string asset, OptionType type, double notional, double strike,
                               double expiry)
    : PathwiseTrade(std::move(id)),
      asset_(std::move(asset)),
      type_(type),
      notional_(notional),
      strike_(strike),
      expiry_(expiry) {
  if (expiry_ <= 0.0 || strike_ <= 0.0) throw std::invalid_argument("EuropeanOption: invalid expiry or strike");
}

void EuropeanOption::valueAtTime(const ScenarioSet& s, std::size_t j, double* out) const {
  const double t = s.grid[j];
  if (t >= expiry_ - 1e-10) {
    for (std::size_t p = 0; p < s.numPaths; ++p) out[p] = 0.0;
    return;
  }
  const std::size_t a = s.assetIndex(asset_);
  const LognormalAsset& model = *s.assetModels[a];
  const double carry = model.carryDiscount(t, expiry_);
  const double stdDev = model.volatility() * std::sqrt(expiry_ - t);
  for (std::size_t p = 0; p < s.numPaths; ++p) {
    const double df = s.zeroBond(p, j, expiry_);
    const double forward = s.assetValues[a](p, j) * carry / df;
    out[p] = notional_ * blackFormula(type_, forward, strike_, stdDev, df);
  }
}

}  // namespace ccr
