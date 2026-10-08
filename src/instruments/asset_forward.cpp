#include "ccr/instruments/asset_forward.hpp"

#include <stdexcept>
#include <utility>

namespace ccr {

AssetForward::AssetForward(std::string id, std::string asset, double notional, double strike, double maturity)
    : PathwiseTrade(std::move(id)), asset_(std::move(asset)), notional_(notional), strike_(strike), maturity_(maturity) {
  if (maturity_ <= 0.0) throw std::invalid_argument("AssetForward: maturity must be positive");
}

void AssetForward::valueAtTime(const ScenarioSet& s, std::size_t j, double* out) const {
  const double t = s.grid[j];
  if (t >= maturity_ - 1e-10) {
    for (std::size_t p = 0; p < s.numPaths; ++p) out[p] = 0.0;
    return;
  }
  const std::size_t a = s.assetIndex(asset_);
  const double carry = s.assetModels[a]->carryDiscount(t, maturity_);
  for (std::size_t p = 0; p < s.numPaths; ++p) {
    const double spot = s.assetValues[a](p, j);
    out[p] = notional_ * (spot * carry - strike_ * s.zeroBond(p, j, maturity_));
  }
}

}  // namespace ccr
