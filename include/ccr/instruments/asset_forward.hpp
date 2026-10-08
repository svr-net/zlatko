#pragma once

#include <string>
#include <vector>

#include "ccr/instruments/trade.hpp"

namespace ccr {

/// Forward purchase of `notional` units of a simulated asset (FX rate or equity)
/// at `strike` (domestic currency per unit) for settlement at `maturity`:
///
///   V(t) = N (S(t) Q(t, T) - K P(t, T)).
///
/// A negative notional is a forward sale.
class AssetForward : public PathwiseTrade {
 public:
  AssetForward(std::string id, std::string asset, double notional, double strike, double maturity);

  double maturity() const override { return maturity_; }
  std::vector<double> eventTimes() const override { return {maturity_}; }
  void valueAtTime(const ScenarioSet& scenarios, std::size_t j, double* out) const override;

  const std::string& asset() const { return asset_; }
  double notional() const { return notional_; }
  double strike() const { return strike_; }

 private:
  std::string asset_;
  double notional_;
  double strike_;
  double maturity_;
};

}  // namespace ccr
