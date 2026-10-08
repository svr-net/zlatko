#pragma once

#include <string>
#include <vector>

#include "ccr/core/math.hpp"
#include "ccr/instruments/trade.hpp"

namespace ccr {

/// European option on a simulated asset (FX or equity), cash-settled at expiry.
///
/// Revalued on each scenario with the Black formula on the forward
/// S(t) Q(t, T) / P(t, T), using the simulated domestic discount factor P(t, T)
/// and the asset's volatility. The contribution of interest-rate volatility to the
/// forward's volatility is neglected (exact when the rate volatility is zero).
/// A negative notional is a short position.
class EuropeanOption : public PathwiseTrade {
 public:
  EuropeanOption(std::string id, std::string asset, OptionType type, double notional, double strike, double expiry);

  double maturity() const override { return expiry_; }
  std::vector<double> eventTimes() const override { return {expiry_}; }
  void valueAtTime(const ScenarioSet& scenarios, std::size_t j, double* out) const override;

  const std::string& asset() const { return asset_; }
  OptionType type() const { return type_; }
  double notional() const { return notional_; }
  double strike() const { return strike_; }
  double expiry() const { return expiry_; }

 private:
  std::string asset_;
  OptionType type_;
  double notional_;
  double strike_;
  double expiry_;
};

}  // namespace ccr
