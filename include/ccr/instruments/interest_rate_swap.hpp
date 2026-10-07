#pragma once

#include <vector>

#include "ccr/instruments/trade.hpp"
#include "ccr/market/yield_curve.hpp"

namespace ccr {

/// Single-currency fixed-vs-floating interest rate swap on the domestic Hull-White curve.
///
/// The schedule T_0 < T_1 < ... < T_n defines the accrual periods of both legs: the
/// floating rate for (T_{i-1}, T_i] fixes at T_{i-1} and both coupons pay at T_i.
/// When a period has already fixed at a valuation date, the fixing is recovered from
/// the simulated curve at the last grid date on or before the fixing date (exact when
/// the fixing date is on the grid, which the exposure engine ensures via eventTimes()).
class InterestRateSwap : public PathwiseTrade {
 public:
  enum class Direction { PayFixed, ReceiveFixed };

  InterestRateSwap(std::string id, double notional, double fixedRate, std::vector<double> schedule,
                   Direction direction);

  /// Regular schedule from `start` with `tenor` years and `paymentsPerYear` periods per year.
  static InterestRateSwap vanilla(std::string id, double notional, double fixedRate, double start, double tenor,
                                  int paymentsPerYear, Direction direction);

  double maturity() const override { return schedule_.back(); }
  std::vector<double> eventTimes() const override { return schedule_; }
  void valueAtTime(const ScenarioSet& scenarios, std::size_t j, double* out) const override;

  /// Value at time 0 off the curve.
  double npv(const YieldCurve& curve) const;
  /// Fixed rate that makes the swap worth zero today.
  double parRate(const YieldCurve& curve) const;

  /// Swap made of the periods starting on or after `fromTime` (the underlying entered
  /// into on exercise of a Bermudan swaption at `fromTime`).
  InterestRateSwap tail(double fromTime) const;
  /// Copy with a different fixed rate.
  InterestRateSwap withFixedRate(double fixedRate) const;

  double notional() const { return notional_; }
  double fixedRate() const { return fixedRate_; }
  Direction direction() const { return direction_; }
  const std::vector<double>& schedule() const { return schedule_; }
  std::vector<double> paymentTimes() const;
  std::vector<double> accruals() const;

 private:
  double sign() const { return direction_ == Direction::PayFixed ? 1.0 : -1.0; }

  double notional_;
  double fixedRate_;
  std::vector<double> schedule_;
  Direction direction_;
};

}  // namespace ccr
