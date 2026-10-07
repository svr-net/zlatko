#pragma once

#include <cstddef>
#include <vector>

#include "ccr/instruments/interest_rate_swap.hpp"
#include "ccr/instruments/trade.hpp"

namespace ccr {

enum class Settlement {
  Physical,  ///< exercise enters the remaining swap: exposure continues after exercise
  Cash       ///< exercise pays the swap value in cash: exposure stops at exercise
};

/// Bermudan swaption valued along the scenarios by American Monte Carlo.
///
/// A backward Longstaff-Schwartz induction over the exercise dates estimates the
/// exercise policy by regressing the deflated realised value of the option on the
/// Hull-White state. The same regression machinery then gives the mark-to-market at
/// every grid date as the conditional expectation of the deflated future payoff,
/// so the exposure of an exercisable product is obtained on the very scenarios used
/// for all other trades. Paths on which the option was exercised carry the value
/// of the underlying swap (physical settlement) or zero (cash settlement).
///
/// Exercise dates must belong to the underlying swap schedule; exercising at e
/// enters the swap periods that start on or after e.
class BermudanSwaption : public Trade {
 public:
  /// `position` +1 for a long option, -1 for a short one (scaled by any other value).
  BermudanSwaption(std::string id, InterestRateSwap underlying, std::vector<double> exerciseTimes,
                   Settlement settlement = Settlement::Physical, double position = 1.0,
                   std::size_t regressionDegree = 3);

  double maturity() const override;
  std::vector<double> eventTimes() const override;
  Matrix valueCube(const ScenarioSet& scenarios) const override;

  struct Valuation {
    Matrix value;                            ///< numPaths x numTimes mark-to-market
    std::vector<std::size_t> exerciseIndex;  ///< per path, index into exerciseTimes(), or npos if never exercised
    double price = 0.0;                      ///< Monte Carlo price at t = 0
  };
  Valuation valueWithExercise(const ScenarioSet& scenarios) const;

  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  const InterestRateSwap& underlying() const { return underlying_; }
  const std::vector<double>& exerciseTimes() const { return exerciseTimes_; }
  Settlement settlement() const { return settlement_; }

 private:
  InterestRateSwap underlying_;
  std::vector<double> exerciseTimes_;
  Settlement settlement_;
  double position_;
  std::size_t degree_;
};

}  // namespace ccr
