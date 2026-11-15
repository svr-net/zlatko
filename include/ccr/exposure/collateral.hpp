#pragma once

#include <limits>

#include "ccr/core/matrix.hpp"
#include "ccr/core/time_grid.hpp"

namespace ccr {

/// Terms of a collateral agreement (CSA) attached to a netting set.
///
/// Values are from our point of view: a positive netting-set value is an
/// exposure to the counterparty, who then posts collateral to us.
struct CollateralAgreement {
  /// Uncollateralised exposure tolerated before the counterparty must post.
  double thresholdCounterparty = 0.0;
  /// Threshold on our side; infinity means a one-way CSA where we never post.
  double thresholdOwn = std::numeric_limits<double>::infinity();
  /// Collateral moves only when the required change is at least this amount.
  double minimumTransferAmount = 0.0;
  /// Independent amount posted by the counterparty (negative: posted by us).
  double independentAmount = 0.0;
  /// Margin period of risk: time between the last successful margin call and the
  /// close-out of the portfolio after a default (e.g. 10 business days).
  double marginPeriodOfRisk = 10.0 / 250.0;

  /// What the variation margin is posted in. Securities are delivered in units at the margin
  /// call (amount / ((1 - haircut) * price at the call)) and are worth their price at close-out:
  /// the collateral value moves over the margin period of risk with the security, which also
  /// captures the wrong-way risk between the collateral and the exposure.
  enum class Type { Cash, Bond, Asset };
  Type type = Type::Cash;
  double haircut = 0.0;        ///< valuation haircut of securities, in [0, 1)
  double bondMaturity = 0.0;   ///< Bond: maturity of the government zero-coupon bond (domestic Hull-White)
  std::size_t assetIndex = 0;  ///< Asset: index of the FX / equity factor in the scenario generator
};

/// Simulates the variation margin held on every path and grid date.
///
/// The collateral held at t is the one called on the portfolio value at t - MPR
/// (the last margin call the defaulting counterparty met), subject to thresholds and
/// the minimum transfer amount, plus the independent amount. The portfolio value at
/// t - MPR is read at the last grid date on or before it, so the grid should contain
/// the lagged dates (TimeGrid::withLaggedTimes). Positive = held by us. With cash collateral the
/// value held is the amount called; with securities it is the units called revalued at t.
/// `unitPrice` (paths x dates) is the price of one unit of the collateral security on every
/// path and date (Bond / Asset collateral); the result is the value held, units x price, plus
/// the independent amount (posted in cash).
Matrix simulateCollateral(const CollateralAgreement& csa, const Matrix& portfolioValue, const TimeGrid& grid,
                          const Matrix* unitPrice = nullptr);

}  // namespace ccr
