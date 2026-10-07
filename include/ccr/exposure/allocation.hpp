#pragma once

#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/core/time_grid.hpp"

namespace ccr {

/// Marginal (Euler) allocation of the expected exposure of an uncollateralised
/// netting set to its trades:
///
///   c_i(t) = E[ w(t) V_i(t) 1{V(t) > 0} ],   sum_i c_i(t) = E[ w(t) max(V(t), 0) ],
///
/// with weights w = 1 (EE) or w = D(0, t) (discounted EE, pass the deflator). Because
/// CVA is linear in discounted EE, applying the CVA formula to each c_i allocates CVA
/// additively across the trades, accounting for netting benefits.
/// Result: one profile per trade.
std::vector<std::vector<double>> marginalExposureContributions(const std::vector<Matrix>& tradeValues,
                                                               const Matrix& nettedValue,
                                                               const Matrix* deflator = nullptr);

/// Incremental exposure of adding `newTrade` to a netting set: EE(V + V_new) - EE(V)
/// (discounted if a deflator is given). Not additive across trades, but the right
/// measure for pricing a new trade against an existing portfolio.
std::vector<double> incrementalExposure(const Matrix& nettedValue, const Matrix& newTrade,
                                        const Matrix* deflator = nullptr);

}  // namespace ccr
