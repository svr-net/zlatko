#pragma once

#include <functional>
#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/instruments/credit_default_swap.hpp"
#include "ccr/market/credit_curve.hpp"
#include "ccr/market/yield_curve.hpp"

namespace ccr {

/// Central finite difference (f(+h) - f(-h)) / (2h) of a function of a bump size.
/// For Monte Carlo prices f must regenerate scenarios with the same seed (common
/// random numbers) so that the difference is not swamped by simulation noise.
double centralDifference(const std::function<double(double)>& f, double h);

/// Bucketed credit-spread sensitivities ("CS01") of a credit-dependent value such as CVA:
/// for each quote, the change in value when that par spread moves by `bump` (default 1bp),
/// re-bootstrapping the hazard curve (central difference, scaled to one bump).
std::vector<double> creditSpreadSensitivities(const std::function<double(const CreditCurve&)>& valueOfCurve,
                                              const YieldCurve& yc, const std::vector<CdsQuote>& quotes,
                                              double recovery, double bump = 1e-4);

/// Sensitivity matrix of the hedge instruments: J(k, i) = change in value of a unit-notional
/// par CDS (protection bought) with maturity quotes[k].maturity when quote i moves by `bump`.
Matrix cdsHedgeJacobian(const YieldCurve& yc, const std::vector<CdsQuote>& quotes, double recovery,
                        double bump = 1e-4);

/// CDS notionals n_k such that the hedge offsets the bucketed sensitivities of the CVA:
/// sum_k n_k J(k, i) = cvaSensitivities[i] for every bucket i. Positive notional = buy protection
/// (CVA is a short credit position on the counterparty, hedged by buying protection).
std::vector<double> cdsHedgeNotionals(const std::vector<double>& cvaSensitivities, const Matrix& jacobian);

}  // namespace ccr
