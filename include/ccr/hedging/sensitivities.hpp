#pragma once

#include <functional>
#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/instruments/credit_default_swap.hpp"
#include "ccr/market/credit_curve.hpp"
#include "ccr/market/yield_curve.hpp"
#include "ccr/xva/xva.hpp"

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

/// Credit and funding sensitivities of the total XVA (CVA - DVA + FCA - FBA + MVA + KVA) of an exposure profile.
struct XvaSensitivities {
  std::vector<double> counterpartyCs01;  ///< per counterparty CDS quote; hedge with cdsHedgeNotionals
  std::vector<double> ownCs01;           ///< per own CDS quote (DVA, FBA and the own-survival weights)
  double fundingSpread01 = 0.0;          ///< change for a parallel `bump` in the borrow, lend and margin funding spreads
};

/// Bump-and-revalue sensitivities of xva(): each quote of the counterparty and own CDS curves is bumped
/// and re-bootstrapped (central difference per bump), and the funding spreads are shifted together.
XvaSensitivities xvaSensitivities(const ExposureProfile& profile, const std::vector<double>& discountedInitialMargin,
                                  const YieldCurve& yc, const std::vector<CdsQuote>& counterpartyQuotes,
                                  double recoveryCounterparty, const std::vector<CdsQuote>& ownQuotes,
                                  double recoveryOwn, const XvaParameters& parameters, double bump = 1e-4);

}  // namespace ccr
