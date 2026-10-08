#pragma once

#include <cstddef>
#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/exposure/exposure_profile.hpp"
#include "ccr/market/credit_curve.hpp"
#include "ccr/market/yield_curve.hpp"
#include "ccr/models/scenario_generator.hpp"

namespace ccr {

/// Unilateral credit valuation adjustment assuming independence between exposure and default:
///
///   CVA = (1 - R) int_0^T EE*(t) dPD(t)
///       ~ (1 - R) sum_i 1/2 (EE*(t_{i-1}) + EE*(t_i)) (Q(t_{i-1}) - Q(t_i)),
///
/// with EE* the discounted expected exposure on the grid `times`.
double unilateralCva(const std::vector<double>& times, const std::vector<double>& discountedEe,
                     const CreditCurve& counterparty, double recovery);
double unilateralCva(const ExposureProfile& profile, const CreditCurve& counterparty, double recovery);

/// Per-period contributions to the unilateral CVA (they sum to the CVA).
std::vector<double> cvaTermStructure(const std::vector<double>& times, const std::vector<double>& discountedEe,
                                     const CreditCurve& counterparty, double recovery);

struct BilateralCva {
  double cva = 0.0;  ///< loss from the counterparty defaulting first
  double dva = 0.0;  ///< gain from defaulting first ourselves
  double total() const { return cva - dva; }
};

/// Bilateral CVA with first-to-default, defaults independent of each other and of exposure:
///   CVA = (1 - R_c) int EE*(t)  Q_own(t)  dPD_c(t)
///   DVA = (1 - R_o) int |ENE*(t)| Q_cpty(t) dPD_o(t)
BilateralCva bilateralCva(const ExposureProfile& profile, const CreditCurve& counterparty, double recoveryCounterparty,
                          const CreditCurve& own, double recoveryOwn);

/// CVA charged as a running spread over the life of the trade: CVA / (notional * risky annuity).
double cvaRunningSpread(double cva, const YieldCurve& yc, const CreditCurve& counterparty, double maturity,
                        double notional, int paymentsPerYear = 4);

/// CVA computed path by path with a stochastic default intensity simulated jointly with
/// the market (ScenarioSet::survival), which captures wrong-way / right-way risk:
///
///   CVA = (1 - R) E[ sum_i D(0, t_i) V^+(t_i) (Q_path(t_{i-1}) - Q_path(t_i)) ]
///
/// (trapezoidal in the exposure). With zero correlation it agrees with unilateralCva.
/// `dates` restricts the sum to these grid indices (e.g. ExposureResult::reportingIndices);
/// empty means every grid date.
double pathwiseCva(const ScenarioSet& scenarios, const Matrix& exposureValue, std::size_t creditIndex,
                   double recovery, const std::vector<std::size_t>& dates = {});

}  // namespace ccr
