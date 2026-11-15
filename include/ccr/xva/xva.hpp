#pragma once

#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/core/time_grid.hpp"
#include "ccr/exposure/exposure_profile.hpp"
#include "ccr/market/credit_curve.hpp"

namespace ccr {

/// Inputs of the funding, margin and capital adjustments that come on top of CVA/DVA.
struct XvaParameters {
  double fundingSpreadBorrow = 0.0100;  ///< s_b: spread over the risk-free rate we pay to fund positive exposure
  double fundingSpreadLend = 0.0;       ///< s_l: spread we earn on the funding released by negative exposure
  double marginFundingSpread = 0.0100;  ///< spread paid to fund the initial margin we post (MVA)
  double hurdleRate = 0.10;             ///< cost of capital (KVA)
  double capitalRatio = 0.08;           ///< regulatory capital as a fraction of risk-weighted exposure
  double riskWeight = 1.0;              ///< counterparty risk weight
  double alpha = 1.4;                   ///< exposure-at-default multiplier applied to the expected exposure
};

/// The adjustments that are linear in the discounted expected exposure EE* and can therefore be
/// allocated to trades with the marginal (Euler) contributions of `exposure/allocation.hpp`.
struct ExposureXva {
  double cva = 0.0;  ///< (1 - R_c) int EE* Q_own dPD_c, as in bilateralCva
  double fca = 0.0;  ///< funding cost adjustment s_b int EE*(t) Q_c(t) Q_own(t) dt
  double kva = 0.0;  ///< hurdle * ratio * RW * alpha int EE*(t) Q_c(t) Q_own(t) dt
  double total() const { return cva + fca + kva; }
};

/// Every adjustment, as positive costs and benefits; total() is the cost charged to the trade.
struct Xva {
  double cva = 0.0;  ///< loss from the counterparty defaulting first
  double dva = 0.0;  ///< benefit of defaulting first ourselves
  double fca = 0.0;  ///< funding cost of the positive exposure
  double fba = 0.0;  ///< funding benefit of the negative exposure
  double mva = 0.0;  ///< cost of funding the initial margin
  double kva = 0.0;  ///< cost of the regulatory capital
  double fva() const { return fca - fba; }
  double total() const { return cva - dva + fca - fba + mva + kva; }
};

/// CVA, FCA and KVA for a discounted EE profile on `times` (trapezoidal in time; survival
/// probabilities of both parties are evaluated at the interval mid-points).
ExposureXva exposureXva(const std::vector<double>& times, const std::vector<double>& discountedEe,
                        const CreditCurve& counterparty, double recoveryCounterparty, const CreditCurve& own,
                        const XvaParameters& parameters);

/// Funding benefit adjustment s_l int |ENE*(t)| Q_c(t) Q_own(t) dt from the discounted ENE (<= 0).
double fundingBenefit(const std::vector<double>& times, const std::vector<double>& discountedEne,
                      const CreditCurve& counterparty, const CreditCurve& own, double fundingSpreadLend);

/// Margin valuation adjustment s_m int IM*(t) Q_c(t) Q_own(t) dt from the discounted initial margin profile.
double marginValuationAdjustment(const std::vector<double>& times, const std::vector<double>& discountedInitialMargin,
                                 const CreditCurve& counterparty, const CreditCurve& own, double marginFundingSpread);

/// Simulated initial margin proxy: at each date the `quantile` of |V(t_k) - V(t_{k-1})| over the
/// paths, scaled by sqrt(mpor / dt) to the margin period of risk, discounted with the mean
/// deflator. One value per grid date (0 at t = 0), ready for marginValuationAdjustment.
std::vector<double> initialMarginProfile(const Matrix& values, const TimeGrid& grid, const Matrix& deflator,
                                         double quantile = 0.99, double marginPeriodOfRisk = 10.0 / 250.0);

/// All adjustments of an exposure profile. `discountedInitialMargin` may be empty (MVA = 0).
Xva xva(const ExposureProfile& profile, const std::vector<double>& discountedInitialMargin,
        const CreditCurve& counterparty, double recoveryCounterparty, const CreditCurve& own, double recoveryOwn,
        const XvaParameters& parameters);

/// Allocation of CVA, FCA and KVA to trades: `discountedContributions[i]` is the marginal discounted
/// EE of trade i (marginalExposureContributions with the deflator). The results add up exactly to
/// exposureXva of the netting-set EE*.
std::vector<ExposureXva> allocateExposureXva(const std::vector<double>& times,
                                             const std::vector<std::vector<double>>& discountedContributions,
                                             const CreditCurve& counterparty, double recoveryCounterparty,
                                             const CreditCurve& own, const XvaParameters& parameters);

}  // namespace ccr
