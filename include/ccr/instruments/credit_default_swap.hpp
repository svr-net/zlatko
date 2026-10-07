#pragma once

#include <vector>

#include "ccr/market/credit_curve.hpp"
#include "ccr/market/yield_curve.hpp"

namespace ccr {

/// Credit default swap valued off deterministic yield and credit curves.
/// It is the instrument used to calibrate counterparty credit curves and to
/// hedge the credit-spread sensitivity of CVA.
class CreditDefaultSwap {
 public:
  CreditDefaultSwap(double maturity, double spread, double notional = 1.0, double recovery = 0.4,
                    int paymentsPerYear = 4, bool protectionBuyer = true);

  /// Risky PV01 per unit notional and unit spread, including accrual on default.
  double riskyAnnuity(const YieldCurve& yc, const CreditCurve& cc) const;
  /// Present value of the protection leg per unit notional: (1 - R) int P(0, t) dQ(tau <= t).
  double protectionLeg(const YieldCurve& yc, const CreditCurve& cc) const;
  /// Running spread that sets the value to zero.
  double parSpread(const YieldCurve& yc, const CreditCurve& cc) const;
  /// Value to the protection buyer (sign flipped for the seller), scaled by the notional.
  double npv(const YieldCurve& yc, const CreditCurve& cc) const;

  double maturity() const { return maturity_; }
  double spread() const { return spread_; }
  double notional() const { return notional_; }
  double recovery() const { return recovery_; }
  std::vector<double> paymentTimes() const;

 private:
  double maturity_;
  double spread_;
  double notional_;
  double recovery_;
  int frequency_;
  bool buyer_;
};

struct CdsQuote {
  double maturity;
  double spread;  ///< par spread (decimal, e.g. 0.01 = 100bp)
};

/// Bootstraps a piecewise-constant hazard curve reproducing the CDS par spreads.
CreditCurve bootstrapCreditCurve(const YieldCurve& yc, std::vector<CdsQuote> quotes, double recovery,
                                 int paymentsPerYear = 4);

/// Risky annuity sum tau_i P(0, t_i) Q(tau > t_i) of a running premium paid to `maturity`
/// (used to convert an upfront CVA into a running spread).
double riskyAnnuity(const YieldCurve& yc, const CreditCurve& cc, double maturity, int paymentsPerYear = 4);

}  // namespace ccr
