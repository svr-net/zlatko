#include "ccr/instruments/interest_rate_swap.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ccr {

namespace {
constexpr double kEps = 1e-10;
}

InterestRateSwap::InterestRateSwap(std::string id, double notional, double fixedRate, std::vector<double> schedule,
                                   Direction direction)
    : PathwiseTrade(std::move(id)),
      notional_(notional),
      fixedRate_(fixedRate),
      schedule_(std::move(schedule)),
      direction_(direction) {
  if (schedule_.size() < 2) throw std::invalid_argument("InterestRateSwap: schedule needs at least two dates");
  if (schedule_.front() < 0.0) throw std::invalid_argument("InterestRateSwap: schedule must start at t >= 0");
  for (std::size_t i = 1; i < schedule_.size(); ++i)
    if (schedule_[i] <= schedule_[i - 1])
      throw std::invalid_argument("InterestRateSwap: schedule must be strictly increasing");
}

InterestRateSwap InterestRateSwap::vanilla(std::string id, double notional, double fixedRate, double start,
                                           double tenor, int paymentsPerYear, Direction direction) {
  if (tenor <= 0.0 || paymentsPerYear <= 0) throw std::invalid_argument("InterestRateSwap::vanilla: bad arguments");
  const auto periods = static_cast<std::size_t>(std::lround(tenor * paymentsPerYear));
  std::vector<double> schedule(periods + 1);
  for (std::size_t i = 0; i <= periods; ++i) schedule[i] = start + static_cast<double>(i) / paymentsPerYear;
  schedule.back() = start + tenor;
  return InterestRateSwap(std::move(id), notional, fixedRate, std::move(schedule), direction);
}

std::vector<double> InterestRateSwap::paymentTimes() const {
  return std::vector<double>(schedule_.begin() + 1, schedule_.end());
}

std::vector<double> InterestRateSwap::accruals() const {
  std::vector<double> tau(schedule_.size() - 1);
  for (std::size_t i = 1; i < schedule_.size(); ++i) tau[i - 1] = schedule_[i] - schedule_[i - 1];
  return tau;
}

void InterestRateSwap::valueAtTime(const ScenarioSet& s, std::size_t j, double* out) const {
  const double t = s.grid[j];
  const HullWhite1F& model = *s.rateModel;
  for (std::size_t p = 0; p < s.numPaths; ++p) out[p] = 0.0;
  if (t >= schedule_.back() - kEps) return;

  for (std::size_t i = 1; i < schedule_.size(); ++i) {
    const double ts = schedule_[i - 1];
    const double te = schedule_[i];
    if (te <= t + kEps) continue;  // already paid
    const double tau = te - ts;
    if (ts >= t - kEps) {
      // Floating coupon not yet fixed: N (P(t, Ts) - P(t, Te)).
      for (std::size_t p = 0; p < s.numPaths; ++p) {
        const double x = s.rateState(p, j);
        const double pe = model.zeroBond(t, te, x);
        const double floating = model.zeroBond(t, ts, x) - pe;
        out[p] += sign() * notional_ * (floating - fixedRate_ * tau * pe);
      }
    } else {
      // Coupon fixed at Ts < t: recover the fixing from the curve simulated at Ts.
      const std::size_t k = s.grid.indexAtOrBefore(ts);
      const double tk = s.grid[k];
      for (std::size_t p = 0; p < s.numPaths; ++p) {
        const double xk = s.rateState(p, k);
        const double fixing = (model.zeroBond(tk, ts, xk) / model.zeroBond(tk, te, xk) - 1.0) / tau;
        const double pe = model.zeroBond(t, te, s.rateState(p, j));
        out[p] += sign() * notional_ * (fixing - fixedRate_) * tau * pe;
      }
    }
  }
}

double InterestRateSwap::npv(const YieldCurve& curve) const {
  double fixedAnnuity = 0.0;
  double floating = 0.0;
  for (std::size_t i = 1; i < schedule_.size(); ++i) {
    fixedAnnuity += (schedule_[i] - schedule_[i - 1]) * curve.discount(schedule_[i]);
    floating += curve.discount(schedule_[i - 1]) - curve.discount(schedule_[i]);
  }
  return sign() * notional_ * (floating - fixedRate_ * fixedAnnuity);
}

double InterestRateSwap::parRate(const YieldCurve& curve) const {
  double annuity = 0.0;
  for (std::size_t i = 1; i < schedule_.size(); ++i)
    annuity += (schedule_[i] - schedule_[i - 1]) * curve.discount(schedule_[i]);
  return (curve.discount(schedule_.front()) - curve.discount(schedule_.back())) / annuity;
}

InterestRateSwap InterestRateSwap::tail(double fromTime) const {
  std::vector<double> schedule;
  for (double t : schedule_)
    if (t >= fromTime - kEps) schedule.push_back(t);
  if (schedule.size() < 2) throw std::invalid_argument("InterestRateSwap::tail: no period starts after fromTime");
  return InterestRateSwap(id() + "/tail", notional_, fixedRate_, std::move(schedule), direction_);
}

InterestRateSwap InterestRateSwap::withFixedRate(double fixedRate) const {
  return InterestRateSwap(id(), notional_, fixedRate, schedule_, direction_);
}

}  // namespace ccr
