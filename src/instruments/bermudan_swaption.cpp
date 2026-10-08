#include "ccr/instruments/bermudan_swaption.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "ccr/core/regression.hpp"

namespace ccr {

namespace {
constexpr double kEps = 1e-10;
}

BermudanSwaption::BermudanSwaption(std::string id, InterestRateSwap underlying, std::vector<double> exerciseTimes,
                                   Settlement settlement, double position, std::size_t regressionDegree)
    : Trade(std::move(id)),
      underlying_(std::move(underlying)),
      exerciseTimes_(std::move(exerciseTimes)),
      settlement_(settlement),
      position_(position),
      degree_(regressionDegree) {
  if (exerciseTimes_.empty()) throw std::invalid_argument("BermudanSwaption: no exercise dates");
  std::sort(exerciseTimes_.begin(), exerciseTimes_.end());
  const auto& schedule = underlying_.schedule();
  for (double e : exerciseTimes_) {
    const bool onSchedule = std::any_of(schedule.begin(), schedule.end() - 1,
                                        [e](double t) { return std::fabs(t - e) < 1e-8; });
    if (!onSchedule)
      throw std::invalid_argument("BermudanSwaption: exercise dates must be start dates of the underlying periods");
  }
}

double BermudanSwaption::maturity() const {
  return settlement_ == Settlement::Physical ? underlying_.maturity() : exerciseTimes_.back();
}

std::vector<double> BermudanSwaption::eventTimes() const {
  std::vector<double> times = underlying_.eventTimes();
  times.insert(times.end(), exerciseTimes_.begin(), exerciseTimes_.end());
  return times;
}

Matrix BermudanSwaption::valueCube(const ScenarioSet& scenarios) const {
  return valueWithExercise(scenarios).value;
}

BermudanSwaption::Valuation BermudanSwaption::valueWithExercise(const ScenarioSet& s) const {
  const std::size_t nP = s.numPaths;
  const std::size_t nT = s.numTimes();
  const std::size_t nE = exerciseTimes_.size();

  std::vector<std::size_t> exerciseGridIndex(nE);
  std::vector<InterestRateSwap> tails;
  tails.reserve(nE);
  for (std::size_t k = 0; k < nE; ++k) {
    const auto idx = s.grid.find(exerciseTimes_[k], 1e-8);
    if (!idx) throw std::invalid_argument("BermudanSwaption: exercise date not on the simulation grid");
    exerciseGridIndex[k] = *idx;
    tails.push_back(underlying_.tail(exerciseTimes_[k]));
  }

  // Exercise values: the swap entered into at each exercise date.
  std::vector<std::vector<double>> exerciseValue(nE, std::vector<double>(nP));
  for (std::size_t k = 0; k < nE; ++k) tails[k].valueAtTime(s, exerciseGridIndex[k], exerciseValue[k].data());

  // Backward Longstaff-Schwartz induction. strategy[k] holds, per path, the deflated value
  // D(0, tau) * swap(tau) realised by the estimated policy given no exercise before e_k.
  std::vector<std::vector<double>> strategy(nE + 1, std::vector<double>(nP, 0.0));
  std::vector<std::vector<bool>> exercise(nE, std::vector<bool>(nP, false));
  std::vector<double> realised(nP, 0.0);
  std::vector<double> target(nP);
  std::vector<bool> inTheMoney(nP);
  PolynomialRegression regression(degree_);

  for (std::size_t k = nE; k-- > 0;) {
    const std::size_t j = exerciseGridIndex[k];
    const std::vector<double> x = s.rateState.column(j);
    for (std::size_t p = 0; p < nP; ++p) {
      target[p] = realised[p] / s.deflator(p, j);
      inTheMoney[p] = exerciseValue[k][p] > 0.0;
    }
    regression.fit(x, target, inTheMoney);
    for (std::size_t p = 0; p < nP; ++p) {
      const double e = exerciseValue[k][p];
      if (e > 0.0 && e > regression.predict(x[p])) {
        exercise[k][p] = true;
        realised[p] = s.deflator(p, j) * e;
      }
    }
    strategy[k] = realised;
  }

  Valuation result;
  result.exerciseIndex.assign(nP, npos);
  for (std::size_t p = 0; p < nP; ++p)
    for (std::size_t k = 0; k < nE; ++k)
      if (exercise[k][p]) {
        result.exerciseIndex[p] = k;
        break;
      }

  // Mark-to-market on every grid date.
  result.value = Matrix(nP, nT, 0.0);
  std::vector<double> continuation(nP, 0.0);
  std::vector<std::vector<double>> tailValues(nE);
  for (std::size_t j = 0; j < nT; ++j) {
    const double t = s.grid[j];
    std::size_t nextK = 0;
    while (nextK < nE && exerciseTimes_[nextK] <= t + kEps) ++nextK;

    // Unexercised paths: E[ D(0, tau) swap(tau) / D(0, t) | x(t) ] over exercises after t.
    if (nextK < nE) {
      const std::vector<double> x = s.rateState.column(j);
      for (std::size_t p = 0; p < nP; ++p) target[p] = strategy[nextK][p] / s.deflator(p, j);
      regression.fit(x, target);
      for (std::size_t p = 0; p < nP; ++p) continuation[p] = std::max(regression.predict(x[p]), 0.0);
    } else {
      std::fill(continuation.begin(), continuation.end(), 0.0);
    }

    // Exercised paths: value of the swap entered into (physical) or nothing (cash).
    if (settlement_ == Settlement::Physical) {
      for (std::size_t k = 0; k < nextK; ++k) {
        tailValues[k].resize(nP);
        tails[k].valueAtTime(s, j, tailValues[k].data());
      }
    }
    for (std::size_t p = 0; p < nP; ++p) {
      const std::size_t k = result.exerciseIndex[p];
      double v;
      if (k != npos && k < nextK) {
        v = settlement_ == Settlement::Physical ? tailValues[k][p] : 0.0;
      } else {
        v = continuation[p];
      }
      result.value(p, j) = position_ * v;
    }
  }

  double sum = 0.0;
  for (std::size_t p = 0; p < nP; ++p) sum += result.value(p, 0);
  result.price = sum / static_cast<double>(nP);
  return result;
}

}  // namespace ccr
