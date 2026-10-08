#include "ccr/core/time_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ccr {

namespace {
constexpr double kTimeTolerance = 1e-9;
}

TimeGrid::TimeGrid(std::vector<double> times) {
  times.erase(std::remove_if(times.begin(), times.end(), [](double t) { return t < 0.0 || !std::isfinite(t); }),
              times.end());
  times.push_back(0.0);
  std::sort(times.begin(), times.end());
  times_.clear();
  for (double t : times)
    if (times_.empty() || t - times_.back() > kTimeTolerance) times_.push_back(t);
}

TimeGrid TimeGrid::uniform(double horizon, std::size_t steps) {
  if (horizon <= 0.0 || steps == 0) throw std::invalid_argument("TimeGrid::uniform: invalid arguments");
  std::vector<double> t(steps + 1);
  for (std::size_t i = 0; i <= steps; ++i) t[i] = horizon * static_cast<double>(i) / static_cast<double>(steps);
  return TimeGrid(std::move(t));
}

TimeGrid TimeGrid::standardExposureGrid(double horizon) {
  if (horizon <= 0.0) throw std::invalid_argument("TimeGrid::standardExposureGrid: horizon must be positive");
  std::vector<double> t;
  for (int w = 1; w <= 13; ++w) t.push_back(w * 7.0 / 365.0);
  for (int m = 4; m <= 12; ++m) t.push_back(m / 12.0);
  for (double q = 1.25; q < horizon + kTimeTolerance; q += 0.25) t.push_back(q);
  t.erase(std::remove_if(t.begin(), t.end(), [horizon](double x) { return x > horizon; }), t.end());
  t.push_back(horizon);
  return TimeGrid(std::move(t));
}

TimeGrid TimeGrid::merged(const std::vector<double>& extra) const {
  std::vector<double> t = times_;
  t.insert(t.end(), extra.begin(), extra.end());
  return TimeGrid(std::move(t));
}

TimeGrid TimeGrid::withLaggedTimes(double lag) const {
  if (lag <= 0.0) return *this;
  std::vector<double> t = times_;
  for (double x : times_)
    if (x > lag) t.push_back(x - lag);
  return TimeGrid(std::move(t));
}

std::size_t TimeGrid::indexAtOrBefore(double t) const {
  const auto it = std::upper_bound(times_.begin(), times_.end(), t + kTimeTolerance);
  if (it == times_.begin()) return 0;
  return static_cast<std::size_t>(std::distance(times_.begin(), it)) - 1;
}

std::optional<std::size_t> TimeGrid::find(double t, double tolerance) const {
  const std::size_t i = indexAtOrBefore(t);
  if (std::fabs(times_[i] - t) <= tolerance) return i;
  if (i + 1 < times_.size() && std::fabs(times_[i + 1] - t) <= tolerance) return i + 1;
  return std::nullopt;
}

}  // namespace ccr
