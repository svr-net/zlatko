#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace ccr {

/// Sorted set of simulation dates (year fractions), always starting at t = 0.
///
/// Exposure is only observed on the grid, so the grid must contain the dates
/// where the portfolio changes (cash flows, fixings, exercise dates) and, for
/// collateralised netting sets, the margin call dates t - MPR.
class TimeGrid {
 public:
  TimeGrid() : times_{0.0} {}
  /// Times are sorted, de-duplicated (tolerance 1e-9); negative values are dropped and 0 is added.
  explicit TimeGrid(std::vector<double> times);

  /// `steps` equal steps from 0 to `horizon`.
  static TimeGrid uniform(double horizon, std::size_t steps);
  /// Typical exposure grid: weekly for three months, monthly to one year, quarterly thereafter.
  static TimeGrid standardExposureGrid(double horizon);

  /// Grid with additional dates (e.g. trade event dates) merged in.
  TimeGrid merged(const std::vector<double>& extra) const;
  /// Grid with t - lag added for every t > lag (margin call dates for a margin period of risk `lag`).
  TimeGrid withLaggedTimes(double lag) const;

  std::size_t size() const { return times_.size(); }
  double operator[](std::size_t i) const { return times_[i]; }
  const std::vector<double>& times() const { return times_; }
  double horizon() const { return times_.back(); }

  /// Largest index i with times[i] <= t (up to a small tolerance); 0 for t < 0.
  std::size_t indexAtOrBefore(double t) const;
  /// Index of t if it is on the grid.
  std::optional<std::size_t> find(double t, double tolerance = 1e-9) const;

 private:
  std::vector<double> times_;
};

}  // namespace ccr
