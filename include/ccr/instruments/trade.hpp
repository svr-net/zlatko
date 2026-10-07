#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/models/scenario_generator.hpp"

namespace ccr {

/// A trade that can be marked to market on every simulated scenario.
///
/// valueCube returns the mark-to-market (domestic currency, as seen at t_j and
/// after any cash flow paid exactly at t_j) on every path and grid date.
class Trade {
 public:
  explicit Trade(std::string id) : id_(std::move(id)) {}
  virtual ~Trade() = default;

  const std::string& id() const { return id_; }
  virtual double maturity() const = 0;
  /// Dates at which the trade's value jumps or that the valuation needs on the grid
  /// (fixings, payments, exercise dates). The exposure engine merges them into the grid.
  virtual std::vector<double> eventTimes() const { return {}; }

  virtual Matrix valueCube(const ScenarioSet& scenarios) const = 0;

 private:
  std::string id_;
};

/// A trade whose value at t_j is a closed-form function of the scenario state up to t_j.
class PathwiseTrade : public Trade {
 public:
  using Trade::Trade;

  /// Writes the value on every path at grid index j into out[0 .. numPaths).
  virtual void valueAtTime(const ScenarioSet& scenarios, std::size_t j, double* out) const = 0;

  Matrix valueCube(const ScenarioSet& scenarios) const override;
};

}  // namespace ccr
