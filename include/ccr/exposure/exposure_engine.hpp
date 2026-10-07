#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/core/time_grid.hpp"
#include "ccr/exposure/collateral.hpp"
#include "ccr/exposure/exposure_profile.hpp"
#include "ccr/instruments/trade.hpp"
#include "ccr/models/scenario_generator.hpp"

namespace ccr {

/// Trades whose values can be offset against each other on default (legally
/// enforceable netting agreement), optionally covered by a collateral agreement.
struct NettingSet {
  std::string id;
  std::vector<std::shared_ptr<const Trade>> trades;
  std::optional<CollateralAgreement> collateral;
};

/// Everything produced by one exposure run.
struct ExposureResult {
  ScenarioSet scenarios;            ///< simulated on the full grid (reporting dates + margin call dates)
  /// Indices into scenarios.grid of the reporting dates (base grid + trade events). Under a CSA
  /// the auxiliary margin call dates t - MPR are excluded: their own call dates are not on the
  /// grid, so exposure is only meaningful on the reporting dates.
  std::vector<std::size_t> reportingIndices;
  std::vector<Matrix> tradeValues;  ///< per trade, numPaths x numTimes
  Matrix nettedValue;               ///< sum of trade values (before collateral)
  Matrix collateral;                ///< collateral held (zero without a CSA)
  Matrix exposureValue;             ///< nettedValue - collateral: the value at risk on default
  ExposureProfile profile;          ///< statistics of exposureValue on the reporting dates
};

/// Sum of trade values on every path and date.
Matrix netValues(const std::vector<Matrix>& tradeValues);
/// Sum of positive trade values: the exposure without netting.
Matrix grossPositiveValues(const std::vector<Matrix>& tradeValues);
/// Matrix made of the given columns of `m` (e.g. the reporting dates of a path matrix).
Matrix selectColumns(const Matrix& m, const std::vector<std::size_t>& columns);

/// Monte Carlo exposure engine: scenario generation, revaluation of every trade on
/// every scenario, netting, collateral and exposure statistics.
class ExposureEngine {
 public:
  ExposureEngine(ScenarioGenerator generator, SimulationConfig config, TimeGrid baseGrid);

  /// Base grid with the trades' event dates merged in: the dates exposure is reported on.
  TimeGrid reportingGrid(const NettingSet& nettingSet) const;
  /// Reporting grid plus, under a CSA, the margin call dates t - MPR.
  TimeGrid simulationGrid(const NettingSet& nettingSet) const;
  ExposureResult run(const NettingSet& nettingSet, double pfeQuantile = 0.95) const;

  const ScenarioGenerator& generator() const { return generator_; }
  const SimulationConfig& config() const { return config_; }
  const TimeGrid& baseGrid() const { return baseGrid_; }

 private:
  ScenarioGenerator generator_;
  SimulationConfig config_;
  TimeGrid baseGrid_;
};

}  // namespace ccr
