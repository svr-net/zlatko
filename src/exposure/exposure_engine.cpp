#include "ccr/exposure/exposure_engine.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ccr {

Matrix netValues(const std::vector<Matrix>& tradeValues) {
  if (tradeValues.empty()) throw std::invalid_argument("netValues: no trades");
  Matrix total = tradeValues.front();
  for (std::size_t i = 1; i < tradeValues.size(); ++i) total += tradeValues[i];
  return total;
}

Matrix grossPositiveValues(const std::vector<Matrix>& tradeValues) {
  if (tradeValues.empty()) throw std::invalid_argument("grossPositiveValues: no trades");
  Matrix total(tradeValues.front().rows(), tradeValues.front().cols(), 0.0);
  for (const Matrix& m : tradeValues)
    for (std::size_t p = 0; p < m.rows(); ++p)
      for (std::size_t j = 0; j < m.cols(); ++j) total(p, j) += std::max(m(p, j), 0.0);
  return total;
}

Matrix selectColumns(const Matrix& m, const std::vector<std::size_t>& columns) {
  Matrix out(m.rows(), columns.size());
  for (std::size_t r = 0; r < m.rows(); ++r)
    for (std::size_t c = 0; c < columns.size(); ++c) out(r, c) = m(r, columns.at(c));
  return out;
}

ExposureEngine::ExposureEngine(ScenarioGenerator generator, SimulationConfig config, TimeGrid baseGrid)
    : generator_(std::move(generator)), config_(config), baseGrid_(std::move(baseGrid)) {}

TimeGrid ExposureEngine::reportingGrid(const NettingSet& nettingSet) const {
  std::vector<double> events;
  for (const auto& trade : nettingSet.trades) {
    const auto times = trade->eventTimes();
    events.insert(events.end(), times.begin(), times.end());
  }
  return baseGrid_.merged(events);
}

TimeGrid ExposureEngine::simulationGrid(const NettingSet& nettingSet) const {
  TimeGrid grid = reportingGrid(nettingSet);
  if (nettingSet.collateral) grid = grid.withLaggedTimes(nettingSet.collateral->marginPeriodOfRisk);
  return grid;
}

ExposureResult ExposureEngine::run(const NettingSet& nettingSet, double pfeQuantile) const {
  if (nettingSet.trades.empty()) throw std::invalid_argument("ExposureEngine::run: empty netting set");
  ExposureResult result;
  const TimeGrid reporting = reportingGrid(nettingSet);
  result.scenarios = generator_.generate(simulationGrid(nettingSet), config_);
  const ScenarioSet& s = result.scenarios;
  for (double t : reporting.times()) result.reportingIndices.push_back(*s.grid.find(t));

  result.tradeValues.reserve(nettingSet.trades.size());
  for (const auto& trade : nettingSet.trades) result.tradeValues.push_back(trade->valueCube(s));
  result.nettedValue = netValues(result.tradeValues);

  if (nettingSet.collateral) {
    result.collateral = simulateCollateral(*nettingSet.collateral, result.nettedValue, s.grid);
  } else {
    result.collateral = Matrix(s.numPaths, s.numTimes(), 0.0);
  }
  result.exposureValue = result.nettedValue - result.collateral;
  result.profile = ExposureProfile::compute(selectColumns(result.exposureValue, result.reportingIndices), reporting,
                                           selectColumns(s.deflator, result.reportingIndices), pfeQuantile);
  return result;
}

}  // namespace ccr
