#include "ccr/instruments/trade.hpp"

namespace ccr {

Matrix PathwiseTrade::valueCube(const ScenarioSet& scenarios) const {
  Matrix cube(scenarios.numPaths, scenarios.numTimes(), 0.0);
  std::vector<double> slice(scenarios.numPaths);
  for (std::size_t j = 0; j < scenarios.numTimes(); ++j) {
    valueAtTime(scenarios, j, slice.data());
    cube.setColumn(j, slice);
  }
  return cube;
}

}  // namespace ccr
