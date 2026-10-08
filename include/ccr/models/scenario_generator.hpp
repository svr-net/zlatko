#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/core/time_grid.hpp"
#include "ccr/models/cir_intensity.hpp"
#include "ccr/models/hull_white.hpp"
#include "ccr/models/lognormal_asset.hpp"

namespace ccr {

/// Simulated market scenarios on a time grid ("the scenario cube").
///
/// All matrices are numPaths x grid.size(). Valuation at future dates uses only
/// the state on each path (x for rates, spot for assets), so any instrument with
/// a closed form in the models can be revalued path by path; instruments without
/// one are priced by regression (American Monte Carlo).
struct ScenarioSet {
  TimeGrid grid;
  std::size_t numPaths = 0;

  std::shared_ptr<const HullWhite1F> rateModel;
  Matrix rateState;  ///< Hull-White state x(t)
  Matrix deflator;   ///< D(0, t) = exp(-int_0^t r(s) ds), inverse of the bank-account numeraire

  std::vector<std::shared_ptr<const LognormalAsset>> assetModels;
  std::vector<Matrix> assetValues;

  std::vector<std::shared_ptr<const CirIntensity>> creditModels;
  std::vector<Matrix> intensityState;  ///< CIR component y(t)
  std::vector<Matrix> survival;        ///< Pathwise survival exp(-int_0^t lambda(s) ds)

  std::size_t numTimes() const { return grid.size(); }
  std::size_t assetIndex(const std::string& name) const;
  std::size_t creditIndex(const std::string& name) const;

  /// Domestic zero-coupon bond P(t_j, T) on a path.
  double zeroBond(std::size_t path, std::size_t timeIndex, double maturity) const {
    return rateModel->zeroBond(grid[timeIndex], maturity, rateState(path, timeIndex));
  }
};

struct SimulationConfig {
  std::size_t numPaths = 5000;
  std::uint64_t seed = 42;
  bool antithetic = true;  ///< pair every path with its mirror image (-Z)
};

/// Joint, correlated simulation of all risk factors under the domestic risk-neutral
/// measure with the bank account as numeraire.
///
/// Correlated drivers are ordered [rates, assets..., credits...]; `correlation`
/// must have that dimension (identity if left empty).
class ScenarioGenerator {
 public:
  ScenarioGenerator(std::shared_ptr<const HullWhite1F> rates,
                    std::vector<std::shared_ptr<const LognormalAsset>> assets = {},
                    std::vector<std::shared_ptr<const CirIntensity>> credits = {}, Matrix correlation = Matrix());

  ScenarioSet generate(const TimeGrid& grid, const SimulationConfig& config) const;

  std::size_t numFactors() const { return 1 + assets_.size() + credits_.size(); }
  const std::shared_ptr<const HullWhite1F>& rateModel() const { return rates_; }
  const std::vector<std::shared_ptr<const LognormalAsset>>& assets() const { return assets_; }
  const std::vector<std::shared_ptr<const CirIntensity>>& credits() const { return credits_; }
  const Matrix& correlation() const { return correlation_; }

  /// Copies with one model replaced (bump-and-revalue).
  ScenarioGenerator withRateModel(std::shared_ptr<const HullWhite1F> rates) const;
  ScenarioGenerator withAsset(std::size_t index, std::shared_ptr<const LognormalAsset> asset) const;
  ScenarioGenerator withCredit(std::size_t index, std::shared_ptr<const CirIntensity> credit) const;

 private:
  std::shared_ptr<const HullWhite1F> rates_;
  std::vector<std::shared_ptr<const LognormalAsset>> assets_;
  std::vector<std::shared_ptr<const CirIntensity>> credits_;
  Matrix correlation_;
  Matrix choleskyFactor_;
};

}  // namespace ccr
