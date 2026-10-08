#include "ccr/models/scenario_generator.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "ccr/core/random.hpp"

namespace ccr {

std::size_t ScenarioSet::assetIndex(const std::string& name) const {
  for (std::size_t i = 0; i < assetModels.size(); ++i)
    if (assetModels[i]->name() == name) return i;
  throw std::out_of_range("ScenarioSet: unknown asset '" + name + "'");
}

std::size_t ScenarioSet::creditIndex(const std::string& name) const {
  for (std::size_t i = 0; i < creditModels.size(); ++i)
    if (creditModels[i]->name() == name) return i;
  throw std::out_of_range("ScenarioSet: unknown credit '" + name + "'");
}

ScenarioGenerator::ScenarioGenerator(std::shared_ptr<const HullWhite1F> rates,
                                     std::vector<std::shared_ptr<const LognormalAsset>> assets,
                                     std::vector<std::shared_ptr<const CirIntensity>> credits, Matrix correlation)
    : rates_(std::move(rates)),
      assets_(std::move(assets)),
      credits_(std::move(credits)),
      correlation_(std::move(correlation)) {
  if (!rates_) throw std::invalid_argument("ScenarioGenerator: null rate model");
  for (const auto& a : assets_)
    if (!a) throw std::invalid_argument("ScenarioGenerator: null asset model");
  for (const auto& c : credits_)
    if (!c) throw std::invalid_argument("ScenarioGenerator: null credit model");
  const std::size_t n = numFactors();
  if (correlation_.empty()) correlation_ = Matrix::identity(n);
  if (correlation_.rows() != n || correlation_.cols() != n)
    throw std::invalid_argument("ScenarioGenerator: correlation matrix has wrong dimension");
  for (std::size_t i = 0; i < n; ++i)
    if (std::fabs(correlation_(i, i) - 1.0) > 1e-12)
      throw std::invalid_argument("ScenarioGenerator: correlation matrix must have unit diagonal");
  choleskyFactor_ = cholesky(correlation_);
}

ScenarioGenerator ScenarioGenerator::withRateModel(std::shared_ptr<const HullWhite1F> rates) const {
  return ScenarioGenerator(std::move(rates), assets_, credits_, correlation_);
}

ScenarioGenerator ScenarioGenerator::withAsset(std::size_t index, std::shared_ptr<const LognormalAsset> asset) const {
  auto assets = assets_;
  assets.at(index) = std::move(asset);
  return ScenarioGenerator(rates_, std::move(assets), credits_, correlation_);
}

ScenarioGenerator ScenarioGenerator::withCredit(std::size_t index, std::shared_ptr<const CirIntensity> credit) const {
  auto credits = credits_;
  credits.at(index) = std::move(credit);
  return ScenarioGenerator(rates_, assets_, std::move(credits), correlation_);
}

ScenarioSet ScenarioGenerator::generate(const TimeGrid& grid, const SimulationConfig& config) const {
  if (config.numPaths == 0) throw std::invalid_argument("ScenarioGenerator::generate: numPaths must be positive");
  const std::size_t nT = grid.size();
  const std::size_t nP = config.numPaths;
  const std::size_t nA = assets_.size();
  const std::size_t nC = credits_.size();
  const std::size_t nCorr = numFactors();

  // Layout of the normals drawn per step: [correlated drivers | HW integral | CIR bridges...]
  std::vector<std::size_t> creditOffset(nC);
  std::size_t nNormals = nCorr + 1;
  for (std::size_t c = 0; c < nC; ++c) {
    creditOffset[c] = nNormals;
    nNormals += credits_[c]->extraNormals();
  }

  ScenarioSet s;
  s.grid = grid;
  s.numPaths = nP;
  s.rateModel = rates_;
  s.rateState = Matrix(nP, nT, 0.0);
  s.deflator = Matrix(nP, nT, 1.0);
  s.assetModels = assets_;
  s.assetValues.assign(nA, Matrix(nP, nT, 0.0));
  s.creditModels = credits_;
  s.intensityState.assign(nC, Matrix(nP, nT, 0.0));
  s.survival.assign(nC, Matrix(nP, nT, 1.0));

  NormalGenerator rng(config.seed);
  const std::size_t stepCount = nT > 0 ? nT - 1 : 0;
  std::vector<double> normals(stepCount * nNormals);
  std::vector<double> w(nCorr);

  for (std::size_t p = 0; p < nP; ++p) {
    const bool mirror = config.antithetic && (p % 2 == 1);
    if (mirror) {
      for (double& z : normals) z = -z;
    } else {
      rng.fill(normals.data(), normals.size());
    }

    double x = 0.0;
    double deflator = 1.0;
    std::vector<double> spot(nA);
    for (std::size_t a = 0; a < nA; ++a) {
      spot[a] = assets_[a]->spot();
      s.assetValues[a](p, 0) = spot[a];
    }
    std::vector<double> y(nC), integratedY(nC, 0.0);
    for (std::size_t c = 0; c < nC; ++c) {
      y[c] = credits_[c]->y0();
      s.intensityState[c](p, 0) = y[c];
    }

    for (std::size_t j = 1; j < nT; ++j) {
      const double t0 = grid[j - 1];
      const double t1 = grid[j];
      const double* z = normals.data() + (j - 1) * nNormals;
      for (std::size_t i = 0; i < nCorr; ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k <= i; ++k) sum += choleskyFactor_(i, k) * z[k];
        w[i] = sum;
      }

      double integratedRate = 0.0;
      rates_->evolve(t0, t1, x, w[0], z[nCorr], x, integratedRate);
      deflator *= std::exp(-integratedRate);
      s.rateState(p, j) = x;
      s.deflator(p, j) = deflator;

      for (std::size_t a = 0; a < nA; ++a) {
        spot[a] = assets_[a]->evolve(t0, t1, spot[a], integratedRate, w[1 + a]);
        s.assetValues[a](p, j) = spot[a];
      }
      for (std::size_t c = 0; c < nC; ++c) {
        credits_[c]->evolve(t1 - t0, w[1 + nA + c], z + creditOffset[c], y[c], integratedY[c]);
        s.intensityState[c](p, j) = y[c];
        s.survival[c](p, j) = credits_[c]->survival(t1, integratedY[c]);
      }
    }
  }
  return s;
}

}  // namespace ccr
