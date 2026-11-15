#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ccr/core/parallel.hpp"

namespace ccr::ai {

/// Machine-learning surrogates of the XVA engine and their back test (chapter X10).
///
/// A surrogate maps a market state x = (counterparty CDS multiplier, own CDS multiplier, parallel
/// rate shift, EURUSD volatility, funding spread) to the six adjustments of the default netting
/// set (CVA, DVA, FCA, FBA, MVA, KVA). Models: polynomial ridge regression (degree 1 is the
/// first-order sensitivity approximation, degree 2 the quadratic one) and a small tanh multilayer
/// perceptron trained with full-batch Adam, its gradient computed on the lock-free pool over fixed
/// chunks of samples (so the result does not depend on the number of threads).

using Rows = std::vector<std::vector<double>>;
constexpr std::size_t kTargets = 6;
constexpr std::array<double, kTargets> kSigns{1, -1, 1, -1, 1, 1};  ///< total = cva - dva + fca - fba + mva + kva
inline const std::array<const char*, kTargets> kTargetNames{"cva", "dva", "fca", "fba", "mva", "kva"};
double totalOf(const std::vector<double>& y);

struct Standardiser {
  std::vector<double> mean, sd;
  static Standardiser fit(const Rows& rows);
  std::vector<double> apply(const std::vector<double>& x) const;
  std::vector<double> invert(const std::vector<double>& z) const;
};

struct Ridge {
  int degree = 2;
  Standardiser sx, sy;
  Rows coefficients;  ///< per target, over the polynomial features
  std::size_t parameters() const;
  std::vector<double> predict(const std::vector<double>& x) const;
};
Ridge fitRidge(const Rows& X, const Rows& Y, int degree, double lambda);

struct Mlp {
  Standardiser sx, sy;
  std::array<Rows, 3> W;                        ///< layers [in, h, h, out]
  std::array<std::vector<double>, 3> B;
  std::size_t parameters() const;
  std::vector<double> predict(const std::vector<double>& x) const;
};
struct MlpOptions {
  std::size_t hidden = 16, epochs = 2500;
  double learningRate = 0.01, weightDecay = 1e-6;
  std::uint32_t seed = 1;
  const Mlp* warm = nullptr;  ///< continue training from these weights and standardisers
};
Mlp fitMlp(const Rows& X, const Rows& Y, const MlpOptions& options, SpinPool& pool = defaultPool());

struct ErrorStats {
  double mae = 0, rmse = 0, bias = 0, max = 0, p95 = 0, maeBp = 0, mape = 0;
};
/// Errors of predictions p against truths t, maeBp as basis points of `scale` (gross notional).
ErrorStats errorStats(const std::vector<double>& t, const std::vector<double>& p, double scale);

struct PnlExplain {
  double effectiveness = 0, signHit = 0, residualSd = 0, actualSd = 0;
};
/// Daily P&L explain: how much of the change of the true XVA the surrogate's change reproduces.
PnlExplain pnlExplain(const std::vector<double>& t, const std::vector<double>& p);

struct Dataset {
  Rows trainX, trainY, testX, testY;  ///< Y rows hold at least the six adjustments
  std::vector<double> low, high;      ///< the training box
  double grossNotional = 1.0;
};

struct Score {
  std::vector<double> predTotal, errorTotal;  ///< per day: predicted total and its error vs the engine
  ErrorStats total;
  std::optional<ErrorStats> inRange, outRange;
  std::array<ErrorStats, kTargets> components;
  PnlExplain pnl;
};

struct ModelResult {
  std::string key, label;
  std::size_t parameters = 0;
  double fitMs = 0, inferMicros = 0;
  Score fittedOnce, walkForward;
};

struct BacktestOptions {
  std::size_t refitEvery = 21, mlpEpochs = 2500, mlpRefitEpochs = 400;
};

struct Backtest {
  std::size_t days = 0, outsideDays = 0;
  std::vector<double> truthTotal;
  std::vector<bool> outside;
  Score lag1, weekly;  ///< baselines: yesterday's engine value; the weekly engine run carried forward
  std::vector<ModelResult> models;  ///< linear, poly2, mlp
};

/// The back test on the daily market path: each model fitted once on the training window, and
/// walk-forward (refitted every refitEvery days on the window plus every earlier day).
Backtest runBacktest(const Dataset& data, const BacktestOptions& options = {}, SpinPool& pool = defaultPool());

}  // namespace ccr::ai
