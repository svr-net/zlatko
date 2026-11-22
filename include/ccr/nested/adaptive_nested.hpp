#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ccr/core/matrix.hpp"
#include "ccr/core/parallel.hpp"

namespace ccr::nested {

/// Adaptive nested Monte Carlo on top of a non-nested simulation (chapter X11).
///
/// A statistic of conditional values V(X) = E[Y | X] on one date (EE*, PFE, ...) is estimated from
/// the paths of an ordinary simulation. Each outer path's own payoff Y, discounted to the date, is a
/// one-sample unbiased estimate of V at that path's state, so the ordinary run already is a nest with
/// one inner path. The estimator
///  1. splits the outer scenarios into a training half and an evaluation half;
///  2. fits a least-squares proxy of V on the training half (HermiteProxy), with the estimation
///     variance of every prediction;
///  3. bins the evaluation scenarios by proxy value and, in a stratified audit, simulates inner paths
///     for a few scenarios per bin to measure the proxy's bias per bin;
///  4. asks each statistic which scenarios could change it and would gain from more inner paths, and
///     simulates inner batches for those only, until none asks, the budget is spent or rounds run out.
/// Every scenario's estimate blends the proxy with its own samples in proportion to their precision.
/// Statistics that select scenarios (a sign, a tail) and then average them keep two independent
/// sample folds per scenario, so the samples that select a scenario never also value it.

/// The outer scenarios on one date.
struct OuterSample {
  Matrix features;             ///< one row per scenario: regression features of its (Markov) state
  std::vector<double> payoff;  ///< Y_i: the outer path's own payoff discounted to the date
  std::vector<double> weight;  ///< w_i of weighted statistics (D(0, t) for EE*); empty: all 1
  std::size_t size() const { return payoff.size(); }
};

/// Inner simulation of scenario `scenario` (a row of OuterSample): writes n independent payoffs,
/// discounted to the date, from that scenario's state into out[0 .. n). All randomness must come from
/// `seed`. It is called concurrently for different scenarios.
using InnerSimulator = std::function<void(std::size_t scenario, std::uint64_t seed, double* out, std::size_t n)>;

/// Mixes (seed, a, b, c) into an independent 64-bit stream seed (SplitMix64 finaliser).
std::uint64_t deriveSeed(std::uint64_t seed, std::uint64_t a, std::uint64_t b = 0, std::uint64_t c = 0);

/// Counter-based standard normals (SplitMix64 stream, Box-Muller pairs). The numbers depend only on
/// the seed, not on the platform's <random> or the thread count, so a nest seeded per (scenario,
/// batch) is reproducible however the work is split.
class CounterNormals {
 public:
  explicit CounterNormals(std::uint64_t seed) : state_(seed) {}
  double next();
  void fill(double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = next();
  }

 private:
  std::uint64_t state_;
  double spare_ = 0.0;
  bool hasSpare_ = false;
};

/// Least-squares proxy of V: probabilists' Hermite polynomials of the standardised features, all
/// products up to a total degree, fitted by ridge-stabilised normal equations, with the
/// heteroscedasticity-robust (HC0 sandwich) variance of every prediction. Features with no spread in
/// the training sample are dropped. The variance covers finite training data only; misspecification
/// is measured separately by the audit.
class HermiteProxy {
 public:
  explicit HermiteProxy(std::size_t degree = 3, double ridge = 1e-10) : degree_(degree), ridge_(ridge) {}

  /// Fits on rows `rows` of `features` against y[row]. Lowers the degree until there are at least
  /// two training rows per basis function.
  void fit(const Matrix& features, const std::vector<double>& y, const std::vector<std::size_t>& rows);
  /// Proxy value at a feature row; `variance` receives its estimation variance.
  double predict(const double* features, double& variance) const;

  std::size_t degree() const { return fittedDegree_; }
  std::size_t numBasis() const { return terms_.size(); }
  const std::vector<double>& coefficients() const { return beta_; }

 private:
  void basis(const double* x, double* phi) const;

  std::size_t degree_;
  double ridge_;
  std::size_t fittedDegree_ = 0;
  std::size_t dim_ = 0;
  std::vector<std::size_t> active_;               // features with spread
  std::vector<std::vector<unsigned>> terms_;      // per basis function: the power of each active feature
  std::vector<double> mean_, scale_, beta_, cov_; // cov_: p x p
};

/// What is known about every evaluation scenario: its proxy value and uncertainty, its bin, and its
/// inner samples in two folds. Fold 0 starts with the outer payoff Y and receives the first half of
/// every inner batch; fold 1 receives the second half.
class ScenarioEstimates {
 public:
  ScenarioEstimates(std::vector<double> outerPayoff, std::vector<double> weight, std::vector<double> proxyMean,
                    std::vector<double> proxyVariance, std::vector<std::size_t> bin, std::vector<double> residualVarianceByBin,
                    std::size_t minInnerForVariance, std::size_t maxInnerPerScenario);

  std::size_t size() const { return proxyMean_.size(); }
  std::size_t numBins() const { return residualVariance_.size(); }
  double weight(std::size_t i) const { return weight_[i]; }
  /// The proxy corrected by its bin's measured bias.
  double proxyMean(std::size_t i) const { return proxyMean_[i] + shift_[bin_[i]]; }
  double rawProxyMean(std::size_t i) const { return proxyMean_[i]; }
  std::size_t bin(std::size_t i) const { return bin_[i]; }
  double residualVariance(std::size_t b) const { return residualVariance_[b]; }
  double misspecificationVariance(std::size_t b) const { return misspecification_[b]; }
  /// The bin's systematic proxy bias (mean of inner mean - proxy), added to the proxy.
  double biasShift(std::size_t b) const { return shift_[b]; }
  /// Proxy error variance: estimation variance plus the variance of the bias around the bin's mean.
  double proxyVariance(std::size_t i) const { return proxyVariance_[i] + misspecification_[bin_[i]]; }
  /// Inner paths simulated for the scenario (its outer payoff not counted).
  std::size_t innerPaths(std::size_t i) const { return count0_[i] + count1_[i] - 1; }
  std::size_t batches(std::size_t i) const { return batches_[i]; }

  /// Variance of one payoff given the state: the sample variance once there are enough inner paths,
  /// else the bin's training residual variance (conservative: it also holds the proxy's bias).
  double withinVariance(std::size_t i) const;
  /// Best estimate of V: the proxy blended with all samples by precision.
  double estimate(std::size_t i) const;
  double stdError(std::size_t i) const;
  /// Share of the error variance of estimate(i) that refining up to the per-scenario cap would remove.
  double refinementGain(std::size_t i) const;
  /// Estimate for selecting the scenario (proxy + fold 0); independent of valueEstimate(i).
  double selectionEstimate(std::size_t i) const;
  /// Unbiased, noisy value: the fold-1 mean, else the fold-0 mean (outer payoff and odd batches).
  double valueEstimate(std::size_t i) const;
  double valueVariance(std::size_t i) const;

  /// Merges a batch of inner payoffs: the first half into fold 0, the rest into fold 1. Thread-safe
  /// across scenarios.
  void addInnerBatch(std::size_t i, const double* payoffs, std::size_t n);
  /// Re-estimates the proxy's bias per bin from every scenario with enough inner paths: the mean of
  /// d = inner mean - proxy shifts the proxy, and Var(d) - sampling variance - estimation variance is
  /// the remaining error variance (bins with too few such scenarios get no shift and the global one).
  void updateMisspecification(std::size_t minScenariosPerBin);

 private:
  void pooled(std::size_t i, double& n, double& mean, double& m2) const;
  void blend(std::size_t i, double sampleMean, double sampleVariance, double& value, double& variance) const;

  std::vector<double> weight_, proxyMean_, proxyVariance_;
  std::vector<std::size_t> bin_;
  std::vector<double> residualVariance_, misspecification_, shift_;
  std::size_t minInnerForVariance_, maxInnerPerScenario_;
  std::vector<std::size_t> count0_, count1_, batches_;
  std::vector<double> mean0_, mean1_, m20_, m21_;
};

/// A statistic of the conditional values on one date, with the rule that says which scenarios need
/// more inner paths.
class Statistic {
 public:
  virtual ~Statistic() = default;
  virtual std::string name() const = 0;
  /// The statistic from the scenario estimates.
  virtual double evaluate(const ScenarioEstimates& s) const = 0;
  /// The statistic of plain point values (exact values, the proxy alone, a uniform nest).
  virtual double plugIn(const std::vector<double>& values, const std::vector<double>& weights) const = 0;
  /// Raises urgency[i] to this statistic's urgency of scenario i where it is higher; urgency above 1
  /// asks for another inner batch, larger first.
  virtual void urgency(const ScenarioEstimates& s, std::vector<double>& urgency) const = 0;
};

/// E[w max(V, 0)]: EE with w = 1, EE* with w = D(0, t). The sign of each scenario is decided by the
/// selection estimate and its value comes from independent, unbiased samples, so the proxy enters only
/// through misclassified signs. Half the tolerance max(absoluteTolerance, relativeTolerance |EE|) goes
/// to the sampling noise of the values counted (the noisiest are refined), half to the expected
/// misclassification error sum_i w_i |V_i| Phi(-|V_i| / se_i) / n (the largest contributors are refined).
class PositivePart : public Statistic {
 public:
  explicit PositivePart(double relativeTolerance = 0.01, double absoluteTolerance = 0.0, std::string label = "EE");
  std::string name() const override { return label_; }
  double evaluate(const ScenarioEstimates& s) const override;
  double plugIn(const std::vector<double>& values, const std::vector<double>& weights) const override;
  void urgency(const ScenarioEstimates& s, std::vector<double>& urgency) const override;

 private:
  double relative_, absolute_;
  std::string label_;
};

/// The `level` quantile of V across scenarios (ccr::quantile); with `positivePart` the quantile of
/// max(V, 0), as PFE. A scenario is refined while the probability that it lies on the wrong side of
/// the quantile, times its refinement gain, exceeds Phi(-zCritical).
class Quantile : public Statistic {
 public:
  explicit Quantile(double level, bool positivePart = false, double zCritical = 2.0, std::string label = "quantile");
  std::string name() const override { return label_; }
  double evaluate(const ScenarioEstimates& s) const override;
  double plugIn(const std::vector<double>& values, const std::vector<double>& weights) const override;
  void urgency(const ScenarioEstimates& s, std::vector<double>& urgency) const override;

 private:
  double level_;
  bool positivePart_;
  double zCritical_;
  std::string label_;
};

struct AdaptiveOptions {
  double trainingFraction = 0.5;         ///< outer scenarios that fit the proxy; the rest are evaluated
  std::size_t hermiteDegree = 3;
  std::size_t innerBatch = 64;           ///< inner paths per refinement step
  std::size_t minInnerForVariance = 32;  ///< inner paths before a scenario's own sample variance is used
  std::size_t maxInnerPerScenario = 4096;
  std::size_t innerBudget = 2000000;     ///< inner paths of the whole estimate, audit included
  std::size_t maxScenariosPerRound = 2000;
  std::size_t maxRounds = 50;
  std::size_t auditPerBin = 16;          ///< random scenarios per bin refined regardless of the statistics (at most a fifth of the budget)
  std::size_t minPerBinForMisspecification = 8;
  /// Cut points of the bins over proxy values (probabilities), fine in both tails.
  std::vector<double> binProbabilities{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995};
  bool refineOutsideTrainingRange = true;  ///< always refine scenarios whose features extrapolate
  std::uint64_t seed = 20261122;

  /// Throws std::invalid_argument on inconsistent settings.
  void validate() const;
};

enum class StopReason { Converged, ScenarioCaps, Budget, MaxRounds };
const char* toString(StopReason reason);

struct BinDiagnostics {
  double lowerProbability = 0.0, upperProbability = 1.0;
  std::size_t scenarios = 0, refined = 0;
  double residualStdDev = 0.0;  ///< payoff noise around the proxy (training residuals)
  double biasShift = 0.0;       ///< measured systematic proxy bias (added to the proxy)
  double biasRms = 0.0;         ///< spread of the proxy bias around it: sqrt of the misspecification variance
};

struct AdaptiveResult {
  std::vector<std::string> names;      ///< per statistic
  std::vector<double> estimate;        ///< per statistic, after refinement
  std::vector<double> baseline;        ///< per statistic, before any inner path (proxy selects, outer payoffs value)
  std::vector<double> regression;      ///< per statistic, plug-in of the proxy alone (least-squares Monte Carlo)
  std::vector<std::size_t> training, evaluation;  ///< scenario indices of the split
  std::vector<double> proxy;           ///< proxy value per evaluation scenario
  std::vector<double> value;           ///< final blended estimate per evaluation scenario
  std::vector<std::size_t> inner;      ///< inner paths per evaluation scenario
  std::size_t innerPaths = 0, auditInnerPaths = 0, refinedScenarios = 0, auditedScenarios = 0, rounds = 0, unresolved = 0;
  std::size_t proxyDegree = 0, proxyBasis = 0;
  StopReason stop = StopReason::Converged;
  std::vector<BinDiagnostics> bins;
};

/// The adaptive nested estimate of `statistics` (refined jointly: a scenario is refined when any
/// statistic asks for it).
AdaptiveResult estimateAdaptive(const OuterSample& outer, const InnerSimulator& inner, const std::vector<const Statistic*>& statistics,
                                const AdaptiveOptions& options, SpinPool& pool = defaultPool());

/// The classical nested estimate on given scenarios: `innerPerScenario` inner paths for each, V
/// estimated by their mean, and each statistic's plug-in of those means. Seeds are independent of
/// estimateAdaptive's, so the two do not share samples.
struct UniformResult {
  std::vector<double> estimate;  ///< per statistic
  std::vector<double> value;     ///< per scenario
  std::size_t innerPaths = 0;
};
UniformResult estimateUniform(const OuterSample& outer, const InnerSimulator& inner, const std::vector<std::size_t>& scenarios,
                              std::size_t innerPerScenario, const std::vector<const Statistic*>& statistics, std::uint64_t seed,
                              SpinPool& pool = defaultPool());

}  // namespace ccr::nested
