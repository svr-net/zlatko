// Adaptive nested Monte Carlo (chapter X11): a least-squares proxy on the outer paths, a stratified
// audit of its bias, and inner simulations only where a statistic needs them. See adaptive_nested.hpp.
#include "ccr/nested/adaptive_nested.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "ccr/core/math.hpp"

namespace ccr::nested {

namespace {

constexpr double kMinGain = 0.05;  // refining a scenario that removes less of its error variance is skipped
constexpr std::uint64_t kAdaptiveTag = 0x6164617074697665ull;  // "adaptive"
constexpr std::uint64_t kUniformTag = 0x756e69666f726d00ull;   // "uniform"
constexpr std::uint64_t kSplitTag = 0x73706c6974000000ull;     // "split"

std::uint64_t splitMix(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Deterministic Fisher-Yates shuffle from a seed.
void shuffle(std::vector<std::size_t>& v, std::uint64_t seed) {
  std::uint64_t s = seed;
  for (std::size_t i = v.size(); i > 1; --i) {
    s = splitMix(s);
    const auto j = static_cast<std::size_t>(s % i);
    std::swap(v[i - 1], v[j]);
  }
}

// Forward and back substitution with a lower-triangular Cholesky factor: solves L L^T x = b.
std::vector<double> choleskySolve(const Matrix& l, std::vector<double> b) {
  const std::size_t n = l.rows();
  for (std::size_t i = 0; i < n; ++i) {
    double s = b[i];
    for (std::size_t k = 0; k < i; ++k) s -= l(i, k) * b[k];
    b[i] = s / l(i, i);
  }
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t k = i + 1; k < n; ++k) s -= l(k, i) * b[k];
    b[i] = s / l(i, i);
  }
  return b;
}

// All exponent vectors over `dims` features with total degree <= degree, in graded order.
std::vector<std::vector<unsigned>> totalDegreeTerms(std::size_t dims, std::size_t degree) {
  std::vector<std::vector<unsigned>> out;
  std::vector<unsigned> current(dims, 0);
  std::function<void(std::size_t, std::size_t)> rec = [&](std::size_t position, std::size_t remaining) {
    if (position == dims) {
      out.push_back(current);
      return;
    }
    for (std::size_t k = 0; k <= remaining; ++k) {
      current[position] = static_cast<unsigned>(k);
      rec(position + 1, remaining - k);
    }
    current[position] = 0;
  };
  rec(0, degree);
  std::stable_sort(out.begin(), out.end(), [](const std::vector<unsigned>& a, const std::vector<unsigned>& b) {
    return std::accumulate(a.begin(), a.end(), 0u) < std::accumulate(b.begin(), b.end(), 0u);
  });
  return out;
}

double misclassificationProbability(double gap, double stdError) {
  return stdError > 0.0 ? normalCdf(-gap / stdError) : 0.0;
}

// Expected error of a scenario on the wrong side of a cutoff: its distance times the probability.
double misclassificationError(double gap, double stdError) { return gap * misclassificationProbability(gap, stdError); }

// Error-budget rule: marks the scenarios with the largest reducible error (contribution x refinement
// gain) until together they cover `excess`.
void markLargestContributors(const std::vector<double>& contribution, const ScenarioEstimates& s, double excess,
                             std::vector<double>& urgency) {
  const std::size_t n = contribution.size();
  std::vector<double> gain(n), reducible(n);
  for (std::size_t i = 0; i < n; ++i) {
    gain[i] = s.refinementGain(i);
    reducible[i] = contribution[i] * gain[i];
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return reducible[a] > reducible[b]; });
  double covered = 0.0;
  for (std::size_t i : order) {
    if (covered >= excess || reducible[i] <= 0.0) break;
    if (gain[i] < kMinGain) continue;
    urgency[i] = std::max(urgency[i], 2.0);
    covered += reducible[i];
  }
}

// Number of cut points strictly below the value.
std::size_t binOf(const std::vector<double>& thresholds, double value) {
  return static_cast<std::size_t>(std::lower_bound(thresholds.begin(), thresholds.end(), value) - thresholds.begin());
}

// Index of the empirical p-quantile in an ascending sample of size n (order statistic, no interpolation).
std::size_t quantileIndex(std::size_t n, double p) {
  const double k = std::ceil(static_cast<double>(n) * p - 1e-9) - 1.0;
  return static_cast<std::size_t>(std::clamp(k, 0.0, static_cast<double>(n - 1)));
}

}  // namespace

std::uint64_t deriveSeed(std::uint64_t seed, std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  std::uint64_t x = splitMix(seed + 0x9E3779B97F4A7C15ull);
  x = splitMix(x ^ a);
  x = splitMix(x ^ (b * 0xD1B54A32D192ED03ull));
  return splitMix(x ^ (c * 0x8CB92BA72F3D8DD7ull));
}

double CounterNormals::next() {
  if (hasSpare_) {
    hasSpare_ = false;
    return spare_;
  }
  state_ += 0x9E3779B97F4A7C15ull;
  const std::uint64_t h1 = splitMix(state_);
  const std::uint64_t h2 = splitMix(h1 ^ 0x68E31DA4B5297A4Dull);
  const double u1 = (static_cast<double>(h1 >> 11) + 0.5) * 0x1.0p-53;  // (0, 1)
  const double u2 = static_cast<double>(h2 >> 11) * 0x1.0p-53;          // [0, 1)
  const double r = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
  spare_ = r * std::sin(a);
  hasSpare_ = true;
  return r * std::cos(a);
}

// ------------------------------------------------------------------ HermiteProxy

void HermiteProxy::basis(const double* x, double* phi) const {
  const std::size_t m = active_.size(), d = fittedDegree_;
  std::vector<double> h(m * (d + 1));
  for (std::size_t f = 0; f < m; ++f) {
    const std::size_t j = active_[f];
    const double z = (x[j] - mean_[j]) / scale_[j];
    double* hf = &h[f * (d + 1)];
    hf[0] = 1.0;
    if (d >= 1) hf[1] = z;
    for (std::size_t k = 1; k < d; ++k) hf[k + 1] = z * hf[k] - static_cast<double>(k) * hf[k - 1];
  }
  for (std::size_t t = 0; t < terms_.size(); ++t) {
    double v = 1.0;
    for (std::size_t f = 0; f < m; ++f)
      if (terms_[t][f] > 0) v *= h[f * (d + 1) + terms_[t][f]];
    phi[t] = v;
  }
}

void HermiteProxy::fit(const Matrix& features, const std::vector<double>& y, const std::vector<std::size_t>& rows) {
  const std::size_t n = rows.size();
  if (n < 2) throw std::invalid_argument("HermiteProxy: needs at least two training rows");
  if (features.rows() != y.size()) throw std::invalid_argument("HermiteProxy: features and targets differ in length");
  dim_ = features.cols();
  mean_.assign(dim_, 0.0);
  scale_.assign(dim_, 1.0);
  for (std::size_t r : rows)
    for (std::size_t j = 0; j < dim_; ++j) mean_[j] += features(r, j);
  for (double& m : mean_) m /= static_cast<double>(n);
  std::vector<double> ss(dim_, 0.0);
  for (std::size_t r : rows)
    for (std::size_t j = 0; j < dim_; ++j) ss[j] += (features(r, j) - mean_[j]) * (features(r, j) - mean_[j]);
  active_.clear();
  for (std::size_t j = 0; j < dim_; ++j) {
    const double sd = std::sqrt(ss[j] / static_cast<double>(n - 1));
    if (sd > 1e-12 * std::max(1.0, std::fabs(mean_[j]))) {
      scale_[j] = sd;
      active_.push_back(j);
    }
  }
  // The largest degree with at least two training rows per basis function.
  for (std::size_t d = degree_;; --d) {
    terms_ = totalDegreeTerms(active_.size(), d);
    fittedDegree_ = d;
    if (2 * terms_.size() <= n || d == 0) break;
  }
  if (terms_.size() > 2000) throw std::invalid_argument("HermiteProxy: more than 2000 basis functions");

  const std::size_t p = terms_.size();
  Matrix gram(p, p);
  std::vector<double> rhs(p, 0.0), phi(p);
  for (std::size_t r : rows) {
    basis(features.row(r), phi.data());
    for (std::size_t a = 0; a < p; ++a) {
      rhs[a] += phi[a] * y[r];
      for (std::size_t c = 0; c <= a; ++c) gram(a, c) += phi[a] * phi[c];
    }
  }
  double trace = 0.0;
  for (std::size_t a = 0; a < p; ++a) {
    trace += gram(a, a);
    for (std::size_t c = 0; c < a; ++c) gram(c, a) = gram(a, c);
  }
  // Ridge relative to the mean diagonal, raised until the factorisation succeeds.
  Matrix l;
  for (double ridge = ridge_;; ridge *= 1e3) {
    Matrix g = gram;
    const double lambda = ridge * std::max(trace / static_cast<double>(p), 1e-300);
    for (std::size_t a = 0; a < p; ++a) g(a, a) += lambda;
    try {
      l = cholesky(g);
      break;
    } catch (const std::invalid_argument&) {
      if (ridge > 1e-2) throw std::invalid_argument("HermiteProxy: normal equations are singular");
    }
  }
  beta_ = choleskySolve(l, rhs);

  // HC0 sandwich: Cov(beta) = G^-1 (sum e_i^2 phi_i phi_i^T) G^-1.
  Matrix meat(p, p);
  for (std::size_t r : rows) {
    basis(features.row(r), phi.data());
    double fitted = 0.0;
    for (std::size_t a = 0; a < p; ++a) fitted += phi[a] * beta_[a];
    const double e2 = (y[r] - fitted) * (y[r] - fitted);
    for (std::size_t a = 0; a < p; ++a)
      for (std::size_t c = 0; c <= a; ++c) meat(a, c) += e2 * phi[a] * phi[c];
  }
  for (std::size_t a = 0; a < p; ++a)
    for (std::size_t c = 0; c < a; ++c) meat(c, a) = meat(a, c);
  Matrix inverse(p, p);
  for (std::size_t c = 0; c < p; ++c) {
    std::vector<double> unit(p, 0.0);
    unit[c] = 1.0;
    const std::vector<double> col = choleskySolve(l, unit);
    for (std::size_t r = 0; r < p; ++r) inverse(r, c) = col[r];
  }
  Matrix tmp(p, p);
  for (std::size_t a = 0; a < p; ++a)
    for (std::size_t k = 0; k < p; ++k) {
      const double v = inverse(a, k);
      if (v == 0.0) continue;
      for (std::size_t c = 0; c < p; ++c) tmp(a, c) += v * meat(k, c);
    }
  cov_.assign(p * p, 0.0);
  for (std::size_t a = 0; a < p; ++a)
    for (std::size_t k = 0; k < p; ++k) {
      const double v = tmp(a, k);
      if (v == 0.0) continue;
      for (std::size_t c = 0; c < p; ++c) cov_[a * p + c] += v * inverse(k, c);
    }
}

double HermiteProxy::predict(const double* features, double& variance) const {
  if (beta_.empty()) throw std::logic_error("HermiteProxy: predict before fit");
  const std::size_t p = terms_.size();
  std::vector<double> phi(p);
  basis(features, phi.data());
  double value = 0.0, var = 0.0;
  for (std::size_t a = 0; a < p; ++a) {
    value += phi[a] * beta_[a];
    double acc = 0.0;
    for (std::size_t c = 0; c < p; ++c) acc += cov_[a * p + c] * phi[c];
    var += phi[a] * acc;
  }
  variance = std::max(var, 0.0);
  return value;
}

// ------------------------------------------------------------------ ScenarioEstimates

ScenarioEstimates::ScenarioEstimates(std::vector<double> outerPayoff, std::vector<double> weight, std::vector<double> proxyMean,
                                     std::vector<double> proxyVariance, std::vector<std::size_t> bin,
                                     std::vector<double> residualVarianceByBin, std::size_t minInnerForVariance,
                                     std::size_t maxInnerPerScenario)
    : weight_(std::move(weight)),
      proxyMean_(std::move(proxyMean)),
      proxyVariance_(std::move(proxyVariance)),
      bin_(std::move(bin)),
      residualVariance_(std::move(residualVarianceByBin)),
      misspecification_(residualVariance_.size(), 0.0),
      shift_(residualVariance_.size(), 0.0),
      minInnerForVariance_(minInnerForVariance),
      maxInnerPerScenario_(maxInnerPerScenario) {
  const std::size_t n = proxyMean_.size();
  if (outerPayoff.size() != n || weight_.size() != n || proxyVariance_.size() != n || bin_.size() != n)
    throw std::invalid_argument("ScenarioEstimates: inconsistent sizes");
  count0_.assign(n, 1);
  count1_.assign(n, 0);
  batches_.assign(n, 0);
  mean0_ = std::move(outerPayoff);
  mean1_.assign(n, 0.0);
  m20_.assign(n, 0.0);
  m21_.assign(n, 0.0);
}

void ScenarioEstimates::pooled(std::size_t i, double& n, double& mean, double& m2) const {
  const auto n0 = static_cast<double>(count0_[i]), n1 = static_cast<double>(count1_[i]);
  n = n0 + n1;
  if (count1_[i] == 0) {
    mean = mean0_[i];
    m2 = m20_[i];
    return;
  }
  const double d = mean1_[i] - mean0_[i];
  mean = mean0_[i] + d * n1 / n;
  m2 = m20_[i] + m21_[i] + d * d * (n0 * n1 / n);
}

void ScenarioEstimates::blend(std::size_t i, double sampleMean, double sampleVariance, double& value, double& variance) const {
  const double proxyVar = proxyVariance(i), proxy = proxyMean(i);
  const double denominator = proxyVar + sampleVariance;
  if (denominator <= 0.0) {
    value = proxy;
    variance = 0.0;
    return;
  }
  value = proxy + proxyVar / denominator * (sampleMean - proxy);
  variance = proxyVar * sampleVariance / denominator;
}

double ScenarioEstimates::withinVariance(std::size_t i) const {
  double n = 0.0, mean = 0.0, m2 = 0.0;
  pooled(i, n, mean, m2);
  return n >= 1.0 + static_cast<double>(minInnerForVariance_) ? m2 / (n - 1.0) : residualVariance_[bin_[i]];
}

double ScenarioEstimates::estimate(std::size_t i) const {
  double n = 0.0, mean = 0.0, m2 = 0.0, value = 0.0, variance = 0.0;
  pooled(i, n, mean, m2);
  blend(i, mean, withinVariance(i) / n, value, variance);
  return value;
}

double ScenarioEstimates::stdError(std::size_t i) const {
  double n = 0.0, mean = 0.0, m2 = 0.0, value = 0.0, variance = 0.0;
  pooled(i, n, mean, m2);
  blend(i, mean, withinVariance(i) / n, value, variance);
  return std::sqrt(variance);
}

double ScenarioEstimates::refinementGain(std::size_t i) const {
  double n = 0.0, mean = 0.0, m2 = 0.0, value = 0.0, current = 0.0, best = 0.0;
  pooled(i, n, mean, m2);
  const double within = withinVariance(i);
  blend(i, mean, within / n, value, current);
  if (current <= 0.0) return 0.0;
  const auto used = static_cast<double>(innerPaths(i));
  const double remaining = std::max(0.0, static_cast<double>(maxInnerPerScenario_) - used);
  blend(i, mean, within / (n + remaining), value, best);
  return 1.0 - best / current;
}

double ScenarioEstimates::selectionEstimate(std::size_t i) const {
  if (count1_[i] == 0) return proxyMean(i);
  double value = 0.0, variance = 0.0;
  blend(i, mean0_[i], withinVariance(i) / static_cast<double>(count0_[i]), value, variance);
  return value;
}

double ScenarioEstimates::valueEstimate(std::size_t i) const { return count1_[i] > 0 ? mean1_[i] : mean0_[i]; }

double ScenarioEstimates::valueVariance(std::size_t i) const {
  return withinVariance(i) / static_cast<double>(count1_[i] > 0 ? count1_[i] : count0_[i]);
}

namespace {

// Chan et al. merge of n payoffs into a running (count, mean, M2).
void merge(std::size_t& count, double& mean, double& m2, const double* payoffs, std::size_t n) {
  if (n == 0) return;
  double sum = 0.0;
  for (std::size_t k = 0; k < n; ++k) sum += payoffs[k];
  const double batchMean = sum / static_cast<double>(n);
  double batchM2 = 0.0;
  for (std::size_t k = 0; k < n; ++k) batchM2 += (payoffs[k] - batchMean) * (payoffs[k] - batchMean);
  const auto na = static_cast<double>(count), nb = static_cast<double>(n), total = na + nb;
  const double delta = batchMean - mean;
  mean += delta * nb / total;
  m2 += batchM2 + delta * delta * (na * nb / total);
  count += n;
}

}  // namespace

void ScenarioEstimates::addInnerBatch(std::size_t i, const double* payoffs, std::size_t n) {
  if (n == 0) return;
  for (std::size_t k = 0; k < n; ++k)
    if (!std::isfinite(payoffs[k])) throw std::runtime_error("nested Monte Carlo: inner simulation returned a non-finite payoff");
  // Every batch is split between the folds: its first half sharpens the selection (fold 0), the rest
  // values the scenario (fold 1), so both improve from the first batch and never share a sample.
  const std::size_t toSelection = n / 2;
  merge(count0_[i], mean0_[i], m20_[i], payoffs, toSelection);
  merge(count1_[i], mean1_[i], m21_[i], payoffs + toSelection, n - toSelection);
  ++batches_[i];
}

void ScenarioEstimates::updateMisspecification(std::size_t minScenariosPerBin) {
  // Per bin, over the scenarios with enough inner paths: d = mean - proxy. Its average is the bin's
  // systematic bias (it shifts the proxy); its spread, net of the sampling and estimation variances,
  // is the proxy's remaining error variance.
  const std::size_t bins = numBins();
  std::vector<double> sum(bins, 0.0), sumSq(bins, 0.0), noise(bins, 0.0);
  std::vector<std::size_t> count(bins, 0);
  double globalSq = 0.0, globalNoise = 0.0;
  std::size_t globalCount = 0;
  for (std::size_t i = 0; i < size(); ++i) {
    double n = 0.0, mean = 0.0, m2 = 0.0;
    pooled(i, n, mean, m2);
    if (n < 1.0 + static_cast<double>(minInnerForVariance_)) continue;
    const double d = mean - proxyMean_[i];
    const double sampling = m2 / (n - 1.0) / n + proxyVariance_[i];
    const std::size_t b = bin_[i];
    sum[b] += d;
    sumSq[b] += d * d;
    noise[b] += sampling;
    ++count[b];
    globalSq += d * d;
    globalNoise += sampling;
    ++globalCount;
  }
  const double global = globalCount > 0 ? std::max(0.0, (globalSq - globalNoise) / static_cast<double>(globalCount)) : 0.0;
  for (std::size_t b = 0; b < bins; ++b) {
    if (count[b] >= minScenariosPerBin) {
      const auto c = static_cast<double>(count[b]);
      shift_[b] = sum[b] / c;
      misspecification_[b] = std::max(0.0, (sumSq[b] - c * shift_[b] * shift_[b] - noise[b]) / c);
    } else {
      shift_[b] = 0.0;
      misspecification_[b] = global;
    }
  }
}

// ------------------------------------------------------------------ statistics

PositivePart::PositivePart(double relativeTolerance, double absoluteTolerance, std::string label)
    : relative_(relativeTolerance), absolute_(absoluteTolerance), label_(std::move(label)) {
  if (!(relative_ >= 0.0) || !(absolute_ >= 0.0) || relative_ + absolute_ == 0.0)
    throw std::invalid_argument("PositivePart: tolerances must be non-negative and not both zero");
}

double PositivePart::evaluate(const ScenarioEstimates& s) const {
  double sum = 0.0;
  for (std::size_t i = 0; i < s.size(); ++i)
    if (s.selectionEstimate(i) > 0.0) sum += s.weight(i) * s.valueEstimate(i);
  return s.size() ? sum / static_cast<double>(s.size()) : 0.0;
}

double PositivePart::plugIn(const std::vector<double>& values, const std::vector<double>& weights) const {
  double sum = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) sum += (weights.empty() ? 1.0 : weights[i]) * std::max(values[i], 0.0);
  return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

void PositivePart::urgency(const ScenarioEstimates& s, std::vector<double>& urgency) const {
  const std::size_t n = s.size();
  if (n == 0) return;
  const double half = 0.5 * std::max(absolute_, relative_ * std::fabs(evaluate(s)));
  if (!(half > 0.0)) return;
  const auto nn = static_cast<double>(n);
  // 1) Sampling noise of the values counted (independent across scenarios, so it adds in quadrature):
  //    refine the noisiest until it is within half the tolerance, each to an equal share of it.
  double noise = 0.0;
  std::size_t counted = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (s.selectionEstimate(i) > 0.0) {
      noise += s.weight(i) * s.weight(i) * s.valueVariance(i);
      ++counted;
    }
  if (counted > 0 && std::sqrt(noise) / nn > half) {
    const double share = half * half * nn * nn / static_cast<double>(counted);
    for (std::size_t i = 0; i < n; ++i)
      if (s.selectionEstimate(i) > 0.0) urgency[i] = std::max(urgency[i], s.weight(i) * s.weight(i) * s.valueVariance(i) / share);
  }
  // 2) Scenarios whose sign may be wrong: refine the largest expected errors until within the other half.
  std::vector<double> contribution(n);
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    contribution[i] = s.weight(i) * misclassificationError(std::fabs(s.estimate(i)), s.stdError(i)) / nn;
    total += contribution[i];
  }
  if (total > half) markLargestContributors(contribution, s, total - half, urgency);
}

Quantile::Quantile(double level, bool positivePart, double zCritical, std::string label)
    : level_(level), positivePart_(positivePart), zCritical_(zCritical), label_(std::move(label)) {
  if (!(level_ > 0.0 && level_ < 1.0)) throw std::invalid_argument("Quantile: level must be in (0, 1)");
  if (!(zCritical_ > 0.0)) throw std::invalid_argument("Quantile: zCritical must be positive");
}

double Quantile::plugIn(const std::vector<double>& values, const std::vector<double>&) const {
  if (values.empty()) return 0.0;
  const double q = quantile(values, level_);
  return positivePart_ ? std::max(q, 0.0) : q;
}

double Quantile::evaluate(const ScenarioEstimates& s) const {
  std::vector<double> v(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) v[i] = s.estimate(i);
  return plugIn(v, {});
}

void Quantile::urgency(const ScenarioEstimates& s, std::vector<double>& urgency) const {
  const std::size_t n = s.size();
  if (n == 0) return;
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = s.estimate(i);
  const double q = quantile(v, level_);
  if (positivePart_ && q < 0.0) return;  // the positive part is 0 however far below zero the quantile is
  const double threshold = normalCdf(-zCritical_);
  for (std::size_t i = 0; i < n; ++i) {
    const double p = misclassificationProbability(std::fabs(v[i] - q), s.stdError(i));
    urgency[i] = std::max(urgency[i], s.refinementGain(i) * p / threshold);
  }
}

// ------------------------------------------------------------------ the estimator

void AdaptiveOptions::validate() const {
  if (!(trainingFraction > 0.0 && trainingFraction < 1.0)) throw std::invalid_argument("AdaptiveOptions: trainingFraction must be in (0, 1)");
  if (innerBatch < 1) throw std::invalid_argument("AdaptiveOptions: innerBatch must be at least 1");
  if (minInnerForVariance < 2) throw std::invalid_argument("AdaptiveOptions: minInnerForVariance must be at least 2");
  if (maxInnerPerScenario < std::max(innerBatch, minInnerForVariance))
    throw std::invalid_argument("AdaptiveOptions: maxInnerPerScenario must be at least innerBatch and minInnerForVariance");
  if (maxScenariosPerRound < 1) throw std::invalid_argument("AdaptiveOptions: maxScenariosPerRound must be at least 1");
  if (minPerBinForMisspecification < 1) throw std::invalid_argument("AdaptiveOptions: minPerBinForMisspecification must be at least 1");
  for (std::size_t k = 0; k < binProbabilities.size(); ++k) {
    const double p = binProbabilities[k];
    if (!(p > 0.0 && p < 1.0) || (k > 0 && !(p > binProbabilities[k - 1])))
      throw std::invalid_argument("AdaptiveOptions: binProbabilities must be strictly increasing in (0, 1)");
  }
}

const char* toString(StopReason reason) {
  switch (reason) {
    case StopReason::Converged: return "converged";
    case StopReason::ScenarioCaps: return "scenario caps";
    case StopReason::Budget: return "budget";
    case StopReason::MaxRounds: return "max rounds";
  }
  return "unknown";
}

namespace {

// One inner batch for each listed evaluation scenario that fits in the budget; returns the paths used.
std::size_t runBatch(ScenarioEstimates& s, const std::vector<std::size_t>& evaluation, const std::vector<std::size_t>& chosen,
                     std::size_t batchSize, std::size_t maxInner, std::size_t budget, std::uint64_t seed,
                     const InnerSimulator& inner, SpinPool& pool) {
  struct Job {
    std::size_t i, paths;
    std::uint64_t seed;
  };
  std::vector<Job> jobs;
  std::size_t total = 0;
  for (std::size_t i : chosen) {
    const std::size_t used = s.innerPaths(i);
    const std::size_t m = used >= maxInner ? 0 : std::min(batchSize, maxInner - used);
    if (m == 0) continue;
    if (total + m > budget) break;
    // The seed depends on the scenario and its batch number only, not on the order or the threads.
    jobs.push_back({i, m, deriveSeed(seed, evaluation[i], s.batches(i), kAdaptiveTag)});
    total += m;
  }
  pool.parallelFor(jobs.size(), 1, [&](std::size_t begin, std::size_t end) {
    std::vector<double> buffer;
    for (std::size_t k = begin; k < end; ++k) {
      buffer.resize(jobs[k].paths);
      inner(evaluation[jobs[k].i], jobs[k].seed, buffer.data(), jobs[k].paths);
      s.addInnerBatch(jobs[k].i, buffer.data(), jobs[k].paths);
    }
  });
  return total;
}

// The scenarios to refine this round. Each statistic ranks its own requests (urgency above 1) and the
// round takes them in turns, so one statistic's urgencies, on their own scale, cannot crowd out
// another's; scenarios outside the training range come first.
std::vector<std::size_t> candidates(const ScenarioEstimates& s, const std::vector<const Statistic*>& statistics,
                                    const std::vector<char>& outside, const AdaptiveOptions& o, std::size_t& capped) {
  const std::size_t n = s.size();
  std::vector<std::vector<std::size_t>> queues;
  std::vector<char> wanted(n, 0);
  capped = 0;
  for (const Statistic* st : statistics) {
    std::vector<double> urgency(n, 0.0);
    st->urgency(s, urgency);
    std::vector<std::pair<double, std::size_t>> keyed;
    for (std::size_t i = 0; i < n; ++i) {
      if (!(urgency[i] > 1.0)) continue;  // also skips NaN
      if (s.innerPaths(i) >= o.maxInnerPerScenario) {
        if (!wanted[i]) ++capped;
        wanted[i] = 1;
        continue;
      }
      keyed.emplace_back(urgency[i], i);
    }
    std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::size_t> queue;
    for (const auto& k : keyed) queue.push_back(k.second);
    queues.push_back(std::move(queue));
  }
  std::vector<std::size_t> out;
  std::vector<char> taken(n, 0);
  if (o.refineOutsideTrainingRange)
    for (std::size_t i = 0; i < n && out.size() < o.maxScenariosPerRound; ++i)
      if (outside[i] && s.innerPaths(i) < o.minInnerForVariance) {
        out.push_back(i);
        taken[i] = 1;
      }
  for (std::size_t depth = 0; out.size() < o.maxScenariosPerRound; ++depth) {
    bool any = false;
    for (const auto& queue : queues) {
      if (depth >= queue.size()) continue;
      any = true;
      const std::size_t i = queue[depth];
      if (!taken[i] && out.size() < o.maxScenariosPerRound) {
        out.push_back(i);
        taken[i] = 1;
      }
    }
    if (!any) break;
  }
  return out;
}

}  // namespace

AdaptiveResult estimateAdaptive(const OuterSample& outer, const InnerSimulator& inner, const std::vector<const Statistic*>& statistics,
                                const AdaptiveOptions& o, SpinPool& pool) {
  o.validate();
  const std::size_t n = outer.size();
  if (outer.features.rows() != n) throw std::invalid_argument("estimateAdaptive: features and payoffs differ in length");
  if (!outer.weight.empty() && outer.weight.size() != n) throw std::invalid_argument("estimateAdaptive: weights differ in length");
  if (statistics.empty()) throw std::invalid_argument("estimateAdaptive: no statistic");
  if (!inner) throw std::invalid_argument("estimateAdaptive: no inner simulator");

  // 1. Training / evaluation split.
  const auto nTrain = static_cast<std::size_t>(std::llround(static_cast<double>(n) * o.trainingFraction));
  if (nTrain < 2 || n - nTrain < 1) throw std::invalid_argument("estimateAdaptive: too few outer scenarios for the split");
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  shuffle(order, deriveSeed(o.seed, kSplitTag));
  AdaptiveResult r;
  r.training.assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(nTrain));
  r.evaluation.assign(order.begin() + static_cast<std::ptrdiff_t>(nTrain), order.end());
  std::sort(r.training.begin(), r.training.end());
  std::sort(r.evaluation.begin(), r.evaluation.end());
  const std::size_t nEval = r.evaluation.size();

  // 2. The proxy on the training half, predicted out of sample on the evaluation half.
  HermiteProxy proxy(o.hermiteDegree);
  proxy.fit(outer.features, outer.payoff, r.training);
  r.proxyDegree = proxy.degree();
  r.proxyBasis = proxy.numBasis();
  std::vector<double> mu(nEval), estVar(nEval), weight(nEval), payoff(nEval);
  for (std::size_t k = 0; k < nEval; ++k) {
    const std::size_t i = r.evaluation[k];
    mu[k] = proxy.predict(outer.features.row(i), estVar[k]);
    weight[k] = outer.weight.empty() ? 1.0 : outer.weight[i];
    payoff[k] = outer.payoff[i];
  }

  // 3. Bins over the proxy values, and the payoff noise per bin from the training residuals.
  std::vector<double> sorted = mu;
  std::sort(sorted.begin(), sorted.end());
  std::vector<double> thresholds;
  for (double p : o.binProbabilities) thresholds.push_back(sorted[quantileIndex(nEval, p)]);
  const std::size_t numBins = thresholds.size() + 1;
  std::vector<std::size_t> bin(nEval);
  for (std::size_t k = 0; k < nEval; ++k) bin[k] = binOf(thresholds, mu[k]);
  std::vector<double> residualSum(numBins, 0.0);
  std::vector<std::size_t> residualCount(numBins, 0);
  double residualTotal = 0.0;
  for (std::size_t i : r.training) {
    double v = 0.0;
    const double fitted = proxy.predict(outer.features.row(i), v);
    const double e2 = (outer.payoff[i] - fitted) * (outer.payoff[i] - fitted);
    const std::size_t b = binOf(thresholds, fitted);
    residualSum[b] += e2;
    ++residualCount[b];
    residualTotal += e2;
  }
  const double residualGlobal = residualTotal / static_cast<double>(nTrain);
  std::vector<double> residualVariance(numBins);
  for (std::size_t b = 0; b < numBins; ++b)
    residualVariance[b] = residualCount[b] >= 2 ? residualSum[b] / static_cast<double>(residualCount[b]) : residualGlobal;

  ScenarioEstimates s(payoff, weight, mu, estVar, bin, residualVariance, o.minInnerForVariance, o.maxInnerPerScenario);
  for (const Statistic* st : statistics) {
    r.names.push_back(st->name());
    r.baseline.push_back(st->evaluate(s));
    r.regression.push_back(st->plugIn(mu, weight));
  }
  r.proxy = mu;

  // Extrapolation: evaluation scenarios with a feature outside the training range.
  const std::size_t dims = outer.features.cols();
  std::vector<double> lo(dims, std::numeric_limits<double>::infinity()), hi(dims, -std::numeric_limits<double>::infinity());
  for (std::size_t i : r.training)
    for (std::size_t j = 0; j < dims; ++j) {
      lo[j] = std::min(lo[j], outer.features(i, j));
      hi[j] = std::max(hi[j], outer.features(i, j));
    }
  std::vector<char> outside(nEval, 0);
  for (std::size_t k = 0; k < nEval; ++k)
    for (std::size_t j = 0; j < dims; ++j)
      if (outer.features(r.evaluation[k], j) < lo[j] || outer.features(r.evaluation[k], j) > hi[j]) outside[k] = 1;

  // 4. Audit: random scenarios per bin, refined regardless of the statistics, measure the bias.
  // The audit takes at most a fifth of the budget (fewer scenarios per bin when the budget is small).
  const std::size_t auditBatch = std::max(o.innerBatch, o.minInnerForVariance);
  const std::size_t auditPerBin = std::min(o.auditPerBin, std::max<std::size_t>(2, o.innerBudget / 5 / (numBins * auditBatch)));
  std::vector<std::size_t> audit;
  if (o.auditPerBin > 0) {
    std::vector<std::vector<std::size_t>> members(numBins);
    for (std::size_t k = 0; k < nEval; ++k) members[bin[k]].push_back(k);
    for (std::size_t b = 0; b < numBins; ++b) {
      shuffle(members[b], deriveSeed(o.seed, kSplitTag, b + 1));
      for (std::size_t q = 0; q < std::min(auditPerBin, members[b].size()); ++q) audit.push_back(members[b][q]);
    }
  }
  std::size_t used = runBatch(s, r.evaluation, audit, auditBatch, o.maxInnerPerScenario, o.innerBudget, o.seed, inner, pool);
  r.auditInnerPaths = used;
  s.updateMisspecification(o.minPerBinForMisspecification);

  // 5. Refinement rounds.
  std::size_t capped = 0;
  for (;;) {
    if (r.rounds >= o.maxRounds) {
      r.stop = StopReason::MaxRounds;
      break;
    }
    const std::vector<std::size_t> chosen = candidates(s, statistics, outside, o, capped);
    if (chosen.empty()) {
      r.stop = capped > 0 ? StopReason::ScenarioCaps : StopReason::Converged;
      break;
    }
    const std::size_t spent = runBatch(s, r.evaluation, chosen, o.innerBatch, o.maxInnerPerScenario, o.innerBudget - used, o.seed, inner, pool);
    if (spent == 0) {
      r.stop = StopReason::Budget;
      break;
    }
    used += spent;
    ++r.rounds;
    s.updateMisspecification(o.minPerBinForMisspecification);
  }
  r.unresolved = candidates(s, statistics, outside, o, capped).size() + capped;

  for (const Statistic* st : statistics) r.estimate.push_back(st->evaluate(s));
  r.innerPaths = used;
  r.value.resize(nEval);
  r.inner.resize(nEval);
  std::vector<std::size_t> perBin(numBins, 0), refinedPerBin(numBins, 0);
  for (std::size_t k = 0; k < nEval; ++k) {
    r.value[k] = s.estimate(k);
    r.inner[k] = s.innerPaths(k);
    ++perBin[bin[k]];
    if (r.inner[k] > 0) {
      ++r.refinedScenarios;
      ++refinedPerBin[bin[k]];
    }
  }
  for (std::size_t k : audit)
    if (s.innerPaths(k) > 0) ++r.auditedScenarios;
  for (std::size_t b = 0; b < numBins; ++b) {
    BinDiagnostics d;
    d.lowerProbability = b == 0 ? 0.0 : o.binProbabilities[b - 1];
    d.upperProbability = b + 1 == numBins ? 1.0 : o.binProbabilities[b];
    d.scenarios = perBin[b];
    d.refined = refinedPerBin[b];
    d.residualStdDev = std::sqrt(s.residualVariance(b));
    d.biasShift = s.biasShift(b);
    d.biasRms = std::sqrt(s.misspecificationVariance(b));
    r.bins.push_back(d);
  }
  return r;
}

UniformResult estimateUniform(const OuterSample& outer, const InnerSimulator& inner, const std::vector<std::size_t>& scenarios,
                              std::size_t innerPerScenario, const std::vector<const Statistic*>& statistics, std::uint64_t seed,
                              SpinPool& pool) {
  if (innerPerScenario < 1) throw std::invalid_argument("estimateUniform: at least one inner path per scenario");
  UniformResult u;
  u.value.assign(scenarios.size(), 0.0);
  pool.parallelFor(scenarios.size(), 1, [&](std::size_t begin, std::size_t end) {
    std::vector<double> buffer(innerPerScenario);
    for (std::size_t k = begin; k < end; ++k) {
      inner(scenarios[k], deriveSeed(seed, scenarios[k], 0, kUniformTag), buffer.data(), innerPerScenario);
      u.value[k] = std::accumulate(buffer.begin(), buffer.end(), 0.0) / static_cast<double>(innerPerScenario);
    }
  });
  std::vector<double> weight(scenarios.size(), 1.0);
  if (!outer.weight.empty())
    for (std::size_t k = 0; k < scenarios.size(); ++k) weight[k] = outer.weight[scenarios[k]];
  for (const Statistic* st : statistics) u.estimate.push_back(st->plugIn(u.value, weight));
  u.innerPaths = scenarios.size() * innerPerScenario;
  return u;
}

}  // namespace ccr::nested
