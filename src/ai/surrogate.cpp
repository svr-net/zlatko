#include "ccr/ai/surrogate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ccr::ai {

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// mulberry32, the deterministic generator of the initial MLP weights.
struct Mulberry32 {
  std::uint32_t a;
  double operator()() {
    a += 0x6D2B79F5u;
    std::uint32_t t = a;
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + (t ^ (t >> 7)) * (t | 61u);
    return static_cast<double>(t ^ (t >> 14)) / 4294967296.0;
  }
};

// Solves the symmetric positive definite system A x = b (Cholesky).
std::vector<double> solveSpd(Rows A, std::vector<double> b) {
  const std::size_t n = b.size();
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      double s = A[i][j];
      for (std::size_t k = 0; k < j; ++k) s -= A[i][k] * A[j][k];
      A[i][j] = i == j ? std::sqrt(s) : s / A[j][j];
    }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < i; ++k) b[i] -= A[i][k] * b[k];
    b[i] /= A[i][i];
  }
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t k = i + 1; k < n; ++k) b[i] -= A[k][i] * b[k];
    b[i] /= A[i][i];
  }
  return b;
}

std::vector<double> polyFeatures(const std::vector<double>& z, int degree) {
  std::vector<double> f{1.0};
  f.insert(f.end(), z.begin(), z.end());
  if (degree >= 2)
    for (std::size_t i = 0; i < z.size(); ++i)
      for (std::size_t j = i; j < z.size(); ++j) f.push_back(z[i] * z[j]);
  return f;
}

double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

}  // namespace

double totalOf(const std::vector<double>& y) {
  double s = 0.0;
  for (std::size_t k = 0; k < kTargets; ++k) s += kSigns[k] * y.at(k);
  return s;
}

Standardiser Standardiser::fit(const Rows& rows) {
  if (rows.empty()) throw std::invalid_argument("Standardiser: no rows");
  const std::size_t d = rows[0].size();
  const auto n = static_cast<double>(rows.size());
  Standardiser s{std::vector<double>(d, 0.0), std::vector<double>(d, 0.0)};
  for (const auto& r : rows)
    for (std::size_t j = 0; j < d; ++j) s.mean[j] += r[j] / n;
  for (const auto& r : rows)
    for (std::size_t j = 0; j < d; ++j) s.sd[j] += (r[j] - s.mean[j]) * (r[j] - s.mean[j]) / n;
  for (double& v : s.sd) v = std::sqrt(v) > 0.0 ? std::sqrt(v) : 1.0;
  return s;
}

std::vector<double> Standardiser::apply(const std::vector<double>& x) const {
  std::vector<double> z(mean.size());
  for (std::size_t j = 0; j < z.size(); ++j) z[j] = (x[j] - mean[j]) / sd[j];
  return z;
}

std::vector<double> Standardiser::invert(const std::vector<double>& z) const {
  std::vector<double> x(mean.size());
  for (std::size_t j = 0; j < x.size(); ++j) x[j] = z[j] * sd[j] + mean[j];
  return x;
}

// ------------------------------------------------------------------ ridge

std::size_t Ridge::parameters() const { return coefficients.size() * (coefficients.empty() ? 0 : coefficients[0].size()); }

std::vector<double> Ridge::predict(const std::vector<double>& x) const {
  const auto f = polyFeatures(sx.apply(x), degree);
  std::vector<double> z(coefficients.size(), 0.0);
  for (std::size_t k = 0; k < z.size(); ++k)
    for (std::size_t i = 0; i < f.size(); ++i) z[k] += coefficients[k][i] * f[i];
  return sy.invert(z);
}

Ridge fitRidge(const Rows& X, const Rows& Y, int degree, double lambda) {
  Ridge m;
  m.degree = degree;
  m.sx = Standardiser::fit(X);
  m.sy = Standardiser::fit(Y);
  Rows F;
  for (const auto& x : X) F.push_back(polyFeatures(m.sx.apply(x), degree));
  const std::size_t p = F[0].size();
  Rows A(p, std::vector<double>(p, 0.0));
  for (const auto& f : F)
    for (std::size_t i = 0; i < p; ++i)
      for (std::size_t j = 0; j <= i; ++j) A[i][j] += f[i] * f[j];
  for (std::size_t i = 0; i < p; ++i) {
    for (std::size_t j = 0; j < i; ++j) A[j][i] = A[i][j];
    if (i > 0) A[i][i] += lambda * static_cast<double>(X.size());
  }
  Rows Z;
  for (const auto& y : Y) Z.push_back(m.sy.apply(y));
  for (std::size_t k = 0; k < Y[0].size(); ++k) {
    std::vector<double> b(p, 0.0);
    for (std::size_t r = 0; r < F.size(); ++r)
      for (std::size_t i = 0; i < p; ++i) b[i] += F[r][i] * Z[r][k];
    m.coefficients.push_back(solveSpd(A, b));
  }
  return m;
}

// ------------------------------------------------------------------ MLP

std::size_t Mlp::parameters() const {
  std::size_t n = 0;
  for (std::size_t l = 0; l < 3; ++l) n += W[l].size() * W[l][0].size() + B[l].size();
  return n;
}

namespace {

// Activations of every layer (tanh hidden layers, linear output).
std::array<std::vector<double>, 4> forward(const Mlp& m, const std::vector<double>& x) {
  std::array<std::vector<double>, 4> a;
  a[0] = x;
  for (std::size_t l = 0; l < 3; ++l) {
    a[l + 1].resize(m.W[l].size());
    for (std::size_t i = 0; i < m.W[l].size(); ++i) {
      double s = m.B[l][i];
      for (std::size_t j = 0; j < a[l].size(); ++j) s += m.W[l][i][j] * a[l][j];
      a[l + 1][i] = l < 2 ? std::tanh(s) : s;
    }
  }
  return a;
}

}  // namespace

std::vector<double> Mlp::predict(const std::vector<double>& x) const { return sy.invert(forward(*this, sx.apply(x))[3]); }

Mlp fitMlp(const Rows& X, const Rows& Y, const MlpOptions& o, SpinPool& pool) {
  Mlp m;
  if (o.warm) {
    m = *o.warm;
  } else {
    m.sx = Standardiser::fit(X);
    m.sy = Standardiser::fit(Y);
    const std::array<std::size_t, 4> sizes{X[0].size(), o.hidden, o.hidden, Y[0].size()};
    Mulberry32 rnd{o.seed};
    auto gauss = [&] { return std::sqrt(-2.0 * std::log(rnd() + 1e-12)) * std::cos(2.0 * M_PI * rnd()); };
    for (std::size_t l = 0; l < 3; ++l) {
      m.W[l].assign(sizes[l + 1], std::vector<double>(sizes[l]));
      for (auto& row : m.W[l])
        for (double& w : row) w = gauss() * std::sqrt(1.0 / static_cast<double>(sizes[l]));
      m.B[l].assign(sizes[l + 1], 0.0);
    }
  }
  Rows Xs, Ys;
  for (const auto& x : X) Xs.push_back(m.sx.apply(x));
  for (const auto& y : Y) Ys.push_back(m.sy.apply(y));
  const std::size_t n = Xs.size(), nOut = Ys[0].size();

  // Parameters flattened: W0, B0, W1, B1, W2, B2.
  std::size_t nParams = 0;
  std::array<std::size_t, 3> offW{}, offB{};
  for (std::size_t l = 0; l < 3; ++l) {
    offW[l] = nParams, nParams += m.W[l].size() * m.W[l][0].size();
    offB[l] = nParams, nParams += m.B[l].size();
  }
  // The gradient over fixed chunks of samples, summed in chunk order: the same result on any
  // number of threads.
  constexpr std::size_t kChunk = 16;
  const std::size_t nChunks = (n + kChunk - 1) / kChunk;
  std::vector<std::vector<double>> partial(nChunks, std::vector<double>(nParams));
  std::vector<double> g(nParams), mom(nParams, 0.0), vel(nParams, 0.0);
  const double b1 = 0.9, b2 = 0.999;
  for (std::size_t ep = 1; ep <= o.epochs; ++ep) {
    pool.parallelFor(nChunks, 1, [&](std::size_t cb, std::size_t ce) {
      for (std::size_t c = cb; c < ce; ++c) {
        auto& gr = partial[c];
        std::fill(gr.begin(), gr.end(), 0.0);
        for (std::size_t s = c * kChunk; s < std::min(n, (c + 1) * kChunk); ++s) {
          const auto a = forward(m, Xs[s]);
          std::vector<double> delta(nOut);
          for (std::size_t k = 0; k < nOut; ++k) delta[k] = 2.0 * (a[3][k] - Ys[s][k]) / static_cast<double>(n * nOut);
          for (std::size_t l = 3; l-- > 0;) {
            const auto& prev = a[l];
            const std::size_t nPrev = prev.size();
            for (std::size_t i = 0; i < delta.size(); ++i) {
              gr[offB[l] + i] += delta[i];
              for (std::size_t j = 0; j < nPrev; ++j) gr[offW[l] + i * nPrev + j] += delta[i] * prev[j];
            }
            if (l > 0) {
              std::vector<double> next(nPrev, 0.0);
              for (std::size_t j = 0; j < nPrev; ++j) {
                double t = 0.0;
                for (std::size_t i = 0; i < delta.size(); ++i) t += m.W[l][i][j] * delta[i];
                next[j] = t * (1.0 - prev[j] * prev[j]);
              }
              delta = std::move(next);
            }
          }
        }
      }
    });
    std::fill(g.begin(), g.end(), 0.0);
    for (const auto& gr : partial)
      for (std::size_t q = 0; q < nParams; ++q) g[q] += gr[q];
    const double c1 = 1.0 - std::pow(b1, static_cast<double>(ep)), c2 = 1.0 - std::pow(b2, static_cast<double>(ep));
    auto adam = [&](double& w, std::size_t q, bool decay) {
      const double gq = g[q] + (decay ? o.weightDecay * w : 0.0);
      mom[q] = b1 * mom[q] + (1.0 - b1) * gq;
      vel[q] = b2 * vel[q] + (1.0 - b2) * gq * gq;
      w -= o.learningRate * (mom[q] / c1) / (std::sqrt(vel[q] / c2) + 1e-8);
    };
    for (std::size_t l = 0; l < 3; ++l) {
      const std::size_t nPrev = m.W[l][0].size();
      for (std::size_t i = 0; i < m.W[l].size(); ++i) {
        for (std::size_t j = 0; j < nPrev; ++j) adam(m.W[l][i][j], offW[l] + i * nPrev + j, true);
        adam(m.B[l][i], offB[l] + i, false);
      }
    }
  }
  return m;
}

// ------------------------------------------------------------------ metrics

ErrorStats errorStats(const std::vector<double>& t, const std::vector<double>& p, double scale) {
  ErrorStats r;
  std::vector<double> e(t.size()), abs(t.size()), sq(t.size()), rel(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    e[i] = p[i] - t[i];
    abs[i] = std::fabs(e[i]);
    sq[i] = e[i] * e[i];
    rel[i] = abs[i] / std::max(std::fabs(t[i]), 1e-9);
  }
  std::vector<double> sorted = abs;
  std::sort(sorted.begin(), sorted.end());
  r.mae = mean(abs);
  r.rmse = std::sqrt(mean(sq));
  r.bias = mean(e);
  r.max = sorted.empty() ? 0.0 : sorted.back();
  r.p95 = sorted.empty() ? 0.0 : sorted[std::min(sorted.size() - 1, static_cast<std::size_t>(0.95 * static_cast<double>(sorted.size())))];
  r.maeBp = 1e4 * r.mae / scale;
  r.mape = mean(rel);
  return r;
}

PnlExplain pnlExplain(const std::vector<double>& t, const std::vector<double>& p) {
  std::vector<double> dt, dp, resid, dt2, r2, hit;
  for (std::size_t i = 1; i < t.size(); ++i) {
    dt.push_back(t[i] - t[i - 1]);
    dp.push_back(p[i] - p[i - 1]);
    resid.push_back(dt.back() - dp.back());
    dt2.push_back(dt.back() * dt.back());
    r2.push_back(resid.back() * resid.back());
    const auto sgn = [](double v) { return (v > 0) - (v < 0); };
    hit.push_back(sgn(dt.back()) == sgn(dp.back()) || std::fabs(dt.back()) < 1.0 ? 1.0 : 0.0);
  }
  PnlExplain r;
  const double varT = mean(dt2) - mean(dt) * mean(dt), varR = mean(r2) - mean(resid) * mean(resid);
  r.effectiveness = 1.0 - varR / varT;
  r.signHit = mean(hit);
  r.residualSd = std::sqrt(varR);
  r.actualSd = std::sqrt(varT);
  return r;
}

// ------------------------------------------------------------------ back test

Backtest runBacktest(const Dataset& data, const BacktestOptions& o, SpinPool& pool) {
  auto six = [](const Rows& Y) {
    Rows out;
    for (const auto& y : Y) out.emplace_back(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(kTargets));
    return out;
  };
  const Rows trX = data.trainX, trY = six(data.trainY), X = data.testX, Y = six(data.testY);
  Backtest bt;
  bt.days = X.size();
  for (const auto& y : Y) bt.truthTotal.push_back(totalOf(y));
  for (const auto& x : X) {
    bool out = false;
    for (std::size_t j = 0; j < x.size(); ++j) out = out || x[j] < data.low[j] || x[j] > data.high[j];
    bt.outside.push_back(out);
    bt.outsideDays += out ? 1 : 0;
  }
  auto score = [&](const Rows& pred) {
    Score s;
    for (std::size_t i = 0; i < pred.size(); ++i) {
      s.predTotal.push_back(totalOf(pred[i]));
      s.errorTotal.push_back(s.predTotal.back() - bt.truthTotal[i]);
    }
    auto subset = [&](int which) {  // 0 all, 1 inside the training box, 2 outside
      std::vector<double> t, p;
      for (std::size_t i = 0; i < pred.size(); ++i)
        if (which == 0 || (which == 1) == !bt.outside[i]) t.push_back(bt.truthTotal[i]), p.push_back(s.predTotal[i]);
      return std::make_pair(t, p);
    };
    const auto [ta, pa] = subset(0);
    s.total = errorStats(ta, pa, data.grossNotional);
    if (const auto [ti, pi] = subset(1); !ti.empty()) s.inRange = errorStats(ti, pi, data.grossNotional);
    if (const auto [to, po] = subset(2); !to.empty()) s.outRange = errorStats(to, po, data.grossNotional);
    for (std::size_t k = 0; k < kTargets; ++k) {
      std::vector<double> t, p;
      for (std::size_t i = 0; i < pred.size(); ++i) t.push_back(Y[i][k]), p.push_back(pred[i][k]);
      s.components[k] = errorStats(t, p, data.grossNotional);
    }
    s.pnl = pnlExplain(bt.truthTotal, s.predTotal);
    return s;
  };
  Rows lag1, weekly;
  for (std::size_t i = 0; i < Y.size(); ++i) lag1.push_back(Y[i == 0 ? 0 : i - 1]), weekly.push_back(Y[i - i % 5]);
  bt.lag1 = score(lag1);
  bt.weekly = score(weekly);

  struct Spec { const char* key; const char* label; };
  for (const Spec spec : {Spec{"linear", "Linear (first order)"}, Spec{"poly2", "Quadratic ridge"}, Spec{"mlp", "MLP 5-16-16-6"}}) {
    const std::string key = spec.key;
    ModelResult r;
    r.key = key;
    r.label = spec.label;
    const auto fitRidgeFor = [&](const Rows& aX, const Rows& aY) { return fitRidge(aX, aY, key == "linear" ? 1 : 2, key == "linear" ? 1e-8 : 1e-6); };
    double t0 = nowMs();
    std::optional<Ridge> ridge;
    std::optional<Mlp> mlp;
    MlpOptions mo;
    mo.seed = 7;
    mo.epochs = o.mlpEpochs;
    if (key == "mlp") mlp = fitMlp(trX, trY, mo, pool);
    else ridge = fitRidgeFor(trX, trY);
    r.fitMs = nowMs() - t0;
    r.parameters = mlp ? mlp->parameters() : ridge->parameters();
    t0 = nowMs();
    Rows fixed;
    for (const auto& x : X) fixed.push_back(mlp ? mlp->predict(x) : ridge->predict(x));
    r.inferMicros = 1000.0 * (nowMs() - t0) / static_cast<double>(std::max<std::size_t>(1, X.size()));
    Rows rolling;
    for (std::size_t i = 0; i < bt.days; ++i) {
      if (i > 0 && i % o.refitEvery == 0) {
        Rows aX = trX, aY = trY;
        aX.insert(aX.end(), X.begin(), X.begin() + static_cast<std::ptrdiff_t>(i));
        aY.insert(aY.end(), Y.begin(), Y.begin() + static_cast<std::ptrdiff_t>(i));
        if (mlp) {
          MlpOptions refit;
          refit.seed = 7, refit.epochs = o.mlpRefitEpochs, refit.learningRate = 0.003;
          const Mlp warm = *mlp;
          refit.warm = &warm;
          mlp = fitMlp(aX, aY, refit, pool);
        } else {
          ridge = fitRidgeFor(aX, aY);
        }
      }
      rolling.push_back(mlp ? mlp->predict(X[i]) : ridge->predict(X[i]));
    }
    r.fittedOnce = score(fixed);
    r.walkForward = score(rolling);
    bt.models.push_back(std::move(r));
  }
  return bt;
}

}  // namespace ccr::ai
