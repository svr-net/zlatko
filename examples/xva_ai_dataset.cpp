// Ground truth for the AI chapter (docs/xva_chapters.md, X10) and web/ai.html.
//
// Prices the XVA of one netting set (the default portfolio of the web front end) with the fused
// Monte Carlo kernels of the library (the CPU build, on every core) on
//   1. a Latin-hypercube "training window" of market states, and
//   2. a synthetic daily market path (calm, a stress episode that leaves the training range,
//      recovery) used as the back test.
// A market state is x = (counterparty CDS multiplier, own CDS multiplier, parallel rate shift,
// EURUSD volatility, funding spread). The same random numbers (seed 2024) are used for every
// state, so each XVA is a smooth deterministic function of x: the learning problem is free of
// Monte Carlo noise in the inputs-to-outputs map, and the surrogate is judged against the engine.
//
//   ./build/xva_ai_dataset > web/js/ai/backtest-data.js
//
// The market path and the sampling use a private generator with portable arithmetic, and the
// kernels draw counter-based normals, so the dataset is the same on every platform.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "ccr/ccr.hpp"

using namespace ccr;

namespace {

constexpr std::size_t kPaths = 2000;
constexpr int kTrain = 160;
constexpr int kDays = 250;
constexpr double kGrossNotional = 10e6 + 6e6 + 5e6 + 4e6;

struct Rng {
  std::mt19937_64 e;
  explicit Rng(std::uint64_t seed) : e(seed) {}
  double uniform() { return ((e() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
  double normal() {
    const double u = uniform(), v = uniform();
    return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * 3.14159265358979323846 * v);
  }
};

using State = std::vector<double>;  // cptyMult, ownMult, rateShift, fxVol, fundingSpread
const double kLow[5] = {0.6, 0.5, -0.010, 0.06, 0.003};
const double kHigh[5] = {3.5, 2.5, 0.020, 0.20, 0.020};

struct Result {
  double cva, dva, fca, fba, mva, kva, total, ms;
};

Result price(const State& x) {
  const double mc = x[0], mo = x[1], dr = x[2], fxVol = x[3], fs = x[4];
  const auto usd = std::make_shared<YieldCurve>(std::vector<double>{0.5, 1.0, 2.0, 5.0, 10.0},
                                                std::vector<double>{0.030 + dr, 0.031 + dr, 0.033 + dr, 0.035 + dr,
                                                                    0.037 + dr});
  const auto eur = std::make_shared<YieldCurve>(std::vector<double>{1.0, 10.0}, std::vector<double>{0.020, 0.025});
  const double recovery = 0.4;
  const std::vector<CdsQuote> base = {{1.0, 0.0070}, {3.0, 0.0095}, {5.0, 0.0120}, {7.0, 0.0130}, {10.0, 0.0140}};
  std::vector<CdsQuote> quotes = base;
  for (auto& q : quotes) q.spread *= mc;
  const auto cptyCurve = std::make_shared<CreditCurve>(bootstrapCreditCurve(*usd, quotes, recovery));
  const CreditCurve ownCurve = bootstrapCreditCurve(*usd, {{5.0, 0.0060 * mo}}, recovery);

  const auto hw = std::make_shared<HullWhite1F>(usd, 0.04, 0.009);
  const auto eurusd = std::make_shared<LognormalAsset>("EURUSD", 1.10, fxVol, eur);
  // CIR++ parameters scale with the spread level so that the Feller condition is preserved.
  const auto intensity = std::make_shared<CirIntensity>("CPTY", cptyCurve, 0.4, 0.02 * mc, 0.10 * std::sqrt(mc),
                                                        0.012 * mc, 4);
  Matrix corr = Matrix::identity(3);
  corr(0, 1) = corr(1, 0) = 0.2;
  const ExposureEngine engine(ScenarioGenerator(hw, {eurusd}, {intensity}, corr), {kPaths, 2024, true},
                              TimeGrid::standardExposureGrid(10.0));

  NettingSet ns;
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("IRS 10y payer", 10e6, 0.0355, 0.0, 10.0, 2, InterestRateSwap::Direction::PayFixed)));
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("IRS 5y receiver", 6e6, 0.0340, 0.0, 5.0, 2, InterestRateSwap::Direction::ReceiveFixed)));
  ns.trades.push_back(std::make_shared<AssetForward>("EURUSD fwd 3y", "EURUSD", 5e6, 1.12, 3.0));
  ns.trades.push_back(std::make_shared<EuropeanOption>("EURUSD call 2y (short)", "EURUSD", OptionType::Call, -4e6,
                                                       1.15, 2.0));

  XvaParameters params;
  params.fundingSpreadBorrow = fs;
  params.fundingSpreadLend = 0.5 * fs;
  params.marginFundingSpread = fs;

  const auto t0 = std::chrono::steady_clock::now();
  gpu::FusedOptions o;
  o.imQuantile = 0.99;  // initial margin: 99% quantile of the value change, scaled to a 10-day MPR
  // (the unconditional proxy, imInnerPaths = 0; the AI page's agent tools ask for the same)
  const auto plan = gpu::compile(engine, ns, o);
  const auto f = gpu::summarise(plan, gpu::runFusedReference(plan));
  const auto& t = f.profile.times;
  std::vector<double> im(t.size(), 0.0);
  for (std::size_t k = 1; k < t.size(); ++k) im[k] = f.valueChangeQuantile[k] * std::sqrt(10.0 / 250.0 / (t[k] - t[k - 1])) * f.meanDeflator[k];
  const Xva v = xva(f.profile, im, *cptyCurve, recovery, ownCurve, recovery, params);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return {v.cva, v.dva, v.fca, v.fba, v.mva, v.kva, v.total(), ms};
}

void printState(const State& x) {
  std::printf("[%.6g,%.6g,%.6g,%.6g,%.6g]", x[0], x[1], x[2], x[3], x[4]);
}
void printResult(const Result& r) {
  std::printf("[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]", r.cva, r.dva, r.fca, r.fba, r.mva, r.kva, r.total);
}

}  // namespace

int main() {
  // 1. Latin hypercube over the training box.
  Rng rng(20240229);
  std::vector<State> train(kTrain, State(5));
  for (int d = 0; d < 5; ++d) {
    std::vector<int> perm(kTrain);
    for (int i = 0; i < kTrain; ++i) perm[i] = i;
    for (int i = kTrain - 1; i > 0; --i) std::swap(perm[i], perm[static_cast<int>(rng.uniform() * (i + 1))]);
    for (int i = 0; i < kTrain; ++i) train[i][d] = kLow[d] + (kHigh[d] - kLow[d]) * (perm[i] + rng.uniform()) / kTrain;
  }

  // 2. Synthetic daily market path: mean-reverting log factors (rate shift: level), a stress
  //    episode on days 150-200 that pushes every driver beyond the training box, then recovery.
  Rng prng(777);
  std::vector<State> path;
  double lmc = std::log(1.1), lmo = std::log(1.0), lfx = std::log(0.10), lfs = std::log(0.008), dr = 0.002;
  for (int t = 0; t < kDays; ++t) {
    double stress = 0.0;  // 0..1 envelope of the stress episode
    if (t >= 150 && t < 200) stress = std::min({1.0, (t - 150) / 15.0, (200 - t) / 15.0});
    const double tmc = std::log(1.1) + stress * (std::log(4.6) - std::log(1.1));
    const double tmo = std::log(1.0) + stress * (std::log(2.2) - std::log(1.0));
    const double tfx = std::log(0.10) + stress * (std::log(0.235) - std::log(0.10));
    const double tfs = std::log(0.008) + stress * (std::log(0.024) - std::log(0.008));
    const double tdr = 0.002 + stress * (-0.0125 - 0.002);
    const double k = 0.08;
    lmc += k * (tmc - lmc) + 0.035 * prng.normal();
    lmo += k * (tmo - lmo) + 0.025 * prng.normal();
    lfx += k * (tfx - lfx) + 0.030 * prng.normal();
    lfs += k * (tfs - lfs) + 0.030 * prng.normal();
    dr += k * (tdr - dr) + 0.0007 * prng.normal();
    path.push_back({std::exp(lmc), std::exp(lmo), dr, std::exp(lfx), std::exp(lfs)});
  }

  std::fprintf(stderr, "pricing %d training states and %d market days (%zu paths each)...\n", kTrain, kDays, kPaths);
  std::printf("// Generated by examples/xva_ai_dataset.cpp (full Monte Carlo on the fused kernels of libccr). Do not edit.\n");
  std::printf("export default {\n  meta: { paths: %zu, seed: 2024, grossNotional: %.0f, trainStates: %d, days: %d,\n", kPaths,
              kGrossNotional, kTrain, kDays);
  std::printf("    low: [%.6g,%.6g,%.6g,%.6g,%.6g], high: [%.6g,%.6g,%.6g,%.6g,%.6g] },\n", kLow[0], kLow[1], kLow[2],
              kLow[3], kLow[4], kHigh[0], kHigh[1], kHigh[2], kHigh[3], kHigh[4]);
  std::printf("  features: ['cptySpreadMultiplier','ownSpreadMultiplier','rateShift','fxVol','fundingSpread'],\n");
  std::printf("  targets: ['cva','dva','fca','fba','mva','kva','total'],\n");

  double totalMs = 0.0;
  int n = 0;
  auto block = [&](const char* name, const std::vector<State>& xs) {
    std::vector<Result> rs;
    for (const auto& x : xs) {
      rs.push_back(price(x));
      totalMs += rs.back().ms;
      ++n;
    }
    std::printf("  %s: {\n    x: [", name);
    for (std::size_t i = 0; i < xs.size(); ++i) {
      std::printf(i ? ",\n      " : "\n      ");
      printState(xs[i]);
    }
    std::printf("],\n    y: [");
    for (std::size_t i = 0; i < rs.size(); ++i) {
      std::printf(i ? ",\n      " : "\n      ");
      printResult(rs[i]);
    }
    std::printf("],\n  },\n");
  };
  block("train", train);
  block("backtest", path);
  std::printf("  nativeMsPerRun: %.1f,\n};\n", totalMs / n);
  return 0;
}
