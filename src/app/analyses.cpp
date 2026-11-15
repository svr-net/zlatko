// The library's entry points for the front end (see ccr/app/value.hpp).
//
// Every entry point takes a "spec" Value (market, models, portfolio, CSA, simulation
// settings) and returns a Value of numbers and typed arrays. Errors are returned as
// { error: "message" } rather than thrown. The WebAssembly binding only converts Values.

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccr/app/value.hpp"
#include "ccr/ccr.hpp"

namespace ccr::app {

using val = Value;

namespace {

// ------------------------------------------------------------------ JS conversion helpers

bool has(const val& o, const char* key) {
  if (o.isUndefined() || o.isNull()) return false;
  const val v = o[key];
  return !(v.isUndefined() || v.isNull());
}

double num(const val& o, const char* key, double fallback) { return has(o, key) ? o[key].as<double>() : fallback; }

std::string str(const val& o, const char* key, const std::string& fallback) {
  return has(o, key) ? o[key].as<std::string>() : fallback;
}

std::vector<double> vec(const val& v) { return v.numbers(); }

val arr(const std::vector<double>& v) { return Value::f64(v); }

val typedArray(const std::vector<float>& v) { return Value::f32(v); }
val typedArray(const std::vector<std::uint32_t>& v) { return Value::u32(v); }

val arrOfArr(const std::vector<std::vector<double>>& rows) {
  val out = val::array();
  for (const auto& r : rows) out.push(arr(r));
  return out;
}

val matrixToJs(const Matrix& m) {
  val o = val::object();
  o.set("rows", static_cast<double>(m.rows()));
  o.set("cols", static_cast<double>(m.cols()));
  std::vector<double> data;
  for (std::size_t r = 0; r < m.rows(); ++r) data.insert(data.end(), m.row(r), m.row(r) + m.cols());
  o.set("data", arr(data));
  return o;
}

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
val guarded(F&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    val o = val::object();
    o.set("error", std::string(e.what()));
    return o;
  } catch (...) {
    val o = val::object();
    o.set("error", std::string("unknown C++ exception"));
    return o;
  }
}

// ------------------------------------------------------------------ spec parsing

std::shared_ptr<YieldCurve> parseCurve(const val& c, double fallbackRate) {
  if (c.isUndefined() || c.isNull()) return std::make_shared<YieldCurve>(YieldCurve::flat(fallbackRate));
  if (has(c, "flat")) return std::make_shared<YieldCurve>(YieldCurve::flat(c["flat"].as<double>()));
  return std::make_shared<YieldCurve>(vec(c["times"]), vec(c["rates"]));
}

std::vector<CdsQuote> parseQuotes(const val& q) {
  std::vector<CdsQuote> out;
  const auto n = q.size();
  for (unsigned i = 0; i < n; ++i) out.push_back({q[i]["maturity"].as<double>(), q[i]["spread"].as<double>()});
  return out;
}

struct TradeInfo {
  std::shared_ptr<const Trade> trade;
  std::string type;
  double reference = std::numeric_limits<double>::quiet_NaN();  // analytic value at t = 0 if known
  std::optional<InterestRateSwap> swap;                         // swap / Bermudan underlying
  std::vector<double> exercises;
  double notional = 0.0;
};

struct Env {
  std::shared_ptr<YieldCurve> domestic;
  std::shared_ptr<HullWhite1F> hw;
  std::vector<std::shared_ptr<const LognormalAsset>> assets;
  std::vector<std::shared_ptr<const CirIntensity>> credits;
  std::vector<std::shared_ptr<const CreditCurve>> creditCurves;
  std::vector<std::vector<CdsQuote>> creditQuotes;
  std::vector<double> recoveries;
  std::vector<std::string> creditNames;
  Matrix correlation;
  SimulationConfig sim;
  TimeGrid baseGrid;
  std::vector<TradeInfo> trades;
  std::optional<CollateralAgreement> csa;
  double pfeQuantile = 0.95;
  std::size_t counterparty = 0;
  std::shared_ptr<const CreditCurve> ownCurve;
  double ownRecovery = 0.4;
  std::vector<CdsQuote> ownQuotes;

  ScenarioGenerator generator() const { return ScenarioGenerator(hw, assets, credits, correlation); }
  NettingSet nettingSet(bool withCsa = true) const {
    NettingSet ns;
    ns.id = "netting set";
    for (const auto& t : trades) ns.trades.push_back(t.trade);
    if (withCsa) ns.collateral = csa;
    return ns;
  }
  const CreditCurve& cptyCurve() const {
    if (creditCurves.empty()) throw std::invalid_argument("spec has no counterparty credit");
    return *creditCurves.at(counterparty);
  }
  double cptyRecovery() const { return recoveries.at(counterparty); }
};

InterestRateSwap parseSwap(const val& t, const YieldCurve& yc, const std::string& id) {
  const auto direction =
      str(t, "direction", "payer") == "payer" ? InterestRateSwap::Direction::PayFixed : InterestRateSwap::Direction::ReceiveFixed;
  auto swap = InterestRateSwap::vanilla(id, num(t, "notional", 1e6), 0.0, num(t, "start", 0.0), num(t, "tenor", 5.0),
                                        static_cast<int>(num(t, "freq", 2)), direction);
  const double rate = has(t, "fixedRate") ? t["fixedRate"].as<double>() : swap.parRate(yc);
  return swap.withFixedRate(rate);
}

TradeInfo parseTrade(const val& t, const Env& env) {
  TradeInfo info;
  info.type = str(t, "type", "swap");
  const std::string id = str(t, "id", info.type);
  const YieldCurve& yc = *env.domestic;
  if (info.type == "swap") {
    auto swap = parseSwap(t, yc, id);
    info.reference = swap.npv(yc);
    info.notional = swap.notional();
    info.swap = swap;
    info.trade = std::make_shared<InterestRateSwap>(swap);
  } else if (info.type == "forward") {
    const std::string asset = str(t, "asset", "FX");
    const double n = num(t, "notional", 1e6), k = num(t, "strike", 1.0), m = num(t, "maturity", 1.0);
    for (const auto& a : env.assets)
      if (a->name() == asset) info.reference = n * (a->spot() * a->carryCurve().discount(m) - k * yc.discount(m));
    info.notional = n;
    info.trade = std::make_shared<AssetForward>(id, asset, n, k, m);
  } else if (info.type == "option") {
    const std::string asset = str(t, "asset", "FX");
    const auto type = str(t, "optionType", "call") == "call" ? OptionType::Call : OptionType::Put;
    const double n = num(t, "notional", 1e6), k = num(t, "strike", 1.0), e = num(t, "expiry", 1.0);
    for (const auto& a : env.assets)
      if (a->name() == asset) {
        const double df = yc.discount(e);
        const double fwd = a->spot() * a->carryCurve().discount(e) / df;
        info.reference = n * blackFormula(type, fwd, k, a->volatility() * std::sqrt(e), df);
      }
    info.notional = n;
    info.trade = std::make_shared<EuropeanOption>(id, asset, type, n, k, e);
  } else if (info.type == "bermudan") {
    auto swap = parseSwap(t, yc, id + "/underlying");
    std::vector<double> exercises;
    if (has(t, "exercises")) {
      exercises = vec(t["exercises"]);
    } else {
      const auto& s = swap.schedule();
      exercises.assign(s.begin(), s.end() - 1);
    }
    const auto settlement = str(t, "settlement", "physical") == "physical" ? Settlement::Physical : Settlement::Cash;
    info.swap = swap;
    info.exercises = exercises;
    info.notional = swap.notional();
    info.trade = std::make_shared<BermudanSwaption>(id, swap, exercises, settlement, num(t, "position", 1.0),
                                                    static_cast<std::size_t>(num(t, "degree", 3)));
  } else {
    throw std::invalid_argument("unknown trade type '" + info.type + "'");
  }
  return info;
}

CollateralAgreement parseCsa(const val& c) {
  CollateralAgreement csa;
  csa.thresholdCounterparty = num(c, "thresholdCounterparty", 0.0);
  csa.thresholdOwn = has(c, "thresholdOwn") ? c["thresholdOwn"].as<double>() : std::numeric_limits<double>::infinity();
  if (csa.thresholdOwn < 0.0) csa.thresholdOwn = std::numeric_limits<double>::infinity();
  csa.minimumTransferAmount = num(c, "mta", 0.0);
  csa.independentAmount = num(c, "independentAmount", 0.0);
  csa.marginPeriodOfRisk = num(c, "mpr", 10.0 / 250.0);
  return csa;
}

Env parseEnv(const val& spec) {
  Env env;
  env.domestic = parseCurve(spec["domestic"], 0.03);
  const val hw = spec["hw"];
  env.hw = std::make_shared<HullWhite1F>(env.domestic, num(hw, "a", 0.05), num(hw, "sigma", 0.01));

  if (has(spec, "assets")) {
    const val a = spec["assets"];
    for (unsigned i = 0; i < a.size(); ++i)
      env.assets.push_back(std::make_shared<LognormalAsset>(str(a[i], "name", "FX"), num(a[i], "spot", 1.0),
                                                            num(a[i], "vol", 0.1), parseCurve(a[i]["carry"], 0.01)));
  }
  if (has(spec, "credits")) {
    const val c = spec["credits"];
    for (unsigned i = 0; i < c.size(); ++i) {
      const double recovery = num(c[i], "recovery", 0.4);
      const auto quotes = parseQuotes(c[i]["quotes"]);
      auto curve = std::make_shared<CreditCurve>(bootstrapCreditCurve(*env.domestic, quotes, recovery));
      const val cir = c[i]["cir"];
      const std::string name = str(c[i], "name", "CPTY");
      env.credits.push_back(std::make_shared<CirIntensity>(name, curve, num(cir, "kappa", 0.4), num(cir, "theta", 0.02),
                                                           num(cir, "xi", 0.08), num(cir, "y0", 0.01),
                                                           static_cast<std::size_t>(num(cir, "substeps", 4))));
      env.creditCurves.push_back(curve);
      env.creditQuotes.push_back(quotes);
      env.recoveries.push_back(recovery);
      env.creditNames.push_back(name);
    }
  }
  const std::size_t nF = 1 + env.assets.size() + env.credits.size();
  env.correlation = Matrix::identity(nF);
  if (has(spec, "correlation")) {
    const val c = spec["correlation"];
    if (c.size() != nF) throw std::invalid_argument("correlation matrix has wrong dimension");
    for (unsigned i = 0; i < nF; ++i)
      for (unsigned j = 0; j < nF; ++j) env.correlation(i, j) = c[i][j].as<double>();
  }
  const val sim = spec["sim"];
  env.sim.numPaths = static_cast<std::size_t>(num(sim, "numPaths", 2000));
  env.sim.seed = static_cast<std::uint64_t>(num(sim, "seed", 42));
  env.sim.antithetic = has(sim, "antithetic") ? sim["antithetic"].as<bool>() : true;

  const val grid = spec["grid"];
  if (has(grid, "times")) {
    env.baseGrid = TimeGrid(vec(grid["times"]));
  } else if (str(grid, "type", "standard") == "uniform") {
    env.baseGrid = TimeGrid::uniform(num(grid, "horizon", 5.0), static_cast<std::size_t>(num(grid, "steps", 20)));
  } else {
    env.baseGrid = TimeGrid::standardExposureGrid(num(grid, "horizon", 5.0));
  }

  if (has(spec, "csa") && !(has(spec["csa"], "enabled") && !spec["csa"]["enabled"].as<bool>())) env.csa = parseCsa(spec["csa"]);
  env.pfeQuantile = num(spec, "pfeQuantile", 0.95);
  env.counterparty = static_cast<std::size_t>(num(spec, "counterparty", 0));
  if (has(spec, "own")) {
    env.ownRecovery = num(spec["own"], "recovery", 0.4);
    env.ownQuotes = parseQuotes(spec["own"]["quotes"]);
    env.ownCurve = std::make_shared<CreditCurve>(bootstrapCreditCurve(*env.domestic, env.ownQuotes, env.ownRecovery));
  }
  if (has(spec, "trades")) {
    const val t = spec["trades"];
    for (unsigned i = 0; i < t.size(); ++i) env.trades.push_back(parseTrade(t[i], env));
  }
  return env;
}

// ------------------------------------------------------------------ shared result builders

val profileToJs(const ExposureProfile& p) {
  val o = val::object();
  o.set("times", arr(p.times));
  o.set("expectedValue", arr(p.expectedValue));
  o.set("ee", arr(p.expectedExposure));
  o.set("ene", arr(p.expectedNegativeExposure));
  o.set("pfe", arr(p.potentialFutureExposure));
  o.set("discountedEe", arr(p.discountedExpectedExposure));
  o.set("discountedEne", arr(p.discountedExpectedNegativeExposure));
  o.set("effectiveEe", arr(p.effectiveExpectedExposure()));
  o.set("pfeQuantile", p.pfeQuantile);
  o.set("epe1y", p.expectedPositiveExposure(1.0));
  o.set("eepe1y", p.effectiveExpectedPositiveExposure(1.0));
  o.set("epeLife", p.expectedPositiveExposure(p.times.empty() ? 0.0 : p.times.back()));
  o.set("maxPfe", p.maxPotentialFutureExposure());
  return o;
}

// Percentile bands over time: result[k][j] = quantile qs[k] at date j.
// ------------------------------------------------------------------ entry points

val version() {
  val o = val::object();
  o.set("library", std::string("zlatko ccr ") + CCR_VERSION);  // from the CMake project VERSION
  o.set("wasm", true);
  return o;
}

// Core numerics: correlation (Cholesky), regression, time grids, normal distribution, Brent.
val coreDemo(val spec) {
  return guarded([&] {
    val out = val::object();
    // Correlation via Cholesky and the realised correlation of simulated normals.
    const Env env = parseEnv(spec);
    const Matrix l = cholesky(env.correlation);
    const std::size_t n = env.correlation.rows();
    const auto samples = static_cast<std::size_t>(num(spec, "numSamples", 20000));
    NormalGenerator rng(env.sim.seed);
    std::vector<double> z(n), w(n), sum(n * n, 0.0);
    std::vector<std::vector<double>> scatter(2);
    for (std::size_t s = 0; s < samples; ++s) {
      rng.fill(z.data(), n);
      for (std::size_t i = 0; i < n; ++i) {
        w[i] = 0.0;
        for (std::size_t k = 0; k <= i; ++k) w[i] += l(i, k) * z[k];
      }
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) sum[i * n + j] += w[i] * w[j];
      if (s < 1500 && n >= 2) {
        scatter[0].push_back(w[0]);
        scatter[1].push_back(w[n - 1]);
      }
    }
    Matrix realised(n, n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        realised(i, j) = sum[i * n + j] / std::sqrt(sum[i * n + i] * sum[j * n + j]);
    out.set("cholesky", matrixToJs(l));
    out.set("correlation", matrixToJs(env.correlation));
    out.set("realisedCorrelation", matrixToJs(realised));
    out.set("scatter", arrOfArr(scatter));

    // Polynomial regression (the AMC workhorse) on noisy data.
    const val reg = spec["regression"];
    const auto nPoints = static_cast<std::size_t>(num(reg, "points", 400));
    const double noise = num(reg, "noise", 0.3);
    std::vector<double> x(nPoints), y(nPoints);
    for (std::size_t i = 0; i < nPoints; ++i) {
      x[i] = rng.next();
      y[i] = std::max(x[i] - 0.3, 0.0) + 0.2 * std::sin(2.0 * x[i]) + noise * rng.next();
    }
    std::vector<double> grid;
    for (int i = 0; i <= 100; ++i) grid.push_back(-3.0 + 0.06 * i);
    val fits = val::array();
    for (int degree : {1, 2, 3, 5}) {
      PolynomialRegression r(static_cast<std::size_t>(degree));
      r.fit(x, y);
      double ssr = 0.0, sst = 0.0;
      const double my = mean(y);
      for (std::size_t i = 0; i < nPoints; ++i) {
        ssr += std::pow(y[i] - r.predict(x[i]), 2);
        sst += std::pow(y[i] - my, 2);
      }
      val f = val::object();
      f.set("degree", degree);
      f.set("fitted", arr(r.predict(grid)));
      f.set("rSquared", 1.0 - ssr / sst);
      fits.push(f);
    }
    val regression = val::object();
    regression.set("x", arr(x));
    regression.set("y", arr(y));
    regression.set("grid", arr(grid));
    regression.set("fits", fits);
    out.set("regression", regression);

    // Time grids.
    const double mpr = num(spec, "mpr", 10.0 / 250.0);
    const auto stepsOf = [](const std::vector<double>& t) {  // (t_k, t_k - t_{k-1}) for k >= 1
      val o = val::object();
      std::vector<double> at(t.begin() + 1, t.end()), dt;
      for (std::size_t k = 1; k < t.size(); ++k) dt.push_back(t[k] - t[k - 1]);
      o.set("times", arr(at));
      o.set("steps", arr(dt));
      o.set("dates", static_cast<double>(t.size()));
      return o;
    };
    out.set("standardGrid", stepsOf(env.baseGrid.times()));
    out.set("laggedGrid", stepsOf(env.baseGrid.withLaggedTimes(mpr).times()));
    out.set("mprDays", mpr * 250.0);

    // Normal distribution, Black formula, Brent implied volatility.
    std::vector<double> xs, cdf, pdf;
    for (int i = 0; i <= 120; ++i) {
      xs.push_back(-4.0 + i / 15.0);
      cdf.push_back(normalCdf(xs.back()));
      pdf.push_back(normalPdf(xs.back()));
    }
    out.set("normalX", arr(xs));
    out.set("normalCdf", arr(cdf));
    out.set("normalPdf", arr(pdf));
    const double trueVol = num(spec, "impliedVolTarget", 0.23);
    const double price = blackFormula(OptionType::Call, 100.0, 105.0, trueVol * std::sqrt(2.0), 0.95);
    int evaluations = 0;
    const double implied = solveBrent(
        [&](double v) {
          ++evaluations;
          return blackFormula(OptionType::Call, 100.0, 105.0, v * std::sqrt(2.0), 0.95) - price;
        },
        1e-4, 3.0, 1e-14);
    val brent = val::object();
    brent.set("price", price);
    brent.set("trueVol", trueVol);
    brent.set("impliedVol", implied);
    brent.set("evaluations", evaluations);
    brent.set("recovered", std::fabs(implied - trueVol) < 1e-10);
    out.set("brent", brent);

    // Empirical quantiles of a normal sample against the exact ones.
    std::vector<double> sample(5000);
    rng.fill(sample.data(), sample.size());
    val qs = val::array();
    for (double p : {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99}) {
      val q = val::object();
      q.set("p", p);
      q.set("empirical", quantile(sample, p));
      q.set("exact", solveBrent([p](double v) { return normalCdf(v) - p; }, -10.0, 10.0));
      qs.push(q);
    }
    out.set("quantiles", qs);
    return out;
  });
}

// Market data: yield curve, credit curve bootstrap, CDS pricing.
val marketDemo(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const double horizon = num(spec, "horizon", 10.0);
    std::vector<double> t, zero, df, fwd, fwd3m;
    for (int i = 0; i <= 200; ++i) {
      t.push_back(horizon * i / 200.0);
      zero.push_back(env.domestic->zeroRate(t.back()));
      df.push_back(env.domestic->discount(t.back()));
      fwd.push_back(env.domestic->instantaneousForward(t.back()));
      fwd3m.push_back(env.domestic->forwardRate(t.back(), t.back() + 0.25));
    }
    val out = val::object();
    val yc = val::object();
    yc.set("times", arr(t));
    yc.set("zero", arr(zero));
    yc.set("discount", arr(df));
    yc.set("instantaneousForward", arr(fwd));
    yc.set("forward3m", arr(fwd3m));
    yc.set("pillars", arr(env.domestic->times()));
    yc.set("pillarRates", arr(env.domestic->zeroRates()));
    out.set("yieldCurve", yc);

    val credits = val::array();
    for (std::size_t c = 0; c < env.creditCurves.size(); ++c) {
      const CreditCurve& cc = *env.creditCurves[c];
      std::vector<double> surv, haz, pd;
      for (double x : t) {
        surv.push_back(cc.survival(x));
        haz.push_back(cc.hazard(x));
        pd.push_back(1.0 - cc.survival(x));
      }
      val quotes = val::array();
      for (const auto& q : env.creditQuotes[c]) {
        const CreditDefaultSwap cds(q.maturity, q.spread, 1.0, env.recoveries[c]);
        val r = val::object();
        r.set("maturity", q.maturity);
        r.set("quote", q.spread);
        r.set("parSpread", cds.parSpread(*env.domestic, cc));
        r.set("riskyAnnuity", cds.riskyAnnuity(*env.domestic, cc));
        r.set("protectionLeg", cds.protectionLeg(*env.domestic, cc));
        r.set("npv", cds.npv(*env.domestic, cc));
        r.set("survival", cc.survival(q.maturity));
        r.set("creditTriangle", cc.hazard(q.maturity) * (1.0 - env.recoveries[c]));
        quotes.push(r);
      }
      val o = val::object();
      o.set("name", env.creditNames[c]);
      o.set("recovery", env.recoveries[c]);
      o.set("pillars", arr(cc.pillars()));
      o.set("hazards", arr(cc.hazards()));
      o.set("survival", arr(surv));
      o.set("hazard", arr(haz));
      o.set("defaultProbability", arr(pd));
      o.set("quotes", quotes);
      credits.push(o);
    }
    out.set("credits", credits);
    return out;
  });
}

// Risk-factor simulation: fan charts and martingale checks.
XvaParameters parseXva(const val& spec) {
  XvaParameters p;
  if (!has(spec, "xva")) return p;
  const val x = spec["xva"];
  p.fundingSpreadBorrow = num(x, "fundingSpreadBorrow", p.fundingSpreadBorrow);
  p.fundingSpreadLend = num(x, "fundingSpreadLend", p.fundingSpreadLend);
  p.marginFundingSpread = num(x, "marginFundingSpread", p.marginFundingSpread);
  p.hurdleRate = num(x, "hurdleRate", p.hurdleRate);
  p.capitalRatio = num(x, "capitalRatio", p.capitalRatio);
  p.riskWeight = num(x, "riskWeight", p.riskWeight);
  p.alpha = num(x, "alpha", p.alpha);
  return p;
}

// Our own credit curve for XVA: without a spec.own section we are treated as default-free.
const CreditCurve& ownCurveOrRiskless(const Env& env) {
  static const CreditCurve riskless = CreditCurve::flat(0.0);
  return env.ownCurve ? *env.ownCurve : riskless;
}

val xvaToJs(const Xva& x) {
  val o = val::object();
  o.set("cva", x.cva);
  o.set("dva", x.dva);
  o.set("fca", x.fca);
  o.set("fba", x.fba);
  o.set("fva", x.fva());
  o.set("mva", x.mva);
  o.set("kva", x.kva);
  o.set("total", x.total());
  return o;
}

// ------------------------------------------------------------------ Monte Carlo analyses
//
// Each Monte Carlo analysis is a list of exposure jobs (one netting set under one model
// set-up each) and a combine step. A job runs on the CPU (ExposureEngine) or on the WebGPU
// fused kernels (gpuJobs -> GPU -> gpuAnalyse). The combine step is the same C++ code for
// both engines, so the results have the same shape and the JavaScript layer does no maths.

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Job {
  std::string label;
  std::optional<ExposureEngine> engine;  // the model set-up (compiled for the fused kernels); empty if invalid
  NettingSet nettingSet;
  gpu::FusedOptions options;             // the views this job asks of the kernels
  std::string error;
};

struct JobResult {
  ExposureProfile profile;
  double pathwiseCva = kNaN;                     // stochastic intensity, NaN without credits
  std::vector<double> conditionalDiscountedEe;   // E[D V+ | default in period k], NaN at k = 0
  std::vector<double> meanSurvival;              // E[Q(t)] of the simulated intensity
  std::size_t simulationDates = 0;
  std::size_t numPaths = 0;
  double elapsedMs = 0.0;
  std::string error;
  std::vector<double> meanDeflator;              // E[D]
  std::vector<double> valueChangeQuantile;       // quantile of |V(t_k) - V(t_{k-1})| (initial margin), 0 at k = 0
  std::vector<gpu::FusedResult::TradeStats> trades;  // Euler, incremental and standalone statistics per trade
  gpu::FusedResult fused;                        // the path views
};

Job makeJob(std::string label, const Env& env, const std::function<ScenarioGenerator()>& generator,
            const SimulationConfig& sim, const std::optional<CollateralAgreement>& csa,
            const std::optional<NettingSet>& nettingSet = std::nullopt) {
  Job job;
  job.label = std::move(label);
  job.nettingSet = nettingSet ? *nettingSet : env.nettingSet(false);
  job.nettingSet.collateral = csa;
  try {
    job.engine.emplace(generator(), sim, env.baseGrid);
  } catch (const std::exception& e) {
    job.error = e.what();
  }
  return job;
}

struct CsaVariant {
  std::string name;
  std::optional<CollateralAgreement> csa;
};

// The CSA comparison of the collateral page: the CSA in the spec (enabled or not) and variants.
std::vector<CsaVariant> csaVariants(const val& spec) {
  const CollateralAgreement c = has(spec, "csa") ? parseCsa(spec["csa"]) : CollateralAgreement{};
  CollateralAgreement zero = c, mpr20 = c, oneWay = c;
  zero.thresholdCounterparty = zero.thresholdOwn = zero.minimumTransferAmount = 0.0;
  mpr20.marginPeriodOfRisk = 20.0 / 250.0;
  oneWay.thresholdOwn = std::numeric_limits<double>::infinity();
  return {{"no CSA", std::nullopt}, {"specified CSA", c}, {"zero threshold & MTA", zero}, {"MPR 20 days", mpr20},
          {"one-way (we never post)", oneWay}};
}

std::vector<double> wwrRhos(const val& spec) {
  return has(spec, "rhos") ? vec(spec["rhos"]) : std::vector<double>{-0.6, -0.3, 0.0, 0.3, 0.6};
}

std::size_t wwrDriver(const val& spec, const Env& env) {
  return static_cast<std::size_t>(num(spec, "wwrFactor", env.assets.empty() ? 0 : 1));
}

std::vector<double> hedgingBumps(const val& spec) {
  return has(spec, "bumps") ? vec(spec["bumps"]) : std::vector<double>{0.04, 0.02, 0.01, 0.005};
}

Value withMarketState(const Value& spec);  // see the AI chapter below

// spec with its marketState replaced by spec.shockState, applied.
Value shockedSpec(const Value& spec) {
  Value s = Value::object();
  for (const auto& [k, v] : spec.items())
    if (k != "marketState" && k != "shockState") s.set(k, v);
  s.set("marketState", spec["shockState"]);
  return withMarketState(s);
}

struct BermudanSpec {
  InterestRateSwap swap;
  std::vector<double> exercises;
  std::size_t degree = 3;
};

// The AMC page's Bermudan: spec.bermudan with its underlying, exercise dates and degree.
BermudanSpec parseBermudan(const val& spec, const Env& env) {
  const val t = spec["bermudan"];
  BermudanSpec b{parseSwap(t, *env.domestic, "underlying"), {}, std::min<std::size_t>(static_cast<std::size_t>(num(t, "degree", 3)), gpu::kMaxAmcDegree)};
  if (has(t, "exercises")) {
    b.exercises = vec(t["exercises"]);
  } else {
    const auto& sch = b.swap.schedule();
    b.exercises.assign(sch.begin(), sch.end() - 1);
  }
  return b;
}

std::vector<Job> analysisJobs(const val& spec, const Env& env, const std::string& analysis) {
  const ScenarioGenerator gen = env.generator();
  auto fixed = [&](const ScenarioGenerator& g) { return [g] { return g; }; };
  std::vector<Job> jobs;
  const auto samples = static_cast<std::size_t>(num(spec, "samplePaths", 0));
  if (analysis == "exposure") {
    // The exposure page: profile, sample paths, gross exposure (netting benefit), per-trade
    // profiles and the exposure histogram.
    jobs.push_back(makeJob("base", env, fixed(gen), env.sim, env.csa));
    auto& o = jobs.back().options;
    o.samples = samples;
    o.histogramTime = num(spec, "histogramTime", 1.0);
    o.gross = o.tradeViews = env.trades.size() <= gpu::kMaxTradeGroups;
    o.collateralBands = env.csa.has_value();
  } else if (analysis == "cva" || analysis == "validation" || analysis == "xva") {
    jobs.push_back(makeJob("base", env, fixed(gen), env.sim, env.csa));
    if (analysis == "xva") jobs.back().options.perTrade = true, jobs.back().options.imQuantile = num(spec, "imQuantile", 0.99);
  } else if (analysis == "xvaCompare") {
    // The base market and the shocked one (spec.shockState) as two jobs of one analysis.
    jobs.push_back(makeJob("base", env, fixed(gen), env.sim, env.csa));
    jobs.back().options.perTrade = true, jobs.back().options.imQuantile = num(spec, "imQuantile", 0.99);
    const Env shocked = parseEnv(shockedSpec(spec));
    jobs.push_back(makeJob("shocked", shocked, [g = shocked.generator()] { return g; }, shocked.sim, shocked.csa,
                           shocked.nettingSet(false)));
    jobs.back().options = jobs.front().options;
  } else if (analysis == "instruments") {
    jobs.push_back(makeJob("trades", env, fixed(gen), env.sim, std::nullopt));
    jobs.back().options.tradeViews = true;
    jobs.back().options.samples = samples;
  } else if (analysis == "models") {
    // Risk factors only: an empty netting set and the factor views.
    jobs.push_back(makeJob("factors", env, fixed(gen), env.sim, std::nullopt, NettingSet{}));
    auto& o = jobs.back().options;
    o.factorViews = true;
    o.samples = samples;
    o.bondMaturity = env.baseGrid.horizon() + num(spec, "bondTenor", 5.0);
  } else if (analysis == "amc") {
    // A Bermudan with physical and with cash settlement in one netting set (exercise views),
    // then the price by regression degree.
    const auto berm = parseBermudan(spec, env);
    NettingSet both;
    for (const auto settle : {Settlement::Physical, Settlement::Cash})
      both.trades.push_back(std::make_shared<BermudanSwaption>(settle == Settlement::Physical ? "physical" : "cash", berm.swap,
                                                               berm.exercises, settle, 1.0, berm.degree));
    jobs.push_back(makeJob("settlement", env, fixed(gen), env.sim, std::nullopt, both));
    auto& o = jobs.back().options;
    o.tradeViews = o.exerciseViews = true;
    o.samples = samples;
    for (std::size_t d = 1; d <= gpu::kMaxAmcDegree; ++d) {
      NettingSet one;
      one.trades.push_back(std::make_shared<BermudanSwaption>("degree", berm.swap, berm.exercises, Settlement::Physical, 1.0, d));
      jobs.push_back(makeJob("degree " + std::to_string(d), env, fixed(gen), env.sim, std::nullopt, one));
    }
  } else if (analysis == "allocation") {
    jobs.push_back(makeJob("base", env, fixed(gen), env.sim, std::nullopt));  // allocation of the uncollateralised set
    jobs.back().options.perTrade = true;
  } else if (analysis == "collateral") {
    for (const auto& v : csaVariants(spec)) jobs.push_back(makeJob(v.name, env, fixed(gen), env.sim, v.csa));
    // The specified CSA (job 1) also returns sample paths of value and collateral and the collateral bands.
    jobs.at(1).options.samples = samples;
    jobs.at(1).options.collateralBands = true;
  } else if (analysis == "wrongWayRisk") {
    if (env.credits.empty()) throw std::invalid_argument("wrong-way risk needs a stochastic counterparty intensity");
    const std::size_t driver = wwrDriver(spec, env), creditFactor = 1 + env.assets.size() + env.counterparty;
    for (double rho : wwrRhos(spec)) {
      Matrix corr = env.correlation;
      corr(driver, creditFactor) = corr(creditFactor, driver) = rho;
      jobs.push_back(makeJob("rho " + std::to_string(rho), env,
                             [&env, corr] { return ScenarioGenerator(env.hw, env.assets, env.credits, corr); }, env.sim, env.csa));
    }
  } else if (analysis == "hedging") {
    // Base run; for each asset and bump size the up/down runs with common random numbers and
    // with independent seeds; finally the +/- 1bp parallel rate shifts (common random numbers).
    jobs.push_back(makeJob("base", env, fixed(gen), env.sim, env.csa));
    SimulationConfig seed1 = env.sim, seed2 = env.sim;
    seed1.seed += 1;
    seed2.seed += 2;
    for (std::size_t a = 0; a < env.assets.size(); ++a)
      for (double h : hedgingBumps(spec)) {
        auto bumped = [&](double b) {
          return gen.withAsset(a, std::make_shared<LognormalAsset>(env.assets[a]->withSpot(env.assets[a]->spot() + b)));
        };
        jobs.push_back(makeJob("up", env, fixed(bumped(h)), env.sim, env.csa));
        jobs.push_back(makeJob("down", env, fixed(bumped(-h)), env.sim, env.csa));
        jobs.push_back(makeJob("up, seed + 1", env, fixed(bumped(h)), seed1, env.csa));
        jobs.push_back(makeJob("down, seed + 2", env, fixed(bumped(-h)), seed2, env.csa));
      }
    for (double bp : {1.0, -1.0}) {
      auto curve = std::make_shared<YieldCurve>(env.domestic->shifted(bp * 1e-4));
      jobs.push_back(makeJob("rates shifted", env,
                             fixed(gen.withRateModel(std::make_shared<HullWhite1F>(curve, env.hw->meanReversion(), env.hw->volatility()))),
                             env.sim, env.csa));
    }
  } else {
    throw std::invalid_argument("unknown analysis '" + analysis + "'");
  }
  return jobs;
}

JobResult fromFused(const gpu::FusedPlan& plan, const gpu::FusedOutput& output, const Env& env) {
  const gpu::FusedResult f = gpu::summarise(plan, output);
  JobResult r;
  r.profile = f.profile;
  r.simulationDates = plan.numDates();
  r.numPaths = plan.numPaths;
  if (!env.credits.empty()) {
    r.pathwiseCva = f.pathwiseCva(env.cptyRecovery());
    r.conditionalDiscountedEe = f.conditionalDiscountedEe;
    r.meanSurvival = f.meanSurvival;
  }
  r.meanDeflator = f.meanDeflator;
  r.valueChangeQuantile = f.valueChangeQuantile;
  r.trades = f.trades;
  r.fused = f;
  return r;
}

// Closed-form CVA analytics on an exposure profile: unilateral CVA, its term structure and
// accumulation, bilateral CVA/DVA, running spread and recovery sensitivity.
val cvaAnalytics(const val& spec, const Env& env, const ExposureProfile& profile) {
  const CreditCurve& cc = env.cptyCurve();
  const double rec = env.cptyRecovery();
  const auto& times = profile.times;

  val out = val::object();
  out.set("profile", profileToJs(profile));
  out.set("cva", unilateralCva(profile, cc, rec));
  const auto terms = cvaTermStructure(times, profile.discountedExpectedExposure, cc, rec);
  std::vector<double> cumulative;
  double running = 0.0;
  for (double t : terms) cumulative.push_back(running += t);
  out.set("termStructure", arr(terms));
  out.set("cumulativeCva", arr(cumulative));
  std::vector<double> surv, own;
  for (double t : times) {
    surv.push_back(cc.survival(t));
    if (env.ownCurve) own.push_back(env.ownCurve->survival(t));
  }
  out.set("counterpartySurvival", arr(surv));
  if (env.ownCurve) {
    const auto b = bilateralCva(profile, cc, rec, *env.ownCurve, env.ownRecovery);
    val bo = val::object();
    bo.set("cva", b.cva);
    bo.set("dva", b.dva);
    bo.set("total", b.total());
    out.set("bilateral", bo);
    out.set("ownSurvival", arr(own));
  }
  double maturity = 0.0, notional = 0.0;
  for (const auto& t : env.trades) {
    maturity = std::max(maturity, t.trade->maturity());
    notional = std::max(notional, std::fabs(t.notional));
  }
  notional = num(spec, "spreadNotional", notional);
  if (maturity > 0.0 && notional > 0.0) {
    out.set("runningSpread", cvaRunningSpread(unilateralCva(profile, cc, rec), *env.domestic, cc, maturity, notional));
    out.set("spreadNotional", notional);
    out.set("spreadMaturity", maturity);
  }
  // CVA as a function of recovery (linear in LGD under independence).
  std::vector<double> recs, cvas;
  for (int i = 0; i <= 10; ++i) {
    recs.push_back(i / 10.0);
    cvas.push_back(unilateralCva(profile, cc, recs.back()));
  }
  out.set("recoveryGrid", arr(recs));
  out.set("cvaByRecovery", arr(cvas));
  return out;
}

// Credit hedging of CVA for an exposure profile: bucketed CS01, the CDS hedge Jacobian and
// notionals, and the P&L of the hedged and unhedged CVA under random spread scenarios.
val creditHedges(const val& spec, const Env& env, const ExposureProfile& profile) {
  const auto& quotes = env.creditQuotes.at(env.counterparty);
  const double rec = env.cptyRecovery();
  auto cvaOf = [&](const CreditCurve& c) { return unilateralCva(profile, c, rec); };

  val out = val::object();
  out.set("cva", cvaOf(env.cptyCurve()));
  const auto cs01 = creditSpreadSensitivities(cvaOf, *env.domestic, quotes, rec);
  const Matrix jac = cdsHedgeJacobian(*env.domestic, quotes, rec);
  const auto notionals = cdsHedgeNotionals(cs01, jac);
  std::vector<double> mats;
  for (const auto& q : quotes) mats.push_back(q.maturity);
  double parallel = 0.0;
  for (double c : cs01) parallel += c;
  out.set("maturities", arr(mats));
  out.set("cs01", arr(cs01));
  out.set("parallelCs01", parallel);
  out.set("jacobian", matrixToJs(jac));
  out.set("hedgeNotionals", arr(notionals));

  // Hedge effectiveness under random spread scenarios.
  NormalGenerator rng(env.sim.seed + 1000);
  const CreditCurve base = env.cptyCurve();
  const double baseCva = cvaOf(base);
  std::vector<double> unhedged, hedged;
  const double shockBp = num(spec, "spreadShockBp", 10.0);
  for (int n = 0; n < 60; ++n) {
    auto moved = quotes;
    const double common = rng.next();
    for (auto& q : moved) q.spread = std::max(q.spread + 1e-4 * shockBp * (0.7 * common + 0.3 * rng.next()), 1e-5);
    const CreditCurve curve = bootstrapCreditCurve(*env.domestic, moved, rec);
    const double dCva = cvaOf(curve) - baseCva;
    double dHedge = 0.0;
    for (std::size_t k = 0; k < quotes.size(); ++k) {
      const CreditDefaultSwap cds(quotes[k].maturity, quotes[k].spread, notionals[k], rec);
      dHedge += cds.npv(*env.domestic, curve) - cds.npv(*env.domestic, base);
    }
    unhedged.push_back(-dCva);  // our P&L: CVA is a liability
    hedged.push_back(-dCva + dHedge);
  }
  auto rms = [](const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s / static_cast<double>(v.size()));
  };
  out.set("pnlUnhedged", arr(unhedged));
  out.set("pnlHedged", arr(hedged));
  out.set("pnlVolRatio", rms(hedged) / rms(unhedged));

  // The same hedge for the whole XVA (CVA, DVA, FVA and KVA; MVA needs paths, see the xva page).
  const XvaParameters xvaParams = parseXva(spec);
  val xv = val::object();
  const Xva total = xva(profile, {}, env.cptyCurve(), rec, ownCurveOrRiskless(env), env.ownRecovery, xvaParams);
  xv.set("xva", xvaToJs(total));
  if (!env.ownQuotes.empty()) {
    const XvaSensitivities s = xvaSensitivities(profile, {}, *env.domestic, quotes, rec, env.ownQuotes, env.ownRecovery, xvaParams);
    double parallelXva = 0.0, parallelOwn = 0.0;
    for (double c : s.counterpartyCs01) parallelXva += c;
    for (double c : s.ownCs01) parallelOwn += c;
    xv.set("counterpartyCs01", arr(s.counterpartyCs01));
    xv.set("counterpartyHedgeNotionals", arr(cdsHedgeNotionals(s.counterpartyCs01, jac)));
    xv.set("parallelCounterpartyCs01", parallelXva);
    xv.set("ownCs01", arr(s.ownCs01));
    xv.set("parallelOwnCs01", parallelOwn);
    xv.set("fundingSpread01", s.fundingSpread01);
  }
  out.set("xva", xv);
  return out;
}

val csaToJs(const std::optional<CollateralAgreement>& c) {
  val o = val::object();
  o.set("enabled", c.has_value());
  if (c) {
    o.set("thresholdCounterparty", c->thresholdCounterparty);
    o.set("thresholdOwn", std::isfinite(c->thresholdOwn) ? c->thresholdOwn : -1.0);
    o.set("oneWay", !std::isfinite(c->thresholdOwn));
    o.set("mta", c->minimumTransferAmount);
    o.set("mprDays", c->marginPeriodOfRisk * 250.0);
  }
  return o;
}

val firstError(const std::vector<JobResult>& results) {
  for (const auto& r : results)
    if (!r.error.empty()) return val(r.error);
  return val::undefined();
}

// XVA from the exposure statistics of one job (CPU paths or fused kernels alike): the six
// adjustments, their build-up, the funding and hurdle sweeps, the credit sensitivities and the
// Euler allocation of CVA, FCA and KVA to trades.
val xvaResult(const val& spec, const Env& env, const JobResult& r) {
  const CreditCurve& cc = env.cptyCurve();
  const CreditCurve& own = ownCurveOrRiskless(env);
  const double rec = env.cptyRecovery();
  const XvaParameters params = parseXva(spec);
  const auto& times = r.profile.times;
  // Initial margin: the |dV| quantile scaled to the margin period of risk, discounted with E[D].
  const double mpr = num(spec, "imMprDays", 10.0) / 250.0;
  std::vector<double> im(times.size(), 0.0);
  for (std::size_t k = 1; k < times.size() && k < r.valueChangeQuantile.size(); ++k)
    im[k] = r.valueChangeQuantile[k] * std::sqrt(mpr / (times[k] - times[k - 1])) * r.meanDeflator[k];
  const Xva total = xva(r.profile, im, cc, rec, own, env.ownRecovery, params);

  val out = val::object();
  out.set("profile", profileToJs(r.profile));
  out.set("initialMargin", arr(im));
  out.set("xva", xvaToJs(total));
  out.set("identityHolds", std::fabs(total.cva - total.dva + total.fca - total.fba + total.mva + total.kva - total.total()) <=
                               1e-6 * std::max(1.0, std::fabs(total.total())));
  out.set("numPaths", static_cast<double>(r.numPaths));

  // Cumulative cost of each adjustment up to every date (each is an integral over time).
  auto part = [&](double Xva::*field) {
    std::vector<double> c(times.size(), 0.0);
    for (std::size_t k = 1; k < times.size(); ++k) {
      ExposureProfile head = r.profile;
      head.times.resize(k + 1);
      head.discountedExpectedExposure.resize(k + 1);
      head.discountedExpectedNegativeExposure.resize(k + 1);
      const std::vector<double> imHead(im.begin(), im.begin() + static_cast<std::ptrdiff_t>(k + 1));
      c[k] = xva(head, imHead, cc, rec, own, env.ownRecovery, params).*field;
    }
    return c;
  };
  out.set("cumulativeCva", arr(part(&Xva::cva)));
  out.set("cumulativeFca", arr(part(&Xva::fca)));
  out.set("cumulativeMva", arr(part(&Xva::mva)));
  out.set("cumulativeKva", arr(part(&Xva::kva)));

  // Funding spread and hurdle rate sweeps: FVA is linear in the spread, KVA in the hurdle rate.
  std::vector<double> spreads, fvaBySpread, hurdles, kvaByHurdle;
  for (int i = 0; i <= 10; ++i) {
    XvaParameters p = params;
    p.fundingSpreadBorrow = p.fundingSpreadLend = 0.0005 * i;
    spreads.push_back(p.fundingSpreadBorrow);
    fvaBySpread.push_back(xva(r.profile, {}, cc, rec, own, env.ownRecovery, p).fva());
    p = params;
    p.hurdleRate = 0.02 * i;
    hurdles.push_back(p.hurdleRate);
    kvaByHurdle.push_back(xva(r.profile, {}, cc, rec, own, env.ownRecovery, p).kva);
  }
  out.set("spreadGrid", arr(spreads));
  out.set("fvaBySpread", arr(fvaBySpread));
  out.set("hurdleGrid", arr(hurdles));
  out.set("kvaByHurdle", arr(kvaByHurdle));

  if (!env.ownQuotes.empty()) {
    const XvaSensitivities s = xvaSensitivities(r.profile, im, *env.domestic, env.creditQuotes.at(env.counterparty), rec,
                                                env.ownQuotes, env.ownRecovery, params);
    std::vector<double> mats;
    for (const auto& q : env.creditQuotes.at(env.counterparty)) mats.push_back(q.maturity);
    val sens = val::object();
    sens.set("maturities", arr(mats));
    sens.set("counterpartyCs01", arr(s.counterpartyCs01));
    sens.set("ownCs01", arr(s.ownCs01));
    sens.set("fundingSpread01", s.fundingSpread01);
    out.set("sensitivities", sens);
  }

  // Marginal allocation of the EE-driven adjustments (CVA, FCA, KVA) to trades.
  std::vector<std::vector<double>> marginal;
  for (const auto& t : r.trades) marginal.push_back(t.marginalDiscounted);
  const auto parts = allocateExposureXva(times, marginal, cc, rec, own, params);
  val trades = val::array();
  for (std::size_t i = 0; i < parts.size(); ++i) {
    val o = val::object();
    o.set("id", env.trades[i].trade->id());
    o.set("cva", parts[i].cva);
    o.set("fca", parts[i].fca);
    o.set("kva", parts[i].kva);
    trades.push(o);
  }
  out.set("trades", trades);
  return out;
}

// Marginal (Euler), incremental and standalone allocation of exposure and CVA to trades.
val allocationResult(const val& spec, const Env& env, const JobResult& r) {
  const auto& times = r.profile.times;
  const CreditCurve& cc = env.cptyCurve();
  const double rec = env.cptyRecovery();
  const XvaParameters xvaParams = parseXva(spec);
  std::vector<std::vector<double>> marginal;
  for (const auto& t : r.trades) marginal.push_back(t.marginalDiscounted);
  const auto marginalXva = allocateExposureXva(times, marginal, cc, rec, ownCurveOrRiskless(env), xvaParams);
  const ExposureXva totalXva = exposureXva(times, r.profile.discountedExpectedExposure, cc, rec, ownCurveOrRiskless(env), xvaParams);

  val out = val::object();
  out.set("times", arr(times));
  out.set("totalCva", unilateralCva(r.profile, cc, rec));
  out.set("totalFca", totalXva.fca);
  out.set("totalKva", totalXva.kva);
  out.set("totalEe", arr(r.profile.expectedExposure));
  out.set("numPaths", static_cast<double>(r.numPaths));
  const double totalCva = unilateralCva(r.profile, cc, rec);
  double marginalSum = 0.0, incrementalSum = 0.0, standaloneSum = 0.0;
  val trades = val::array();
  for (std::size_t i = 0; i < r.trades.size(); ++i) {
    const auto& t = r.trades[i];
    const double marginalCva = unilateralCva(times, t.marginalDiscounted, cc, rec);
    const double incrementalCva = unilateralCva(times, t.incrementalDiscounted, cc, rec);
    const double standaloneCva = unilateralCva(times, t.standaloneDiscounted, cc, rec);
    marginalSum += marginalCva, incrementalSum += incrementalCva, standaloneSum += standaloneCva;
    val o = val::object();
    o.set("id", env.trades[i].trade->id());
    o.set("marginalEe", arr(t.marginal));
    o.set("marginalCva", marginalCva);
    o.set("marginalShare", totalCva != 0.0 ? marginalCva / totalCva : kNaN);
    o.set("marginalFca", marginalXva[i].fca);
    o.set("marginalKva", marginalXva[i].kva);
    o.set("incrementalCva", incrementalCva);
    o.set("standaloneCva", standaloneCva);
    o.set("standaloneEe", arr(t.standalone));
    trades.push(o);
  }
  out.set("trades", trades);
  out.set("marginalCvaSum", marginalSum);
  out.set("incrementalCvaSum", incrementalSum);
  out.set("standaloneCvaSum", standaloneSum);
  out.set("nettingSaving", standaloneSum != 0.0 ? 1.0 - totalCva / standaloneSum : kNaN);
  // The Euler contributions add up exactly in exact arithmetic; the f32 kernels to about 1e-6.
  out.set("addsUp", std::fabs(marginalSum - totalCva) < 1e-4 * std::fabs(totalCva));
  return out;
}

// The exposure distribution on one date: bin edges and the share of paths in each bin.
val histogramToJs(const gpu::FusedResult& f) {
  val h = val::object();
  h.set("time", f.histogramTime);
  h.set("edges", arr(f.histogramEdges));
  h.set("shares", arr(f.histogramCounts));
  return h;
}

// A Monte Carlo estimate against its exact value: relative error and pass at `tolerance`.
val checkToJs(const std::string& name, double mc, double exact, double tolerance) {
  val o = val::object();
  const double rel = std::fabs(mc / exact - 1.0);
  o.set("name", name);
  o.set("value", mc);
  o.set("exact", exact);
  o.set("relError", rel);
  o.set("tolerance", tolerance);
  o.set("pass", rel < tolerance);
  return o;
}

double bestEuropean(const Env& env, const InterestRateSwap& u, const std::vector<double>& exercises, std::vector<double>* each = nullptr) {
  double best = 0.0;
  const bool payer = u.direction() == InterestRateSwap::Direction::PayFixed;
  for (double e : exercises) {
    const auto tail = u.tail(e);
    const double v = u.notional() * env.hw->europeanSwaption(e, tail.paymentTimes(), tail.accruals(), u.fixedRate(), payer);
    if (each) each->push_back(v);
    best = std::max(best, v);
  }
  return best;
}

// The instruments page: every trade's mark-to-market on the scenarios (fused per-trade rows):
// sample paths, bands, its standalone profile, the deflated mean and the check at t = 0.
val instrumentsResult(const Env& env, const JobResult& r) {
  if (!r.error.empty()) throw std::invalid_argument(r.error);
  const auto& f = r.fused;
  val out = val::object();
  out.set("times", arr(r.profile.times));
  out.set("numPaths", static_cast<double>(r.numPaths));
  val trades = val::array();
  for (std::size_t i = 0; i < f.tradeViews.size(); ++i) {
    const auto& t = env.trades[i];
    const auto& v = f.tradeViews[i];
    const double value0 = v.profile.expectedValue.at(0);
    val o = val::object();
    o.set("id", t.trade->id());
    o.set("type", t.type);
    o.set("maturity", t.trade->maturity());
    o.set("value0", value0);
    o.set("samples", arrOfArr(v.samples));
    o.set("bands", arrOfArr(v.bands));
    o.set("meanDeflatedValue", arr(v.meanDeflated));
    o.set("profile", profileToJs(v.profile));
    o.set("eePeak", *std::max_element(v.profile.expectedExposure.begin(), v.profile.expectedExposure.end()));
    if (t.type == "bermudan") {
      const double lower = bestEuropean(env, *t.swap, t.exercises);
      o.set("benchmark", lower);
      o.set("benchmarkLabel", "max European");
      o.set("pass", value0 >= 0.97 * lower);
      o.set("checkLabel", "≥ max European");
    } else {
      // The kernels price in f32: t = 0 against the analytic value to 1e-5 of the notional.
      o.set("benchmark", t.reference);
      o.set("benchmarkLabel", "analytic");
      o.set("pass", std::fabs(value0 - t.reference) <= 1e-5 * std::max({1.0, std::fabs(t.reference), std::fabs(t.notional)}));
      o.set("checkLabel", "matches");
    }
    trades.push(o);
  }
  out.set("trades", trades);
  return out;
}

// The models page: risk-factor fans, sample paths and martingale tests from the probe rows.
val modelsResult(const val& spec, const Env& env, const JobResult& r) {
  if (!r.error.empty()) throw std::invalid_argument(r.error);
  const auto& F = *r.fused.factors;
  const auto& times = r.profile.times;
  const std::size_t nR = times.size(), last = nR - 1;
  auto shifted = [](std::vector<std::vector<double>> rows, const std::vector<double>& shift) {
    for (auto& row : rows)
      for (std::size_t k = 0; k < row.size(); ++k) row[k] += shift[k];
    return rows;
  };
  std::vector<double> phi(nR), curveD(nR);
  for (std::size_t k = 0; k < nR; ++k) phi[k] = env.hw->phi(times[k]), curveD[k] = env.domestic->discount(times[k]);
  const double bondMaturity = env.baseGrid.horizon() + num(spec, "bondTenor", 5.0);
  val out = val::object();
  out.set("times", arr(times));
  out.set("numPaths", static_cast<double>(r.numPaths));
  val checks = val::array();
  val rates = val::object();
  rates.set("samples", arrOfArr(shifted(F.rateSamples, phi)));
  rates.set("bands", arrOfArr(shifted(F.rateBands, phi)));
  std::vector<double> meanR(nR);
  for (std::size_t k = 0; k < nR; ++k) meanR[k] = F.rateMean[k] + phi[k];
  rates.set("meanShortRate", arr(meanR));
  rates.set("meanDeflator", arr(F.deflatorMean));
  rates.set("curveDiscount", arr(curveD));
  rates.set("bondMaturity", bondMaturity);
  rates.set("meanDeflatedBond", arr(F.bondMean));
  rates.set("bondTarget", env.domestic->discount(bondMaturity));
  out.set("rates", rates);
  checks.push(checkToJs("E[D(0,T)] vs P(0,T)", F.deflatorMean[last], curveD[last], 0.01));
  checks.push(checkToJs("E[D(0,t)P(t,T)] vs P(0,T)", F.bondMean[last], env.domestic->discount(bondMaturity), 0.01));
  val assets = val::array();
  for (std::size_t a = 0; a < F.assets.size(); ++a) {
    const auto& m = *env.assets[a];
    std::vector<double> target(nR);
    for (std::size_t k = 0; k < nR; ++k) target[k] = m.spot() * m.carryCurve().discount(times[k]);
    val o = val::object();
    o.set("name", m.name());
    o.set("samples", arrOfArr(F.assets[a].samples));
    o.set("bands", arrOfArr(F.assets[a].bands));
    o.set("meanDeflated", arr(F.assets[a].deflatedMean));
    o.set("target", arr(target));
    assets.push(o);
    checks.push(checkToJs("E[D·S] vs S0·Q (" + m.name() + ")", F.assets[a].deflatedMean[last], target[last], 0.01));
  }
  out.set("assets", assets);
  val credits = val::array();
  for (std::size_t c = 0; c < F.credits.size(); ++c) {
    const CirIntensity& model = *env.credits[c];
    std::vector<double> psi(nR), marketQ(nR);
    for (std::size_t k = 0; k < nR; ++k) {
      const double h = 1e-4, t = std::max(times[k], h);
      psi[k] = (model.integratedShift(t + h) - model.integratedShift(t - h)) / (2 * h);  // lambda = y + psi
      marketQ[k] = model.marketCurve().survival(times[k]);
    }
    val o = val::object();
    o.set("name", model.name());
    o.set("intensitySamples", arrOfArr(shifted(F.credits[c].intensitySamples, psi)));
    o.set("intensityBands", arrOfArr(shifted(F.credits[c].intensityBands, psi)));
    o.set("survivalSamples", arrOfArr(F.credits[c].survivalSamples));
    o.set("meanSurvival", arr(F.credits[c].survivalMean));
    o.set("marketSurvival", arr(marketQ));
    o.set("shift", arr(psi));
    credits.push(o);
    checks.push(checkToJs("E[Q] vs market (" + model.name() + ")", F.credits[c].survivalMean[last], marketQ[last], 0.01));
  }
  out.set("credits", credits);
  out.set("checks", checks);
  if (F.incrementCorrelation.rows() > 0) {
    out.set("realisedCorrelation", matrixToJs(F.incrementCorrelation));
    out.set("targetCorrelation", matrixToJs(env.correlation));
  }
  return out;
}

// The AMC page: the Bermudan with physical and cash settlement, its exercise policy, the
// European bounds and the price by regression degree, all from the fused kernels.
val amcResult(const val& spec, const Env& env, const std::vector<JobResult>& results) {
  if (val e = firstError(results); !e.isUndefined()) throw std::invalid_argument(e.as<std::string>());
  const auto berm = parseBermudan(spec, env);
  const auto& f = results.at(0).fused;
  const auto& physical = f.tradeViews.at(0), &cash = f.tradeViews.at(1);
  const double price = physical.profile.expectedValue.at(0);
  std::vector<double> european;
  const double best = bestEuropean(env, berm.swap, berm.exercises, &european);
  const auto& ex = f.exercise.at(0);
  val out = val::object();
  out.set("times", arr(results.at(0).profile.times));
  out.set("price", price);
  out.set("exercises", arr(berm.exercises));
  out.set("fixedRate", berm.swap.fixedRate());
  out.set("parRate", berm.swap.parRate(*env.domestic));
  out.set("european", arr(european));
  out.set("bestEuropean", best);
  out.set("switchValue", price - best);
  out.set("switchPremium", best > 0.0 ? price / best - 1.0 : kNaN);
  out.set("exerciseDistribution", arr(ex.distribution));
  out.set("neverExercised", ex.distribution.back());
  std::vector<double> sampleTimes;
  for (double k : ex.sampleIndex) sampleTimes.push_back(k < 0.0 ? -1.0 : berm.exercises.at(static_cast<std::size_t>(k)));
  out.set("sampleExerciseTimes", arr(sampleTimes));
  out.set("physicalSamples", arrOfArr(physical.samples));
  out.set("cashSamples", arrOfArr(cash.samples));
  out.set("physicalProfile", profileToJs(physical.profile));
  out.set("cashProfile", profileToJs(cash.profile));
  val degrees = val::array();
  const double reference = results.back().profile.expectedValue.at(0);  // the highest degree
  for (std::size_t d = 1; d < results.size(); ++d) {
    val o = val::object();
    const double p = results[d].profile.expectedValue.at(0);
    o.set("degree", static_cast<double>(d));
    o.set("price", p);
    o.set("vsHighest", p / reference - 1.0);
    degrees.push(o);
  }
  out.set("degreeSensitivity", degrees);
  return out;
}

// The combine step of every analysis: identical for CPU and GPU job results.
val combine(const val& spec, const Env& env, const std::string& analysis, const std::vector<Job>& jobs,
            const std::vector<JobResult>& results) {
  const CreditCurve* cc = env.creditCurves.empty() ? nullptr : &env.cptyCurve();
  const double rec = env.creditCurves.empty() ? 0.4 : env.cptyRecovery();
  val out = val::object();
  if (analysis == "exposure") {
    const JobResult& r = results.at(0);
    if (!r.error.empty()) throw std::invalid_argument(r.error);
    const auto& f = r.fused;
    out.set("profile", profileToJs(r.profile));
    out.set("numPaths", static_cast<double>(r.numPaths));
    out.set("simulationDates", static_cast<double>(r.simulationDates));
    out.set("reportingDates", static_cast<double>(r.profile.times.size()));
    out.set("collateralised", env.csa.has_value());
    out.set("exposureSamples", arrOfArr(f.exposureSamples));
    out.set("nettedSamples", arrOfArr(f.nettedSamples));
    if (env.csa) {
      out.set("collateralSamples", arrOfArr(f.collateralSamples));
      out.set("collateralBands", arrOfArr(f.collateralBands));
    }
    if (f.grossProfile) {
      out.set("grossProfile", profileToJs(*f.grossProfile));
      const double gross = f.grossProfile->effectiveExpectedPositiveExposure(1.0);
      out.set("nettingBenefit", gross > 0.0 ? 1.0 - r.profile.effectiveExpectedPositiveExposure(1.0) / gross : 0.0);
    }
    val trades = val::array();
    for (std::size_t i = 0; i < f.tradeViews.size(); ++i) {
      val o = val::object();
      o.set("id", env.trades[i].trade->id());
      o.set("value0", f.tradeViews[i].profile.expectedValue.at(0));
      o.set("profile", profileToJs(f.tradeViews[i].profile));
      trades.push(o);
    }
    out.set("trades", trades);
    out.set("histogram", histogramToJs(f));
  } else if (analysis == "cva") {
    const JobResult& r = results.at(0);
    if (!r.error.empty()) throw std::invalid_argument(r.error);
    out = cvaAnalytics(spec, env, r.profile);
    const double cva = unilateralCva(r.profile, *cc, rec);
    out.set("pathwiseCva", r.pathwiseCva);
    out.set("pathwiseExcess", r.pathwiseCva / cva - 1.0);
  } else if (analysis == "collateral") {
    const auto variants = csaVariants(spec);
    if (val e = firstError(results); !e.isUndefined()) throw std::invalid_argument(e.as<std::string>());
    const double noCsa = results.at(0).profile.effectiveExpectedPositiveExposure(1.0);
    val list = val::array();
    for (std::size_t i = 0; i < results.size(); ++i) {
      val o = val::object();
      o.set("name", variants[i].name);
      o.set("csa", csaToJs(variants[i].csa));
      o.set("profile", profileToJs(results[i].profile));
      o.set("eepeVsNoCsa", noCsa > 0.0 ? results[i].profile.effectiveExpectedPositiveExposure(1.0) / noCsa : kNaN);
      list.push(o);
    }
    out.set("variants", list);
    const auto& f = results.at(1).fused;  // the specified CSA
    out.set("nettedSamples", arrOfArr(f.nettedSamples));
    out.set("collateralBands", arrOfArr(f.collateralBands));
  } else if (analysis == "wrongWayRisk") {
    const auto rhos = wwrRhos(spec);
    val list = val::array();
    for (std::size_t i = 0; i < results.size(); ++i) {
      const JobResult& r = results[i];
      val o = val::object();
      o.set("rho", rhos[i]);
      if (!r.error.empty()) {
        o.set("error", r.error);
      } else {
        const double independent = unilateralCva(r.profile, *cc, rec);
        o.set("pathwiseCva", r.pathwiseCva);
        o.set("independentCva", independent);
        o.set("multiplier", r.pathwiseCva / independent);
        o.set("times", arr(r.profile.times));
        o.set("discountedEe", arr(r.profile.discountedExpectedExposure));
        o.set("conditionalDiscountedEe", arr(r.conditionalDiscountedEe));
      }
      list.push(o);
    }
    out.set("results", list);
    out.set("driver", static_cast<double>(wwrDriver(spec, env)));
  } else if (analysis == "hedging") {
    if (val e = firstError(results); !e.isUndefined()) throw std::invalid_argument(e.as<std::string>());
    out = creditHedges(spec, env, results.at(0).profile);
    auto cvaAt = [&](std::size_t i) { return unilateralCva(results.at(i).profile, *cc, rec); };
    const auto bumps = hedgingBumps(spec);
    std::size_t i = 1;
    val deltas = val::array();
    for (std::size_t a = 0; a < env.assets.size(); ++a) {
      std::vector<double> crn, independent;
      for (double h : bumps) {
        crn.push_back((cvaAt(i) - cvaAt(i + 1)) / (2 * h));
        independent.push_back((cvaAt(i + 2) - cvaAt(i + 3)) / (2 * h));
        i += 4;
      }
      const double carry = env.assets[a]->carryCurve().discount(env.baseGrid.horizon());
      const double hedgeDelta = crn.size() >= 2 ? crn[crn.size() - 2] : crn.back();
      val o = val::object();
      o.set("name", env.assets[a]->name());
      o.set("bumps", arr(bumps));
      o.set("deltaCrn", arr(crn));
      o.set("deltaIndependent", arr(independent));
      o.set("hedgeBump", crn.size() >= 2 ? bumps[bumps.size() - 2] : bumps.back());
      o.set("hedgeDelta", hedgeDelta);
      o.set("hedgeUnits", hedgeDelta / carry);  // forward units at the horizon (+ buy, - sell)
      deltas.push(o);
    }
    out.set("assetDeltas", deltas);
    out.set("cvaDv01", 0.5 * (cvaAt(i) - cvaAt(i + 1)));
  } else if (analysis == "xvaCompare") {
    if (val e = firstError(results); !e.isUndefined()) throw std::invalid_argument(e.as<std::string>());
    const Value shocked = shockedSpec(spec);
    const Env shockedEnv = parseEnv(shocked);
    const val base = xvaResult(spec, env, results.at(0)), bumped = xvaResult(shocked, shockedEnv, results.at(1));
    val change = val::object();
    for (const char* k : {"cva", "dva", "fca", "fba", "fva", "mva", "kva", "total"})
      change.set(k, bumped["xva"][k].as<double>() - base["xva"][k].as<double>());
    out.set("base", base);
    out.set("shocked", bumped);
    out.set("change", change);
  } else if (analysis == "instruments") {
    out = instrumentsResult(env, results.at(0));
  } else if (analysis == "models") {
    out = modelsResult(spec, env, results.at(0));
  } else if (analysis == "amc") {
    out = amcResult(spec, env, results);
  } else if (analysis == "xva") {
    const JobResult& r = results.at(0);
    if (!r.error.empty()) throw std::invalid_argument(r.error);
    out = xvaResult(spec, env, r);
  } else if (analysis == "allocation") {
    const JobResult& r = results.at(0);
    if (!r.error.empty()) throw std::invalid_argument(r.error);
    out = allocationResult(spec, env, r);
  } else {
    throw std::invalid_argument("unknown analysis '" + analysis + "'");
  }
  (void)jobs;
  return out;
}

// ------------------------------------------------------------------ WebGPU entry points
//
// gpuKernels() returns the WGSL sources. gpuJobs(spec) compiles the jobs of spec.analysis
// into GPU plans: ready-made buffers, dispatch sizes and buffer sizes, so the host only
// uploads, dispatches and reads back. gpuAnalyse(spec) takes the read-backs
// (spec.gpuOutputs[i] = { sums, pfe }, null for an invalid job) and returns the result in
// the shape of the CPU analysis. cpuKernels(spec) runs the same kernels on the CPU instead.

Value withMarketState(const Value& spec);  // see the AI chapter below

val gpuKernels(val) {
  val o = val::object();
  o.set("fusedExposure", gpu::fusedExposureKernel(false));
  o.set("fusedExposureTrades", gpu::fusedExposureKernel(true));
  o.set("reducePartials", gpu::reducePartialsKernel());
  o.set("pfeQuantile", gpu::pfeQuantileKernel());
  return o;
}

val planToJs(const gpu::FusedPlan& p, const std::string& label) {
  val o = val::object();
  o.set("label", label);
  o.set("numPaths", static_cast<double>(p.numPaths));
  o.set("numDates", static_cast<double>(p.numDates()));
  o.set("numTerms", static_cast<double>(p.numTerms()));
  o.set("perTrade", p.perTrade());  // needs the per-trade variant of the fused kernel
  o.set("header", typedArray(p.header));
  o.set("indices", typedArray(p.indices));
  o.set("params", typedArray(p.params));
  o.set("steps", typedArray(p.steps));
  o.set("terms", typedArray(p.terms.empty() ? std::vector<float>{0.0f} : p.terms));
  o.set("requests", typedArray(p.requests.empty() ? std::vector<float>(gpu::kRequestStride, 0.0f) : p.requests));
  val dispatch = val::object();
  dispatch.set("fused", static_cast<double>(p.fusedDispatch()));
  dispatch.set("reduce", static_cast<double>(p.reduceDispatch()));
  dispatch.set("stats", static_cast<double>(p.pfeDispatch()));
  o.set("dispatch", dispatch);
  val bytes = val::object();
  bytes.set("cube", static_cast<double>(p.cubeBytes()));
  bytes.set("vals", static_cast<double>(p.valsBytes()));
  bytes.set("expo", static_cast<double>(p.expoBytes()));
  bytes.set("partials", static_cast<double>(p.partialsBytes()));
  bytes.set("sums", static_cast<double>(p.sumsBytes()));
  bytes.set("values", static_cast<double>(p.pfeBytes()));
  o.set("bytes", bytes);
  return o;
}

std::string analysisOf(const val& spec) { return str(spec, "analysis", "exposure"); }

// The plans of an analysis, or the reason the kernels cannot run it.
struct GpuJobs {
  std::vector<Job> jobs;
  std::vector<std::optional<gpu::FusedPlan>> plans;
  std::string unsupported;
};

GpuJobs compileJobs(const val& spec, const Env& env) {
  GpuJobs g;
  g.jobs = analysisJobs(spec, env, analysisOf(spec) == "validation" ? "cva" : analysisOf(spec));
  for (const auto& job : g.jobs) {
    if (!job.engine) {
      g.plans.emplace_back();
      continue;
    }
    gpu::FusedOptions o = job.options;
    o.pfeQuantile = env.pfeQuantile;
    o.counterparty = env.counterparty;
    const bool perTrade = o.perTrade || o.gross || o.tradeViews;
    if (const std::string why = gpu::limitation(*job.engine, job.nettingSet, perTrade); !why.empty()) {
      g.unsupported = why;
      return g;
    }
    g.plans.emplace_back(gpu::compile(*job.engine, job.nettingSet, o));
  }
  return g;
}

val gpuJobs(val input) {
  const val spec = withMarketState(input);
  return guarded([&] {
    const Env env = parseEnv(spec);
    const double t0 = nowMs();
    const GpuJobs g = compileJobs(spec, env);
    val out = val::object();
    out.set("compileMs", nowMs() - t0);
    if (!g.unsupported.empty()) {
      out.set("unsupported", g.unsupported);
      return out;
    }
    val plans = val::array();
    for (std::size_t i = 0; i < g.jobs.size(); ++i)
      plans.push(g.plans[i] ? planToJs(*g.plans[i], g.jobs[i].label) : val::null());
    out.set("plans", plans);
    return out;
  });
}

// GPU validation: the fused kernels against the CPU library on the same specification.
// WebGPU validation: the GPU read-back against the same plans on the CPU fused kernels (the same
// algorithm in C++: they agree to f32 rounding), plus the exact checks: the t = 0 value against
// the analytic trade values and the simulated CIR++ survival against the market curve.
val validation(const val& spec, const Env& env, const JobResult& g, const JobResult& c) {
  (void)spec;
  const CreditCurve& cc = env.cptyCurve();
  const double rec = env.cptyRecovery();
  auto maxRelDiff = [](const std::vector<double>& a, const std::vector<double>& b) {
    double scale = 1e-12, diff = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      scale = std::max(scale, std::fabs(b[i]));
      diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff / scale;
  };
  const double gpuCva = unilateralCva(g.profile, cc, rec), cpuCva = unilateralCva(c.profile, cc, rec);
  std::vector<double> market;
  for (double t : g.profile.times) market.push_back(cc.survival(t));
  double grossNotional = 0.0, analytic0 = 0.0;
  bool exact0 = true;
  for (const auto& t : env.trades) {
    grossNotional += std::fabs(t.notional);
    if (std::isfinite(t.reference)) analytic0 += t.reference;
    else exact0 = false;
  }
  val checks = val::object();
  checks.set("tol", 1e-3);  // the same kernels: f32 rounding and GPU transcendental functions only
  checks.set("t0Diff", exact0 ? std::fabs(g.profile.expectedValue[0] - analytic0) / std::max(grossNotional, 1.0) : 0.0);
  checks.set("eeDiff", maxRelDiff(g.profile.expectedExposure, c.profile.expectedExposure));
  checks.set("cvaDiff", std::fabs(gpuCva - cpuCva) / std::max(std::fabs(cpuCva), 1e-12));
  checks.set("survDiff", g.meanSurvival.empty() ? 0.0 : maxRelDiff(g.meanSurvival, market));
  checks.set("t0Pass", checks["t0Diff"].as<double>() < 1e-5);
  checks.set("eePass", checks["eeDiff"].as<double>() < 1e-3);
  checks.set("cvaPass", checks["cvaDiff"].as<double>() < 1e-3);
  checks.set("survPass", checks["survDiff"].as<double>() < 0.01);
  auto side = [&](const JobResult& r, double cva) {
    val o = val::object();
    o.set("profile", profileToJs(r.profile));
    o.set("cva", cva);
    o.set("pathwiseCva", r.pathwiseCva);
    o.set("meanSurvival", arr(r.meanSurvival));
    o.set("elapsedMs", r.elapsedMs);
    o.set("simulationDates", static_cast<double>(r.simulationDates));
    return o;
  };
  val out = val::object();
  out.set("gpu", side(g, gpuCva));
  out.set("reference", side(c, cpuCva));
  out.set("marketSurvival", arr(market));
  out.set("checks", checks);
  out.set("numPaths", static_cast<double>(env.sim.numPaths));
  return out;
}

val analyseFused(const val& spec, const std::function<std::optional<gpu::FusedOutput>(std::size_t, const gpu::FusedPlan&)>& output) {
  const Env env = parseEnv(spec);
  const GpuJobs g = compileJobs(spec, env);
  if (!g.unsupported.empty()) {  // e.g. a Bermudan: the caller falls back to ExposureEngine
    val out = val::object();
    out.set("unsupported", g.unsupported);
    return out;
  }
  std::vector<JobResult> results;
  for (std::size_t i = 0; i < g.jobs.size(); ++i) {
    if (!g.plans[i]) {
      JobResult r;
      r.error = g.jobs[i].error;
      results.push_back(r);
      continue;
    }
    const auto o = output(i, *g.plans[i]);
    if (!o) throw std::invalid_argument("missing GPU output for job " + std::to_string(i));
    results.push_back(fromFused(*g.plans[i], *o, env));
  }
  const std::string analysis = analysisOf(spec);
  if (analysis == "validation") {
    const double t0 = nowMs();
    JobResult cpu = fromFused(*g.plans.at(0), gpu::runFusedReference(*g.plans.at(0)), env);
    cpu.elapsedMs = nowMs() - t0;
    return validation(spec, env, results.at(0), cpu);
  }
  return combine(spec, env, analysis, g.jobs, results);
}

val gpuAnalyse(val input) {
  const val spec = withMarketState(input);
  return guarded([&] {
    const val outputs = spec["gpuOutputs"];
    return analyseFused(spec, [&](std::size_t i, const gpu::FusedPlan&) -> std::optional<gpu::FusedOutput> {
      if (i >= outputs.size() || outputs[i].isNull() || outputs[i].isUndefined()) return std::nullopt;
      gpu::FusedOutput o;
      o.sums = (outputs[i]["sums"]).floats();
      o.values = (outputs[i]["values"]).floats();
      return o;
    });
  });
}

// The CPU backend of the fused pipeline: the same kernels in C++ (runFusedReference), parallel
// on the lock-free SpinPool where the build has threads (natively; one thread in this
// single-threaded WebAssembly build). One call per analysis, the same result shape as gpuAnalyse.
val cpuKernels(val input) {
  const val spec = withMarketState(input);
  return guarded([&] {
    double kernelMs = 0.0;  // the kernels only (validation also runs ExposureEngine in this call)
    val out = analyseFused(spec, [&](std::size_t, const gpu::FusedPlan& plan) -> std::optional<gpu::FusedOutput> {
      const double t0 = nowMs();
      gpu::FusedOutput o = gpu::runFusedReference(plan);
      kernelMs += nowMs() - t0;
      return o;
    });
    if (out["unsupported"].isUndefined()) {
      out.set("kernelMs", kernelMs);
      out.set("threads", static_cast<double>(defaultPool().concurrency()));
    }
    return out;
  });
}

// ------------------------------------------------------------------ AI chapter (X10)

// A deep copy of a value (Values share their contents, like JavaScript objects).
Value deepCopy(const Value& v) {
  switch (v.type()) {
    case Value::Type::Object: {
      Value o = Value::object();
      for (const auto& [k, e] : v.items()) o.set(k, deepCopy(e));
      return o;
    }
    case Value::Type::Array: {
      Value a = Value::array();
      for (const auto& e : v.elements()) a.push(deepCopy(e));
      return a;
    }
    default: return v;  // scalars and typed arrays are not modified below
  }
}

// The market state of the XVA agent tools and the surrogates, applied to a specification:
// counterparty CDS quotes and CIR parameters scaled, own quotes scaled, a parallel rate shift,
// the EURUSD volatility and the funding spreads (as price() of examples/xva_ai_dataset.cpp).
Value withMarketState(const Value& spec) {
  const Value& st = spec["marketState"];
  if (st.isUndefined() || st.isNull()) return spec;
  const double mc = num(st, "cptySpreadMultiplier", 1.0), mo = num(st, "ownSpreadMultiplier", 1.0);
  const double dr = num(st, "rateShift", 0.0), vol = num(st, "fxVol", 0.1), fs = num(st, "fundingSpread", 0.01);
  Value out = deepCopy(spec);
  auto scaled = [](const Value& quotes, double m) {
    Value q = Value::array();
    for (const auto& e : quotes.elements()) {
      Value o = deepCopy(e);
      o.set("spread", e["spread"].as<double>() * m);
      q.push(o);
    }
    return q;
  };
  Value domestic = deepCopy(spec["domestic"]);
  Value rates = Value::array();
  for (double r : vec(spec["domestic"]["rates"])) rates.push(r + dr);
  domestic.set("rates", rates);
  out.set("domestic", domestic);
  if (spec["assets"].size() > 0) {
    Value list = Value::array();
    Value first = deepCopy(spec["assets"][0]);
    first.set("vol", vol);
    list.push(first);
    for (std::size_t i = 1; i < spec["assets"].size(); ++i) list.push(spec["assets"][i]);
    out.set("assets", list);
  }
  if (spec["credits"].size() > 0) {
    Value credits = Value::array();
    Value c = deepCopy(spec["credits"][0]);
    c.set("quotes", scaled(spec["credits"][0]["quotes"], mc));
    const Value& cir = spec["credits"][0]["cir"];
    Value ncir = deepCopy(cir);
    ncir.set("theta", num(cir, "theta", 0.02) * mc);
    ncir.set("xi", num(cir, "xi", 0.1) * std::sqrt(mc));
    ncir.set("y0", num(cir, "y0", 0.012) * mc);
    c.set("cir", ncir);
    credits.push(c);
    for (std::size_t i = 1; i < spec["credits"].size(); ++i) credits.push(spec["credits"][i]);
    out.set("credits", credits);
  }
  if (has(spec, "own")) {
    Value own = deepCopy(spec["own"]);
    own.set("quotes", scaled(spec["own"]["quotes"], mo));
    out.set("own", own);
  }
  Value x = has(spec, "xva") ? deepCopy(spec["xva"]) : Value::object();
  x.set("fundingSpreadBorrow", fs);
  x.set("fundingSpreadLend", 0.5 * fs);
  x.set("marginFundingSpread", fs);
  out.set("xva", x);
  return out;
}

ai::Rows rowsOf(const Value& v) {
  ai::Rows rows;
  for (const auto& r : v.elements()) rows.push_back(r.numbers());
  return rows;
}

Value statsToJs(const ai::ErrorStats& e) {
  Value o = Value::object();
  o.set("mae", e.mae), o.set("rmse", e.rmse), o.set("bias", e.bias), o.set("max", e.max);
  o.set("p95", e.p95), o.set("maeBp", e.maeBp), o.set("mape", e.mape);
  return o;
}

Value scoreToJs(const ai::Score& s) {
  Value o = Value::object();
  o.set("predTotal", arr(s.predTotal));
  o.set("errorTotal", arr(s.errorTotal));
  o.set("total", statsToJs(s.total));
  o.set("inRange", s.inRange ? statsToJs(*s.inRange) : Value::null());
  o.set("outRange", s.outRange ? statsToJs(*s.outRange) : Value::null());
  Value comp = Value::object();
  for (std::size_t k = 0; k < ai::kTargets; ++k) comp.set(ai::kTargetNames[k], statsToJs(s.components[k]));
  o.set("components", comp);
  Value pnl = Value::object();
  pnl.set("effectiveness", s.pnl.effectiveness), pnl.set("signHit", s.pnl.signHit);
  pnl.set("residualSd", s.pnl.residualSd), pnl.set("actualSd", s.pnl.actualSd);
  o.set("pnl", pnl);
  return o;
}

// The surrogate back test of the AI page on the committed dataset (spec.dataset, the content of
// web/js/ai/backtest-data.js), fitted and scored in C++ (the MLP on the thread pool natively).
Value surrogateBacktest(Value spec) {
  return guarded([&] {
    const Value& d = spec["dataset"];
    ai::Dataset data;
    data.trainX = rowsOf(d["train"]["x"]);
    data.trainY = rowsOf(d["train"]["y"]);
    data.testX = rowsOf(d["backtest"]["x"]);
    data.testY = rowsOf(d["backtest"]["y"]);
    data.low = d["meta"]["low"].numbers();
    data.high = d["meta"]["high"].numbers();
    data.grossNotional = d["meta"]["grossNotional"].as<double>();
    ai::BacktestOptions o;
    o.mlpEpochs = static_cast<std::size_t>(num(spec, "mlpEpochs", 2500));
    o.mlpRefitEpochs = static_cast<std::size_t>(num(spec, "mlpRefitEpochs", 400));
    o.refitEvery = static_cast<std::size_t>(num(spec, "refitEvery", 21));
    const double t0 = nowMs();
    const ai::Backtest bt = ai::runBacktest(data, o);
    Value out = Value::object();
    out.set("days", static_cast<double>(bt.days));
    out.set("outsideDays", static_cast<double>(bt.outsideDays));
    out.set("truthTotal", arr(bt.truthTotal));
    out.set("elapsedMs", nowMs() - t0);
    Value baseline = Value::object();
    Value lag1 = scoreToJs(bt.lag1), weekly = scoreToJs(bt.weekly);
    lag1.set("label", "Yesterday's engine value");
    weekly.set("label", "Engine run weekly, carried forward");
    baseline.set("lag1", lag1);
    baseline.set("weekly", weekly);
    out.set("baseline", baseline);
    const double engineMs = num(d, "nativeMsPerRun", 0.0);
    Value models = Value::object();
    for (const auto& m : bt.models) {
      Value o2 = Value::object();
      o2.set("label", m.label);
      o2.set("parameters", static_cast<double>(m.parameters));
      o2.set("fitMs", m.fitMs);
      o2.set("inferMicros", m.inferMicros);
      o2.set("speedUp", m.inferMicros > 0.0 ? 1000.0 * engineMs / m.inferMicros : kNaN);  // engine run vs one prediction
      o2.set("static", scoreToJs(m.fittedOnce));
      o2.set("walkForward", scoreToJs(m.walkForward));
      models.set(m.key, o2);
    }
    out.set("models", models);
    return out;
  });
}

}  // namespace


namespace {

using Entry = Value (*)(Value);
const std::vector<std::pair<std::string, Entry>>& table() {
  static const std::vector<std::pair<std::string, Entry>> t = {
      {"version", [](Value) { return version(); }},
      {"coreDemo", &coreDemo},
      {"marketDemo", &marketDemo},
      // Every Monte Carlo analysis (spec.analysis) runs on the fused kernels: compiled for the GPU
      // (gpuJobs), finished from its read-backs (gpuAnalyse), or run on the CPU (cpuKernels).
      {"gpuKernels", &gpuKernels},
      {"gpuJobs", &gpuJobs},
      {"gpuAnalyse", &gpuAnalyse},
      {"cpuKernels", &cpuKernels},
      {"surrogateBacktest", &surrogateBacktest},
  };
  return t;
}

}  // namespace

Value call(const std::string& function, const Value& spec) {
  for (const auto& [name, entry] : table())
    if (name == function) return entry(spec);
  Value o = Value::object();
  o.set("error", "unknown entry point '" + function + "'");
  return o;
}

const std::vector<std::string>& functions() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> n;
    for (const auto& e : table()) n.push_back(e.first);
    return n;
  }();
  return names;
}

}  // namespace ccr::app
