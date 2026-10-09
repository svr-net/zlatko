// Embind wrapper exposing the library to JavaScript.
//
// Every entry point takes a plain JS "spec" object (market, models, portfolio, CSA,
// simulation settings) and returns a plain JS object of numbers and Float64Arrays.
// Errors are returned as { error: "message" } rather than thrown across the boundary.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccr/ccr.hpp"

using emscripten::val;
using namespace ccr;

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

std::vector<double> vec(const val& v) { return emscripten::vecFromJSArray<double>(v); }

val arr(const std::vector<double>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

template <class T>
val typedArray(const std::vector<T>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

val u32arr(const std::vector<unsigned>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

val arrOfArr(const std::vector<std::vector<double>>& rows) {
  val out = val::array();
  for (const auto& r : rows) out.call<void>("push", arr(r));
  return out;
}

val matrixRows(const Matrix& m, std::size_t maxRows) {
  std::vector<std::vector<double>> rows;
  for (std::size_t r = 0; r < std::min(maxRows, m.rows()); ++r) rows.emplace_back(m.row(r), m.row(r) + m.cols());
  return arrOfArr(rows);
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
  const auto n = q["length"].as<unsigned>();
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
    for (unsigned i = 0; i < a["length"].as<unsigned>(); ++i)
      env.assets.push_back(std::make_shared<LognormalAsset>(str(a[i], "name", "FX"), num(a[i], "spot", 1.0),
                                                            num(a[i], "vol", 0.1), parseCurve(a[i]["carry"], 0.01)));
  }
  if (has(spec, "credits")) {
    const val c = spec["credits"];
    for (unsigned i = 0; i < c["length"].as<unsigned>(); ++i) {
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
    if (c["length"].as<unsigned>() != nF) throw std::invalid_argument("correlation matrix has wrong dimension");
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
    env.ownCurve = std::make_shared<CreditCurve>(
        bootstrapCreditCurve(*env.domestic, parseQuotes(spec["own"]["quotes"]), env.ownRecovery));
  }
  if (has(spec, "trades")) {
    const val t = spec["trades"];
    for (unsigned i = 0; i < t["length"].as<unsigned>(); ++i) env.trades.push_back(parseTrade(t[i], env));
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

std::vector<double> percentileRow(const Matrix& m, std::size_t j, const std::vector<double>& qs) {
  const std::vector<double> col = m.column(j);
  std::vector<double> out;
  for (double q : qs) out.push_back(quantile(col, q));
  return out;
}

// Percentile bands over time: result[k][j] = quantile qs[k] at date j.
val bands(const Matrix& m, const std::vector<double>& qs, const std::function<double(double, std::size_t)>& transform = nullptr) {
  std::vector<std::vector<double>> out(qs.size(), std::vector<double>(m.cols()));
  for (std::size_t j = 0; j < m.cols(); ++j) {
    std::vector<double> col = m.column(j);
    if (transform)
      for (double& v : col) v = transform(v, j);
    for (std::size_t k = 0; k < qs.size(); ++k) out[k][j] = quantile(col, qs[k]);
  }
  return arrOfArr(out);
}

ExposureProfile profileOn(const ExposureResult& r, const Matrix& values, double q) {
  return ExposureProfile::compute(selectColumns(values, r.reportingIndices), TimeGrid(r.profile.times),
                                  selectColumns(r.scenarios.deflator, r.reportingIndices), q);
}

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
      fits.call<void>("push", f);
    }
    val regression = val::object();
    regression.set("x", arr(x));
    regression.set("y", arr(y));
    regression.set("grid", arr(grid));
    regression.set("fits", fits);
    out.set("regression", regression);

    // Time grids.
    const double mpr = num(spec, "mpr", 10.0 / 250.0);
    out.set("standardGrid", arr(env.baseGrid.times()));
    out.set("laggedGrid", arr(env.baseGrid.withLaggedTimes(mpr).times()));

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
      qs.call<void>("push", q);
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
        quotes.call<void>("push", r);
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
      credits.call<void>("push", o);
    }
    out.set("credits", credits);
    return out;
  });
}

// Risk-factor simulation: fan charts and martingale checks.
val simulate(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const double t0 = nowMs();
    const ScenarioSet s = env.generator().generate(env.baseGrid, env.sim);
    const double elapsed = nowMs() - t0;
    const auto sampleCount = static_cast<std::size_t>(num(spec, "samplePaths", 30));
    const std::vector<double> qs = {0.05, 0.25, 0.5, 0.75, 0.95};
    const auto& times = s.grid.times();
    const std::size_t nT = s.numTimes(), nP = s.numPaths;

    val out = val::object();
    out.set("times", arr(times));
    out.set("numPaths", static_cast<double>(nP));
    out.set("elapsedMs", elapsed);

    // Rates: short rate, deflator, bond martingale.
    Matrix shortRate(nP, nT);
    for (std::size_t p = 0; p < nP; ++p)
      for (std::size_t j = 0; j < nT; ++j) shortRate(p, j) = env.hw->shortRate(times[j], s.rateState(p, j));
    const double bondMaturity = times.back() + num(spec, "bondTenor", 5.0);
    std::vector<double> meanD(nT), curveD(nT), meanBond(nT), meanR(nT);
    for (std::size_t j = 0; j < nT; ++j) {
      double d = 0.0, b = 0.0, r = 0.0;
      for (std::size_t p = 0; p < nP; ++p) {
        d += s.deflator(p, j);
        b += s.deflator(p, j) * s.zeroBond(p, j, bondMaturity);
        r += shortRate(p, j);
      }
      meanD[j] = d / nP;
      meanBond[j] = b / nP;
      meanR[j] = r / nP;
      curveD[j] = env.domestic->discount(times[j]);
    }
    val rates = val::object();
    rates.set("samples", matrixRows(shortRate, sampleCount));
    rates.set("bands", bands(shortRate, qs));
    rates.set("meanShortRate", arr(meanR));
    rates.set("meanDeflator", arr(meanD));
    rates.set("curveDiscount", arr(curveD));
    rates.set("bondMaturity", bondMaturity);
    rates.set("meanDeflatedBond", arr(meanBond));
    rates.set("bondTarget", env.domestic->discount(bondMaturity));
    out.set("rates", rates);

    val assets = val::array();
    for (std::size_t a = 0; a < s.assetValues.size(); ++a) {
      const Matrix& m = s.assetValues[a];
      std::vector<double> martingale(nT), target(nT);
      for (std::size_t j = 0; j < nT; ++j) {
        double v = 0.0;
        for (std::size_t p = 0; p < nP; ++p) v += s.deflator(p, j) * m(p, j);
        martingale[j] = v / nP;
        target[j] = s.assetModels[a]->spot() * s.assetModels[a]->carryCurve().discount(times[j]);
      }
      val o = val::object();
      o.set("name", s.assetModels[a]->name());
      o.set("samples", matrixRows(m, sampleCount));
      o.set("bands", bands(m, qs));
      o.set("meanDeflated", arr(martingale));
      o.set("target", arr(target));
      assets.call<void>("push", o);
    }
    out.set("assets", assets);

    val credits = val::array();
    for (std::size_t c = 0; c < s.survival.size(); ++c) {
      const CirIntensity& model = *s.creditModels[c];
      // lambda = y + psi(t), psi = d/dt of the integrated shift.
      std::vector<double> psi(nT);
      for (std::size_t j = 0; j < nT; ++j) {
        const double h = 1e-4, t = std::max(times[j], h);
        psi[j] = (model.integratedShift(t + h) - model.integratedShift(t - h)) / (2 * h);
      }
      Matrix lambda(nP, nT);
      for (std::size_t p = 0; p < nP; ++p)
        for (std::size_t j = 0; j < nT; ++j) lambda(p, j) = s.intensityState[c](p, j) + psi[j];
      std::vector<double> meanQ(nT), marketQ(nT);
      for (std::size_t j = 0; j < nT; ++j) {
        double q = 0.0;
        for (std::size_t p = 0; p < nP; ++p) q += s.survival[c](p, j);
        meanQ[j] = q / nP;
        marketQ[j] = model.marketCurve().survival(times[j]);
      }
      val o = val::object();
      o.set("name", model.name());
      o.set("intensitySamples", matrixRows(lambda, sampleCount));
      o.set("intensityBands", bands(lambda, qs));
      o.set("survivalSamples", matrixRows(s.survival[c], sampleCount));
      o.set("meanSurvival", arr(meanQ));
      o.set("marketSurvival", arr(marketQ));
      o.set("shift", arr(psi));
      credits.call<void>("push", o);
    }
    out.set("credits", credits);

    // Realised correlation of the factor increments over the first step.
    if (nT > 1) {
      std::vector<std::vector<double>> inc;
      std::vector<double> dx(nP);
      for (std::size_t p = 0; p < nP; ++p) dx[p] = s.rateState(p, 1);
      inc.push_back(dx);
      for (const auto& m : s.assetValues) {
        for (std::size_t p = 0; p < nP; ++p) dx[p] = std::log(m(p, 1) / m(p, 0));
        inc.push_back(dx);
      }
      for (const auto& m : s.intensityState) {
        for (std::size_t p = 0; p < nP; ++p) dx[p] = m(p, 1) - m(p, 0);
        inc.push_back(dx);
      }
      const std::size_t n = inc.size();
      Matrix corr(n, n);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
          const double mi = mean(inc[i]), mk = mean(inc[k]);
          double cov = 0.0, vi = 0.0, vk = 0.0;
          for (std::size_t p = 0; p < nP; ++p) {
            cov += (inc[i][p] - mi) * (inc[k][p] - mk);
            vi += std::pow(inc[i][p] - mi, 2);
            vk += std::pow(inc[k][p] - mk, 2);
          }
          corr(i, k) = cov / std::sqrt(vi * vk);
        }
      out.set("realisedCorrelation", matrixToJs(corr));
      out.set("targetCorrelation", matrixToJs(env.correlation));
    }
    return out;
  });
}

// Pricing every trade on every scenario.
val priceTrades(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const ExposureEngine engine(env.generator(), env.sim, env.baseGrid);
    const double t0 = nowMs();
    const ExposureResult r = engine.run(env.nettingSet(false), env.pfeQuantile);
    const double elapsed = nowMs() - t0;
    const auto& s = r.scenarios;
    const auto sampleCount = static_cast<std::size_t>(num(spec, "samplePaths", 25));

    val out = val::object();
    out.set("times", arr(r.profile.times));
    out.set("elapsedMs", elapsed);
    val trades = val::array();
    for (std::size_t i = 0; i < env.trades.size(); ++i) {
      const Matrix values = selectColumns(r.tradeValues[i], r.reportingIndices);
      const ExposureProfile p = profileOn(r, r.tradeValues[i], env.pfeQuantile);
      std::vector<double> deflated(values.cols());
      for (std::size_t k = 0; k < values.cols(); ++k) {
        double sum = 0.0;
        for (std::size_t q = 0; q < s.numPaths; ++q) sum += s.deflator(q, r.reportingIndices[k]) * values(q, k);
        deflated[k] = sum / s.numPaths;
      }
      val o = val::object();
      o.set("id", env.trades[i].trade->id());
      o.set("type", env.trades[i].type);
      o.set("maturity", env.trades[i].trade->maturity());
      o.set("value0", r.tradeValues[i](0, 0));
      o.set("reference", env.trades[i].reference);
      o.set("samples", matrixRows(values, sampleCount));
      o.set("bands", bands(values, {0.05, 0.5, 0.95}));
      o.set("meanDeflatedValue", arr(deflated));
      o.set("profile", profileToJs(p));
      if (env.trades[i].type == "bermudan") {
        const auto& u = *env.trades[i].swap;
        double best = 0.0;
        for (double e : env.trades[i].exercises) {
          const auto tail = u.tail(e);
          best = std::max(best, env.hw->europeanSwaption(e, tail.paymentTimes(), tail.accruals(), u.fixedRate(),
                                                         u.direction() == InterestRateSwap::Direction::PayFixed));
        }
        o.set("europeanLowerBound", best * u.notional());
      }
      trades.call<void>("push", o);
    }
    out.set("trades", trades);
    return out;
  });
}

// American Monte Carlo: Bermudan swaption exercise policy and exposure.
val amc(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const val t = spec["bermudan"];
    const auto swap = parseSwap(t, *env.domestic, "underlying");
    std::vector<double> exercises;
    if (has(t, "exercises")) {
      exercises = vec(t["exercises"]);
    } else {
      const auto& sch = swap.schedule();
      exercises.assign(sch.begin(), sch.end() - 1);
    }
    const auto degree = static_cast<std::size_t>(num(t, "degree", 3));
    const BermudanSwaption physical("physical", swap, exercises, Settlement::Physical, 1.0, degree);
    const BermudanSwaption cash("cash", swap, exercises, Settlement::Cash, 1.0, degree);
    const TimeGrid grid = env.baseGrid.merged(physical.eventTimes());
    const ScenarioSet s = env.generator().generate(grid, env.sim);
    const double t0 = nowMs();
    const auto vp = physical.valueWithExercise(s);
    const double elapsed = nowMs() - t0;
    const auto vc = cash.valueWithExercise(s);
    const auto sampleCount = static_cast<std::size_t>(num(spec, "samplePaths", 25));
    const bool payer = swap.direction() == InterestRateSwap::Direction::PayFixed;

    val out = val::object();
    out.set("times", arr(grid.times()));
    out.set("price", vp.price);
    out.set("elapsedMs", elapsed);
    out.set("exercises", arr(exercises));
    out.set("fixedRate", swap.fixedRate());
    out.set("parRate", swap.parRate(*env.domestic));

    std::vector<double> european, counts(exercises.size() + 1, 0.0);
    for (double e : exercises) {
      const auto tail = swap.tail(e);
      european.push_back(swap.notional() *
                         env.hw->europeanSwaption(e, tail.paymentTimes(), tail.accruals(), swap.fixedRate(), payer));
    }
    for (std::size_t k : vp.exerciseIndex) counts[k == BermudanSwaption::npos ? exercises.size() : k] += 1.0;
    for (double& c : counts) c /= static_cast<double>(s.numPaths);
    out.set("european", arr(european));
    out.set("exerciseDistribution", arr(counts));

    std::vector<double> sampleExercise;
    for (std::size_t p = 0; p < std::min(sampleCount, s.numPaths); ++p) {
      const std::size_t k = vp.exerciseIndex[p];
      sampleExercise.push_back(k == BermudanSwaption::npos ? -1.0 : exercises[k]);
    }
    out.set("sampleExerciseTimes", arr(sampleExercise));
    out.set("physicalSamples", matrixRows(vp.value, sampleCount));
    out.set("cashSamples", matrixRows(vc.value, sampleCount));
    out.set("physicalProfile", profileToJs(ExposureProfile::compute(vp.value, grid, s.deflator, env.pfeQuantile)));
    out.set("cashProfile", profileToJs(ExposureProfile::compute(vc.value, grid, s.deflator, env.pfeQuantile)));

    // Regression degree sensitivity of the price.
    val degrees = val::array();
    for (std::size_t d = 1; d <= 5; ++d) {
      const BermudanSwaption b("d", swap, exercises, Settlement::Physical, 1.0, d);
      val o = val::object();
      o.set("degree", static_cast<double>(d));
      o.set("price", b.valueWithExercise(s).price);
      degrees.call<void>("push", o);
    }
    out.set("degreeSensitivity", degrees);
    return out;
  });
}

// Netting-set exposure (with optional collateral).
val exposure(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const ExposureEngine engine(env.generator(), env.sim, env.baseGrid);
    const NettingSet ns = env.nettingSet(true);
    const double t0 = nowMs();
    const ExposureResult r = engine.run(ns, env.pfeQuantile);
    const double elapsed = nowMs() - t0;
    const auto sampleCount = static_cast<std::size_t>(num(spec, "samplePaths", 30));

    val out = val::object();
    out.set("elapsedMs", elapsed);
    out.set("numPaths", static_cast<double>(r.scenarios.numPaths));
    out.set("simulationDates", static_cast<double>(r.scenarios.numTimes()));
    out.set("reportingDates", static_cast<double>(r.reportingIndices.size()));
    out.set("profile", profileToJs(r.profile));
    out.set("collateralised", static_cast<bool>(ns.collateral));

    const Matrix exposureRep = selectColumns(r.exposureValue, r.reportingIndices);
    const Matrix nettedRep = selectColumns(r.nettedValue, r.reportingIndices);
    out.set("exposureSamples", matrixRows(exposureRep, sampleCount));
    out.set("nettedSamples", matrixRows(nettedRep, sampleCount));
    if (ns.collateral) {
      const Matrix coll = selectColumns(r.collateral, r.reportingIndices);
      out.set("collateralSamples", matrixRows(coll, sampleCount));
      out.set("collateralBands", bands(coll, {0.05, 0.5, 0.95}));
      out.set("uncollateralisedProfile", profileToJs(profileOn(r, r.nettedValue, env.pfeQuantile)));
    }
    const ExposureProfile gross = profileOn(r, grossPositiveValues(r.tradeValues), env.pfeQuantile);
    out.set("grossProfile", profileToJs(gross));
    const double grossEepe = gross.effectiveExpectedPositiveExposure(1.0);
    out.set("nettingBenefit", grossEepe > 0.0 ? 1.0 - r.profile.effectiveExpectedPositiveExposure(1.0) / grossEepe : 0.0);

    val trades = val::array();
    for (std::size_t i = 0; i < env.trades.size(); ++i) {
      val o = val::object();
      o.set("id", env.trades[i].trade->id());
      o.set("value0", r.tradeValues[i](0, 0));
      o.set("profile", profileToJs(profileOn(r, r.tradeValues[i], env.pfeQuantile)));
      trades.call<void>("push", o);
    }
    out.set("trades", trades);

    // Exposure distribution at a chosen horizon (for histograms).
    const double histTime = num(spec, "histogramTime", 1.0);
    const TimeGrid rep(r.profile.times);
    const std::size_t k = rep.indexAtOrBefore(histTime);
    out.set("histogramTime", rep[k]);
    out.set("histogramValues", arr(exposureRep.column(k)));
    return out;
  });
}

// Marginal / incremental / standalone allocation of exposure and CVA.
val allocation(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const ExposureEngine engine(env.generator(), env.sim, env.baseGrid);
    const ExposureResult r = engine.run(env.nettingSet(false), env.pfeQuantile);
    const auto& s = r.scenarios;
    const Matrix deflator = selectColumns(s.deflator, r.reportingIndices);
    std::vector<Matrix> values;
    for (const auto& m : r.tradeValues) values.push_back(selectColumns(m, r.reportingIndices));
    const Matrix net = selectColumns(r.nettedValue, r.reportingIndices);
    const auto& times = r.profile.times;
    const CreditCurve& cc = env.cptyCurve();
    const double rec = env.cptyRecovery();

    const auto marginal = marginalExposureContributions(values, net, &deflator);
    const auto marginalEe = marginalExposureContributions(values, net);
    const double total = unilateralCva(r.profile, cc, rec);

    val out = val::object();
    out.set("times", arr(times));
    out.set("totalCva", total);
    out.set("totalEe", arr(r.profile.expectedExposure));
    val trades = val::array();
    for (std::size_t i = 0; i < values.size(); ++i) {
      Matrix without(net.rows(), net.cols(), 0.0);
      for (std::size_t k = 0; k < values.size(); ++k)
        if (k != i) without += values[k];
      const auto inc = incrementalExposure(without, values[i], &deflator);
      const auto standalone = ExposureProfile::compute(values[i], TimeGrid(times), deflator);
      val o = val::object();
      o.set("id", env.trades[i].trade->id());
      o.set("marginalEe", arr(marginalEe[i]));
      o.set("marginalCva", unilateralCva(times, marginal[i], cc, rec));
      o.set("incrementalCva", unilateralCva(times, inc, cc, rec));
      o.set("standaloneCva", unilateralCva(standalone, cc, rec));
      o.set("standaloneEe", arr(standalone.expectedExposure));
      trades.call<void>("push", o);
    }
    out.set("trades", trades);
    return out;
  });
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
  std::optional<ExposureEngine> engine;  // empty when the set-up is invalid (see error)
  NettingSet nettingSet;
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
  std::optional<ExposureResult> paths;           // CPU only, when requested
};

Job makeJob(std::string label, const Env& env, const std::function<ScenarioGenerator()>& generator,
            const SimulationConfig& sim, const std::optional<CollateralAgreement>& csa) {
  Job job;
  job.label = std::move(label);
  job.nettingSet = env.nettingSet(false);
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

std::vector<Job> analysisJobs(const val& spec, const Env& env, const std::string& analysis) {
  const ScenarioGenerator gen = env.generator();
  auto fixed = [&](const ScenarioGenerator& g) { return [g] { return g; }; };
  std::vector<Job> jobs;
  if (analysis == "exposure" || analysis == "cva" || analysis == "validation") {
    jobs.push_back(makeJob("base", env, fixed(gen), env.sim, env.csa));
  } else if (analysis == "collateral") {
    for (const auto& v : csaVariants(spec)) jobs.push_back(makeJob(v.name, env, fixed(gen), env.sim, v.csa));
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

JobResult runOnCpu(const Job& job, const Env& env, bool keepPaths = false) {
  JobResult r;
  if (!job.engine) {
    r.error = job.error;
    return r;
  }
  const double t0 = nowMs();
  ExposureResult x = job.engine->run(job.nettingSet, env.pfeQuantile);
  r.elapsedMs = nowMs() - t0;
  r.profile = x.profile;
  r.simulationDates = x.scenarios.numTimes();
  r.numPaths = x.scenarios.numPaths;
  if (!env.credits.empty()) {
    const auto& s = x.scenarios;
    const auto& idx = x.reportingIndices;
    const Matrix& q = s.survival[env.counterparty];
    r.pathwiseCva = pathwiseCva(s, x.exposureValue, env.counterparty, env.cptyRecovery(), idx);
    r.conditionalDiscountedEe.assign(idx.size(), kNaN);
    for (std::size_t k = 0; k < idx.size(); ++k) {
      double sum = 0.0;
      for (std::size_t p = 0; p < s.numPaths; ++p) sum += q(p, idx[k]);
      r.meanSurvival.push_back(sum / static_cast<double>(s.numPaths));
    }
    for (std::size_t k = 1; k < idx.size(); ++k) {
      double num_ = 0.0, den = 0.0;
      for (std::size_t p = 0; p < s.numPaths; ++p) {
        const double dq = q(p, idx[k - 1]) - q(p, idx[k]);
        num_ += s.deflator(p, idx[k]) * std::max(x.exposureValue(p, idx[k]), 0.0) * dq;
        den += dq;
      }
      if (den != 0.0) r.conditionalDiscountedEe[k] = num_ / den;
    }
  }
  if (keepPaths) r.paths = std::move(x);
  return r;
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

// The combine step of every analysis: identical for CPU and GPU job results.
val combine(const val& spec, const Env& env, const std::string& analysis, const std::vector<Job>& jobs,
            const std::vector<JobResult>& results) {
  const CreditCurve* cc = env.creditCurves.empty() ? nullptr : &env.cptyCurve();
  const double rec = env.creditCurves.empty() ? 0.4 : env.cptyRecovery();
  val out = val::object();
  if (analysis == "exposure") {
    const JobResult& r = results.at(0);
    if (!r.error.empty()) throw std::invalid_argument(r.error);
    out.set("profile", profileToJs(r.profile));
    out.set("numPaths", static_cast<double>(r.numPaths));
    out.set("simulationDates", static_cast<double>(r.simulationDates));
    out.set("reportingDates", static_cast<double>(r.profile.times.size()));
    out.set("elapsedMs", r.elapsedMs);
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
      list.call<void>("push", o);
    }
    out.set("variants", list);
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
      list.call<void>("push", o);
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
      deltas.call<void>("push", o);
    }
    out.set("assetDeltas", deltas);
    out.set("cvaDv01", 0.5 * (cvaAt(i) - cvaAt(i + 1)));
  } else {
    throw std::invalid_argument("unknown analysis '" + analysis + "'");
  }
  (void)jobs;
  return out;
}

val runAnalysisOnCpu(const val& spec, const std::string& analysis) {
  const Env env = parseEnv(spec);
  const auto jobs = analysisJobs(spec, env, analysis);
  std::vector<JobResult> results;
  for (const auto& job : jobs) results.push_back(runOnCpu(job, env));
  return combine(spec, env, analysis, jobs, results);
}

// ------------------------------------------------------------------ CPU entry points

val cva(val spec) {
  return guarded([&] { return runAnalysisOnCpu(spec, "cva"); });
}

val wrongWayRisk(val spec) {
  return guarded([&] { return runAnalysisOnCpu(spec, "wrongWayRisk"); });
}

val hedging(val spec) {
  return guarded([&] { return runAnalysisOnCpu(spec, "hedging"); });
}

// CSA comparison; the specified CSA also returns sample paths and collateral bands.
val collateral(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto jobs = analysisJobs(spec, env, "collateral");
    std::vector<JobResult> results;
    for (std::size_t i = 0; i < jobs.size(); ++i) results.push_back(runOnCpu(jobs[i], env, i == 1));
    val out = combine(spec, env, "collateral", jobs, results);
    const ExposureResult& r = *results[1].paths;
    const auto sampleCount = static_cast<std::size_t>(num(spec, "samplePaths", 12));
    out.set("nettedSamples", matrixRows(selectColumns(r.nettedValue, r.reportingIndices), sampleCount));
    out.set("collateralBands", bands(selectColumns(r.collateral, r.reportingIndices), {0.05, 0.5, 0.95}));
    return out;
  });
}

// ------------------------------------------------------------------ WebGPU entry points
//
// gpuKernels() returns the WGSL sources. gpuJobs(spec) compiles the jobs of spec.analysis
// into GPU plans: ready-made buffers, dispatch sizes and buffer sizes, so the host only
// uploads, dispatches and reads back. gpuAnalyse(spec) takes the read-backs
// (spec.gpuOutputs[i] = { sums, pfe }, null for an invalid job) and returns the result in
// the shape of the CPU analysis. gpuEmulate(spec) runs the kernels on the CPU instead.

val gpuKernels(val) {
  val o = val::object();
  o.set("fusedExposure", gpu::fusedExposureKernel());
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
  o.set("header", typedArray(p.header));
  o.set("indices", typedArray(p.indices));
  o.set("params", typedArray(p.params));
  o.set("steps", typedArray(p.steps));
  o.set("terms", typedArray(p.terms.empty() ? std::vector<float>{0.0f} : p.terms));
  val dispatch = val::object();
  dispatch.set("fused", static_cast<double>(p.fusedDispatch()));
  dispatch.set("reduce", static_cast<double>(p.reduceDispatch()));
  dispatch.set("pfe", static_cast<double>(p.pfeDispatch()));
  o.set("dispatch", dispatch);
  val bytes = val::object();
  bytes.set("cube", static_cast<double>(p.cubeBytes()));
  bytes.set("partials", static_cast<double>(p.partialsBytes()));
  bytes.set("sums", static_cast<double>(p.sumsBytes()));
  bytes.set("pfe", static_cast<double>(p.pfeBytes()));
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
    if (const std::string why = gpu::limitation(*job.engine, job.nettingSet); !why.empty()) {
      g.unsupported = why;
      return g;
    }
    g.plans.emplace_back(gpu::compile(*job.engine, job.nettingSet, env.pfeQuantile, env.counterparty));
  }
  return g;
}

val gpuJobs(val spec) {
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
      plans.call<void>("push", g.plans[i] ? planToJs(*g.plans[i], g.jobs[i].label) : val::null());
    out.set("plans", plans);
    return out;
  });
}

// GPU validation: the fused kernels against the CPU library on the same specification.
val validation(const val& spec, const Env& env, const JobResult& g) {
  const JobResult w = runOnCpu(analysisJobs(spec, env, "cva").at(0), env);
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
  const double gpuCva = unilateralCva(g.profile, cc, rec), cpuCva = unilateralCva(w.profile, cc, rec);
  std::vector<double> market;
  for (double t : g.profile.times) market.push_back(cc.survival(t));
  double grossNotional = 0.0;
  for (const auto& t : env.trades) grossNotional += std::fabs(t.notional);
  const double tol = 4.0 / std::sqrt(static_cast<double>(env.sim.numPaths));

  val checks = val::object();
  checks.set("tol", tol);
  // f32 term cancellation: the t = 0 difference per unit of gross notional.
  checks.set("t0Diff", std::fabs(g.profile.expectedValue[0] - w.profile.expectedValue[0]) / std::max(grossNotional, 1.0));
  checks.set("eeDiff", maxRelDiff(g.profile.expectedExposure, w.profile.expectedExposure));
  checks.set("cvaDiff", std::fabs(gpuCva - cpuCva) / std::max(std::fabs(cpuCva), 1e-12));
  checks.set("survDiff", g.meanSurvival.empty() ? 0.0 : maxRelDiff(g.meanSurvival, market));
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
  out.set("wasm", side(w, cpuCva));
  out.set("marketSurvival", arr(market));
  out.set("checks", checks);
  out.set("numPaths", static_cast<double>(env.sim.numPaths));
  return out;
}

val analyseFused(const val& spec, const std::function<std::optional<gpu::FusedOutput>(std::size_t, const gpu::FusedPlan&)>& output) {
  const Env env = parseEnv(spec);
  const GpuJobs g = compileJobs(spec, env);
  if (!g.unsupported.empty()) throw std::invalid_argument(g.unsupported);
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
  if (analysis == "validation") return validation(spec, env, results.at(0));
  return combine(spec, env, analysis, g.jobs, results);
}

val gpuAnalyse(val spec) {
  return guarded([&] {
    const val outputs = spec["gpuOutputs"];
    return analyseFused(spec, [&](std::size_t i, const gpu::FusedPlan&) -> std::optional<gpu::FusedOutput> {
      if (i >= outputs["length"].as<std::size_t>() || outputs[i].isNull() || outputs[i].isUndefined()) return std::nullopt;
      gpu::FusedOutput o;
      o.sums = emscripten::convertJSArrayToNumberVector<float>(outputs[i]["sums"]);
      o.pfe = emscripten::convertJSArrayToNumberVector<float>(outputs[i]["pfe"]);
      return o;
    });
  });
}

val gpuEmulate(val spec) {
  return guarded([&] {
    return analyseFused(spec, [](std::size_t, const gpu::FusedPlan& plan) -> std::optional<gpu::FusedOutput> {
      return gpu::runFusedReference(plan);
    });
  });
}

// ------------------------------------------------------------------ CPU fused-kernel backend
//
// The fused kernels on the CPU, split across Web Workers like workgroups across GPU cores:
// each worker calls cpuKernelSlices with spec.part / spec.parts and runs that share of the
// workgroups of every job; cpuKernelAnalyse(spec) with spec.parts = the workers' results (in
// part order) finishes each job exactly as the GPU's reduce and PFE kernels do and returns
// the same result as the GPU path. The result does not depend on the number of workers.

val cpuKernelSlices(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const GpuJobs g = compileJobs(spec, env);
    val out = val::object();
    if (!g.unsupported.empty()) {
      out.set("unsupported", g.unsupported);
      return out;
    }
    const auto part = static_cast<std::size_t>(num(spec, "part", 0));
    const auto parts = static_cast<std::size_t>(std::max(1.0, num(spec, "parts", 1)));
    if (part >= parts) throw std::invalid_argument("cpuKernelSlices: part out of range");
    const double t0 = nowMs();
    val jobs = val::array();
    for (const auto& plan : g.plans) {
      if (!plan) {
        jobs.call<void>("push", val::null());
        continue;
      }
      const std::size_t numWG = plan->numWorkgroups();
      const gpu::FusedSlice sl = gpu::runFusedWorkgroups(*plan, numWG * part / parts, numWG * (part + 1) / parts);
      val o = val::object();
      o.set("wgBegin", static_cast<double>(sl.wgBegin));
      o.set("wgEnd", static_cast<double>(sl.wgEnd));
      o.set("pathBegin", static_cast<double>(sl.pathBegin));
      o.set("pathEnd", static_cast<double>(sl.pathEnd));
      o.set("partials", typedArray(sl.partials));
      o.set("exposure", typedArray(sl.exposure));
      jobs.call<void>("push", o);
    }
    out.set("jobs", jobs);
    out.set("kernelMs", nowMs() - t0);
    return out;
  });
}

val cpuKernelAnalyse(val spec) {
  return guarded([&] {
    const val parts = spec["parts"];
    const auto nParts = parts["length"].as<std::size_t>();
    return analyseFused(spec, [&](std::size_t job, const gpu::FusedPlan& plan) -> std::optional<gpu::FusedOutput> {
      std::vector<gpu::FusedSlice> slices;
      for (std::size_t k = 0; k < nParts; ++k) {
        const val o = parts[k]["jobs"][job];
        if (o.isNull() || o.isUndefined()) return std::nullopt;
        gpu::FusedSlice sl;
        sl.wgBegin = o["wgBegin"].as<std::size_t>();
        sl.wgEnd = o["wgEnd"].as<std::size_t>();
        sl.pathBegin = o["pathBegin"].as<std::size_t>();
        sl.pathEnd = o["pathEnd"].as<std::size_t>();
        sl.partials = emscripten::convertJSArrayToNumberVector<float>(o["partials"]);
        sl.exposure = emscripten::convertJSArrayToNumberVector<float>(o["exposure"]);
        slices.push_back(std::move(sl));
      }
      return gpu::finishFused(plan, slices);
    });
  });
}

}  // namespace

EMSCRIPTEN_BINDINGS(ccr) {
  emscripten::function("version", &version);
  emscripten::function("coreDemo", &coreDemo);
  emscripten::function("marketDemo", &marketDemo);
  emscripten::function("simulate", &simulate);
  emscripten::function("priceTrades", &priceTrades);
  emscripten::function("amc", &amc);
  emscripten::function("exposure", &exposure);
  emscripten::function("allocation", &allocation);
  emscripten::function("cva", &cva);
  emscripten::function("collateral", &collateral);
  emscripten::function("wrongWayRisk", &wrongWayRisk);
  emscripten::function("hedging", &hedging);
  emscripten::function("gpuKernels", &gpuKernels);
  emscripten::function("gpuJobs", &gpuJobs);
  emscripten::function("gpuAnalyse", &gpuAnalyse);
  emscripten::function("gpuEmulate", &gpuEmulate);
  emscripten::function("cpuKernelSlices", &cpuKernelSlices);
  emscripten::function("cpuKernelAnalyse", &cpuKernelAnalyse);
}
