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

  if (has(spec, "csa") && !(has(spec["csa"], "enabled") && !spec["csa"]["enabled"].as<bool>())) {
    const val c = spec["csa"];
    CollateralAgreement csa;
    csa.thresholdCounterparty = num(c, "thresholdCounterparty", 0.0);
    csa.thresholdOwn = has(c, "thresholdOwn") ? c["thresholdOwn"].as<double>() : std::numeric_limits<double>::infinity();
    if (csa.thresholdOwn < 0.0) csa.thresholdOwn = std::numeric_limits<double>::infinity();
    csa.minimumTransferAmount = num(c, "mta", 0.0);
    csa.independentAmount = num(c, "independentAmount", 0.0);
    csa.marginPeriodOfRisk = num(c, "mpr", 10.0 / 250.0);
    env.csa = csa;
  }
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
  o.set("library", std::string("zlatko ccr 0.1.0"));
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
    out.set("grossProfile", profileToJs(profileOn(r, grossPositiveValues(r.tradeValues), env.pfeQuantile)));

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

// CVA, DVA, term structure, running spread, pathwise CVA.
// Closed-form CVA analytics on an exposure profile: unilateral CVA and its term structure,
// bilateral CVA/DVA, running spread and recovery sensitivity. Shared by cva() and
// cvaFromProfile(), so a profile simulated on the GPU gets exactly the same analytics.
val cvaAnalytics(const val& spec, const Env& env, const ExposureProfile& profile) {
  const CreditCurve& cc = env.cptyCurve();
  const double rec = env.cptyRecovery();
  const auto& times = profile.times;

  val out = val::object();
  out.set("profile", profileToJs(profile));
  out.set("cva", unilateralCva(profile, cc, rec));
  out.set("termStructure", arr(cvaTermStructure(times, profile.discountedExpectedExposure, cc, rec)));
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

// CVA, DVA, term structure, running spread, pathwise CVA.
val cva(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const ExposureEngine engine(env.generator(), env.sim, env.baseGrid);
    const ExposureResult r = engine.run(env.nettingSet(true), env.pfeQuantile);
    val out = cvaAnalytics(spec, env, r.profile);
    out.set("pathwiseCva", pathwiseCva(r.scenarios, r.exposureValue, env.counterparty, env.cptyRecovery(), r.reportingIndices));
    return out;
  });
}

// CVA analytics on an exposure profile computed elsewhere (the WebGPU kernels):
// spec.profile = { times, ee, ene, pfe, discountedEe, discountedEne, expectedValue }.
ExposureProfile profileFromJs(const val& p, double pfeQuantile, const char* caller) {
  ExposureProfile profile;
  profile.times = vec(p["times"]);
  profile.expectedExposure = vec(p["ee"]);
  profile.expectedNegativeExposure = vec(p["ene"]);
  profile.discountedExpectedExposure = vec(p["discountedEe"]);
  profile.discountedExpectedNegativeExposure = vec(p["discountedEne"]);
  profile.potentialFutureExposure = has(p, "pfe") ? vec(p["pfe"]) : std::vector<double>(profile.times.size(), 0.0);
  profile.expectedValue = has(p, "expectedValue") ? vec(p["expectedValue"]) : std::vector<double>(profile.times.size(), 0.0);
  profile.pfeQuantile = pfeQuantile;
  const std::size_t n = profile.times.size();
  if (n == 0 || profile.expectedExposure.size() != n || profile.discountedExpectedExposure.size() != n ||
      profile.discountedExpectedNegativeExposure.size() != n || profile.expectedNegativeExposure.size() != n)
    throw std::invalid_argument(std::string(caller) + ": profile arrays must have one value per date");
  return profile;
}

val cvaFromProfile(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    return cvaAnalytics(spec, env, profileFromJs(spec["profile"], env.pfeQuantile, "cvaFromProfile"));
  });
}

// Wrong-way risk: pathwise CVA as a function of the exposure/intensity correlation.
val wrongWayRisk(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    if (env.credits.empty()) throw std::invalid_argument("wrong-way risk needs a stochastic counterparty intensity");
    const std::vector<double> rhos = has(spec, "rhos") ? vec(spec["rhos"]) : std::vector<double>{-0.6, -0.3, 0.0, 0.3, 0.6};
    const auto driver = static_cast<std::size_t>(num(spec, "wwrFactor", env.assets.empty() ? 0 : 1));
    const std::size_t creditFactor = 1 + env.assets.size() + env.counterparty;
    const CreditCurve& cc = env.cptyCurve();
    const double rec = env.cptyRecovery();

    val out = val::object();
    val results = val::array();
    for (double rho : rhos) {
      Matrix corr = env.correlation;
      corr(driver, creditFactor) = corr(creditFactor, driver) = rho;
      val o = val::object();
      o.set("rho", rho);
      try {
        const ExposureEngine engine(ScenarioGenerator(env.hw, env.assets, env.credits, corr), env.sim, env.baseGrid);
        const ExposureResult r = engine.run(env.nettingSet(true), env.pfeQuantile);
        const auto& s = r.scenarios;
        const auto& idx = r.reportingIndices;
        const Matrix& q = s.survival[env.counterparty];
        // Discounted exposure conditional on default in each period.
        std::vector<double> conditional(idx.size(), 0.0);
        for (std::size_t k = 1; k < idx.size(); ++k) {
          double num_ = 0.0, den = 0.0;
          for (std::size_t p = 0; p < s.numPaths; ++p) {
            const double dq = q(p, idx[k - 1]) - q(p, idx[k]);
            num_ += s.deflator(p, idx[k]) * std::max(r.exposureValue(p, idx[k]), 0.0) * dq;
            den += dq;
          }
          conditional[k] = den != 0.0 ? num_ / den : 0.0;
        }
        o.set("pathwiseCva", pathwiseCva(s, r.exposureValue, env.counterparty, rec, idx));
        o.set("independentCva", unilateralCva(r.profile, cc, rec));
        o.set("times", arr(r.profile.times));
        o.set("discountedEe", arr(r.profile.discountedExpectedExposure));
        o.set("conditionalDiscountedEe", arr(conditional));
      } catch (const std::exception& e) {
        o.set("error", std::string(e.what()));
      }
      results.call<void>("push", o);
    }
    out.set("results", results);
    out.set("driver", static_cast<double>(driver));
    return out;
  });
}

// CVA hedging: CS01 buckets and CDS hedge, market deltas with/without common random numbers.
// Credit hedging of CVA for a given exposure profile: bucketed CS01, the CDS hedge Jacobian and
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
  out.set("maturities", arr(mats));
  out.set("cs01", arr(cs01));
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
    const double parallel = rng.next();
    for (auto& q : moved) q.spread = std::max(q.spread + 1e-4 * shockBp * (0.7 * parallel + 0.3 * rng.next()), 1e-5);
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
  out.set("pnlUnhedged", arr(unhedged));
  out.set("pnlHedged", arr(hedged));
  return out;
}

// The credit part of hedging() for a profile simulated elsewhere (the WebGPU kernels).
val creditHedgingFromProfile(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    val out = creditHedges(spec, env, profileFromJs(spec["profile"], env.pfeQuantile, "creditHedgingFromProfile"));
    std::vector<double> carry;
    for (const auto& a : env.assets) carry.push_back(a->carryCurve().discount(env.baseGrid.horizon()));
    out.set("carryDiscountAtHorizon", arr(carry));
    return out;
  });
}

val hedging(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const ExposureEngine engine(env.generator(), env.sim, env.baseGrid);
    const NettingSet ns = env.nettingSet(true);
    const ExposureProfile profile = engine.run(ns, env.pfeQuantile).profile;
    const double rec = env.cptyRecovery();
    val out = creditHedges(spec, env, profile);

    // Market-risk deltas of CVA by bump-and-revalue.
    const CreditCurve& cc = env.cptyCurve();
    auto cvaWith = [&](const ScenarioGenerator& g, std::uint64_t seed) {
      SimulationConfig cfg = env.sim;
      cfg.seed = seed;
      return unilateralCva(ExposureEngine(g, cfg, env.baseGrid).run(ns, env.pfeQuantile).profile, cc, rec);
    };
    const ScenarioGenerator gen = env.generator();
    const std::vector<double> bumps = has(spec, "bumps") ? vec(spec["bumps"]) : std::vector<double>{0.04, 0.02, 0.01, 0.005};
    val deltas = val::array();
    for (std::size_t a = 0; a < env.assets.size(); ++a) {
      std::vector<double> crn, independent;
      for (double h : bumps) {
        const double spot = env.assets[a]->spot();
        auto bumped = [&](double b) {
          return gen.withAsset(a, std::make_shared<LognormalAsset>(env.assets[a]->withSpot(spot + b)));
        };
        crn.push_back((cvaWith(bumped(h), env.sim.seed) - cvaWith(bumped(-h), env.sim.seed)) / (2 * h));
        independent.push_back((cvaWith(bumped(h), env.sim.seed + 1) - cvaWith(bumped(-h), env.sim.seed + 2)) / (2 * h));
      }
      val o = val::object();
      o.set("name", env.assets[a]->name());
      o.set("bumps", arr(bumps));
      o.set("deltaCrn", arr(crn));
      o.set("deltaIndependent", arr(independent));
      o.set("carryDiscountAtHorizon", env.assets[a]->carryCurve().discount(env.baseGrid.horizon()));
      deltas.call<void>("push", o);
    }
    out.set("assetDeltas", deltas);

    // Parallel rate shift (DV01 of CVA), common random numbers.
    auto shifted = [&](double bp) {
      auto curve = std::make_shared<YieldCurve>(env.domestic->shifted(bp * 1e-4));
      return gen.withRateModel(std::make_shared<HullWhite1F>(curve, env.hw->meanReversion(), env.hw->volatility()));
    };
    out.set("cvaDv01", 0.5 * (cvaWith(shifted(1.0), env.sim.seed) - cvaWith(shifted(-1.0), env.sim.seed)));
    return out;
  });
}

// Compiles a netting set into the flat tables consumed by the WebGPU fused kernel.
val gpuPlan(val spec) {
  return guarded([&] {
    constexpr std::size_t kMaxAssets = 4, kMaxCredits = 2, kMaxFactors = 7, kMaxSlots = 32, kMaxNormals = 64;
    constexpr std::size_t kStepStride = 7 + 3 * kMaxAssets + kMaxCredits;
    const Env env = parseEnv(spec);
    if (env.assets.size() > kMaxAssets) throw std::invalid_argument("GPU kernel supports at most 4 assets");
    if (env.credits.size() > kMaxCredits) throw std::invalid_argument("GPU kernel supports at most 2 credits");

    const ExposureEngine engine(env.generator(), env.sim, env.baseGrid);
    const NettingSet ns = env.nettingSet(true);
    const TimeGrid reporting = engine.reportingGrid(ns);
    const TimeGrid grid = engine.simulationGrid(ns);
    const std::size_t nT = grid.size();
    const HullWhite1F& hw = *env.hw;
    const double a = hw.meanReversion(), sigma = hw.volatility();

    // Per-step deterministic coefficients (same exact transitions as the C++ scenario generator).
    std::vector<double> steps(nT * kStepStride, 0.0);
    for (std::size_t j = 1; j < nT; ++j) {
      const double t0 = grid[j - 1], t1 = grid[j], dt = t1 - t0, e = std::exp(-a * dt), s2 = sigma * sigma;
      const double varX = s2 * (1 - e * e) / (2 * a);
      const double varI = s2 / (a * a) * (dt - 2 * (1 - e) / a + (1 - e * e) / (2 * a));
      const double cov = s2 / (2 * a * a) * (1 - e) * (1 - e);
      const double sdX = std::sqrt(varX);
      const double c1 = sdX > 0 ? cov / sdX : 0.0;
      const double c2 = std::sqrt(std::max(varI - c1 * c1, 0.0));
      const double g2 = dt - 2 / a * (std::exp(-a * t0) - std::exp(-a * t1)) + 0.5 / a * (std::exp(-2 * a * t0) - std::exp(-2 * a * t1));
      const double intPhi = std::log(env.domestic->discount(t0) / env.domestic->discount(t1)) + s2 / (2 * a * a) * g2;
      double* row = &steps[j * kStepStride];
      row[0] = e;
      row[1] = sdX;
      row[2] = c1;
      row[3] = c2;
      row[4] = intPhi;
      row[5] = (1 - e) / a;
      row[6] = dt;
      for (std::size_t k = 0; k < env.assets.size(); ++k) {
        const auto& m = *env.assets[k];
        row[7 + 3 * k] = std::log(m.carryCurve().discount(t0) / m.carryCurve().discount(t1));
        row[8 + 3 * k] = m.volatility() * std::sqrt(dt);
        row[9 + 3 * k] = 0.5 * m.volatility() * m.volatility() * dt;
      }
      for (std::size_t c = 0; c < env.credits.size(); ++c) row[7 + 3 * kMaxAssets + c] = env.credits[c]->integratedShift(t1);
    }

    // Valuation terms per grid date: [type, slot, c, A, B, p1, p2, p3].
    enum : int { kBond = 0, kFixedFloat = 1, kAsset = 2, kSetFixing = 3, kOption = 4 };
    std::vector<std::vector<std::vector<double>>> termsAt(nT);
    auto addTerm = [&](std::size_t j, int type, double slot, double c, double A, double B, double p1 = 0, double p2 = 0,
                       double p3 = 0) { termsAt[j].push_back({double(type), slot, c, A, B, p1, p2, p3}); };
    auto bondA = [&](double t, double T) { return hw.zeroBond(t, T, 0.0); };
    auto bondB = [&](double t, double T) { return T <= t ? 0.0 : hw.B(t, T); };
    std::size_t slots = 0;
    std::vector<std::string> unsupported;
    for (const auto& info : env.trades) {
      if (info.type == "swap") {
        const InterestRateSwap& sw = *info.swap;
        const double sign = sw.direction() == InterestRateSwap::Direction::PayFixed ? 1.0 : -1.0;
        const double n = sw.notional(), k = sw.fixedRate();
        const auto& sch = sw.schedule();
        if (slots >= kMaxSlots) throw std::invalid_argument("too many swaps for the GPU kernel");
        const double slot = static_cast<double>(slots++);
        for (std::size_t i = 1; i < sch.size(); ++i) {
          const double ts = sch[i - 1], te = sch[i], tau = te - ts;
          // Record the fixing on the path at the last grid date on or before ts (as the C++ swap does).
          // Appended after the previous period's terms at that date, which still use the old fixing.
          const std::size_t kFix = grid.indexAtOrBefore(ts);
          addTerm(kFix, kSetFixing, slot, tau, bondA(grid[kFix], ts), bondB(grid[kFix], ts), bondA(grid[kFix], te),
                  bondB(grid[kFix], te));
          for (std::size_t j = 0; j < nT; ++j) {
            const double t = grid[j];
            if (te <= t + 1e-10) continue;
            if (ts >= t - 1e-10) {
              addTerm(j, kBond, 0, sign * n, bondA(t, ts), bondB(t, ts));
              addTerm(j, kBond, 0, -sign * n * (1 + k * tau), bondA(t, te), bondB(t, te));
            } else {
              addTerm(j, kFixedFloat, slot, sign * n * tau, bondA(t, te), bondB(t, te));
              addTerm(j, kBond, 0, -sign * n * k * tau, bondA(t, te), bondB(t, te));
            }
          }
        }
      } else if (info.type == "forward" || info.type == "option") {
        const val t = [&] {
          const val trades = spec["trades"];
          for (unsigned i = 0; i < trades["length"].as<unsigned>(); ++i)
            if (str(trades[i], "id", "") == info.trade->id()) return trades[i];
          throw std::invalid_argument("trade spec not found");
        }();
        const std::string assetName = str(t, "asset", "FX");
        std::size_t ai = env.assets.size();
        for (std::size_t q = 0; q < env.assets.size(); ++q)
          if (env.assets[q]->name() == assetName) ai = q;
        if (ai == env.assets.size()) throw std::invalid_argument("unknown asset " + assetName);
        const auto& m = *env.assets[ai];
        const double n = num(t, "notional", 1e6), strike = num(t, "strike", 1.0);
        const double T = info.type == "forward" ? num(t, "maturity", 1.0) : num(t, "expiry", 1.0);
        const bool call = str(t, "optionType", "call") == "call";
        for (std::size_t j = 0; j < nT; ++j) {
          const double tj = grid[j];
          if (tj >= T - 1e-10) continue;
          const double carry = m.carryCurve().discount(T) / m.carryCurve().discount(tj);
          if (info.type == "forward") {
            addTerm(j, kAsset, double(ai), n, carry, 0.0);
            addTerm(j, kBond, 0, -n * strike, bondA(tj, T), bondB(tj, T));
          } else {
            const double sd = m.volatility() * std::sqrt(T - tj);
            addTerm(j, kOption, double(ai), n, bondA(tj, T), bondB(tj, T), carry, strike, call ? sd : -sd);
          }
        }
      } else {
        unsupported.push_back(info.trade->id() + " (" + info.type + ": needs AMC regression, priced in WASM only)");
      }
    }

    std::vector<double> terms;
    std::vector<unsigned> termStart(nT + 1, 0);
    for (std::size_t j = 0; j < nT; ++j) {
      termStart[j] = static_cast<unsigned>(terms.size() / 8);
      for (const auto& rec : termsAt[j]) terms.insert(terms.end(), rec.begin(), rec.end());
    }
    termStart[nT] = static_cast<unsigned>(terms.size() / 8);

    std::vector<unsigned> callIndex(nT, 0), isReporting(nT, 0);
    const double mpr = env.csa ? env.csa->marginPeriodOfRisk : 0.0;
    for (std::size_t j = 0; j < nT; ++j) {
      callIndex[j] = static_cast<unsigned>(grid.indexAtOrBefore(std::max(grid[j] - mpr, 0.0)));
      isReporting[j] = reporting.find(grid[j]).has_value() ? 1u : 0u;
    }

    const std::size_t nCorr = 1 + env.assets.size() + env.credits.size();
    std::size_t nNormals = nCorr + 1;
    for (const auto& c : env.credits) nNormals += c->extraNormals();
    if (nNormals > kMaxNormals) throw std::invalid_argument("too many normals per step for the GPU kernel");
    const Matrix l = cholesky(env.correlation);
    std::vector<double> fparams(16 + kMaxFactors * kMaxFactors, 0.0);
    for (std::size_t c = 0; c < env.credits.size(); ++c) {
      fparams[4 * c + 0] = env.credits[c]->kappa();
      fparams[4 * c + 1] = env.credits[c]->theta();
      fparams[4 * c + 2] = env.credits[c]->xi();
      fparams[4 * c + 3] = env.credits[c]->y0();
    }
    if (env.csa) {
      fparams[8] = env.csa->thresholdCounterparty;
      fparams[9] = std::isfinite(env.csa->thresholdOwn) ? env.csa->thresholdOwn : -1.0;
      fparams[10] = env.csa->minimumTransferAmount;
      fparams[11] = env.csa->independentAmount;
    }
    for (std::size_t k = 0; k < env.assets.size(); ++k) fparams[12 + k] = env.assets[k]->spot();
    for (std::size_t i = 0; i < nCorr; ++i)
      for (std::size_t k = 0; k <= i; ++k) fparams[16 + i * kMaxFactors + k] = l(i, k);

    std::vector<double> marketSurvival;
    for (double t : grid.times()) marketSurvival.push_back(env.creditCurves.empty() ? 1.0 : env.cptyCurve().survival(t));

    val out = val::object();
    out.set("times", arr(grid.times()));
    out.set("reportingTimes", arr(reporting.times()));
    out.set("numPaths", static_cast<double>(env.sim.numPaths));
    out.set("seed", static_cast<double>(env.sim.seed % 4294967296ull));
    out.set("antithetic", env.sim.antithetic);
    out.set("numAssets", static_cast<double>(env.assets.size()));
    out.set("numCredits", static_cast<double>(env.credits.size()));
    out.set("counterparty", static_cast<double>(env.counterparty));
    out.set("substeps", static_cast<double>(env.credits.empty() ? 1 : env.credits.front()->substeps()));
    out.set("numCorrelated", static_cast<double>(nCorr));
    out.set("numNormals", static_cast<double>(nNormals));
    out.set("hasCsa", static_cast<bool>(env.csa));
    out.set("stepStride", static_cast<double>(kStepStride));
    out.set("maxFactors", static_cast<double>(kMaxFactors));
    out.set("steps", arr(steps));
    out.set("terms", arr(terms));
    out.set("termStart", u32arr(termStart));
    out.set("callIndex", u32arr(callIndex));
    out.set("isReporting", u32arr(isReporting));
    out.set("fparams", arr(fparams));
    out.set("pfeQuantile", env.pfeQuantile);
    out.set("recovery", env.creditCurves.empty() ? 0.4 : env.cptyRecovery());
    out.set("marketSurvival", arr(marketSurvival));
    out.set("numTerms", static_cast<double>(terms.size() / 8));
    val un = val::array();
    for (const auto& u : unsupported) un.call<void>("push", u);
    out.set("unsupported", un);
    for (const auto& c : env.credits)
      if (c->substeps() != env.credits.front()->substeps())
        throw std::invalid_argument("GPU kernel needs the same CIR substeps for all credits");
    return out;
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
  emscripten::function("cvaFromProfile", &cvaFromProfile);
  emscripten::function("wrongWayRisk", &wrongWayRisk);
  emscripten::function("hedging", &hedging);
  emscripten::function("creditHedgingFromProfile", &creditHedgingFromProfile);
  emscripten::function("gpuPlan", &gpuPlan);
}
