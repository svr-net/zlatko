// Exposure by adaptive nested Monte Carlo: the netting set's discounted cash flows on the outer paths
// and on inner paths restarted from a path's state. See nested_exposure.hpp.
#include "ccr/nested/nested_exposure.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "ccr/core/math.hpp"
#include "ccr/instruments/asset_forward.hpp"
#include "ccr/instruments/european_option.hpp"
#include "ccr/instruments/interest_rate_swap.hpp"

namespace ccr::nested {

namespace {

constexpr double kEps = 1e-10;  // the trades' own tolerance for "on or before t"

struct SwapPeriod {
  double ts, te, tau, amount, fixedRate;  // amount = sign * notional: pays amount (L - K) tau at te
};
struct TerminalFlow {
  std::size_t asset;
  bool option;
  OptionType type;
  double notional, strike, maturity;
  double payoff(double spot) const {
    if (!option) return notional * (spot - strike);
    return notional * std::max(type == OptionType::Call ? spot - strike : strike - spot, 0.0);
  }
};

struct CashFlows {
  std::vector<SwapPeriod> periods;
  std::vector<TerminalFlow> terminals;
  std::vector<const PathwiseTrade*> trades;  // the supported trades, for the closed form
  std::vector<std::string> unsupported;
  std::vector<double> eventTimes() const {
    std::vector<double> t;
    for (const auto& p : periods) t.insert(t.end(), {p.ts, p.te});
    for (const auto& f : terminals) t.push_back(f.maturity);
    return t;
  }
  double lastPayment() const {
    double last = 0.0;
    for (const auto& p : periods) last = std::max(last, p.te);
    for (const auto& f : terminals) last = std::max(last, f.maturity);
    return last;
  }
};

std::size_t assetIndex(const ScenarioGenerator& g, const std::string& name) {
  for (std::size_t a = 0; a < g.assets().size(); ++a)
    if (g.assets()[a]->name() == name) return a;
  throw std::invalid_argument("nestedExposure: unknown asset '" + name + "'");
}

CashFlows collect(const ScenarioGenerator& g, const NettingSet& ns) {
  CashFlows cf;
  for (const auto& t : ns.trades) {
    if (const auto* swap = dynamic_cast<const InterestRateSwap*>(t.get())) {
      const auto& s = swap->schedule();
      const double sign = swap->direction() == InterestRateSwap::Direction::PayFixed ? 1.0 : -1.0;
      for (std::size_t i = 1; i < s.size(); ++i)
        cf.periods.push_back({s[i - 1], s[i], s[i] - s[i - 1], sign * swap->notional(), swap->fixedRate()});
      cf.trades.push_back(swap);
    } else if (const auto* fwd = dynamic_cast<const AssetForward*>(t.get())) {
      cf.terminals.push_back({assetIndex(g, fwd->asset()), false, OptionType::Call, fwd->notional(), fwd->strike(), fwd->maturity()});
      cf.trades.push_back(fwd);
    } else if (const auto* opt = dynamic_cast<const EuropeanOption*>(t.get())) {
      cf.terminals.push_back({assetIndex(g, opt->asset()), true, opt->type(), opt->notional(), opt->strike(), opt->expiry()});
      cf.trades.push_back(opt);
    } else {
      cf.unsupported.push_back(t->id());
    }
  }
  return cf;
}

// What happens on one future event date of an inner path.
struct Event {
  double time;
  std::vector<std::size_t> fix, pay, terminal;  // swap periods fixing / paying, terminal flows paying
};

// The program of an inner path from date t: the event dates after t in order.
std::vector<Event> program(const CashFlows& cf, double t) {
  std::vector<double> times;
  for (const auto& p : cf.periods) {
    if (p.te <= t + kEps) continue;
    if (p.ts > t + kEps) times.push_back(p.ts);
    times.push_back(p.te);
  }
  for (const auto& f : cf.terminals)
    if (f.maturity > t + kEps) times.push_back(f.maturity);
  std::sort(times.begin(), times.end());
  std::vector<Event> events;
  for (double u : times)
    if (events.empty() || u > events.back().time + 1e-9) events.push_back({u, {}, {}, {}});
  auto at = [&](double u) -> Event& {
    for (auto& e : events)
      if (std::fabs(e.time - u) <= 1e-9) return e;
    throw std::logic_error("nestedExposure: event date not in the program");
  };
  for (std::size_t i = 0; i < cf.periods.size(); ++i) {
    const auto& p = cf.periods[i];
    if (p.te <= t + kEps) continue;
    if (p.ts > t + kEps) at(p.ts).fix.push_back(i);
    at(p.te).pay.push_back(i);
  }
  for (std::size_t f = 0; f < cf.terminals.size(); ++f)
    if (cf.terminals[f].maturity > t + kEps) at(cf.terminals[f].maturity).terminal.push_back(f);
  return events;
}

double fixing(const HullWhite1F& hw, const SwapPeriod& p, double x) {
  return (1.0 / hw.zeroBond(p.ts, p.te, x) - 1.0) / p.tau;
}

}  // namespace

NestedExposureResult nestedExposure(const ScenarioGenerator& generator, const NettingSet& nettingSet, const TimeGrid& reporting,
                                    const SimulationConfig& simulation, const NestedExposureOptions& options, SpinPool& pool) {
  options.adaptive.validate();
  const CashFlows cf = collect(generator, nettingSet);
  NestedExposureResult result;
  result.unsupported = cf.unsupported;
  if (cf.trades.empty()) throw std::invalid_argument("nestedExposure: no supported trade (swaps, asset forwards, European options)");

  // The outer simulation: reporting dates plus every fixing and payment date.
  const TimeGrid grid = reporting.merged(cf.eventTimes());
  const ScenarioSet outer = generator.generate(grid, simulation);
  const std::size_t nP = outer.numPaths, nA = generator.assets().size();
  result.outerPaths = nP;
  const HullWhite1F& hw = *generator.rateModel();
  const Matrix chol = cholesky(generator.correlation());
  auto gridIndex = [&](double t) {
    const auto k = grid.find(t, 1e-8);
    if (!k) throw std::logic_error("nestedExposure: date not on the simulation grid");
    return *k;
  };

  for (ExposureProfile* profile : {&result.exact, &result.regression, &result.uniform, &result.adaptive})
    profile->pfeQuantile = options.pfeQuantile;
  // Grid indices of every fixing and payment date.
  std::vector<std::size_t> fixAt(cf.periods.size()), payAt(cf.periods.size()), terminalAt(cf.terminals.size());
  for (std::size_t i = 0; i < cf.periods.size(); ++i) {
    fixAt[i] = gridIndex(cf.periods[i].ts);
    payAt[i] = gridIndex(cf.periods[i].te);
  }
  for (std::size_t f = 0; f < cf.terminals.size(); ++f) terminalAt[f] = gridIndex(cf.terminals[f].maturity);

  for (std::size_t r = 0; r < reporting.size(); ++r) {
    const double t = reporting[r];
    const std::size_t k = gridIndex(t);
    NestedExposurePoint point;
    point.time = t;
    point.live = cf.lastPayment() > t + kEps;
    if (point.live) {
      const std::vector<Event> events = program(cf, t);

      // Features: x, the log spot of every asset a live terminal flow refers to, and the fixing of
      // every coupon that is fixed but not yet paid (its value no longer follows from x).
      std::vector<std::size_t> liveAssets, running;
      for (const auto& f : cf.terminals)
        if (f.maturity > t + kEps && std::find(liveAssets.begin(), liveAssets.end(), f.asset) == liveAssets.end()) liveAssets.push_back(f.asset);
      std::sort(liveAssets.begin(), liveAssets.end());
      for (std::size_t i = 0; i < cf.periods.size(); ++i)
        if (cf.periods[i].ts < t - kEps && cf.periods[i].te > t + kEps) running.push_back(i);
      // Fixings already set at t on each path (for the running coupons).
      Matrix known(nP, cf.periods.size(), 0.0);
      for (std::size_t i : running)
        for (std::size_t p = 0; p < nP; ++p) known(p, i) = fixing(hw, cf.periods[i], outer.rateState(p, fixAt[i]));

      OuterSample sample;
      sample.features = Matrix(nP, 1 + liveAssets.size() + running.size());
      sample.payoff.assign(nP, 0.0);
      sample.weight.assign(nP, 0.0);
      for (std::size_t p = 0; p < nP; ++p) {
        double* row = sample.features.row(p);
        std::size_t c = 0;
        row[c++] = outer.rateState(p, k);
        for (std::size_t a : liveAssets) row[c++] = std::log(outer.assetValues[a](p, k));
        for (std::size_t i : running) row[c++] = known(p, i);
        sample.weight[p] = outer.deflator(p, k);
        // The outer path's own cash flows after t, discounted to t.
        const double dk = outer.deflator(p, k);
        double y = 0.0;
        for (std::size_t i = 0; i < cf.periods.size(); ++i) {
          const auto& per = cf.periods[i];
          if (per.te <= t + kEps) continue;
          const double L = fixing(hw, per, outer.rateState(p, fixAt[i]));
          y += per.amount * (L - per.fixedRate) * per.tau * outer.deflator(p, payAt[i]) / dk;
        }
        for (std::size_t f = 0; f < cf.terminals.size(); ++f) {
          const auto& fl = cf.terminals[f];
          if (fl.maturity <= t + kEps) continue;
          y += fl.payoff(outer.assetValues[fl.asset](p, terminalAt[f])) * outer.deflator(p, terminalAt[f]) / dk;
        }
        sample.payoff[p] = y;
      }

      // The closed form on the outer paths (the reference).
      std::vector<double> exact(nP, 0.0), tmp(nP);
      for (const PathwiseTrade* trade : cf.trades) {
        trade->valueAtTime(outer, k, tmp.data());
        for (std::size_t p = 0; p < nP; ++p) exact[p] += tmp[p];
      }

      // Inner simulation from path p's state at t: exact Hull-White and log-normal steps between the
      // event dates, correlated as the outer simulation (the first 1 + nA rows of its Cholesky factor).
      const InnerSimulator inner = [&](std::size_t p, std::uint64_t seed, double* out, std::size_t n) {
        CounterNormals rng(seed);
        std::vector<double> z(nA + 2), w(nA + 1), spot(nA), fix(cf.periods.size());
        for (std::size_t q = 0; q < n; ++q) {
          double x = outer.rateState(p, k), lnD = 0.0, prev = t, y = 0.0;
          for (std::size_t a = 0; a < nA; ++a) spot[a] = outer.assetValues[a](p, k);
          for (std::size_t i = 0; i < cf.periods.size(); ++i) {
            const auto& per = cf.periods[i];
            if (per.te <= t + kEps) continue;
            if (per.ts < t - kEps) fix[i] = known(p, i);
            else if (per.ts <= t + kEps) fix[i] = fixing(hw, per, x);  // fixing today
          }
          for (const Event& e : events) {
            rng.fill(z.data(), nA + 2);
            for (std::size_t i = 0; i <= nA; ++i) {
              double sum = 0.0;
              for (std::size_t j = 0; j <= i; ++j) sum += chol(i, j) * z[j];
              w[i] = sum;
            }
            double integrated = 0.0;
            hw.evolve(prev, e.time, x, w[0], z[nA + 1], x, integrated);
            lnD -= integrated;
            for (std::size_t a = 0; a < nA; ++a) spot[a] = generator.assets()[a]->evolve(prev, e.time, spot[a], integrated, w[1 + a]);
            prev = e.time;
            const double D = std::exp(lnD);
            for (std::size_t i : e.pay) {
              const auto& per = cf.periods[i];
              y += per.amount * (fix[i] - per.fixedRate) * per.tau * D;
            }
            for (std::size_t i : e.fix) fix[i] = fixing(hw, cf.periods[i], x);
            for (std::size_t f : e.terminal) y += cf.terminals[f].payoff(spot[cf.terminals[f].asset]) * D;
          }
          out[q] = y;
        }
      };

      // EE* and PFE, refined jointly. The absolute tolerance of EE* is 0.1% of the size of the
      // discounted cash flows, so a netting set out of the money everywhere does not refine forever.
      double size = 0.0;
      for (std::size_t p = 0; p < nP; ++p) size += sample.weight[p] * std::fabs(sample.payoff[p]);
      size /= static_cast<double>(nP);
      const PositivePart ee(0.01, 1e-3 * size, "EE*");
      const Quantile pfe(options.pfeQuantile, true, 2.0, "PFE");
      const std::vector<const Statistic*> stats{&ee, &pfe};

      bool constant = true;  // every path in the same state (t = 0): V is one number, plain Monte Carlo
      for (std::size_t p = 1; p < nP && constant; ++p)
        for (std::size_t c = 0; c < sample.features.cols(); ++c)
          if (sample.features(p, c) != sample.features(0, c)) constant = false;

      if (constant) {
        const double v = mean(sample.payoff), w0 = sample.weight[0];
        point.eeExact = w0 * std::max(exact[0], 0.0);
        point.pfeExact = std::max(exact[0], 0.0);
        point.eeRegression = point.eeUniform = point.eeAdaptive = w0 * std::max(v, 0.0);
        point.pfeRegression = point.pfeUniform = point.pfeAdaptive = std::max(v, 0.0);
        point.evaluated = nP;
      } else {
        AdaptiveOptions o = options.adaptive;
        o.seed = deriveSeed(options.adaptive.seed, r);
        const AdaptiveResult a = estimateAdaptive(sample, inner, stats, o, pool);
        std::vector<double> exactEval, weightEval;
        for (std::size_t i : a.evaluation) {
          exactEval.push_back(exact[i]);
          weightEval.push_back(sample.weight[i]);
        }
        point.eeExact = ee.plugIn(exactEval, weightEval);
        point.pfeExact = pfe.plugIn(exactEval, weightEval);
        point.eeRegression = a.regression[0];
        point.pfeRegression = a.regression[1];
        point.eeAdaptive = a.estimate[0];
        point.pfeAdaptive = a.estimate[1];
        point.evaluated = a.evaluation.size();
        point.innerPaths = a.innerPaths;
        point.auditInnerPaths = a.auditInnerPaths;
        point.refined = a.refinedScenarios;
        point.rounds = a.rounds;
        point.stop = a.stop;
        point.proxyBasis = a.proxyBasis;
        for (const auto& b : a.bins) point.maxBiasRms = std::max(point.maxBiasRms, b.biasRms);
        if (options.uniform) {
          const auto perScenario = static_cast<std::size_t>(std::max<long long>(
              2, std::llround(static_cast<double>(a.innerPaths) / static_cast<double>(a.evaluation.size()))));
          const UniformResult u = estimateUniform(sample, inner, a.evaluation, perScenario, stats, o.seed, pool);
          point.eeUniform = u.estimate[0];
          point.pfeUniform = u.estimate[1];
          point.uniformInnerPerScenario = perScenario;
        }
      }
    }
    result.innerPaths += point.innerPaths;
    const std::pair<ExposureProfile*, std::pair<double, double>> rows[] = {
        {&result.exact, {point.eeExact, point.pfeExact}},
        {&result.regression, {point.eeRegression, point.pfeRegression}},
        {&result.uniform, {point.eeUniform, point.pfeUniform}},
        {&result.adaptive, {point.eeAdaptive, point.pfeAdaptive}}};
    for (const auto& [profile, values] : rows) {
      profile->times.push_back(t);
      profile->discountedExpectedExposure.push_back(values.first);
      profile->potentialFutureExposure.push_back(values.second);
      profile->expectedValue.push_back(0.0);
      profile->expectedExposure.push_back(0.0);
      profile->expectedNegativeExposure.push_back(0.0);
      profile->discountedExpectedNegativeExposure.push_back(0.0);
    }
    result.points.push_back(point);
  }
  return result;
}

}  // namespace ccr::nested
