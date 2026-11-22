#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ccr/core/parallel.hpp"
#include "ccr/core/time_grid.hpp"
#include "ccr/exposure/exposure_engine.hpp"
#include "ccr/exposure/exposure_profile.hpp"
#include "ccr/models/scenario_generator.hpp"
#include "ccr/nested/adaptive_nested.hpp"

namespace ccr::nested {

/// Exposure of a netting set by adaptive nested Monte Carlo (chapter X11).
///
/// On every reporting date t the conditional value of the netting set,
///
///   V(t, path) = E[ sum_{t_j > t} D(t, t_j) CF_j | state of the path at t ],
///
/// is treated as unknown: the outer paths of one ordinary simulation carry their own discounted cash
/// flows after t (one-sample estimates of V), and inner simulations restart the Hull-White and
/// log-normal models from a path's state at t with their exact transitions between the cash-flow
/// dates. The regression features are the Markov state at t: the Hull-White state x, the log spot of
/// every asset a live trade refers to, and the fixing of every swap coupon that is fixed but not yet
/// paid. Supported trades are InterestRateSwap, AssetForward and EuropeanOption (others are listed in
/// `unsupported` and left out). All three have closed forms on the scenarios, so every estimate is
/// compared with the exact statistic on the same evaluation scenarios. (EuropeanOption's closed form
/// ignores the rate volatility in the forward's volatility, so for options that reference carries a
/// small model difference that the nest, which simulates the full dynamics, does not.)
struct NestedExposureOptions {
  double pfeQuantile = 0.95;
  AdaptiveOptions adaptive;  ///< per reporting date (its innerBudget too)
  bool uniform = true;       ///< also the classical nest with the same inner paths per date
};

/// One reporting date: EE* = E[D(0, t) max(V, 0)] and PFE = quantile of max(V, 0) by four methods on
/// the same evaluation scenarios: the closed form (exact), the proxy alone (least-squares Monte Carlo),
/// the classical nest with the adaptive run's inner paths spread evenly, and the adaptive nest.
struct NestedExposurePoint {
  double time = 0.0;
  double eeExact = 0.0, eeRegression = 0.0, eeUniform = 0.0, eeAdaptive = 0.0;
  double pfeExact = 0.0, pfeRegression = 0.0, pfeUniform = 0.0, pfeAdaptive = 0.0;
  std::size_t evaluated = 0;         ///< evaluation scenarios
  std::size_t innerPaths = 0;        ///< inner paths of the adaptive nest (audit included)
  std::size_t auditInnerPaths = 0;
  std::size_t refined = 0;           ///< evaluation scenarios with inner paths
  std::size_t rounds = 0;
  std::size_t uniformInnerPerScenario = 0;
  double maxBiasRms = 0.0;           ///< largest measured proxy bias over the bins
  std::size_t proxyBasis = 0;
  StopReason stop = StopReason::Converged;
  bool live = false;                 ///< a cash flow remains after t (else every value is 0)
};

struct NestedExposureResult {
  std::vector<NestedExposurePoint> points;
  std::size_t outerPaths = 0;
  std::vector<std::string> unsupported;  ///< trades left out
  /// The four EE* / PFE profiles (times, discountedExpectedExposure, potentialFutureExposure), for CVA.
  ExposureProfile exact, regression, uniform, adaptive;
  std::size_t innerPaths = 0;            ///< adaptive, all dates
};

/// Runs the nest on the reporting dates of `reporting` (dates after the horizon of the simulation are
/// dropped). The outer simulation is `generator` on the reporting dates merged with the trades' fixing
/// and payment dates, with `simulation`'s paths and seed.
NestedExposureResult nestedExposure(const ScenarioGenerator& generator, const NettingSet& nettingSet, const TimeGrid& reporting,
                                    const SimulationConfig& simulation, const NestedExposureOptions& options,
                                    SpinPool& pool = defaultPool());

}  // namespace ccr::nested
