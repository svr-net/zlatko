# zlatko — counterparty credit exposure library

A C++17 library that models the techniques in

> G. Cesari, J. Aquilina, N. Charpillon, Z. Filipović, G. Lee, I. Manda,
> *Modelling, Pricing, and Hedging Counterparty Credit Exposure: A Technical Guide*,
> Springer Finance, 2009.

The book explains how to build a counterparty exposure system. You simulate the risk factors jointly
under the risk-neutral measure, revalue every trade on every scenario (using American Monte Carlo
for trades without a closed form), and apply netting and collateral. The resulting exposure
distribution gives the regulatory exposure measures, the price of counterparty risk (CVA/DVA,
including wrong-way risk), and the hedge sensitivities.
This library is an independent implementation of that framework. It does not reproduce the text of
the book.

It has no dependencies beyond the C++17 standard library.

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build            # or ./build/ccr_tests [name-filter]
./build/counterparty_exposure_demo
```

Options: `-DCCR_BUILD_TESTS=OFF` and `-DCCR_BUILD_EXAMPLES=OFF`. `cmake --install` installs the
headers and a `ccr::ccr` CMake target.

## How the library maps onto the book

| Topic | Module | What is implemented |
|---|---|---|
| Exposure definitions and measures | `exposure/exposure_profile.hpp` | EE, ENE, PFE (any quantile), discounted EE/ENE, EPE, effective EE, effective EPE (Basel), peak PFE |
| Monte Carlo framework and time grid | `core/time_grid.hpp`, `models/scenario_generator.hpp` | Exposure grids (weekly → monthly → quarterly), event dates merged into the grid, joint correlated simulation (Cholesky), antithetic variates, reproducible seeds |
| Interest-rate model | `models/hull_white.hpp` | One-factor Hull–White (G1++) fitted to the initial curve, exact joint transition of x(t) and ∫r dt (exact bank-account numeraire), analytic bonds, bond options and Jamshidian swaptions |
| FX / equity models | `models/lognormal_asset.hpp` | Log-normal factors drifting at the simulated domestic short rate minus foreign rate/dividend yield (exact martingales under stochastic rates) |
| Credit model | `models/cir_intensity.hpp`, `market/credit_curve.hpp` | CIR++ stochastic intensity fitted exactly to the market survival curve. Brownian-bridge sub-stepping keeps its driver correlated with the market factors. Piecewise-flat hazard curves |
| Pricing on scenarios | `instruments/*` | Swaps (with fixings recovered on the path), FX/equity forwards, European options, CDS (bootstrapping of hazard curves) |
| American Monte Carlo | `instruments/bermudan_swaption.hpp`, `core/regression.hpp` | Longstaff–Schwartz exercise policy plus regression-based mark-to-market at every grid date. Physical vs cash settlement (exposure carries on, or stops, after exercise) |
| Netting | `exposure/exposure_engine.hpp` | Netting sets, netted vs gross exposure |
| Collateral | `exposure/collateral.hpp` | CSA with two-way thresholds, minimum transfer amount, independent amount and margin period of risk. Margin call dates t − MPR are simulated as auxiliary grid dates |
| Exposure allocation | `exposure/allocation.hpp` | Marginal (Euler) contributions that add up to the netting-set EE/CVA, and incremental exposure of a new trade |
| Pricing counterparty risk | `cva/cva.hpp` | Unilateral CVA, CVA term structure, bilateral CVA/DVA with first-to-default, CVA as a running spread |
| Wrong-way risk | `cva/cva.hpp` (`pathwiseCva`) | Path-by-path CVA using the simulated survival probabilities, with the intensity correlated to the exposure drivers |
| Hedging | `hedging/sensitivities.hpp` | Bucketed CS01 of CVA with re-bootstrapped curves, CDS hedge notionals from the hedge Jacobian, and market-risk deltas by bump-and-revalue with common random numbers |

## Conventions

- Times are year fractions from today (t = 0). Values are in the domestic currency, from our
  point of view. A positive value is an exposure to the counterparty.
- Simulated quantities are `Matrix` objects with one row per path and one column per grid date.
- `Trade::valueCube` gives the value at t_j *after* any cash flow paid at t_j.
- The numeraire is the bank account, so `ScenarioSet::deflator` holds D(0, t) = exp(−∫r).
  E[D(0, t) V(t)] equals V(0) for a trade with no cash flows before t. The tests check this
  property throughout.

## Example

```cpp
#include "ccr/ccr.hpp"
using namespace ccr;

auto usd = std::make_shared<YieldCurve>(YieldCurve::flat(0.03));
auto hw  = std::make_shared<HullWhite1F>(usd, 0.05, 0.01);
ExposureEngine engine(ScenarioGenerator(hw), {5000, 42, true}, TimeGrid::standardExposureGrid(10.0));

NettingSet ns;
ns.trades.push_back(std::make_shared<InterestRateSwap>(
    InterestRateSwap::vanilla("IRS", 10e6, 0.03, 0.0, 10.0, 2, InterestRateSwap::Direction::PayFixed)));
ns.collateral = CollateralAgreement{};          // zero-threshold one-way CSA, 10-day MPR

ExposureResult r = engine.run(ns);
double eepe = r.profile.effectiveExpectedPositiveExposure(1.0);
double cva  = unilateralCva(r.profile, bootstrapCreditCurve(*usd, {{5.0, 0.01}}, 0.4), 0.4);
```

`examples/counterparty_exposure_demo.cpp` runs the whole workflow on a mixed IR/FX netting set:
exposure profiles with and without a CSA, the netting benefit, Bermudan swaption exposure by AMC
under both settlement types, CVA/DVA, CVA allocation, wrong-way risk as correlation varies, and CVA
hedging.

## Modelling notes and limitations

- **Single curve, one rate factor.** Floating legs are projected and discounted on the same
  Hull–White curve. Foreign rates and dividend yields are deterministic.
- **European options** are revalued with the Black formula using the simulated domestic discount
  factor. The contribution of rate volatility to the forward's volatility is ignored.
- **AMC** regresses on the Hull–White state with a polynomial basis, and it estimates the exercise
  policy in-sample. For production use, estimate the policy on an independent pre-simulation to
  remove the small upward bias.
- **Collateral and cash flows.** Over the margin period of risk the model keeps paying trade cash
  flows, but collateral is not returned. So exposure spikes on payment and maturity dates when we
  had posted collateral beforehand. The book discusses this effect.
- **CIR++.** Full-truncation Euler on sub-steps. The deterministic shift can be negative if the
  CIR parameters imply more default risk than the market curve.
