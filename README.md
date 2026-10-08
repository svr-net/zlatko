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

## Standalone, copy-deployable build

`tools/standalone/build.mjs` turns `web/` into one folder of static files that runs anywhere you copy it.
- Open `index.html` straight from disk (`file://`), or upload the folder unchanged to any static host (S3, GitHub Pages, IIS, nginx...).
- There is no server code, no MIME setup and no build step on the target.
- The `.wasm` is embedded in the worker script, and every page is bundled into one classic script, so nothing is fetched at runtime.
- Asset names carry content hashes, so a host can cache them indefinitely.

```sh
docker build --target standalone-artifacts --output dist .     # -> dist/standalone/ + dist/zlatko-ccr-standalone.zip
# or locally (needs web/wasm built):
npm --prefix tools/standalone ci && node tools/standalone/build.mjs
```

```
dist/standalone/
  index.html, core.html, ... gpu.html      13 pages
  assets/app.<hash>.js                     all pages (classic script, ~85 KiB)
  assets/ccr-runtime.<hash>.js             WASM worker with embedded library (~360 KiB)
  assets/style.<hash>.css
  manifest.json, README.txt                build commit, file sizes and SHA-256
```

The whole site is about 460 KiB (about 185 KiB zipped). `node web/tests/e2e.mjs --root dist/standalone --file` opens every page from
disk. It checks that the pages make no network request at all, and that the WebGPU kernels still agree with WASM.

## Docker

The multi-stage `Dockerfile` builds everything from source. Each stage runs its tests, so a successful build means they passed.

```sh
docker build -t zlatko-ccr .                       # WASM from source + checks -> standalone site on nginx (default)
docker run --rm -p 8080:80 zlatko-ccr              # web front end at http://localhost:8080

docker build --target native -t zlatko-ccr:native .   # native C++ build, ctest
docker run --rm zlatko-ccr:native                     # end-to-end demo

docker build --target e2e .                        # every page in headless Chromium, WebGPU on SwiftShader:
                                                   # dev site over HTTP, standalone from file:// and over HTTP
docker build --target standalone-artifacts --output dist .  # export the standalone site and zip
docker build --target wasm-artifacts --output web/wasm .   # export ccr.js / ccr.wasm to the host
```

| Target | Base image | Contents |
|---|---|---|
| `native` | `ubuntu:24.04` | Library, unit tests and demo binary; installed headers and CMake package in `/opt/ccr` |
| `wasm` | `emscripten/emsdk:4.0.10` | Embind module built from source and checked with `web/tests/wasm.test.mjs` |
| `wasm-artifacts` | `scratch` | Only `ccr.js` and `ccr.wasm`, for `--output` |
| `standalone` | `node:22-alpine` | Copy-deployable site built with the pinned esbuild |
| `standalone-artifacts` | `scratch` | `standalone/` folder and `zlatko-ccr-standalone.zip`, for `--output` |
| `e2e` | `mcr.microsoft.com/playwright` | Browser test of all pages in three modes, including GPU-vs-WASM agreement |
| `web` (default) | `nginx:1.27-alpine` | The standalone site, about 75 MB; hashed assets are cached as immutable |

If the network goes through a TLS-intercepting proxy, give the npm stages the proxy's CA so they can reach the registry:
`docker build --secret id=ca,src=/path/to/ca.pem ...`. Pass `--build-arg GIT_COMMIT=$(git rev-parse --short HEAD)` to stamp the commit into `manifest.json`.

## Web front end (WebAssembly + WebGPU)

`web/` contains one interactive page per library context. Each page runs the C++ library compiled to WebAssembly,
and the exposure pipeline also runs as fused WebGPU compute kernels.

```sh
python3 -m http.server -d web 8000      # any static server; file:// will not load WASM or module workers
# open http://localhost:8000
```

| Page | Context | Entry point |
|---|---|---|
| `core.html` | Cholesky correlation, regression, time grids, Brent, quantiles | `coreDemo` |
| `market.html` | Yield curve, CDS pricing, hazard bootstrapping | `marketDemo` |
| `models.html` | Hull–White, FX/equity, CIR++ fan charts and martingale tests | `simulate` |
| `instruments.html` | Swaps, forwards, options, Bermudans on scenarios | `priceTrades` |
| `amc.html` | Longstaff–Schwartz policy, physical vs cash exposure | `amc` |
| `exposure.html` | EE/ENE/PFE/EPE/EEPE, netting benefit, distribution | `exposure` |
| `collateral.html` | Thresholds, MTA, independent amount, margin period of risk | `exposure` |
| `allocation.html` | Marginal (Euler), incremental and standalone CVA | `allocation` |
| `cva.html` | Unilateral/bilateral CVA, term structure, running spread | `cva` |
| `wwr.html` | Pathwise CVA against exposure–intensity correlation | `wrongWayRisk` |
| `hedging.html` | CS01 buckets, CDS hedge, CRN vs independent-seed deltas | `hedging` |
| `gpu.html` | Fused WebGPU kernels validated against WASM, with a benchmark | `gpuPlan` + `web/js/gpu` |

The pages share one specification: market, models, correlation, portfolio, CSA and simulation settings. You edit it on any
page, and the browser's local storage keeps it. The WASM module runs in a module worker, so long Monte Carlo runs don't block the page.

**WebGPU fused kernels.** `gpuPlan(spec)` uses the library's own models to compile the netting set into flat tables:
- exact Hull–White step coefficients;
- per-date bond terms `A·e^(−Bx)`, fixing records and option terms;
- the margin-call look-back;
- the Cholesky factor and the CIR++ shift.

Three compute kernels then run on these tables:
1. **fused-exposure** runs once per path. It draws counter-based normals, correlates them, steps rates, FX and CIR++,
   revalues the netting set, applies the CSA and reduces EE/ENE/EE*/pathwise-CVA across the workgroup. It never builds a scenario cube.
2. **reduce-partials** sums the workgroup results.
3. **pfe-quantile** builds a histogram per date and reads off the quantile.

The GPU works in f32 with its own random number generator, so it agrees with WASM within Monte Carlo error. Bermudans need AMC regression and stay in WASM.

**Rebuilding the WASM module.** The built `web/wasm/ccr.{js,wasm}` is committed. Rebuild it with Docker (`--target wasm-artifacts` above), or with a local Emscripten:

```sh
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm                 # writes web/wasm/ccr.js and ccr.wasm
node web/tests/wasm.test.mjs             # every entry point plus invariant checks, in Node
node web/tests/e2e.mjs                   # every page in headless Chromium (Playwright; WebGPU via SwiftShader)
```

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
