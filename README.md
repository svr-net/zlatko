# zlatko — counterparty credit exposure library

[![CI](https://github.com/svr-net/zlatko/actions/workflows/ci.yml/badge.svg)](https://github.com/svr-net/zlatko/actions/workflows/ci.yml)

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

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main`, on pull requests and on manual dispatch:

| Job | What it checks |
|---|---|
| `native (gcc)`, `native (clang)` | Release build with warnings as errors, `ctest`, example program |
| `native (ASan + UBSan)` | Debug build with address and undefined-behaviour sanitizers, `ctest` |
| `wasm` | Embind module built in `emscripten/emsdk:4.0.10`. Runs the Node checks of every entry point, and fails if the committed `web/wasm/` is out of date |
| `standalone` | Copy-deployable site and zip, uploaded as the `zlatko-ccr-standalone` artifact |
| `e2e` | Every page in headless Chromium (WebGPU on SwiftShader): dev site over HTTP, standalone from `file://` and over HTTP. Screenshots are uploaded as an artifact |
| `docker` | `native` and `web` images with BuildKit layer caching, plus a smoke test of the nginx image |

GitHub Pages: `.github/workflows/pages.yml` deploys the standalone site after CI passes on every push to `main`. Pages must be enabled once under *Settings → Pages* with **Source: GitHub Actions**.

Releases: push a tag that matches the CMake project version (`git tag v0.1.0 && git push origin v0.1.0`), or run the **Release** workflow manually on `main`, which creates that tag itself. `.github/workflows/release.yml` runs the full CI, then publishes a GitHub release with `zlatko-ccr-standalone-<tag>.zip` and its SHA-256.

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
  index.html, core.html, ... xva.html      15 pages
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

Every Monte Carlo page runs one analysis of the library (`spec.analysis`) on the fused kernels:

| Page | Context | Analysis / entry point |
|---|---|---|
| `core.html` | Cholesky correlation, regression, time grids, Brent, quantiles | `coreDemo` |
| `market.html` | Yield curve, CDS pricing, hazard bootstrapping | `marketDemo` |
| `models.html` | Hull–White, FX/equity, CIR++ fan charts and martingale tests (probe rows of the kernel) | `models` |
| `instruments.html` | Swaps, forwards, options, Bermudans on scenarios, against their analytic prices | `instruments` |
| `amc.html` | Longstaff–Schwartz policy inside the kernel, physical vs cash exposure | `amc` |
| `exposure.html` | EE/ENE/PFE/EPE/EEPE, sample paths, netting benefit, per-trade profiles, distribution | `exposure` |
| `collateral.html` | Thresholds, MTA, independent amount, margin period of risk | `collateral` |
| `allocation.html` | Marginal (Euler), incremental and standalone CVA | `allocation` |
| `cva.html` | Unilateral/bilateral CVA, term structure, running spread | `cva` |
| `wwr.html` | Pathwise CVA against exposure–intensity correlation | `wrongWayRisk` |
| `xva.html` | CVA, DVA, FVA, MVA and KVA with cumulative profiles, sweeps, allocation and sensitivities | `xva` |
| `hedging.html` | CS01 buckets, CDS hedge, CRN vs independent-seed deltas, XVA credit and funding sensitivities | `hedging` |
| `ai.html` | Surrogate XVA (ridge, MLP) fitted and back tested in C++, and a provider-neutral LLM agent skill (OpenAI, Anthropic, Gemini, MCP) whose tools call `xva` / `xvaCompare` | `surrogateBacktest` · dataset from `xva_ai_dataset` |
| `gpu.html` | Fused WebGPU kernels validated against the same kernels in C++, with a benchmark | `validation` |

The pages share one specification: market, models, correlation, portfolio, CSA and simulation settings. You edit it on any
page, and the browser's local storage keeps it. The WASM module runs in a module worker, so long Monte Carlo runs don't block the page.

**Compute engine.** Every Monte Carlo page has an *Engine* selector:
- **Auto** (the default) uses the fused **WebGPU** kernels wherever the browser offers a WebGPU adapter, on desktop and mobile alike. On Android it also tries a WebGPU compatibility-mode adapter.
- Where there is no adapter, for example where Chrome blocks WebGPU for the GPU or driver, Auto runs the **same fused kernels on the CPU**: one call into C++ (`cpuKernels`), which compiles, runs and analyses them in the worker's WebAssembly instance. The result agrees with WebGPU to about 0.005% (the same algorithm and random numbers in f32). The WebAssembly build is single-threaded, so the standalone zip still opens from `file://`; natively the CPU kernels run on every core.
- **WebGPU** or **CPU fused kernels** forces one engine. The status line names the engine that ran and why.
- `ExposureEngine` (double precision, its own random numbers) is not part of the front end: it is the native reference the unit tests check the kernels against.

**All numerics are in C++, on the fused kernels.** The analyses live in the library (`src/app/analyses.cpp`, on the JSON-like `ccr::app::Value`), so they build and test natively; the WebAssembly binding (`wasm/bindings.cpp`, about 100 lines) only converts values and exposes `call(name, spec)`. Each Monte Carlo analysis is a list of jobs (one netting set under one model set-up each, with the views it needs from the kernels: `gpu::FusedOptions`) and a combine step, which also computes everything the pages show (checks, sums, shares, histograms). The pages only render. On WebGPU:

1. `gpuJobs(spec)` compiles each job with `ccr::gpu::compile` into a plan of ready-made GPU buffers, dispatch sizes and buffer sizes.
2. `web/js/gpu/engine.js` uploads each plan, dispatches the three kernels and reads back two arrays. The kernel sources come from the library (`gpuKernels`).
3. `gpuAnalyse(spec)` turns the read-backs into profiles (`ccr::gpu::summarise`) and runs the same combine step as the CPU path.

The JavaScript layer only routes calls and talks to WebGPU (`web/js/gpu`), and one Web Worker hosts the library (`web/js/ccr-client.js`); the pages only render results. `cpuKernels(spec)` runs the kernels on the CPU instead, which is also how Node tests every analysis (`web/tests/wasm.test.mjs`).

**AI chapter.** The surrogates (ridge regression of degree 1 and 2, a tanh MLP trained with Adam on the thread pool) and their back test are C++ (`ccr::ai`, `include/ccr/ai/surrogate.hpp`). The agent skill's market state (spread multipliers, rate shift, FX vol, funding spread) is applied by the library (`spec.marketState`), and `xvaCompare` prices a base and a shocked market as two jobs. The dataset (`examples/xva_ai_dataset.cpp`) is priced on the fused kernels too: 410 XVA runs in about 5 s natively (12.7 ms each, was 200 ms on `ExposureEngine`), reproducible on every platform.

**WebGPU fused kernels** (`include/ccr/gpu/fused_exposure.hpp`). `compile()` uses the library's own models to turn the netting set into flat tables:
- exact Hull–White step coefficients;
- per-date bond terms `A·e^(−Bx)`, fixing records and option terms. Terms are merged per cash-flow date in double precision: bonds of one maturity become a single term, a running coupon carries its fixed leg, forwards on one asset are summed and a fixing is one bond ratio. The kernel evaluates one exponential per distinct date instead of one per cash flow, and offsetting flows cancel before the f32 cast (a 10-swap book drops from 31,849 to 2,237 terms);
- the margin-call look-back;
- the Cholesky factor and the CIR++ shift.

Three WGSL compute kernels (`src/gpu/fused_kernels.cpp`) then run on these tables:
1. **fused-exposure** runs once per path. It draws counter-based normals, correlates them, steps rates, FX and CIR++,
   revalues the netting set, applies the CSA and reduces E[V−C]/EE/ENE/EE*/ENE*/pathwise-CVA/E[Q] across the workgroup. It never builds a scenario cube: it writes the netted value only when a CSA needs the margin-call look-back, and the positive exposure only on the reporting dates. The discount factor and the asset levels are accumulated as sums of logs, so their f32 error does not compound date by date.
   - *Vector unrolling:* normals come in Box–Muller pairs packed four to a `vec4` (one hash pair, one `log` and one `sqrt` per two normals). Each Cholesky row is two `vec4` dot products. The four FX/equity factors step as one `vec4`, and the statistics reduce as two `vec4`s per invocation.
   - *Matrix tiling:* inputs that every path reads are staged once per workgroup in shared memory instead of being fetched by each of the 64 invocations. That covers the Cholesky factor, each date's step row and the valuation-term table. The term table of a date (terms × 8) streams through in tiles of 64 terms, loaded cooperatively as `vec4`s, and each invocation applies the tile to its own path state.
2. **reduce-partials** sums the workgroup results.
   - *XVA and allocation:* the `xva` and `allocation` plans also ask for the quantile of the value change between reporting dates (the initial-margin proxy behind MVA) and for per-trade statistics. Each term then carries its trade (terms are merged within a trade only), a per-trade build of the kernel tracks every trade's value on the path, and five statistics per trade are reduced with the netting set's: the Euler contributions to EE* and EE, the incremental EE* and the standalone EE* and EE, for up to 16 trades. MVA, the allocation of CVA, FCA and KVA, and the incremental and standalone CVAs therefore run on WebGPU too, and so do the AI page's agent tools.
   - *American Monte Carlo:* Bermudan swaptions are priced by Longstaff–Schwartz in two passes of the same kernel. `compile()` first runs the kernel on the CPU (on the thread pool) to record each path's Hull–White state, deflator and exercise values, fits the regressions backwards over the exercise dates in double precision exactly as `BermudanSwaption` does, and writes the standardisation and coefficients into the plan's policy and continuation terms. The kernel then decides exercise path by path (`kBermDecide`), values unexercised paths by the regressed continuation (`kBermContinue`), and exercised ones by the tail swap they entered (`kBermBond` / `kBermFloat`, physical settlement). Up to 4 Bermudans, 64 exercise dates and degree 3 per netting set; the kernel agrees with `ExposureEngine` within Monte Carlo error (price within 0.3% at 128,000 paths).
3. **row statistics** (`pfeQuantileKernel`) runs one workgroup per request on the rows the fused kernel wrote (per-path quantities on one date): an empirical quantile interpolated between order statistics as `ccr::quantile` does, found without sorting (a coarse histogram over [min, max], a fine histogram inside the order statistic's bin, then its neighbours; signed values via order-preserving keys), a mean, a cross moment, a histogram or sample paths. PFE (to about 5e-8 of the library's quantile), the initial-margin quantile and the path views are all requests: sample paths of V − C, V and C, collateral bands, the gross exposure Σ max(Vᵢ, 0) and its profile, the exposure histogram, per-trade value profiles, samples and bands, Bermudan exercise distributions, and the risk-factor views (state samples and bands, martingale tests of E[D], E[D·P(t,T)] and E[D·S], CIR++ survival, realised correlation of the first-step increments, written by probe terms). `FusedOptions` selects them; `FusedResult` returns them.

`runFusedReference()` (`src/gpu/fused_reference.cpp`) executes the same three kernels on the CPU in single precision, so the unit tests check the kernel algorithm natively against `ExposureEngine`. It is split into `runFusedWorkgroups()` (any range of workgroups) and `finishFused()` (the reduce and PFE kernels), and both run in parallel on `ccr::SpinPool` (`include/ccr/core/parallel.hpp`), a lock-free fork-join pool. Its workers wait for work by spinning on an atomic epoch (pause, yield, then short sleeps when idle for long), claim chunks of workgroups, statistics or quantile rows by compare-exchange on a shared cursor, and signal completion through an atomic counter the caller spins on; there is no mutex or condition variable, and a nested or concurrent loop that loses the compare-exchange on the busy flag runs serially instead of blocking. Every chunk writes its own outputs, so the result is bit-identical on any number of threads (tested, also under ThreadSanitizer). Natively the kernels scale 3.2× on 4 cores; a 32,768-path collateralised 10-swap book takes 0.47 s against 108 s for the double-precision `ExposureEngine`. `CCR_THREADS` sets the thread count. The GPU works in f32 with its own random number generator, so it agrees with the double-precision engine within Monte Carlo error.

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
| GPU exposure | `gpu/fused_exposure.hpp` | The exposure pipeline as fused WGSL kernels: plan compiler, kernel sources, read-back summary, and a CPU reference of the kernels |
| XVA | `xva/xva.hpp` | FCA, FBA (FVA), MVA from a simulated initial-margin proxy, KVA, the combined total, and exact allocation of CVA, FCA and KVA to trades |
| Hedging | `hedging/sensitivities.hpp` | XVA credit (counterparty and own) and funding sensitivities; bucketed CS01 of CVA with re-bootstrapped curves, CDS hedge notionals from the hedge Jacobian, and market-risk deltas by bump-and-revalue with common random numbers |

## Proposed extension chapters: XVA

The book stops at CVA/DVA, wrong-way risk and hedging. [`docs/xva_chapters.md`](docs/xva_chapters.md)
proposes ten chapters that extend it, in the same style (definition, formula, discretisation on the
exposure grid, Monte Carlo estimator, worked example, code and page), to cover everything implemented
in `xva/xva.hpp`, `hedging/sensitivities.hpp` and [`web/xva.html`](web/xva.html):

| Proposed chapter | Content | Code / page |
|---|---|---|
| X1. From CVA to XVA | Costs beyond credit; replication argument; total adjustment and sign conventions | `exposureXva`, [xva](web/xva.html) |
| X2. Funding valuation adjustment | FCA, FBA, FVA; the overlap with DVA and the FVA debate | `fundingBenefit` |
| X3. Margin valuation adjustment | Simulated initial margin proxy over the margin period of risk; MVA | `initialMarginProfile`, `marginValuationAdjustment` |
| X4. Capital valuation adjustment | KVA from the Basel exposure at default; simplifications | `xva` |
| X5. XVA allocation to trades | Exact Euler allocation of CVA, FCA, KVA; incremental XVA | `allocateExposureXva`, [allocation](web/allocation.html) |
| X6. XVA sensitivities and hedging | Bucketed CS01, CDS hedge notionals, own-credit paradox, common random numbers | [hedging](web/hedging.html) |
| X7. Numerical practice | Monte Carlo error, grid, regression, validation checks, engines | [gpu](web/gpu.html) |
| X8. Nested Monte Carlo in XVA | Inner/outer simulation, bias and the `B^{-2/3}` convergence, bias reduction, regression versus nesting, conditional initial margin, dual bounds, design | not implemented; builds on `core/regression.hpp`, [amc](web/amc.html) |
| X9. Beyond this implementation | Collateralised funding, SA-CCR capital, stochastic spreads, conditional IM, AAD | open |
| X10. Artificial intelligence and XVA | Surrogates, differential ML, generative scenarios, deep hedging, LLM agent skill, model risk, back test | [ai](web/ai.html) · `examples/xva_ai_dataset.cpp` · `web/js/ai/` |

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
