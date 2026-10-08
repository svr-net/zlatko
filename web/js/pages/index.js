import { run } from '../ccr-client.js';
import { fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, gpuExposure, runWithEngine } from '../gpu/backend.js';
import { PAGES, card, el, grid, initPage, runButton, tiles } from '../ui.js';

const page = initPage({
  id: 'index',
  title: 'Counterparty credit exposure, in the browser',
  context: 'zlatko · libccr (C++17) → WebAssembly + WebGPU',
  description:
    'Interactive front end for the library modelling <i>Modelling, Pricing, and Hedging Counterparty Credit Exposure</i> ' +
    '(Cesari, Aquilina, Charpillon, Filipović, Lee, Manda; Springer 2009). The C++ library is compiled to WebAssembly and runs in a worker. ' +
    'The exposure pipeline also runs as fused WebGPU compute kernels. Each page below exercises one context of the library on a shared, editable specification.',
});

const DESCRIPTIONS = {
  core: 'Cholesky correlation, seeded normals, least-squares regression, exposure time grids, Brent root finding, quantiles.',
  market: 'Yield-curve interpolation and forwards, CDS pricing and hazard-rate bootstrapping.',
  models: 'Hull–White rates with exact numeraire, log-normal FX/equity, CIR++ intensity. Fan charts and martingale tests.',
  instruments: 'Swaps (with path fixings), forwards, European options and Bermudans marked to market on every scenario.',
  amc: 'Longstaff–Schwartz exercise policy and regression-based exposure of a Bermudan swaption, physical vs cash settlement.',
  exposure: 'EE, ENE, PFE, EPE, effective EPE, netting benefit and the exposure distribution.',
  collateral: 'Thresholds, minimum transfer amount, independent amount and margin period of risk.',
  allocation: 'Marginal (Euler), incremental and standalone allocation of exposure and CVA.',
  cva: 'Unilateral and bilateral CVA/DVA, term structure, running spread, recovery sensitivity.',
  wwr: 'Pathwise CVA with a stochastic intensity correlated with the exposure drivers.',
  hedging: 'Bucketed CS01, CDS hedge notionals, delta with and without common random numbers.',
  gpu: 'The whole exposure pipeline fused into one WebGPU kernel, validated against WASM and benchmarked.',
};

const map = card(page.main, 'Contexts', 'Each page maps onto one part of the library and one topic of the book.');
const g = el('div', { class: 'grid' });
for (const group of PAGES.slice(1))
  for (const p of group.items)
    g.append(el('a', { href: p.href, class: 'tile', style: 'text-decoration:none;color:inherit' },
      el('div', { class: 'label', text: group.group }), el('div', { style: 'font-weight:650;font-size:15px', text: p.title }),
      el('div', { class: 'hint', text: DESCRIPTIONS[p.id] })));
map.append(g);
page.main.insertBefore(map.parentNode, page.toolbar);

const button = runButton(page, 'Run quick exposure', async (spec) => {
  const t0 = performance.now();
  const runResult = await runWithEngine(spec, {
    wasm: () => run('exposure', spec),
    gpu: async () => {
      const g = await gpuExposure(spec);
      return { profile: g.profile, numPaths: spec.sim.numPaths, simulationDates: g.plan.times.length, elapsedMs: g.gpuMs, adapter: g.adapter };
    },
  });
  const r = runResult.result;
  const onGpu = runResult.engine === 'gpu';
  const p = r.profile;
  window.__ccrEngineRun = { page: 'index', engine: runResult.engine, eepe1y: p.eepe1y };
  tiles(page.content, [
    { label: 'Effective EPE (1y)', value: fmt.compact(p.eepe1y) },
    { label: 'Peak PFE', value: fmt.compact(p.maxPfe) },
    onGpu
      ? { label: 'Netting benefit (EEPE)', value: '–', hint: 'WebAssembly engine only' }
      : { label: 'Netting benefit (EEPE)', value: fmt.pct(1 - p.eepe1y / r.grossProfile.eepe1y, 1), hint: 'vs sum of positive trade values' },
    { label: 'Paths × dates', value: `${fmt.num(r.numPaths)} × ${r.simulationDates}`, hint: `${fmt.num(r.elapsedMs)} ms on ${onGpu ? 'WebGPU' : 'WebAssembly'}` },
  ]);
  const gr = grid(page.content);
  lineChart(card(gr, 'Exposure profile of the default netting set', 'Full analysis on the <a href="exposure.html">exposure page</a>.'), {
    x: Array.from(p.times), xLabel: 't (years)', yLabel: 'exposure',
    series: [
      { name: 'EE', y: Array.from(p.ee) },
      { name: `PFE ${Math.round(p.pfeQuantile * 100)}%`, y: Array.from(p.pfe) },
      { name: 'ENE', y: Array.from(p.ene), colorIndex: 2 },
    ],
  });
  const arch = card(gr, 'How it runs', '');
  arch.append(el('pre', { class: 'code', text:
`C++17 library (include/ccr, src/)
   │  emcmake cmake -S . -B build-wasm   (Embind wrapper: wasm/bindings.cpp)
   ▼
web/wasm/ccr.{js,wasm}  ── run in a Worker (js/ccr-worker.js)
   │   standalone build: worker + .wasm embedded, started from a blob URL
   │   coreDemo · marketDemo · simulate · priceTrades · amc · exposure
   │   allocation · cva · wrongWayRisk · hedging · gpuPlan
   ▼
gpuPlan(spec) → flat tables (HW step coefficients, bond terms A·e^(−Bx),
                fixings, options, CSA look-back, Cholesky, CIR++ shift)
   ▼
WebGPU (js/gpu): fused-exposure ▸ reduce-partials ▸ pfe-quantile` }));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
