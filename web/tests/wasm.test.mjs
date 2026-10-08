// Node smoke test of the WASM wrapper: every entry point on the default spec,
// with sanity checks against the library's own invariants.
//   node web/tests/wasm.test.mjs
import createCcrModule from '../wasm/ccr.js';
import { defaultSpec, defaultBermudan } from '../js/spec.js';

const ccr = await createCcrModule();
let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL', msg); } };
const near = (a, b, tol, msg) => check(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);
const spec = defaultSpec();

function run(name, fn) {
  const t0 = performance.now();
  const r = fn();
  if (r && r.error) { failures++; console.log(`FAIL ${name}: ${r.error}`); return null; }
  console.log(`ok   ${name} (${(performance.now() - t0).toFixed(0)} ms)`);
  return r;
}

const ver = run('version', () => ccr.version());
const cmakeVersion = (await import('node:fs')).readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8').match(/^\s*VERSION ([0-9.]+)$/m)[1];
if (ver) check(ver.library === `zlatko ccr ${cmakeVersion}`, `version() reports the CMake project version: ${ver.library}`);
const core = run('coreDemo', () => ccr.coreDemo({ ...spec, regression: { points: 300, noise: 0.2 } }));
if (core) {
  near(core.brent.impliedVol, core.brent.trueVol, 1e-10, 'Brent implied vol');
  near(core.realisedCorrelation.data[1], 0.2, 0.03, 'realised correlation');
}
const market = run('marketDemo', () => ccr.marketDemo(spec));
if (market) for (const q of market.credits[0].quotes) near(q.parSpread, q.quote, 1e-10, 'CDS reprices');
const sim = run('simulate', () => ccr.simulate(spec));
if (sim) {
  const last = sim.times.length - 1;
  near(sim.rates.meanDeflator[last], sim.rates.curveDiscount[last], 5e-3, 'E[D] = P(0,T)');
  near(sim.credits[0].meanSurvival[last], sim.credits[0].marketSurvival[last], 5e-3, 'CIR++ fit');
}
const priced = run('priceTrades', () => ccr.priceTrades(spec));
if (priced) for (const t of priced.trades) near(t.value0, t.reference, 1e-6 * Math.max(1, Math.abs(t.reference)) + 1e-6, `${t.id} t0 value`);
const amc = run('amc', () => ccr.amc({ ...spec, sim: { ...spec.sim, numPaths: 1000 }, bermudan: defaultBermudan() }));
if (amc) check(amc.price >= Math.max(...amc.european) * 0.98, 'Bermudan >= max European');
const expo = run('exposure', () => ccr.exposure(spec));
const collateralised = run('exposure (CSA)', () => ccr.exposure({ ...spec, csa: { ...spec.csa, enabled: true } }));
if (expo && collateralised) check(collateralised.profile.eepe1y < expo.profile.eepe1y, 'CSA reduces EEPE');
const alloc = run('allocation', () => ccr.allocation(spec));
if (alloc) near(alloc.trades.reduce((s, t) => s + t.marginalCva, 0), alloc.totalCva, 1e-6 * alloc.totalCva, 'Euler allocation adds up');
const cva = run('cva', () => ccr.cva(spec));
if (cva) {
  check(cva.bilateral && cva.bilateral.dva > 0 && cva.cva > 0, 'CVA and DVA positive');
  near(cva.cumulativeCva[cva.cumulativeCva.length - 1], cva.cva, 1e-9 * cva.cva, 'cumulative CVA ends at CVA');
}
const coll = run('collateral', () => ccr.collateral({ ...spec, sim: { ...spec.sim, numPaths: 1000 } }));
if (coll) {
  check(coll.variants.length === 5 && coll.variants[0].eepeVsNoCsa === 1, 'CSA variants');
  check(coll.variants[2].profile.eepe1y < coll.variants[0].profile.eepe1y, 'zero-threshold CSA reduces EEPE');
}
const wwr = run('wrongWayRisk', () => ccr.wrongWayRisk({ ...spec, sim: { ...spec.sim, numPaths: 1000 }, rhos: [-0.5, 0, 0.5] }));
if (wwr) check(wwr.results[2].pathwiseCva > wwr.results[0].pathwiseCva, 'WWR increases CVA');
const hedgeSpec = { ...spec, sim: { ...spec.sim, numPaths: 500 }, bumps: [0.02, 0.01] };
const hedge = run('hedging', () => ccr.hedging(hedgeSpec));
if (hedge) check(hedge.pnlVolRatio < 0.1, 'CDS hedge removes most spread P&L');

// WebGPU path: plans and the combine step. No GPU in Node, so the kernels run through the
// C++ CPU reference (gpuEmulate); results must agree with the CPU library within MC error.
const kernels = run('gpuKernels', () => ccr.gpuKernels({}));
if (kernels) check(kernels.fusedExposure.includes('fn main') && kernels.pfeQuantile.includes('3072'), 'WGSL sources');
const jobs = run('gpuJobs', () => ccr.gpuJobs({ ...spec, analysis: 'hedging', bumps: [0.02, 0.01] }));
if (jobs) {
  check(jobs.plans.length === 1 + 4 * 2 + 2, 'hedging plans: base, 4 per bump, 2 rate shifts');
  const p = jobs.plans[0];
  check(p.header instanceof Uint32Array && p.params instanceof Float32Array && p.header[1] === p.numDates, 'plan buffers');
}
const berm = ccr.gpuJobs({ ...spec, analysis: 'exposure', trades: [{ type: 'bermudan', id: 'B', ...defaultBermudan() }] });
check(typeof berm.unsupported === 'string', 'Bermudan reported as unsupported on the GPU');
const pathTol = 4 / Math.sqrt(4000);
const pairSpec = { ...spec, sim: { ...spec.sim, numPaths: 4000 } };
const emuCva = run('gpuEmulate cva', () => ccr.gpuEmulate({ ...pairSpec, analysis: 'cva' }));
const cpuCva = ccr.cva(pairSpec);
if (emuCva) {
  near(emuCva.cva, cpuCva.cva, pathTol * cpuCva.cva, 'emulated kernels CVA vs CPU');
  near(emuCva.pathwiseCva, cpuCva.pathwiseCva, pathTol * cpuCva.pathwiseCva, 'emulated kernels pathwise CVA vs CPU');
}
const emuHedge = run('gpuEmulate hedging', () => ccr.gpuEmulate({ ...hedgeSpec, analysis: 'hedging' }));
if (emuHedge && hedge) {
  near(emuHedge.parallelCs01, hedge.parallelCs01, 0.2 * Math.abs(hedge.parallelCs01), 'emulated CS01 vs CPU');
  near(emuHedge.cvaDv01, hedge.cvaDv01, 0.2 * Math.abs(hedge.cvaDv01), 'emulated DV01 vs CPU');
}
const valid = run('gpuEmulate validation', () => ccr.gpuEmulate({ ...pairSpec, analysis: 'validation' }));
if (valid) {
  const c = valid.checks;
  check(c.t0Diff < 1e-6 && c.eeDiff < c.tol && c.cvaDiff < c.tol && c.survDiff < 0.01, `validation checks ${JSON.stringify(c)}`);
}
// gpuAnalyse with read-backs from the emulated kernels' shape: wrong sizes are rejected.
const badOut = ccr.gpuAnalyse({ ...spec, analysis: 'exposure', gpuOutputs: [{ sums: new Float32Array(3), pfe: new Float32Array(1) }] });
check(typeof badOut.error === 'string', 'gpuAnalyse rejects read-backs that do not match the plan');
const bad = ccr.exposure({ ...spec, trades: [{ type: 'nonsense' }] });
check(typeof bad.error === 'string', 'errors are reported, not thrown');

console.log(failures ? `\n${failures} failure(s)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
