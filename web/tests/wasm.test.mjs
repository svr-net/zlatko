// Node test of the WebAssembly build: every entry point through ccr::app::call, and every Monte
// Carlo analysis on the fused kernels (cpuKernels: the CPU build of the kernels; no GPU in Node),
// with checks against exact values and the library's invariants. The kernels themselves are
// validated against the double-precision ExposureEngine by the native tests.
//   node web/tests/wasm.test.mjs
import createCcrModule from '../wasm/ccr.js';
import { defaultSpec, defaultBermudan } from '../js/spec.js';

const mod = await createCcrModule();
const call = (name, spec = {}) => mod.call(name, spec);
let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL', msg); } };
const near = (a, b, tol, msg) => check(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);
const spec = { ...defaultSpec(), sim: { numPaths: 4000, seed: 2024, antithetic: true } };
const sum = (xs) => Array.from(xs).reduce((s, x) => s + x, 0);

function run(name, fn) {
  const t0 = performance.now();
  const r = fn();
  if (r && r.error) { failures++; console.log(`FAIL ${name}: ${r.error}`); return null; }
  console.log(`ok   ${name} (${(performance.now() - t0).toFixed(0)} ms)`);
  return r;
}
const fused = (analysis, extra = {}) => run(`cpuKernels ${analysis}`, () => call('cpuKernels', { ...spec, analysis, ...extra }));

check(Array.from(mod.functions()).join() === 'version,coreDemo,marketDemo,gpuKernels,gpuJobs,gpuAnalyse,cpuKernels,surrogateBacktest', 'entry points');
const ver = run('version', () => call('version'));
const cmakeVersion = (await import('node:fs')).readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8').match(/^\s*VERSION ([0-9.]+)$/m)[1];
if (ver) check(ver.library === `zlatko ccr ${cmakeVersion}`, `version() reports the CMake project version: ${ver.library}`);
const core = run('coreDemo', () => call('coreDemo', { ...spec, regression: { points: 300, noise: 0.2 } }));
if (core) {
  near(core.brent.impliedVol, core.brent.trueVol, 1e-10, 'Brent implied vol');
  near(core.realisedCorrelation.data[1], 0.2, 0.03, 'realised correlation');
}
const market = run('marketDemo', () => call('marketDemo', spec));
if (market) for (const q of market.credits[0].quotes) near(q.parSpread, q.quote, 1e-10, 'CDS reprices');

// Every Monte Carlo analysis on the fused kernels.
const expo = fused('exposure', { samplePaths: 12, histogramTime: 2 });
const expoCsa = fused('exposure', { csa: { ...spec.csa, enabled: true }, samplePaths: 12 });
if (expo && expoCsa) {
  check(expo.exposureSamples.length === 12 && expo.exposureSamples[0].length === expo.profile.times.length, 'exposure sample paths');
  near(sum(expo.histogram.shares), 1, 1e-6, 'histogram shares');
  check(expo.nettingBenefit > 0 && expo.nettingBenefit < 1 && expo.trades.length === 4, 'netting benefit and per-trade profiles');
  check(expoCsa.profile.eepe1y < expo.profile.eepe1y && expoCsa.collateralBands.length === 3, 'CSA reduces EEPE; collateral bands');
}
const coll = fused('collateral', { samplePaths: 12 });
if (coll) {
  check(coll.variants.length === 5 && coll.variants[0].eepeVsNoCsa === 1, 'CSA variants');
  check(coll.variants[2].profile.eepe1y < coll.variants[0].profile.eepe1y && coll.nettedSamples.length === 12, 'zero-threshold CSA; samples');
}
const alloc = fused('allocation');
if (alloc) near(alloc.marginalCvaSum, alloc.totalCva, 1e-4 * alloc.totalCva, 'Euler allocation adds up (f32)');
const cva = fused('cva');
if (cva) {
  check(cva.bilateral && cva.bilateral.dva > 0 && cva.cva > 0, 'CVA and DVA positive');
  near(cva.cumulativeCva[cva.cumulativeCva.length - 1], cva.cva, 1e-9 * cva.cva, 'cumulative CVA ends at CVA');
}
const xva = fused('xva');
if (xva) {
  const x = xva.xva;
  near(x.total, x.cva - x.dva + x.fca - x.fba + x.mva + x.kva, 1e-6 * Math.abs(x.total), 'XVA identity');
  near(sum(xva.trades.map((t) => t.cva)), x.cva, 1e-4 * x.cva, 'XVA allocation adds up (f32)');
}
const wwr = fused('wrongWayRisk', { rhos: [-0.5, 0, 0.5] });
if (wwr) check(wwr.results[2].pathwiseCva > wwr.results[0].pathwiseCva, 'WWR increases CVA');
const hedge = fused('hedging', { sim: { ...spec.sim, numPaths: 1000 }, bumps: [0.02, 0.01] });
if (hedge) check(hedge.pnlVolRatio < 0.1, 'CDS hedge removes most spread P&L');
const inst = fused('instruments', { samplePaths: 5, trades: [...spec.trades, { type: 'bermudan', id: 'B', ...defaultBermudan() }] });
if (inst) for (const t of inst.trades) check(t.pass && t.samples.length === 5 && t.bands.length === 3, `${t.id}: t = 0 value ${t.value0} vs ${t.benchmarkLabel} ${t.benchmark}`);
const models = fused('models', { samplePaths: 5 });
if (models) {
  for (const c of models.checks) check(c.pass, `${c.name}: ${c.value} vs ${c.exact}`);
  near(models.realisedCorrelation.data[1], 0.2, 0.05, 'realised rate/FX correlation');
}
const amc = fused('amc', { samplePaths: 5, bermudan: defaultBermudan() });
if (amc) {
  check(amc.price >= 0.98 * amc.bestEuropean, `Bermudan ${amc.price} >= max European ${amc.bestEuropean}`);
  near(sum(amc.exerciseDistribution), 1, 1e-6, 'exercise distribution');
  check(amc.degreeSensitivity.length === 3 && Math.abs(amc.degreeSensitivity[1].vsHighest) < 0.02, 'price stable in the regression degree');
}
const valid = fused('validation');
if (valid) {
  const c = valid.checks;
  check(c.t0Diff < 1e-5 && c.eeDiff === 0 && c.survDiff < 0.01, `validation checks ${JSON.stringify(c)}`);
}

// The GPU entry points: plans for every job, read-backs checked against the plan.
const jobs = run('gpuJobs', () => call('gpuJobs', { ...spec, analysis: 'hedging', bumps: [0.02, 0.01] }));
if (jobs) {
  check(jobs.plans.length === 1 + 4 * 2 + 2, 'hedging plans: base, 4 per bump, 2 rate shifts');
  const p = jobs.plans[0];
  check(p.header instanceof Uint32Array && p.requests instanceof Float32Array && p.header[1] === p.numDates, 'plan buffers');
}
const kernels = run('gpuKernels', () => call('gpuKernels'));
if (kernels) check(kernels.fusedExposure.includes('fn main') && kernels.fusedExposureTrades.includes('MAX_TRADES : u32 = 16u'), 'WGSL sources');
const badOut = call('gpuAnalyse', { ...spec, analysis: 'exposure', gpuOutputs: [{ sums: new Float32Array(3), values: new Float32Array(1) }] });
check(typeof badOut.error === 'string', 'gpuAnalyse rejects read-backs that do not match the plan');
check(typeof call('cpuKernels', { ...spec, analysis: 'exposure', trades: [{ type: 'nonsense' }] }).error === 'string', 'errors are reported, not thrown');
check(typeof call('nonsense').error === 'string', 'unknown entry points are reported');

console.log(failures ? `\n${failures} failure(s)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
