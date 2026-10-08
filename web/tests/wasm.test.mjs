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

run('version', () => ccr.version());
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
if (cva) check(cva.bilateral && cva.bilateral.dva > 0 && cva.cva > 0, 'CVA and DVA positive');
// The analytics applied to a profile from elsewhere (the WebGPU kernels) must match exactly.
const fromProfile = cva && run('cvaFromProfile', () => ccr.cvaFromProfile({ ...spec, profile: cva.profile }));
if (fromProfile) {
  near(fromProfile.cva, cva.cva, 1e-9 * cva.cva, 'cvaFromProfile CVA');
  near(fromProfile.bilateral.dva, cva.bilateral.dva, 1e-9 * cva.bilateral.dva, 'cvaFromProfile DVA');
  near(fromProfile.runningSpread, cva.runningSpread, 1e-15, 'cvaFromProfile running spread');
}
check(typeof ccr.cvaFromProfile({ ...spec, profile: { times: [0, 1], ee: [0], ene: [0], discountedEe: [0], discountedEne: [0] } }).error === 'string',
  'cvaFromProfile rejects mismatched arrays');
const wwr = run('wrongWayRisk', () => ccr.wrongWayRisk({ ...spec, sim: { ...spec.sim, numPaths: 1000 }, rhos: [-0.5, 0, 0.5] }));
if (wwr) check(wwr.results[2].pathwiseCva > wwr.results[0].pathwiseCva, 'WWR increases CVA');
const hedge = run('hedging', () => ccr.hedging({ ...spec, sim: { ...spec.sim, numPaths: 500 }, bumps: [0.02, 0.01] }));
if (hedge) {
  const sd = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0) / a.length);
  check(sd(hedge.pnlHedged) < 0.1 * sd(hedge.pnlUnhedged), 'CDS hedge removes most spread P&L');
  // The credit part from a profile simulated elsewhere must match hedging() on the same paths.
  const base = ccr.exposure({ ...spec, sim: { ...spec.sim, numPaths: 500 } });
  const credit = run('creditHedgingFromProfile', () => ccr.creditHedgingFromProfile({ ...spec, sim: { ...spec.sim, numPaths: 500 }, profile: base.profile }));
  if (credit) {
    near(credit.cva, hedge.cva, 1e-9 * hedge.cva, 'creditHedgingFromProfile CVA');
    hedge.cs01.forEach((v, k) => near(credit.cs01[k], v, 1e-9 * Math.abs(v) + 1e-9, `creditHedgingFromProfile CS01[${k}]`));
    near(credit.pnlHedged[7], hedge.pnlHedged[7], 1e-6 * Math.abs(hedge.pnlUnhedged[7]), 'creditHedgingFromProfile hedged P&L');
  }
}
const plan = run('gpuPlan', () => ccr.gpuPlan({ ...spec, csa: { ...spec.csa, enabled: true } }));
if (plan) check(plan.numTerms > 0 && plan.termStart.length === plan.times.length + 1, 'GPU plan tables');
const bad = ccr.exposure({ ...spec, trades: [{ type: 'nonsense' }] });
check(typeof bad.error === 'string', 'errors are reported, not thrown');

console.log(failures ? `\n${failures} failure(s)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
