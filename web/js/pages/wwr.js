import { run } from '../ccr-client.js';
import { fmt, lineChart } from '../charts.js';
import { factorNames } from '../spec.js';
import { engineNote, engineSelector, gpuExposure, runWithEngine } from '../gpu/backend.js';
import { card, el, grid, initPage, runButton, specEditor, table } from '../ui.js';

const page = initPage({
  id: 'wwr',
  title: 'Wrong-way risk',
  context: 'ccr/models: CirIntensity   ccr/cva: pathwiseCva',
  description: 'Exposure and default are not independent when the counterparty\'s intensity is correlated with the factors driving the exposure. ' +
    'The page sweeps the correlation between one market driver and the CIR++ intensity and prices CVA path by path. ' +
    'It also shows the discounted exposure <i>conditional on default</i> in each period. That conditional exposure sits above EE* under wrong-way risk and below it under right-way risk.',
});
specEditor(page, ['market', 'assets', 'credit', 'simulation', 'portfolio', 'csa']);

const names = factorNames(page.spec);
const driver = el('select', {}, names.slice(0, 1 + page.spec.assets.length).map((n, i) => el('option', { value: i, text: n, selected: i === Math.min(1, page.spec.assets.length) })));
page.toolbar.insertBefore(el('label', { class: 'status' }, 'driver correlated with intensity: ', driver), page.status);

// The same sweep on the WebGPU kernels: one fused run per correlation. The discounted exposure
// conditional on default in a period is the pathwise-CVA increment of that period divided by
// the simulated default probability E[Q_{k-1} - Q_k].
async function sweepOnGpu(spec, rhos, driverIndex) {
  const creditFactor = 1 + spec.assets.length + (spec.counterparty || 0);
  const results = [];
  let adapter;
  for (const rho of rhos) {
    const correlation = spec.correlation.map((row) => row.slice());
    correlation[driverIndex][creditFactor] = correlation[creditFactor][driverIndex] = rho;
    try {
      const g = await gpuExposure({ ...spec, correlation });
      adapter = g.adapter;
      const p = g.profile;
      const conditional = p.times.map((_, k) => {
        const pd = k ? p.meanSurvival[k - 1] - p.meanSurvival[k] : 0;
        return k && pd > 0 ? p.cvaIncrements[k] / pd : NaN;
      });
      results.push({ rho, pathwiseCva: g.pathwiseCva, independentCva: g.cva, times: p.times, discountedEe: p.discountedEe, conditionalDiscountedEe: conditional });
    } catch (e) {
      results.push({ rho, error: e.message });
    }
  }
  return { results, driver: driverIndex, adapter };
}

const button = runButton(page, 'Sweep correlation', async (spec) => {
  const t0 = performance.now();
  const rhos = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8];
  const runResult = await runWithEngine(spec, {
    wasm: () => run('wrongWayRisk', { ...spec, rhos, wwrFactor: Number(driver.value) }),
    gpu: () => sweepOnGpu(spec, rhos, Number(driver.value)),
  });
  const r = runResult.result;
  const ok = r.results.filter((x) => !x.error);
  if (!ok.length) throw new Error(r.results[0]?.error || 'no correlation could be simulated');
  window.__ccrEngineRun = { page: 'wwr', engine: runResult.engine, pathwiseCva: ok.map((x) => x.pathwiseCva), rhos: ok.map((x) => x.rho) };
  const g = grid(page.content);
  lineChart(card(g, 'CVA against correlation', `Correlation between <b>${names[r.driver]}</b> and the ${spec.credits[spec.counterparty || 0].name} intensity. Correlations that make the matrix non-positive-definite are skipped.`), {
    x: ok.map((x) => x.rho), xLabel: 'correlation ρ', xFormat: (v) => v.toFixed(1),
    series: [{ name: 'pathwise CVA', y: ok.map((x) => x.pathwiseCva), markers: true }, { name: 'CVA under independence', y: ok.map((x) => x.independentCva), dash: true, colorIndex: 1 }],
  });
  const lo = ok[0], mid = ok.find((x) => x.rho === 0) || ok[Math.floor(ok.length / 2)], hi = ok[ok.length - 1];
  const t = Array.from(mid.times);
  lineChart(card(g, 'Discounted exposure conditional on default', 'E[D·V⁺ | default in period] against the unconditional EE*.'), {
    x: t, xLabel: 't (years)',
    series: [
      { name: 'unconditional EE*', y: Array.from(mid.discountedEe), dash: true, colorIndex: 7 },
      { name: `conditional, ρ = ${lo.rho}`, y: Array.from(lo.conditionalDiscountedEe).map((v, i) => (i ? v : NaN)), colorIndex: 2 },
      { name: `conditional, ρ = ${mid.rho}`, y: Array.from(mid.conditionalDiscountedEe).map((v, i) => (i ? v : NaN)), colorIndex: 0 },
      { name: `conditional, ρ = ${hi.rho}`, y: Array.from(hi.conditionalDiscountedEe).map((v, i) => (i ? v : NaN)), colorIndex: 1 },
    ],
  });
  const box = card(g, 'Results', '');
  table(box, ['ρ', 'pathwise CVA', 'independence CVA', 'WWR multiplier'], r.results.map((x) => x.error
    ? [x.rho, x.error, '', '']
    : [x.rho, fmt.num(x.pathwiseCva), fmt.num(x.independentCva), (x.pathwiseCva / x.independentCva).toFixed(3)]));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
