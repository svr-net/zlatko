import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { saveSpec } from '../spec.js';
import { card, el, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

// Chapter X11: the exposure by adaptive nested Monte Carlo, on the CPU in the worker (the nest is not a
// fused-kernel job), against the closed form, the regression alone and the classical nest with the
// same inner paths. All numbers come from the library (nestedExposure); the page only draws them.
const page = initPage({
  id: 'nested',
  title: 'Adaptive nested Monte Carlo of the exposure',
  context: 'ccr/nested: estimateAdaptive · HermiteProxy · ScenarioEstimates · PositivePart · Quantile · nestedExposure · estimateUniform',
  description: 'One ordinary simulation is a nest with one inner path: every outer path already carries a noisy sample of V(t, path), its own discounted cash flows after t. ' +
    'Half of the outer paths fit a least-squares proxy of V on the Markov state at t (Hermite polynomials of the Hull–White factor, the log spots and the running fixings). ' +
    'An audit with inner paths in every bin of proxy values measures its bias and misspecification, and the rest of the inner budget goes only to scenarios whose sign of V (for EE*) or side of the quantile (for PFE) is uncertain enough to matter. ' +
    'Each scenario blends the proxy with its inner samples by precision, with two sample folds so that the choice of scenarios does not bias the values. ' +
    'The inner paths evolve exactly from the path\'s state, so the closed form on the same scenarios is the reference, against the regression alone and the classical nest that spreads the same inner paths evenly. ' +
    'Swaps, forwards and European options are nested; the netting set is taken uncollateralised. Chapter X11; the <a href="xva.html">XVA page</a> uses the same audit to remove the bias of the nested initial margin.',
});
specEditor(page, ['market', 'assets', 'credit', 'simulation', 'portfolio']);

const DEFAULTS = { dates: 10, numPaths: 1000, innerPerScenario: 60, degree: 3 };
page.spec.nested = { ...DEFAULTS, ...(page.spec.nested || {}) };
const cfg = page.spec.nested;
// Kept on page.spec (also after "Reset to defaults" replaces it) so the settings persist with the spec.
const store = () => { page.spec.nested = cfg; saveSpec(page.spec); };
const form = el('details', { class: 'spec', open: true }, el('summary', { text: 'Nested Monte Carlo' }));
const numField = (label, key, step = 1) => {
  const input = el('input', { type: 'number', step, min: 1, value: cfg[key] });
  input.addEventListener('change', () => { cfg[key] = Math.max(1, Math.round(Number(input.value) || DEFAULTS[key])); input.value = cfg[key]; store(); });
  return el('label', { class: 'field' }, el('span', { text: label }), input);
};
form.append(el('div', { class: 'spec-sections' }, el('div', {},
  numField('exposure dates', 'dates'), numField('outer paths', 'numPaths', 100),
  numField('inner paths per evaluated scenario (budget)', 'innerPerScenario'), numField('proxy degree (Hermite, total)', 'degree'))));
page.main.insertBefore(form, page.toolbar);

const METHODS = [
  ['exact', 'closed form', 3, true],
  ['regression', 'regression only', 2, false],
  ['uniform', 'classical nest, same inner paths', 1, false],
  ['adaptive', 'adaptive nest', 0, false],
];

runButton(page, 'Run nested Monte Carlo', async (spec) => {
  const n = await run('nestedExposure', { ...spec, nested: cfg });
  const t = Array.from(n.times);
  const err = n.rmsError;
  tiles(page.content, [
    { label: 'EE* error, adaptive', value: fmt.pct(err.adaptive.ee, 2), hint: `classical nest ${fmt.pct(err.uniform.ee, 2)} · regression ${fmt.pct(err.regression.ee, 2)}` },
    { label: `PFE ${fmt.pct(n.pfeQuantile, 0)} error, adaptive`, value: fmt.pct(err.adaptive.pfe, 2), hint: `classical nest ${fmt.pct(err.uniform.pfe, 2)} · regression ${fmt.pct(err.regression.pfe, 2)}` },
    ...(n.cva ? [{ label: 'CVA, adaptive', value: fmt.num(n.cva.adaptive), hint: `closed form ${fmt.num(n.cva.exact)} · classical nest ${fmt.num(n.cva.uniform)}` }] : []),
    { label: 'Inner paths', value: fmt.num(n.totalInnerPaths), hint: `at most ${fmt.num(n.innerBudgetPerDate)} per date · ${fmt.num(n.elapsedMs)} ms` },
  ]);

  const g = grid(page.content);
  const lines = (what) => METHODS.map(([key, name, colorIndex, dash]) => ({ name, y: Array.from(n[what][key]), colorIndex, dash }));
  const scope = `${fmt.num(n.outerPaths)} outer paths (half fit the proxy, ${fmt.num(n.evaluatedPerDate)} are evaluated per date), ` +
    `${fmt.num(n.innerBudgetPerDate)} inner paths per date at most` +
    (n.unsupported.length ? `; left out (no nested valuation): ${n.unsupported.join(', ')}` : '') + (n.csaIgnored ? '; the CSA is ignored (uncollateralised)' : '') + '.';
  lineChart(card(g, 'EE* by adaptive nested Monte Carlo', `V(t, path) is treated as unknown: a least-squares proxy on the outer paths' own discounted cash flows, an audit of its bias per bin, and inner simulations from a path's state only where the sign of V or its value is uncertain enough to move EE*. ${scope}`), {
    x: t, xLabel: 't (years)', zero: true, series: lines('ee'),
  });
  lineChart(card(g, `PFE ${fmt.pct(n.pfeQuantile, 0)} by adaptive nested Monte Carlo`, 'Scenarios that may lie on the wrong side of the quantile are refined; the closed form is the reference on the same scenarios.'), {
    x: t, xLabel: 't (years)', zero: true, series: lines('pfe'),
  });
  barChart(card(g, 'Error against the closed form', 'Relative root-mean-square error over the dates, on the same evaluation scenarios. The classical nest gets the adaptive run\'s inner paths spread evenly.'), {
    labels: ['EE*', 'PFE'],
    series: METHODS.slice(1).map(([key, name, colorIndex]) => ({ name, values: [err[key].ee, err[key].pfe], colorIndex })),
    yFormat: (v) => fmt.pct(v, 1), tooltipFormat: (v) => fmt.pct(v, 2),
  });
  lineChart(card(g, 'Where the inner paths went', 'Inner paths per date: the audit (random scenarios in every bin of proxy values) and the refinement on top of it.'), {
    x: t, xLabel: 't (years)', zero: true,
    series: [
      { name: 'inner paths, adaptive', y: Array.from(n.innerPaths), colorIndex: 0, markers: true },
      { name: 'of which audit', y: Array.from(n.auditInnerPaths), colorIndex: 2, dash: true },
    ],
  });
  const wide = grid(page.content, true);
  table(card(wide, 'Summary', `Adaptive run: ${fmt.num(n.totalInnerPaths)} inner paths in ${fmt.num(n.elapsedMs)} ms.`),
    ['method', 'EE* error', 'PFE error', ...(n.cva ? ['CVA'] : [])],
    METHODS.map(([key, name]) => [name, fmt.pct(err[key].ee, 2), fmt.pct(err[key].pfe, 2), ...(n.cva ? [fmt.num(n.cva[key])] : [])]));
  // Seven columns: full width so none is cut off.
  table(card(wide, 'Per date', 'Inner paths of the adaptive run and of its audit, the share of evaluation scenarios that got any, the largest bias of the proxy measured in a bin, the inner paths per scenario of the classical nest, and why the adaptive loop stopped.'),
    ['t', 'inner paths', 'audit', 'refined', 'max bin bias (rms)', 'classical nest per scenario', 'stop'],
    t.map((ti, k) => [fmt.years(ti), fmt.num(n.innerPaths[k]), fmt.num(n.auditInnerPaths[k]), fmt.pct(n.refinedShare[k], 1),
      fmt.num(n.maxBiasRms[k]), fmt.num(n.uniformInnerPerScenario[k]), n.stop[k]]));
  window.__ccrNestedRun = { page: 'nested', rmsError: err, totalInnerPaths: n.totalInnerPaths };
});
