import { barChart, fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { card, grid, initPage, runButton, specEditor, table, toArrays } from '../ui.js';

const page = initPage({
  id: 'collateral',
  title: 'Collateral: CSA terms and the margin period of risk',
  context: 'ccr/exposure: CollateralAgreement · simulateCollateral',
  description: 'The collateral held at t is the amount called on the portfolio value at <i>t − MPR</i>, the last call the defaulting counterparty met. ' +
    'Thresholds and the minimum transfer amount apply. The page compares the CSA you specify with variants. Margin-call dates are simulated as auxiliary grid dates, ' +
    'and exposure is reported only on the primary dates. The spikes on cash-flow dates are the known effect of collateral posted ' +
    'before a payment still being held during the margin period of risk. ' +
    'Collateral can be cash or securities: a government bond or the FX / equity asset is delivered in units at the call (the amount over (1 − haircut) times its price) ' +
    'and is worth its price at close-out, so its value moves over the margin period of risk and can fall just as the exposure rises (wrong-way collateral).',
});
specEditor(page, ['market', 'assets', 'simulation', 'portfolio', 'csa'], { open: true });

const button = runButton(page, 'Compare CSAs', async (spec) => {
  const t0 = performance.now();
  // The library defines the variants: no CSA, the specified CSA, zero thresholds and MTA,
  // a 20-day MPR and a one-way CSA.
  const runResult = await runAnalysis('collateral', { ...spec, samplePaths: 12 });
  const r = runResult.result;
  const v = r.variants;
  window.__ccrEngineRun = { page: 'collateral', engine: runResult.engine, eepe1y: v.map((x) => x.profile.eepe1y) };
  const t = Array.from(v[0].profile.times);
  const g = grid(page.content);
  lineChart(card(g, 'Expected exposure by CSA', ''), {
    x: t, xLabel: 't (years)', series: v.map((x, i) => ({ name: x.name, y: Array.from(x.profile.ee), colorIndex: i })),
  });
  lineChart(card(g, `PFE ${fmt.pct(spec.pfeQuantile, 0)} by CSA`, ''), {
    x: t, xLabel: 't (years)', series: v.map((x, i) => ({ name: x.name, y: Array.from(x.profile.pfe), colorIndex: i })),
  });
  if (r.nettedSamples) {
    lineChart(card(g, 'Specified CSA: value vs collateral held', 'Sample paths of the netted value (grey) and the median collateral held with its 5–95% band.'), {
      x: t, xLabel: 't (years)', zero: true, samples: toArrays(r.nettedSamples),
      series: [{ name: 'collateral median, 5–95%', y: Array.from(r.collateralBands[1]), band: { lower: Array.from(r.collateralBands[0]), upper: Array.from(r.collateralBands[2]) }, colorIndex: 1 }],
    });
  }
  barChart(card(g, 'Effective EPE (1y) and peak PFE', ''), {
    labels: v.map((x) => x.name),
    series: [{ name: 'EEPE (1y)', values: v.map((x) => x.profile.eepe1y) }, { name: 'peak PFE', values: v.map((x) => x.profile.maxPfe), colorIndex: 1 }],
  });
  const box = card(g, 'Summary', '');
  table(box, ['CSA', 'collateral', 'haircut', 'H cpty', 'H own', 'MTA', 'MPR (days)', 'EEPE 1y', 'peak PFE', 'EEPE vs no CSA'], v.map((x) => {
    const c = x.csa;
    return [x.name, c.enabled ? c.collateral : '–', c.enabled ? fmt.pct(c.haircut, 0) : '–', c.enabled ? fmt.compact(c.thresholdCounterparty) : '–',
      c.enabled ? (c.oneWay ? '∞' : fmt.compact(c.thresholdOwn)) : '–', c.enabled ? fmt.compact(c.mta) : '–', c.enabled ? fmt.num(c.mprDays) : '–',
      fmt.compact(x.profile.eepe1y), fmt.compact(x.profile.maxPfe), fmt.pct(x.eepeVsNoCsa, 1)];
  }));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
