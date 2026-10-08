import { barChart, fmt, lineChart, scatterChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'hedging',
  title: 'Hedging CVA',
  context: 'ccr/hedging: creditSpreadSensitivities · cdsHedgeJacobian · cdsHedgeNotionals · centralDifference',
  description: 'Credit risk: bump each CDS quote by 1bp, re-bootstrap the curve and reprice CVA to get bucketed CS01s. Then solve for the CDS notionals whose ' +
    'sensitivities offset them. Market risk: bump-and-revalue the whole Monte Carlo. Re-using the same random numbers for the up and down runs ' +
    '(common random numbers) is what makes the finite-difference delta stable as the bump shrinks.',
});
specEditor(page, ['market', 'assets', 'credit', 'simulation', 'portfolio', 'csa']);

const button = runButton(page, 'Compute hedges', async (spec) => {
  const t0 = performance.now();
  // Every delta is a pair of full Monte Carlo runs per bump, so the path count is capped.
  const sim = { ...spec.sim, numPaths: Math.min(spec.sim.numPaths, 1500) };
  const bumps = [0.04, 0.02, 0.01, 0.005, 0.0025];
  const runResult = await runAnalysis('hedging', { ...spec, sim, bumps });
  const r = runResult.result;
  window.__ccrEngineRun = { page: 'hedging', engine: runResult.engine, cva: r.cva, cvaDv01: r.cvaDv01, parallelCs01: r.parallelCs01, deltaCrn: r.assetDeltas.map((a) => Array.from(a.deltaCrn)) };
  const pnlU = Array.from(r.pnlUnhedged), pnlH = Array.from(r.pnlHedged);
  tiles(page.content, [
    { label: 'CVA', value: fmt.num(r.cva) },
    { label: 'Parallel CS01 (per 1bp)', value: fmt.num(r.parallelCs01, 1) },
    { label: 'CVA DV01 (rates +1bp)', value: fmt.num(r.cvaDv01, 1), hint: 'common random numbers' },
    { label: 'Spread P&L vol, hedged / unhedged', value: fmt.pct(r.pnlVolRatio, 2), hint: '60 random spread scenarios' },
  ]);
  const g = grid(page.content);
  const labels = Array.from(r.maturities).map((m) => fmt.years(m));
  barChart(card(g, 'Bucketed CS01 of CVA', 'Change in CVA for a 1bp move in each CDS quote. A short bucket can be negative, because re-bootstrapping lowers the hazard rate just after it.'), {
    labels, series: [{ name: 'CS01', values: Array.from(r.cs01) }], tooltipFormat: (v) => fmt.num(v, 2),
  });
  barChart(card(g, 'CDS hedge notionals', 'Protection to buy (+) or sell (−) at each tenor so that the hedge offsets every CS01 bucket.'), {
    labels, series: [{ name: 'CDS notional', values: Array.from(r.hedgeNotionals), colorIndex: 1 }],
  });
  scatterChart(card(g, 'Hedge effectiveness under spread shocks', 'Our P&L (−ΔCVA) unhedged against hedged, for random curve moves. Hedged points should sit close to zero.'), {
    series: [{ name: 'scenario', x: pnlU, y: pnlH }], xLabel: 'unhedged P&L', yLabel: 'hedged P&L',
  });
  for (const a of r.assetDeltas) {
    const bumps = Array.from(a.bumps);
    lineChart(card(g, `${a.name} delta of CVA vs bump size`, 'With common random numbers the delta converges as the bump shrinks. With independent seeds it is swamped by noise that grows like 1/h.'), {
      x: bumps, xLabel: 'bump h', xFormat: (v) => v.toFixed(4), zero: true,
      series: [{ name: 'common random numbers', y: Array.from(a.deltaCrn), markers: true }, { name: 'independent seeds', y: Array.from(a.deltaIndependent), markers: true, colorIndex: 1 }],
    });
    const box = card(g, `${a.name} hedge`, '');
    table(box, ['quantity', 'value'], [
      [`CVA delta (CRN, h = ${a.hedgeBump})`, fmt.num(a.hedgeDelta, 0)],
      ['hedge', `${a.hedgeUnits > 0 ? 'buy' : 'sell'} ${fmt.num(Math.abs(a.hedgeUnits))} units forward (horizon)`],
    ]);
  }
  const box = card(g, 'Hedge Jacobian', 'Change in value of a unit-notional par CDS (rows) for a 1bp move in each quote (columns).');
  const j = r.jacobian;
  table(box, ['CDS \\ quote', ...labels], labels.map((l, i) => [l, ...labels.map((_, k) => j.data[i * j.cols + k].toExponential(2))]));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
