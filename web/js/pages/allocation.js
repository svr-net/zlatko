import { barChart, fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { card, grid, initPage, passFail, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'allocation',
  title: 'Allocating exposure and CVA to trades',
  context: 'ccr/exposure: marginalExposureContributions · incrementalExposure · ccr/xva: allocateExposureXva',
  description: 'Netting makes the netting-set CVA smaller than the sum of the standalone CVAs, so it has to be split among the trades. The marginal (Euler) ' +
    'contribution E[D·V<sub>i</sub>·1{V &gt; 0}] adds up exactly to the total. The incremental CVA is the change from removing a trade, ' +
    'which is the right price for a new deal but does not add up. Hedging trades get negative contributions. ' +
    'On WebGPU (or the CPU kernels) the fused kernel tracks every trade\'s value on each path and accumulates these statistics directly, for up to 16 trades.',
});
specEditor(page, ['market', 'assets', 'credit', 'own', 'xva', 'simulation', 'portfolio']);

const button = runButton(page, 'Allocate', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('allocation', spec);
  const r = runResult.result;
  tiles(page.content, [
    { label: 'Netting-set CVA', value: fmt.num(r.totalCva) },
    { label: 'Σ marginal CVA', value: fmt.num(r.marginalCvaSum), hint: 'adds up' },
    { label: 'Σ standalone CVA', value: fmt.num(r.standaloneCvaSum), hint: `netting saves ${fmt.pct(r.nettingSaving, 1)}` },
    { label: 'Σ incremental CVA', value: fmt.num(r.incrementalCvaSum), hint: 'does not add up' },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'CVA by trade', ''), {
    labels: r.trades.map((t) => t.id),
    series: [
      { name: 'marginal (Euler)', values: r.trades.map((t) => t.marginalCva) },
      { name: 'incremental', values: r.trades.map((t) => t.incrementalCva) },
      { name: 'standalone', values: r.trades.map((t) => t.standaloneCva) },
    ],
  });
  const t = Array.from(r.times);
  lineChart(card(g, 'Marginal EE contributions over time', 'The contributions sum to the netting-set EE (dashed).'), {
    x: t, xLabel: 't (years)', zero: true,
    series: [...r.trades.map((tr, i) => ({ name: tr.id, y: Array.from(tr.marginalEe), colorIndex: i % 8 })), { name: 'netting-set EE', y: Array.from(r.totalEe), dash: true, colorIndex: 7 }],
  });
  const box = card(g, 'Allocation table', '');
  table(box, ['trade', 'marginal', 'share', 'incremental', 'standalone'], [
    ...r.trades.map((tr) => [tr.id, fmt.num(tr.marginalCva), fmt.pct(tr.marginalShare, 1), fmt.num(tr.incrementalCva), fmt.num(tr.standaloneCva)]),
    ['total', fmt.num(r.marginalCvaSum), passFail(r.addsUp, 'adds up (f32)'), fmt.num(r.incrementalCvaSum), fmt.num(r.standaloneCvaSum)],
  ]);
  const xvaBox = card(g, 'XVA allocation', 'CVA, FCA and KVA are linear in the discounted EE, so the marginal contributions allocate them exactly.');
  table(xvaBox, ['trade', 'CVA', 'FCA', 'KVA'], [
    ...r.trades.map((tr) => [tr.id, fmt.num(tr.marginalCva), fmt.num(tr.marginalFca), fmt.num(tr.marginalKva)]),
    ['netting set', fmt.num(r.totalCva), fmt.num(r.totalFca), fmt.num(r.totalKva)],
  ]);
  window.__ccrEngineRun = { page: 'allocation', engine: runResult.engine, totalCva: r.totalCva, marginalSum: r.marginalCvaSum };
  return engineNote(runResult, performance.now() - t0);
});

engineSelector(page, () => button.click());
