import { run } from '../ccr-client.js';
import { fmt, histogram, lineChart } from '../charts.js';
import { card, el, grid, initPage, runButton, specEditor, table, tiles, toArrays } from '../ui.js';

const page = initPage({
  id: 'exposure',
  title: 'Exposure profiles and netting',
  context: 'ccr/exposure: ExposureEngine · NettingSet · ExposureProfile',
  description: 'The engine merges trade event dates into the grid, simulates, revalues every trade, nets them and computes the exposure statistics. ' +
    'EE and PFE describe the distribution of max(V, 0). The Basel measures are EPE (time-averaged EE), effective EE (non-decreasing) and effective EPE. ' +
    'The netting benefit compares the netted EE with the sum of positive trade values.',
});
specEditor(page, ['market', 'assets', 'correlation', 'simulation', 'portfolio', 'csa'], { tradeTypes: ['swap', 'forward', 'option', 'bermudan'] });

const histInput = el('input', { type: 'number', step: 'any', value: 2, style: 'width:80px' });
page.toolbar.insertBefore(el('label', { class: 'status' }, 'histogram at t = ', histInput, ' y'), page.status);

runButton(page, 'Compute exposure', async (spec) => {
  const r = await run('exposure', { ...spec, samplePaths: 30, histogramTime: Number(histInput.value) });
  const p = r.profile, gp = r.grossProfile;
  const t = Array.from(p.times);
  tiles(page.content, [
    { label: 'EPE (1y)', value: fmt.compact(p.epe1y), hint: 'time-average of EE' },
    { label: 'Effective EPE (1y)', value: fmt.compact(p.eepe1y), hint: 'Basel EAD = α · EEPE' },
    { label: 'Peak PFE', value: fmt.compact(p.maxPfe), hint: `${Math.round(p.pfeQuantile * 100)}% quantile` },
    { label: 'Lifetime EPE', value: fmt.compact(p.epeLife) },
    { label: 'Netting benefit', value: fmt.pct(1 - p.eepe1y / gp.eepe1y, 1), hint: `EEPE netted ${fmt.compact(p.eepe1y)} vs gross ${fmt.compact(gp.eepe1y)}` },
    { label: 'Simulation', value: `${fmt.num(r.numPaths)} × ${r.simulationDates}`, hint: `${r.reportingDates} reporting dates, ${fmt.num(r.elapsedMs)} ms` },
  ]);
  const g = grid(page.content);
  lineChart(card(g, 'Exposure profile', 'Grey lines are sample paths of the value at risk on default (netted, net of collateral).'), {
    x: t, xLabel: 't (years)', zero: true, samples: toArrays(r.exposureSamples),
    series: [
      { name: 'EE', y: Array.from(p.ee) },
      { name: `PFE ${Math.round(p.pfeQuantile * 100)}%`, y: Array.from(p.pfe) },
      { name: 'effective EE', y: Array.from(p.effectiveEe), dash: true, colorIndex: 0 },
      { name: 'ENE', y: Array.from(p.ene), colorIndex: 2 },
    ],
  });
  lineChart(card(g, 'Netting benefit', 'Netted EE against the gross exposure (sum of positive trade values).'), {
    x: t, xLabel: 't (years)',
    series: [
      { name: 'EE netted', y: Array.from(p.ee) }, { name: 'EE gross', y: Array.from(gp.ee), colorIndex: 1 },
      { name: 'PFE netted', y: Array.from(p.pfe), dash: true, colorIndex: 0 }, { name: 'PFE gross', y: Array.from(gp.pfe), dash: true, colorIndex: 1 },
    ],
  });
  lineChart(card(g, 'Standalone EE by trade', 'Without netting, each trade on its own.'), {
    x: t, xLabel: 't (years)', series: r.trades.map((tr, i) => ({ name: tr.id, y: Array.from(tr.profile.ee), colorIndex: i % 8 })),
  });
  histogram(card(g, `Exposure distribution at t = ${fmt.years(r.histogramTime)}`, 'Distribution of the netted value net of collateral across paths. Its positive tail is the PFE.'),
    Array.from(r.histogramValues), { bins: 40 });
  const box = card(g, 'Profile table', '');
  table(box, ['t', 'E[V]', 'EE', 'ENE', 'PFE', 'eff. EE', 'EE*'], t.map((x, j) => [fmt.years(x), fmt.num(p.expectedValue[j]), fmt.num(p.ee[j]), fmt.num(p.ene[j]), fmt.num(p.pfe[j]), fmt.num(p.effectiveEe[j]), fmt.num(p.discountedEe[j])]));
});
