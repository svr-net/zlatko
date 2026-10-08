import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'cva',
  title: 'Pricing counterparty risk: CVA and DVA',
  context: 'ccr/cva: unilateralCva · cvaTermStructure · bilateralCva · cvaRunningSpread · pathwiseCva',
  description: 'Under independence of exposure and default, CVA = (1 − R) ∫ EE*(t) dPD(t), where EE* is the discounted expected exposure. ' +
    'Bilateral CVA uses first-to-default: our own survival weights CVA, and the counterparty\'s weights DVA, which comes from our own default. ' +
    'Pathwise CVA uses the simulated CIR++ survival on each path instead of the market curve. ' +
    'It matches unilateral CVA when the intensity is uncorrelated with the exposure (see <a href="wwr.html">wrong-way risk</a>).',
});
specEditor(page, ['market', 'assets', 'credit', 'own', 'correlation', 'simulation', 'portfolio', 'csa']);

runButton(page, 'Price CVA', async (spec) => {
  const r = await run('cva', spec);
  const p = r.profile;
  const t = Array.from(p.times);
  tiles(page.content, [
    { label: 'Unilateral CVA', value: fmt.num(r.cva) },
    { label: 'Bilateral: CVA − DVA', value: r.bilateral ? fmt.num(r.bilateral.total) : '–', hint: r.bilateral ? `CVA ${fmt.num(r.bilateral.cva)} · DVA ${fmt.num(r.bilateral.dva)}` : '' },
    { label: 'CVA running spread', value: r.runningSpread !== undefined ? fmt.bp(r.runningSpread, 2) : '–', hint: r.spreadNotional ? `on ${fmt.compact(r.spreadNotional)} over ${fmt.years(r.spreadMaturity)}` : '' },
    { label: 'Pathwise CVA (CIR++)', value: fmt.num(r.pathwiseCva), hint: `${fmt.pct(r.pathwiseCva / r.cva - 1, 1)} vs independence` },
  ]);
  const g = grid(page.content);
  lineChart(card(g, 'Discounted exposures', 'EE* drives CVA and ENE* drives DVA.'), {
    x: t, xLabel: 't (years)', zero: true,
    series: [{ name: 'EE* (discounted EE)', y: Array.from(p.discountedEe) }, { name: 'ENE* (discounted ENE)', y: Array.from(p.discountedEne), colorIndex: 2 }],
  });
  const terms = Array.from(r.termStructure);
  let cum = 0;
  const cumulative = terms.map((v) => (cum += v));
  lineChart(card(g, 'CVA accumulation over time', 'Cumulative CVA by default date. Its slope is LGD · EE* · default density.'), {
    x: t, xLabel: 't (years)', series: [{ name: 'cumulative CVA', y: cumulative, step: false }],
  });
  barChart(card(g, 'CVA contribution per period', ''), {
    labels: t.slice(1).map((x) => x.toFixed(2)),
    series: [{ name: 'CVA per period', values: terms.slice(1) }],
  });
  lineChart(card(g, 'Survival curves', 'Counterparty and own survival, from the bootstrapped CDS curves.'), {
    x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(3),
    series: [{ name: 'counterparty', y: Array.from(r.counterpartySurvival) }, ...(r.ownSurvival ? [{ name: 'own', y: Array.from(r.ownSurvival), colorIndex: 1 }] : [])],
  });
  lineChart(card(g, 'CVA against recovery', 'CVA is linear in the loss given default 1 − R.'), {
    x: Array.from(r.recoveryGrid), xLabel: 'recovery', xFormat: (v) => fmt.pct(v, 0), series: [{ name: 'CVA', y: Array.from(r.cvaByRecovery), markers: true }],
  });
  const box = card(g, 'Inputs', '');
  table(box, ['quantity', 'value'], [
    ['effective EPE (1y)', fmt.num(p.eepe1y)],
    ['lifetime EPE', fmt.num(p.epeLife)],
    ['CSA', spec.csa.enabled ? 'enabled' : 'none'],
    ['recovery', fmt.pct(spec.credits[spec.counterparty || 0].recovery, 0)],
  ]);
});
