import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { defaultBermudan, saveSpec } from '../spec.js';
import { card, el, grid, initPage, runButton, specEditor, table, tiles, toArrays } from '../ui.js';

const page = initPage({
  id: 'amc',
  title: 'American Monte Carlo: Bermudan swaption exposure',
  context: 'ccr/instruments: BermudanSwaption   ccr/core: PolynomialRegression',
  description: 'A backward Longstaff–Schwartz pass regresses the deflated realised value on the Hull–White state to estimate the exercise policy. ' +
    'The same regressions then give the mark-to-market at every date, so an exercisable product gets an exposure profile from the ' +
    'scenarios used for the rest of the portfolio. After exercise, physical settlement carries the swap while cash settlement has nothing left.',
});
specEditor(page, ['market', 'simulation']);

page.spec.bermudan = { ...defaultBermudan(), ...(page.spec.bermudan || {}) };
const b = page.spec.bermudan;
const form = el('details', { class: 'spec', open: true }, el('summary', { text: 'Bermudan swaption' }));
const fields = el('div', { class: 'spec-sections' });
const numField = (label, key, scale = 1) => {
  const input = el('input', { type: 'number', step: 'any', value: +(b[key] * scale).toFixed(6) });
  input.addEventListener('change', () => { b[key] = Number(input.value) / scale; saveSpec(page.spec); });
  return el('label', { class: 'field' }, el('span', { text: label }), input);
};
const dir = el('select', {}, ['receiver', 'payer'].map((d) => el('option', { value: d, text: d, selected: d === b.direction })));
dir.addEventListener('change', () => { b.direction = dir.value; saveSpec(page.spec); });
fields.append(el('div', {},
  numField('notional', 'notional'), numField('strike (fixed %)', 'fixedRate', 100), numField('first exercise (years)', 'start'),
  numField('underlying tenor (years)', 'tenor'), numField('payments & exercises per year', 'freq'), numField('regression degree', 'degree'),
  el('label', { class: 'field' }, el('span', { text: 'right to receive/pay fixed' }), dir)));
form.append(fields);
page.main.insertBefore(form, page.toolbar);

runButton(page, 'Value by AMC', async (spec) => {
  const r = await run('amc', { ...spec, grid: { ...spec.grid, horizon: Math.max(spec.grid.horizon, b.start + b.tenor) }, samplePaths: 20 });
  const t = Array.from(r.times);
  const maxEur = Math.max(...r.european);
  tiles(page.content, [
    { label: 'Bermudan price (AMC)', value: fmt.num(r.price), hint: `strike ${fmt.pct(r.fixedRate, 3)} vs par ${fmt.pct(r.parRate, 3)}` },
    { label: 'Best European (Jamshidian)', value: fmt.num(maxEur), hint: 'lower bound' },
    { label: 'Switch option value', value: fmt.num(r.price - maxEur), hint: fmt.pct(r.price / maxEur - 1, 1) + ' over best European' },
    { label: 'Never exercised', value: fmt.pct(r.exerciseDistribution[r.exerciseDistribution.length - 1], 1), hint: `${fmt.num(r.elapsedMs)} ms for value cube` },
  ]);
  const g = grid(page.content);
  const pp = r.physicalProfile, cp = r.cashProfile;
  lineChart(card(g, 'Exposure: physical vs cash settlement', 'Both are identical before the first exercise. Cash settlement drops to zero on the exercised paths.'), {
    x: t, xLabel: 't (years)',
    series: [
      { name: 'EE physical', y: Array.from(pp.ee) }, { name: 'PFE physical', y: Array.from(pp.pfe), dash: true, colorIndex: 0 },
      { name: 'EE cash', y: Array.from(cp.ee), colorIndex: 1 }, { name: 'PFE cash', y: Array.from(cp.pfe), dash: true, colorIndex: 1 },
    ],
  });
  lineChart(card(g, 'Value paths, physical settlement', 'After exercise a path carries the underlying swap and can go negative. Before that, the option value is ≥ 0.'), {
    x: t, xLabel: 't (years)', zero: true, samples: toArrays(r.physicalSamples),
    series: [{ name: 'E[V]', y: Array.from(pp.expectedValue) }, { name: 'ENE', y: Array.from(pp.ene), colorIndex: 2 }],
  });
  lineChart(card(g, 'Value paths, cash settlement', 'Exposure stops at exercise.'), {
    x: t, xLabel: 't (years)', zero: true, samples: toArrays(r.cashSamples), series: [{ name: 'E[V]', y: Array.from(cp.expectedValue), colorIndex: 1 }],
  });
  barChart(card(g, 'Exercise policy: when are paths exercised?', 'Share of paths exercising at each date, from the estimated policy.'), {
    labels: [...Array.from(r.exercises).map((e) => fmt.years(e)), 'never'],
    series: [{ name: 'share of paths', values: Array.from(r.exerciseDistribution) }], yFormat: (v) => fmt.pct(v, 0),
  });
  barChart(card(g, 'European swaptions into the remaining swap', 'Jamshidian closed form for each exercise date, compared with the Bermudan price.'), {
    labels: Array.from(r.exercises).map((e) => fmt.years(e)),
    series: [{ name: 'European', values: Array.from(r.european) }, { name: 'Bermudan', values: r.european.map(() => r.price), colorIndex: 1 }],
  });
  const box = card(g, 'Regression basis sensitivity', 'Price against the degree of the polynomial in x(t). It should be stable from degree 2 or 3.');
  table(box, ['degree', 'price', 'vs degree 3'], r.degreeSensitivity.map((d) => [d.degree, fmt.num(d.price), fmt.pct(d.price / r.degreeSensitivity[2].price - 1, 2)]));
});
