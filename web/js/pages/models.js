import { run } from '../ccr-client.js';
import { fmt, lineChart } from '../charts.js';
import { factorNames } from '../spec.js';
import { card, grid, initPage, passFail, runButton, specEditor, table, toArrays } from '../ui.js';

const page = initPage({
  id: 'models',
  title: 'Risk-factor models and scenario generation',
  context: 'ccr/models: HullWhite1F · LognormalAsset · CirIntensity · ScenarioGenerator',
  description: 'All risk factors are simulated jointly under the domestic risk-neutral measure, with the bank account as numeraire. Hull–White ' +
    'steps x(t) and ∫r dt exactly, so the deflator D(0,t) = e<sup>−∫r</sup> reproduces the initial curve. FX and equities drift at the simulated short rate. The CIR++ intensity ' +
    'is shifted to fit the market survival curve exactly. Every check below is a martingale test that must hold within Monte Carlo error.',
});
specEditor(page, ['market', 'assets', 'credit', 'correlation', 'simulation']);

const bandSeries = (bands, name, colorIndex = 0) => [
  { name: `${name} 5–95%`, y: Array.from(bands[2]), band: { lower: Array.from(bands[0]), upper: Array.from(bands[4]) }, colorIndex },
  { name: `${name} 25–75%`, y: Array.from(bands[2]), band: { lower: Array.from(bands[1]), upper: Array.from(bands[3]) }, colorIndex, hideLegend: true },
];

runButton(page, 'Simulate', async (spec) => {
  const r = await run('simulate', { ...spec, samplePaths: 20 });
  const t = Array.from(r.times);
  const g = grid(page.content);
  const last = t.length - 1;
  const checks = [];

  lineChart(card(g, 'Hull–White short rate r(t)', 'Median with 25–75% and 5–95% bands, plus 20 sample paths.'), {
    x: t, xLabel: 't (years)', yFormat: (v) => fmt.pct(v, 1), samples: toArrays(r.rates.samples),
    series: [...bandSeries(r.rates.bands, 'r(t)'), { name: 'mean', y: Array.from(r.rates.meanShortRate), colorIndex: 1, dash: true }],
  });
  lineChart(card(g, 'Numeraire martingale tests', `E[D(0,t)] must equal P(0,t), and E[D(0,t)·P(t,T)] must equal P(0,T) for T = ${r.rates.bondMaturity.toFixed(1)}y.`), {
    x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(4),
    series: [
      { name: 'E[D(0,t)]', y: Array.from(r.rates.meanDeflator) },
      { name: 'P(0,t)', y: Array.from(r.rates.curveDiscount), dash: true },
      { name: 'E[D·P(t,T)]', y: Array.from(r.rates.meanDeflatedBond), colorIndex: 2 },
      { name: 'P(0,T)', y: t.map(() => r.rates.bondTarget), colorIndex: 2, dash: true },
    ],
  });
  checks.push(['E[D(0,T)] vs P(0,T)', r.rates.meanDeflator[last], r.rates.curveDiscount[last]]);
  checks.push(['E[D(0,t)P(t,T)] vs P(0,T)', r.rates.meanDeflatedBond[last], r.rates.bondTarget]);

  r.assets.forEach((a, i) => {
    lineChart(card(g, `${a.name} (log-normal)`, 'Fan chart and sample paths.'), {
      x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(3), samples: toArrays(a.samples), series: bandSeries(a.bands, a.name, 3 + i),
    });
    lineChart(card(g, `${a.name}: martingale test`, 'E[D(0,t)·S(t)] = S(0)·Q(0,t), where Q is the foreign or dividend discount factor. This holds exactly under stochastic rates.'), {
      x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(4),
      series: [{ name: 'E[D·S]', y: Array.from(a.meanDeflated) }, { name: 'S(0)·Q(0,t)', y: Array.from(a.target), dash: true }],
    });
    checks.push([`E[D·S] vs S0·Q (${a.name})`, a.meanDeflated[last], a.target[last]]);
  });
  r.credits.forEach((c) => {
    lineChart(card(g, `${c.name}: CIR++ default intensity λ(t)`, 'λ = y + ψ(t). The CIR component y is floored at zero (full truncation), and ψ is the deterministic shift that fits the market curve.'), {
      x: t, xLabel: 't (years)', yFormat: (v) => fmt.pct(v, 1), samples: toArrays(c.intensitySamples),
      series: [...bandSeries(c.intensityBands, 'λ', 2), { name: 'shift ψ(t)', y: Array.from(c.shift), colorIndex: 7, dash: true }],
    });
    lineChart(card(g, `${c.name}: survival fit`, 'The mean pathwise survival E[e^(−∫λ)] reproduces the bootstrapped market curve.'), {
      x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(3), samples: toArrays(c.survivalSamples),
      series: [{ name: 'E[Q_path]', y: Array.from(c.meanSurvival) }, { name: 'market Q', y: Array.from(c.marketSurvival), dash: true }],
    });
    checks.push([`E[Q] vs market (${c.name})`, c.meanSurvival[last], c.marketSurvival[last]]);
  });

  const box = card(g, 'Checks', `${fmt.num(r.numPaths)} paths, ${t.length} dates, simulated in ${fmt.num(r.elapsedMs)} ms. Tolerance: 1% relative.`);
  table(box, ['test at horizon', 'Monte Carlo', 'exact', 'rel. error', ''], checks.map(([n, mc, ex]) => [n, mc.toFixed(5), ex.toFixed(5), fmt.pct(Math.abs(mc / ex - 1), 3), passFail(Math.abs(mc / ex - 1) < 0.01)]));
  if (r.realisedCorrelation) {
    const names = factorNames(spec);
    const m = r.realisedCorrelation;
    table(box, ['realised increment correlation', ...names], names.map((n, i) => [n, ...names.map((_, j) => `${m.data[i * m.cols + j].toFixed(3)} (${spec.correlation[i][j]})`)]));
  }
});
