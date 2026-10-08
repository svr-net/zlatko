import { run } from '../ccr-client.js';
import { fmt, lineChart, scatterChart } from '../charts.js';
import { factorNames } from '../spec.js';
import { card, grid, initPage, passFail, runButton, specEditor, table } from '../ui.js';

const page = initPage({
  id: 'core',
  title: 'Core numerics',
  context: 'ccr/core: matrix · random · regression · time_grid · math',
  description: 'These are the building blocks of the Monte Carlo framework. Correlated drivers come from a Cholesky factor. ' +
    'Polynomial least squares supplies the conditional expectations for American Monte Carlo. Exposure grids are dense at the short end and gain the ' +
    'margin-call dates <i>t − MPR</i> under a CSA. Brent root finding drives the curve bootstrapping, and empirical quantiles give PFE.',
});
specEditor(page, ['assets', 'credit', 'correlation', 'simulation', 'csa']);

const matrixRows = (m, names) => names.map((n, i) => [n, ...names.map((_, j) => m.data[i * m.cols + j].toFixed(4))]);

runButton(page, 'Run', async (spec) => {
  const r = await run('coreDemo', { ...spec, numSamples: 20000, regression: { points: 400, noise: 0.25 }, mpr: spec.csa.mpr });
  const names = factorNames(spec);
  const g = grid(page.content);

  const corr = card(g, 'Correlated normals via Cholesky', 'Target correlation, its lower-triangular factor L (Σ = LLᵀ), and the realised correlation of 20,000 draws w = Lz.');
  table(corr, ['target', ...names], matrixRows(r.correlation, names));
  table(corr, ['Cholesky L', ...names], matrixRows(r.cholesky, names));
  table(corr, ['realised', ...names], matrixRows(r.realisedCorrelation, names));

  scatterChart(card(g, `Draws: ${names[0]} vs ${names[names.length - 1]} driver`, 'The first 1,500 correlated draws.'), {
    series: [{ name: 'draws', x: Array.from(r.scatter[0]), y: Array.from(r.scatter[1]) }],
    xLabel: names[0], yLabel: names[names.length - 1], square: true,
  });

  const reg = r.regression;
  lineChart(card(g, 'Least-squares polynomial regression (AMC conditional expectation)', 'Noisy payoff-like data, fitted with polynomials of degree 1–5 on a standardised regressor.'), {
    x: Array.from(reg.grid), xLabel: 'regressor', yLabel: 'y',
    series: [
      { name: 'samples', x: Array.from(reg.x), y: Array.from(reg.y), pointsOnly: true, colorIndex: 7, hideLegend: true },
      ...reg.fits.map((f, i) => ({ name: `degree ${f.degree} (R² ${f.rSquared.toFixed(3)})`, y: Array.from(f.fitted), colorIndex: i })),
    ],
  });

  const std = Array.from(r.standardGrid), lag = Array.from(r.laggedGrid);
  const steps = (t) => t.slice(1).map((v, i) => v - t[i]);
  lineChart(card(g, 'Exposure time grids', `Step size along the standard grid (${std.length} dates) and with the margin-call dates for a ${fmt.num(spec.csa.mpr * 250)}-day MPR added (${lag.length} dates).`), {
    x: std.slice(1), xLabel: 't (years)', yLabel: 'Δt (years)', yFormat: (v) => v.toFixed(3),
    series: [
      { name: 'standard grid', y: steps(std), step: true },
      { name: 'with t − MPR dates', x: lag.slice(1), y: steps(lag), step: true, colorIndex: 1 },
    ],
  });

  lineChart(card(g, 'Standard normal distribution', 'Φ is computed via erfc and is used throughout: Black, bond options, swaptions.'), {
    x: Array.from(r.normalX), xLabel: 'x', yFormat: (v) => v.toFixed(2),
    series: [{ name: 'Φ(x)', y: Array.from(r.normalCdf) }, { name: 'φ(x)', y: Array.from(r.normalPdf) }],
  });

  const b = r.brent;
  const misc = card(g, 'Brent root finding & empirical quantiles', 'Implied volatility recovered from a Black price, and quantiles of 5,000 normals against Φ⁻¹.');
  table(misc, ['Brent', 'value'], [
    ['Black call price (F=100, K=105, T=2)', b.price.toFixed(6)],
    ['true volatility', fmt.pct(b.trueVol, 4)],
    ['implied volatility', fmt.pct(b.impliedVol, 8)],
    ['function evaluations', b.evaluations],
    ['recovered', passFail(Math.abs(b.impliedVol - b.trueVol) < 1e-10)],
  ]);
  table(misc, ['p', 'empirical quantile', 'Φ⁻¹(p)'], r.quantiles.map((q) => [q.p, q.empirical.toFixed(4), q.exact.toFixed(4)]));
});
