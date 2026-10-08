import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, toArrays } from '../ui.js';

const page = initPage({
  id: 'collateral',
  title: 'Collateral: CSA terms and the margin period of risk',
  context: 'ccr/exposure: CollateralAgreement · simulateCollateral',
  description: 'The collateral held at t is the amount called on the portfolio value at <i>t − MPR</i>, the last call the defaulting counterparty met. ' +
    'Thresholds and the minimum transfer amount apply. The page compares the CSA you specify with variants. Margin-call dates are simulated as auxiliary grid dates, ' +
    'and exposure is reported only on the primary dates. The spikes on cash-flow dates are the known effect of collateral posted ' +
    'before a payment still being held during the margin period of risk.',
});
specEditor(page, ['market', 'assets', 'simulation', 'portfolio', 'csa'], { open: true });

runButton(page, 'Compare CSAs', async (spec) => {
  const csa = { ...spec.csa, enabled: true };
  const variants = [
    { name: 'no CSA', csa: { ...csa, enabled: false } },
    { name: 'specified CSA', csa },
    { name: 'zero threshold & MTA', csa: { ...csa, thresholdCounterparty: 0, thresholdOwn: 0, mta: 0 } },
    { name: 'MPR 20 days', csa: { ...csa, mpr: 20 / 250 } },
    { name: 'one-way (we never post)', csa: { ...csa, thresholdOwn: -1 } },
  ];
  const results = [];
  for (const v of variants) results.push(await run('exposure', { ...spec, csa: v.csa, samplePaths: 12 }));
  const t = Array.from(results[0].profile.times);
  const g = grid(page.content);
  lineChart(card(g, 'Expected exposure by CSA', ''), {
    x: t, xLabel: 't (years)', series: results.map((r, i) => ({ name: variants[i].name, y: Array.from(r.profile.ee), colorIndex: i })),
  });
  lineChart(card(g, `PFE ${Math.round(spec.pfeQuantile * 100)}% by CSA`, ''), {
    x: t, xLabel: 't (years)', series: results.map((r, i) => ({ name: variants[i].name, y: Array.from(r.profile.pfe), colorIndex: i })),
  });
  const sc = results[1];
  lineChart(card(g, 'Specified CSA: value vs collateral held', 'Sample paths of the netted value (grey) and the median collateral held with its 5–95% band.'), {
    x: t, xLabel: 't (years)', zero: true, samples: toArrays(sc.nettedSamples),
    series: [{ name: 'collateral median, 5–95%', y: Array.from(sc.collateralBands[1]), band: { lower: Array.from(sc.collateralBands[0]), upper: Array.from(sc.collateralBands[2]) }, colorIndex: 1 }],
  });
  barChart(card(g, 'Effective EPE (1y) and peak PFE', ''), {
    labels: variants.map((v) => v.name),
    series: [{ name: 'EEPE (1y)', values: results.map((r) => r.profile.eepe1y) }, { name: 'peak PFE', values: results.map((r) => r.profile.maxPfe), colorIndex: 1 }],
  });
  const box = card(g, 'Summary', '');
  table(box, ['CSA', 'H cpty', 'H own', 'MTA', 'MPR (days)', 'EEPE 1y', 'peak PFE', 'EEPE vs no CSA'], results.map((r, i) => {
    const c = variants[i].csa;
    return [variants[i].name, c.enabled ? fmt.compact(c.thresholdCounterparty) : '–', c.enabled ? (c.thresholdOwn < 0 ? '∞' : fmt.compact(c.thresholdOwn)) : '–',
      c.enabled ? fmt.compact(c.mta) : '–', c.enabled ? fmt.num(c.mpr * 250) : '–', fmt.compact(r.profile.eepe1y), fmt.compact(r.profile.maxPfe),
      fmt.pct(r.profile.eepe1y / results[0].profile.eepe1y, 1)];
  }));
});
