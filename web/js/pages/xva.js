import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'xva',
  title: 'XVA: CVA, DVA, FVA, MVA and KVA',
  context: 'ccr/xva: exposureXva · fundingBenefit · initialMarginProfile · marginValuationAdjustment · xva · allocateExposureXva',
  description: 'Beyond credit risk, a derivative costs money to fund, to margin and to hold capital against. With the discounted exposures EE* and ENE* and the survival probabilities Q: ' +
    'FCA = s<sub>b</sub> ∫ EE* Q<sub>c</sub> Q<sub>o</sub> dt and FBA = s<sub>l</sub> ∫ |ENE*| Q<sub>c</sub> Q<sub>o</sub> dt give FVA = FCA − FBA. ' +
    'MVA = s<sub>m</sub> ∫ IM* Q<sub>c</sub> Q<sub>o</sub> dt funds the initial margin, here a simulated 10-day quantile proxy. ' +
    'KVA = hurdle · capital ratio · RW · α ∫ EE* Q<sub>c</sub> Q<sub>o</sub> dt. The total cost is CVA − DVA + FCA − FBA + MVA + KVA. ' +
    'The initial margin needs the simulated paths, so this page runs on the WebAssembly engine. CVA, FCA and KVA are linear in EE*, so they allocate to trades like CVA does (<a href="allocation.html">allocation</a>), ' +
    'and <a href="hedging.html">hedging</a> shows their credit and funding sensitivities.',
});
specEditor(page, ['market', 'assets', 'credit', 'own', 'xva', 'simulation', 'portfolio', 'csa']);

runButton(page, 'Price XVA', async (spec) => {
  const r = await run('xva', spec);
  const x = r.xva;
  const t = Array.from(r.profile.times);
  tiles(page.content, [
    { label: 'Total XVA', value: fmt.num(x.total), hint: 'CVA − DVA + FCA − FBA + MVA + KVA' },
    { label: 'CVA − DVA', value: fmt.num(x.cva - x.dva), hint: `CVA ${fmt.num(x.cva)} · DVA ${fmt.num(x.dva)}` },
    { label: 'FVA', value: fmt.num(x.fva), hint: `FCA ${fmt.num(x.fca)} · FBA ${fmt.num(x.fba)}` },
    { label: 'MVA', value: fmt.num(x.mva) },
    { label: 'KVA', value: fmt.num(x.kva) },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'XVA components', 'Costs are positive. DVA and FBA are benefits and reduce the total.'), {
    labels: ['CVA', 'DVA', 'FCA', 'FBA', 'MVA', 'KVA'],
    series: [{ name: 'XVA', values: [x.cva, -x.dva, x.fca, -x.fba, x.mva, x.kva] }],
  });
  lineChart(card(g, 'Cumulative cost over time', 'Each adjustment accumulated up to every date.'), {
    x: t, xLabel: 't (years)', zero: true,
    series: [
      { name: 'CVA', y: Array.from(r.cumulativeCva) },
      { name: 'FCA', y: Array.from(r.cumulativeFca), colorIndex: 1 },
      { name: 'MVA', y: Array.from(r.cumulativeMva), colorIndex: 2 },
      { name: 'KVA', y: Array.from(r.cumulativeKva), colorIndex: 3 },
    ],
  });
  lineChart(card(g, 'Exposure and initial margin profiles', 'EE* drives FCA and KVA, |ENE*| drives FBA and the discounted IM drives MVA.'), {
    x: t, xLabel: 't (years)', zero: true,
    series: [
      { name: 'EE*', y: Array.from(r.profile.discountedEe) },
      { name: 'ENE*', y: Array.from(r.profile.discountedEne), colorIndex: 2 },
      { name: 'IM* (initial margin)', y: Array.from(r.initialMargin), colorIndex: 1 },
    ],
  });
  lineChart(card(g, 'FVA against the funding spread', 'FVA is linear in the funding spread (borrow = lend).'), {
    x: Array.from(r.spreadGrid), xLabel: 'funding spread', xFormat: (v) => fmt.bp(v, 0),
    series: [{ name: 'FVA', y: Array.from(r.fvaBySpread), markers: true }],
  });
  lineChart(card(g, 'KVA against the hurdle rate', 'KVA is linear in the cost of capital.'), {
    x: Array.from(r.hurdleGrid), xLabel: 'hurdle rate', xFormat: (v) => fmt.pct(v, 0),
    series: [{ name: 'KVA', y: Array.from(r.kvaByHurdle), markers: true }],
  });
  barChart(card(g, 'Allocation of CVA, FCA and KVA to trades', 'Marginal (Euler) contributions that add up to the netting-set values.'), {
    labels: r.trades.map((tr) => tr.id),
    series: [
      { name: 'CVA', values: r.trades.map((tr) => tr.cva) },
      { name: 'FCA', values: r.trades.map((tr) => tr.fca) },
      { name: 'KVA', values: r.trades.map((tr) => tr.kva) },
    ],
  });
  if (r.sensitivities) {
    const s = r.sensitivities;
    barChart(card(g, 'Credit sensitivities of the total XVA', 'Per 1bp move of each counterparty and own CDS quote.'), {
      labels: Array.from(s.maturities).map((m) => fmt.years(m)),
      series: [{ name: 'counterparty CS01', values: Array.from(s.counterpartyCs01) }, { name: 'own CS01', values: Array.from(s.ownCs01), colorIndex: 2 }],
      tooltipFormat: (v) => fmt.num(v, 2),
    });
    table(card(g, 'Funding sensitivity', ''), ['quantity', 'value'], [['FVA01: all funding spreads +1bp', fmt.num(s.fundingSpread01, 1)]]);
  }
});
