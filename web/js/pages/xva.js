import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'xva',
  title: 'XVA: CVA, DVA, FVA, MVA and KVA',
  context: 'ccr/xva: exposureXva · fundingBenefit · initialMarginProfile · marginValuationAdjustment · xva · allocateExposureXva',
  description: 'Beyond credit risk, a derivative costs money to fund, to margin and to hold capital against. With the discounted exposures EE* and ENE* and the survival probabilities Q: ' +
    'FCA = s<sub>b</sub> ∫ EE* Q<sub>c</sub> Q<sub>o</sub> dt and FBA = s<sub>l</sub> ∫ |ENE*| Q<sub>c</sub> Q<sub>o</sub> dt give FVA = FCA − FBA. ' +
    'MVA = s<sub>m</sub> ∫ IM* Q<sub>c</sub> Q<sub>o</sub> dt funds the initial margin: IM(t) is the 99% quantile of the 10-day value change <i>conditional on each path</i>, by nested Monte Carlo (inner simulations from every path inside the fused kernel), compared below with the unconditional proxy. ' +
    'An audit on the first paths with many more inner paths removes the low bias of the quantile of a few inner samples (chapter X11), and an adaptive nested Monte Carlo of the exposure, below the XVA charts, spends inner paths only where EE* and PFE need them. ' +
    'KVA = hurdle · capital ratio · RW · α ∫ EE* Q<sub>c</sub> Q<sub>o</sub> dt. The total cost is CVA − DVA + FCA − FBA + MVA + KVA. ' +
    'The fused kernels compute everything here in one pass on WebGPU (or the CPU kernels): the exposure profile, the initial-margin quantile of the value change between dates and the per-trade Euler contributions. CVA, FCA and KVA are linear in EE*, so they allocate to trades like CVA does (<a href="allocation.html">allocation</a>), ' +
    'and <a href="hedging.html">hedging</a> shows their credit and funding sensitivities.',
});
specEditor(page, ['market', 'assets', 'credit', 'own', 'xva', 'simulation', 'portfolio', 'csa']);

const button = runButton(page, 'Price XVA', async (spec) => {
  const t0 = performance.now();
  // The page audits the nested initial margin by default (spec.imAudit = false turns it off).
  const runResult = await runAnalysis('xva', { ...spec, imAudit: spec.imAudit ?? true });
  const r = runResult.result;
  const x = r.xva;
  const t = Array.from(r.profile.times);
  tiles(page.content, [
    { label: 'Total XVA', value: fmt.num(x.total), hint: 'CVA − DVA + FCA − FBA + MVA + KVA' },
    { label: 'CVA − DVA', value: fmt.num(x.cva - x.dva), hint: `CVA ${fmt.num(x.cva)} · DVA ${fmt.num(x.dva)}` },
    { label: 'FVA', value: fmt.num(x.fva), hint: `FCA ${fmt.num(x.fca)} · FBA ${fmt.num(x.fba)}` },
    { label: 'MVA', value: fmt.num(x.mva), hint: r.imMethod !== 'nested' ? 'unconditional IM proxy'
      : `conditional IM, ${fmt.num(r.imInnerPaths)} inner paths${r.imAudit ? ` · before the audit ${fmt.num(r.imAudit.mvaUncorrected)}` : ''} · proxy ${fmt.num(r.mvaProxy)}` },
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
  if (r.initialMarginBands)
    lineChart(card(g, 'Initial margin: nested conditional vs proxy', 'IM(t, path) is the 99% quantile of the netting-set value change over the margin period of risk conditional on the path, from inner simulations launched from every path (nested Monte Carlo in the fused kernel). Median with its 5–95% band across paths, and the discounted means that drive MVA: nested E[D·IM] against the unconditional proxy (one quantile across paths for all of them).'), {
      x: t, xLabel: 't (years)', zero: true,
      series: [
        { name: 'IM median, 5–95% across paths', y: Array.from(r.initialMarginBands[1]), band: { lower: Array.from(r.initialMarginBands[0]), upper: Array.from(r.initialMarginBands[2]) }, colorIndex: 1 },
        { name: r.imAudit ? 'E[D·IM] nested, audited (MVA)' : 'E[D·IM] nested (MVA)', y: Array.from(r.initialMargin), colorIndex: 0 },
        ...(r.imAudit ? [{ name: `E[D·IM] with ${fmt.num(r.imInnerPaths)} inner paths, before the audit`, y: Array.from(r.imAudit.initialMarginUncorrected), colorIndex: 0, dash: true }] : []),
        { name: 'IM* proxy', y: Array.from(r.initialMarginProxy), colorIndex: 3, dash: true },
      ],
    });
  if (r.imAudit)
    lineChart(card(g, 'Audit of the conditional initial margin', `The 99% quantile of ${fmt.num(r.imInnerPaths)} inner samples is biased low, and MVA, linear in IM, keeps the bias however many paths there are. The first ${fmt.num(r.imAudit.paths)} paths are rerun with ${fmt.num(r.imAudit.innerPaths)} inner paths (the kernel's own samples, extended), and IM on each date is scaled by the ratio of the two E[D·IM]. Chapter X11.`), {
      x: t, xLabel: 't (years)',
      series: [{ name: `E[D·IM] with ${fmt.num(r.imAudit.innerPaths)} / with ${fmt.num(r.imInnerPaths)} inner paths`, y: Array.from(r.imAudit.ratio), markers: true }],
      tooltipFormat: (v) => fmt.num(v, 4), yFormat: (v) => fmt.num(v, 3),
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
  const nested = await nestedSection(spec);
  window.__ccrEngineRun = { page: 'xva', engine: runResult.engine, total: x.total, cva: x.cva, mva: x.mva, kva: x.kva, nested: nested ? nested.rmsError : null };
  return engineNote(runResult, performance.now() - t0);
});

// Chapter X11: the exposure by adaptive nested Monte Carlo, on the CPU in the worker (the nest is not a
// fused-kernel job), against the closed form, the regression alone and the classical nest with the
// same inner paths. All numbers come from the library (nestedExposure); the page only draws them.
async function nestedSection(spec) {
  const g = grid(page.content);
  const title = 'Adaptive nested Monte Carlo of the exposure (chapter X11)';
  let n;
  try {
    n = await run('nestedExposure', { ...spec, nested: { dates: 10, numPaths: 1000, innerPerScenario: 60, ...(spec.nested || {}) } });
  } catch (e) {
    console.warn('nestedExposure:', e);
    card(g, title, `Not available: ${e.message}. The WebAssembly module in web/wasm may predate this entry point; rebuild it from the sources.`);
    return null;
  }
  const t = Array.from(n.times);
  const methods = [
    ['exact', 'closed form', 3, true],
    ['regression', 'regression only', 2, false],
    ['uniform', `classical nest, same inner paths`, 1, false],
    ['adaptive', 'adaptive nest', 0, false],
  ];
  const lines = (what) => methods.map(([key, name, colorIndex, dash]) => ({ name, y: Array.from(n[what][key]), colorIndex, dash }));
  const scope = `${fmt.num(n.outerPaths)} outer paths (half fit the proxy, ${fmt.num(n.evaluatedPerDate)} are evaluated per date), ` +
    `${fmt.num(n.innerBudgetPerDate)} inner paths per date at most` +
    (n.unsupported.length ? `; left out (no nested valuation): ${n.unsupported.join(', ')}` : '') + (n.csaIgnored ? '; uncollateralised' : '') + '.';
  lineChart(card(g, 'EE* by adaptive nested Monte Carlo', `V(t, path) is treated as unknown: a least-squares proxy on the outer paths' own discounted cash flows, an audit of its bias per bin, and inner simulations from a path's state only where the sign of V or its value is uncertain enough to move EE*. ${scope}`), {
    x: t, xLabel: 't (years)', zero: true, series: lines('ee'),
  });
  lineChart(card(g, `PFE ${fmt.pct(n.pfeQuantile, 0)} by adaptive nested Monte Carlo`, 'Scenarios that may lie on the wrong side of the quantile are refined; the closed form is the reference on the same scenarios.'), {
    x: t, xLabel: 't (years)', zero: true, series: lines('pfe'),
  });
  barChart(card(g, 'Error against the closed form', 'Relative root-mean-square error over the dates, on the same evaluation scenarios. The classical nest gets the adaptive run\'s inner paths spread evenly.'), {
    labels: ['EE*', 'PFE'],
    series: methods.slice(1).map(([key, name, colorIndex]) => ({ name, values: [n.rmsError[key].ee, n.rmsError[key].pfe], colorIndex })),
    yFormat: (v) => fmt.pct(v, 1), tooltipFormat: (v) => fmt.pct(v, 2),
  });
  lineChart(card(g, 'Where the inner paths went', 'Inner paths per date: the audit (random scenarios in every bin of proxy values) and the refinement, and the share of evaluation scenarios that got any.'), {
    x: t, xLabel: 't (years)', zero: true,
    series: [
      { name: 'inner paths, adaptive', y: Array.from(n.innerPaths), colorIndex: 0, markers: true },
      { name: 'of which audit', y: Array.from(n.auditInnerPaths), colorIndex: 2, dash: true },
    ],
  });
  table(card(g, 'Summary', `Adaptive run: ${fmt.num(n.totalInnerPaths)} inner paths in ${fmt.num(n.elapsedMs)} ms.`),
    ['method', 'EE* error', 'PFE error', ...(n.cva ? ['CVA'] : [])],
    methods.map(([key, name]) => [name, fmt.pct(n.rmsError[key].ee, 2), fmt.pct(n.rmsError[key].pfe, 2), ...(n.cva ? [fmt.num(n.cva[key])] : [])]));
  return n;
}

engineSelector(page, () => button.click());
