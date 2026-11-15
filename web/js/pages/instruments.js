import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { fmt, lineChart } from '../charts.js';
import { card, grid, initPage, passFail, runButton, specEditor, table, toArrays } from '../ui.js';

const page = initPage({
  id: 'instruments',
  title: 'Instruments priced on scenarios',
  context: 'ccr/instruments: Trade · PathwiseTrade · InterestRateSwap · AssetForward · EuropeanOption · BermudanSwaption',
  description: 'Every trade is revalued on every path and date from the simulated state. Swaps recover past fixings from the curve simulated at ' +
    'the fixing date. Forwards and options use the simulated domestic discount factors. Bermudans are valued by regression (American Monte Carlo inside the fused kernels). ' +
    'Values are taken <i>after</i> cash flows at each date, which is why swap exposures have a saw-tooth shape. At t = 0 each value must equal its closed-form price.',
});
specEditor(page, ['market', 'assets', 'simulation', 'portfolio'], { tradeTypes: ['swap', 'forward', 'option', 'bermudan'] });

const button = runButton(page, 'Price portfolio', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('instruments', { ...spec, samplePaths: 20 });
  const r = runResult.result;
  const t = Array.from(r.times);
  const summary = card(page.content, 'Values at t = 0', `Scenario valuation against the analytic price (Bermudans: against the best European), ${t.length} reporting dates.`);
  table(summary, ['trade', 'type', 'maturity', 'MtM(0) on scenarios', 'benchmark', 'check', 'EE peak', 'PFE peak'],
    r.trades.map((tr) => [tr.id, tr.type, fmt.years(tr.maturity), fmt.num(tr.value0), `${fmt.num(tr.benchmark)} (${tr.benchmarkLabel})`,
      passFail(tr.pass, tr.checkLabel), fmt.compact(tr.eePeak), fmt.compact(tr.profile.maxPfe)]));
  const g = grid(page.content);
  r.trades.forEach((tr, i) => {
    lineChart(card(g, tr.id, 'Sample paths of the mark-to-market with the 5/50/95% bands. EE and PFE are those of the trade on its own.'), {
      x: t, xLabel: 't (years)', zero: true, samples: toArrays(tr.samples),
      series: [
        { name: 'median, 5–95%', y: Array.from(tr.bands[1]), band: { lower: Array.from(tr.bands[0]), upper: Array.from(tr.bands[2]) }, colorIndex: i % 8 },
        { name: 'EE', y: Array.from(tr.profile.ee), colorIndex: (i + 1) % 8 },
        { name: 'PFE', y: Array.from(tr.profile.pfe), colorIndex: (i + 1) % 8, dash: true },
      ],
    });
  });
  lineChart(card(g, 'Deflated expected value E[D(0,t)·V(t)]', 'Constant (a martingale) until the first cash flow. After that it falls by the value of the flows already paid.'), {
    x: t, xLabel: 't (years)', zero: true,
    series: r.trades.map((tr, i) => ({ name: tr.id, y: Array.from(tr.meanDeflatedValue), colorIndex: i % 8 })),
  });
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
