import { run } from '../ccr-client.js';
import { fmt, lineChart } from '../charts.js';
import { card, grid, initPage, passFail, runButton, specEditor, table, toArrays } from '../ui.js';

const page = initPage({
  id: 'instruments',
  title: 'Instruments priced on scenarios',
  context: 'ccr/instruments: Trade · PathwiseTrade · InterestRateSwap · AssetForward · EuropeanOption · BermudanSwaption',
  description: 'Every trade is revalued on every path and date from the simulated state. Swaps recover past fixings from the curve simulated at ' +
    'the fixing date. Forwards and options use the simulated domestic discount factors. Bermudans are valued by regression. ' +
    'Values are taken <i>after</i> cash flows at each date, which is why swap exposures have a saw-tooth shape. At t = 0 each value must equal its closed-form price.',
});
specEditor(page, ['market', 'assets', 'simulation', 'portfolio'], { tradeTypes: ['swap', 'forward', 'option', 'bermudan'] });

runButton(page, 'Price portfolio', async (spec) => {
  const r = await run('priceTrades', { ...spec, samplePaths: 20 });
  const t = Array.from(r.times);
  const summary = card(page.content, 'Values at t = 0', `Scenario valuation against the analytic price, ${t.length} reporting dates, ${fmt.num(r.elapsedMs)} ms.`);
  table(summary, ['trade', 'type', 'maturity', 'MtM(0) on scenarios', 'analytic', 'check', 'EE peak', 'PFE peak'],
    r.trades.map((tr) => {
      const ref = Number.isFinite(tr.reference) ? tr.reference : tr.europeanLowerBound;
      const check = tr.type === 'bermudan'
        ? passFail(tr.value0 >= 0.97 * tr.europeanLowerBound, '≥ max European')
        : passFail(Math.abs(tr.value0 - tr.reference) <= 1e-6 * Math.max(1, Math.abs(tr.reference)));
      return [tr.id, tr.type, fmt.years(tr.maturity), fmt.num(tr.value0), fmt.num(ref) + (tr.type === 'bermudan' ? ' (max Eur.)' : ''), check,
        fmt.compact(Math.max(...tr.profile.ee)), fmt.compact(tr.profile.maxPfe)];
    }));
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
});
