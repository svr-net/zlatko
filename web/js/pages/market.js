import { run } from '../ccr-client.js';
import { fmt, lineChart } from '../charts.js';
import { card, grid, initPage, passFail, runButton, specEditor, table } from '../ui.js';

const page = initPage({
  id: 'market',
  title: 'Market data: yield and credit curves',
  context: 'ccr/market: YieldCurve · CreditCurve   ccr/instruments: CreditDefaultSwap, bootstrapCreditCurve',
  description: 'Zero rates are linearly interpolated, which gives consistent discount factors and instantaneous forwards. That forward curve feeds the ' +
    'Hull–White drift. Counterparty hazard rates are piecewise flat and bootstrapped so that each CDS quote reprices at par. ' +
    'Those curves are what CVA is computed against, and what the CS01 hedge bumps.',
});
specEditor(page, ['market', 'credit', 'own'], { open: true });

runButton(page, 'Run', async (spec) => {
  const r = await run('marketDemo', { ...spec, horizon: Math.max(10, spec.grid.horizon) });
  const yc = r.yieldCurve;
  const t = Array.from(yc.times);
  const g = grid(page.content);
  lineChart(card(g, 'Domestic zero and forward rates', 'Zero rates are linear between the pillars. The instantaneous forward f(0,t) = r(t) + t·r′(t) jumps at the pillars.'), {
    x: t, xLabel: 't (years)', yFormat: (v) => fmt.pct(v, 2), yLabel: 'rate',
    series: [
      { name: 'zero rate', y: Array.from(yc.zero) },
      { name: 'instantaneous forward', y: Array.from(yc.instantaneousForward) },
      { name: '3m simple forward', y: Array.from(yc.forward3m), dash: true },
    ],
  });
  lineChart(card(g, 'Discount factors P(0, t)', ''), {
    x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(3), series: [{ name: 'P(0,t)', y: Array.from(yc.discount) }],
  });
  for (const c of r.credits) {
    lineChart(card(g, `${c.name}: bootstrapped hazard rate`, `Piecewise-flat hazard that reprices every CDS quote (recovery ${fmt.pct(c.recovery, 0)}).`), {
      x: t, xLabel: 't (years)', yFormat: (v) => fmt.pct(v, 2), series: [{ name: 'hazard λ(t)', y: Array.from(c.hazard), step: true }],
    });
    lineChart(card(g, `${c.name}: survival and default probability`, ''), {
      x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(3),
      series: [{ name: 'survival Q(τ > t)', y: Array.from(c.survival) }, { name: 'default probability', y: Array.from(c.defaultProbability) }],
    });
    const box = card(g, `${c.name}: CDS repricing`, 'Each quote is repriced off the bootstrapped curve. Protection leg and risky annuity are per unit notional. The credit triangle λ(1−R) is shown for comparison.');
    table(box, ['tenor', 'quote', 'model par spread', 'risky annuity', 'protection leg', 'Q(T)', 'λ(1−R)', 'reprices'],
      c.quotes.map((q) => [fmt.years(q.maturity), fmt.bp(q.quote, 2), fmt.bp(q.parSpread, 4), q.riskyAnnuity.toFixed(4), q.protectionLeg.toFixed(5),
        q.survival.toFixed(4), fmt.bp(q.creditTriangle, 1), passFail(Math.abs(q.parSpread - q.quote) < 1e-9)]));
  }
});
