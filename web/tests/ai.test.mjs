// Node tests for the AI chapter (web/js/ai): agent-skill tool layer and surrogate back test.
// The engine is replaced by a closed-form fake runner (WASM is not needed); the surrogate test uses the
// committed Monte Carlo dataset, so it checks the real back test.
import assert from 'node:assert/strict';
import data from '../js/ai/backtest-data.js';
import { SKILL_MD, TOOLS, adapters, executeTool, runAgentLoop, runHarness, scriptedLlm, stateToSpec } from '../js/ai/skill.js';
import { runBacktest } from '../js/ai/surrogate.js';

const fake = async (fn, spec) => {
  assert.equal(fn, 'xva');
  const m = spec.credits[0].quotes[0].spread / 0.007;
  const cva = 50000 * m, dva = 8000, fca = 20000 + 1e6 * spec.xva.fundingSpreadBorrow, fba = 3000, mva = 9000, kva = 12000;
  return {
    xva: { cva, dva, fca, fba, mva, kva, total: cva - dva + fca - fba + mva + kva },
    sensitivities: { maturities: [1, 3], counterpartyCs01: [1, 2], ownCs01: [-1], fundingSpread01: 5 },
    trades: [{ id: 'a', cva: 1, fca: 2, kva: 3 }],
  };
};

const spec = stateToSpec({ cptySpreadMultiplier: 2, rateShift: 0.01, fundingSpread: 0.02 });
assert.ok(Math.abs(spec.credits[0].quotes[0].spread - 0.014) < 1e-12);
assert.ok(Math.abs(spec.domestic.rates[0] - 0.04) < 1e-12);
assert.equal(spec.xva.fundingSpreadLend, 0.01);

const h = await runHarness(fake);
assert.equal(h.passed, h.total, JSON.stringify(h.rows.filter((r) => !r.pass)));

const cmp = await executeTool('compare_scenarios', { shock: { cptySpreadMultiplier: 2 } }, fake);
assert.ok(Math.abs(cmp.change.cva - 50000) < 1e-6);

for (const t of adapters.openai()) assert.equal(t.type, 'function');
assert.equal(adapters.gemini()[0].functionDeclarations.length, TOOLS.length);
assert.equal(adapters.gemini()[0].functionDeclarations[0].parameters.type, 'OBJECT');
assert.ok(SKILL_MD.startsWith('---\nname: xva-pricing'));

// the loop recovers from a bad call: the scripted model fixes its units after the error
const run = await runAgentLoop({
  runner: fake, prompt: 'vol 10%?',
  llm: scriptedLlm([
    { toolCalls: [{ id: '1', name: 'price_xva', arguments: { fxVol: 10 } }] },
    { toolCalls: [{ id: '2', name: 'price_xva', arguments: '{"fxVol":0.1}' }] },
    { text: 'done' },
  ]),
});
assert.ok(run.completed && run.steps === 3);
assert.equal(JSON.parse(run.messages.find((m) => m.role === 'tool').content).ok, false);

const bt = await runBacktest(data, { mlpEpochs: 800, mlpRefitEpochs: 100 });
const { linear, poly2, mlp } = bt.models;
assert.ok(poly2.static.inRange.maeBp < linear.static.inRange.maeBp);
assert.ok(mlp.static.total.maeBp < linear.static.total.maeBp);
assert.ok(bt.outsideDays > 0 && poly2.static.outRange.maeBp > poly2.static.inRange.maeBp);
console.log(`ai tests ok: harness ${h.passed}/${h.total}; MLP walk-forward MAE ${mlp.walkForward.total.maeBp.toFixed(2)}bp of gross notional`);
