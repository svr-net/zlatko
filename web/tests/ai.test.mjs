// Node tests for the AI chapter (web/js/ai): the agent-skill tool layer on the library's XVA
// analysis (the fused kernels in the WebAssembly build) and the surrogate back test (C++).
import assert from 'node:assert/strict';
import createCcrModule from '../wasm/ccr.js';
import data from '../js/ai/backtest-data.js';
import { SKILL_MD, TOOLS, adapters, executeTool, runAgentLoop, runHarness, scriptedLlm, stateToSpec } from '../js/ai/skill.js';

const mod = await createCcrModule();
const sim = { numPaths: 1000, seed: 2024, antithetic: true };
// The runner of the browser, on the CPU fused kernels: one library call per tool call.
const engine = async (fn, spec) => {
  const r = mod.call('cpuKernels', { ...spec, sim, analysis: fn });
  if (r.error) throw new Error(r.error);
  return r;
};

// The market state is applied by the library: a doubled counterparty spread doubles its CDS quotes.
const spec = stateToSpec({ cptySpreadMultiplier: 2, rateShift: 0.01, fundingSpread: 0.02 });
assert.equal(spec.marketState.cptySpreadMultiplier, 2);
assert.equal(spec.marketState.fxVol, 0.1);  // defaults filled in

const h = await runHarness(engine);
assert.equal(h.passed, h.total, JSON.stringify(h.rows.filter((r) => !r.pass)));

const cmp = await executeTool('compare_scenarios', { shock: { cptySpreadMultiplier: 2 } }, engine);
assert.ok(cmp.ok && cmp.change.cva > 0.5 * cmp.base.cva && cmp.base.invariant.holds && cmp.shocked.invariant.holds, JSON.stringify(cmp));

for (const t of adapters.openai()) assert.equal(t.type, 'function');
assert.equal(adapters.gemini()[0].functionDeclarations.length, TOOLS.length);
assert.equal(adapters.gemini()[0].functionDeclarations[0].parameters.type, 'OBJECT');
assert.ok(SKILL_MD.startsWith('---\nname: xva-pricing'));

// the loop recovers from a bad call: the scripted model fixes its units after the error
const run = await runAgentLoop({
  runner: engine, prompt: 'vol 10%?',
  llm: scriptedLlm([
    { toolCalls: [{ id: '1', name: 'price_xva', arguments: { fxVol: 10 } }] },
    { toolCalls: [{ id: '2', name: 'price_xva', arguments: '{"fxVol":0.1}' }] },
    { text: 'done' },
  ]),
});
assert.ok(run.completed && run.steps === 3);
assert.equal(JSON.parse(run.messages.find((m) => m.role === 'tool').content).ok, false);

const bt = mod.call('surrogateBacktest', { dataset: data, mlpEpochs: 800, mlpRefitEpochs: 100 });
assert.ok(!bt.error, bt.error);
const { linear, poly2, mlp } = bt.models;
assert.ok(poly2.static.inRange.maeBp < linear.static.inRange.maeBp);
assert.ok(mlp.static.total.maeBp < linear.static.total.maeBp);
assert.ok(bt.outsideDays > 0 && poly2.static.outRange.maeBp > poly2.static.inRange.maeBp);
console.log(`ai tests ok: harness ${h.passed}/${h.total}; MLP walk-forward MAE ${mlp.walkForward.total.maeBp.toFixed(2)}bp of gross notional`);
