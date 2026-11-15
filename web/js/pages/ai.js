import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import data from '../ai/backtest-data.js';
import { SKILL_MD, STATE_FIELDS, TOOLS, adapters, executeTool, runAgentLoop, runHarness, scriptedLlm } from '../ai/skill.js';
import { engineSelector, runAnalysis } from '../gpu/backend.js';
import { card, el, grid, initPage, runButton, table, tiles } from '../ui.js';

const page = initPage({
  id: 'ai',
  title: 'AI and XVA: surrogates and an LLM agent skill',
  context: 'docs/xva_chapters.md · X10 · examples/xva_ai_dataset.cpp · web/js/ai',
  description:
    'Two uses of AI around the XVA engine. <b>1. Surrogates:</b> a polynomial and a neural network learn the map from a market state to the six adjustments of the default netting set ' +
    'from full Monte Carlo prices, and are <i>back tested</i> on a 250-day market path that includes a stress episode outside the training range. ' +
    '<b>2. Agent skill:</b> typed tools, a SKILL.md and a tool-use loop that let any LLM price, shock and explain XVA by calling the engine, never by guessing numbers. ' +
    'The ground truth is the library\'s own Monte Carlo (generated natively with <code>examples/xva_ai_dataset.cpp</code>, committed as <code>web/js/ai/backtest-data.js</code>). ' +
    '<b>The market path is synthetic, not market history, and the page measures surrogates and the tool layer, not the accuracy of any language model.</b>',
});

const { meta } = data;
const usd = (v) => fmt.compact(v);
const bp = (v) => (v === null || v === undefined ? '–' : v.toFixed(2) + ' bp');
const pctS = (v) => (v === null || v === undefined ? '–' : (100 * v).toFixed(1) + '%');

// ------------------------------------------------------------------ 1. back test (automatic, from the committed dataset)

runButton(page, 'Run surrogate back test', async () => {
  const t0 = performance.now();
  page.status.textContent = 'Fitting and back-testing the surrogates in C++…';
  const bt = await run('surrogateBacktest', { dataset: data });
  const day = Array.from(bt.truthTotal, (_, i) => i);
  const mlp = bt.models.mlp, poly = bt.models.poly2, lin = bt.models.linear;
  window.__aiBacktest = { mlpMaeBp: mlp.walkForward.total.maeBp, polyMaeBp: poly.static.total.maeBp };

  tiles(page.content, [
    { label: 'MLP walk-forward error', value: bp(mlp.walkForward.total.maeBp), hint: `mean abs. error of total XVA, % of the ${usd(meta.grossNotional)} gross notional` },
    { label: 'MLP P&L-explain', value: pctS(mlp.walkForward.pnl.effectiveness), hint: 'share of daily XVA P&L variance explained' },
    { label: 'Stress days (outside training box)', value: `${bt.outsideDays} of ${bt.days}`, hint: 'days 150-200 leave the Latin-hypercube range' },
    { label: 'Speed-up', value: `${fmt.num(mlp.speedUp)}×`, hint: `${fmt.num(data.nativeMsPerRun)} ms per Monte Carlo run vs ${mlp.inferMicros.toFixed(1)} µs per surrogate call` },
  ]);

  const g = grid(page.content, true);
  lineChart(card(g, 'Total XVA over the back-test path', 'Engine (full Monte Carlo) against the surrogates fitted once on the training window. Days 150-200 are the stress episode.'), {
    x: day, xLabel: 'day', yLabel: 'total XVA (USD)', zero: true,
    series: [
      { name: 'Monte Carlo engine', y: bt.truthTotal },
      { name: 'Linear', y: lin.static.predTotal, colorIndex: 3, dash: true },
      { name: 'Quadratic ridge', y: poly.static.predTotal, colorIndex: 1, dash: true },
      { name: 'MLP', y: mlp.static.predTotal, colorIndex: 2, dash: true },
    ],
  });
  lineChart(card(g, 'Pricing error (surrogate − engine)', 'Static models. The error grows when the market leaves the training range.'), {
    x: day, xLabel: 'day', yLabel: 'error (USD)', zero: true,
    series: [
      { name: 'Linear', y: Array.from(lin.static.errorTotal), colorIndex: 3 },
      { name: 'Quadratic ridge', y: Array.from(poly.static.errorTotal), colorIndex: 1 },
      { name: 'MLP', y: Array.from(mlp.static.errorTotal), colorIndex: 2 },
      { name: 'MLP walk-forward', y: Array.from(mlp.walkForward.errorTotal), colorIndex: 4 },
    ],
  });

  const rows = [];
  const add = (label, m, extra = {}) => rows.push([label, extra.params ?? '–', bp(m.total.maeBp), bp(m.inRange?.maeBp), bp(m.outRange?.maeBp), pctS(m.total.mape), usd(m.total.max), pctS(m.pnl.effectiveness), pctS(m.pnl.signHit)]);
  add(bt.baseline.lag1.label, bt.baseline.lag1);
  add(bt.baseline.weekly.label, bt.baseline.weekly);
  for (const m of [lin, poly, mlp]) {
    add(`${m.label}, fitted once`, m.static, { params: m.parameters });
    add(`${m.label}, walk-forward`, m.walkForward, { params: m.parameters });
  }
  table(card(page.content, 'Back-test metrics',
    'Error of the total XVA as basis points of gross notional (mean absolute), split into days inside and outside the training box; MAPE and maximum error; P&L-explain effectiveness = 1 − Var(ΔXVA − Δsurrogate) / Var(ΔXVA); sign hit rate of the daily change. ' +
    'Walk-forward refits every 21 days on the training window plus all earlier days (their engine prices are known overnight).'),
  ['model', 'parameters', 'MAE (all)', 'MAE (inside)', 'MAE (stress)', 'MAPE', 'max error', 'P&L explain', 'sign hit'], rows);

  const comps = Object.entries(mlp.walkForward.components);
  barChart(card(page.content, 'Error by adjustment (MLP, walk-forward)', 'Mean absolute error of each adjustment, in USD.'), {
    labels: comps.map(([k]) => k.toUpperCase()), series: [{ name: 'MAE', values: comps.map(([, c]) => c.mae) }],
  });
  table(card(page.content, 'Cost of the models', 'Training once is cheap; the expensive part is the Monte Carlo labels (160 engine runs).'),
    ['model', 'fit time', 'per prediction', 'engine runs to train'],
    [lin, poly, mlp].map((m) => [m.label, `${fmt.num(m.fitMs)} ms`, `${m.inferMicros.toFixed(1)} µs`, `${meta.trainStates}`]));
  return `Done in ${fmt.num(performance.now() - t0)} ms (${bt.days} days, ${meta.trainStates} training states, fitted in C++)`;
});

// ------------------------------------------------------------------ 2. agent skill (static sections)

const extra = el('div');
page.main.append(extra);

// Tool calls price XVA on the engine chosen by the selector (Auto: the fused kernels on WebGPU,
// else on the CPU); the agent layer only sees the library's result.
let lastEngine = '';
const engineRunner = async (fn, spec) => {
  if (fn !== 'xva' && fn !== 'xvaCompare') return run(fn, spec);
  const r = await runAnalysis(fn, spec);
  lastEngine = `${r.label} (${r.result.adapter})`;
  return r.result;
};
engineSelector(page, () => {});

{
  const body = card(extra, 'The agent skill',
    'A skill is a folder an agent can load: instructions (<code>SKILL.md</code>) plus tools. Here the tools are four typed functions over the engine. The model supplies <i>market inputs</i>; the engine supplies <i>numbers</i>. ' +
    'Arguments are validated against the schema before the engine runs, and errors come back as data so the model can correct itself. The same JSON-Schema definitions work as OpenAI function calling, Anthropic tool use, Gemini function declarations or an MCP server\'s <code>tools/list</code>.');
  table(body, ['tool', 'purpose'], TOOLS.map((t) => [el('code', { text: t.name }), t.description]));
  table(body, ['input', 'range', 'default', 'meaning'], Object.entries(STATE_FIELDS).map(([k, f]) => [el('code', { text: k }), `[${f.min}, ${f.max}]`, String(f.default), f.description]));

  const sel = el('select', {}, ['anthropic', 'openai', 'gemini', 'mcp'].map((p) => el('option', { value: p, text: p })));
  const pre = el('pre', { class: 'code' });
  const show = () => { pre.textContent = JSON.stringify(adapters[sel.value](), null, 2); };
  sel.addEventListener('change', show);
  show();
  const download = el('button', { text: 'Download SKILL.md' });
  download.addEventListener('click', () => {
    const a = el('a', { href: URL.createObjectURL(new Blob([SKILL_MD], { type: 'text/markdown' })), download: 'SKILL.md' });
    a.click();
    URL.revokeObjectURL(a.href);
  });
  body.append(el('div', { class: 'toolbar' }, el('label', { class: 'field' }, el('span', { text: 'Tool definitions for' }), sel), download), pre);
  body.append(el('h3', { text: 'SKILL.md' }), el('pre', { class: 'code', text: SKILL_MD }));
}

{
  const body = card(extra, 'Wiring it to a model',
    'The loop is provider neutral: you pass a function <code>llm(messages, tools)</code> that calls your model and returns <code>{ text }</code> or <code>{ toolCalls }</code>. Everything else (validation, execution, the error-recovery loop) is in <code>web/js/ai/skill.js</code>.');
  body.append(el('pre', {
    class: 'code',
    text: `import { runAgentLoop, adapters } from './js/ai/skill.js';
import { run } from './js/ccr-client.js';       // the WebAssembly engine

const llm = async (messages, tools) => {
  // 1. translate messages to your provider's format, send adapters.openai(tools) (or anthropic/gemini/mcp)
  // 2. return { text } for a final answer, or
  //           { toolCalls: [{ id, name, arguments }] } when the model wants a tool
};

const { answer } = await runAgentLoop({
  llm, runner: (fn, spec) => run(fn, spec),
  prompt: 'What happens to our XVA if counterparty spreads double?',
});`,
  }));
  body.append(el('p', { class: 'note', html: 'No API key is entered or stored on this page and nothing is sent anywhere: the page runs the tools locally. ' +
    'Examples of prompts and the calls a capable model should make:' }));
  table(body, ['prompt', 'expected tool call'], [
    ['What is our total XVA today?', el('code', { text: 'price_xva {}' })],
    ['What if counterparty CDS spreads double?', el('code', { text: 'compare_scenarios {"shock":{"cptySpreadMultiplier":2}}' })],
    ['Rates +50bp and FX vol at 15%: how much does MVA change?', el('code', { text: 'compare_scenarios {"shock":{"rateShift":0.005,"fxVol":0.15}}' })],
    ['Which trade drives the CVA?', el('code', { text: 'allocate_xva {}' })],
    ['How much do we lose per bp of funding spread?', el('code', { text: 'xva_sensitivities {}  → fundingSpread01' })],
  ]);
}

{
  const body = card(extra, 'Try the tools on the engine',
    'Paste a tool call, or use the scripted replay (a fixed script standing in for a model: <b>it is not a language model</b>). Each call runs the XVA engine in this tab, on the engine chosen above (Auto: the fused kernels on WebGPU).');
  const name = el('select', {}, TOOLS.map((t) => el('option', { value: t.name, text: t.name })));
  const args = el('textarea', { rows: 4, style: 'width:100%;font-family:monospace;font-size:12px', text: '{"cptySpreadMultiplier": 2, "fxVol": 0.15}' });
  const out = el('pre', { class: 'code', text: '(no call yet)' });
  const status = el('span', { class: 'status' });
  const guard = async (btn, fn) => {
    btn.disabled = true; status.className = 'status'; status.textContent = 'Running…';
    try { await fn(); status.textContent = 'Done'; } catch (e) { status.className = 'status error'; status.textContent = 'Error: ' + e.message; } finally { btn.disabled = false; }
  };
  const call = el('button', { class: 'primary', text: 'Run tool call' });
  call.addEventListener('click', () => guard(call, async () => {
    let parsed;
    try { parsed = JSON.parse(args.value); } catch (e) { out.textContent = JSON.stringify({ ok: false, error: 'arguments are not valid JSON: ' + e.message }, null, 2); return; }
    const t0 = performance.now();
    const r = await executeTool(name.value, parsed, engineRunner);
    out.textContent = JSON.stringify(r, null, 2) + `\n// ${fmt.num(performance.now() - t0)} ms · ${lastEngine}`;
  }));
  const replay = el('button', { text: 'Replay a scripted agent session' });
  replay.addEventListener('click', () => guard(replay, async () => {
    const log = [];
    const r = await runAgentLoop({
      runner: engineRunner, prompt: 'What happens to our XVA if the counterparty\'s CDS spreads double? (volatility is 10%)',
      llm: scriptedLlm([
        { text: 'I will compare the base market with doubled counterparty spreads.', toolCalls: [{ id: 'c1', name: 'compare_scenarios', arguments: { shock: { cptySpreadMultiplier: 2, fxVol: 10 } } }] },
        { text: 'The volatility was given in percent; the tool takes decimals. Retrying.', toolCalls: [{ id: 'c2', name: 'compare_scenarios', arguments: { shock: { cptySpreadMultiplier: 2 } } }] },
        { text: 'FINAL: doubling counterparty spreads raises CVA and the total as shown in the last tool result (every figure above comes from the engine).' },
      ]),
      onEvent: (e) => log.push(e.type === 'assistant' ? `assistant: ${e.reply.text}` : `tool ${e.call.name}(${JSON.stringify(e.call.arguments)}) → ${JSON.stringify(e.result).slice(0, 420)}`),
    });
    out.textContent = log.join('\n\n') + `\n\ncompleted: ${r.completed} in ${r.steps} steps`;
  }));
  body.append(el('div', { class: 'toolbar' }, el('label', { class: 'field' }, el('span', { text: 'Tool' }), name), call, replay, status), args, out);
}

{
  const body = card(extra, 'Tool-layer harness',
    'Eleven conformance cases: valid calls must succeed and satisfy the XVA identity; malformed calls (percent instead of decimals, unknown inputs, strings, unknown tools) must fail with an error a model can act on. ' +
    'The pass rate measures this harness, not a language model.');
  const status = el('span', { class: 'status' });
  const host = el('div');
  const btn = el('button', { class: 'primary', text: 'Run harness on the engine' });
  btn.addEventListener('click', async () => {
    btn.disabled = true; status.className = 'status'; status.textContent = 'Running…'; host.innerHTML = '';
    try {
      const r = await runHarness(engineRunner);
      table(host, ['case', 'tool', 'expected', 'result', 'detail'], r.rows.map((x) => [x.name, x.tool, x.expect, x.pass ? '✓ pass' : '✗ fail', x.detail]));
      status.textContent = `${r.passed}/${r.total} passed`;
    } catch (e) { status.className = 'status error'; status.textContent = 'Error: ' + e.message; } finally { btn.disabled = false; }
  });
  body.append(el('div', { class: 'toolbar' }, btn, status), host);
}

{
  const body = card(extra, 'Live spot check of the stored ground truth',
    'Re-prices a few days of the back-test path with the engine in this browser and compares with the stored native result. They use different normal generators (<code>std::normal_distribution</code> is implementation defined; the fused kernels have their own), so they agree to Monte Carlo accuracy, not to the last digit.');
  const status = el('span', { class: 'status' });
  const host = el('div');
  const btn = el('button', { text: 'Spot check 4 days on the engine' });
  btn.addEventListener('click', async () => {
    btn.disabled = true; status.className = 'status'; status.textContent = 'Running…'; host.innerHTML = '';
    try {
      const rows = [];
      for (const d of [0, 100, 175, 249]) {
        const [mc, mo, dr, fx, fs] = data.backtest.x[d];
        const r = await executeTool('price_xva', { cptySpreadMultiplier: mc, ownSpreadMultiplier: mo, rateShift: dr, fxVol: fx, fundingSpread: fs }, engineRunner);
        if (!r.ok) throw new Error(r.error);
        const truth = data.backtest.y[d][6];
        rows.push([`day ${d}`, fmt.num(truth), fmt.num(r.xva.total), pctS((r.xva.total - truth) / truth)]);
      }
      table(host, ['day', 'stored total XVA', 'engine now', 'difference'], rows);
      status.textContent = `Done · ${lastEngine}`;
    } catch (e) { status.className = 'status error'; status.textContent = 'Error: ' + e.message; } finally { btn.disabled = false; }
  });
  body.append(el('div', { class: 'toolbar' }, btn, status), host);
}

{
  const body = card(extra, 'How to read the back test');
  body.append(el('ul', { html:
    '<li><b>Synthetic path.</b> Five mean-reverting market drivers with a stress episode (counterparty spread ×4.6, FX vol 23.5%, rates −125bp) that exceeds the training box. It is a controlled test of extrapolation, not an estimate of live performance.</li>' +
    '<li><b>Common random numbers.</b> The engine uses the same seed for every state, so XVA is a smooth deterministic function of the market and the surrogate learns the model, not Monte Carlo noise. With fresh noise per label, errors would be larger.</li>' +
    '<li><b>Baselines matter.</b> Yesterday\'s engine value is a strong benchmark for a smooth day; surrogates win by being <i>current</i> intraday and by explaining the daily P&L (the baselines explain none of it).</li>' +
    '<li><b>Extrapolation.</b> Polynomials degrade outside the training range; the network degraded least here, but a network has no guarantee outside the data. Production use needs a range check and a fallback to the engine, which is what the stress-day column measures.</li>' +
    '<li><b>One netting set, five inputs.</b> Real portfolios have hundreds of risk factors and trades that change daily; see chapter X10 for what that implies.</li>' +
    '' }));
}
