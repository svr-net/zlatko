// LLM agent skill for the XVA engine (no DOM, runs in Node and in the browser).
//
// Four layers, all provider neutral:
//   1. TOOLS      JSON-Schema tool definitions (function calling / MCP) with adapters for the OpenAI,
//                 Anthropic, Gemini and MCP formats.
//   2. executeTool  validates the arguments of a tool call and runs it through an injected `runner`
//                 (fn, spec) => result. In the browser the runner is the library's XVA analysis on
//                 the fused kernels (gpu/backend.js runAnalysis). The answer is a compact JSON object whose numbers the model can quote,
//                 with the invariants (total = cva - dva + fca - fba + mva + kva) checked.
//   3. SKILL_MD   the same knowledge as an Anthropic-style Agent Skill (SKILL.md with frontmatter).
//   4. runAgentLoop  the generic tool-use loop. The model is a callback `llm(messages, tools)`, so any
//                 provider can be plugged in; scriptedLlm is a deterministic stand-in used in tests (not an LLM).

import { defaultSpec } from '../spec.js';

// ----------------------------------------------------------------------------- market state

export const STATE_FIELDS = {
  cptySpreadMultiplier: { min: 0.2, max: 6, default: 1, description: 'Multiplier applied to every counterparty CDS quote (1 = base curve 70-140bp).' },
  ownSpreadMultiplier: { min: 0.2, max: 4, default: 1, description: 'Multiplier applied to the own (bank) 5y CDS quote (base 60bp).' },
  rateShift: { min: -0.02, max: 0.04, default: 0, description: 'Parallel shift of the USD zero curve as a decimal (0.01 = +100bp).' },
  fxVol: { min: 0.02, max: 0.4, default: 0.1, description: 'EURUSD lognormal volatility as a decimal (0.10 = 10%).' },
  fundingSpread: { min: 0, max: 0.05, default: 0.01, description: 'Unsecured funding spread as a decimal; borrow = spread, lend = half of it, margin funding = spread.' },
};

/** A market state as an engine specification: the default netting set plus the state, which the
 *  library applies (C++ withMarketState, as price() of examples/xva_ai_dataset.cpp). */
export function stateToSpec(state = {}) {
  const s = {};
  for (const [k, f] of Object.entries(STATE_FIELDS)) s[k] = state[k] ?? f.default;
  // The unconditional IM proxy, as in examples/xva_ai_dataset.cpp: the agent's engine prices and the
  // surrogates' ground truth then share one MVA definition (the XVA page defaults to nested IM).
  return { ...defaultSpec(), imMethod: 'proxy', marketState: s };
}

// ----------------------------------------------------------------------------- tools

const stateSchema = {
  type: 'object',
  properties: Object.fromEntries(Object.entries(STATE_FIELDS).map(([k, f]) => [k, { type: 'number', minimum: f.min, maximum: f.max, description: f.description }])),
  additionalProperties: false,
};

export const TOOLS = [
  {
    name: 'price_xva',
    description: 'Prices CVA, DVA, FCA, FBA, MVA, KVA and the total XVA of the default 4-trade netting set (two swaps, an FX forward, a short FX option) ' +
      'by full Monte Carlo (2000 paths, 10y). All amounts are in USD. Costs are positive; DVA and FBA are benefits. total = cva - dva + fca - fba + mva + kva. Omitted inputs take the base market.',
    input_schema: stateSchema,
  },
  {
    name: 'xva_sensitivities',
    description: 'XVA and its first-order risk: CS01 of the counterparty and own CDS curves per maturity bucket (value change for +1bp) and funding spread 01 (per +1bp).',
    input_schema: stateSchema,
  },
  {
    name: 'allocate_xva',
    description: 'Allocates CVA, FCA and KVA of the netting set to its trades by marginal (Euler) contributions; the parts add up to the netting-set values.',
    input_schema: stateSchema,
  },
  {
    name: 'compare_scenarios',
    description: 'Prices a base market and a shocked market (the base with the given overrides) and reports the change of every adjustment. Use it for what-if questions such as "what if counterparty spreads double".',
    input_schema: {
      type: 'object',
      properties: { base: stateSchema, shock: { ...stateSchema, description: 'Overrides applied on top of base.' } },
      required: ['shock'],
      additionalProperties: false,
    },
  },
];

export class ToolError extends Error {}

function validateState(args, label = 'arguments') {
  if (args === null || typeof args !== 'object' || Array.isArray(args)) throw new ToolError(`${label} must be an object`);
  for (const [k, v] of Object.entries(args)) {
    const f = STATE_FIELDS[k];
    if (!f) throw new ToolError(`unknown input "${k}"; allowed: ${Object.keys(STATE_FIELDS).join(', ')}`);
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new ToolError(`${k} must be a finite number`);
    if (v < f.min || v > f.max) throw new ToolError(`${k}=${v} is outside the supported range [${f.min}, ${f.max}]`);
  }
  return args;
}

const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
const arr = (a) => Array.from(a ?? []);

function summarise(r) {
  const x = r.xva;
  return {
    currency: 'USD',
    cva: round(x.cva), dva: round(x.dva), fca: round(x.fca), fba: round(x.fba), fva: round(x.fva),
    mva: round(x.mva), kva: round(x.kva), total: round(x.total),
    invariant: { identity: 'total = cva - dva + fca - fba + mva + kva', holds: r.identityHolds },  // checked by the library
  };
}

/** Runs one tool call. Never throws: errors come back as { ok: false, error } so the model can correct itself. */
export async function executeTool(name, args, runner) {
  try {
    const tool = TOOLS.find((t) => t.name === name);
    if (!tool) throw new ToolError(`unknown tool "${name}"; available: ${TOOLS.map((t) => t.name).join(', ')}`);
    args = args ?? {};
    if (name === 'compare_scenarios') {
      if (typeof args !== 'object' || Array.isArray(args)) throw new ToolError('arguments must be an object');
      for (const k of Object.keys(args)) if (k !== 'base' && k !== 'shock') throw new ToolError(`unknown input "${k}"; allowed: base, shock`);
      if (args.shock === undefined) throw new ToolError('"shock" is required');
      const base = validateState(args.base ?? {}, 'base'), shock = validateState(args.shock, 'shock');
      // One library call prices both markets (two jobs) and returns the changes.
      const r = await runner('xvaCompare', { ...stateToSpec(base), shockState: stateToSpec({ ...base, ...shock }).marketState });
      const change = Object.fromEntries(Object.entries(r.change).map(([k, v]) => [k, round(v)]));
      return { ok: true, tool: name, base: summarise(r.base), shocked: summarise(r.shocked), change };
    }
    const state = validateState(args);
    const r = await runner('xva', stateToSpec(state));
    const out = { ok: true, tool: name, state: { ...Object.fromEntries(Object.entries(STATE_FIELDS).map(([k, f]) => [k, f.default])), ...state }, xva: summarise(r) };
    if (name === 'xva_sensitivities') {
      const s = r.sensitivities;
      if (!s) throw new ToolError('the engine returned no sensitivities');
      out.sensitivities = {
        maturities: arr(s.maturities), counterpartyCs01: arr(s.counterpartyCs01).map((v) => round(v)), ownCs01: arr(s.ownCs01).map((v) => round(v)),
        fundingSpread01: round(s.fundingSpread01), unit: 'USD per +1bp',
      };
    }
    if (name === 'allocate_xva') {
      out.allocation = (r.trades ?? []).map((t) => ({ trade: t.id, cva: round(t.cva), fca: round(t.fca), kva: round(t.kva) }));
    }
    return out;
  } catch (e) {
    return { ok: false, tool: name, error: e instanceof ToolError ? e.message : `engine error: ${e?.message ?? e}` };
  }
}

// ----------------------------------------------------------------------------- provider adapters

export const adapters = {
  openai: (tools = TOOLS) => tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } })),
  anthropic: (tools = TOOLS) => tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.input_schema })),
  gemini: (tools = TOOLS) => [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: toGeminiSchema(t.input_schema) })) }],
  mcp: (tools = TOOLS) => ({ tools: tools.map((t) => ({ name: t.name, description: t.description, inputSchema: t.input_schema })) }),
};

// Gemini's schema dialect has upper-case types and no additionalProperties.
function toGeminiSchema(s) {
  if (Array.isArray(s)) return s.map(toGeminiSchema);
  if (s === null || typeof s !== 'object') return s;
  const o = {};
  for (const [k, v] of Object.entries(s)) {
    if (k === 'additionalProperties') continue;
    o[k] = k === 'type' && typeof v === 'string' ? v.toUpperCase() : toGeminiSchema(v);
  }
  return o;
}

// ----------------------------------------------------------------------------- Agent Skill document

export const SKILL_MD = `---
name: xva-pricing
description: Prices and explains XVA (CVA, DVA, FCA/FBA, MVA, KVA) of a derivatives netting set with a Monte Carlo engine, runs what-if market scenarios, and reports sensitivities and trade allocation. Use when asked about counterparty credit risk, funding, margin or capital costs of a derivatives portfolio.
---

# XVA pricing skill

## When to use
Questions about the cost of counterparty credit, funding, initial margin or capital for the demo netting set
(IRS 10y payer 10m, IRS 5y receiver 6m, EURUSD forward 3y 5m, short EURUSD call 2y 4m), or how those costs
move with the market. Do not use it for other portfolios: the engine prices exactly this set.

## Tools
${TOOLS.map((t) => `- \`${t.name}\`: ${t.description}`).join('\n')}

Inputs (all optional, decimals not percent):
${Object.entries(STATE_FIELDS).map(([k, f]) => `- \`${k}\` in [${f.min}, ${f.max}], default ${f.default}: ${f.description}`).join('\n')}

## Rules
1. Never quote an XVA number that did not come from a tool result. If a tool returns \`ok: false\`, read the error, fix the arguments and retry once; otherwise say what failed.
2. Convert units before calling: "+50bp" is \`rateShift: 0.005\`; "spreads double" is \`cptySpreadMultiplier: 2\`.
3. Costs are positive. DVA and FBA are benefits and are subtracted: total = CVA - DVA + FCA - FBA + MVA + KVA. Check \`invariant.holds\`.
4. Results are Monte Carlo with fixed random numbers (2000 paths, seed 2024): differences between scenarios are reliable, absolute levels carry sampling error of a few percent.
5. State assumptions: no CSA, 10-year horizon, USD amounts, flat recovery 40%.
6. Do not give trading or hedging advice as fact; present sensitivities as model output.

## Example
User: "What happens to our XVA if the counterparty's CDS spreads double?"
1. Call \`compare_scenarios\` with \`{"shock": {"cptySpreadMultiplier": 2}}\`.
2. Report base total, shocked total and the change per adjustment (CVA moves most; check the sign of FCA/KVA changes), and mention the Monte Carlo caveat.
`;

// ----------------------------------------------------------------------------- agent loop

/**
 * Generic tool-use loop. `llm(messages, tools)` returns either { text } (final answer) or
 * { toolCalls: [{ id, name, arguments }] } and may also return text alongside the calls.
 * Messages use a neutral shape: { role: 'user'|'assistant'|'tool', content, toolCalls?, toolCallId?, name? }.
 */
export async function runAgentLoop({ llm, runner, prompt, tools = TOOLS, maxSteps = 6, onEvent = () => {} }) {
  const messages = [{ role: 'user', content: prompt }];
  for (let step = 0; step < maxSteps; step++) {
    const reply = await llm(messages, tools);
    messages.push({ role: 'assistant', content: reply.text ?? '', toolCalls: reply.toolCalls });
    onEvent({ type: 'assistant', step, reply });
    if (!reply.toolCalls?.length) return { answer: reply.text ?? '', messages, steps: step + 1, completed: true };
    for (const call of reply.toolCalls) {
      let args = call.arguments;
      if (typeof args === 'string') { try { args = JSON.parse(args); } catch { args = null; } }
      const result = args === null ? { ok: false, tool: call.name, error: 'arguments are not valid JSON' } : await executeTool(call.name, args, runner);
      messages.push({ role: 'tool', toolCallId: call.id, name: call.name, content: JSON.stringify(result) });
      onEvent({ type: 'tool', step, call, result });
    }
  }
  return { answer: '', messages, steps: maxSteps, completed: false };
}

/** Deterministic stand-in for a model: replays a list of replies. It is NOT a language model. */
export function scriptedLlm(replies) {
  let i = 0;
  return async () => replies[Math.min(i++, replies.length - 1)];
}

// ----------------------------------------------------------------------------- harness checks

/**
 * Conformance cases for the tool layer: what a model's tool calls must achieve. Each case is a call and
 * the expected outcome. The pass rate measures the harness (validation, units, errors), not a model.
 */
export const HARNESS_CASES = [
  { name: 'base price', tool: 'price_xva', args: {}, expect: 'ok' },
  { name: 'stress inputs', tool: 'price_xva', args: { cptySpreadMultiplier: 4, fxVol: 0.2, fundingSpread: 0.02 }, expect: 'ok' },
  { name: 'sensitivities', tool: 'xva_sensitivities', args: { rateShift: 0.005 }, expect: 'ok' },
  { name: 'allocation', tool: 'allocate_xva', args: {}, expect: 'ok' },
  { name: 'what-if', tool: 'compare_scenarios', args: { shock: { cptySpreadMultiplier: 2 } }, expect: 'ok' },
  { name: 'percent instead of decimal', tool: 'price_xva', args: { fxVol: 10 }, expect: 'error', match: /outside the supported range/ },
  { name: 'unknown input', tool: 'price_xva', args: { cdsSpread: 100 }, expect: 'error', match: /unknown input/ },
  { name: 'string number', tool: 'price_xva', args: { rateShift: '50bp' }, expect: 'error', match: /finite number/ },
  { name: 'unknown tool', tool: 'delete_trades', args: {}, expect: 'error', match: /unknown tool/ },
  { name: 'missing shock', tool: 'compare_scenarios', args: { base: {} }, expect: 'error', match: /required/ },
  { name: 'array arguments', tool: 'price_xva', args: [1, 2], expect: 'error', match: /object/ },
];

export async function runHarness(runner) {
  const rows = [];
  for (const c of HARNESS_CASES) {
    const r = await executeTool(c.tool, c.args, runner);
    const pass = c.expect === 'ok' ? r.ok === true && (r.xva ?? r.base).invariant.holds : r.ok === false && c.match.test(r.error);
    rows.push({ name: c.name, tool: c.tool, expect: c.expect, pass: !!pass, detail: r.ok ? 'ok' : r.error });
  }
  return { rows, passed: rows.filter((r) => r.pass).length, total: rows.length };
}
