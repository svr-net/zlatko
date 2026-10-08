// Chooses between the WebAssembly library and the WebGPU fused kernels for the Monte Carlo
// pages (Auto: WebGPU wherever the browser has it), and runs the GPU path: the WASM build compiles the plan (gpuPlan), the GPU
// simulates, prices and aggregates, and WASM applies the closed-form CVA analytics
// (cvaFromProfile) so results have the same shape whichever engine produced them.
import { run } from '../ccr-client.js';
import { el } from '../ui.js';
import { GpuEngine } from './engine.js';

const STORAGE_KEY = 'ccr-engine';
const MODES = { auto: 'Auto', gpu: 'WebGPU', wasm: 'WebAssembly' };
const GPU_TRADE_TYPES = new Set(['swap', 'forward', 'option']);

export function getEngineMode() {
  try { return MODES[localStorage.getItem(STORAGE_KEY)] ? localStorage.getItem(STORAGE_KEY) : 'auto'; } catch (_) { return 'auto'; }
}

export function setEngineMode(mode) {
  try { localStorage.setItem(STORAGE_KEY, mode); } catch (_) { /* storage unavailable */ }
}

let engine = null;
export function gpuEngine() {
  if (!engine) engine = GpuEngine.create().catch((e) => { engine = null; throw e; });
  return engine;
}

/** Why the GPU path cannot run this spec, or null if it can. */
export function gpuLimitation(spec) {
  const amc = spec.trades.filter((t) => !GPU_TRADE_TYPES.has(t.type));
  if (amc.length) return `${amc.map((t) => t.id).join(', ')} need${amc.length === 1 ? 's' : ''} American Monte Carlo, which runs in WebAssembly only`;
  if (spec.assets.length > 4) return 'the GPU kernel supports at most 4 FX/equity assets';
  if (spec.credits.length > 2) return 'the GPU kernel supports at most 2 credit curves';
  if (spec.credits.some((c) => c.cir.substeps !== spec.credits[0].cir.substeps)) return 'the GPU kernel needs the same CIR sub-steps for all credits';
  return null;
}

/**
 * Decides the engine for a run: { engine: 'gpu' | 'wasm', reason }.
 * Auto uses the WebGPU kernels whenever the browser supports WebGPU and the portfolio fits them,
 * on desktop and mobile alike, and WebAssembly otherwise.
 */
export async function chooseEngine(spec) {
  const mode = getEngineMode();
  if (mode === 'wasm') return { engine: 'wasm', reason: 'WebAssembly selected' };
  if (!('gpu' in navigator)) return { engine: 'wasm', reason: 'WebGPU is not available in this browser' };
  const limitation = gpuLimitation(spec);
  if (limitation) return { engine: 'wasm', reason: limitation };
  try {
    await gpuEngine();
  } catch (e) {
    return { engine: 'wasm', reason: `WebGPU could not start (${e.message})` };
  }
  return { engine: 'gpu', reason: mode === 'auto' ? 'Auto: WebGPU available' : 'WebGPU selected' };
}

/** Exposure profile simulated on the GPU, in the same shape as the WASM profile. */
export async function gpuExposure(spec) {
  const plan = await run('gpuPlan', spec);
  if (plan.unsupported.length) throw new Error('not supported on the GPU: ' + plan.unsupported.join(', '));
  const gpu = await gpuEngine();
  const r = await gpu.run(plan);
  return { ...r, plan, adapter: [gpu.info.vendor, gpu.info.architecture].filter(Boolean).join(' ') || 'GPU' };
}

/** CVA on the GPU: exposure from the kernels, analytics from the WASM library. */
export async function gpuCva(spec) {
  const g = await gpuExposure(spec);
  const analytics = await run('cvaFromProfile', { ...spec, profile: g.profile });
  return { ...analytics, pathwiseCva: g.pathwiseCva, gpu: g };
}

/**
 * The hedging analysis on the GPU, in the same shape as the WASM `hedging` result. The kernels
 * simulate the base and every bumped exposure profile with the same seed (common random numbers)
 * or with other seeds; CS01s, the hedge Jacobian and the spread scenarios come from the WASM
 * library applied to the base profile. Bumped CVAs use the base counterparty curve, as in WASM.
 */
export async function gpuHedging(spec, bumps) {
  const base = await gpuExposure(spec);
  const credit = await run('creditHedgingFromProfile', { ...spec, profile: base.profile });
  const surv = base.profile.marketSurvival, lgd = 1 - base.plan.recovery;
  const cvaOf = async (s) => {
    const e = (await gpuExposure(s)).profile.discountedEe;
    let cva = 0;
    for (let k = 1; k < e.length; k++) cva += lgd * 0.5 * (e[k - 1] + e[k]) * (surv[k - 1] - surv[k]);
    return cva;
  };
  const withSeed = (s, seed) => ({ ...s, sim: { ...s.sim, seed } });
  const seed = spec.sim.seed;
  const assetDeltas = [];
  for (let a = 0; a < spec.assets.length; a++) {
    const asset = spec.assets[a];
    const bumped = (b) => ({ ...spec, assets: spec.assets.map((x, i) => (i === a ? { ...x, spot: x.spot + b } : x)) });
    const crn = [], independent = [];
    for (const h of bumps) {
      crn.push((await cvaOf(bumped(h)) - await cvaOf(bumped(-h))) / (2 * h));
      independent.push((await cvaOf(withSeed(bumped(h), seed + 1)) - await cvaOf(withSeed(bumped(-h), seed + 2))) / (2 * h));
    }
    assetDeltas.push({ name: asset.name, bumps, deltaCrn: crn, deltaIndependent: independent, carryDiscountAtHorizon: credit.carryDiscountAtHorizon[a] });
  }
  const shifted = (bp) => ({ ...spec, domestic: shiftCurve(spec.domestic, bp * 1e-4) });
  const cvaDv01 = 0.5 * (await cvaOf(shifted(1)) - await cvaOf(shifted(-1)));
  return { ...credit, assetDeltas, cvaDv01, adapter: base.adapter };
}

// Parallel shift of the zero rates, as ccr::YieldCurve::shifted.
function shiftCurve(curve, shift) {
  if (!curve) return { flat: 0.03 + shift };
  if (curve.flat !== undefined) return { flat: curve.flat + shift };
  return { times: curve.times, rates: curve.rates.map((r) => r + shift) };
}

/**
 * Runs `tasks.gpu` or `tasks.wasm` according to the engine choice. If the GPU path throws,
 * falls back to WebAssembly. Resolves to { result, engine, reason }.
 */
export async function runWithEngine(spec, tasks) {
  const choice = await chooseEngine(spec);
  if (choice.engine === 'gpu') {
    try {
      return { result: await tasks.gpu(), ...choice };
    } catch (e) {
      console.warn('WebGPU run failed, falling back to WebAssembly:', e);
      return { result: await tasks.wasm(), engine: 'wasm', reason: `WebGPU failed (${e.message}); used WebAssembly` };
    }
  }
  return { result: await tasks.wasm(), ...choice };
}

/** Toolbar control: Engine [Auto | WebGPU | WebAssembly]. Changing it re-runs via `onChange`. */
export function engineSelector(page, onChange) {
  const select = el('select', { 'aria-label': 'Compute engine' },
    Object.entries(MODES).map(([value, label]) => el('option', {
      value, selected: value === getEngineMode(),
      text: value === 'auto' ? 'Auto (WebGPU if available)' : label,
    })));
  select.addEventListener('change', () => { setEngineMode(select.value); onChange(); });
  page.toolbar.insertBefore(el('label', { class: 'engine-select' }, 'Engine ', select), page.status);
  return select;
}

/** One-line note for the status bar and the page: which engine ran and why. */
export function engineNote(run, ms) {
  const name = run.engine === 'gpu' ? `WebGPU${run.result?.gpu?.adapter || run.result?.adapter ? ` (${run.result.gpu?.adapter || run.result.adapter})` : ''}` : 'WebAssembly';
  return `Done in ${Math.round(ms).toLocaleString('en-US')} ms · ${name} · ${run.reason}`;
}

/** Card shown in GPU mode listing what needs path-level data from the WASM engine. */
export function gpuScopeNote(parent, items) {
  parent.append(el('p', { class: 'note engine-note' },
    el('b', { text: 'Computed on WebGPU. ' }),
    `The fused kernels aggregate on the GPU and never return individual paths, so ${items} are only available with the WebAssembly engine.`));
}
