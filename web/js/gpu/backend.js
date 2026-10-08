// Chooses between the WebAssembly library and the WebGPU fused kernels for the Monte Carlo
// exposure pages, and runs the GPU path: the WASM build compiles the plan (gpuPlan), the GPU
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

/** Phones and tablets (including iPadOS, which reports a desktop user agent). */
export function isMobileDevice() {
  const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
  const ua = navigator.userAgent || '';
  return coarse || /Mobi|Android|iPhone|iPad|iPod/i.test(ua) || (navigator.maxTouchPoints > 1 && /Macintosh/.test(ua));
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
 * Auto uses WebGPU on phones and tablets that support it, WebAssembly elsewhere.
 */
export async function chooseEngine(spec) {
  const mode = getEngineMode();
  if (mode === 'wasm') return { engine: 'wasm', reason: 'WebAssembly selected' };
  if (mode === 'auto' && !isMobileDevice()) return { engine: 'wasm', reason: 'Auto: WebAssembly on desktop' };
  if (!('gpu' in navigator)) return { engine: 'wasm', reason: 'WebGPU is not available in this browser' };
  const limitation = gpuLimitation(spec);
  if (limitation) return { engine: 'wasm', reason: limitation };
  try {
    await gpuEngine();
  } catch (e) {
    return { engine: 'wasm', reason: `WebGPU could not start (${e.message})` };
  }
  return { engine: 'gpu', reason: mode === 'auto' ? 'Auto: WebGPU on mobile' : 'WebGPU selected' };
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
      text: value === 'auto' ? 'Auto (WebGPU on mobile)' : label,
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
