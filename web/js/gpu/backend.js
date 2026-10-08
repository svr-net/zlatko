// Runs a Monte Carlo analysis of the library on the WebAssembly build or on the WebGPU
// fused kernels. Auto picks WebGPU whenever the browser supports it and the library can
// compile the portfolio for the kernels.
//
// GPU path: gpuJobs (C++ compiles the plans) -> GpuEngine.run per plan (upload, dispatch,
// read back) -> gpuAnalyse (C++ turns the read-backs into the same result as the WASM
// analysis). This file only routes calls; all numerics are in the C++ library.
import { run } from '../ccr-client.js';
import { el } from '../ui.js';
import { GpuEngine } from './engine.js';

const STORAGE_KEY = 'ccr-engine';
const MODES = { auto: 'Auto', gpu: 'WebGPU', wasm: 'WebAssembly' };

export function getEngineMode() {
  try { return MODES[localStorage.getItem(STORAGE_KEY)] ? localStorage.getItem(STORAGE_KEY) : 'auto'; } catch (_) { return 'auto'; }
}

export function setEngineMode(mode) {
  try { localStorage.setItem(STORAGE_KEY, mode); } catch (_) { /* storage unavailable */ }
}

let engine = null;
export function gpuEngine() {
  if (!engine) engine = run('gpuKernels', {}).then((k) => GpuEngine.create(k)).catch((e) => { engine = null; throw e; });
  return engine;
}

/** Runs every plan of gpuJobs on the GPU and hands the read-backs to gpuAnalyse. */
export async function runOnGpu(analysis, spec, { warmUp = false } = {}) {
  const jobs = await run('gpuJobs', { ...spec, analysis });
  if (jobs.unsupported) return { unsupported: jobs.unsupported };
  const gpu = await gpuEngine();
  if (warmUp && jobs.plans[0]) await gpu.run(jobs.plans[0]); // pipeline and driver caches
  const gpuOutputs = [];
  let gpuMs = 0;
  for (const plan of jobs.plans) {
    const out = plan ? await gpu.run(plan) : null;
    gpuOutputs.push(out && { sums: out.sums, pfe: out.pfe });
    gpuMs += out ? out.gpuMs : 0;
  }
  const result = await run('gpuAnalyse', { ...spec, analysis, gpuOutputs });
  const first = jobs.plans.find(Boolean);
  return { result: { ...result, gpuMs, adapter: gpu.name, plans: jobs.plans.length, compileMs: jobs.compileMs, numTerms: first?.numTerms, numDates: first?.numDates } };
}

/**
 * Runs `analysis` (a library entry point: exposure, collateral, cva, wrongWayRisk, hedging)
 * on the engine chosen by the selector. Resolves to { result, engine, reason }; a GPU that
 * fails or a portfolio the kernels cannot price falls back to WebAssembly.
 */
export async function runAnalysis(analysis, spec) {
  const mode = getEngineMode();
  const wasm = async (reason) => ({ result: await run(analysis, spec), engine: 'wasm', reason });
  if (mode === 'wasm') return wasm('WebAssembly selected');
  if (!('gpu' in navigator)) return wasm('WebGPU is not available in this browser');
  try {
    const g = await runOnGpu(analysis, spec);
    if (g.unsupported) return wasm(g.unsupported);
    return { ...g, engine: 'gpu', reason: mode === 'auto' ? 'Auto: WebGPU available' : 'WebGPU selected' };
  } catch (e) {
    console.warn('WebGPU run failed, falling back to WebAssembly:', e);
    // No adapter is a property of the device, not a failure of the run: say so plainly.
    if (/no WebGPU adapter/.test(e.message)) return wasm(`WebGPU unavailable: ${e.message.replace('no WebGPU adapter: ', '')}`);
    return wasm(`WebGPU failed (${e.message}); used WebAssembly`);
  }
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

/** One-line note for the status bar: which engine ran and why. */
export function engineNote(r, ms) {
  const name = r.engine === 'gpu' ? `WebGPU (${r.result.adapter})` : 'WebAssembly';
  return `Done in ${Math.round(ms).toLocaleString('en-US')} ms · ${name} · ${r.reason}`;
}

/** Card shown in GPU mode listing what needs path-level data from the WASM engine. */
export function gpuScopeNote(parent, items) {
  parent.append(el('p', { class: 'note engine-note' },
    el('b', { text: 'Computed on WebGPU. ' }),
    `The fused kernels aggregate on the GPU and never return individual paths, so ${items} are only available with the WebAssembly engine.`));
}
