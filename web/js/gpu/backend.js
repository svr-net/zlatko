// Runs a Monte Carlo analysis of the library on one of three engines:
//   WebGPU       the fused kernels on the GPU;
//   CPU kernels  the same fused kernels on the CPU, workgroups split across Web Workers;
//   WebAssembly  the library's ExposureEngine (path-level views, AMC).
// Auto tries them in that order: WebGPU where the browser offers an adapter, the CPU kernels
// where it does not, WebAssembly for portfolios the kernels cannot price (Bermudans).
//
// GPU path: gpuJobs (C++ compiles the plans) -> GpuEngine.run per plan (upload, dispatch,
// read back) -> gpuAnalyse. CPU kernel path: cpuKernelSlices on every worker -> cpuKernelAnalyse.
// Both end in the same C++ analysis; this file only routes calls.
import { callOn, run, workerCount } from '../ccr-client.js';
import { el } from '../ui.js';
import { GpuEngine } from './engine.js';

const STORAGE_KEY = 'ccr-engine';
const MODES = { auto: 'Auto', gpu: 'WebGPU', cpu: 'CPU fused kernels', wasm: 'WebAssembly' };
const LABELS = { gpu: 'WebGPU', cpu: 'CPU fused kernels', wasm: 'WebAssembly' };

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
 * The fused kernels on the CPU: every worker runs its share of the workgroups of every job
 * (cpuKernelSlices), then the library finishes and analyses them (cpuKernelAnalyse). The
 * result does not depend on the number of workers.
 */
export async function runOnCpuKernels(analysis, spec) {
  const n = workerCount();
  const t0 = performance.now();
  const parts = (await Promise.all(Array.from({ length: n }, (_, part) =>
    callOn(part, 'cpuKernelSlices', { ...spec, analysis, part, parts: n })))).map((r) => r.result);
  if (parts[0].unsupported) return { unsupported: parts[0].unsupported };
  const kernelMs = performance.now() - t0;
  globalThis.__ccrCpuKernelRuns = (globalThis.__ccrCpuKernelRuns || 0) + 1; // checked by the e2e test
  const result = await run('cpuKernelAnalyse', { ...spec, analysis, parts });
  return { result: { ...result, gpuMs: kernelMs, adapter: `${n} worker${n === 1 ? '' : 's'}`, workers: n } };
}

const outcome = (engine, result, reason) => ({ result, engine, fused: engine !== 'wasm', label: LABELS[engine], reason });

/**
 * Runs `analysis` (a library entry point: exposure, collateral, cva, wrongWayRisk, hedging)
 * on the engine chosen by the selector. Resolves to { result, engine, fused, label, reason }.
 */
export async function runAnalysis(analysis, spec) {
  const mode = getEngineMode();
  const wasm = async (reason) => outcome('wasm', await run(analysis, spec), reason);
  const cpu = async (why) => {
    try {
      const c = await runOnCpuKernels(analysis, spec);
      if (c.unsupported) return wasm(c.unsupported);
      return outcome('cpu', c.result, why);
    } catch (e) {
      console.warn('CPU kernels failed, falling back to WebAssembly:', e);
      return wasm(`CPU kernels failed (${e.message}); used WebAssembly`);
    }
  };
  if (mode === 'wasm') return wasm('WebAssembly selected');
  if (mode === 'cpu') return cpu('CPU fused kernels selected');
  if (!('gpu' in navigator)) return cpu('WebGPU is not available in this browser');
  try {
    const g = await runOnGpu(analysis, spec);
    if (g.unsupported) return wasm(g.unsupported);
    return outcome('gpu', g.result, mode === 'auto' ? 'Auto: WebGPU available' : 'WebGPU selected');
  } catch (e) {
    console.warn('WebGPU run failed, falling back to the CPU kernels:', e);
    // No adapter is a property of the device, not a failure of the run: say so plainly.
    if (/no WebGPU adapter|too limited/.test(e.message)) return cpu(`WebGPU unavailable: ${e.message.replace('no WebGPU adapter: ', '')}`);
    return cpu(`WebGPU failed (${e.message})`);
  }
}

/** Toolbar control: Engine [Auto | WebGPU | WebAssembly]. Changing it re-runs via `onChange`. */
export function engineSelector(page, onChange) {
  const select = el('select', { 'aria-label': 'Compute engine' },
    Object.entries(MODES).map(([value, label]) => el('option', {
      value, selected: value === getEngineMode(),
      text: value === 'auto' ? 'Auto (WebGPU, else CPU kernels)' : label,
    })));
  select.addEventListener('change', () => { setEngineMode(select.value); onChange(); });
  page.toolbar.insertBefore(el('label', { class: 'engine-select' }, 'Engine ', select), page.status);
  return select;
}

/** One-line note for the status bar: which engine ran and why. */
export function engineNote(r, ms) {
  const name = r.engine === 'wasm' ? 'WebAssembly' : `${r.label} (${r.result.adapter})`;
  return `Done in ${Math.round(ms).toLocaleString('en-US')} ms · ${name} · ${r.reason}`;
}

/** Card shown for the fused kernels listing what needs path-level data from the WASM engine. */
export function gpuScopeNote(parent, items, label = 'WebGPU') {
  parent.append(el('p', { class: 'note engine-note' },
    el('b', { text: `Computed with the ${label}. ` }),
    `The fused kernels aggregate per workgroup and never return individual paths, so ${items} are only available with the WebAssembly engine.`));
}
