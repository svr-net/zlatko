// Runs a Monte Carlo analysis of the library on the fused kernels, on one of two engines:
//   WebGPU       the fused kernels on the GPU;
//   CPU kernels  the same fused kernels in C++ on the CPU (one call).
// Auto uses WebGPU where the browser offers an adapter and the CPU kernels where it does not.
//
// GPU path: gpuJobs (C++ compiles the plans) -> GpuEngine.run per plan (upload, dispatch,
// read back) -> gpuAnalyse. CPU path: cpuKernels. Both end in the same C++ analysis; this file
// only routes calls.
import { run } from '../ccr-client.js';
import { el } from '../ui.js';
import { GpuEngine } from './engine.js';

const STORAGE_KEY = 'ccr-engine';
const MODES = { auto: 'Auto', gpu: 'WebGPU', cpu: 'CPU fused kernels' };
const LABELS = { gpu: 'WebGPU', cpu: 'CPU fused kernels' };

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
    gpuOutputs.push(out && { sums: out.sums, values: out.values });
    gpuMs += out ? out.gpuMs : 0;
  }
  const result = await run('gpuAnalyse', { ...spec, analysis, gpuOutputs });
  const first = jobs.plans.find(Boolean);
  return { result: { ...result, gpuMs, adapter: gpu.name, plans: jobs.plans.length, compileMs: jobs.compileMs, numTerms: first?.numTerms, numDates: first?.numDates } };
}

/** The fused kernels on the CPU: one call into C++, which runs and analyses them. */
export async function runOnCpuKernels(analysis, spec) {
  const result = await run('cpuKernels', { ...spec, analysis });
  if (result.unsupported) return { unsupported: result.unsupported };
  globalThis.__ccrCpuKernelRuns = (globalThis.__ccrCpuKernelRuns || 0) + 1; // checked by the e2e test
  const n = result.threads;
  return { result: { ...result, gpuMs: result.kernelMs, adapter: `${n} thread${n === 1 ? '' : 's'}` } };
}

const outcome = (engine, result, reason) => ({ result, engine, fused: true, label: LABELS[engine], reason });

/**
 * Runs `analysis` (exposure, collateral, allocation, cva, xva, wrongWayRisk, hedging,
 * instruments, models, amc, validation) on the engine chosen by the selector.
 * Resolves to { result, engine, fused, label, reason }.
 */
export async function runAnalysis(analysis, spec) {
  const mode = getEngineMode();
  const cpu = async (why) => {
    const c = await runOnCpuKernels(analysis, spec);
    if (c.unsupported) throw new Error(c.unsupported);
    return outcome('cpu', c.result, why);
  };
  if (mode === 'cpu') return cpu('CPU fused kernels selected');
  if (!('gpu' in navigator)) return cpu('WebGPU is not available in this browser');
  try {
    const g = await runOnGpu(analysis, spec);
    if (g.unsupported) throw new Error(g.unsupported);
    return outcome('gpu', g.result, mode === 'auto' ? 'Auto: WebGPU available' : 'WebGPU selected');
  } catch (e) {
    console.warn('WebGPU run failed, falling back to the CPU kernels:', e);
    // No adapter is a property of the device, not a failure of the run: say so plainly.
    if (/no WebGPU adapter|too limited/.test(e.message)) return cpu(`WebGPU unavailable: ${e.message.replace('no WebGPU adapter: ', '')}`);
    return cpu(`WebGPU failed (${e.message})`);
  }
}

/** Toolbar control: Engine [Auto | WebGPU | CPU fused kernels]. Changing it re-runs via `onChange`. */
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
  return `Done in ${Math.round(ms).toLocaleString('en-US')} ms · ${r.label} (${r.result.adapter}) · ${r.reason}`;
}
