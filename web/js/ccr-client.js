// Promise-based client for the WASM worker: await ccr.call('exposure', spec).

// Worker 0 serves every call; the fused-kernel CPU backend also spreads its workgroups over
// extra workers (one WebAssembly instance each), created on first use.
const workers = [];
let nextId = 1;
const pending = new Map();

function createWorker() {
  // Standalone build: the whole worker (glue + embedded .wasm) ships as a string in a classic
  // script, so it runs from a blob URL without any fetch, from file:// or any static host.
  if (globalThis.__CCR_WORKER_SOURCE__) {
    const blob = new Blob([globalThis.__CCR_WORKER_SOURCE__], { type: 'text/javascript' });
    return new Worker(URL.createObjectURL(blob));
  }
  return new Worker(new URL('./ccr-worker.js', import.meta.url), { type: 'module' });
}

function startWorker() {
  const w = createWorker();
  w.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    if (e.data.error) p.reject(new Error(e.data.error));
    else p.resolve({ result: e.data.result, ms: e.data.ms });
  };
  w.onerror = (e) => {
    for (const [id, p] of pending) if (p.worker === w) { p.reject(new Error(e.message || 'WASM worker failed to load')); pending.delete(id); }
  };
  return w;
}

function workerAt(i) {
  while (workers.length <= i) workers.push(startWorker());
  return workers[i];
}

/** Number of workers the CPU kernel backend uses: the device's cores, at most 8. */
export function workerCount() {
  return Math.max(1, Math.min(8, navigator.hardwareConcurrency || 4));
}

/** Calls a library entry point in worker `i`. Resolves to { result, ms }. */
export function callOn(i, fn, spec) {
  const w = workerAt(i);
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, worker: w });
    w.postMessage({ id, fn, spec }); // structured clone: keeps the typed arrays of GPU read-backs
  });
}

/** Calls a library entry point in the main worker. Resolves to { result, ms }. */
export function call(fn, spec) {
  return callOn(0, fn, spec);
}

/** Convenience: resolves to the result only. */
export async function run(fn, spec) {
  return (await call(fn, spec)).result;
}
