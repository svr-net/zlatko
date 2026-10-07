// Promise-based client for the WASM worker: await ccr.call('exposure', spec).

let worker = null;
let nextId = 1;
const pending = new Map();

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./ccr-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    if (e.data.error) p.reject(new Error(e.data.error));
    else p.resolve({ result: e.data.result, ms: e.data.ms });
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || 'WASM worker failed to load'));
    pending.clear();
  };
  return worker;
}

/** Calls a library entry point in the worker. Resolves to { result, ms }. */
export function call(fn, spec) {
  const w = ensureWorker();
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage({ id, fn, spec: JSON.parse(JSON.stringify(spec)) });
  });
}

/** Convenience: resolves to the result only. */
export async function run(fn, spec) {
  return (await call(fn, spec)).result;
}
