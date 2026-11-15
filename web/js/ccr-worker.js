// Module worker hosting the WebAssembly build of the library, so long Monte Carlo
// runs never block the page. Messages: { id, fn, spec } -> { id, result, ms } | { id, error }.
import createCcrModule from '../wasm/ccr.js';

// The standalone build embeds the .wasm bytes and passes them here (no fetch, works from file://).
const ready = createCcrModule(self.__CCR_MODULE_OPTIONS__ || {});

self.onmessage = async (event) => {
  const { id, fn, spec } = event.data;
  try {
    const ccr = await ready;
    const t0 = performance.now();
    const result = ccr.call(fn, spec ?? {}); // every entry point is ccr::app::call(fn, spec) in C++
    const ms = performance.now() - t0;
    if (result && result.error) self.postMessage({ id, error: result.error });
    else self.postMessage({ id, result, ms });
  } catch (e) {
    self.postMessage({ id, error: String(e && e.message ? e.message : e) });
  }
};
