// End-to-end test of the web front end in headless Chromium (Playwright).
//
//   node web/tests/e2e.mjs [page-filter] [--root DIR] [--file]
//
// Serves web/ (or --root DIR, e.g. dist/standalone) on a local port, or with --file opens the
// pages straight from disk as file:// URLs (the copy-and-open deployment of the standalone
// build, which must then make no network request at all). Opens every page, waits for its automatic run and
// checks that it finished without errors and rendered charts. On the WebGPU page
// it also checks that the fused GPU kernel agrees with the WASM library.
// WebGPU runs on SwiftShader in headless mode, so the GPU page works without a GPU.
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const argv = process.argv.slice(2);
const option = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
const fileMode = argv.includes('--file');
const root = path.resolve(option('--root') || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
const filter = argv.find((a, i) => !a.startsWith('--') && argv[i - 1] !== '--root');
const { chromium, devices } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright').catch(() =>
  import('/opt/node22/lib/node_modules/playwright/index.mjs'));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.png': 'image/png' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.join(root, url === '/' ? 'index.html' : url);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || (fs.existsSync('/opt/pw-browsers/chromium-1194/chrome-linux/chrome') ? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' : undefined),
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'],
});

// Small path counts keep the run fast; the pages pick the spec up from localStorage.
const testSpec = { sim: { numPaths: 1000, seed: 11, antithetic: true } };
const gpuSpec = { sim: { numPaths: 8000, seed: 11, antithetic: true } };
const pairSpec = { sim: { numPaths: 4000, seed: 11, antithetic: true } };
// Pages with the engine selector. With no stored choice they run on Auto, which must pick the
// WebGPU kernels on desktop too; each also runs once forced onto WebAssembly.
const enginePages = ['index', 'exposure', 'collateral', 'cva', 'wwr', 'hedging'];
const cases = [
  ...['index', 'core', 'market', 'models', 'instruments', 'amc', 'exposure', 'collateral', 'allocation', 'cva', 'wwr', 'hedging']
    .map((name) => ({ name, file: name, spec: testSpec, ...(enginePages.includes(name) ? { engine: 'auto', expectEngine: 'WebGPU' } : {}) })),
  ...['index', 'collateral', 'wwr'].map((name) => ({ name: `${name}-wasm`, file: name, spec: testSpec, engine: 'wasm', expectEngine: 'WebAssembly' })),
  // A first visit (nothing stored): the default is Auto, which must dispatch the fused kernels.
  { name: 'cva-first-visit', file: 'cva', spec: testSpec, expectEngine: 'WebGPU' },
  // A Bermudan needs American Monte Carlo: Auto must fall back to WebAssembly and say why.
  { name: 'exposure-amc', file: 'exposure', spec: { ...testSpec, trades: [{ type: 'bermudan', id: 'Berm 1y×4y', notional: 1e6, fixedRate: 0.035, start: 1, tenor: 4, freq: 1, direction: 'receiver', degree: 2 }] }, engine: 'auto', expectEngine: 'WebAssembly' },
  { name: 'gpu', file: 'gpu', spec: gpuSpec },
  // Collateralised netting set: exercises the margin-call look-back inside the fused kernel.
  { name: 'gpu-csa', file: 'gpu', spec: { ...gpuSpec, csa: { enabled: true, thresholdCounterparty: 250e3, thresholdOwn: 250e3, mta: 50e3, independentAmount: 0, mpr: 10 / 250 } } },
  // WASM/GPU pairs on the same spec whose headline numbers must agree within Monte Carlo error.
  ...['exposure', 'cva', 'hedging'].flatMap((name) => [
    { name: `${name}-wasm-pair`, file: name, spec: pairSpec, engine: 'wasm', expectEngine: 'WebAssembly', pair: name },
    { name: `${name}-gpu-pair`, file: name, spec: pairSpec, engine: 'gpu', expectEngine: 'WebGPU', pair: name },
    { name: `${name}-cpu-pair`, file: name, spec: pairSpec, engine: 'cpu', expectEngine: 'CPU fused kernels', pair: name },
  ]),
  // Android drivers: no adapter for a high-performance request but one for a plain request
  // (Auto must still reach WebGPU), and no adapter at all (WebAssembly, with the reason).
  { name: 'exposure-android-quirk', file: 'exposure', spec: testSpec, engine: 'auto', expectEngine: 'WebGPU', gpuStub: 'no-high-performance', device: 'Pixel 7' },
  // Chrome offers only a WebGPU compatibility-mode adapter (OpenGL ES): Auto must use it.
  { name: 'exposure-compat-only', file: 'exposure', spec: testSpec, engine: 'auto', expectEngine: 'WebGPU', gpuStub: 'compat-only', device: 'Pixel 7', expectReason: 'compatibility mode' },
  // No adapter at all (WebGPU blocked by Chrome): Auto runs the same fused kernels on the CPU.
  { name: 'exposure-no-adapter', file: 'exposure', spec: testSpec, engine: 'auto', expectEngine: 'CPU fused kernels', gpuStub: 'no-adapter', expectReason: 'WebGPU unavailable', device: 'Pixel 7' },
  { name: 'gpu-no-adapter', file: 'gpu', spec: gpuSpec, gpuStub: 'no-adapter', expectEngine: 'CPU fused kernels' },
  { name: 'hedging-no-adapter', file: 'hedging', spec: testSpec, engine: 'auto', expectEngine: 'CPU fused kernels', gpuStub: 'no-adapter' },
  // A phone: Auto must pick WebGPU, and the collapsed menu must leave the page content in view.
  { name: 'exposure-mobile', file: 'exposure', spec: testSpec, engine: 'auto', expectEngine: 'WebGPU', device: 'iPhone 14' },
];
const pairs = {};
let failures = 0;
console.log(`testing ${root} ${fileMode ? 'from file:// (no server)' : `over ${base}`}`);

for (const { name, file, spec, engine, expectEngine, pair, device, gpuStub, expectReason } of cases.filter((c) => !filter || c.name.includes(filter))) {
  const context = await browser.newContext(device ? { ...devices[device] } : { viewport: { width: 1400, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const network = [];
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) network.push(r.url()); });
  if (gpuStub) await page.addInitScript((stub) => {
    const gpu = navigator.gpu;
    if (!gpu) return;
    const original = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = (options = {}) =>
      stub === 'no-adapter' || (stub === 'compat-only' ? !('featureLevel' in options) : options.powerPreference === 'high-performance')
        ? Promise.resolve(null) : original(options);
  }, gpuStub);
  await page.addInitScript(({ spec, engine }) => {
    if (!sessionStorage.getItem('seeded')) {
      localStorage.setItem('ccr-spec-v1', JSON.stringify(spec));
      if (engine) localStorage.setItem('ccr-engine', engine); else localStorage.removeItem('ccr-engine');
      sessionStorage.setItem('seeded', '1');
    }
  }, { spec, engine });
  const t0 = Date.now();
  await page.goto(fileMode ? pathToFileURL(path.join(root, `${file}.html`)).href : `${base}/${file}.html`);
  let status = '';
  try {
    await page.waitForFunction(() => {
      const s = document.querySelector('#run-status');
      return s && /Done|GPU .* ms|Error|ready/.test(s.textContent);
    }, null, { timeout: 240000 });
    status = await page.textContent('#run-status');
  } catch (e) {
    status = 'timeout';
  }
  const charts = await page.$$eval('.chart canvas', (cs) => cs.filter((c) => c.width > 0).length);
  const problems = [...errors];
  if (/Error|timeout/.test(status)) problems.push('status: ' + status);
  if (charts === 0) problems.push('no charts rendered');
  if (fileMode && network.length) problems.push(`network requests from a file:// page: ${network.slice(0, 3).join(', ')}`);
  if (expectEngine && !status.includes(`· ${expectEngine}`)) problems.push(`expected the ${expectEngine} engine: ${status}`);
  if (expectReason && !status.includes(expectReason)) problems.push(`expected the reason "${expectReason}": ${status}`);
  // The status line is not enough: the fused kernels must really have been dispatched on WebGPU
  // (and never when WebAssembly ran).
  const gpuRuns = await page.evaluate(() => globalThis.__ccrGpuRuns || 0);
  const cpuRuns = await page.evaluate(() => globalThis.__ccrCpuKernelRuns || 0);
  if (expectEngine === 'WebGPU' && gpuRuns === 0) problems.push('no fused pipeline was dispatched on WebGPU');
  if (expectEngine === 'CPU fused kernels' && cpuRuns === 0) problems.push('the CPU fused kernels did not run');
  if (expectEngine && expectEngine !== 'WebGPU' && gpuRuns > 0) problems.push(`${gpuRuns} WebGPU dispatches while the ${expectEngine} engine ran`);
  if (expectEngine && expectEngine !== 'CPU fused kernels' && cpuRuns > 0) problems.push(`CPU kernels ran while the ${expectEngine} engine ran`);
  if (pair) pairs[pair] = { ...(pairs[pair] || {}), [engine]: await page.evaluate(() => window.__ccrEngineRun) };
  if (device) {
    const layout = await page.evaluate(() => ({
      menuHidden: document.getElementById('nav-links').offsetParent === null,
      titleTop: document.querySelector('main h1').getBoundingClientRect().top,
      scrollW: document.documentElement.scrollWidth, width: document.documentElement.clientWidth,
    }));
    if (!layout.menuHidden) problems.push('mobile menu is not collapsed');
    if (layout.titleTop > 200) problems.push(`page title starts ${Math.round(layout.titleTop)}px down: content hidden below the menu`);
    if (layout.scrollW > layout.width + 1) problems.push(`horizontal scroll on mobile (${layout.scrollW} > ${layout.width})`);
    await page.click('.menu-toggle');
    if (await page.evaluate(() => document.getElementById('nav-links').offsetParent === null)) problems.push('menu button does not open the menu');
  }
  if (file === 'gpu' && !gpuStub) {
    const probe = await page.waitForFunction(() => window.__ccrWebGpuProbe, null, { timeout: 30000 }).then((h) => h.jsonValue()).catch(() => null);
    if (!probe || !probe.adapters.some((a) => a.adapter)) problems.push(`WebGPU diagnostics found no adapter: ${JSON.stringify(probe)}`);
  }
  if (file === 'gpu' && !problems.length) {
    const r = await page.evaluate(() => window.__ccrLastRun);
    if (!r) problems.push('no GPU result');
    else {
      const c = r.checks;
      if (!(c.t0Diff < 1e-6)) problems.push(`t=0 value mismatch ${c.t0Diff}`);
      if (!(c.eeDiff < c.tol)) problems.push(`EE mismatch ${c.eeDiff} > ${c.tol}`);
      if (!(c.cvaDiff < c.tol)) problems.push(`CVA mismatch ${c.cvaDiff} > ${c.tol}`);
      if (!(c.survDiff < 0.01)) problems.push(`CIR++ survival mismatch ${c.survDiff}`);
      console.log(`       GPU checks: EE ${(100 * c.eeDiff).toFixed(2)}%, CVA ${(100 * c.cvaDiff).toFixed(2)}% (tol ${(100 * c.tol).toFixed(1)}%), Q ${(100 * c.survDiff).toFixed(3)}%, t0 ${c.t0Diff.toExponential(1)}`);
    }
  }
  if (process.env.SCREENSHOTS) await page.screenshot({ path: path.join(process.env.SCREENSHOTS, `${name}.png`), fullPage: true });
  console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name.padEnd(12)} ${String(Date.now() - t0).padStart(6)} ms  charts=${charts}  gpu=${gpuRuns}  ${status.trim().slice(0, 80)}`);
  for (const p of problems) console.log('       ' + p);
  failures += problems.length ? 1 : 0;
  await context.close();
}

// The same spec on both engines: GPU (f32, its own random numbers) and WASM must agree within
// Monte Carlo error at 4,000 paths (hedging caps its runs at 1,500 paths).
const metrics = {
  exposure: [['EEPE 1y', (r) => r.eepe1y]],
  cva: [['CVA', (r) => r.cva]],
  hedging: [['CVA', (r) => r.cva], ['parallel CS01', (r) => r.parallelCs01], ['CVA DV01', (r) => r.cvaDv01],
    ['delta (CRN, h=0.005)', (r) => r.deltaCrn[0][3]]],
};
for (const [name, r] of Object.entries(pairs)) {
  const tol = 4 / Math.sqrt(name === 'hedging' ? 1500 : pairSpec.sim.numPaths);
  for (const fused of ['gpu', 'cpu']) {
    if (!r.wasm || !r[fused]) continue;
    for (const [label, get] of metrics[name]) {
      const w = get(r.wasm), g = get(r[fused]);
      const diff = Math.abs(g - w) / Math.abs(w);
      const ok = diff < tol;
      console.log(`${ok ? 'ok  ' : 'FAIL'} ${`${name} WASM/${fused}`.padEnd(16)} ${label}: WASM ${w.toFixed(1)} vs ${fused.toUpperCase()} ${g.toFixed(1)} (${(100 * diff).toFixed(2)}%, tol ${(100 * tol).toFixed(1)}%)`);
      if (!ok) failures++;
    }
  }
  // The CPU kernels run the GPU's algorithm and random numbers in f32: they agree with the GPU
  // far more closely than either agrees with WebAssembly.
  if (r.gpu && r.cpu) {
    const [label, get] = metrics[name][0];
    const diff = Math.abs(get(r.cpu) - get(r.gpu)) / Math.abs(get(r.gpu));
    const ok = diff < 1e-3;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${`${name} GPU/CPU`.padEnd(16)} ${label}: GPU ${get(r.gpu).toFixed(2)} vs CPU kernels ${get(r.cpu).toFixed(2)} (${(100 * diff).toFixed(4)}%, tol 0.1%)`);
    if (!ok) failures++;
  }
}

await browser.close();
server.close();
console.log(failures ? `\n${failures} page(s) failed` : '\nall pages passed');
process.exit(failures ? 1 : 0);
