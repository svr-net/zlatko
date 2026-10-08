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
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright').catch(() =>
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
const cases = [
  ...['index', 'core', 'market', 'models', 'instruments', 'amc', 'exposure', 'collateral', 'allocation', 'cva', 'wwr', 'hedging']
    .map((name) => ({ name, file: name, spec: testSpec })),
  { name: 'gpu', file: 'gpu', spec: gpuSpec },
  // Collateralised netting set: exercises the margin-call look-back inside the fused kernel.
  { name: 'gpu-csa', file: 'gpu', spec: { ...gpuSpec, csa: { enabled: true, thresholdCounterparty: 250e3, thresholdOwn: 250e3, mta: 50e3, independentAmount: 0, mpr: 10 / 250 } } },
];
let failures = 0;
console.log(`testing ${root} ${fileMode ? 'from file:// (no server)' : `over ${base}`}`);

for (const { name, file, spec } of cases.filter((c) => !filter || c.name.includes(filter))) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const network = [];
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) network.push(r.url()); });
  await page.addInitScript((spec) => {
    if (!sessionStorage.getItem('seeded')) {
      localStorage.setItem('ccr-spec-v1', JSON.stringify(spec));
      sessionStorage.setItem('seeded', '1');
    }
  }, spec);
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
  console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name.padEnd(12)} ${String(Date.now() - t0).padStart(6)} ms  charts=${charts}  ${status.trim().slice(0, 90)}`);
  for (const p of problems) console.log('       ' + p);
  failures += problems.length ? 1 : 0;
  await page.close();
}

await browser.close();
server.close();
console.log(failures ? `\n${failures} page(s) failed` : '\nall pages passed');
process.exit(failures ? 1 : 0);
