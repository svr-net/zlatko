import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { GpuEngine } from '../gpu/engine.js';
import { FUSED_EXPOSURE, PFE_QUANTILE, REDUCE_PARTIALS } from '../gpu/kernels.js';
import { card, el, grid, initPage, passFail, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'gpu',
  title: 'WebGPU fused exposure kernels',
  context: 'web/js/gpu · wasm gpuPlan()',
  description:
    'The WASM build of the library compiles the netting set into flat tables. These hold the exact Hull–White step coefficients, ' +
    'the bond terms <code>A·e<sup>−Bx</sup></code> for each date, fixing records, option terms and the margin-call look-back. A single WebGPU compute kernel then ' +
    '<b>simulates, prices, nets, collateralises and aggregates</b> every path in one pass, without ever storing a scenario cube. ' +
    'The results are checked against the WASM Monte Carlo run on the same specification. The GPU works in 32-bit floats with a different random number generator, ' +
    'so the two agree within Monte Carlo error, not to the last digit. Bermudan swaptions need AMC regression, so they stay on the WASM side.',
});
specEditor(page, ['market', 'assets', 'credit', 'correlation', 'simulation', 'csa', 'portfolio']);

let engine = null;
async function getEngine() {
  if (!engine) engine = await GpuEngine.create();
  return engine;
}

function maxRelDiff(a, b) {
  const scale = Math.max(...b.map(Math.abs), 1e-12);
  return Math.max(...a.map((v, i) => Math.abs(v - b[i]))) / scale;
}

runButton(page, 'Run on GPU and validate against WASM', async (spec) => {
  const { result: plan, ms: planMs } = await (async () => {
    const t0 = performance.now();
    const r = await run('gpuPlan', spec);
    return { result: r, ms: performance.now() - t0 };
  })();
  const gpu = await getEngine();
  await gpu.run({ ...plan, numPaths: Math.min(plan.numPaths, 256) }); // warm-up (pipeline & driver caches)
  const g = await gpu.run(plan);
  const t0 = performance.now();
  const w = await run('cva', spec);
  const wasmMs = performance.now() - t0;
  const wp = w.profile;

  const c = page.content;
  if (plan.unsupported.length) c.append(el('p', { class: 'status error', text: 'Not on GPU: ' + plan.unsupported.join(', ') }));
  tiles(c, [
    { label: 'GPU fused pipeline', value: fmt.num(g.gpuMs, 1) + ' ms', hint: `${fmt.num(plan.numPaths)} paths × ${plan.times.length} dates` },
    { label: 'WASM (CPU, 1 thread)', value: fmt.num(wasmMs, 0) + ' ms', hint: 'scenarios + pricing + CVA' },
    { label: 'Speed-up', value: (wasmMs / g.gpuMs).toFixed(1) + '×', hint: 'includes upload & read-back' },
    { label: 'Plan compile (WASM)', value: fmt.num(planMs, 1) + ' ms', hint: `${fmt.num(plan.numTerms)} valuation terms` },
    { label: 'CVA (GPU / WASM)', value: `${fmt.compact(g.cva)} / ${fmt.compact(w.cva)}`, hint: 'independence, market survival' },
    { label: 'Pathwise CVA (GPU / WASM)', value: `${fmt.compact(g.pathwiseCva)} / ${fmt.compact(w.pathwiseCva)}`, hint: 'stochastic CIR++ intensity' },
  ]);

  const gr = grid(c);
  lineChart(card(gr, 'Expected exposure: GPU vs WASM', 'Solid lines are the fused GPU kernel and dashed lines the WASM library. They use independent random numbers.'), {
    x: g.profile.times, xLabel: 't (years)', yLabel: 'exposure',
    series: [
      { name: 'EE GPU', y: g.profile.ee, colorIndex: 0 },
      { name: 'EE WASM', y: Array.from(wp.ee), colorIndex: 0, dash: true },
      { name: `PFE${Math.round(plan.pfeQuantile * 100)} GPU`, y: g.profile.pfe, colorIndex: 1 },
      { name: `PFE${Math.round(plan.pfeQuantile * 100)} WASM`, y: Array.from(wp.pfe), colorIndex: 1, dash: true },
    ],
  });
  lineChart(card(gr, 'Negative exposure & expected value', 'ENE drives DVA. E[V] is the mean mark-to-market.'), {
    x: g.profile.times, xLabel: 't (years)', zero: true,
    series: [
      { name: 'ENE GPU', y: g.profile.ene, colorIndex: 2 },
      { name: 'ENE WASM', y: Array.from(wp.ene), colorIndex: 2, dash: true },
      { name: 'E[V] GPU', y: g.profile.expectedValue, colorIndex: 6 },
      { name: 'E[V] WASM', y: Array.from(wp.expectedValue), colorIndex: 6, dash: true },
    ],
  });
  if (plan.numCredits > 0)
    lineChart(card(gr, 'CIR++ survival simulated on the GPU', 'The mean pathwise survival must reproduce the bootstrapped market curve.'), {
      x: g.profile.times, xLabel: 't (years)', yFormat: (v) => v.toFixed(3),
      series: [
        { name: 'E[Q] GPU', y: g.profile.meanSurvival, colorIndex: 0 },
        { name: 'market Q', y: plan.marketSurvival.filter((_, j) => plan.isReporting[j]), colorIndex: 1, dash: true },
      ],
    });

  const checks = card(gr, 'Agreement checks', 'Tolerances allow for Monte Carlo noise at the chosen path count.');
  const tol = 4 / Math.sqrt(plan.numPaths);
  const eeDiff = maxRelDiff(g.profile.ee, Array.from(wp.ee));
  const cvaDiff = Math.abs(g.cva - w.cva) / Math.max(Math.abs(w.cva), 1e-12);
  const survDiff = plan.numCredits ? maxRelDiff(g.profile.meanSurvival, plan.marketSurvival.filter((_, j) => plan.isReporting[j])) : 0;
  // f32 term cancellation: measure the t = 0 difference per unit of gross notional.
  const grossNotional = spec.trades.reduce((s, t) => s + Math.abs(t.notional || 0), 0) || 1;
  const t0Diff = Math.abs(g.profile.expectedValue[0] - wp.expectedValue[0]) / grossNotional;
  table(checks, ['check', 'value', 'tolerance', 'result'], [
    ['t = 0 value: |Δ| per unit gross notional (f32)', t0Diff.toExponential(1), '1e-6', passFail(t0Diff < 1e-6)],
    ['max |ΔEE| / max EE', fmt.pct(eeDiff, 2), fmt.pct(tol, 1), passFail(eeDiff < tol)],
    ['|ΔCVA| / CVA', fmt.pct(cvaDiff, 2), fmt.pct(tol, 1), passFail(cvaDiff < tol)],
    ['CIR++ fit: max |E[Q] − Q| / Q', fmt.pct(survDiff, 3), '1%', passFail(survDiff < 0.01)],
    ['EEPE (1y) GPU / WASM', `${fmt.compact(g.profile.eepe1y)} / ${fmt.compact(wp.eepe1y)}`, '', ''],
  ]);
  window.__ccrLastRun = { gpu: g, wasmCva: w.cva, wasmPathwiseCva: w.pathwiseCva, checks: { eeDiff, cvaDiff, survDiff, t0Diff, tol } };
  return `GPU ${fmt.num(g.gpuMs, 1)} ms · WASM ${fmt.num(wasmMs)} ms · adapter: ${[gpu.info.vendor, gpu.info.architecture].filter(Boolean).join(' ') || 'unknown'}`;
});

const bench = el('button', { text: 'Benchmark path scaling' });
page.toolbar.insertBefore(bench, page.status);
bench.addEventListener('click', async () => {
  bench.disabled = true;
  page.status.textContent = 'Benchmarking…';
  try {
    const gpu = await getEngine();
    const counts = [1000, 4000, 16000, 64000];
    const gpuMs = [], wasmMs = [];
    for (const n of counts) {
      const spec = { ...page.spec, sim: { ...page.spec.sim, numPaths: n } };
      const plan = await run('gpuPlan', spec);
      gpuMs.push((await gpu.run(plan)).gpuMs);
      if (n <= 4000) {
        const t0 = performance.now();
        await run('exposure', spec);
        wasmMs.push(performance.now() - t0);
      } else wasmMs.push(wasmMs[wasmMs.length - 1] * (n / counts[wasmMs.length - 1])); // linear extrapolation
    }
    const box = card(page.content, 'Throughput (paths per second)', 'WASM figures above 4,000 paths are extrapolated linearly, because CPU cost is linear in the path count.');
    barChart(box, {
      labels: counts.map((n) => fmt.num(n) + ' paths'),
      series: [
        { name: 'GPU fused kernel', values: counts.map((n, i) => n / (gpuMs[i] / 1000)) },
        { name: 'WASM library', values: counts.map((n, i) => n / (wasmMs[i] / 1000)) },
      ],
    });
    table(box, ['paths', 'GPU ms', 'WASM ms', 'speed-up'], counts.map((n, i) => [fmt.num(n), fmt.num(gpuMs[i], 1), fmt.num(wasmMs[i]) + (n > 4000 ? ' (extrap.)' : ''), (wasmMs[i] / gpuMs[i]).toFixed(1) + '×']));
    page.status.textContent = 'Benchmark done';
  } catch (e) {
    page.status.className = 'status error';
    page.status.textContent = 'Error: ' + e.message;
  } finally {
    bench.disabled = false;
  }
});

const src = el('details', { class: 'spec' }, el('summary', { text: 'WGSL source of the kernels' }),
  el('h3', { text: 'fused-exposure' }), el('pre', { class: 'code', text: FUSED_EXPOSURE }),
  el('h3', { text: 'reduce-partials' }), el('pre', { class: 'code', text: REDUCE_PARTIALS }),
  el('h3', { text: 'pfe-quantile' }), el('pre', { class: 'code', text: PFE_QUANTILE }));
page.main.append(src);
