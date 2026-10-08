import { run } from '../ccr-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { runOnGpu } from '../gpu/backend.js';
import { card, el, grid, initPage, passFail, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'gpu',
  title: 'WebGPU fused exposure kernels',
  context: 'ccr/gpu: compile · summarise · runFusedReference · WGSL kernels',
  description:
    'The C++ library compiles the netting set into flat tables. These hold the exact Hull–White step coefficients, ' +
    'the bond terms <code>A·e<sup>−Bx</sup></code> for each date, fixing records, option terms and the margin-call look-back. A single WebGPU compute kernel, also part of the library, then ' +
    '<b>simulates, prices, nets, collateralises and aggregates</b> every path in one pass, without ever storing a scenario cube. ' +
    'The browser only uploads the tables, dispatches the kernels and hands the read-back to the library, which builds the profile, CVA and the checks below against its own CPU Monte Carlo run. ' +
    'The GPU works in 32-bit floats with a different random number generator, so the two agree within Monte Carlo error, not to the last digit. Bermudan swaptions need AMC regression, so they stay on the WASM side.',
});
specEditor(page, ['market', 'assets', 'credit', 'correlation', 'simulation', 'csa', 'portfolio']);

runButton(page, 'Run on GPU and validate against WASM', async (spec) => {
  const g = await runOnGpu('validation', spec, { warmUp: true });
  if (g.unsupported) throw new Error('not on GPU: ' + g.unsupported);
  const r = g.result;
  const gp = r.gpu.profile, wp = r.wasm.profile, c = r.checks;

  tiles(page.content, [
    { label: 'GPU fused pipeline', value: fmt.num(r.gpuMs, 1) + ' ms', hint: `${fmt.num(r.numPaths)} paths × ${r.numDates} dates` },
    { label: 'WASM (CPU, 1 thread)', value: fmt.num(r.wasm.elapsedMs, 0) + ' ms', hint: 'scenarios + pricing' },
    { label: 'Speed-up', value: (r.wasm.elapsedMs / r.gpuMs).toFixed(1) + '×', hint: 'includes upload & read-back' },
    { label: 'Plan compile (C++)', value: fmt.num(r.compileMs, 1) + ' ms', hint: `${fmt.num(r.numTerms)} valuation terms` },
    { label: 'CVA (GPU / WASM)', value: `${fmt.compact(r.gpu.cva)} / ${fmt.compact(r.wasm.cva)}`, hint: 'independence, market survival' },
    { label: 'Pathwise CVA (GPU / WASM)', value: `${fmt.compact(r.gpu.pathwiseCva)} / ${fmt.compact(r.wasm.pathwiseCva)}`, hint: 'stochastic CIR++ intensity' },
  ]);

  const gr = grid(page.content);
  const t = Array.from(gp.times);
  const q = fmt.pct(gp.pfeQuantile, 0);
  lineChart(card(gr, 'Expected exposure: GPU vs WASM', 'Solid lines are the fused GPU kernel and dashed lines the WASM library. They use independent random numbers.'), {
    x: t, xLabel: 't (years)', yLabel: 'exposure',
    series: [
      { name: 'EE GPU', y: Array.from(gp.ee), colorIndex: 0 },
      { name: 'EE WASM', y: Array.from(wp.ee), colorIndex: 0, dash: true },
      { name: `PFE${q} GPU`, y: Array.from(gp.pfe), colorIndex: 1 },
      { name: `PFE${q} WASM`, y: Array.from(wp.pfe), colorIndex: 1, dash: true },
    ],
  });
  lineChart(card(gr, 'Negative exposure & expected value', 'ENE drives DVA. E[V] is the mean mark-to-market net of collateral.'), {
    x: t, xLabel: 't (years)', zero: true,
    series: [
      { name: 'ENE GPU', y: Array.from(gp.ene), colorIndex: 2 },
      { name: 'ENE WASM', y: Array.from(wp.ene), colorIndex: 2, dash: true },
      { name: 'E[V] GPU', y: Array.from(gp.expectedValue), colorIndex: 6 },
      { name: 'E[V] WASM', y: Array.from(wp.expectedValue), colorIndex: 6, dash: true },
    ],
  });
  if (r.gpu.meanSurvival.length)
    lineChart(card(gr, 'CIR++ survival simulated on the GPU', 'The mean pathwise survival must reproduce the bootstrapped market curve.'), {
      x: t, xLabel: 't (years)', yFormat: (v) => v.toFixed(3),
      series: [
        { name: 'E[Q] GPU', y: Array.from(r.gpu.meanSurvival), colorIndex: 0 },
        { name: 'market Q', y: Array.from(r.marketSurvival), colorIndex: 1, dash: true },
      ],
    });

  const checks = card(gr, 'Agreement checks', 'Computed by the library. Tolerances allow for Monte Carlo noise at the chosen path count.');
  table(checks, ['check', 'value', 'tolerance', 'result'], [
    ['t = 0 value: |Δ| per unit gross notional (f32)', c.t0Diff.toExponential(1), '1e-6', passFail(c.t0Diff < 1e-6)],
    ['max |ΔEE| / max EE', fmt.pct(c.eeDiff, 2), fmt.pct(c.tol, 1), passFail(c.eeDiff < c.tol)],
    ['|ΔCVA| / CVA', fmt.pct(c.cvaDiff, 2), fmt.pct(c.tol, 1), passFail(c.cvaDiff < c.tol)],
    ['CIR++ fit: max |E[Q] − Q| / Q', fmt.pct(c.survDiff, 3), '1%', passFail(c.survDiff < 0.01)],
    ['EEPE (1y) GPU / WASM', `${fmt.compact(gp.eepe1y)} / ${fmt.compact(wp.eepe1y)}`, '', ''],
  ]);
  window.__ccrLastRun = { checks: c };
  return `GPU ${fmt.num(r.gpuMs, 1)} ms · WASM ${fmt.num(r.wasm.elapsedMs)} ms · adapter: ${r.adapter}`;
});

const bench = el('button', { text: 'Benchmark path scaling' });
page.toolbar.insertBefore(bench, page.status);
bench.addEventListener('click', async () => {
  bench.disabled = true;
  page.status.textContent = 'Benchmarking…';
  try {
    const counts = [1000, 4000, 16000, 64000];
    const rows = [];
    for (const n of counts) {
      const spec = { ...page.spec, sim: { ...page.spec.sim, numPaths: n } };
      const g = await runOnGpu('exposure', spec);
      if (g.unsupported) throw new Error('not on GPU: ' + g.unsupported);
      // The CPU cost is linear in the path count: measured up to 4,000 paths, extrapolated above.
      const wasmMs = n <= 4000 ? (await run('exposure', spec)).elapsedMs : rows[rows.length - 1].wasmMs * (n / rows[rows.length - 1].n);
      rows.push({ n, gpuMs: g.result.gpuMs, wasmMs });
    }
    const box = card(page.content, 'Throughput (paths per second)', 'WASM figures above 4,000 paths are extrapolated linearly, because CPU cost is linear in the path count.');
    barChart(box, {
      labels: rows.map((x) => fmt.num(x.n) + ' paths'),
      series: [
        { name: 'GPU fused kernel', values: rows.map((x) => (1000 * x.n) / x.gpuMs) },
        { name: 'WASM library', values: rows.map((x) => (1000 * x.n) / x.wasmMs) },
      ],
    });
    table(box, ['paths', 'GPU ms', 'WASM ms', 'speed-up'], rows.map((x) => [fmt.num(x.n), fmt.num(x.gpuMs, 1), fmt.num(x.wasmMs) + (x.n > 4000 ? ' (extrap.)' : ''), (x.wasmMs / x.gpuMs).toFixed(1) + '×']));
    page.status.textContent = 'Benchmark done';
  } catch (e) {
    page.status.className = 'status error';
    page.status.textContent = 'Error: ' + e.message;
  } finally {
    bench.disabled = false;
  }
});

run('gpuKernels', {}).then((k) => page.main.append(el('details', { class: 'spec' }, el('summary', { text: 'WGSL source of the kernels (from the C++ library)' }),
  el('h3', { text: 'fused-exposure' }), el('pre', { class: 'code', text: k.fusedExposure }),
  el('h3', { text: 'reduce-partials' }), el('pre', { class: 'code', text: k.reducePartials }),
  el('h3', { text: 'pfe-quantile' }), el('pre', { class: 'code', text: k.pfeQuantile }))));
