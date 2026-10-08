// WebGPU engine running the fused exposure kernels on a plan compiled by the WASM library.
import { FUSED_EXPOSURE, NF, PFE_QUANTILE, REDUCE_PARTIALS, WG } from './kernels.js';

export class GpuEngine {
  static async create() {
    if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
      },
    });
    const engine = new GpuEngine(adapter, device);
    await engine.compile();
    return engine;
  }

  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.info = adapter.info || {};
    device.lost.then((info) => { this.lost = info; });
  }

  async compile() {
    const make = async (code, label) => {
      const module = this.device.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
      return this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    };
    [this.fused, this.reduce, this.pfe] = await Promise.all([
      make(FUSED_EXPOSURE, 'fused-exposure'),
      make(REDUCE_PARTIALS, 'reduce-partials'),
      make(PFE_QUANTILE, 'pfe-quantile'),
    ]);
  }

  buffer(data, usage, label) {
    const size = Math.max(16, Math.ceil(data.byteLength / 4) * 4);
    const buf = this.device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST, label, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
  }

  /**
   * Runs the fused pipeline. plan: result of the WASM gpuPlan(spec).
   * Returns per-date statistics on the reporting dates plus CVA and timings.
   */
  async run(plan) {
    const d = this.device;
    const nP = plan.numPaths;
    const nT = plan.times.length;
    const numWG = Math.ceil(nP / WG);
    const cubeBytes = nP * nT * 4;
    const maxBinding = d.limits.maxStorageBufferBindingSize;
    if (cubeBytes > maxBinding)
      throw new Error(`paths × dates too large for one storage buffer (${(cubeBytes / 2 ** 20).toFixed(0)} MiB > ${(maxBinding / 2 ** 20).toFixed(0)} MiB)`);
    if (numWG > d.limits.maxComputeWorkgroupsPerDimension) throw new Error('too many paths for one dispatch');

    const header = new ArrayBuffer(64);
    const hu = new Uint32Array(header);
    const hf = new Float32Array(header);
    const offCall = nT + 1;
    const offReporting = offCall + nT;
    hu.set([nP, nT, plan.numAssets, plan.numCredits, plan.substeps, plan.numNormals, plan.seed >>> 0, plan.antithetic ? 1 : 0,
      plan.hasCsa ? 1 : 0, plan.counterparty, plan.stepStride, plan.numCorrelated, offCall, offReporting]);
    hf[14] = plan.pfeQuantile;
    hu[15] = numWG;

    const idx = new Uint32Array(offReporting + nT);
    idx.set(plan.termStart, 0);
    idx.set(plan.callIndex, offCall);
    idx.set(plan.isReporting, offReporting);

    const S = GPUBufferUsage.STORAGE;
    const bufs = {
      header: this.buffer(new Uint8Array(header), GPUBufferUsage.UNIFORM, 'header'),
      idx: this.buffer(idx, S, 'idx'),
      fp: this.buffer(Float32Array.from(plan.fparams), S, 'fparams'),
      steps: this.buffer(Float32Array.from(plan.steps), S, 'steps'),
      terms: this.buffer(Float32Array.from(plan.terms.length ? plan.terms : [0]), S, 'terms'),
      vals: d.createBuffer({ size: cubeBytes, usage: S, label: 'vals' }),
      expo: d.createBuffer({ size: cubeBytes, usage: S, label: 'expo' }),
      partials: d.createBuffer({ size: numWG * nT * NF * 4, usage: S, label: 'partials' }),
      sums: d.createBuffer({ size: nT * NF * 4, usage: S | GPUBufferUsage.COPY_SRC, label: 'sums' }),
      pfe: d.createBuffer({ size: nT * 4, usage: S | GPUBufferUsage.COPY_SRC, label: 'pfe' }),
      readSums: d.createBuffer({ size: nT * NF * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      readPfe: d.createBuffer({ size: nT * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
    const group = (pipeline, entries) => d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const fusedGroup = group(this.fused, [bufs.header, bufs.idx, bufs.fp, bufs.steps, bufs.terms, bufs.vals, bufs.expo, bufs.partials]);
    const reduceGroup = group(this.reduce, [bufs.header, bufs.partials, bufs.sums]);
    const pfeGroup = group(this.pfe, [bufs.header, bufs.expo, bufs.pfe]);

    d.pushErrorScope('validation');
    d.pushErrorScope('out-of-memory');
    const t0 = performance.now();
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.fused); pass.setBindGroup(0, fusedGroup); pass.dispatchWorkgroups(numWG);
    pass.setPipeline(this.reduce); pass.setBindGroup(0, reduceGroup); pass.dispatchWorkgroups(Math.ceil((nT * NF) / 64));
    pass.setPipeline(this.pfe); pass.setBindGroup(0, pfeGroup); pass.dispatchWorkgroups(nT);
    pass.end();
    enc.copyBufferToBuffer(bufs.sums, 0, bufs.readSums, 0, nT * NF * 4);
    enc.copyBufferToBuffer(bufs.pfe, 0, bufs.readPfe, 0, nT * 4);
    d.queue.submit([enc.finish()]);
    await Promise.all([bufs.readSums.mapAsync(GPUMapMode.READ), bufs.readPfe.mapAsync(GPUMapMode.READ)]);
    const gpuMs = performance.now() - t0;
    const oom = await d.popErrorScope();
    const validation = await d.popErrorScope();
    if (validation || oom) {
      for (const b of Object.values(bufs)) b.destroy();
      throw new Error((validation || oom).message);
    }
    const sums = new Float32Array(bufs.readSums.getMappedRange().slice(0));
    const pfe = new Float32Array(bufs.readPfe.getMappedRange().slice(0));
    for (const b of Object.values(bufs)) b.destroy();
    if (this.lost) throw new Error('GPU device lost: ' + this.lost.message);
    return summarise(plan, sums, pfe, gpuMs);
  }
}

/** Turns the raw per-date sums into an exposure profile on the reporting dates, plus CVA. */
export function summarise(plan, sums, pfe, gpuMs) {
  const nT = plan.times.length;
  const rep = [];
  for (let j = 0; j < nT; j++) if (plan.isReporting[j]) rep.push(j);
  const pick = (f) => rep.map((j) => sums[j * NF + f]);
  const times = rep.map((j) => plan.times[j]);
  const profile = {
    times,
    expectedValue: pick(0),
    ee: pick(1),
    ene: pick(2),
    discountedEe: pick(3),
    discountedEne: pick(4),
    pfe: rep.map((j) => pfe[j]),
    meanSurvival: pick(6),
  };
  let running = 0;
  profile.effectiveEe = profile.ee.map((v) => (running = Math.max(running, v)));
  profile.epe1y = timeAverage(times, profile.ee, 1);
  profile.eepe1y = timeAverage(times, profile.effectiveEe, 1);
  profile.epeLife = timeAverage(times, profile.ee, times[times.length - 1]);
  profile.maxPfe = Math.max(...profile.pfe);
  profile.pfeQuantile = plan.pfeQuantile;

  const surv = rep.map((j) => plan.marketSurvival[j]);
  profile.marketSurvival = surv;
  // Per-date pathwise CVA increments E[1/2 (D V+_{k-1} + D V+_k) (Q_{k-1} - Q_k)], before the LGD factor.
  profile.cvaIncrements = pick(5);
  let cva = 0;
  for (let k = 1; k < rep.length; k++)
    cva += (1 - plan.recovery) * 0.5 * (profile.discountedEe[k - 1] + profile.discountedEe[k]) * (surv[k - 1] - surv[k]);
  const pathwiseCva = plan.numCredits > 0 ? (1 - plan.recovery) * pick(5).reduce((a, b) => a + b, 0) : NaN;
  return { profile, cva, pathwiseCva, gpuMs };
}

function timeAverage(times, values, horizon) {
  let integral = 0, end = 0;
  for (let j = 1; j < times.length && times[j] <= horizon + 1e-10; j++) {
    integral += values[j] * (times[j] - times[j - 1]);
    end = times[j];
  }
  return end > 0 ? integral / end : values[0] || 0;
}
