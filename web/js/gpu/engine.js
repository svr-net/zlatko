// WebGPU host for the fused exposure kernels of the C++ library (ccr::gpu).
//
// No numerics live here. The library supplies the WGSL sources (gpuKernels) and, for every
// job, a plan with ready-made buffers, dispatch sizes and buffer sizes (gpuJobs). This class
// uploads the buffers, dispatches fused-exposure -> reduce-partials -> pfe-quantile and reads
// back two arrays, which go back to the library (gpuAnalyse) for all post-processing.

// Adapter requests, in order. Some mobile drivers (seen on Android) return no adapter for a
// high-performance request but do for a plain one, so a null answer is not final. The last
// request asks for a WebGPU *compatibility* adapter: where Chrome cannot offer core WebGPU
// (seen on Android with an OpenGL compositor and a blocklisted Vulkan/GL interop) it may still
// offer WebGPU on OpenGL ES in compatibility mode, which is enough for compute kernels.
const ADAPTER_OPTIONS = [
  { powerPreference: 'high-performance' },
  {},
  { powerPreference: 'low-power' },
  { featureLevel: 'compatibility', compatibilityMode: true },
];

// What the fused kernels need beyond the compatibility-mode defaults: 7 storage buffers in the
// fused kernel, 256-invocation workgroups in pfe-quantile and its 13 KiB histogram.
const KERNEL_LIMITS = {
  maxStorageBuffersPerShaderStage: 7,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupStorageSize: 13316,
};

/** The kernel limits this adapter cannot meet, e.g. ["maxStorageBuffersPerShaderStage 4 < 7"]. */
export function missingLimits(adapter) {
  return Object.entries(KERNEL_LIMITS)
    .filter(([k, need]) => !(adapter.limits[k] >= need))
    .map(([k, need]) => `${k} ${adapter.limits[k]} < ${need}`);
}

/** Why WebGPU cannot start here, with what the user can check. */
export function noAdapterReason() {
  const ua = navigator.userAgent || '';
  const android = /Android/.test(ua);
  return android
    ? 'no WebGPU adapter: Chrome has WebGPU blocked or unsupported on this device (it needs Android 12+ and a supported GPU; see chrome://gpu)'
    : 'no WebGPU adapter: WebGPU is blocked or unsupported for this GPU or driver (see chrome://gpu)';
}

async function requestAdapter() {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser');
  const tooLimited = [];
  for (const options of ADAPTER_OPTIONS) {
    const adapter = await navigator.gpu.requestAdapter(options).catch(() => null);
    if (!adapter) continue;
    const missing = missingLimits(adapter);
    if (!missing.length) return { adapter, compatibility: 'featureLevel' in options };
    tooLimited.push(missing.join(', '));
  }
  if (tooLimited.length) throw new Error(`WebGPU adapter too limited for the kernels (${tooLimited[0]})`);
  throw new Error(noAdapterReason());
}

// Ask for the adapter's own limits where the kernels need more than the defaults (required in
// compatibility mode) and for its largest buffers; without the large buffers if refused.
async function requestDevice(adapter) {
  const lim = adapter.limits;
  const needed = Object.fromEntries(Object.keys(KERNEL_LIMITS).map((k) => [k, lim[k]]));
  try {
    return await adapter.requestDevice({
      requiredLimits: { ...needed, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
    });
  } catch (_) {
    return adapter.requestDevice({ requiredLimits: needed });
  }
}

/** Every adapter request with its answer, for the diagnostics panel. */
export async function probeAdapters() {
  const report = { secureContext: globalThis.isSecureContext, api: 'gpu' in navigator, userAgent: navigator.userAgent, adapters: [] };
  if (!report.api) return report;
  for (const options of ADAPTER_OPTIONS) {
    const entry = { request: JSON.stringify(options) };
    try {
      const a = await navigator.gpu.requestAdapter(options);
      if (a) {
        const info = a.info || {};
        entry.adapter = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'adapter (no info)';
        entry.maxStorageBufferBindingSize = a.limits.maxStorageBufferBindingSize;
        entry.maxComputeWorkgroupStorageSize = a.limits.maxComputeWorkgroupStorageSize;
        entry.missing = missingLimits(a);
      } else entry.adapter = null;
    } catch (e) {
      entry.error = e.message;
    }
    report.adapters.push(entry);
  }
  return report;
}

export class GpuEngine {
  static async create(kernels) {
    const { adapter, compatibility } = await requestAdapter();
    const device = await requestDevice(adapter);
    const engine = new GpuEngine(adapter, device, compatibility);
    await engine.compile(kernels);
    return engine;
  }

  constructor(adapter, device, compatibility = false) {
    this.adapter = adapter;
    this.device = device;
    this.info = adapter.info || {};
    this.name = ([this.info.vendor, this.info.architecture].filter(Boolean).join(' ') || 'GPU') + (compatibility ? ', compatibility mode' : '');
    device.lost.then((info) => { this.lost = info; });
  }

  async compile(kernels) {
    const make = async (code, label) => {
      const module = this.device.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
      return this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    };
    [this.fused, this.reduce, this.pfe] = await Promise.all([
      make(kernels.fusedExposure, 'fused-exposure'),
      make(kernels.reducePartials, 'reduce-partials'),
      make(kernels.pfeQuantile, 'pfe-quantile'),
    ]);
  }

  upload(data, usage, label) {
    const buf = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, label, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
  }

  /** Runs one plan from gpuJobs. Resolves to { sums, pfe, gpuMs }. */
  async run(plan) {
    const d = this.device;
    const { bytes, dispatch } = plan;
    if (bytes.cube > d.limits.maxStorageBufferBindingSize)
      throw new Error(`paths × dates too large for one storage buffer (${(bytes.cube / 2 ** 20).toFixed(0)} MiB > ${(d.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB)`);
    if (dispatch.fused > d.limits.maxComputeWorkgroupsPerDimension) throw new Error('too many paths for one dispatch');

    const S = GPUBufferUsage.STORAGE;
    const scratch = (size, usage, label) => d.createBuffer({ size, usage, label });
    const bufs = {
      header: this.upload(plan.header, GPUBufferUsage.UNIFORM, 'header'),
      indices: this.upload(plan.indices, S, 'indices'),
      params: this.upload(plan.params, S, 'params'),
      steps: this.upload(plan.steps, S, 'steps'),
      terms: this.upload(plan.terms, S, 'terms'),
      vals: scratch(bytes.cube, S, 'vals'),
      expo: scratch(bytes.cube, S, 'expo'),
      partials: scratch(bytes.partials, S, 'partials'),
      sums: scratch(bytes.sums, S | GPUBufferUsage.COPY_SRC, 'sums'),
      pfe: scratch(bytes.pfe, S | GPUBufferUsage.COPY_SRC, 'pfe'),
      readSums: scratch(bytes.sums, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-sums'),
      readPfe: scratch(bytes.pfe, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-pfe'),
    };
    const group = (pipeline, entries) => d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    d.pushErrorScope('validation');
    d.pushErrorScope('out-of-memory');
    const t0 = performance.now();
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.fused);
    pass.setBindGroup(0, group(this.fused, [bufs.header, bufs.indices, bufs.params, bufs.steps, bufs.terms, bufs.vals, bufs.expo, bufs.partials]));
    pass.dispatchWorkgroups(dispatch.fused);
    pass.setPipeline(this.reduce);
    pass.setBindGroup(0, group(this.reduce, [bufs.header, bufs.partials, bufs.sums]));
    pass.dispatchWorkgroups(dispatch.reduce);
    pass.setPipeline(this.pfe);
    pass.setBindGroup(0, group(this.pfe, [bufs.header, bufs.expo, bufs.pfe]));
    pass.dispatchWorkgroups(dispatch.pfe);
    pass.end();
    enc.copyBufferToBuffer(bufs.sums, 0, bufs.readSums, 0, bytes.sums);
    enc.copyBufferToBuffer(bufs.pfe, 0, bufs.readPfe, 0, bytes.pfe);
    d.queue.submit([enc.finish()]);
    globalThis.__ccrGpuRuns = (globalThis.__ccrGpuRuns || 0) + 1; // fused pipelines submitted (checked by the e2e test)
    await Promise.all([bufs.readSums.mapAsync(GPUMapMode.READ), bufs.readPfe.mapAsync(GPUMapMode.READ)]);
    const gpuMs = performance.now() - t0;
    const oom = await d.popErrorScope();
    const validation = await d.popErrorScope();
    const out = validation || oom ? null : {
      sums: new Float32Array(bufs.readSums.getMappedRange().slice(0)),
      pfe: new Float32Array(bufs.readPfe.getMappedRange().slice(0)),
      gpuMs,
    };
    for (const b of Object.values(bufs)) b.destroy();
    if (!out) throw new Error((validation || oom).message);
    if (this.lost) throw new Error('GPU device lost: ' + this.lost.message);
    return out;
  }
}
