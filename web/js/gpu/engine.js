// WebGPU host for the fused exposure kernels of the C++ library (ccr::gpu).
//
// No numerics live here. The library supplies the WGSL sources (gpuKernels) and, for every
// job, a plan with ready-made buffers, dispatch sizes and buffer sizes (gpuJobs). This class
// uploads the buffers, dispatches fused-exposure -> reduce-partials -> pfe-quantile and reads
// back two arrays, which go back to the library (gpuAnalyse) for all post-processing.

export class GpuEngine {
  static async create(kernels) {
    if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
    });
    const engine = new GpuEngine(adapter, device);
    await engine.compile(kernels);
    return engine;
  }

  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.info = adapter.info || {};
    this.name = [this.info.vendor, this.info.architecture].filter(Boolean).join(' ') || 'GPU';
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
