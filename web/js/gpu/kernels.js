// WGSL compute kernels for the fused exposure pipeline.
//
// 1. FUSED_EXPOSURE: one invocation per Monte Carlo path. For every simulation date it
//    draws counter-based normals, correlates them (Cholesky), advances Hull-White rates
//    (exact joint step of x and the integrated short rate), the log-normal assets and
//    the CIR++ intensities (Brownian-bridge sub-steps), revalues the whole netting set
//    from the compiled term table, applies the CSA (thresholds, MTA, independent amount,
//    margin period of risk), and reduces the exposure statistics across the workgroup.
//    No scenario cube is ever materialised: only the netted value (needed for the margin
//    call look-back) and the positive exposure (for PFE) are written.
// 2. REDUCE_PARTIALS: sums the per-workgroup partial statistics.
// 3. PFE_QUANTILE: one workgroup per date builds a histogram of the exposure distribution
//    and reads off the requested quantile.
//
// Table layouts are produced by the WASM build of the C++ library (gpuPlan).

export const WG = 64;
export const NF = 7; // statistics per date: E[V], EE, ENE, EE*, ENE*, pathwise CVA increment, E[Q]

const HEADER = /* wgsl */ `
struct Header {
  nP : u32, nT : u32, nA : u32, nC : u32,
  substeps : u32, nNormals : u32, seed : u32, antithetic : u32,
  hasCsa : u32, cpty : u32, stepStride : u32, nCorr : u32,
  offCall : u32, offReporting : u32, quantile : f32, numWG : u32,
};
`;

export const FUSED_EXPOSURE = /* wgsl */ `
${HEADER}
const WG : u32 = ${WG}u;
const NF : u32 = ${NF}u;
const MAXA : u32 = 4u;
const MAXF : u32 = 7u;

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> IDX : array<u32>;      // termStart[nT+1] | callIndex[nT] | isReporting[nT]
@group(0) @binding(2) var<storage, read> FP : array<f32>;       // CIR params, CSA, spots, Cholesky factor
@group(0) @binding(3) var<storage, read> STEPS : array<f32>;    // per-date transition coefficients
@group(0) @binding(4) var<storage, read> TERMS : array<f32>;    // valuation terms, 8 floats each
@group(0) @binding(5) var<storage, read_write> VALS : array<f32>;     // [nT][nP] netted value
@group(0) @binding(6) var<storage, read_write> EXPO : array<f32>;     // [nT][nP] positive exposure
@group(0) @binding(7) var<storage, read_write> PARTIALS : array<f32>; // [numWG][nT][NF]

var<workgroup> red : array<f32, ${WG * NF}>;

fn pcg(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn gauss(stream : u32, j : u32, k : u32) -> f32 {
  let h1 = pcg(stream ^ pcg(j * 131u + k));
  let h2 = pcg(h1 ^ 0x68E31DA4u);
  let u1 = (f32(h1 >> 8u) + 0.5) / 16777216.0;
  let u2 = (f32(h2 >> 8u) + 0.5) / 16777216.0;
  return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

fn ncdf(x : f32) -> f32 {
  let z = abs(x) * 0.70710678;
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  let erf = 1.0 - poly * exp(-z * z);
  return select(0.5 * (1.0 - erf), 0.5 * (1.0 + erf), x >= 0.0);
}

fn black(F : f32, K : f32, sd : f32, df : f32, isCall : bool) -> f32 {
  let s = select(-1.0, 1.0, isCall);
  if (sd <= 1e-7 || F <= 0.0) { return df * max(s * (F - K), 0.0); }
  let d1 = log(F / K) / sd + 0.5 * sd;
  let d2 = d1 - sd;
  return df * s * (F * ncdf(s * d1) - K * ncdf(s * d2));
}

fn requiredCollateral(v : f32) -> f32 {
  let hc = FP[8];
  let hb = FP[9];
  var c = 0.0;
  if (v > hc) { c = c + (v - hc); }
  if (hb >= 0.0 && -v > hb) { c = c - (-v - hb); }
  return c;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let nP = H.nP;
  let nT = H.nT;
  let p = gid.x;
  let isActive = p < nP;
  let pp = min(p, nP - 1u);

  // Antithetic pairs share a stream and flip the sign of every normal.
  var sign = 1.0;
  var pathStream = p;
  if (H.antithetic == 1u) {
    pathStream = p >> 1u;
    if ((p & 1u) == 1u) { sign = -1.0; }
  }
  let stream = pcg(pcg(H.seed) ^ pcg(pathStream + 0x9E3779B9u));

  var x = 0.0;
  var D = 1.0;
  var S : array<f32, 4>;
  for (var a = 0u; a < MAXA; a++) { S[a] = FP[12u + a]; }
  var y : array<f32, 2>;
  var Y : array<f32, 2>;
  for (var c = 0u; c < 2u; c++) { y[c] = FP[4u * c + 3u]; Y[c] = 0.0; }
  var fix : array<f32, 32>;
  var z : array<f32, 64>;
  var w : array<f32, 7>;
  var held = 0.0;
  var prevDE = 0.0;
  var prevQ = 1.0;
  let ia = FP[11];
  let mta = FP[10];

  for (var j = 0u; j < nT; j++) {
    var Q = 1.0;
    if (j > 0u) {
      let base = j * H.stepStride;
      for (var k = 0u; k < H.nNormals; k++) { z[k] = sign * gauss(stream, j, k); }
      for (var i = 0u; i < H.nCorr; i++) {
        var acc = 0.0;
        for (var k = 0u; k <= i; k++) { acc = acc + FP[16u + i * MAXF + k] * z[k]; }
        w[i] = acc;
      }
      // Hull-White: exact joint transition of x and int r dt.
      let integratedRate = x * STEPS[base + 5u] + STEPS[base + 2u] * w[0] + STEPS[base + 3u] * z[H.nCorr] + STEPS[base + 4u];
      x = x * STEPS[base] + STEPS[base + 1u] * w[0];
      D = D * exp(-integratedRate);
      // Log-normal assets drifting at the simulated domestic rate.
      for (var a = 0u; a < H.nA; a++) {
        let o = base + 7u + 3u * a;
        S[a] = S[a] * exp(integratedRate - STEPS[o] - STEPS[o + 2u] + STEPS[o + 1u] * w[1u + a]);
      }
      // CIR++ intensities: full-truncation Euler on Brownian-bridge sub-steps.
      let dt = STEPS[base + 6u];
      var off = H.nCorr + 1u;
      for (var c = 0u; c < H.nC; c++) {
        let kappa = FP[4u * c];
        let theta = FP[4u * c + 1u];
        let xi = FP[4u * c + 2u];
        let h = dt / f32(H.substeps);
        var rem = sqrt(dt) * w[1u + H.nA + c];
        var remT = dt;
        var yy = y[c];
        var YY = Y[c];
        for (var s = 0u; s < H.substeps; s++) {
          var dW = rem;
          if (s + 1u < H.substeps) {
            dW = h / remT * rem + sqrt(max(h * (remT - h) / remT, 0.0)) * z[off + s];
          }
          rem = rem - dW;
          remT = remT - h;
          let yp = max(yy, 0.0);
          let yn = yy + kappa * (theta - yp) * h + xi * sqrt(yp) * dW;
          YY = YY + 0.5 * (yp + max(yn, 0.0)) * h;
          yy = yn;
        }
        y[c] = yy;
        Y[c] = YY;
        off = off + H.substeps - 1u;
      }
      if (H.nC > 0u) { Q = exp(-Y[H.cpty] - STEPS[base + 7u + 3u * MAXA + H.cpty]); }
    }

    // Revalue the netting set from the compiled term table.
    var v = 0.0;
    let t1 = IDX[j + 1u];
    for (var t = IDX[j]; t < t1; t++) {
      let r = t * 8u;
      let kind = u32(TERMS[r]);
      let slot = u32(TERMS[r + 1u]);
      let c = TERMS[r + 2u];
      let bond = TERMS[r + 3u] * exp(-TERMS[r + 4u] * x);
      switch kind {
        case 0u: { v = v + c * bond; }
        case 1u: { v = v + c * fix[slot] * bond; }
        case 2u: { v = v + c * S[slot] * TERMS[r + 3u]; }
        case 3u: { fix[slot] = (bond / (TERMS[r + 5u] * exp(-TERMS[r + 6u] * x)) - 1.0) / c; }
        default: {
          let p3 = TERMS[r + 7u];
          v = v + c * black(S[slot] * TERMS[r + 5u] / bond, TERMS[r + 6u], abs(p3), bond, p3 > 0.0);
        }
      }
    }

    // Collateral called on the value at t - MPR (already written by this invocation).
    var collateral = 0.0;
    if (isActive) { VALS[j * nP + p] = v; }
    if (H.hasCsa == 1u) {
      let callJ = IDX[H.offCall + j];
      var vc = v;
      if (callJ != j) { vc = VALS[callJ * nP + pp]; }
      let required = requiredCollateral(vc);
      if (j == 0u || abs(required - held) >= mta) { held = required; }
      collateral = held + ia;
    }
    let e = v - collateral;
    let ep = max(e, 0.0);
    let en = min(e, 0.0);
    if (isActive) { EXPO[j * nP + p] = ep; }

    var cvaInc = 0.0;
    if (IDX[H.offReporting + j] == 1u) {
      let de = D * ep;
      if (j > 0u) { cvaInc = 0.5 * (prevDE + de) * (prevQ - Q); }
      prevDE = de;
      prevQ = Q;
    }

    // Workgroup tree reduction of the statistics for this date.
    let b = lid.x * NF;
    let m = select(0.0, 1.0, isActive);
    red[b] = m * v;
    red[b + 1u] = m * ep;
    red[b + 2u] = m * en;
    red[b + 3u] = m * D * ep;
    red[b + 4u] = m * D * en;
    red[b + 5u] = m * cvaInc;
    red[b + 6u] = m * Q;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
      if (lid.x < s) {
        for (var f = 0u; f < NF; f++) { red[b + f] = red[b + f] + red[(lid.x + s) * NF + f]; }
      }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      for (var f = 0u; f < NF; f++) { PARTIALS[(wid.x * nT + j) * NF + f] = red[f]; }
    }
    workgroupBarrier();
  }
}
`;

export const REDUCE_PARTIALS = /* wgsl */ `
${HEADER}
@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> PARTIALS : array<f32>;
@group(0) @binding(2) var<storage, read_write> SUMS : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = H.nT * ${NF}u;
  let i = gid.x;
  if (i >= n) { return; }
  // Kahan summation across workgroups.
  var sum = 0.0;
  var comp = 0.0;
  for (var g = 0u; g < H.numWG; g++) {
    let yv = PARTIALS[g * n + i] - comp;
    let tv = sum + yv;
    comp = (tv - sum) - yv;
    sum = tv;
  }
  SUMS[i] = sum / f32(H.nP);
}
`;

export const PFE_QUANTILE = /* wgsl */ `
${HEADER}
const BINS : u32 = 3072u;  // 12 KiB of histogram: fits the portable 16 KiB workgroup-storage limit
@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> EXPO : array<f32>;
@group(0) @binding(2) var<storage, read_write> PFE : array<f32>;

var<workgroup> hist : array<atomic<u32>, 3072>;
var<workgroup> wmax : array<f32, 256>;
var<workgroup> zeros : atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let j = wid.x;
  let nP = H.nP;
  var m = 0.0;
  for (var p = lid.x; p < nP; p += 256u) { m = max(m, EXPO[j * nP + p]); }
  wmax[lid.x] = m;
  for (var i = lid.x; i < BINS; i += 256u) { atomicStore(&hist[i], 0u); }
  if (lid.x == 0u) { atomicStore(&zeros, 0u); }
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { wmax[lid.x] = max(wmax[lid.x], wmax[lid.x + s]); }
    workgroupBarrier();
  }
  let mx = wmax[0];
  for (var p = lid.x; p < nP; p += 256u) {
    let v = EXPO[j * nP + p];
    if (v <= 0.0) {
      atomicAdd(&zeros, 1u);
    } else {
      atomicAdd(&hist[min(u32(v / mx * f32(BINS)), BINS - 1u)], 1u);
    }
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    let rank = H.quantile * f32(nP - 1u);
    var cum = f32(atomicLoad(&zeros));
    var result = 0.0;
    if (rank >= cum && mx > 0.0) {
      result = mx;
      let width = mx / f32(BINS);
      for (var b = 0u; b < BINS; b++) {
        let c = f32(atomicLoad(&hist[b]));
        if (cum + c > rank) {
          result = (f32(b) + (rank - cum + 0.5) / c) * width;
          break;
        }
        cum = cum + c;
      }
    }
    PFE[j] = result;
  }
}
`;
