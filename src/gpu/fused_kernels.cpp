// WGSL compute kernels of the fused exposure pipeline (see fused_exposure.hpp).
//
// 1. fusedExposureKernel: one invocation per Monte Carlo path. For every simulation date it
//    draws counter-based normals, correlates them (Cholesky), advances Hull-White rates
//    (exact joint step of x and the integrated short rate), the log-normal assets and
//    the CIR++ intensities (Brownian-bridge sub-steps), revalues the whole netting set
//    from the compiled term table, applies the CSA (thresholds, MTA, independent amount,
//    margin period of risk), and reduces the exposure statistics across the workgroup.
//    No scenario cube is ever materialised: only the netted value (needed for the margin
//    call look-back) and the positive exposure (for PFE) are written.
// 2. reducePartialsKernel: sums the per-workgroup partial statistics.
// 3. pfeQuantileKernel: one workgroup per date builds a histogram of the exposure distribution
//    and reads off the requested quantile.
//
// Buffer layouts are produced by gpu::compile(); constants match fused_exposure.hpp
// (workgroup 64, 7 statistics per date, 3072 PFE bins, 256 PFE threads), and
// runFusedReference() mirrors this code on the CPU.
#include "ccr/gpu/fused_exposure.hpp"

namespace ccr::gpu {

static_assert(kWorkgroupSize == 64 && kStatsPerDate == 7 && kPfeBins == 3072 && kPfeWorkgroupSize == 256,
              "the WGSL sources below hard-code these constants");

const std::string& fusedExposureKernel() {
  static const std::string source = R"wgsl(
struct Header {
  nP : u32, nT : u32, nA : u32, nC : u32,
  substeps : u32, nNormals : u32, seed : u32, antithetic : u32,
  hasCsa : u32, cpty : u32, stepStride : u32, nCorr : u32,
  offCall : u32, offReporting : u32, quantile : f32, numWG : u32,
};

const WG : u32 = 64u;
const NF : u32 = 7u;
const STEP_VEC : u32 = 6u;    // vec4s per date in STEPS
const TERM_TILE : u32 = 64u;  // valuation terms staged in workgroup memory at a time
const CHOL_VEC : u32 = 16u;   // Cholesky factor: 8 padded rows of 2 vec4s

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> IDX : array<u32>;            // termStart[nT+1] | callIndex[nT] | isReporting[nT]
@group(0) @binding(2) var<storage, read> FP : array<vec4<f32>>;       // CIR x2 | CSA | spots | Cholesky rows
@group(0) @binding(3) var<storage, read> STEPS : array<vec4<f32>>;    // per date: HW | HW | carry | vol | var/2 | CIR++ shift
@group(0) @binding(4) var<storage, read> TERMS : array<vec4<f32>>;    // per term: (kind, slot, c, A) (B, p1, p2, p3)
@group(0) @binding(5) var<storage, read_write> VALS : array<f32>;     // [nT][nP] netted value (margin-call look-back)
@group(0) @binding(6) var<storage, read_write> EXPO : array<f32>;     // [nT][nP] positive exposure
@group(0) @binding(7) var<storage, read_write> PARTIALS : array<f32>; // [numWG][nT][NF]

// Workgroup tiles: data every path of the workgroup reads is staged once in fast shared
// memory instead of being fetched from storage by each of the 64 invocations.
var<workgroup> chol : array<vec4<f32>, 16>;
var<workgroup> stepTile : array<vec4<f32>, 6>;
var<workgroup> termTile : array<vec4<f32>, 128>;
var<workgroup> termRange : vec2<u32>;
var<workgroup> red : array<vec4<f32>, 128>;  // statistics reduction, 2 vec4 per invocation

fn pcg(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Box-Muller pair: one hash pair, one log and one sqrt give two independent normals.
fn gauss2(stream : u32, j : u32, k : u32) -> vec2<f32> {
  let h1 = pcg(stream ^ pcg(j * 131u + k));
  let h2 = pcg(h1 ^ 0x68E31DA4u);
  let u1 = (f32(h1 >> 8u) + 0.5) / 16777216.0;
  let u2 = (f32(h2 >> 8u) + 0.5) / 16777216.0;
  let a = 6.2831853 * u2;
  return sqrt(-2.0 * log(u1)) * vec2<f32>(cos(a), sin(a));
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

fn requiredCollateral(v : f32, csa : vec4<f32>) -> f32 {
  var c = 0.0;
  if (v > csa.x) { c = c + (v - csa.x); }
  if (csa.y >= 0.0 && -v > csa.y) { c = c - (-v - csa.y); }
  return c;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let nP = H.nP;
  let nT = H.nT;
  let p = gid.x;
  let l = lid.x;
  let isActive = p < nP;
  let pp = min(p, nP - 1u);

  // Stage the Cholesky factor once per workgroup.
  if (l < CHOL_VEC) { chol[l] = FP[4u + l]; }

  // Antithetic pairs share a stream and flip the sign of every normal.
  var sign = 1.0;
  var pathStream = p;
  if (H.antithetic == 1u) {
    pathStream = p >> 1u;
    if ((p & 1u) == 1u) { sign = -1.0; }
  }
  let stream = pcg(pcg(H.seed) ^ pcg(pathStream + 0x9E3779B9u));

  var cir = array<vec4<f32>, 2>(FP[0], FP[1]);  // (kappa, theta, xi, y0) per credit
  let csa = FP[2];                               // (H cpty, H own, MTA, IA)
  var x = 0.0;
  var D = 1.0;
  var S = FP[3];                                 // the four asset spots as one vector
  var y = vec2<f32>(cir[0].w, cir[1].w);
  var Y = vec2<f32>(0.0, 0.0);
  var fix : array<f32, 32>;
  var z : array<vec4<f32>, 16>;                  // 64 normals, 4 per vector
  var w : array<f32, 8>;                         // correlated drivers (7 used)
  var held = 0.0;
  var prevDE = 0.0;
  var prevQ = 1.0;
  let nZ = (H.nNormals + 3u) / 4u;

  for (var j = 0u; j < nT; j++) {
    // Stage this date's step coefficients and term range for the whole workgroup.
    if (l < STEP_VEC) { stepTile[l] = STEPS[j * STEP_VEC + l]; }
    if (l == 0u) { termRange = vec2<u32>(IDX[j], IDX[j + 1u]); }
    let range = workgroupUniformLoad(&termRange);  // includes the barrier

    var Q = 1.0;
    if (j > 0u) {
      for (var q = 0u; q < nZ; q++) {
        z[q] = sign * vec4<f32>(gauss2(stream, j, 2u * q), gauss2(stream, j, 2u * q + 1u));
      }
      // Correlate: each padded Cholesky row is two vec4 dot products (fully unrolled).
      for (var i = 0u; i < 8u; i++) { w[i] = dot(chol[2u * i], z[0]) + dot(chol[2u * i + 1u], z[1]); }

      let hw0 = stepTile[0];  // (e^{-a dt}, sd x, c1, c2)
      let hw1 = stepTile[1];  // (int phi, (1 - e^{-a dt}) / a, dt, -)
      let nC = H.nCorr;
      let zI = z[nC >> 2u][nC & 3u];
      // Hull-White: exact joint transition of x and int r dt.
      let integratedRate = x * hw1.y + hw0.z * w[0] + hw0.w * zI + hw1.x;
      x = x * hw0.x + hw0.y * w[0];
      D = D * exp(-integratedRate);
      // All four log-normal assets in one vector step (unused lanes have zero volatility).
      let wa = vec4<f32>(w[1], w[2], w[3], w[4]);
      S = S * exp(vec4<f32>(integratedRate) - stepTile[2] - stepTile[4] + stepTile[3] * wa);
      // CIR++ intensities: full-truncation Euler on Brownian-bridge sub-steps.
      let dt = hw1.z;
      var off = nC + 1u;
      for (var c = 0u; c < H.nC; c++) {
        let prm = cir[c];
        let h = dt / f32(H.substeps);
        var rem = sqrt(dt) * w[1u + H.nA + c];
        var remT = dt;
        var yy = y[c];
        var YY = Y[c];
        for (var s = 0u; s < H.substeps; s++) {
          var dW = rem;
          if (s + 1u < H.substeps) {
            let k = off + s;
            dW = h / remT * rem + sqrt(max(h * (remT - h) / remT, 0.0)) * z[k >> 2u][k & 3u];
          }
          rem = rem - dW;
          remT = remT - h;
          let yp = max(yy, 0.0);
          let yn = yy + prm.x * (prm.y - yp) * h + prm.z * sqrt(yp) * dW;
          YY = YY + 0.5 * (yp + max(yn, 0.0)) * h;
          yy = yn;
        }
        y[c] = yy;
        Y[c] = YY;
        off = off + H.substeps - 1u;
      }
      if (H.nC > 0u) { Q = exp(-Y[H.cpty] - stepTile[5][H.cpty]); }
    }

    // Revalue the netting set. The term table of this date is a matrix (terms x 8) shared by
    // every path: it is streamed through workgroup memory in tiles of TERM_TILE terms, loaded
    // cooperatively as vec4s, and each invocation applies the tile to its own path state.
    var v = 0.0;
    for (var t0 = range.x; t0 < range.y; t0 += TERM_TILE) {
      let n = min(TERM_TILE, range.y - t0);
      for (var q = l; q < 2u * n; q += WG) { termTile[q] = TERMS[2u * t0 + q]; }
      workgroupBarrier();
      for (var t = 0u; t < n; t++) {
        let a = termTile[2u * t];       // (kind, slot, c, A)
        let b = termTile[2u * t + 1u];  // (B, p1, p2, p3)
        let slot = u32(a.y);
        let bond = a.w * exp(-b.x * x);
        switch u32(a.x) {
          case 0u: { v = v + a.z * bond; }
          case 1u: { v = v + a.z * fix[slot] * bond; }
          case 2u: { v = v + a.z * S[slot] * a.w; }
          case 3u: { fix[slot] = (bond / (b.y * exp(-b.z * x)) - 1.0) / a.z; }
          default: { v = v + a.z * black(S[slot] * b.y / bond, b.z, abs(b.w), bond, b.w > 0.0); }
        }
      }
      workgroupBarrier();
    }

    // Collateral called on the value at t - MPR (already written by this invocation).
    var collateral = 0.0;
    if (isActive) { VALS[j * nP + p] = v; }
    if (H.hasCsa == 1u) {
      let callJ = IDX[H.offCall + j];
      var vc = v;
      if (callJ != j) { vc = VALS[callJ * nP + pp]; }
      let required = requiredCollateral(vc, csa);
      if (j == 0u || abs(required - held) >= csa.z) { held = required; }
      collateral = held + csa.w;
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

    // Workgroup tree reduction of the statistics, two vec4s per invocation.
    let m = select(0.0, 1.0, isActive);
    red[2u * l] = m * vec4<f32>(e, ep, en, D * ep);
    red[2u * l + 1u] = m * vec4<f32>(D * en, cvaInc, Q, 0.0);
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
      if (l < s) {
        red[2u * l] = red[2u * l] + red[2u * (l + s)];
        red[2u * l + 1u] = red[2u * l + 1u] + red[2u * (l + s) + 1u];
      }
      workgroupBarrier();
    }
    if (l == 0u) {
      let o = (wid.x * nT + j) * NF;
      let r0 = red[0];
      let r1 = red[1];
      PARTIALS[o] = r0.x; PARTIALS[o + 1u] = r0.y; PARTIALS[o + 2u] = r0.z; PARTIALS[o + 3u] = r0.w;
      PARTIALS[o + 4u] = r1.x; PARTIALS[o + 5u] = r1.y; PARTIALS[o + 6u] = r1.z;
    }
    workgroupBarrier();
  }
}
)wgsl";
  return source;
}

const std::string& reducePartialsKernel() {
  static const std::string source = R"wgsl(
struct Header {
  nP : u32, nT : u32, nA : u32, nC : u32,
  substeps : u32, nNormals : u32, seed : u32, antithetic : u32,
  hasCsa : u32, cpty : u32, stepStride : u32, nCorr : u32,
  offCall : u32, offReporting : u32, quantile : f32, numWG : u32,
};

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> PARTIALS : array<f32>;
@group(0) @binding(2) var<storage, read_write> SUMS : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = H.nT * 7u;
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
)wgsl";
  return source;
}

const std::string& pfeQuantileKernel() {
  static const std::string source = R"wgsl(
struct Header {
  nP : u32, nT : u32, nA : u32, nC : u32,
  substeps : u32, nNormals : u32, seed : u32, antithetic : u32,
  hasCsa : u32, cpty : u32, stepStride : u32, nCorr : u32,
  offCall : u32, offReporting : u32, quantile : f32, numWG : u32,
};

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
)wgsl";
  return source;
}

}  // namespace ccr::gpu
