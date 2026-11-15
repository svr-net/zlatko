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
// 3. pfeQuantileKernel (row statistics): one workgroup per request computes a quantile (the
//    empirical quantile interpolated between order statistics, as ccr::quantile, from a
//    two-level histogram without sorting), a mean, a cross moment, a histogram or sample paths
//    of one row of the exposure buffer: PFE, initial margin and every path view.
//
// Buffer layouts are produced by gpu::compile(); constants match fused_exposure.hpp
// (workgroup 64, 7 statistics per date, 3072 PFE bins, 256 PFE threads), and
// runFusedReference() mirrors this code on the CPU.
#include "ccr/gpu/fused_exposure.hpp"

namespace ccr::gpu {

static_assert(kWorkgroupSize == 64 && kStatsPerDate == 8 && kTradeStats == 5 && kMaxTradeGroups == 16 && kMaxBermudans == 4 &&
                  kPfeBins == 3072 && kPfeWorkgroupSize == 256 && kHeaderWords == 32,
              "the WGSL sources below hard-code these constants");

namespace {

// The uniform block shared by the three kernels (kHeaderWords words, written by gpu::compile).
const char* const kHeaderWgsl = R"wgsl(
struct Header {
  nP : u32, nT : u32, nA : u32, nC : u32,
  substeps : u32, nNormals : u32, seed : u32, antithetic : u32,
  hasCsa : u32, cpty : u32, stepStride : u32, nCorr : u32,
  offCall : u32, offReporting : u32, quantile : f32, numWG : u32,
  nS : u32, nR : u32, nTr : u32, rowDV : u32,       // stats per date, reporting dates, trades; rows (see RowLayout)
  rowE : u32, rowV : u32, rowC : u32, rowGross : u32,
  rowTrades : u32, rowEx : u32, nReq : u32, nBerm : u32,
  rowEP : u32, pad0 : u32, pad1 : u32, pad2 : u32,
};
)wgsl";

std::string withHeader(const char* body) { return std::string(kHeaderWgsl) + body; }

}  // namespace

const std::string& fusedExposureKernel(bool perTrade) {
  static const char* const body = R"wgsl(

const WG : u32 = 64u;
const NB : u32 = 8u;          // netting-set statistics per date, before the per-trade ones
const NB_BERM : u32 = 4u;     // Bermudan swaptions per netting set
const NOT_EXERCISED : u32 = 0xFFFFFFFFu;
const NO_ROW : u32 = 0xFFFFFFFFu;
const STEP_VEC : u32 = 6u;    // vec4s per date in STEPS
const TERM_TILE : u32 = 64u;  // valuation terms staged in workgroup memory at a time
const CHOL_VEC : u32 = 16u;   // Cholesky factor: 8 padded rows of 2 vec4s

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> IDX : array<u32>;            // termStart[nT+1] | callIndex[nT] | reportingSlot[nT]
@group(0) @binding(2) var<storage, read> FP : array<vec4<f32>>;       // CIR x2 | CSA | spots | Cholesky rows
@group(0) @binding(3) var<storage, read> STEPS : array<vec4<f32>>;    // per date: HW | HW | carry | vol | var/2 | CIR++ shift
@group(0) @binding(4) var<storage, read> TERMS : array<vec4<f32>>;    // per term: (kind, slot, c, A) (B, p1, p2, p3)
@group(0) @binding(5) var<storage, read_write> VALS : array<f32>;     // [nT][nP] netted value (margin-call look-back, CSA only)
@group(0) @binding(6) var<storage, read_write> EXPO : array<f32>;     // [nR][nP] positive exposure, then [nR][nP] |dV| (margin)
@group(0) @binding(7) var<storage, read_write> PARTIALS : array<f32>; // [numWG][nT][nS]

// Workgroup tiles: data every path of the workgroup reads is staged once in fast shared
// memory instead of being fetched from storage by each of the 64 invocations.
var<workgroup> chol : array<vec4<f32>, 16>;
var<workgroup> stepTile : array<vec4<f32>, 6>;
var<workgroup> termTile : array<vec4<f32>, 128>;
var<workgroup> termRange : vec2<u32>;
var<workgroup> red : array<vec4<f32>, 128>;  // statistics reduction, 8 statistics (2 vec4) per invocation at a time

// AMC regression of a policy / continuation term: z = (x - c) * A, B + p1 z + p2 z^2 + p3 z^3.
fn amcPoly(a : vec4<f32>, b : vec4<f32>, x : f32) -> f32 {
  let z = (x - a.z) * a.w;
  return ((b.w * z + b.z) * z + b.y) * z + b.x;
}

// Per-trade statistic m of trade value vi in netting-set value v (uncollateralised), weight D.
fn tradeStat(m : u32, vi : f32, v : f32, D : f32) -> f32 {
  let pos = select(0.0, 1.0, v > 0.0);
  switch m {
    case 0u: { return D * vi * pos; }                       // Euler contribution to EE*
    case 1u: { return vi * pos; }                           // Euler contribution to EE
    case 2u: { return D * (max(v, 0.0) - max(v - vi, 0.0)); } // incremental EE*
    case 3u: { return D * max(vi, 0.0); }                   // standalone EE*
    default: { return max(vi, 0.0); }                       // standalone EE
  }
}

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
  // Discount factor and assets are accumulated in log space: the f32 error of a running
  // product grows with every date, that of a running sum of logs does not.
  var lnD = 0.0;
  var D = 1.0;
  let S0 = FP[3];                                // the four asset spots as one vector
  var lnS = vec4<f32>(0.0);
  var S = S0;
  var y = vec2<f32>(cir[0].w, cir[1].w);
  var Y = vec2<f32>(0.0, 0.0);
  var fix : array<f32, 32>;
  var z : array<vec4<f32>, 16>;                  // 64 normals, 4 per vector
  var w : array<f32, 8>;                         // correlated drivers (7 used)
  var held = 0.0;
  var prevDE = 0.0;
  var prevQ = 1.0;
  var prevRV = 0.0;  // netted value on the previous reporting date (margin)
  var exercised : array<u32, 4>;  // exercise index of each Bermudan on this path
  for (var i = 0u; i < NB_BERM; i++) { exercised[i] = NOT_EXERCISED; }
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
      lnD = lnD - integratedRate;
      D = exp(lnD);
      // All four log-normal assets in one vector step (unused lanes have zero volatility).
      let wa = vec4<f32>(w[1], w[2], w[3], w[4]);
      lnS = lnS + (vec4<f32>(integratedRate) - stepTile[2] - stepTile[4] + stepTile[3] * wa);
      S = S0 * exp(lnS);
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
    // Per-trade values (per-trade mode). Reset explicitly on every date: shader compilers may
    // hoist a function-scope var to the function entry and zero it only once.
    var vt : array<f32, MAX_TRADES>;
    for (var i = 0u; i < MAX_TRADES; i++) { vt[i] = 0.0; }
    var ev : array<f32, 4>;  // Bermudan exercise values on this date (reset explicitly, see vt)
    for (var i = 0u; i < NB_BERM; i++) { ev[i] = 0.0; }
    for (var t0 = range.x; t0 < range.y; t0 += TERM_TILE) {
      let n = min(TERM_TILE, range.y - t0);
      for (var q = l; q < 2u * n; q += WG) { termTile[q] = TERMS[2u * t0 + q]; }
      workgroupBarrier();
      for (var t = 0u; t < n; t++) {
        let a = termTile[2u * t];       // (kind, slot, c, A)
        let b = termTile[2u * t + 1u];  // (B, p1, p2, p3)
        let slot = u32(a.y);
        let code = u32(a.x);  // kind + 16 trade + 256 bermudan + 4096 extra
        let kind = code & 15u;
        let bm = min((code >> 8u) & 15u, NB_BERM - 1u);
        let extra = code >> 12u;
        var term = 0.0;
        switch kind {
          case 0u: { term = a.z * a.w * exp(-b.x * x); }
          case 1u: { term = (a.z * fix[slot] + b.y) * a.w * exp(-b.x * x); }
          case 2u: { term = a.z * S[slot] * a.w; }
          case 3u: { fix[slot] = (a.w * exp(-b.x * x) - 1.0) / a.z; }
          case 4u: {
            let bond = a.w * exp(-b.x * x);
            term = a.z * black(S[slot] * b.y / bond, b.z, abs(b.w), bond, b.w > 0.0);
          }
          // American Monte Carlo: exercise value, policy, continuation, exercised tail.
          case 5u: { ev[bm] = ev[bm] + a.z * a.w * exp(-b.x * x); }
          case 6u: {
            if (exercised[bm] == NOT_EXERCISED && ev[bm] > 0.0 && ev[bm] > amcPoly(a, b, x)) { exercised[bm] = extra; }
          }
          case 7u: { if (exercised[bm] == NOT_EXERCISED) { term = a.y * max(amcPoly(a, b, x), 0.0); } }
          case 8u: { if (exercised[bm] != NOT_EXERCISED && exercised[bm] <= extra) { term = a.z * a.w * exp(-b.x * x); } }
          case 10u: {  // risk-factor probe into row b.y: what = slot & 15, index = slot >> 4
            let what = slot & 15u;
            let ix = min(slot >> 4u, 3u);
            var q = 0.0;
            switch what {
              case 0u: { q = x; }
              case 1u: { q = D; }
              case 2u: { q = S[ix]; }
              case 3u: { q = lnS[ix]; }
              case 4u: { q = y[min(ix, 1u)]; }
              case 5u: { q = select(exp(-Y[min(ix, 1u)] - stepTile[5][min(ix, 1u)]), 1.0, j == 0u); }
              case 6u: { q = D * S[ix]; }
              default: { q = D * a.w * exp(-b.x * x); }
            }
            if (isActive) { EXPO[u32(b.y) * nP + p] = q + a.z; }
          }
          default: {
            if (exercised[bm] != NOT_EXERCISED && exercised[bm] <= extra) { term = (a.z * fix[slot] + b.y) * a.w * exp(-b.x * x); }
          }
        }
        if (H.nTr > 0u) {
          let tr = min((code >> 4u) & 15u, MAX_TRADES - 1u);
          vt[tr] = vt[tr] + term;
        } else {
          v = v + term;
        }
      }
      workgroupBarrier();
    }
    // Per-trade mode: the netting set is the sum of its trades.
    for (var i = 0u; i < min(H.nTr, MAX_TRADES); i++) { v = v + vt[i]; }

    // Collateral called on the value at t - MPR (already written by this invocation).
    var collateral = 0.0;
    if (H.hasCsa == 1u) {
      if (isActive) { VALS[j * nP + p] = v; }
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
    let rs = IDX[H.offReporting + j];
    if (isActive && rs > 0u) {
      // Rows of the path views on this reporting date (see RowLayout; NO_ROW blocks are absent).
      let k = rs - 1u;
      EXPO[(H.rowEP + k) * nP + p] = ep;
      // Initial margin: |V(t_k) - V(t_{k-1})| of the uncollateralised value between reporting dates.
      if (H.rowDV != NO_ROW) { EXPO[(H.rowDV + k) * nP + p] = select(abs(v - prevRV), 0.0, rs == 1u); }
      if (H.rowE != NO_ROW) { EXPO[(H.rowE + k) * nP + p] = e; }
      if (H.rowV != NO_ROW) { EXPO[(H.rowV + k) * nP + p] = v; }
      if (H.rowC != NO_ROW) { EXPO[(H.rowC + k) * nP + p] = collateral; }
      if (H.rowGross != NO_ROW) {
        var gross = 0.0;
        for (var i = 0u; i < min(H.nTr, MAX_TRADES); i++) { gross = gross + max(vt[i], 0.0); }
        EXPO[(H.rowGross + k) * nP + p] = gross;
      }
      if (H.rowTrades != NO_ROW) {
        for (var i = 0u; i < min(H.nTr, MAX_TRADES); i++) { EXPO[(H.rowTrades + i * H.nR + k) * nP + p] = vt[i]; }
      }
    }
    if (isActive && H.rowEx != NO_ROW) {
      for (var i = 0u; i < min(H.nBerm, NB_BERM); i++) {
        EXPO[(H.rowEx + i) * nP + p] = select(f32(exercised[i]), -1.0, exercised[i] == NOT_EXERCISED);
      }
    }

    var cvaInc = 0.0;
    if (rs > 0u) {
      prevRV = v;
      let de = D * ep;
      if (j > 0u) { cvaInc = 0.5 * (prevDE + de) * (prevQ - Q); }
      prevDE = de;
      prevQ = Q;
    }

    // Workgroup tree reduction of the statistics, 8 at a time (two vec4s per invocation): the
    // netting set's 8, then kTradeStats per trade in per-trade mode.
    let m = select(0.0, 1.0, isActive);
    for (var q = 0u; q < H.nS; q += 8u) {
      var st : array<f32, 8>;
      for (var c = 0u; c < 8u; c++) { st[c] = 0.0; }  // explicit reset (see vt)
      if (q == 0u) {
        st = array<f32, 8>(e, ep, en, D * ep, D * en, cvaInc, Q, D);
      } else {
        for (var c = 0u; c < 8u; c++) {
          let k = q + c - NB;
          let i = k / 5u;
          if (q + c < H.nS && i < min(H.nTr, MAX_TRADES)) { st[c] = tradeStat(k % 5u, vt[i], v, D); }
        }
      }
      red[2u * l] = m * vec4<f32>(st[0], st[1], st[2], st[3]);
      red[2u * l + 1u] = m * vec4<f32>(st[4], st[5], st[6], st[7]);
      workgroupBarrier();
      for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (l < s) {
          red[2u * l] = red[2u * l] + red[2u * (l + s)];
          red[2u * l + 1u] = red[2u * l + 1u] + red[2u * (l + s) + 1u];
        }
        workgroupBarrier();
      }
      if (l == 0u) {
        let o = (wid.x * nT + j) * H.nS + q;
        let r0 = red[0];
        let r1 = red[1];
        PARTIALS[o] = r0.x; PARTIALS[o + 1u] = r0.y; PARTIALS[o + 2u] = r0.z; PARTIALS[o + 3u] = r0.w;
        if (q + 4u < H.nS) { PARTIALS[o + 4u] = r1.x; PARTIALS[o + 5u] = r1.y; PARTIALS[o + 6u] = r1.z; PARTIALS[o + 7u] = r1.w; }
      }
      workgroupBarrier();
    }
  }
}
)wgsl";
  // MAX_TRADES sizes the per-trade value array: 1 in the default kernel (no register cost).
  static const std::string base = withHeader("const MAX_TRADES : u32 = 1u;\n") + body;
  static const std::string trades = withHeader("const MAX_TRADES : u32 = 16u;\n") + body;
  return perTrade ? trades : base;
}

const std::string& reducePartialsKernel() {
  static const std::string source = withHeader(R"wgsl(

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> PARTIALS : array<f32>;
@group(0) @binding(2) var<storage, read_write> SUMS : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = H.nT * H.nS;
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
)wgsl");
  return source;
}

const std::string& pfeQuantileKernel() {
  static const std::string source = withHeader(R"wgsl(
const BINS : u32 = 3072u;  // 12 KiB of histogram: fits the portable 16 KiB workgroup-storage limit
@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> EXPO : array<f32>;
@group(0) @binding(2) var<storage, read> REQ : array<vec4<f32>>;   // per request: (op, row, row2, count) (level, lo, hi, out)
@group(0) @binding(3) var<storage, read_write> OUT : array<f32>;

var<workgroup> hist : array<atomic<u32>, 3072>;
var<workgroup> wsum : array<f32, 256>;
var<workgroup> minKey : atomic<u32>;
var<workgroup> maxKey : atomic<u32>;
var<workgroup> loKey : atomic<u32>;     // extremes of the fine bin holding order statistic k
var<workgroup> hiKey : atomic<u32>;
var<workgroup> aboveKey : atomic<u32>;  // smallest value above that fine bin
var<workgroup> req : array<vec4<f32>, 2>;
var<workgroup> range : vec2<f32>;
var<workgroup> pick : vec4<u32>;        // (coarse bin, paths below it, -, -)
var<workgroup> fine : vec4<u32>;        // (fine bin, paths below it, paths in it, -)

// f32 <-> u32 keys that order like the values, signed ones included, for atomic min / max.
fn toKey(v : f32) -> u32 {
  let b = bitcast<u32>(v);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}
fn fromKey(k : u32) -> f32 { return bitcast<f32>(select(~k, k & 0x7FFFFFFFu, (k & 0x80000000u) != 0u)); }

fn rowValue(op : u32, row : u32, p : u32) -> f32 {
  let x = EXPO[row * H.nP + p];
  return select(x, max(x, 0.0), op == 7u);
}
fn coarseBin(v : f32, lo : f32, span : f32) -> u32 { return min(u32(max((v - lo) / span, 0.0) * f32(BINS)), BINS - 1u); }
fn fineBin(v : f32, lo : f32, width : f32) -> u32 { return min(u32(clamp((v - lo) / width, 0.0, 1.0) * f32(BINS)), BINS - 1u); }

// Row statistics, one workgroup per request (see ccr::gpu::RowOp). Means are Kahan sums per
// invocation and a tree across the workgroup. Quantiles are the empirical quantile interpolated
// between order statistics k = floor(rank) and k + 1 (as ccr::quantile), found without sorting:
// a coarse histogram over [min, max] locates order statistic k, a fine histogram splits that bin
// BINS ways, and a last pass reads the extremes of k's fine bin and the next value above it.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let l = lid.x;
  if (l < 2u) { req[l] = REQ[2u * wid.x + l]; }
  let r0 = workgroupUniformLoad(&req[0]);
  let r1 = workgroupUniformLoad(&req[1]);
  let op = u32(r0.x);
  let row = u32(r0.y);
  let row2 = u32(r0.z);
  let count = u32(r0.w);
  let out = u32(r1.w);
  let nP = H.nP;

  if (op == 4u) {  // samples
    for (var p = l; p < count; p += 256u) { OUT[out + p] = EXPO[row * nP + p]; }
    return;
  }
  if (op != 0u && op != 5u && op != 7u) {  // means
    var sum = 0.0;
    var comp = 0.0;
    for (var p = l; p < nP; p += 256u) {
      let x = EXPO[row * nP + p];
      var f = x;
      switch op {
        case 2u: { f = max(x, 0.0); }
        case 3u: { f = min(x, 0.0); }
        case 6u: { f = x * EXPO[row2 * nP + p]; }
        case 8u: { f = max(x, 0.0) * EXPO[row2 * nP + p]; }
        case 9u: { f = min(x, 0.0) * EXPO[row2 * nP + p]; }
        default: { }
      }
      let yv = f - comp;
      let tv = sum + yv;
      comp = (tv - sum) - yv;
      sum = tv;
    }
    wsum[l] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
      if (l < s) { wsum[l] = wsum[l] + wsum[l + s]; }
      workgroupBarrier();
    }
    if (l == 0u) { OUT[out] = wsum[0] / f32(nP); }
    return;
  }

  // Histogram and quantiles: the range first.
  for (var i = l; i < BINS; i += 256u) { atomicStore(&hist[i], 0u); }
  if (l == 0u) {
    atomicStore(&minKey, 0xFFFFFFFFu);
    atomicStore(&maxKey, 0u);
    atomicStore(&loKey, 0xFFFFFFFFu);
    atomicStore(&hiKey, 0u);
    atomicStore(&aboveKey, 0xFFFFFFFFu);
  }
  workgroupBarrier();
  var kMin = 0xFFFFFFFFu;
  var kMax = 0u;
  for (var p = l; p < nP; p += 256u) {
    let key = toKey(rowValue(op, row, p));
    kMin = min(kMin, key);
    kMax = max(kMax, key);
  }
  atomicMin(&minKey, kMin);
  atomicMax(&maxKey, kMax);
  workgroupBarrier();
  if (l == 0u) { range = vec2<f32>(fromKey(atomicLoad(&minKey)), fromKey(atomicLoad(&maxKey))); }
  let rg = workgroupUniformLoad(&range);  // uniform: the branches below contain barriers
  var lo = rg.x;
  var hi = rg.y;

  if (op == 5u) {  // histogram of `count` bins on [lo, hi] (the given range unless r1.y == r1.z)
    if (r1.y != r1.z) { lo = r1.y; hi = r1.z; }
    let span = max(hi - lo, 1e-30);
    for (var p = l; p < nP; p += 256u) {
      let b = min(u32(max((EXPO[row * nP + p] - lo) / span, 0.0) * f32(count)), count - 1u);
      atomicAdd(&hist[b], 1u);
    }
    workgroupBarrier();
    for (var b = l; b < count; b += 256u) { OUT[out + b] = f32(atomicLoad(&hist[b])); }
    if (l == 0u) { OUT[out + count] = lo; OUT[out + count + 1u] = hi; }
    return;
  }

  let rank = r1.x * f32(nP - 1u);
  let k = min(u32(floor(rank)), nP - 1u);
  let frac = rank - f32(k);
  let span = hi - lo;
  if (span <= 0.0) {  // every path has the same value
    if (l == 0u) { OUT[out] = lo; }
    return;
  }
  for (var p = l; p < nP; p += 256u) { atomicAdd(&hist[coarseBin(rowValue(op, row, p), lo, span)], 1u); }
  workgroupBarrier();
  if (l == 0u) {
    var cum = 0u;
    var b = 0u;
    for (; b < BINS - 1u; b++) {
      let c = atomicLoad(&hist[b]);
      if (cum + c > k) { break; }
      cum = cum + c;
    }
    pick = vec4<u32>(b, cum, 0u, 0u);
  }
  let chosen = workgroupUniformLoad(&pick);

  // Fine histogram inside the coarse bin of order statistic k.
  for (var i = l; i < BINS; i += 256u) { atomicStore(&hist[i], 0u); }
  workgroupBarrier();
  let width = span / f32(BINS);
  let blo = lo + f32(chosen.x) * width;
  for (var p = l; p < nP; p += 256u) {
    let v = rowValue(op, row, p);
    if (coarseBin(v, lo, span) == chosen.x) { atomicAdd(&hist[fineBin(v, blo, width)], 1u); }
  }
  workgroupBarrier();
  if (l == 0u) {
    var cum = chosen.y;
    var b = 0u;
    var c = 0u;
    for (; b < BINS; b++) {
      c = atomicLoad(&hist[b]);
      if (cum + c > k) { break; }
      cum = cum + c;
    }
    fine = vec4<u32>(min(b, BINS - 1u), cum, c, 0u);
  }
  let f = workgroupUniformLoad(&fine);

  // Extremes of k's fine bin and the next value above it.
  var bLo = 0xFFFFFFFFu;
  var bHi = 0u;
  var above = 0xFFFFFFFFu;
  for (var p = l; p < nP; p += 256u) {
    let v = rowValue(op, row, p);
    let cb = coarseBin(v, lo, span);
    var fb = 0u;
    if (cb == chosen.x) { fb = fineBin(v, blo, width); }
    if (cb == chosen.x && fb == f.x) {
      bLo = min(bLo, toKey(v));
      bHi = max(bHi, toKey(v));
    } else if (cb > chosen.x || (cb == chosen.x && fb > f.x)) {
      above = min(above, toKey(v));
    }
  }
  atomicMin(&loKey, bLo);
  atomicMax(&hiKey, bHi);
  atomicMin(&aboveKey, above);
  workgroupBarrier();
  if (l == 0u) {
    let vLo = fromKey(atomicLoad(&loKey));
    let vHi = fromKey(atomicLoad(&hiKey));
    let i = k - f.y;  // position of k inside its fine bin (f.z paths)
    let spread = vHi - vLo;
    var ek = vHi;
    if (f.z > 1u) { ek = vLo + spread * f32(i) / f32(f.z - 1u); }
    var result = ek;
    if (frac > 0.0 && k + 1u < nP) {
      var ek1 = fromKey(atomicLoad(&aboveKey));
      if (i + 1u < f.z) { ek1 = vLo + spread * f32(i + 1u) / f32(f.z - 1u); }
      result = ek + frac * (ek1 - ek);
    }
    OUT[out] = result;
  }
}
)wgsl");
  return source;
}

}  // namespace ccr::gpu
