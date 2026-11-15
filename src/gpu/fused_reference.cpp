// CPU execution of the fused kernels (fused_kernels.cpp), statement for statement in
// single precision. Used to unit-test the GPU algorithm natively, and as the CPU backend of
// the fused pipeline: runFusedWorkgroups() runs any range of workgroups (so the work can be
// split across threads or Web Workers) and finishFused() reduces the slices exactly as the
// reduce-partials and pfe-quantile kernels do, so the result does not depend on the split.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "ccr/core/parallel.hpp"
#include "ccr/gpu/fused_exposure.hpp"

namespace ccr::gpu {

namespace {

std::uint32_t pcg(std::uint32_t v) {
  const std::uint32_t state = v * 747796405u + 2891336453u;
  const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Box-Muller pair k: normals 2k (cosine) and 2k + 1 (sine), as gauss2 in the kernel.
void gauss2(std::uint32_t stream, std::uint32_t j, std::uint32_t k, float* out) {
  const std::uint32_t h1 = pcg(stream ^ pcg(j * 131u + k));
  const std::uint32_t h2 = pcg(h1 ^ 0x68E31DA4u);
  const float u1 = (static_cast<float>(h1 >> 8u) + 0.5f) / 16777216.0f;
  const float u2 = (static_cast<float>(h2 >> 8u) + 0.5f) / 16777216.0f;
  const float r = std::sqrt(-2.0f * std::log(u1)), a = 6.2831853f * u2;
  out[0] = r * std::cos(a);
  out[1] = r * std::sin(a);
}

float ncdf(float x) {
  const float z = std::fabs(x) * 0.70710678f;
  const float t = 1.0f / (1.0f + 0.3275911f * z);
  const float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
  const float erf = 1.0f - poly * std::exp(-z * z);
  return x >= 0.0f ? 0.5f * (1.0f + erf) : 0.5f * (1.0f - erf);
}

float black(float F, float K, float sd, float df, bool isCall) {
  const float s = isCall ? 1.0f : -1.0f;
  if (sd <= 1e-7f || F <= 0.0f) return df * std::max(s * (F - K), 0.0f);
  const float d1 = std::log(F / K) / sd + 0.5f * sd;
  const float d2 = d1 - sd;
  return df * s * (F * ncdf(s * d1) - K * ncdf(s * d2));
}

}  // namespace

namespace {

struct PlanView {
  const FusedPlan& plan;
  std::uint32_t nP, nT, nA, nC, substeps, nNormals, seed, cpty, nCorr, offCall, offReporting, numWG, nS, nR, nTr;
  bool antithetic, hasCsa, hasIm;
  float quantileLevel, imLevel;

  explicit PlanView(const FusedPlan& p) : plan(p) {
    const auto& H = p.header;
    if (H.size() != kHeaderWords) throw std::invalid_argument("fused kernels: malformed plan header");
    nP = H[0], nT = H[1], nA = H[2], nC = H[3], substeps = H[4], nNormals = H[5], seed = H[6];
    antithetic = H[7] == 1u, hasCsa = H[8] == 1u;
    cpty = H[9], nCorr = H[11], offCall = H[12], offReporting = H[13], numWG = H[15];
    std::memcpy(&quantileLevel, &H[14], sizeof quantileLevel);
    nS = H[16], nR = H[17], nTr = H[18], hasIm = H[19] == 1u;
    std::memcpy(&imLevel, &H[20], sizeof imLevel);
    if (nP == 0 || nT == 0) throw std::invalid_argument("fused kernels: empty plan");
  }
};

// Per-trade statistic m of trade value vi in netting-set value v, as tradeStat in the kernel.
float tradeStat(std::uint32_t m, float vi, float v, float D) {
  const float pos = v > 0.0f ? 1.0f : 0.0f;
  switch (m) {
    case 0: return D * vi * pos;
    case 1: return vi * pos;
    case 2: return D * (std::max(v, 0.0f) - std::max(v - vi, 0.0f));
    case 3: return D * std::max(vi, 0.0f);
    default: return std::max(vi, 0.0f);
  }
}

// fused-exposure for one path (invocation p): the per-date statistics into `stats`
// ([nT][nS]) and quantile row r (positive exposure, then margin |dV|) into expo[r * stride].
// The AMC regression pass records the state of every path into these (see runAmcStates).
struct AmcSink {
  float* x;
  float* deflator;
  float* exercise;
  std::size_t nP;
  const std::vector<std::size_t>* offsets;  // first exercise row of each Bermudan
};

constexpr std::uint32_t kNotExercised = 0xFFFFFFFFu;

// The AMC regression z = (x - c) * A, B + p1 z + p2 z^2 + p3 z^3 (Horner), as amcPoly in the kernel.
float amcPoly(const float* rec, float x) {
  const float z = (x - rec[2]) * rec[3];
  return ((rec[7] * z + rec[6]) * z + rec[5]) * z + rec[4];
}

void simulatePath(const PlanView& v, std::uint32_t p, float* stats, float* expo, std::size_t stride, float* vals,
                  const AmcSink* sink = nullptr) {
  const auto& IDX = v.plan.indices;
  const auto& FP = v.plan.params;
  const auto& STEPS = v.plan.steps;
  const auto& TERMS = v.plan.terms;
  const std::uint32_t nT = v.nT, nA = v.nA, nC = v.nC, substeps = v.substeps, nCorr = v.nCorr;
  float sign = 1.0f;
  std::uint32_t pathStream = p;
  if (v.antithetic) {
    pathStream = p >> 1u;
    if ((p & 1u) == 1u) sign = -1.0f;
  }
  const std::uint32_t stream = pcg(pcg(v.seed) ^ pcg(pathStream + 0x9E3779B9u));
  float x = 0.0f, lnD = 0.0f, D = 1.0f;
  float S[kMaxAssets], lnS[kMaxAssets] = {}, y[kMaxCredits], Y[kMaxCredits], fix[kMaxSwapSlots] = {}, z[kMaxNormals] = {}, w[kCholeskyStride];
  for (std::uint32_t a = 0; a < kMaxAssets; ++a) S[a] = FP[12 + a];
  for (std::uint32_t c = 0; c < kMaxCredits; ++c) y[c] = FP[4 * c + 3], Y[c] = 0.0f;
  float held = 0.0f, prevDE = 0.0f, prevQ = 1.0f, prevRV = 0.0f;
  std::uint32_t exercised[kMaxBermudans];  // exercise index of each Bermudan on this path
  for (auto& e : exercised) e = kNotExercised;
  const float ia = FP[11], mta = FP[10];

  for (std::uint32_t j = 0; j < nT; ++j) {
    float Q = 1.0f;
    if (j > 0) {
      const std::size_t base = std::size_t(j) * kStepStride;
      const std::uint32_t nZ = 4 * ((v.nNormals + 3) / 4);
      for (std::uint32_t k = 0; k < nZ; k += 2) gauss2(stream, j, k / 2, &z[k]);
      for (std::uint32_t k = 0; k < nZ; ++k) z[k] *= sign;
      // Each padded Cholesky row as two 4-term dot products, as in the kernel.
      for (std::uint32_t i = 0; i < kCholeskyStride; ++i) {
        const float* row = &FP[16 + i * kCholeskyStride];
        const float d0 = row[0] * z[0] + row[1] * z[1] + row[2] * z[2] + row[3] * z[3];
        const float d1 = row[4] * z[4] + row[5] * z[5] + row[6] * z[6] + row[7] * z[7];
        w[i] = d0 + d1;
      }
      const float integratedRate = x * STEPS[base + 5] + STEPS[base + 2] * w[0] + STEPS[base + 3] * z[nCorr] + STEPS[base + 4];
      x = x * STEPS[base] + STEPS[base + 1] * w[0];
      lnD = lnD - integratedRate;
      D = std::exp(lnD);
      for (std::uint32_t a = 0; a < kMaxAssets; ++a) {
        lnS[a] = lnS[a] + (integratedRate - STEPS[base + 8 + a] - STEPS[base + 16 + a] + STEPS[base + 12 + a] * w[1 + a]);
        S[a] = FP[12 + a] * std::exp(lnS[a]);
      }
      const float dt = STEPS[base + 6];
      std::uint32_t off = nCorr + 1;
      for (std::uint32_t c = 0; c < nC; ++c) {
        const float kappa = FP[4 * c], theta = FP[4 * c + 1], xi = FP[4 * c + 2];
        const float h = dt / static_cast<float>(substeps);
        float rem = std::sqrt(dt) * w[1 + nA + c], remT = dt, yy = y[c], YY = Y[c];
        for (std::uint32_t s = 0; s < substeps; ++s) {
          float dW = rem;
          if (s + 1 < substeps) dW = h / remT * rem + std::sqrt(std::max(h * (remT - h) / remT, 0.0f)) * z[off + s];
          rem -= dW;
          remT -= h;
          const float yp = std::max(yy, 0.0f);
          const float yn = yy + kappa * (theta - yp) * h + xi * std::sqrt(yp) * dW;
          YY += 0.5f * (yp + std::max(yn, 0.0f)) * h;
          yy = yn;
        }
        y[c] = yy;
        Y[c] = YY;
        off += substeps - 1;
      }
      if (nC > 0) Q = std::exp(-Y[v.cpty] - STEPS[base + 20 + v.cpty]);
    }

    float value = 0.0f;
    float vt[kMaxTradeGroups] = {};
    float ev[kMaxBermudans] = {};  // exercise values on this date
    const std::uint32_t nTr = std::min<std::uint32_t>(v.nTr, kMaxTradeGroups);
    if (sink) sink->x[std::size_t(j) * sink->nP + p] = x, sink->deflator[std::size_t(j) * sink->nP + p] = D;
    for (std::uint32_t t = IDX[j]; t < IDX[j + 1]; ++t) {
      const std::size_t r = std::size_t(t) * kTermStride;
      const float* rec = &TERMS[r];
      const auto code = static_cast<std::uint32_t>(rec[0]);  // kind + 16 trade + 256 bermudan + 4096 extra
      const std::uint32_t kind = code & 15u, bm = std::min<std::uint32_t>((code >> 8u) & 15u, kMaxBermudans - 1),
                          extra = code >> 12u;
      const auto slot = static_cast<std::uint32_t>(rec[1]);
      const float c = rec[2];
      auto bond = [&] { return rec[3] * std::exp(-rec[4] * x); };
      float term = 0.0f;
      switch (kind) {
        case kBond: term = c * bond(); break;
        case kFixedFloat: term = (c * fix[slot] + rec[5]) * bond(); break;
        case kAsset: term = c * S[slot] * rec[3]; break;
        case kSetFixing: fix[slot] = (bond() - 1.0f) / c; break;
        case kOption: {
          const float b0 = bond(), p3 = rec[7];
          term = c * black(S[slot] * rec[5] / b0, rec[6], std::fabs(p3), b0, p3 > 0.0f);
          break;
        }
        case kBermExercise: ev[bm] += c * bond(); break;
        case kBermDecide:
          if (sink) {
            sink->exercise[((*sink->offsets)[bm] + extra) * sink->nP + p] = ev[bm];
          } else if (exercised[bm] == kNotExercised && ev[bm] > 0.0f && ev[bm] > amcPoly(rec, x)) {
            exercised[bm] = extra;
          }
          break;
        case kBermContinue:
          if (exercised[bm] == kNotExercised) term = rec[1] * std::max(amcPoly(rec, x), 0.0f);
          break;
        case kBermBond:
          if (exercised[bm] != kNotExercised && exercised[bm] <= extra) term = c * bond();
          break;
        default:  // kBermFloat
          if (exercised[bm] != kNotExercised && exercised[bm] <= extra) term = (c * fix[slot] + rec[5]) * bond();
      }
      if (v.nTr > 0) vt[std::min<std::uint32_t>((code >> 4u) & 15u, kMaxTradeGroups - 1)] += term;
      else value += term;
    }
    for (std::uint32_t i = 0; i < nTr; ++i) value += vt[i];

    // Collateral called on this path's value at t - MPR.
    float collateral = 0.0f;
    vals[j] = value;
    if (v.hasCsa) {
      const std::uint32_t callJ = IDX[v.offCall + j];
      const float vc = callJ != j ? vals[callJ] : value;
      const float hc = FP[8], hb = FP[9];
      float required = 0.0f;
      if (vc > hc) required += vc - hc;
      if (hb >= 0.0f && -vc > hb) required -= -vc - hb;
      if (j == 0 || std::fabs(required - held) >= mta) held = required;
      collateral = held + ia;
    }
    const float e = value - collateral, ep = std::max(e, 0.0f), en = std::min(e, 0.0f);
    const std::uint32_t rs = IDX[v.offReporting + j];
    if (rs > 0) {
      expo[std::size_t(rs - 1) * stride] = ep;
      if (v.hasIm) expo[std::size_t(v.nR + rs - 1) * stride] = rs == 1 ? 0.0f : std::fabs(value - prevRV);
    }

    float cvaInc = 0.0f;
    if (rs > 0) {
      prevRV = value;
      const float de = D * ep;
      if (j > 0) cvaInc = 0.5f * (prevDE + de) * (prevQ - Q);
      prevDE = de;
      prevQ = Q;
    }
    float* st = &stats[std::size_t(j) * v.nS];
    st[0] = e, st[1] = ep, st[2] = en, st[3] = D * ep, st[4] = D * en, st[5] = cvaInc, st[6] = Q, st[7] = D;
    for (std::uint32_t k = kStatsPerDate; k < v.nS; ++k) {
      const std::uint32_t i = (k - kStatsPerDate) / kTradeStats;
      st[k] = i < nTr ? tradeStat((k - kStatsPerDate) % kTradeStats, vt[i], value, D) : 0.0f;
    }
  }
}

}  // namespace

FusedSlice runFusedWorkgroups(const FusedPlan& plan, std::size_t wgBegin, std::size_t wgEnd, SpinPool& pool) {
  const PlanView v(plan);
  wgEnd = std::min<std::size_t>(wgEnd, v.numWG);
  if (wgBegin > wgEnd) throw std::invalid_argument("runFusedWorkgroups: empty or reversed workgroup range");
  FusedSlice slice;
  slice.wgBegin = wgBegin;
  slice.wgEnd = wgEnd;
  slice.pathBegin = std::min<std::size_t>(wgBegin * kWorkgroupSize, v.nP);
  slice.pathEnd = std::min<std::size_t>(wgEnd * kWorkgroupSize, v.nP);
  const std::size_t count = slice.pathEnd - slice.pathBegin, nT = v.nT, nQ = plan.quantileRows(), nS = v.nS;
  slice.partials.assign((wgEnd - wgBegin) * nT * nS, 0.0f);
  slice.exposure.assign(nQ * count, 0.0f);

  // Workgroups are independent (each writes its own partials and its own paths' exposures), so
  // the lock-free pool runs them in parallel and the result is the same bits on any number of
  // threads. Scratch buffers are per chunk of workgroups.
  float* partials = slice.partials.data();
  float* exposure = slice.exposure.data();
  pool.parallelFor(wgEnd - wgBegin, 1, [&](std::size_t cb, std::size_t ce) {
    std::vector<float> stats(kWorkgroupSize * nT * nS), vals(nT), red(kWorkgroupSize * nS);
    for (std::size_t g = wgBegin + cb; g < wgBegin + ce; ++g) {
      // The 64 invocations of the workgroup; inactive ones (past the last path) contribute zeros.
      std::fill(stats.begin(), stats.end(), 0.0f);
      for (std::uint32_t l = 0; l < kWorkgroupSize; ++l) {
        const std::size_t p = g * kWorkgroupSize + l;
        if (p >= v.nP) break;
        simulatePath(v, static_cast<std::uint32_t>(p), &stats[l * nT * nS], exposure + (p - slice.pathBegin), count, vals.data());
      }
      // Workgroup tree reduction, as in the kernel.
      for (std::size_t j = 0; j < nT; ++j) {
        for (std::uint32_t l = 0; l < kWorkgroupSize; ++l)
          for (std::size_t f = 0; f < nS; ++f) red[l * nS + f] = stats[(l * nT + j) * nS + f];
        for (std::uint32_t s = kWorkgroupSize / 2; s > 0; s >>= 1)
          for (std::uint32_t l = 0; l < s; ++l)
            for (std::size_t f = 0; f < nS; ++f) red[l * nS + f] += red[(l + s) * nS + f];
        for (std::size_t f = 0; f < nS; ++f) partials[((g - wgBegin) * nT + j) * nS + f] = red[f];
      }
    }
  });
  return slice;
}

FusedOutput finishFused(const FusedPlan& plan, const std::vector<FusedSlice>& slices, SpinPool& pool) {
  const PlanView v(plan);
  const std::size_t nT = v.nT, n = nT * v.nS, nQ = plan.quantileRows();
  std::size_t next = 0;
  for (const auto& s : slices) {
    if (s.wgBegin != next || s.partials.size() != (s.wgEnd - s.wgBegin) * n ||
        s.exposure.size() != nQ * (s.pathEnd - s.pathBegin))
      throw std::invalid_argument("finishFused: slices must cover the workgroups in order");
    next = s.wgEnd;
  }
  if (next != v.numWG) throw std::invalid_argument("finishFused: slices do not cover every workgroup");

  FusedOutput out;
  // reduce-partials: Kahan summation across workgroups, in workgroup order.
  // Every statistic is independent: the pool sums them in parallel, each in workgroup order.
  out.sums.resize(n);
  pool.parallelFor(n, 0, [&](std::size_t ib, std::size_t ie) {
    for (std::size_t i = ib; i < ie; ++i) {
      float sum = 0.0f, comp = 0.0f;
      for (const auto& s : slices)
        for (std::size_t g = 0; g < s.wgEnd - s.wgBegin; ++g) {
          const float yv = s.partials[g * n + i] - comp;
          const float tv = sum + yv;
          comp = (tv - sum) - yv;
          sum = tv;
        }
      out.sums[i] = sum / static_cast<float>(v.nP);
    }
  });

  // pfe-quantile, as the kernel: the order statistics k = floor(rank) and k + 1 from a coarse
  // and a fine histogram and the extremes of k's fine bin, interpolated as ccr::quantile.
  out.pfe.resize(nQ);
  const float bins = static_cast<float>(kPfeBins), big = std::numeric_limits<float>::max();
  const std::uint32_t last = kPfeBins - 1;
  auto forEach = [&](std::size_t r, auto&& f) {
    for (const auto& s : slices) {
      const std::size_t count = s.pathEnd - s.pathBegin;
      for (std::size_t k = 0; k < count; ++k) f(s.exposure[r * count + k]);
    }
  };
  // One quantile row per task, as one workgroup per row on the GPU.
  pool.parallelFor(nQ, 1, [&](std::size_t rb, std::size_t re) {
  std::vector<std::uint32_t> hist(kPfeBins);
  for (std::size_t r = rb; r < re; ++r) {
    const float level = r >= v.nR ? v.imLevel : v.quantileLevel;  // PFE rows, then margin rows
    const float rank = level * static_cast<float>(v.nP - 1);
    const std::uint32_t k = std::min(static_cast<std::uint32_t>(std::floor(rank)), v.nP - 1);
    const float frac = rank - static_cast<float>(k);
    float mx = 0.0f, mn = big;
    forEach(r, [&](float e) {
      mx = std::max(mx, e);
      if (e > 0.0f) mn = std::min(mn, e);
    });
    auto coarseBin = [&](float e) { return std::min(static_cast<std::uint32_t>(e / mx * bins), last); };
    std::fill(hist.begin(), hist.end(), 0u);
    std::uint32_t zeros = 0;
    forEach(r, [&](float e) {
      if (e <= 0.0f) ++zeros;
      else ++hist[coarseBin(e)];
    });
    if (k < zeros) {
      out.pfe[r] = k + 1 >= zeros && k + 1 < v.nP && frac > 0.0f ? frac * mn : 0.0f;
      continue;
    }
    std::uint32_t below = zeros, cb = 0;
    for (; cb < last; ++cb) {
      if (below + hist[cb] > k) break;
      below += hist[cb];
    }
    // Fine histogram inside coarse bin cb.
    std::fill(hist.begin(), hist.end(), 0u);
    const float width = mx / bins, lo = static_cast<float>(cb) * width;
    auto fineBin = [&](float e) { return std::min(static_cast<std::uint32_t>(std::clamp((e - lo) / width, 0.0f, 1.0f) * bins), last); };
    forEach(r, [&](float e) {
      if (e > 0.0f && coarseBin(e) == cb) ++hist[fineBin(e)];
    });
    std::uint32_t fb = 0, c = 0;
    for (; fb < kPfeBins; ++fb) {
      c = hist[fb];
      if (below + c > k) break;
      below += c;
    }
    fb = std::min(fb, last);
    // Extremes of k's fine bin and the next exposure above it.
    float bLo = big, bHi = 0.0f, above = big;
    forEach(r, [&](float e) {
      if (e <= 0.0f) return;
      const std::uint32_t eb = coarseBin(e), ef = eb == cb ? fineBin(e) : 0;
      if (eb == cb && ef == fb) bLo = std::min(bLo, e), bHi = std::max(bHi, e);
      else if (eb > cb || (eb == cb && ef > fb)) above = std::min(above, e);
    });
    const std::uint32_t i = k - below;
    const float spread = bHi - bLo;
    const float ek = c > 1 ? bLo + spread * static_cast<float>(i) / static_cast<float>(c - 1) : bHi;
    float result = ek;
    if (frac > 0.0f && k + 1 < v.nP) {
      const float ek1 = i + 1 < c ? bLo + spread * static_cast<float>(i + 1) / static_cast<float>(c - 1) : above;
      result = ek + frac * (ek1 - ek);
    }
    out.pfe[r] = result;
  }
  });
  return out;
}

AmcStates runAmcStates(const FusedPlan& plan, SpinPool& pool) {
  const PlanView v(plan);
  const std::size_t nP = v.nP, nT = v.nT, nS = v.nS, nQ = plan.quantileRows();
  std::vector<std::size_t> offsets;
  std::size_t rows = 0;
  for (const auto& a : plan.amc) offsets.push_back(rows), rows += a.exerciseDates.size();
  AmcStates out;
  out.x.assign(nT * nP, 0.0f);
  out.deflator.assign(nT * nP, 0.0f);
  out.exercise.assign(rows * nP, 0.0f);
  const AmcSink sink{out.x.data(), out.deflator.data(), out.exercise.data(), nP, &offsets};
  // Paths are independent: the pool runs them in parallel, each writing its own column.
  pool.parallelFor(nP, 0, [&](std::size_t pb, std::size_t pe) {
    std::vector<float> stats(nT * nS), expo(nQ), vals(nT);
    for (std::size_t p = pb; p < pe; ++p) simulatePath(v, static_cast<std::uint32_t>(p), stats.data(), expo.data(), 1, vals.data(), &sink);
  });
  return out;
}

FusedOutput runFusedReference(const FusedPlan& plan, SpinPool& pool) {
  return finishFused(plan, {runFusedWorkgroups(plan, 0, plan.numWorkgroups(), pool)}, pool);
}

}  // namespace ccr::gpu
