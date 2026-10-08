// CPU execution of the fused kernels (fused_kernels.cpp), statement for statement in
// single precision, so the GPU algorithm is unit-tested natively.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "ccr/gpu/fused_exposure.hpp"

namespace ccr::gpu {

namespace {

std::uint32_t pcg(std::uint32_t v) {
  const std::uint32_t state = v * 747796405u + 2891336453u;
  const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float gauss(std::uint32_t stream, std::uint32_t j, std::uint32_t k) {
  const std::uint32_t h1 = pcg(stream ^ pcg(j * 131u + k));
  const std::uint32_t h2 = pcg(h1 ^ 0x68E31DA4u);
  const float u1 = (static_cast<float>(h1 >> 8u) + 0.5f) / 16777216.0f;
  const float u2 = (static_cast<float>(h2 >> 8u) + 0.5f) / 16777216.0f;
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
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

FusedOutput runFusedReference(const FusedPlan& plan) {
  const auto& H = plan.header;
  const auto& IDX = plan.indices;
  const auto& FP = plan.params;
  const auto& STEPS = plan.steps;
  const auto& TERMS = plan.terms;
  const std::uint32_t nP = H[0], nT = H[1], nA = H[2], nC = H[3], substeps = H[4], nNormals = H[5], seed = H[6];
  const bool antithetic = H[7] == 1u, hasCsa = H[8] == 1u;
  const std::uint32_t cpty = H[9], stepStride = H[10], nCorr = H[11], offCall = H[12], offReporting = H[13];
  float quantileLevel;
  std::memcpy(&quantileLevel, &H[14], sizeof quantileLevel);
  const std::uint32_t numWG = H[15];
  if (nP == 0 || nT == 0) throw std::invalid_argument("runFusedReference: empty plan");

  std::vector<float> vals(std::size_t(nT) * nP), expo(std::size_t(nT) * nP);
  std::vector<float> partials(std::size_t(numWG) * nT * kStatsPerDate, 0.0f);
  std::vector<float> red(kWorkgroupSize * kStatsPerDate);

  // Paths run one after another; their per-date statistics are kept for the workgroup reduction.
  std::vector<float> stats(std::size_t(nP) * nT * kStatsPerDate, 0.0f);
  for (std::uint32_t p = 0; p < nP; ++p) {
    float sign = 1.0f;
    std::uint32_t pathStream = p;
    if (antithetic) {
      pathStream = p >> 1u;
      if ((p & 1u) == 1u) sign = -1.0f;
    }
    const std::uint32_t stream = pcg(pcg(seed) ^ pcg(pathStream + 0x9E3779B9u));
    float x = 0.0f, D = 1.0f;
    float S[kMaxAssets], y[kMaxCredits], Y[kMaxCredits], fix[kMaxSwapSlots] = {}, z[kMaxNormals], w[kMaxFactors];
    for (std::uint32_t a = 0; a < kMaxAssets; ++a) S[a] = FP[12 + a];
    for (std::uint32_t c = 0; c < kMaxCredits; ++c) y[c] = FP[4 * c + 3], Y[c] = 0.0f;
    float held = 0.0f, prevDE = 0.0f, prevQ = 1.0f;
    const float ia = FP[11], mta = FP[10];

    for (std::uint32_t j = 0; j < nT; ++j) {
      float Q = 1.0f;
      if (j > 0) {
        const std::size_t base = std::size_t(j) * stepStride;
        for (std::uint32_t k = 0; k < nNormals; ++k) z[k] = sign * gauss(stream, j, k);
        for (std::uint32_t i = 0; i < nCorr; ++i) {
          float acc = 0.0f;
          for (std::uint32_t k = 0; k <= i; ++k) acc += FP[16 + i * kMaxFactors + k] * z[k];
          w[i] = acc;
        }
        const float integratedRate = x * STEPS[base + 5] + STEPS[base + 2] * w[0] + STEPS[base + 3] * z[nCorr] + STEPS[base + 4];
        x = x * STEPS[base] + STEPS[base + 1] * w[0];
        D = D * std::exp(-integratedRate);
        for (std::uint32_t a = 0; a < nA; ++a) {
          const std::size_t o = base + 7 + 3 * a;
          S[a] = S[a] * std::exp(integratedRate - STEPS[o] - STEPS[o + 2] + STEPS[o + 1] * w[1 + a]);
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
        if (nC > 0) Q = std::exp(-Y[cpty] - STEPS[base + 7 + 3 * kMaxAssets + cpty]);
      }

      float v = 0.0f;
      for (std::uint32_t t = IDX[j]; t < IDX[j + 1]; ++t) {
        const std::size_t r = std::size_t(t) * kTermStride;
        const auto kind = static_cast<std::uint32_t>(TERMS[r]);
        const auto slot = static_cast<std::uint32_t>(TERMS[r + 1]);
        const float c = TERMS[r + 2];
        const float bond = TERMS[r + 3] * std::exp(-TERMS[r + 4] * x);
        switch (kind) {
          case kBond: v += c * bond; break;
          case kFixedFloat: v += c * fix[slot] * bond; break;
          case kAsset: v += c * S[slot] * TERMS[r + 3]; break;
          case kSetFixing: fix[slot] = (bond / (TERMS[r + 5] * std::exp(-TERMS[r + 6] * x)) - 1.0f) / c; break;
          default: {
            const float p3 = TERMS[r + 7];
            v += c * black(S[slot] * TERMS[r + 5] / bond, TERMS[r + 6], std::fabs(p3), bond, p3 > 0.0f);
          }
        }
      }

      float collateral = 0.0f;
      vals[std::size_t(j) * nP + p] = v;
      if (hasCsa) {
        const std::uint32_t callJ = IDX[offCall + j];
        const float vc = callJ != j ? vals[std::size_t(callJ) * nP + p] : v;
        const float hc = FP[8], hb = FP[9];
        float required = 0.0f;
        if (vc > hc) required += vc - hc;
        if (hb >= 0.0f && -vc > hb) required -= -vc - hb;
        if (j == 0 || std::fabs(required - held) >= mta) held = required;
        collateral = held + ia;
      }
      const float e = v - collateral, ep = std::max(e, 0.0f), en = std::min(e, 0.0f);
      expo[std::size_t(j) * nP + p] = ep;

      float cvaInc = 0.0f;
      if (IDX[offReporting + j] == 1u) {
        const float de = D * ep;
        if (j > 0) cvaInc = 0.5f * (prevDE + de) * (prevQ - Q);
        prevDE = de;
        prevQ = Q;
      }
      float* st = &stats[(std::size_t(p) * nT + j) * kStatsPerDate];
      st[0] = e, st[1] = ep, st[2] = en, st[3] = D * ep, st[4] = D * en, st[5] = cvaInc, st[6] = Q;
    }
  }

  // Workgroup tree reduction, as in the kernel (inactive threads contribute zeros).
  for (std::uint32_t g = 0; g < numWG; ++g)
    for (std::uint32_t j = 0; j < nT; ++j) {
      for (std::uint32_t l = 0; l < kWorkgroupSize; ++l) {
        const std::uint32_t p = g * kWorkgroupSize + l;
        for (std::size_t f = 0; f < kStatsPerDate; ++f)
          red[l * kStatsPerDate + f] = p < nP ? stats[(std::size_t(p) * nT + j) * kStatsPerDate + f] : 0.0f;
      }
      for (std::uint32_t s = kWorkgroupSize / 2; s > 0; s >>= 1)
        for (std::uint32_t l = 0; l < s; ++l)
          for (std::size_t f = 0; f < kStatsPerDate; ++f) red[l * kStatsPerDate + f] += red[(l + s) * kStatsPerDate + f];
      for (std::size_t f = 0; f < kStatsPerDate; ++f) partials[(std::size_t(g) * nT + j) * kStatsPerDate + f] = red[f];
    }

  FusedOutput out;
  // reduce-partials: Kahan summation across workgroups.
  const std::size_t n = std::size_t(nT) * kStatsPerDate;
  out.sums.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    float sum = 0.0f, comp = 0.0f;
    for (std::uint32_t g = 0; g < numWG; ++g) {
      const float yv = partials[g * n + i] - comp;
      const float tv = sum + yv;
      comp = (tv - sum) - yv;
      sum = tv;
    }
    out.sums[i] = sum / static_cast<float>(nP);
  }

  // pfe-quantile: histogram of the positive exposure on [0, max] per date.
  out.pfe.resize(nT);
  std::vector<std::uint32_t> hist(kPfeBins);
  for (std::uint32_t j = 0; j < nT; ++j) {
    const float* ex = &expo[std::size_t(j) * nP];
    float mx = 0.0f;
    for (std::uint32_t p = 0; p < nP; ++p) mx = std::max(mx, ex[p]);
    std::fill(hist.begin(), hist.end(), 0u);
    std::uint32_t zeros = 0;
    for (std::uint32_t p = 0; p < nP; ++p) {
      if (ex[p] <= 0.0f) ++zeros;
      else ++hist[std::min(static_cast<std::uint32_t>(ex[p] / mx * static_cast<float>(kPfeBins)), std::uint32_t(kPfeBins - 1))];
    }
    const float rank = quantileLevel * static_cast<float>(nP - 1);
    float cum = static_cast<float>(zeros), result = 0.0f;
    if (rank >= cum && mx > 0.0f) {
      result = mx;
      const float width = mx / static_cast<float>(kPfeBins);
      for (std::uint32_t b = 0; b < kPfeBins; ++b) {
        const float c = static_cast<float>(hist[b]);
        if (cum + c > rank) {
          result = (static_cast<float>(b) + (rank - cum + 0.5f) / c) * width;
          break;
        }
        cum += c;
      }
    }
    out.pfe[j] = result;
  }
  return out;
}

}  // namespace ccr::gpu
