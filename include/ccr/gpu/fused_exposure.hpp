#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ccr/exposure/exposure_engine.hpp"
#include "ccr/exposure/exposure_profile.hpp"
#include "ccr/models/scenario_generator.hpp"

namespace ccr::gpu {

/// The fused exposure pipeline for GPUs (WebGPU / WGSL).
///
/// One kernel invocation per Monte Carlo path simulates the risk factors, revalues the
/// whole netting set, applies the CSA and accumulates the exposure statistics, without
/// ever storing a scenario cube. Everything a GPU needs is compiled here, on the CPU,
/// into flat tables (FusedPlan): exact Hull-White transition coefficients, bond terms
/// A(t, T) e^{-B(t, T) x} per date, fixing records, option terms, the margin-call
/// look-back, the Cholesky factor and the CIR++ shift. The host only uploads the
/// buffers, dispatches the three kernels and reads back two arrays (FusedOutput);
/// summarise() turns them into an ExposureProfile.
///
/// The fused kernel is vectorised and tiled: normals come in Box-Muller pairs packed into
/// vec4s, the Cholesky product is two vec4 dot products per row, the four assets step as
/// one vec4, and the Cholesky factor, each date's step row and the valuation-term table are
/// staged in workgroup memory (the term table in tiles of 64 terms) so that the 64 paths of a
/// workgroup share one load. runFusedReference() executes the same kernels on the CPU in
/// 32-bit floats, so the algorithm is tested natively without a GPU.

constexpr std::size_t kWorkgroupSize = 64;   ///< threads per workgroup of the fused kernel
/// Statistics per date: E[V - C], EE, ENE, EE*, ENE*, pathwise CVA increment, E[Q], E[D]; then,
/// in per-trade mode, kTradeStats per trade (see FusedResult::TradeStats), padded to a multiple of 4.
constexpr std::size_t kStatsPerDate = 8;
constexpr std::size_t kTradeStats = 5;
constexpr std::size_t kMaxTradeGroups = 16;  ///< trades with their own statistics in per-trade mode
constexpr std::size_t kMaxAssets = 4;
constexpr std::size_t kMaxCredits = 2;
constexpr std::size_t kMaxFactors = 1 + kMaxAssets + kMaxCredits;
constexpr std::size_t kMaxSwapSlots = 32;    ///< swaps with a running fixing
constexpr std::size_t kMaxNormals = 64;      ///< normals drawn per path and step
constexpr std::size_t kCholeskyStride = 8;  ///< padded row length of the Cholesky factor (two vec4s)
/// Floats per date in the step table, as six vec4s: HW (e, sd x, c1, c2), HW (int phi, (1-e)/a, dt, 0),
/// carry[4], vol sqrt(dt)[4], vol^2 dt / 2[4], CIR++ shift[2] + padding.
constexpr std::size_t kStepStride = 24;
constexpr std::size_t kParamCount = 16 + kCholeskyStride * kCholeskyStride;
constexpr std::size_t kTermStride = 8;
constexpr std::size_t kHeaderWords = 24;
constexpr std::size_t kPfeBins = 3072;       ///< histogram bins of the PFE kernel (12 KiB of workgroup storage)
constexpr std::size_t kPfeWorkgroupSize = 256;

/// Valuation term kinds in the term table, as (kind, slot, c, A) (B, p1, p2, p3):
///  kBond        c A e^{-B x}                          (bonds of one maturity merged, c A pre-multiplied)
///  kFixedFloat  (c fix[slot] + p1) A e^{-B x}         (running floating coupon with its fixed leg folded in)
///  kAsset       c S[slot] A                            (forwards on one asset merged)
///  kSetFixing   fix[slot] = (A e^{-B x} - 1) / c       (A, B: the ratio P(t, ts) / P(t, te))
///  kOption      c Black(S[slot] p1 / bond, p2, |p3|)   (call when p3 > 0)
enum TermKind : int { kBond = 0, kFixedFloat = 1, kAsset = 2, kSetFixing = 3, kOption = 4 };

struct FusedPlan {
  // Problem size.
  std::size_t numPaths = 0;
  std::vector<double> times;            ///< simulation dates (event and margin-call dates merged in)
  std::vector<std::size_t> reporting;   ///< indices of the reporting dates in `times`
  std::size_t numAssets = 0, numCredits = 0, counterparty = 0;
  std::size_t numCorrelated = 0, numNormals = 0, substeps = 1;
  std::uint32_t seed = 0;
  bool antithetic = true;
  bool hasCsa = false;
  double pfeQuantile = 0.95;
  /// Quantile of |V(t_k) - V(t_{k-1})| between reporting dates (initial margin proxy); 0 if off.
  double imQuantile = 0.0;
  /// Trades with their own statistics (per-trade mode), 0 if off; netting-set order.
  std::size_t numTradeGroups = 0;
  std::size_t statsPerDate = kStatsPerDate;

  // GPU buffers, in the exact binary layout of the WGSL kernels.
  std::vector<std::uint32_t> header;    ///< uniform block (kHeaderWords words; the quantile as f32 bits)
  std::vector<std::uint32_t> indices;   ///< termStart[nT+1] | callIndex[nT] | reportingSlot[nT] (1 + rank, 0 if not reported)
  std::vector<float> params;            ///< CIR parameters x2 | CSA terms | spots | Cholesky rows (kCholeskyStride)
  std::vector<float> steps;             ///< per-date transition coefficients (kStepStride per date, vec4-aligned)
  std::vector<float> terms;             ///< valuation terms (kTermStride per term)

  /// Market survival of the counterparty on the simulation dates (1 without credits).
  std::vector<double> marketSurvival;
  /// Trades the kernel cannot price (Bermudans need AMC regression); empty if all are supported.
  std::vector<std::string> unsupported;

  std::size_t numDates() const { return times.size(); }
  std::size_t numTerms() const { return terms.size() / kTermStride; }
  std::size_t numWorkgroups() const { return (numPaths + kWorkgroupSize - 1) / kWorkgroupSize; }
  /// Workgroups to dispatch for the fused, reduce and PFE kernels.
  std::size_t fusedDispatch() const { return numWorkgroups(); }
  std::size_t reduceDispatch() const { return (numDates() * statsPerDate + 63) / 64; }
  std::size_t pfeDispatch() const { return quantileRows(); }
  bool hasInitialMargin() const { return imQuantile > 0.0; }
  bool perTrade() const { return numTradeGroups > 0; }
  /// Rows of the exposure buffer read by the quantile kernel: the positive exposure on the
  /// reporting dates, then (initial margin) |V(t_k) - V(t_{k-1})| on the reporting dates.
  std::size_t quantileRows() const { return reporting.size() * (hasInitialMargin() ? 2 : 1); }
  /// Buffer sizes in bytes. The netted values are kept only for the margin-call look-back
  /// (with a CSA), and the positive exposure only on the reporting dates (for the PFE).
  std::size_t valsBytes() const { return hasCsa ? numPaths * numDates() * 4 : 16; }
  std::size_t expoBytes() const { return std::max<std::size_t>(16, numPaths * quantileRows() * 4); }
  std::size_t cubeBytes() const { return std::max(valsBytes(), expoBytes()); }  ///< largest storage binding
  std::size_t partialsBytes() const { return numWorkgroups() * numDates() * statsPerDate * 4; }
  std::size_t sumsBytes() const { return numDates() * statsPerDate * 4; }
  std::size_t pfeBytes() const { return quantileRows() * 4; }
};

/// What the GPU returns: the per-date means of the statsPerDate statistics and the quantiles.
struct FusedOutput {
  std::vector<float> sums;  ///< [date][stat], already divided by the number of paths
  std::vector<float> pfe;   ///< [quantile row]: PFE per reporting date, then the margin quantile
};

/// Exposure statistics on the reporting dates.
struct FusedResult {
  ExposureProfile profile;
  std::vector<double> meanSurvival;    ///< E[Q(t)] of the simulated CIR++ intensity
  std::vector<double> marketSurvival;  ///< market Q(t) of the counterparty
  /// E[1/2 (D V+_{k-1} + D V+_k) (Q_{k-1} - Q_k)]: the pathwise CVA of period k before the LGD factor.
  std::vector<double> cvaIncrements;
  /// E[D V+ | default in period k]: the increment divided by the simulated default probability.
  std::vector<double> conditionalDiscountedEe;
  /// E[D(t)]: the mean deflator.
  std::vector<double> meanDeflator;
  /// Quantile of |V(t_k) - V(t_{k-1})| of the uncollateralised netted value (0 at k = 0); empty if off.
  std::vector<double> valueChangeQuantile;

  /// Per-trade statistics of trade i with value V_i in the netting set V (uncollateralised):
  struct TradeStats {
    std::vector<double> marginalDiscounted;     ///< E[D V_i 1{V > 0}]: Euler contribution to EE*
    std::vector<double> marginal;               ///< E[V_i 1{V > 0}]: Euler contribution to EE
    std::vector<double> incrementalDiscounted;  ///< E[D (V+ - (V - V_i)+)]: EE* added by the trade
    std::vector<double> standaloneDiscounted;   ///< E[D V_i+]
    std::vector<double> standalone;             ///< E[V_i+]
  };
  std::vector<TradeStats> trades;  ///< per-trade mode only

  /// Pathwise CVA (stochastic intensity) for recovery `recovery`; NaN without credits.
  double pathwiseCva(double recovery) const;
};

struct FusedOptions {
  double pfeQuantile = 0.95;
  std::size_t counterparty = 0;
  /// > 0: also the quantile of the value change between reporting dates (initial margin).
  double imQuantile = 0.0;
  /// Statistics per trade (allocation); at most kMaxTradeGroups trades.
  bool perTrade = false;
};

/// Compiles a netting set for the fused kernels. Swaps, asset forwards and European
/// options are supported; other trades are listed in FusedPlan::unsupported and left out.
/// Throws std::invalid_argument when the models exceed the kernel's fixed limits.
FusedPlan compile(const ExposureEngine& engine, const NettingSet& nettingSet, const FusedOptions& options);
FusedPlan compile(const ExposureEngine& engine, const NettingSet& nettingSet, double pfeQuantile = 0.95,
                  std::size_t counterparty = 0);

/// Why `compile` cannot run this setup on the GPU, or an empty string if it can.
std::string limitation(const ExposureEngine& engine, const NettingSet& nettingSet, bool perTrade = false);

/// Turns the GPU read-back into an exposure profile on the reporting dates.
FusedResult summarise(const FusedPlan& plan, const FusedOutput& output);

/// The fused-exposure kernel for workgroups [wgBegin, wgEnd) on the CPU: their partial
/// statistics and the positive exposure of their paths ([date][path in the slice]).
struct FusedSlice {
  std::size_t wgBegin = 0, wgEnd = 0;      ///< workgroups covered
  std::size_t pathBegin = 0, pathEnd = 0;  ///< paths covered
  std::vector<float> partials;             ///< [workgroup][date][stat]
  std::vector<float> exposure;             ///< [reporting date][path - pathBegin]
};

/// Runs workgroups [wgBegin, wgEnd) of the fused kernel on the CPU in single precision.
FusedSlice runFusedWorkgroups(const FusedPlan& plan, std::size_t wgBegin, std::size_t wgEnd);

/// The reduce-partials and pfe-quantile kernels over slices that cover every workgroup in
/// order. The result is the same however the workgroups were split.
FusedOutput finishFused(const FusedPlan& plan, const std::vector<FusedSlice>& slices);

/// The three kernels executed on the CPU in single precision, as the GPU runs them.
FusedOutput runFusedReference(const FusedPlan& plan);

/// WGSL sources of the kernels. The per-trade variant of the fused kernel (for plans with
/// perTrade()) carries the per-trade value array; the default one does not pay for it.
const std::string& fusedExposureKernel(bool perTrade = false);
const std::string& reducePartialsKernel();
const std::string& pfeQuantileKernel();

}  // namespace ccr::gpu
