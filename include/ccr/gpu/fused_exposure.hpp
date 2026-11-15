#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ccr/core/parallel.hpp"
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
/// carry[4], vol sqrt(dt)[4], vol^2 dt / 2[4], CIR++ shift[2] + collateral bond (A, B).
constexpr std::size_t kStepStride = 24;
/// CIR x2 | CSA | spots | Cholesky rows | collateral security (type, asset, haircut, -).
constexpr std::size_t kParamCount = 16 + kCholeskyStride * kCholeskyStride + 4;
constexpr std::size_t kTermStride = 8;
constexpr std::size_t kHeaderWords = 32;
constexpr std::size_t kPfeBins = 3072;       ///< histogram bins of the row-statistics kernel (12 KiB of workgroup storage)
constexpr std::size_t kPfeWorkgroupSize = 256;
constexpr std::size_t kMaxSamples = 64;      ///< sample paths of the path views
constexpr std::size_t kMaxInnerPaths = 1024; ///< inner paths of the nested initial margin
constexpr std::size_t kInnerTop = 16;        ///< largest inner samples kept for the quantile
constexpr std::size_t kRequestStride = 8;    ///< (op, row, row2, count) (level, lo, hi, out) per request, as f32
constexpr std::uint32_t kNoRow = 0xFFFFFFFFu;

/// Statistics of one row (a per-path quantity on one date, written by the fused kernel) that
/// the row-statistics kernel computes, one workgroup per request.
enum RowOp : int {
  kRowQuantile = 0,      ///< empirical quantile `level` (interpolated between order statistics)
  kRowMean = 1,          ///< E[x]
  kRowMeanPos = 2,       ///< E[max(x, 0)]
  kRowMeanNeg = 3,       ///< E[min(x, 0)]
  kRowSamples = 4,       ///< x on the first `count` paths
  kRowHistogram = 5,     ///< `count` bins on [lo, hi] (auto range when lo == hi), then lo and hi
  kRowMeanProduct = 6,   ///< E[x y], y = row2
  kRowQuantilePos = 7,   ///< quantile of max(x, 0)
  kRowMeanPosTimes = 8,  ///< E[max(x, 0) y]
  kRowMeanNegTimes = 9,  ///< E[min(x, 0) y]
};

/// The rows the fused kernel writes on the reporting dates: the first row of each block of one
/// row per reporting date (kNoRow when not requested). Risk-factor probes follow `total`.
struct RowLayout {
  std::uint32_t exposure = 0;               ///< max(V - C, 0): PFE
  std::uint32_t valueChange = kNoRow;       ///< |V(t_k) - V(t_{k-1})|: initial margin
  std::uint32_t exposureValue = kNoRow;     ///< V - C
  std::uint32_t netted = kNoRow;            ///< V (uncollateralised)
  std::uint32_t collateral = kNoRow;        ///< C
  std::uint32_t gross = kNoRow;             ///< sum_i max(V_i, 0)
  std::uint32_t trades = kNoRow;            ///< V_i, trade-major
  std::uint32_t exercise = kNoRow;          ///< one row per Bermudan: exercise index, -1 if never exercised
  std::uint32_t conditionalIm = kNoRow;     ///< IM(t, path) of the nested initial margin
  std::uint32_t total = 0;                  ///< rows of the exposure buffer, probes included
};

/// Output offsets of a value row's exposure profile (each a block of one output per reporting date).
struct ProfileOutputs {
  std::size_t mean = 0, positive = 0, negative = 0, pfe = 0, positiveDiscounted = 0, negativeDiscounted = 0;
};

/// Where summarise() finds each view in the outputs of the row-statistics kernel.
struct ViewOutputs {
  static constexpr std::size_t none = static_cast<std::size_t>(-1);
  std::size_t pfe = 0, valueChange = none;
  std::size_t conditionalIm = none, conditionalImMean = none;  ///< E[D IM], E[IM] per reporting date
  std::vector<std::size_t> conditionalImBands;                 ///< 5 / 50 / 95% of IM per reporting date
  std::size_t samples = 0;                                   ///< paths per sample block
  std::size_t exposureSamples = none, nettedSamples = none, collateralSamples = none;  ///< [date][path]
  std::vector<std::size_t> collateralBands;                  ///< per level: [date]
  bool gross = false;
  ProfileOutputs grossProfile;
  std::size_t histogram = none, histogramBins = 0, histogramRank = 0;
  struct Trade {
    ProfileOutputs profile;
    std::size_t samples = none, meanDeflated = none;
    std::vector<std::size_t> bands;
  };
  std::vector<Trade> trades;
  std::vector<std::size_t> exerciseHistogram, exerciseSamples;  ///< per Bermudan
  // Risk factors (factorViews): rows of the state, the deflator and the martingale tests.
  bool factors = false;
  std::size_t rateSamples = none, rateMean = none, deflatorMean = none, bondMean = none;
  std::vector<std::size_t> rateBands;
  struct Asset { std::size_t samples = none, mean = none, deflatedMean = none; std::vector<std::size_t> bands; };
  std::vector<Asset> assets;
  struct Credit { std::size_t intensitySamples = none, survivalSamples = none, survivalMean = none; std::vector<std::size_t> intensityBands; };
  std::vector<Credit> credits;
  std::size_t increments = 0, incrementMeans = none, incrementProducts = none;  ///< first-step factor increments
};

constexpr std::size_t kMaxBermudans = 4;     ///< Bermudan swaptions priced by AMC inside the kernel
constexpr std::size_t kMaxExercises = 64;    ///< exercise dates per Bermudan
constexpr std::size_t kMaxAmcDegree = 3;     ///< polynomial degree of the AMC regressions

/// The first word of a term packs kind + 16 * trade + 256 * bermudan + 4096 * extra (small
/// integers, exact in f32): trade is the trade group in per-trade mode, bermudan and extra
/// (an exercise index) address the AMC terms.
///
/// Valuation term kinds in the term table, as (kind, slot, c, A) (B, p1, p2, p3):
///  kBond        c A e^{-B x}                          (bonds of one maturity merged, c A pre-multiplied)
///  kFixedFloat  (c fix[slot] + p1) A e^{-B x}         (running floating coupon with its fixed leg folded in)
///  kAsset       c S[slot] A                            (forwards on one asset merged)
///  kSetFixing   fix[slot] = (A e^{-B x} - 1) / c       (A, B: the ratio P(t, ts) / P(t, te))
///  kOption      c Black(S[slot] p1 / bond, p2, |p3|)   (call when p3 > 0)
/// American Monte Carlo (Bermudan b, Longstaff-Schwartz in the kernel; z = (x - c) * A, poly = B + p1 z + p2 z^2 + p3 z^3):
///  kBermExercise  ev[b] += c A e^{-B x}                  (the tail swap entered on exercise, at an exercise date)
///  kBermDecide    exercise[b] = extra if not yet exercised, ev[b] > 0 and ev[b] > poly  (policy of exercise `extra`)
///  kBermContinue  slot * max(poly, 0) if not exercised   (continuation value, slot = position)
///  kBermBond      c A e^{-B x}              if exercised at an index <= extra (physical settlement)
///  kBermFloat     (c fix[slot] + p1) A e^{-B x}  if exercised at an index <= extra
enum TermKind : int {
  kBond = 0, kFixedFloat = 1, kAsset = 2, kSetFixing = 3, kOption = 4,
  kBermExercise = 5, kBermDecide = 6, kBermContinue = 7, kBermBond = 8, kBermFloat = 9,
  kProbe = 10,  ///< writes a risk factor into a row: slot = what + 16 index, c = offset, p1 = row (see kernel)
};

/// Where the AMC regressions of one Bermudan live in the term table (filled by compile).
struct AmcTerms {
  std::size_t degree = 2;
  std::vector<std::size_t> exerciseDates;   ///< grid index of each exercise date
  std::vector<std::size_t> decideTerm;      ///< term index of each exercise's policy
  std::vector<std::size_t> continueTerm;    ///< term index of the continuation on each grid date (npos: none)
  std::vector<std::size_t> nextExercise;    ///< per grid date: first exercise strictly after it (as BermudanSwaption)
};

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
  bool securitiesCollateral = false;  ///< the CSA collateral is a bond or an asset, not cash
  double pfeQuantile = 0.95;
  /// Quantile of |V(t_k) - V(t_{k-1})| between reporting dates (initial margin proxy); 0 if off.
  double imQuantile = 0.0;
  /// Trades with their own statistics (per-trade mode), 0 if off; netting-set order.
  std::size_t numTradeGroups = 0;
  std::size_t statsPerDate = kStatsPerDate;
  std::size_t imInnerPaths = 0;         ///< nested initial margin (0: off)
  RowLayout rows;
  std::vector<float> requests;          ///< row-statistics requests (kRequestStride floats each)
  std::size_t numOutputs = 0;           ///< floats the row-statistics kernel writes
  ViewOutputs views;

  // GPU buffers, in the exact binary layout of the WGSL kernels.
  std::vector<std::uint32_t> header;    ///< uniform block (kHeaderWords words; the quantile as f32 bits)
  std::vector<std::uint32_t> indices;   ///< termStart[nT+1] | callIndex[nT] | reportingSlot[nT] (1 + rank, 0 if not reported) | imTarget[nT] (date of t + MPR)
  std::vector<float> params;            ///< CIR parameters x2 | CSA terms | spots | Cholesky rows (kCholeskyStride)
  std::vector<float> steps;             ///< per-date transition coefficients (kStepStride per date, vec4-aligned)
  std::vector<float> terms;             ///< valuation terms (kTermStride per term)

  /// Market survival of the counterparty on the simulation dates (1 without credits).
  std::vector<double> marketSurvival;
  /// Trades the kernel cannot price; empty if all are supported.
  std::vector<std::string> unsupported;
  /// Bermudan swaptions: their regression terms, fitted by compile() on a CPU pass of the kernel.
  std::vector<AmcTerms> amc;

  std::size_t numDates() const { return times.size(); }
  std::size_t numTerms() const { return terms.size() / kTermStride; }
  std::size_t numWorkgroups() const { return (numPaths + kWorkgroupSize - 1) / kWorkgroupSize; }
  /// Workgroups to dispatch for the fused, reduce and PFE kernels.
  std::size_t fusedDispatch() const { return numWorkgroups(); }
  std::size_t reduceDispatch() const { return (numDates() * statsPerDate + 63) / 64; }
  std::size_t numRequests() const { return requests.size() / kRequestStride; }
  std::size_t pfeDispatch() const { return numRequests(); }
  bool hasInitialMargin() const { return imQuantile > 0.0; }
  bool perTrade() const { return numTradeGroups > 0; }
  /// Buffer sizes in bytes. The netted values are kept only for the margin-call look-back
  /// (with a CSA), and the positive exposure only on the reporting dates (for the PFE).
  /// Netted values (and, for securities collateral, the unit price) for the margin-call look-back.
  std::size_t valsBytes() const { return hasCsa ? numPaths * numDates() * 4 * (securitiesCollateral ? 2 : 1) : 16; }
  std::size_t expoBytes() const { return std::max<std::size_t>(16, numPaths * rows.total * 4); }
  std::size_t cubeBytes() const { return std::max(valsBytes(), expoBytes()); }  ///< largest storage binding
  std::size_t partialsBytes() const { return numWorkgroups() * numDates() * statsPerDate * 4; }
  std::size_t sumsBytes() const { return numDates() * statsPerDate * 4; }
  std::size_t pfeBytes() const { return std::max<std::size_t>(4, numOutputs) * 4; }
};

/// What the GPU returns: the per-date means of the statsPerDate statistics and the quantiles.
struct FusedOutput {
  std::vector<float> sums;  ///< [date][stat], already divided by the number of paths
  std::vector<float> values;  ///< outputs of the row-statistics requests (see ViewOutputs)
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
  /// Nested conditional initial margin: E[D IM(t)] (for MVA), E[IM(t)] and its 5 / 50 / 95% bands.
  std::vector<double> conditionalImDiscounted, conditionalImMean;
  std::vector<std::vector<double>> conditionalImBands;

  /// Per-trade statistics of trade i with value V_i in the netting set V (uncollateralised):
  struct TradeStats {
    std::vector<double> marginalDiscounted;     ///< E[D V_i 1{V > 0}]: Euler contribution to EE*
    std::vector<double> marginal;               ///< E[V_i 1{V > 0}]: Euler contribution to EE
    std::vector<double> incrementalDiscounted;  ///< E[D (V+ - (V - V_i)+)]: EE* added by the trade
    std::vector<double> standaloneDiscounted;   ///< E[D V_i+]
    std::vector<double> standalone;             ///< E[V_i+]
  };
  std::vector<TradeStats> trades;  ///< per-trade mode only

  // Path views (when requested in FusedOptions), on the reporting dates:
  std::vector<std::vector<double>> exposureSamples, nettedSamples, collateralSamples;  ///< [path][date]
  std::vector<std::vector<double>> collateralBands;                                   ///< [level][date]
  std::optional<ExposureProfile> grossProfile;
  double histogramTime = 0.0;
  std::vector<double> histogramEdges, histogramCounts;  ///< bins + 1 edges, bins counts (fractions of paths)
  struct TradeView {
    ExposureProfile profile;
    std::vector<std::vector<double>> samples;  ///< [path][date]
    std::vector<std::vector<double>> bands;    ///< [level][date]
    std::vector<double> meanDeflated;          ///< E[D V_i]
  };
  std::vector<TradeView> tradeViews;
  struct ExerciseView {
    std::vector<double> distribution;  ///< share of paths exercising at each date, then never
    std::vector<double> sampleIndex;   ///< exercise index of the sample paths, -1 if never
  };
  std::vector<ExerciseView> exercise;
  struct FactorView {
    std::vector<std::vector<double>> rateSamples, rateBands;  ///< Hull-White state x(t) (add phi(t) for r)
    std::vector<double> rateMean, deflatorMean, bondMean;     ///< E[x], E[D], E[D P(t, T)]
    struct Asset { std::vector<std::vector<double>> samples, bands; std::vector<double> mean, deflatedMean; };
    std::vector<Asset> assets;
    struct Credit { std::vector<std::vector<double>> intensitySamples, intensityBands, survivalSamples; std::vector<double> survivalMean; };
    std::vector<Credit> credits;  ///< intensity rows hold the CIR state y (add psi(t) for lambda)
    Matrix incrementCorrelation;  ///< realised correlation of the first-step increments (x, ln S, y)
  };
  std::optional<FactorView> factors;

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
  /// > 0: conditional initial margin by nested Monte Carlo: from every path on every reporting
  /// date, this many inner paths (antithetic pairs) over imMarginPeriod revalue the netting set,
  /// and IM(t, path) is the imQuantile of |V(t + MPR) - V(t)| over them (at most kMaxInnerPaths).
  std::size_t imInnerPaths = 0;
  double imMarginPeriod = 10.0 / 250.0;
  // Path views (each adds rows to the exposure buffer and requests to the row-statistics kernel):
  std::size_t samples = 0;           ///< sample paths of V - C, V and C (at most kMaxSamples)
  double histogramTime = -1.0;       ///< >= 0: histogram of V - C on the reporting date on or before it
  std::size_t histogramBins = 40;
  bool gross = false;                ///< gross exposure sum_i max(V_i, 0) (netting benefit); implies perTrade
  bool collateralBands = false;      ///< bandLevels quantiles of the collateral held
  bool tradeViews = false;           ///< per-trade value samples, bands, profiles; implies perTrade
  bool exerciseViews = false;        ///< exercise distribution and sample exercise of each Bermudan
  bool factorViews = false;          ///< risk-factor samples, bands, martingale tests, realised correlation
  double bondMaturity = 0.0;         ///< factorViews: maturity of the deflated-bond martingale test
  std::vector<double> bandLevels{0.05, 0.5, 0.95};
  std::vector<double> factorBandLevels{0.05, 0.25, 0.5, 0.75, 0.95};
};

/// Compiles a netting set for the fused kernels. Swaps, asset forwards, European options and
/// Bermudan swaptions are supported; other trades are listed in FusedPlan::unsupported and left
/// out. Bermudans are priced by American Monte Carlo in two passes of the same kernel: compile
/// runs it once on the CPU to record the Hull-White state, the deflator and the exercise values,
/// fits the Longstaff-Schwartz regressions in double precision, and writes their coefficients
/// into the plan's policy and continuation terms; the kernel then exercises path by path.
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
  std::vector<float> exposure;             ///< [row][path - pathBegin]
};

/// Runs workgroups [wgBegin, wgEnd) of the fused kernel on the CPU in single precision. The
/// workgroups run in parallel on `pool` (lock-free; see SpinPool), like workgroups across GPU
/// cores, and the result is bit-identical on any number of threads.
FusedSlice runFusedWorkgroups(const FusedPlan& plan, std::size_t wgBegin, std::size_t wgEnd, SpinPool& pool = defaultPool());

/// The reduce-partials and pfe-quantile kernels over slices that cover every workgroup in
/// order, statistics and quantile rows in parallel. The result is the same however the
/// workgroups were split and on any number of threads.
FusedOutput finishFused(const FusedPlan& plan, const std::vector<FusedSlice>& slices, SpinPool& pool = defaultPool());

/// The AMC regression pass: per path the Hull-White state and the deflator on every date
/// ([date][path]) and the exercise value of every Bermudan exercise ([bermudan][exercise][path]).
struct AmcStates {
  std::vector<float> x, deflator, exercise;
};
AmcStates runAmcStates(const FusedPlan& plan, SpinPool& pool = defaultPool());

/// The three kernels executed on the CPU in single precision, as the GPU runs them: the CPU
/// backend of the fused pipeline, parallel on every core of `pool`.
FusedOutput runFusedReference(const FusedPlan& plan, SpinPool& pool = defaultPool());

/// WGSL sources of the kernels. The per-trade variant of the fused kernel (for plans with
/// perTrade()) carries the per-trade value array; the default one does not pay for it.
const std::string& fusedExposureKernel(bool perTrade = false);
const std::string& reducePartialsKernel();
const std::string& pfeQuantileKernel();

}  // namespace ccr::gpu
