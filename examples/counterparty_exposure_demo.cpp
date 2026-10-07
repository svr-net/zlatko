// End-to-end walk-through of the counterparty credit exposure workflow:
//
//   1. market data and risk-factor models (Hull-White rates, FX, CIR++ credit)
//   2. Monte Carlo scenario generation and revaluation of a netting set
//   3. exposure profiles (EE, PFE, EPE, effective EPE), netting and collateral
//   4. American Monte Carlo exposure of a Bermudan swaption
//   5. CVA / DVA, CVA as a running spread, marginal CVA allocation
//   6. wrong-way risk with a stochastic intensity correlated to the exposure
//   7. CVA hedging: CDS credit hedge and FX delta with common random numbers

#include <cmath>
#include <cstdio>
#include <memory>

#include "ccr/ccr.hpp"

using namespace ccr;

namespace {

void printProfile(const char* title, const ExposureProfile& p) {
  std::printf("\n%s\n%8s %14s %14s %14s %14s\n", title, "t", "E[V]", "EE", "ENE", "PFE95");
  for (double target : {0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0}) {
    for (std::size_t j = 0; j < p.times.size(); ++j) {
      if (std::abs(p.times[j] - target) < 1e-9) {
        std::printf("%8.2f %14.0f %14.0f %14.0f %14.0f\n", p.times[j], p.expectedValue[j], p.expectedExposure[j],
                    p.expectedNegativeExposure[j], p.potentialFutureExposure[j]);
      }
    }
  }
  std::printf("EPE(1y) = %.0f   effective EPE(1y) = %.0f   peak PFE = %.0f\n", p.expectedPositiveExposure(1.0),
              p.effectiveExpectedPositiveExposure(1.0), p.maxPotentialFutureExposure());
}

}  // namespace

int main() {
  // ---------------------------------------------------------------- 1. market and models
  const auto usd = std::make_shared<YieldCurve>(std::vector<double>{0.5, 1.0, 2.0, 5.0, 10.0},
                                                std::vector<double>{0.030, 0.031, 0.033, 0.035, 0.037});
  const auto eur = std::make_shared<YieldCurve>(std::vector<double>{1.0, 10.0}, std::vector<double>{0.020, 0.025});

  const double recovery = 0.4;
  const std::vector<CdsQuote> cptyQuotes = {{1.0, 0.0070}, {3.0, 0.0095}, {5.0, 0.0120}, {7.0, 0.0130}, {10.0, 0.0140}};
  const auto cptyCurve = std::make_shared<CreditCurve>(bootstrapCreditCurve(*usd, cptyQuotes, recovery));
  const CreditCurve ownCurve = bootstrapCreditCurve(*usd, {{5.0, 0.0060}}, recovery);

  const auto hw = std::make_shared<HullWhite1F>(usd, 0.04, 0.009);
  const auto eurusd = std::make_shared<LognormalAsset>("EURUSD", 1.10, 0.10, eur);
  const auto cptyIntensity = std::make_shared<CirIntensity>("CPTY", cptyCurve, 0.4, 0.02, 0.10, 0.012);

  // Correlations [rates, EURUSD, CPTY intensity].
  Matrix corr = Matrix::identity(3);
  corr(0, 1) = corr(1, 0) = 0.2;
  const ScenarioGenerator generator(hw, {eurusd}, {cptyIntensity}, corr);
  const SimulationConfig config{5000, 2024, true};
  const ExposureEngine engine(generator, config, TimeGrid::standardExposureGrid(10.0));

  // ---------------------------------------------------------------- 2. netting set
  NettingSet ns;
  ns.id = "Counterparty A";
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("IRS 10y payer", 10e6, 0.0355, 0.0, 10.0, 2, InterestRateSwap::Direction::PayFixed)));
  ns.trades.push_back(std::make_shared<InterestRateSwap>(
      InterestRateSwap::vanilla("IRS 5y receiver", 6e6, 0.0340, 0.0, 5.0, 2, InterestRateSwap::Direction::ReceiveFixed)));
  ns.trades.push_back(std::make_shared<AssetForward>("EURUSD fwd 3y", "EURUSD", 5e6, 1.12, 3.0));
  ns.trades.push_back(std::make_shared<EuropeanOption>("EURUSD call 2y (short)", "EURUSD", OptionType::Call, -4e6,
                                                       1.15, 2.0));

  const ExposureResult uncollateralised = engine.run(ns);
  std::printf("Netting set '%s': %zu trades, %zu paths, %zu simulation dates\n", ns.id.c_str(), ns.trades.size(),
              config.numPaths, uncollateralised.scenarios.numTimes());
  for (std::size_t i = 0; i < ns.trades.size(); ++i)
    std::printf("  %-24s MtM(0) = %12.0f\n", ns.trades[i]->id().c_str(), uncollateralised.tradeValues[i](0, 0));

  // ---------------------------------------------------------------- 3. exposure, netting, collateral
  printProfile("Uncollateralised exposure", uncollateralised.profile);

  const auto& s = uncollateralised.scenarios;
  const auto grossProfile = ExposureProfile::compute(grossPositiveValues(uncollateralised.tradeValues), s.grid,
                                                     s.deflator);
  std::printf("Netting benefit: effective EPE(1y) %.0f netted vs %.0f gross\n",
              uncollateralised.profile.effectiveExpectedPositiveExposure(1.0),
              grossProfile.effectiveExpectedPositiveExposure(1.0));

  NettingSet collateralised = ns;
  CollateralAgreement csa;
  csa.thresholdCounterparty = 250e3;
  csa.thresholdOwn = 250e3;
  csa.minimumTransferAmount = 50e3;
  csa.marginPeriodOfRisk = 10.0 / 250.0;
  collateralised.collateral = csa;
  const ExposureResult withCsa = engine.run(collateralised);
  printProfile("Collateralised exposure (thresholds 250k, MTA 50k, MPR 10d)", withCsa.profile);

  // ---------------------------------------------------------------- 4. AMC: Bermudan swaption
  const auto underlying =
      InterestRateSwap::vanilla("underlying", 10e6, 0.036, 1.0, 9.0, 1, InterestRateSwap::Direction::ReceiveFixed);
  std::vector<double> exercises;
  for (int i = 1; i <= 9; ++i) exercises.push_back(i);
  for (Settlement settlement : {Settlement::Physical, Settlement::Cash}) {
    NettingSet bermudan;
    bermudan.trades.push_back(std::make_shared<BermudanSwaption>("Bermudan 1y x 9y", underlying, exercises, settlement));
    const ExposureResult r = engine.run(bermudan);
    printProfile(settlement == Settlement::Physical ? "Bermudan receiver swaption, physical settlement (AMC)"
                                                    : "Bermudan receiver swaption, cash settlement (AMC)",
                 r.profile);
  }

  // ---------------------------------------------------------------- 5. CVA / DVA
  const double cva = unilateralCva(uncollateralised.profile, *cptyCurve, recovery);
  const double cvaCsa = unilateralCva(withCsa.profile, *cptyCurve, recovery);
  const BilateralCva bcva = bilateralCva(uncollateralised.profile, *cptyCurve, recovery, ownCurve, recovery);
  std::printf("\nUnilateral CVA: %.0f uncollateralised, %.0f under the CSA\n", cva, cvaCsa);
  std::printf("Bilateral: CVA %.0f  DVA %.0f  BCVA %.0f\n", bcva.cva, bcva.dva, bcva.total());
  std::printf("CVA as running spread on 10m notional over 10y: %.2f bp\n",
              1e4 * cvaRunningSpread(cva, *usd, *cptyCurve, 10.0, 10e6));

  const auto contributions =
      marginalExposureContributions(uncollateralised.tradeValues, uncollateralised.nettedValue, &s.deflator);
  std::printf("Marginal (Euler) CVA allocation:\n");
  double allocated = 0.0;
  for (std::size_t i = 0; i < ns.trades.size(); ++i) {
    const double c = unilateralCva(s.grid.times(), contributions[i], *cptyCurve, recovery);
    allocated += c;
    std::printf("  %-24s %12.0f\n", ns.trades[i]->id().c_str(), c);
  }
  std::printf("  %-24s %12.0f\n", "total", allocated);

  // ---------------------------------------------------------------- 6. wrong-way risk
  NettingSet fxOnly;
  fxOnly.trades.push_back(std::make_shared<AssetForward>("EURUSD fwd 5y", "EURUSD", 10e6, 1.10, 5.0));
  std::printf("\nWrong-way risk, long 5y EURUSD forward (pathwise CVA with CIR++ intensity):\n");
  for (double rho : {-0.6, 0.0, 0.6}) {
    Matrix c = corr;
    c(1, 2) = c(2, 1) = rho;
    const ExposureEngine wwrEngine(ScenarioGenerator(hw, {eurusd}, {cptyIntensity}, c), config,
                                   TimeGrid::standardExposureGrid(5.0));
    const ExposureResult r = wwrEngine.run(fxOnly);
    std::printf("  corr(FX, intensity) = %+.1f   CVA = %10.0f   (independence: %10.0f)\n", rho,
                pathwiseCva(r.scenarios, r.exposureValue, 0, recovery, r.reportingIndices),
                unilateralCva(r.profile, *cptyCurve, recovery));
  }

  // ---------------------------------------------------------------- 7. hedging
  const auto& profile = uncollateralised.profile;
  auto cvaOfCurve = [&](const CreditCurve& curve) { return unilateralCva(profile, curve, recovery); };
  const auto cs01 = creditSpreadSensitivities(cvaOfCurve, *usd, cptyQuotes, recovery);
  const auto hedge = cdsHedgeNotionals(cs01, cdsHedgeJacobian(*usd, cptyQuotes, recovery));
  std::printf("\nCVA credit hedge (CS01 per 1bp, CDS protection notional to buy):\n");
  for (std::size_t i = 0; i < cptyQuotes.size(); ++i)
    std::printf("  %4.0fy  CS01 %10.1f   CDS notional %14.0f\n", cptyQuotes[i].maturity, cs01[i], hedge[i]);

  auto cvaForSpot = [&](double bump) {
    const auto bumped = std::make_shared<LognormalAsset>(eurusd->withSpot(eurusd->spot() + bump));
    const ExposureEngine e(generator.withAsset(0, bumped), config, TimeGrid::standardExposureGrid(10.0));
    return unilateralCva(e.run(ns).profile, *cptyCurve, recovery);
  };
  const double fxDelta = centralDifference(cvaForSpot, 0.01);
  // CVA is a liability: when it rises with the spot, the hedge must gain, i.e. buy EUR forward.
  // A 3y forward on N EUR has a spot delta of N * P_EUR(0, 3).
  std::printf("CVA EURUSD delta: %.0f USD per unit of spot -> %s %.0f EUR 3y forward\n", fxDelta,
              fxDelta > 0.0 ? "buy" : "sell", std::abs(fxDelta) / eur->discount(3.0));
  return 0;
}
