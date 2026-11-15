#include <cmath>
#include <memory>

#include "ccr/cva/cva.hpp"
#include "ccr/exposure/allocation.hpp"
#include "ccr/exposure/exposure_engine.hpp"
#include "ccr/hedging/sensitivities.hpp"
#include "ccr/instruments/asset_forward.hpp"
#include "ccr/xva/xva.hpp"
#include "test_framework.hpp"

using namespace ccr;

namespace {

ExposureProfile humpedProfile() {
  ExposureProfile p;
  for (int i = 0; i <= 20; ++i) {
    const double t = 0.25 * i;
    p.times.push_back(t);
    p.discountedExpectedExposure.push_back(1e6 * 0.02 * std::sqrt(t) * (5.0 - t) / 5.0);
    p.discountedExpectedNegativeExposure.push_back(-0.5e6 * 0.02 * std::sqrt(t) * (5.0 - t) / 5.0);
  }
  return p;
}

}  // namespace

TEST(xva_reduces_to_cva_dva_without_funding_and_capital) {
  const auto p = humpedProfile();
  const CreditCurve cpty = CreditCurve::flat(0.02), own = CreditCurve::flat(0.01);
  XvaParameters none;
  none.fundingSpreadBorrow = none.fundingSpreadLend = none.marginFundingSpread = none.hurdleRate = 0.0;
  const Xva x = xva(p, {}, cpty, 0.4, own, 0.4, none);
  const auto b = bilateralCva(p, cpty, 0.4, own, 0.4);
  CHECK_NEAR(x.cva, b.cva, 1e-9);
  CHECK_NEAR(x.dva, b.dva, 1e-9);
  CHECK_NEAR(x.total(), b.total(), 1e-9);
  CHECK_NEAR(x.fca + x.fba + x.mva + x.kva, 0.0, 0.0);
}

TEST(fva_of_constant_exposure_and_symmetry) {
  std::vector<double> t, ee, ene;
  for (int i = 0; i <= 8; ++i) {
    t.push_back(0.5 * i);
    ee.push_back(100.0);
    ene.push_back(-100.0);
  }
  const CreditCurve none = CreditCurve::flat(0.0);
  XvaParameters p;
  p.fundingSpreadBorrow = 0.02;
  p.fundingSpreadLend = 0.005;
  // No default risk: FCA = s_b * EE * T.
  const auto e = exposureXva(t, ee, none, 0.4, none, p);
  CHECK_NEAR(e.fca, 0.02 * 100.0 * 4.0, 1e-12);
  CHECK_NEAR(e.cva, 0.0, 1e-15);
  CHECK_NEAR(fundingBenefit(t, ene, none, none, p.fundingSpreadLend), 0.005 * 100.0 * 4.0, 1e-12);
  CHECK_NEAR(marginValuationAdjustment(t, ee, none, none, 0.01), 0.01 * 100.0 * 4.0, 1e-12);
  // Default risk shrinks the funding cost.
  CHECK(exposureXva(t, ee, CreditCurve::flat(0.05), 0.4, none, p).fca < e.fca);
  // KVA scales with the hurdle rate and the risk weight.
  XvaParameters q = p;
  q.riskWeight = 2.0;
  CHECK_NEAR(exposureXva(t, ee, none, 0.4, none, q).kva, 2.0 * e.kva, 1e-12);
}

TEST(xva_total_combines_every_adjustment) {
  const auto p = humpedProfile();
  const CreditCurve cpty = CreditCurve::flat(0.02), own = CreditCurve::flat(0.01);
  std::vector<double> im(p.times.size(), 5e4);
  im[0] = 0.0;
  const XvaParameters params;
  const Xva x = xva(p, im, cpty, 0.4, own, 0.4, params);
  CHECK(x.cva > 0.0 && x.dva > 0.0 && x.fca > 0.0 && x.mva > 0.0 && x.kva > 0.0);
  CHECK_NEAR(x.fba, 0.0, 0.0);  // no lending spread by default
  CHECK_NEAR(x.total(), x.cva - x.dva + x.fca - x.fba + x.mva + x.kva, 1e-9);
  XvaParameters lend = params;
  lend.fundingSpreadLend = 0.004;
  CHECK(xva(p, im, cpty, 0.4, own, 0.4, lend).total() < x.total());
}

TEST(xva_allocation_adds_up_and_initial_margin) {
  const auto yc = std::make_shared<YieldCurve>(YieldCurve::flat(0.02));
  const auto foreign = std::make_shared<YieldCurve>(YieldCurve::flat(0.01));
  const auto hw = std::make_shared<HullWhite1F>(yc, 0.05, 0.01);
  const ScenarioGenerator gen(hw, {std::make_shared<LognormalAsset>("FX", 1.0, 0.12, foreign)});
  NettingSet ns;
  ns.trades.push_back(std::make_shared<AssetForward>("long", "FX", 1e6, 1.0, 3.0));
  ns.trades.push_back(std::make_shared<AssetForward>("short", "FX", -4e5, 1.1, 2.0));
  const ExposureEngine engine(gen, {4000, 9, true}, TimeGrid::uniform(3.0, 12));
  const auto r = engine.run(ns);
  const auto& s = r.scenarios;
  const CreditCurve cpty = CreditCurve::flat(0.02), own = CreditCurve::flat(0.01);
  const XvaParameters params;

  const auto contributions = marginalExposureContributions(r.tradeValues, r.nettedValue, &s.deflator);
  const auto parts = allocateExposureXva(s.grid.times(), contributions, cpty, 0.4, own, params);
  const auto whole = exposureXva(r.profile.times, r.profile.discountedExpectedExposure, cpty, 0.4, own, params);
  double cva = 0.0, fca = 0.0, kva = 0.0;
  for (const auto& e : parts) {
    cva += e.cva;
    fca += e.fca;
    kva += e.kva;
  }
  CHECK_NEAR(cva, whole.cva, 1e-8 * whole.cva);
  CHECK_NEAR(fca, whole.fca, 1e-8 * whole.fca);
  CHECK_NEAR(kva, whole.kva, 1e-8 * whole.kva);

  const auto im = initialMarginProfile(r.nettedValue, s.grid, s.deflator);
  CHECK_NEAR(im[0], 0.0, 0.0);
  CHECK(im.back() > 0.0);
  CHECK(marginValuationAdjustment(s.grid.times(), im, cpty, own, 0.01) > 0.0);
}

TEST(xva_sensitivities_funding_and_credit) {
  const YieldCurve yc({1.0, 10.0}, {0.02, 0.03});
  const std::vector<CdsQuote> cptyQuotes = {{1.0, 0.0080}, {3.0, 0.0100}, {5.0, 0.0130}};
  const std::vector<CdsQuote> ownQuotes = {{1.0, 0.0040}, {3.0, 0.0050}, {5.0, 0.0060}};
  const auto p = humpedProfile();
  const XvaParameters params;
  const auto s = xvaSensitivities(p, {}, yc, cptyQuotes, 0.4, ownQuotes, 0.4, params);
  CHECK(s.counterpartyCs01.size() == 3 && s.ownCs01.size() == 3);
  double parallel = 0.0;
  for (double c : s.counterpartyCs01) parallel += c;
  CHECK(parallel > 0.0);   // wider counterparty spreads raise CVA and FCA's survival-weighted decline is smaller
  CHECK(s.fundingSpread01 > 0.0);  // a positive EE profile costs more to fund
  // Linear in the spread: FVA01 matches the finite difference of the total.
  const CreditCurve c = bootstrapCreditCurve(yc, cptyQuotes, 0.4), o = bootstrapCreditCurve(yc, ownQuotes, 0.4);
  XvaParameters up = params;
  up.fundingSpreadBorrow += 1e-4;
  up.fundingSpreadLend += 1e-4;
  up.marginFundingSpread += 1e-4;
  CHECK_NEAR(s.fundingSpread01, xva(p, {}, c, 0.4, o, 0.4, up).total() - xva(p, {}, c, 0.4, o, 0.4, params).total(),
             1e-6 * std::fabs(s.fundingSpread01));
  const auto notionals = cdsHedgeNotionals(s.counterpartyCs01, cdsHedgeJacobian(yc, cptyQuotes, 0.4));
  for (double n : notionals) CHECK(std::isfinite(n));
}
