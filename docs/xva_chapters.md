# Proposed extension chapters: XVA

*Modelling, Pricing, and Hedging Counterparty Credit Exposure* ends with CVA/DVA, wrong-way risk and
hedging of credit risk. The eight chapters below extend it in the same style: a definition, the
continuous-time formula, the discretisation on the exposure grid, the Monte Carlo estimator, a worked
example, and the pointer to the code (`include/ccr/xva/xva.hpp`, `include/ccr/hedging/sensitivities.hpp`)
and to the page of the web front end that reproduces it. The chapters are not numbered against the book.

## Notation

Times are year fractions from t = 0. `D(0,t)` is the deflator of the simulated bank-account numeraire,
`V(t)` the netting-set value (our point of view), `V⁺ = max(V,0)`, `V⁻ = max(−V,0)`. The discounted
exposures are

- `EE*(t) = E[D(0,t) V⁺(t)]`,
- `ENE*(t) = −E[D(0,t) V⁻(t)] ≤ 0`.

`Q_c`, `Q_o` are the survival probabilities of the counterparty and of ourselves, `PD_c = 1 − Q_c`,
`R_c`, `R_o` the recoveries, `λ_c`, `λ_o` the hazard rates. On the grid `t_0 = 0 < … < t_n`, with
`Δ_k = t_k − t_{k−1}`, a quantity `f` is integrated by the trapezoidal rule on `f(t_{k−1}), f(t_k)`,
and the survival probabilities are evaluated at the interval mid-points `t_{k−½}`
(`exposureXva`).

---

## X1. From CVA to XVA

**Motivation.** CVA prices the loss on default. A derivative desk also pays to fund the collateral and
the uncollateralised positive exposure, to post initial margin, and to hold regulatory capital. Since
2008 (the spread between the overnight-indexed and the interbank curves opened) these are priced and
reported as separate adjustments: *funding* (FVA), *margin* (MVA) and *capital* (KVA) valuation
adjustments, together with CVA and DVA called XVA.

**Replication argument.** Following Burgard and Kjaer (2011), the value of a derivative with
counterparty and own default is `V̂ = V + U`, where `V` is the risk-free value and the adjustment `U`
solves a semi-linear PDE (or BSDE) driven by the close-out value, the funding rate and the hedging
instruments. In the limit of small spreads and a one-period expansion `U` splits into additive terms,
each being an integral of a discounted exposure weighted by the *joint survival probability*
`Q_c(t) Q_o(t)`, because every cost accrues only while neither party has defaulted:

```
XVA = CVA − DVA + FCA − FBA + MVA + KVA
```

All the terms except the credit ones in the default-time expectation are of the form
`s ∫ X*(t) Q_c(t) Q_o(t) dt`, with `s` a spread (cost rate) and `X*` a discounted exposure measure.

**Sign convention.** All adjustments are reported as positive costs; DVA and FBA are benefits and are
subtracted (`Xva::total()`). A positive total is added to the price charged to the client.

**Bilateral CVA** (first to default, as in the CVA chapter):

```
CVA = (1 − R_c) ∫₀ᵀ EE*(t) Q_o(t) dPD_c(t)
DVA = (1 − R_o) ∫₀ᵀ |ENE*(t)| Q_c(t) dPD_o(t)
```

**Discretisation.** `CVA ≈ (1 − R_c) Σ_k EE*(t_{k−½}) Q_o(t_{k−½}) (Q_c(t_{k−1}) − Q_c(t_k))`, with the
exposure at the mid-point taken as the average of the two grid values.

**Why discounting with the risk-free numeraire.** The expectation is under the risk-neutral measure of
the bank-account numeraire, exact for the Hull–White model of the book (the numeraire is simulated
jointly with the short rate). Funding costs are measured as *spreads over* this rate, so the risk-free
discounting is retained and only the spread enters the formulas.

**Worked example.** Single at-the-money 5y swap, flat 3% curve, counterparty CDS 200bp, own CDS 100bp:
CVA, DVA, FCA, FBA, MVA and KVA for the default specification of the XVA page, shown as a bar chart and
as cumulative costs over time.

**Code / page.** `exposureXva`, `xva`, `Xva`, `XvaParameters`; [`web/xva.html`](../web/xva.html).

**Further reading.** Burgard and Kjaer (2011, 2013); Gregory (2020, *The xVA Challenge*); Green (2015,
*XVA: Credit, Funding and Capital Valuation Adjustments*); Brigo, Morini and Pallavicini (2013).

---

## X2. Funding valuation adjustment

**Definition.** A trade that is not (fully) collateralised has to be funded. When the portfolio value is
positive (we are owed money) we borrow the amount at the unsecured rate `r + s_b` and pay the spread
`s_b`: the *funding cost adjustment* FCA. When it is negative we have cash in hand, lend it and earn
`s_l`: the *funding benefit adjustment* FBA.

```
FCA = s_b ∫₀ᵀ EE*(t)  Q_c(t) Q_o(t) dt
FBA = s_l ∫₀ᵀ |ENE*(t)| Q_c(t) Q_o(t) dt          FVA = FCA − FBA
```

**Discretisation.** `FCA ≈ s_b Σ_k ½(EE*(t_{k−1}) + EE*(t_k)) Q_c(t_{k−½}) Q_o(t_{k−½}) Δ_k`. FBA is
identical with `|ENE*|` and `s_l`, and is returned as a non-positive number by `fundingBenefit`.

**Parameters.** `fundingSpreadBorrow = s_b`, `fundingSpreadLend = s_l`. The default `s_l = 0` is the
conservative choice: a desk that cannot lend the funding of its negative positions at a spread earns
nothing.

**The overlap with DVA.** With a funding spread tied to own credit, `s_b ≈ (1 − R_o) λ_o`, the FBA of
own funding replicates a part of the DVA: *the funding benefit is the benefit of own default seen from
the funding side*. Adding DVA and FBA in full double counts. Hull and White (2012, 2014) argue that
FVA must not be charged at all for pricing (the funding cost is offset by the DVA benefit and is
shareholder value, not a market value), while Burgard and Kjaer and the practitioner literature
(Laughton and Vaisbrot, Castagna) argue that it is a real cost of a desk that funds at its own spread.
The library does not take sides: both terms are reported separately, `fva() = fca − fba`, and the
analyst can set `s_l` and the own-credit curve to implement either view (for instance `s_l = s_b` for a
symmetric treatment, or `s_b = 0` to switch FVA off).

**Collateralised trades.** With a CSA in force, `EE*` is the exposure *after* collateral (the margin
period of risk of the Collateral chapter), so the FCA charges the uncollateralised remainder only. The
funding cost of the collateral itself is a function of the difference between the CSA remuneration and
the funding rate, not modelled here (see X8).

**Worked example.** Fan chart of `V(t)` and the profiles `EE*`, `|ENE*|` for a receiver swap; the
asymmetric swap makes FCA small and FBA large. A table of FVA against the funding spread shows the
linearity of FVA in `s_b`.

**Code / page.** `exposureXva` (FCA), `fundingBenefit` (FBA); [`web/xva.html`](../web/xva.html).

**Further reading.** Burgard and Kjaer (2011); Hull and White (2012, 2014); Piterbarg (2010,
"Funding beyond discounting"); Pallavicini, Perini and Brigo (2012); Andersen, Duffie and Song (2019).

---

## X3. Margin valuation adjustment

**Definition.** Cleared and, since the uncleared margin rules, most bilateral trades require *initial
margin* (IM) which covers the loss over the margin period of risk (MPOR) in a close-out, in addition to
the variation margin. The IM is posted, segregated and earns no more than the risk-free rate, so
funding it costs a spread `s_m`:

```
MVA = s_m ∫₀ᵀ IM*(t) Q_c(t) Q_o(t) dt
```

**The IM proxy.** The IM of the regulatory models (SIMM for uncleared, a clearing-house VaR for
cleared) is a quantile of the distribution of the portfolio value change over the MPOR, conditional on
the scenario at `t`. The library uses the unconditional proxy implemented by `initialMarginProfile`:

```
IM(t_k) = q_{99%}  |V(t_k) − V(t_{k−1})|  ·  sqrt(MPOR / Δ_k),      MPOR = 10 / 250
IM*(t_k) = IM(t_k) · E[D(0,t_k)]
```

The quantile `q` is taken over the paths at each grid date, the square-root-of-time rule scales the grid
step to the MPOR, and the mean deflator discounts. This uses the simulated paths, so MVA runs on the
WebAssembly engine only (the fused GPU kernels return aggregates).

**Properties.** MVA is not linear in the exposures, so it does not allocate to trades by Euler's
theorem (X5). It rises with volatility and is largest for long-dated, at-the-money positions, where
the exposure profile and the IM profile are both high. Netting and the choice of the quantile have a
first-order effect: the 99% quantile of a normal is 2.33σ, and the portfolio `σ` is lower than the
sum of the trade `σ` by the diversification of the netting set.

**Limitations.** The proxy does not condition on the state of the paths (a regression of `IM` on the
risk factors, as in the nested-simulation-free approach of Andersen, Pykhtin and Sokol 2017, or the
Longstaff–Schwartz regression of the AMC chapter, would); it ignores that IM is `MPOR`-dependent
per trade type; and it does not net the IM received (MVA is the cost of IM *posted*; the IM *received*
and rehypothecation are beyond the model).

**Worked example.** IM profile and MVA as `s_m` varies for a 10y interest-rate swap; the dependence of
the IM on the MPOR (10 versus 20 business days).

**Code / page.** `initialMarginProfile`, `marginValuationAdjustment`; [`web/xva.html`](../web/xva.html).

**Further reading.** Green and Kenyon (2015) "MVA by replication and regression"; Andersen, Pykhtin and
Sokol (2017) "Rethinking the margin period of risk"; Caspers, Giltinan, Lichters and Mazzoni (2017)
"Forecasting initial margin requirements"; ISDA SIMM methodology; BCBS-IOSCO uncleared margin rules.

---

## X4. Capital valuation adjustment

**Definition.** A bank holds regulatory capital against the counterparty exposure and the shareholders
require a hurdle return on it. Over the life of the trade the capital is not free: KVA is the present
value of the cost of that capital (Green, Kenyon and Dennis, 2014):

```
KVA = γ ∫₀ᵀ K(t) Q_c(t) Q_o(t) dt  with  K*(t) = capitalRatio · RW · α · EE*(t),   γ = hurdleRate
```

so that `KVA = hurdle · ratio · RW · α ∫ EE*(t) Q_c(t) Q_o(t) dt`.

**The regulatory capital.** The exposure at default of the Basel II/III internal model method is
`EAD = α · EEPE` with `α = 1.4` (the multiplier of the *Effective EPE* of the Exposure chapter, which
the Basel text fixed to account for wrong-way risk and the granularity of portfolios). The capital is
`ratio · RW · EAD` with `ratio = 8%` and a risk weight `RW` that depends on the counterparty rating. The
library uses the *expected exposure profile* `EE*(t)` in place of the single number EEPE, so that capital
runs down with the trade (a lifetime view, consistent with the other terms).

**Parameters.** `hurdleRate`, `capitalRatio`, `riskWeight`, `alpha` in `XvaParameters`. KVA is linear in
`EE*` and therefore allocates like CVA and FCA.

**Simplifications.** The capital here is the counterparty-default capital on the exposure only. The
full KVA of Green et al. also contains market-risk capital, the CVA capital charge and the leverage
ratio; the Basel III reform (SA-CCR, the standardised and basic CVA frameworks) replaces `α · EEPE` by
formulas that are not proportional to the simulated `EE`. Moreover a hurdle rate is applied to
the capital *net of* the tax shield and the part of the capital funded by the shareholder only; the
tax and the dividend policy are beyond the model.

**Double counting.** The KVA of the risk that has already been charged as CVA and as CVA capital (the
reserve) is partly offset by the CVA reserve itself (Kenyon and Green 2013, "Regulatory costs break
risk neutrality"). The library reports KVA gross of such offsets.

**Worked example.** Total XVA against the hurdle rate (0%, 10%, 15%): KVA is linear in the hurdle; for a
long swap with a risky counterparty KVA exceeds CVA at a hurdle of 10%.

**Code / page.** `exposureXva` (KVA), `XvaParameters`; [`web/xva.html`](../web/xva.html).

**Further reading.** Green, Kenyon and Dennis (2014) "KVA: capital valuation adjustment";
Kenyon and Green (2013); BCBS 128 (2006), 189 (2011) and 279 (SA-CCR, 2014); Elouerkhaoui (2016)
"Credit Correlation: Theory and Practice", on CVA capital.

---

## X5. XVA allocation to trades

**Problem.** The XVA is computed on the netting set; the desk, the P&L and the client price need the
amount attributable to each trade, adding up to the netting-set total.

**Euler allocation.** If a measure `F(w)` is positively homogeneous of degree one in the weights `w` of
the trades in the portfolio, then `F = Σ_i w_i ∂F/∂w_i` (Euler's theorem). `EE*(t) = E[D(0,t)(Σ_i V_i(t))⁺]`
is homogeneous of degree one, with the marginal contribution

```
EE*_i(t) = E[ D(0,t) V_i(t) 1{V(t) > 0} ],     Σ_i EE*_i(t) = EE*(t)
```

(the Exposure Allocation chapter). CVA, FCA and KVA are *linear in EE** for given survival
probabilities, so they inherit the property and the allocation is exact:

```
CVA_i = (1 − R_c) ∫ EE*_i Q_o dPD_c,  FCA_i = s_b ∫ EE*_i Q_c Q_o dt,  KVA_i = κ ∫ EE*_i Q_c Q_o dt
Σ_i CVA_i = CVA,  Σ_i FCA_i = FCA,  Σ_i KVA_i = KVA
```

`allocateExposureXva` returns these for every trade from the marginal discounted EE and the tests check
the additivity to rounding error.

**Not allocated this way.** FBA depends on `ENE*`, which is allocated by the symmetric formula with
`1{V(t) < 0}` and can be added in the same way, whereas MVA does not decompose: `IM` is a *quantile* of
the net portfolio change and, as a function of the weights, is homogeneous of degree one too but its
marginal contributions have to be estimated by regression or by pathwise (likelihood-ratio) derivatives
(Anfuso, Aziz, Giltinan and Loukopoulos 2017, "A sound modelling and backtesting framework for forecasting
initial margin requirements"; Green and Kenyon). The library reports MVA at the netting-set level.

**Incremental XVA.** The price of a *new* trade given the existing book is `XVA(book + trade) −
XVA(book)`; it is not the Euler contribution and not the standalone XVA either. The difference
between the three numbers (standalone, marginal, incremental) is the netting benefit; the order is
standalone ≥ incremental ≈ marginal for a diversifying trade.

**Monte Carlo error.** The marginal contributions are noisy conditional expectations on the event
`{V > 0}` and are estimated with the same paths as the total, so the sum rule holds path by path;
common random numbers keep the error of differences small.

**Worked example.** Three-trade netting set (swap, FX forward, option) on the Allocation page, table of
CVA, FCA and KVA per trade; the offsetting swap receives a negative KVA.

**Code / page.** `allocateExposureXva`, `marginalExposureContributions`; [`web/allocation.html`](../web/allocation.html),
[`web/xva.html`](../web/xva.html).

**Further reading.** Pykhtin and Rosen (2010) "Pricing counterparty risk at the trade level and CVA
allocations"; Tasche (2008) "Capital allocation to business units and sub-portfolios: the Euler principle";
Kenyon and Green (2015) "Efficient XVA allocation".

---

## X6. XVA sensitivities and hedging

**Why hedge.** The XVA of a book is a large, volatile P&L item: it changes with the credit spreads of the
counterparties (CVA), with the own spread (DVA, FBA), with funding spreads (FVA, MVA), and with the
market (through `EE*`). The desk hedges the credit part with CDS and the market part with the
underlying risk factors.

**Credit sensitivities.** CVA is a *short credit position* on the counterparty. With the hazard curve
bootstrapped from par CDS quotes `s_1…s_m`, the bucketed sensitivity (CS01) is

```
CS01_i = ( XVA(s_i + h) − XVA(s_i − h) ) / (2h) · bump,     bump = 1 bp
```

re-bootstrapping the hazard curve at each bump (`creditSpreadSensitivities`). The central difference
removes the first-order error. For XVA, both the counterparty and the own curve are bumped:
`counterpartyCs01` (CVA, FCA and KVA through the weight `Q_c`) and `ownCs01` (DVA, FBA and the
weight `Q_o` of every term). `fundingSpread01` is the change of the total for a parallel shift of the
borrow, lend and margin spreads together.

**Hedge notionals.** A unit-notional par CDS of maturity `T_k` has a sensitivity matrix `J(k,i)` to the
quotes (`cdsHedgeJacobian`); the hedge solves `Σ_k n_k J(k,i) = CS01_i` for all buckets (`cdsHedgeNotionals`),
a square linear system, so the CDS hedge matches the bucketed exposure. Because CVA is short credit,
the notionals are positive: one buys protection.

**The own-credit paradox.** DVA and FBA are gains when our credit deteriorates, and *cannot be hedged*
by trading our own CDS without selling protection on ourselves. The practice (proxy hedges with the
CDS of correlated names, or a regulator that disallows DVA in capital) is an extension; the library
reports `ownCs01` so that the exposure to own credit is visible.

**Market risk and the CRN.** The delta of the XVA with respect to a market factor is by bump-and-revalue.
With Monte Carlo the bump must use *common random numbers* (the same seed), otherwise the sensitivity
is overwhelmed by noise of order `σ/(h √N)`; with CRN the difference of the two simulations is smooth,
as the Hedging page shows by comparing with and without. In the finite-difference of a pathwise payoff
discontinuity (digitals, barriers), the pathwise estimator fails and the likelihood-ratio estimator or
a smoothed payoff has to be used.

**Higher-order effects.** Gamma of CVA (through the stochastic intensity and wrong-way risk), the
cross-gamma between credit and market (`∂²CVA/∂s ∂S`), and the *jump to default* are not captured by
CS01. The Wrong-Way Risk page shows how CVA depends on the correlation.

**Worked example.** Bucketed CS01 of the CVA of a 5y swap with a 4-quote counterparty CDS curve, the CDS
hedge notionals, and the residual after hedging when the whole curve shifts in parallel.

**Code / page.** `creditSpreadSensitivities`, `cdsHedgeJacobian`, `cdsHedgeNotionals`, `centralDifference`,
`xvaSensitivities`; [`web/hedging.html`](../web/hedging.html).

**Further reading.** Capponi (2013); Green (2015, ch. on hedging); Glasserman and Yu (2004), the pathwise
and likelihood-ratio methods; Brigo and Mercurio (2006) on the CDS bootstrapping.

---

## X7. Numerical practice

**Monte Carlo error.** Every term is an expectation of a smooth function of the exposure and the error is
`O(1/√N)`. The profiles `EE*` at different dates use the same paths, so the errors are highly positively
correlated and mostly cancel in the XVA integral; the standard error is estimated by batching (`B`
independent seeds). Antithetic variates (default) remove the linear part of the dependence.

**Grid.** Exposure grids are weekly, monthly and quarterly, with trade event dates (fixings, payments,
exercise dates) and margin call dates `t − MPR` inserted (Collateral chapter). The XVA integrals are
trapezoidal in time, with survival probabilities at the mid-point: the error is `O(Δ²)` for smooth
integrands but the CVA integrand is not smooth at the cash-flow dates. A refinement study halving
the grid is the standard check.

**Regression.** The AMC chapter (Longstaff–Schwartz) gives `V(t)` for early-exercise trades; the XVA of a
Bermudan swaption needs `V(t)` at all dates, in-the-money or not, so the regression basis must cover
the whole range of the states (not only the in-the-money paths, as for the exercise decision).

**Validation checks** that the test suite runs and that a practitioner should repeat:

1. *Martingale test*: `E[D(0,t) V(t)] = V(0)` for trades with no cash flows before `t`.
2. *Additivity*: the allocated CVA, FCA and KVA add up to the netting-set values.
3. *Limits*: with zero spreads FVA, MVA and KVA vanish; with `s_l = s_b` and `Q_o = 1` the FVA is
   `s ∫ (EE* − |ENE*|) Q_c dt = s ∫ E[D V] Q_c dt`, linear in the *signed* value.
4. *Monotonicity*: CVA increases in the counterparty spread, DVA in the own spread.
5. *Consistency with the exposure*: setting `Q_c = Q_o = 1` the costs are `s ∫ EE*`.
6. *Engine parity*: the WebAssembly, the CPU-fused and the WebGPU engines agree on the aggregates
   (EE, PFE) within tolerance.

**Performance.** The exposure pipeline is the bottleneck: the simulation is run once and `EE*`,
`ENE*`, `IM*` reuse it. On GPU the pipeline is a fused kernel with partial-sum reductions and a
quantile kernel; MVA needs the path matrix and so stays on the WebAssembly engine.

**Code / page.** `web/tests/wasm.test.mjs`, `web/tests/e2e.mjs`; [`web/gpu.html`](../web/gpu.html).

**Further reading.** Glasserman (2004) *Monte Carlo Methods in Financial Engineering*; Longstaff and
Schwartz (2001); Joshi (2008) on pricing and the Greeks.

---

## X8. Beyond this implementation (outlook)

No code or web page exists yet for these topics; they are research directions that follow from the
simplifications made above.

1. **Collateralised funding and discounting.** With a CSA, the discount rate is the collateral rate
   (OIS, or the cheapest-to-deliver currency); the funding benefit of the optionality of the currency
   choice is a FVA-type term (Piterbarg 2010, Fujii–Shimada–Takahashi 2010). Requires multi-curve
   Hull–White with a basis spread.
2. **Capital with SA-CCR and the CVA frameworks.** Replace `α · EEPE` by the SA-CCR add-on and
   multiplier (a function of the net MtM and the add-on, not of `EE`), and add the CVA capital charge
   (BA-CVA, SA-CVA) with its own hedge recognition, giving a KVA of the hedged and the unhedged CVA
   (Green–Kenyon–Dennis; Kenyon–Green).
3. **Stochastic funding spreads and wrong-way funding.** `s_b(t)` driven by a mean-reverting process
   correlated with the intensity of the own bank and the exposure; the FVA becomes a pathwise integral
   `∫ D(0,t) s_b(t) V⁺(t) Q dt` with a non-trivial correlation structure, as for the wrong-way
   CVA of the Wrong-way risk page.
4. **Conditional initial margin.** A regression-based or nested-simulation IM per path (SIMM
   sensitivities with the Longstaff–Schwartz machinery, or the full-revaluation historical VaR of CCPs),
   and the IM that we *receive* with its rehypothecation.
5. **XVA as a coherent measure with feedback.** The semi-linear PDE of Burgard–Kjaer with the
   cross-dependence of the funding rate on the adjusted value (the non-linearity, fixed here by taking
   the exposure at the risk-free value). Solvable by BSDE and regression (Henry-Labordère 2012;
   Crépey 2015).
6. **Portfolio-level XVA optimisation.** Choose the netting sets, clearing and the CSA terms to minimise
   the XVA of the book; the optimal hedge in the presence of transaction costs and the funding
   constraint on the hedging instruments.
7. **Accelerated XVA.** Algorithmic differentiation of the whole XVA (adjoint Greeks) instead of the
   bump-and-revalue of X6, which gives all the sensitivities at a small multiple of the cost of
   the XVA itself (Capriotti and Giles 2012; Capriotti, Lee and Peacock 2011).

---

## References

- Andersen, L., Duffie, D., Song, Y. (2019). Funding value adjustments. *Journal of Finance* 74(1).
- Andersen, L., Pykhtin, M., Sokol, A. (2017). Rethinking the margin period of risk. *Journal of Credit Risk* 13(1).
- Brigo, D., Mercurio, F. (2006). *Interest Rate Models: Theory and Practice*. Springer.
- Brigo, D., Morini, M., Pallavicini, A. (2013). *Counterparty Credit Risk, Collateral and Funding*. Wiley.
- Burgard, C., Kjaer, M. (2011). In the balance. *Risk*, October; (2013) Funding strategies, funding costs. *Risk*, December.
- Capriotti, L., Giles, M. (2012). Algorithmic differentiation: adjoint Greeks made easy. *Risk*, September.
- Caspers, P., Giltinan, P., Lichters, R., Mazzoni, N. (2017). Forecasting initial margin requirements – a model evaluation. *Journal of Financial Risk Management*.
- Fujii, M., Shimada, Y., Takahashi, A. (2010). A note on construction of multiple swap curves with and without collateral. CARF working paper.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Green, A. (2015). *XVA: Credit, Funding and Capital Valuation Adjustments*. Wiley.
- Green, A., Kenyon, C., Dennis, C. (2014). KVA: capital valuation adjustment. *Risk*, December.
- Gregory, J. (2020). *The xVA Challenge*, 4th ed. Wiley.
- Henry-Labordère, P. (2012). Cutting CVA's complexity. *Risk*, July.
- Hull, J., White, A. (2012). The FVA debate. *Risk*, July; (2014) Valuing derivatives: funding value adjustments and fair value. *Financial Analysts Journal* 70(3).
- Kenyon, C., Green, A. (2013). Regulatory costs break risk neutrality. *Risk*, September.
- Longstaff, F., Schwartz, E. (2001). Valuing American options by simulation. *Review of Financial Studies* 14(1).
- Pallavicini, A., Perini, D., Brigo, D. (2012). Funding, collateral and hedging: uncovering the bilateral risk. arXiv:1210.3811.
- Piterbarg, V. (2010). Funding beyond discounting: collateral agreements and derivatives pricing. *Risk*, February.
- Pykhtin, M., Rosen, D. (2010). Pricing counterparty risk at the trade level and CVA allocations. *Journal of Credit Risk* 6(4).
- Tasche, D. (2008). Capital allocation to business units and sub-portfolios: the Euler principle. In *Pillar II in the New Basel Accord*.
