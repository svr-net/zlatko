# Proposed extension chapters: XVA

*Modelling, Pricing, and Hedging Counterparty Credit Exposure* ends with CVA/DVA, wrong-way risk and
hedging of credit risk. The eleven chapters below extend it in the same style: a definition, the
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
Every adjustment is computed per netting set; [`web/portfolio.html`](../web/portfolio.html) runs the XVA (and the
other analyses) on a book of netting sets with several counterparties on common scenarios, and adds them up by
counterparty and for the book: CVA, DVA, FCA, FBA, MVA and KVA are all sums over netting sets, since each is an
integral of a netting-set exposure (or initial margin) against that set's counterparty curve.

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
the scenario at `t`. The library offers two estimates: the conditional IM by nested Monte Carlo (X8, the
default of the XVA page) and the unconditional proxy implemented by `initialMarginProfile`:

```
IM(t_k) = q_{99%}  |V(t_k) − V(t_{k−1})|  ·  sqrt(MPOR / Δ_k),      MPOR = 10 / 250
IM*(t_k) = IM(t_k) · E[D(0,t_k)]
```

The quantile `q` is taken over the paths at each grid date, the square-root-of-time rule scales the grid
step to the MPOR, and the mean deflator discounts. In the fused kernels the proxy is a row-statistics
request (the quantile of the value change between reporting dates), so it runs on WebGPU and on the CPU
kernels alike. The nested estimate replaces `IM*(t)` by `E[D(0,t) IM(t, ω)]`, with `IM(t, ω)` the
conditional quantile of `|V(t + MPOR) − V(t)|` given the path; see X8.

**Properties.** MVA is not linear in the exposures, so it does not allocate to trades by Euler's
theorem (X5). It rises with volatility and is largest for long-dated, at-the-money positions, where
the exposure profile and the IM profile are both high. Netting and the choice of the quantile have a
first-order effect: the 99% quantile of a normal is 2.33σ, and the portfolio `σ` is lower than the
sum of the trade `σ` by the diversification of the netting set.

**Limitations.** The proxy does not condition on the state of the paths; the nested estimate of X8
does (at the cost of the inner simulation and a small downward bias of the empirical quantile), and a
regression of `IM` on the risk factors (Andersen, Pykhtin and Sokol 2017, or the Longstaff–Schwartz
machinery of the AMC chapter) would do so more cheaply, with a basis bias instead. Both estimates ignore that IM is `MPOR`-dependent
per trade type; and it does not net the IM received (MVA is the cost of IM *posted*; the IM *received*
and rehypothecation are beyond the model).

**Worked example.** IM profile and MVA as `s_m` varies for a 10y interest-rate swap; the dependence of
the IM on the MPOR (10 versus 20 business days).

**Code / page.** `initialMarginProfile`, `marginValuationAdjustment`; nested IM `FusedOptions::imInnerPaths`
(analysis option `imMethod`: `"nested"` by default, or `"proxy"`); [`web/xva.html`](../web/xva.html) shows both.

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

## X8. Nested Monte Carlo in XVA modelling

**Motivation.** Every XVA term of the previous chapters needs the value `V(t)` of the netting set at each
future date and scenario. For the trades of the book (swaps, forwards, European options) `V(t)` is
analytic given the state, so one simulation suffices (*simple* Monte Carlo). When it is not (Bermudan and
barrier products, path-dependent payoffs, CVA on a CVA, conditional initial margin, XVA as the
underlying of a hedge or of a capital calculation) the value is itself a conditional expectation and has
to be estimated by simulation: *nested* (or "Monte Carlo on Monte Carlo") simulation.

**Structure.** An *outer* simulation of `N_o` paths of the risk factors from `0` to the grid dates `t_k`
produces the scenarios `ω`. At each `(t_k, ω)` an *inner* simulation of `N_i` paths started from the
state `X(t_k, ω)` estimates

```
V(t_k, ω) = E[ Σ_{t_j > t_k} D(t_k, t_j) CF_j | F_{t_k} ](ω)  ≈  (1/N_i) Σ_{m=1}^{N_i} Σ_j D^m(t_k, t_j) CF_j^m
```

and the outer expectation is then taken of a (usually non-linear) function `g` of the inner estimate:

```
Y = E[ g( V(t_k) ) ]  ≈  (1/N_o) Σ_{n=1}^{N_o} g( V̂_{N_i}(t_k, ω_n) )
```

with `g(v) = D(0,t_k) v⁺` for EE*, `g(v) = D(0,t_k) v⁻` for ENE*, and the quantile for IM and PFE. The
cost is `N_o · N_i · K` paths times the number of steps: with `N_o = 10⁴`, `N_i = 10³` and a grid of
`K = 100` dates this is `10⁹` inner paths.

**The bias of the nested estimator.** Because `g` is convex or non-smooth (`v⁺` has a kink at 0), the
estimator is biased even when the inner estimate is unbiased. With `V̂ = V + ε`, `E[ε | V] = 0`,
`Var(ε | V) = σ²(V)/N_i`, a second-order expansion gives

```
E[ g(V̂) ] − E[ g(V) ] ≈ ½ E[ g''(V) σ²(V) ] / N_i
```

and for `g(v) = v⁺`, `g'' = δ(v)`, hence the bias `≈ ½ f_V(0) σ²(0) / N_i`, with `f_V` the density of `V` at
zero. The bias is `O(1/N_i)`, positive for convex `g` (the exposure is *overestimated*, since
`E[(V+ε)⁺] ≥ (E[V+ε])⁺`), and largest where the exposure is near zero: at-the-money, at early dates and for
netted books. The variance of the outer estimate is `O(1/N_o)`, so the mean-squared error is

```
MSE ≈ c₁ / N_i²  +  c₂ / N_o      with budget  N_o N_i = B
```

and is minimised at `N_i ∝ B^{1/3}` and `N_o ∝ B^{2/3}`, giving the *optimal convergence* `MSE ∝ B^{−2/3}`,
i.e. root-MSE `O(B^{−1/3})`: slower than the `B^{−1/2}` of simple Monte Carlo (Gordy and Juneja 2010,
"Nested simulation in portfolio risk measurement"). For a quantile (PFE, IM) the same rate holds under
smoothness conditions on the density.

**Bias reduction.**

1. *Richardson or jackknife.* Combine estimates with `N_i` and `N_i/2` inner paths (or with halves of the
   same sample): `2 g(V̂_{N_i}) − ½[g(V̂_a) + g(V̂_b)]` removes the `1/N_i` term and gives rate `B^{−4/5}`
   (Giles and Haji-Ali 2019, multilevel nested simulation).
2. *Common random numbers across outer scenarios.* The same inner normals (and antithetics) at every outer
   point make the inner errors smooth in the state, which helps the regression-based and the
   differentiated estimators below.
3. *Control variates.* Use a product with an analytic value (the swap, the Jamshidian swaption of the
   Hull–White chapter) as a control for the inner estimate of a Bermudan: the error of `V̂` is the error
   of the *difference* only. The exact martingale property `E[D(0,t)V(t)] = V(0)` checks the sum.
4. *Variance-reducing inner design.* Antithetic variates, stratified or quasi-random inner draws, and the
   exact joint transition of `(x, ∫r)` of the Hull–White model keep the inner discretisation bias at zero.
5. *Smoothing `g`.* Replace the kink by a smooth approximation of the positive part over a width
   `h ∼ N_i^{−1/2}`; this trades a small bias in `h` for a large reduction in the inner-noise bias.

**Regression instead of inner paths (least-squares Monte Carlo).** The cheapest "nesting" is none: the
Longstaff–Schwartz regression of the AMC chapter estimates `V(t_k)` from the *outer* cross-section by
projecting the discounted future cash flows onto basis functions `φ_l(X(t_k))`:

```
V̂(t_k, ω) = Σ_l β_l φ_l(X(t_k, ω)),     β = argmin Σ_n ( Y_n − Σ_l β_l φ_l(X_n) )²
```

With `N_o` paths and `L` basis functions the regression noise is `O(L/N_o)` and the basis bias is the
projection error, so one simulation costs `N_o` paths (not `N_o N_i`), which is why the library
(`core/regression.hpp`, `instruments/bermudan_swaption.hpp`) uses it. Its limits are the choice of the
basis (completeness in a multi-factor state), the out-of-sample bias at the tail where PFE and IM live,
and the cost of `K` regressions. The regression estimate of `V` is *smooth*, which also removes the
kink-bias of the nested estimator at the price of a basis bias: the two are compared in the examples
below. Hybrid schemes (a small inner simulation to compute the regression targets, as in *nested
regression*, Broadie, Du and Moallemi 2015) get the robustness of the first at a cost close to the second.

**Where nesting is genuinely needed in XVA.**

- *Initial margin.* IM at `(t_k, ω)` is the quantile of the value change over the MPOR *conditional on the
  state*: a quantile of the inner distribution, with `N_i` inner paths of length `MPOR` from each outer
  point. The unconditional proxy of X3 replaces this nest by a quantile across the *outer* paths. The
  conditional version makes MVA dependent on the current volatility and on the netting-set state, and
  introduces a `1/√N_i` bias in the quantile (estimated by the order statistic) that can be reduced by the
  Harrell–Davis or kernel quantile estimators.
- *Exercise boundary of the exotics.* Regression gives a sub-optimal policy and therefore a low-biased
  price. A nested inner simulation at the boundary evaluates the *continuation value* without the
  regression bias (the upper bound of Andersen and Broadie 2004, the dual method), so the pair
  lower/upper bounds brackets the true exposure.
- *Wrong-way and right-way risk through default.* The inner simulation conditions on survival and on the
  default of the counterparty at `t_k`, giving `V(t_k)` *given the default*; the close-out value
  (the replacement cost of ISDA) then differs from the risk-free `V`, and the CVA of the book is a nest
  of the close-out valuation at the default time (Brigo and Morini 2011, "Close-out convention
  tensions"; Burgard and Kjaer).
- *Capital and CVA of CVA.* KVA of a *hedged* CVA book needs the regulatory capital at `t_k`, which
  is itself a function of the exposure simulated from `t_k`: the capital is a nested expectation
  (the SA-CCR and the CVA capital formulas are evaluated on the inner exposure), the inner step
  the exposure and the outer step the KVA integral.
- *Forward-looking sensitivities.* The hedge ratios to be held at `t_k` (the "future CS01", needed
  for the cost of hedging and for the MVA of the hedges) are inner derivatives at each outer node. With
  a nest the differences need CRN and a pathwise or likelihood-ratio inner estimator (X6) to remain
  stable.
- *Model risk and the XVA of the XVA.* The valuation of the adjustment at `t_k` as an asset to be sold
  (the *fair value* of the XVA, for the P&L explain and for the IFRS 13 measurement).

**Computational design.**

- *Allocation of the budget.* The rate `B^{−2/3}` above gives `N_i ∼ (c₁/c₂)^{1/3} B^{1/3}`: for a budget of
  `10⁹` paths, `N_i` of order `10²` and `N_o` of order `10⁷`. In practice `N_i` is *small* (`10`–`100`) and
  the bias is removed by (1) or (5).
- *Adaptive nesting.* Outer scenarios far from the kink (deep in or out of the money) need few inner
  paths. Allocate `N_i(ω) ∝ σ(V(ω))/|V(ω)|` or use sequential stopping on the standard error of the inner
  estimate (Broadie, Du and Moallemi 2011, "Efficient risk estimation via nested sequential simulation").
  Implemented in X11 on top of the regression: a proxy, an audit of its bias, and inner paths only for
  the scenarios whose sign or tail membership is uncertain.
- *Re-use of the outer paths.* The inner simulation from `t_k` can re-use the outer paths from `t_k`
  onward as the "inner" ones (a *kernel* or *one-sided* estimator): the inner estimate of the node `ω` is the
  average of the outer continuation over the neighbours of `X(t_k, ω)`. This is the regression with a
  local basis; it avoids the cube cost and the bias is the neighbourhood size.
- *Parallelism and GPU.* The inner simulations are independent given the outer state: `N_o` independent
  tasks, fused on a GPU with the structure of the WebGPU kernel of the GPU chapter (one thread per
  outer path and inner replications in the thread loop; partial-sum reductions), and over Web Workers on
  the CPU. Memory is the constraint: the nest must not store the cube `N_o × N_i × K`, but accumulate
  `g(V̂)` online per outer node.
- *Reproducibility.* Counter-based seeds of the form `(outer path, date, inner index)` make the nest
  reproducible and independent of the work partition, and give the CRN of (2) at no cost.

**Validation.**

1. *Convergence study in `N_i`* at fixed large `N_o`: the exposure decreases as `1/N_i` towards the limit;
   a log-log plot has slope `−1` and the Richardson extrapolate is flat.
2. *Analytic benchmark*: a swap or a European option, for which `V(t)` is known, run through the nest
   and compared with the analytic EE*, to measure the bias term by term.
3. *Martingale test* on the discounted inner estimate, and the equality of the nested EE of a swap with
   the regression one at large sample.
4. *Dual bound check* for a Bermudan swaption: lower (regression policy) `≤` price `≤` upper (nested dual).

**Worked example.** EE*, `EE` bias against `N_i` for an at-the-money 5y swap on a Hull–White model:
nested estimator, with Richardson correction, against the regression estimate and the analytic value; the
MSE for fixed budget `B` as `N_i` varies, with the minimum at `N_i ∼ B^{1/3}`; and the conditional IM
profile against the unconditional proxy of X3, with the corresponding MVA.

**Code / page.** The conditional initial margin is implemented in the fused kernels
(`FusedOptions::imInnerPaths`, `imMarginPeriod`; WGSL in `src/gpu/fused_kernels.cpp`, its CPU mirror in
`src/gpu/fused_reference.cpp`). From every outer path at every reporting date `t` the kernel runs
`N_i` antithetic inner paths of the risk factors to `t + MPOR` (the `t + MPOR` dates are added to the
grid), revalues the netting set on each with the same pricing terms as the outer simulation, and keeps the
16 largest `|V(t + MPOR) − V(t)|` by insertion, so the `q`-quantile (0.99 by default) is read from
order statistics without storing the inner sample. That limits `N_i` to `2..1024` with
`N_i − 1 − ⌊q(N_i − 1)⌋ < 15`. The result is `IM(t, ω)` per path, its mean and 5/50/95% bands, and
MVA = Σ spread · E[D(t) IM(t)] Δt. The XVA analysis (`spec.imMethod`, `"nested"` by default with
`imInnerPaths` = 128, or `"proxy"`) reports the X3 proxy alongside, and [`web/xva.html`](../web/xva.html) plots
both profiles. The unit test `gpu_nested_initial_margin_matches_the_analytic_conditional_quantile`
checks an FX forward against the analytic conditional quantile. The low-biased empirical quantile
(about −4% at 128 inner paths, −1.3% at 512, −0.6% at 1024) is the inner bias discussed above.
The audit of X11 (`FusedOptions::imAuditPaths`) measures and removes that bias on the first paths with
many more inner paths. Not implemented: the Richardson or multilevel bias corrections and the nested
dual bound.

**Further reading.** Gordy and Juneja (2010) "Nested simulation in portfolio risk measurement",
*Management Science* 56(10); Broadie, Du and Moallemi (2011) "Efficient risk estimation via nested
sequential simulation", *Management Science* 57(6); Broadie, Du and Moallemi (2015) "Risk estimation via
regression", *Operations Research* 63(5); Giles and Haji-Ali (2019) "Multilevel nested simulation for
efficient risk estimation", *SIAM/ASA J. Uncertain. Quantif.* 7(2); Andersen and Broadie (2004)
"Primal-dual simulation algorithm for pricing multidimensional American options", *Management Science*
50(9); Lee (1998) *Monte Carlo methods for nested expectations* (PhD thesis); Brigo and Morini (2011)
"Close-out convention tensions", *Risk*, December; Longstaff and Schwartz (2001).

---

## X9. Beyond this implementation (outlook)

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
4. **Conditional initial margin, beyond nesting.** The nested IM per path is implemented (X8); open
   are a regression-based IM (SIMM sensitivities with the Longstaff–Schwartz machinery), the
   full-revaluation historical VaR of CCPs, bias corrections of the nested quantile beyond the audit of
   X11, and the IM that we *receive* with its rehypothecation.
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

## X10. Artificial intelligence and XVA

XVA is expensive for one reason: it is a high-dimensional conditional expectation that must be
re-evaluated for every market move, every trade and every sensitivity. Most uses of AI in XVA attack
that cost, or the human work around it. This chapter classifies the uses, derives the surrogate
approach that is implemented and back tested here, and states the limits honestly.

### X10.1 Where AI touches XVA

| Area | Problem | Typical method | Main risk |
|---|---|---|---|
| Fast revaluation | `XVA(state)` costs seconds to hours | Surrogate: polynomial, neural net, Gaussian process | Extrapolation outside the training domain |
| Exposure simulation | Conditional `V(t)` on each path | Regression (Longstaff–Schwartz), deep regression | Basis bias, see X7 and X8 |
| Sensitivities | Bump-and-revalue is noisy and slow | Differential machine learning, AAD (X9) | Needs pathwise derivatives |
| Scenario generation | Realistic joint risk-factor paths | Generative models (VAE, GAN, diffusion), neural SDEs | Tail fidelity, arbitrage-free conditions |
| Counterparty credit | Default intensity and spread proxies | Gradient boosting, mapping of illiquid names | Data scarcity, bias, fairness |
| Hedging | Optimal CS01 and funding hedges with costs | Deep hedging, reinforcement learning | Reward design, model dependence |
| Calibration | Fit model to market quotes | Neural calibration networks | Stability, regime change |
| Operations | Explain, reconcile, query XVA | LLM agents with tool use | Hallucination, access control |
| Governance | Validation and monitoring | Challenger models, drift detection | Model risk of the AI itself |

### X10.2 Surrogate XVA

Fix the engine of X1–X7 with common random numbers (X6) and a fixed seed. Then the adjustments are
a deterministic, smooth function of the market state `x` (here five inputs: the counterparty and own
spread multipliers, a parallel rate shift, the FX volatility and the funding spread):

```
XVA_k(x) = F_k(x),   k in {CVA, DVA, FCA, FBA, MVA, KVA},   total = CVA - DVA + FCA - FBA + MVA + KVA
```

The MVA of the dataset and of the agent tools uses the unconditional IM proxy of X3 (`imMethod:
"proxy"`), not the nested IM that the XVA page shows by default: the proxy is about a hundred times
cheaper per state, which keeps the 410 training and back-test runs and each tool call fast, and the
surrogates and the tools then share one definition. A nested-IM surrogate is the same exercise with a
noisier, slightly biased target.

Without common random numbers `F_k` carries Monte Carlo noise of order `N^{-1/2}`, which a surrogate
would learn as noise; the fixed seed removes it. A surrogate `F̂_k(x; θ)` is trained on `n` engine
runs at states `x_i` chosen by Latin hypercube sampling over a box `B`:

```
θ* = argmin  (1/n) sum_i || F(x_i) - F̂(x_i; θ) ||²  +  λ ||θ||²
```

Three model classes are implemented in C++ (`include/ccr/ai/surrogate.hpp`, `src/ai/surrogate.cpp`; the MLP gradient runs on the lock-free thread pool):

1. **Linear ridge**: `F̂ = b + w·x`. The first-order Taylor model; it equals the bucketed
   sensitivities of X6 applied to the market move.
2. **Quadratic ridge**: all monomials of degree at most 2. It captures the convexity that matters for
   `V⁺`-type payoffs. Closed-form solution via the normal equations.
3. **MLP**: a 5-16-16-6 network with `tanh` units, standardised inputs and outputs, trained by Adam
   on the full batch, with warm-started refits (walk-forward, below).

Evaluation is a handful of multiply-adds, so the cost ratio to the engine is of order `10^5` or more
(measured below). The price is a bias that must be monitored, not assumed away.

### X10.3 Back-test design

A surrogate is a model and is validated like one. The back test in this repository is
deliberately strict about what it claims:

- **Data**: `examples/xva_ai_dataset.cpp` runs the native engine (2,000 paths, seed 2024) on 160 Latin
  hypercube training states and on a 250-day *synthetic* market path with a stress episode on days
  150–200. The data are synthetic, so the numbers measure surrogate error against the engine, not
  against the market.
- **Metrics**: mean absolute error as basis points of the 25m gross notional (MAE), MAPE, maximum
  error, and the split between days inside the training box and *stress days outside it*.
- **Static versus walk-forward**: static fits once on the training set; walk-forward refits every 21
  business days using only data observed up to that day, so no look-ahead.
- **Baselines**: (a) yesterday's engine value carried forward, (b) a weekly engine run carried
  forward. A surrogate that cannot beat these is not worth running.
- **P&L explain**: for the daily change `ΔXVA`, effectiveness `1 - Var(ΔXVA - ΔF̂)/Var(ΔXVA)`;
  a negative value means the explanation is worse than none. The sign hit rate is the share of days
  on which `ΔF̂` has the sign of `ΔXVA`.

### X10.4 Measured results

Run by `web/js/pages/ai.js` (and by `web/tests/ai.test.mjs` with fewer epochs). MAE in bp of gross
notional; 20 of the 250 days are outside the training box.

| Model | All days | Inside box | Stress days |
|---|---|---|---|
| Yesterday's value | 1.92 | 1.89 | 2.27 |
| Weekly engine run | 2.37 | 2.33 | 2.80 |
| Linear, static | 4.57 | 3.22 | 20.1 |
| Linear, walk-forward | 3.15 | 2.04 | 15.95 |
| Quadratic, static | 0.76 | 0.23 | 6.9 |
| Quadratic, walk-forward | 0.66 | 0.22 | 5.73 |
| MLP, static | 0.35 | 0.32 | 0.60 |
| MLP, walk-forward | 0.14 | 0.11 | 0.45 |

Reading the table:

- The MLP is the most accurate and its error on stress days is not much worse than inside the box. In
  this experiment it extrapolated mildly beyond the box; there is no guarantee that this holds for
  other portfolios or larger excursions.
- The linear and quadratic models degrade sharply outside the box, as the Taylor remainder grows.
- Yesterday's value is a strong baseline in level terms (about 1.9 bp), yet it explains none of
  the daily P&L (effectiveness -115%), because it by construction predicts no change. The surrogates
  earn their place on *changes*, which is what sensitivities and P&L explain need.
- Cost: MLP inference takes about 1.3 µs against about 199.5 ms for one native engine run, a ratio
  above `10^5`, and the one-off training cost amortises after a few hundred evaluations.

### X10.5 Differential machine learning

If the engine can also return pathwise derivatives `∂F/∂x` (adjoints, X9 item 7), train on value *and*
slope: `min Σ ||F - F̂||² + μ ||∇F - ∇F̂||²`. The derivative labels act as many extra equations per
sample, which cuts the number of engine runs needed (Huge and Savine 2020). With CRN bump-and-revalue
(X6) the derivative labels are available approximately, at the price of the bump size error.

### X10.6 Regression and deep learning inside the simulation

The conditional exposure `E[V⁺(t) | X_t]` of X7 is itself a regression; replacing the polynomial basis
by a neural network removes the manual basis choice and scales to many risk factors. The same
bias–variance trade-off as in X8 applies: a network is a flexible basis, but over-fit labels are
biased in the direction that flatters `V⁺`. Use out-of-sample paths for the exposure estimate (the
low/high estimator pairing of X8).

### X10.7 Generative scenarios and deep hedging

Generative models (VAE, GAN, diffusion, neural SDEs) learn the joint distribution of risk factors from
history. Used for XVA they must be checked against the properties the pricing model assumes:
martingale (no-arbitrage) drift under the pricing measure, correct tails and correlation under stress,
and positive-definite dependence. The real-world to risk-neutral link is not provided by the network
and has to be imposed. Deep hedging (Buehler et al. 2019) trains a policy minimising a convex risk
measure of terminal hedging P&L with transaction costs; for XVA the policy hedges the CS01 and funding
sensitivities of X6 with liquid CDS, and the cost of rebalancing is explicit. Results depend on the
simulator that the policy is trained on.

### X10.8 LLM agents and tool use

Large language models cannot be trusted to *compute* XVA, but they can drive the engine. The agent
skill in this repository follows one rule: **the model proposes, the engine computes**. The model sees
four provider-neutral JSON-Schema tools:

| Tool | Engine call | Returns |
|---|---|---|
| `price_xva` | `xva` (IM proxy) | CVA, DVA, FCA, FBA, MVA, KVA, total |
| `xva_sensitivities` | `xva` | bucketed counterparty CS01, own CS01, funding spread 01 |
| `allocate_xva` | `xva` | per-trade CVA, FCA, KVA (Euler allocation of X5) |
| `compare_scenarios` | `xva` x 2 | base versus scenario, differences per adjustment |

`web/js/ai/skill.js` validates every argument (type, range, unknown keys) before the engine runs, and
returns errors as data so that the model can correct itself; adapters emit the same tools in OpenAI,
Anthropic, Gemini and MCP formats; `SKILL_MD` is a ready-to-use skill file in the Agent Skills
format; `runAgentLoop` is a model-agnostic loop with an injected `llm(messages, tools)` callback and
an injected runner. A deterministic harness of 11 cases (valid calls, out-of-range inputs, unknown
tools, scenario comparison, error recovery) runs without any network and passes 11/11; it tests the
*tool layer*. **No accuracy of any real LLM is measured in this repository**; measuring it needs the
user's own model and the harness as a fixture.

Controls that matter in production: read-only tools by default; schema validation as above; a
numeric answer must come from a tool result and be quoted with its inputs; log prompts, tool calls
and results for audit; never let the model set limits, approve trades or edit market data.

### X10.9 Model risk and governance

An AI component is a model under SR 11-7 / the PRA SS1/23 principles. For a surrogate:

1. **Domain of validity**: record the box `B` and flag states outside it (`outsideBox` in the code);
   fall back to the engine there.
2. **Challenger and benchmark**: compare with the engine on a rolling sample, and with the baselines
   above.
3. **Monitoring**: error in bp of notional and P&L-explain effectiveness, with thresholds that trigger a
   refit or a fallback.
4. **Explainability**: for the polynomial models, the coefficients; for a network, local sensitivities
   `∂F̂/∂x` compared with the bucketed CS01 of X6 (a sanity check, not a proof).
5. **Reproducibility**: fixed seeds, versioned datasets, recorded training configuration.
6. **Regulation**: EU AI Act and supervisory guidance on model use; fair-lending style concerns apply
   to learned credit proxies.

### X10.10 Limitations

- Synthetic data: results show the method, not the performance on a live book.
- One portfolio, five factors, one seed: a large book with many factors suffers from the curse of
  dimensionality, so the training set must grow, or the surrogate must be applied to factor-reduced
  states.
- Common random numbers are essential; with changing seeds the labels are noisy.
- Wrong-way risk, collateral thresholds and close-out kinks make `F` less smooth; networks cope, low
  degree polynomials do not.
- The WASM and native engines use different normal generators and agree only to Monte Carlo accuracy,
  so the dataset is generated natively and the page does not mix the two.

### X10.11 Code and page

| Item | Location |
|---|---|
| Dataset generator | `examples/xva_ai_dataset.cpp`, target `xva_ai_dataset`; `./build/xva_ai_dataset > web/js/ai/backtest-data.js` |
| Surrogates and back test | `src/ai/surrogate.cpp` (entry point `surrogateBacktest`) |
| Agent skill, tools, harness | `web/js/ai/skill.js` |
| Page | [`web/ai.html`](../web/ai.html), entry `web/js/pages/ai.js` |
| Node tests | `web/tests/ai.test.mjs` |

**Citations** below are given from memory and have not been checked against the sources.

AI references: Huge and Savine (2020) "Differential machine learning", *Risk*, October (arXiv:2005.02347);
Buehler, Gonon, Teichmann and Wood (2019) "Deep hedging", *Quantitative Finance* 19(8); Ferguson and
Green (2018) "Deeply learning derivatives", SSRN; Hernandez (2016) "Model calibration with neural
networks", *Risk*; Sirignano and Spiliopoulos (2018) "DGM: a deep learning algorithm for solving
partial differential equations", *J. Comput. Phys.* 375; Han, Jentzen and E (2018) "Solving
high-dimensional PDEs using deep learning", *PNAS* 115(34); Henry-Labordère (2017) "Deep primal-dual
algorithm for BSDEs: applications of machine learning to CVA and IM", SSRN; Kondratyev and Schwarz
(2019) "The market generator", SSRN; Federal Reserve SR 11-7 (2011) *Guidance on model risk
management*; Bank of England PRA SS1/23 (2023) *Model risk management principles for banks*.

---

## X11. Adaptive nested Monte Carlo

**Motivation.** X8 leaves adaptive nesting as an outlook: outer scenarios far from the kink need few
inner paths, so a fixed `N_i` per scenario spends most of the budget where it changes nothing. This
chapter builds the adaptive estimator on top of the *non-nested* simulation the library already runs, so
that an existing engine needs only one more ability: to restart its paths from a given state.

**One ordinary simulation is a nest with one inner path.** An outer path from `0` to `T` passes through
`t`. Its own cash flows after `t`, discounted to `t`,

```
Y(t, ω) = Σ_{t_j > t} D(t, t_j; ω) CF_j(ω),        E[ Y(t) | X(t) ] = V(t, X(t))
```

are an unbiased one-sample estimate of the conditional value at that path's state `X(t)`. Far too noisy
scenario by scenario, but a regression across the scenarios removes most of the noise (the
Longstaff–Schwartz idea of X8). The adaptive estimator starts from this regression, measures how wrong
it is, and spends inner paths only on the scenarios whose error could move the statistic.

**The estimator, on one date `t`** (`nested::estimateAdaptive`):

1. *Split.* A seeded shuffle splits the outer scenarios into a training half and an evaluation half
   (`trainingFraction`). The proxy never sees an evaluation scenario's `Y`, so `Y` stays an independent
   sample of that scenario, and the regression's in-sample optimism does not enter.
2. *Proxy.* Least squares of `Y` on probabilists' Hermite polynomials of the standardised features, all
   products up to a total degree (`HermiteProxy`, degree 3 by default, lowered until there are two
   training rows per basis function), with the heteroscedasticity-robust (HC0) variance of each
   prediction:

   ```
   μ(x) = φ(x)ᵀ β̂,   Cov(β̂) = G⁻¹ (Σ_i e_i² φ_i φ_iᵀ) G⁻¹,   v(x) = φ(x)ᵀ Cov(β̂) φ(x),   G = ΦᵀΦ + λ I
   ```

   `v` covers finite training data only, not the misspecification of the basis.
3. *Bins.* The evaluation scenarios are binned by proxy value at the probabilities 0.5%, 1%, 2.5%, 5%,
   10%, 25%, 50%, 75%, 90%, 95%, 97.5%, 99%, 99.5% (14 bins, fine in both tails, where PFE and IM live).
   The training residuals give the payoff noise `r_b²` per bin.
4. *Audit.* A few random scenarios per bin (16, at most a fifth of the budget) get inner paths whatever
   the statistic. On every scenario with enough inner samples, `d_i = m̄_i − μ_i` (inner mean minus proxy)
   measures the proxy's error. Per bin, the mean of `d` is the proxy's *systematic* bias `δ_b`, which
   shifts the proxy, and the rest of its spread is the proxy's error variance:

   ```
   δ_b = mean(d_i),     τ_b² = max(0, mean((d_i − δ_b)²) − mean(s_i²/n_i + v_i)),     σ_p,i² = v_i + τ_b²
   ```

   Both are re-estimated after every refinement round from all scenarios with inner paths: those are
   chosen from proxy values and uncertainties, never from their unseen inner results, so within a bin
   they measure the proxy's error without selection bias, where the decisions are made.
5. *Blend.* Each scenario's estimate weighs the shifted proxy and its own pooled sample mean (the outer
   payoff and the inner paths) by precision:

   ```
   Ṽ_i = μ'_i + w_i (m̄_i − μ'_i),   w_i = σ_p,i² / (σ_p,i² + s_i²/n_i),   Var(Ṽ_i) = σ_p,i² (s_i²/n_i) / (σ_p,i² + s_i²/n_i)
   ```

   with `μ'_i = μ_i + δ_b`. Where the proxy is precise the estimate stays close to it (it pools all
   training paths: with a payoff standard deviation of 10 and a proxy error of 0.1, matching the proxy
   takes 10,000 inner paths); where the audit found it wrong, the inner mean takes over. The *refinement
   gain* `G_i = 1 − Var(Ṽ_i at the cap) / Var(Ṽ_i now)` says how much of the error more inner paths
   could remove (`maxInnerPerScenario` = 4,096).
6. *Two folds.* Every inner batch is split: its first half goes to fold 0 (with the outer payoff), its
   second half to fold 1. A statistic that *selects* scenarios (a sign, a tail) and then *averages* them
   selects with the proxy and fold 0 and values with fold 1, so the samples that put a scenario in never
   also value it. Without the split, `E[max(m̄, 0)] > max(V, 0)` near zero by about 0.4 standard errors (a
   winner's curse), and refining the scenarios near the kink *raises* EE instead of correcting it.
7. *Refine.* Each statistic says which scenarios it wants refined; each round takes the requests of the
   statistics in turns (so one statistic's urgencies cannot crowd out another's), up to 2,000
   scenarios, one batch of 64 inner paths each, until no statistic asks, the budget is spent, the
   requesting scenarios are all at the cap, or 50 rounds have run. Evaluation scenarios whose features lie
   outside the training range are refined first: a polynomial is least reliable where it extrapolates.

**Refinement rules.** With `e(g, se) = g Φ(−g/se)`, the expected error of a scenario at distance `g` from a
cutoff (it moves the statistic by about `g`, with the probability of being on the wrong side):

- `PositivePart` (EE, and EE* with the weight `w_i = D(0, t)`): `EE = (1/n) Σ_i w_i 1{Ṽ_i^sel > 0} V̂_i^val`.
  Half the tolerance (1% of EE, or 0.1% of the size of the discounted cash flows) goes to the sampling
  noise `√(Σ w_i² Var V̂_i^val) / n` of the values counted, refining the noisiest to an equal share of it;
  half to the expected misclassification `Σ_i w_i e(|Ṽ_i|, se_i) / n`, refining the largest
  reducible contributions `e_i G_i` (skipping `G_i < 0.05`) until they cover the excess.
- `Quantile` (PFE, a quantile of `max(V, 0)`): refine scenario `i` while
  `G_i Φ(−|Ṽ_i − q̂| / se_i) > Φ(−2)`, i.e. while it may be on the wrong side of the quantile and more inner
  paths would remove a useful part of its error. The probability is at most ½, so refinement stops by
  itself once `G_i < 0.046`.

**The netting set** (`nested::nestedExposure`). The features are the *Markov* state at `t`: the
Hull–White `x(t)`, the log spot of every asset a live trade refers to, and the fixing of every coupon
fixed before `t` and paid after it (its value no longer follows from `x(t)`). The outer simulation is the
library's `ScenarioGenerator` on the reporting dates merged with every fixing and payment date; the
inner paths restart the Hull–White and log-normal models from a path's state with their *exact*
transitions between the cash-flow dates, correlated as the outer simulation, with counter-based normals
seeded by `(date, scenario, batch)`, so the result is the same on any number of threads (tested).
Swaps, asset forwards and European options are nested; all three have closed forms on the scenarios,
so EE* and PFE are compared with the exact statistic *on the same evaluation scenarios*, which removes
the outer sampling error from the comparison. (The option's closed form ignores the rate volatility in
the forward's volatility; the nest simulates the full dynamics.) At `t = 0` every path has the same
state and the estimate is plain Monte Carlo on the outer payoffs.

**Conditional initial margin: the audit.** For MVA the adaptive loop reduces to its audit. MVA is
linear in `IM(t, ω)`: no sign or tail decides which scenarios count, so no scenario is more urgent than
another, and the noise of the per-path IM averages out over the paths. What does not average out is the
bias of the per-path estimator, the empirical 99% quantile of `N_i` inner samples, which is low by
`O(1/N_i)` (X8: about −4% at 128 inner paths, −9% at 64). The audit measures it on the plan's first `M`
paths with `N_a > N_i` inner paths, the kernel's own samples extended (the kernel's normals are keyed by
path, date and inner index, so the two estimates are paired), and scales IM per reporting date:

```
c(t) = E_M[ D(t) IM_{N_a}(t) ] / E_M[ D(t) IM_{N_i}(t) ],      IM*(t) = c(t) E[ D(t) IM_{N_i}(t) ]
```

`summarise` runs it on the CPU (`gpu::auditConditionalIm`); the fused kernels and their WGSL are
unchanged, so the GPU and the CPU kernels produce the same read-backs and the same correction.

**Validation** (`tests/test_nested.cpp`, `tests/test_gpu.cpp`, `tests/test_app.cpp`; native runs).

| Case | Regression only | Classical nest, same inner paths | Adaptive |
|---|---|---|---|
| Default portfolio, EE*, 2,000 outer paths (1,000 evaluated), 60 inner paths per evaluated scenario, semiannual dates to 10y, 5 seeds | 3.9% | 1.2% | 0.74% |
| The same, PFE 95% | 3.7% | 1.7% | 0.96% |
| The same with a degree-1 proxy (3 seeds), EE* / PFE | 3.9% / 5.3% | 1.4% / 1.7% | 0.74% / 0.75% |
| 30 inner paths per scenario (3 seeds), EE* / PFE | 4.3% / 4.6% | 2.7% / 2.8% | 3.1% / 1.3% |

Relative root-mean-square error against the closed form over the dates. The default portfolio is the one
of the web front end (10y payer and 5y receiver swaps, a 3y FX forward and a short 2y FX call; Hull–White
and EURUSD, correlation 0.2). The toy unit test with a quadratic `V` and a linear proxy shows the same
pattern in a sharper form: the regression misses the 95% quantile by more than 15%, the adaptive nest is
within 2% and closer than the classical nest with the same inner paths.

| Initial margin (FX forward, analytic conditional quantile) | Kernel `N_i` | Before the audit | After the audit |
|---|---|---|---|
| Audit 256 paths × 1,024 inner paths | 64 | −9.3% | −0.4% |
| Audit 256 paths × 1,024 inner paths | 128 | −4.0% | −0.6% |
| Audit 128 paths × 512 inner paths | 64 | −9.3% | −0.8% |
| Audit 128 paths × 512 inner paths | 128 | −4.0% | −1.0% |

Mean relative error of E[D IM] over the dates before maturity (2,000 paths). The audit's own `N_a`
keeps a small bias of its own (about −1% at 512 inner paths). On the default XVA
specification (128 inner paths, 59 reporting dates) the audit raises MVA by 2.6%, from 14,371 to 14,748
with 32 audit paths, 14,782 with 64 and 14,775 with 128, for 0.5 s, 1.6 s and 4.2 s on one thread on top
of a 10.4 s CPU-kernel run.

**What the tests show, and their limits.**

- The adaptive nest wins where the statistic depends on a *decision* (PFE's tail, EE's sign) and where
  the proxy is misspecified: it measures the bias and corrects it, while the regression alone keeps it.
- With a small budget, EE* is not better than the classical nest: the values of the scenarios it does
  not refine come from their single outer payoff, which is noisy, while the classical nest averages a few
  inner paths for every scenario. In the tests the crossover was between 30 and 60 inner paths per
  evaluated scenario; PFE gained at both.
- The bias is modelled per bin of proxy values. A proxy wrong in one corner of a high-dimensional state
  can be missed if no audited or refined scenario lands there.
- The quantile of blended estimates keeps a small `O(Var)` nested bias, and whether a scenario gets
  another batch depends on its earlier samples (an optional-stopping effect); neither was visible in the
  tests.
- The IM audit corrects the mean (MVA) per date; it assumes the relative bias is the same on every path
  of a date, which holds for location-scale value changes and less well for books of options far from
  and close to the money. The IM bands are scaled by the same factor.

**Code / page.** `include/ccr/nested/adaptive_nested.hpp` (the estimator: `HermiteProxy`,
`ScenarioEstimates`, `PositivePart`, `Quantile`, `estimateAdaptive`, `estimateUniform`),
`include/ccr/nested/nested_exposure.hpp` (`nestedExposure`), the IM audit in
`include/ccr/gpu/fused_exposure.hpp` (`FusedOptions::imAuditPaths`, `imAuditInnerPaths`,
`auditConditionalIm`). The front end's entry point `nestedExposure` (spec.nested: `dates`, `numPaths`,
`innerPerScenario`, `degree`) runs on the CPU in the worker: the nest is not a fused-kernel job. The
XVA analysis audits the conditional IM with `spec.imAudit` (`imAuditPaths` = 32, `imAuditInnerPaths` =
512 by default). [`web/nested.html`](../web/nested.html) (menu: Exposure → Adaptive nested Monte Carlo)
shows EE* and PFE by the closed form, the regression, the classical nest and the adaptive nest, their
errors, the CVA of each and, per date, where the inner paths went and why the loop stopped; its settings
are the `spec.nested` fields above. [`web/xva.html`](../web/xva.html) turns the IM audit on and shows its
ratio per date and the MVA before and after it.

**Further reading.** Broadie, Du and Moallemi (2011) "Efficient risk estimation via nested sequential
simulation", *Management Science* 57(6); Broadie, Du and Moallemi (2015) "Risk estimation via
regression", *Operations Research* 63(5); Gordy and Juneja (2010); Liu and Staum (2010) "Stochastic
kriging for efficient nested simulation of expected shortfall", *Journal of Risk* 12(3); Hong, Juneja
and Liu (2017) "Kernel smoothing for nested estimation with application to portfolio risk
measurement", *Operations Research* 65(3); Giles and Haji-Ali (2019); Chernozhukov et al. (2018)
"Double/debiased machine learning for treatment and structural parameters", *The Econometrics
Journal* 21(1) (sample splitting against selection bias).

---

## References

- Andersen, L., Duffie, D., Song, Y. (2019). Funding value adjustments. *Journal of Finance* 74(1).
- Andersen, L., Broadie, M. (2004). Primal-dual simulation algorithm for pricing multidimensional American options. *Management Science* 50(9).
- Andersen, L., Pykhtin, M., Sokol, A. (2017). Rethinking the margin period of risk. *Journal of Credit Risk* 13(1).
- Brigo, D., Mercurio, F. (2006). *Interest Rate Models: Theory and Practice*. Springer.
- Brigo, D., Morini, M., Pallavicini, A. (2013). *Counterparty Credit Risk, Collateral and Funding*. Wiley.
- Broadie, M., Du, Y., Moallemi, C. (2011). Efficient risk estimation via nested sequential simulation. *Management Science* 57(6); (2015) Risk estimation via regression. *Operations Research* 63(5).
- Brigo, D., Morini, M. (2011). Close-out convention tensions. *Risk*, December.
- Burgard, C., Kjaer, M. (2011). In the balance. *Risk*, October; (2013) Funding strategies, funding costs. *Risk*, December.
- Capriotti, L., Giles, M. (2012). Algorithmic differentiation: adjoint Greeks made easy. *Risk*, September.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal* 21(1).
- Caspers, P., Giltinan, P., Lichters, R., Mazzoni, N. (2017). Forecasting initial margin requirements – a model evaluation. *Journal of Financial Risk Management*.
- Fujii, M., Shimada, Y., Takahashi, A. (2010). A note on construction of multiple swap curves with and without collateral. CARF working paper.
- Giles, M., Haji-Ali, A.-L. (2019). Multilevel nested simulation for efficient risk estimation. *SIAM/ASA Journal on Uncertainty Quantification* 7(2).
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Gordy, M., Juneja, S. (2010). Nested simulation in portfolio risk measurement. *Management Science* 56(10).
- Green, A. (2015). *XVA: Credit, Funding and Capital Valuation Adjustments*. Wiley.
- Green, A., Kenyon, C., Dennis, C. (2014). KVA: capital valuation adjustment. *Risk*, December.
- Gregory, J. (2020). *The xVA Challenge*, 4th ed. Wiley.
- Henry-Labordère, P. (2012). Cutting CVA's complexity. *Risk*, July.
- Hong, L. J., Juneja, S., Liu, G. (2017). Kernel smoothing for nested estimation with application to portfolio risk measurement. *Operations Research* 65(3).
- Hull, J., White, A. (2012). The FVA debate. *Risk*, July; (2014) Valuing derivatives: funding value adjustments and fair value. *Financial Analysts Journal* 70(3).
- Kenyon, C., Green, A. (2013). Regulatory costs break risk neutrality. *Risk*, September.
- Liu, M., Staum, J. (2010). Stochastic kriging for efficient nested simulation of expected shortfall. *Journal of Risk* 12(3).
- Longstaff, F., Schwartz, E. (2001). Valuing American options by simulation. *Review of Financial Studies* 14(1).
- Pallavicini, A., Perini, D., Brigo, D. (2012). Funding, collateral and hedging: uncovering the bilateral risk. arXiv:1210.3811.
- Piterbarg, V. (2010). Funding beyond discounting: collateral agreements and derivatives pricing. *Risk*, February.
- Pykhtin, M., Rosen, D. (2010). Pricing counterparty risk at the trade level and CVA allocations. *Journal of Credit Risk* 6(4).
- Tasche, D. (2008). Capital allocation to business units and sub-portfolios: the Euler principle. In *Pillar II in the New Basel Accord*.
