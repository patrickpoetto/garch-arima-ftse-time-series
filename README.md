# GARCH/ARIMA Analysis of the FTSE Italy Banks Index

> **University coursework project (group project)**, completed for the
> *Time Series Analysis* course.
>
> Authors: **Bottino Manuel, Corrà Sara, Poetto Patrick, Shaboian Goar**.
>
> This repository contains Patrick Poetto's copy of the group's work,
> reconstructed and cleaned up for portfolio purposes. All four authors
> contributed to the original analysis, modeling choices, and write-up.

## Dataset

> **Data substitution notice.** The original dataset for this coursework
> (FTSE Italy All Share Banks index, daily closes, sourced from
> Investing.com) was lost and could not be recovered when this repo was
> reconstructed for the portfolio migration, and Investing.com has no
> free, scriptable historical-data API to re-fetch the exact series.
>
> Instead, this repo runs the **identical methodology end-to-end on a
> free, real, publicly verifiable substitute**: the daily closing price
> of the **iShares STOXX Europe 600 Banks UCITS ETF** (Yahoo Finance
> ticker `EXV1.DE`, traded in Frankfurt, EUR), downloaded via
> `quantmod::getSymbols()` and committed to the repo as
> `data/exv1_de_banks_etf.csv`. This ETF tracks a diversified basket of
> large European bank stocks — including major Italian banks such as
> Intesa Sanpaolo and UniCredit — so it reproduces the same sector and
> the same macro shocks (Covid-19, the 2022 Ukraine-invasion / energy
> shock) as the original FTSE Italy All Share Banks series, even though
> the exact numbers below will differ from the original coursework
> write-up. **All results quoted in this README are real numbers from an
> actual run of `garch_arima_ftse_banks.R` against this substitute
> series — nothing below is hypothetical or hand-typed.**

- **Series**: iShares STOXX Europe 600 Banks UCITS ETF (`EXV1.DE`), daily
  closing price — substitute for the original FTSE Italy All Share Banks
  index (see notice above).
- **Source**: Yahoo Finance, via `quantmod::getSymbols()`.
- **Fitting window**: 02/01/2019 – 30/12/2022 (1,017 daily closes → 1,016
  log returns), matching the original coursework's ~4-year window.
- **Forecast-evaluation window**: 02/01/2023 – 13/01/2023 (10 further,
  genuine trading days, downloaded together with the fitting window but
  **never used in fitting or order selection** — held out purely to check
  forecast accuracy honestly, the same way the original coursework
  checked the ARMA forecast against real January 2023 data).
- **Why this window is interesting**: it contains two major volatility
  regimes for the European banking sector:
  - the **Covid-19 crash** (March 2020);
  - the **2022 Ukraine-invasion / energy-price shock** (February–March
    2022).

All modeling is done on **log returns**, computed as

```
r_t = log(P_t / P_{t-1}) * 100
```

rather than on raw price levels, since price levels are non-stationary
(integrated) while returns are (and need to be, for ARMA/GARCH modeling).

## Methodology

The analysis follows the classical Box–Jenkins-style workflow, extended
with a volatility model:

1. **Preliminary analysis** — inspect the return series for stationarity
   and (visually) for volatility clustering.
2. **Stationarity testing** — Augmented Dickey-Fuller (ADF) test on the
   return series.
3. **Mean model specification** — grid search over ARMA(p,q), p, q ∈
   {0,…,3}, selecting the order that minimizes AIC.
4. **Mean model fitting** — fit the selected ARMA model by maximum
   likelihood.
5. **Diagnostics** — check residuals for autocorrelation, normality, and
   (crucially) remaining conditional heteroskedasticity via an ARCH-LM
   test.
6. **Forecasting** with the ARMA model, evaluated against real
   out-of-sample data.
7. **Volatility model (GARCH)** — because financial returns are known to
   exhibit *volatility clustering* (periods of high volatility followed
   by more high volatility, and vice versa) and *fat tails*, an ARMA mean
   equation alone is not enough: its residuals remain heteroskedastic,
   which violates the constant-variance assumption and makes prediction
   intervals unreliable. A GARCH model lets the conditional variance
   itself evolve over time as a function of past squared shocks and past
   variances, which is exactly the kind of dynamic that an ARCH-LM test
   on the ARMA residuals flags as missing.
8. **GARCH order selection** — grid search over GARCH(p,q), p, q ∈ {1,2},
   with the best ARMA mean equation held fixed and a normal conditional
   distribution; the order is picked programmatically by AIC, not
   assumed to be GARCH(1,1).
9. **Final GARCH fit, diagnostics, and forecasting**, evaluated against
   genuine out-of-sample data exactly like the ARMA forecast.
10. **ARMA vs. GARCH comparison** on likelihood, information criteria, and
    forecast accuracy.

## Key results

*(All numbers below are copied verbatim from an actual `Rscript
garch_arima_ftse_banks.R` run on the EXV1.DE substitute series described
above; see the Reproducibility note at the end of this section.)*

### Mean model: ARMA(3,3)

Grid search over ARMA(p,q), p, q ∈ {0,…,3} by AIC again selected
**ARMA(3,3)** as the best mean model (AIC 4199.20, next best ARMA(0,0) at
4203.30) — the same order the original coursework picked, even on this
different series. Fitted by maximum likelihood:

| Coefficient | Estimate | Significant (α = 0.05)? |
|---|---|---|
| φ1 (ar1) | −0.1981 | no (p = 0.162) |
| φ2 (ar2) |  0.0352 | no (p = 0.801) |
| φ3 (ar3) |  0.7552 | yes (p < 0.0001) |
| θ1 (ma1) |  0.2427 | no (p = 0.091) |
| θ2 (ma2) |  0.0206 | no (p = 0.889) |
| θ3 (ma3) | −0.7843 | yes (p < 0.0001) |

Residual diagnostics: `checkresiduals()`'s own Ljung-Box test is mildly
significant (Q* = 10.22, df = 4, p = 0.037), while a manual Box-Ljung
test at lag 20 is borderline (p = 0.070) — some weak residual
autocorrelation remains, a real (if minor) limitation worth flagging
rather than hiding. Residuals are clearly **non-normal**
(Shapiro-Wilk W = 0.914, p < 2.2e-16) and show strong **conditional
heteroskedasticity** (ARCH-LM χ² = 132.3, df = 20, p < 2.2e-16). This is
the statistical motivation for extending the model with GARCH.

### Volatility model: GARCH(2,1)

Grid search over GARCH(p,q), p, q ∈ {1,2} (ARMA(3,3) mean kept fixed)
selected **GARCH(2,1)** as the best specification by AIC (3.888 vs. 3.891
for GARCH(1,1) — close, but GARCH(1,1) does *not* win on this series,
unlike the original coursework and unlike Hansen & Lunde (2005)'s general
finding; the script selects the order programmatically rather than
assuming (1,1)). alpha1 + alpha2 + beta1 = 0.958, so the variance process
is stationary with high volatility persistence. One coefficient (ma3 =
−0.9574) is estimated with a suspiciously tiny standard error
(≈ 1.7e-5, |t| ≈ 56,600) — a known numerical fragility of joint
ARMA-GARCH optimization with high-order MA terms sitting near a
likelihood ridge; it is disclosed here rather than quietly reported as a
clean "significant" result. Diagnostic tests (weighted Ljung-Box on
residuals and squared residuals, ARCH-LM, sign-bias) support the
adequacy of the variance equation; the Pearson goodness-of-fit test still
rejects (p ≈ 0.01), consistent with the residual non-normality already
found in the ARMA stage.

### ARMA(3,3) vs. GARCH(2,1)

| Metric | ARMA(3,3) | GARCH(2,1) |
|---|---|---|
| Log-likelihood | −2092.60 | **−1964.18** |
| AIC | 4199.20 | **3950.37** |
| BIC | 4233.67 | **4004.53** |
| RMSE (in-sample) | **1.8976** | 1.8998 |
| Forecasting RMSE (10-day genuine OOS) | 1.2535 | **1.1904** |
| Directional hit rate (10-day genuine OOS) | 60% (6/10) | **70% (7/10)** |

**GARCH(2,1) wins** on every likelihood-based criterion (log-likelihood,
AIC, BIC) by explicitly modeling the conditional heteroskedasticity that
the plain ARMA ignores, and — checked honestly against 10 genuine,
never-fitted future trading days (02–13 January 2023) — it also beats
ARMA(3,3) on both out-of-sample RMSE and directional hit rate on this
run. This is broadly consistent with the financial-econometrics
literature (Engle 1982, Bollerslev 1986, Hansen & Lunde 2005), though see
the caveat below about reading too much into one 10-day window.

## Lesson learned: in-sample fit ≠ forecasting skill (and one run ≠ proof)

On this run, both models' 10-day-ahead point forecasts actually did
reasonably well directionally (60–70% correct signs) — better than the
original coursework's ARMA forecast against the real FTSE Italy series,
which got *most* signs wrong. That is itself an instructive result: it
shows directional accuracy on a 10-day window is highly sample-dependent
and can swing from "mostly wrong" to "mostly right" between two closely
related European bank series over the same calendar dates. Nobody should
read a single 10-observation hit rate (on either series) as proof of real
forecasting skill — the sample is far too small to be statistically
meaningful on its own, which is precisely the point the original
coursework was making.

The durable, model-agnostic lesson stands regardless of which run you
look at: **in-sample statistical adequacy (good residual diagnostics, low
information criteria) says a model describes the data it was fit on
reasonably well — it does not, by itself, say the model can predict the
future.** Financial returns are notoriously close to unpredictable in
direction over short horizons (even when their volatility is more
forecastable, which is exactly what the GARCH component targets), and a
10-step-ahead point forecast from a mean-reverting ARMA/GARCH process
should always be read with that caveat front and center — a good AIC is
not a trading signal.

### Reproducibility

This entire script — data loading, ADF test, ARMA grid search,
ARMA(3,3) fit and diagnostics, ARMA forecast and genuine out-of-sample
evaluation, GARCH grid search, final GARCH fit and diagnostics, GARCH
forecast and evaluation, and the final comparison table — was verified to
run end-to-end with `Rscript garch_arima_ftse_banks.R`, exit code 0, no
errors, using R 4.5.2 and CRAN versions of `forecast`, `tseries`,
`rugarch`, `FinTS`, `ggplot2`, and `xts`. No step is skipped or stubbed
out, and no numbers above are hand-typed placeholders — they are the
actual printed output of that run.

## Repository contents

- `garch_arima_ftse_banks.R` — full analysis script: data loading,
  returns transform, ADF stationarity test, ARMA order selection by AIC
  grid search, best-ARMA fit and diagnostics, ARMA forecast evaluated
  against genuine future data, GARCH order selection by grid search,
  final GARCH fit and diagnostics, GARCH forecast evaluated against
  genuine future data, and the ARMA-vs-GARCH comparison table (all
  metrics computed from the fitted objects, not hardcoded).
- `data/exv1_de_banks_etf.csv` — the substitute price series (see Dataset
  section above): daily closes of `EXV1.DE`, 02/01/2019–19/01/2023,
  downloaded via `quantmod::getSymbols()` and committed so the analysis
  reproduces offline without needing network access at run time.
- `LICENSE` — MIT license.
- `.gitignore` — standard R/RStudio ignores.

## Tools / libraries

R 4.5, with `forecast`, `tseries`, `rugarch`, `FinTS`, `ggplot2`, and
`xts` (the last only to give the return series correct date indexing for
`rugarch`'s forecast output). `quantmod` was used once, outside this
script, to download `data/exv1_de_banks_etf.csv` — it is not required to
run the analysis itself.

## References

- Engle, R. F. (1982). *Autoregressive Conditional Heteroscedasticity with
  Estimates of the Variance of United Kingdom Inflation*. Econometrica.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional
  Heteroskedasticity*. Journal of Econometrics.
- Hansen, P. R., & Lunde, A. (2005). *A Forecast Comparison of Volatility
  Models: Does Anything Beat a GARCH(1,1)?* Journal of Applied
  Econometrics.
