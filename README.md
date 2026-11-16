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

- **Series**: FTSE Italy All Share Banks index, daily closing prices.
- **Source**: [Investing.com](https://www.investing.com/).
- **Period**: 02/01/2019 – 30/12/2022 (4 years, ~1,019 observations).
- **Why this window is interesting**: it contains two major volatility
  regimes for the Italian banking sector:
  - the **Covid-19 crash** (March 2020): −33% in 16 trading days;
  - the **Ukraine invasion crash** (February 2022): −32% drawdown after a
    +107% bull run from the Covid lows.

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
   with the ARMA(3,3) mean equation held fixed and a normal conditional
   distribution.
9. **Final GARCH(1,1) fit, diagnostics, and forecasting.**
10. **ARMA vs. GARCH comparison** on likelihood, information criteria, and
    forecast accuracy.

## Key results

### Mean model: ARMA(3,3)

Grid search over ARMA(p,q), p, q ∈ {0,…,3} by AIC selected **ARMA(3,3)**
as the best mean model. Fitted by maximum likelihood:

| Coefficient | Estimate | Significant (α = 0.05)? |
|---|---|---|
| φ1 | −0.4748 | yes (p = 0.0150) |
| φ2 |  0.2408 | no |
| φ3 |  0.6416 | yes (p < 0.0001) |
| θ1 |  0.4824 | yes (p = 0.0105) |
| θ2 | −0.1291 | no |
| θ3 | −0.5926 | yes (p < 0.0001) |

Residual diagnostics: no major remaining autocorrelation, but residuals
are clearly **non-normal** (leptokurtic, negatively skewed; Shapiro-Wilk
rejects normality) and show strong **conditional heteroskedasticity**
(ARCH-LM test statistic 312.40, rank test 179.32, both p ≈ 0). This is
the statistical motivation for extending the model with GARCH.

### Volatility model: GARCH(1,1)

Grid search over GARCH(p,q), p, q ∈ {1,2} (ARMA(3,3) mean kept fixed)
selected **GARCH(1,1)** as the best specification (AIC 4104.92, BIC
4154.18), consistent with the well-known result of Hansen & Lunde (2005)
that "nothing beats a GARCH(1,1)" for most financial series. All ten
coefficients of the final ARMA(3,3)-GARCH(1,1) model are significant at
α = 0.05, and diagnostic tests (weighted Ljung-Box on residuals and
squared residuals, ARCH-LM, Pearson goodness-of-fit) mostly support the
adequacy of the fit.

### ARMA(3,3) vs. GARCH(1,1)

| Metric | ARMA(3,3) | GARCH(1,1) |
|---|---|---|
| Log-likelihood | −2197.95 | **−2042.46** |
| AIC | 4409.90 | **4104.92** |
| BIC | 4444.38 | **4154.18** |
| RMSE (in-sample) | **2.09617** | 2.10958 |
| Forecasting RMSE | 1.4658 | **1.43816** |

**GARCH(1,1) wins** on every likelihood-based criterion (log-likelihood,
AIC, BIC) by explicitly modeling the conditional heteroskedasticity that
the plain ARMA ignores, and it also achieves a slightly lower
out-of-sample forecasting RMSE — despite ARMA(3,3) having a marginally
lower in-sample RMSE. This is consistent with the financial-econometrics
literature (Engle 1982, Bollerslev 1986, Hansen & Lunde 2005).

## Lesson learned: in-sample fit ≠ forecasting skill

The honest, slightly humbling result of this project: when the ARMA(3,3)
point forecast was checked against the **actual realized returns for
early January 2023** (true out-of-sample data, never used in fitting or
model selection), it got **most of the directional signs wrong**. A model
that passes its residual diagnostics, has significant coefficients, and
minimizes AIC within its class is not thereby a reliable trading or
directional forecasting tool.

This is an important and often underappreciated distinction in applied
time series work: **in-sample statistical adequacy (good residual
diagnostics, low information criteria) says a model describes the data
it was fit on reasonably well — it does not say the model can predict
the future.** Financial returns are notoriously close to unpredictable in
direction (even if their volatility is forecastable, which is exactly
what the GARCH component is good at), and a 10-step-ahead point forecast
from a mean-reverting ARMA process should be interpreted with that caveat
front and center. GARCH(1,1) improves on volatility forecasting and
overall fit, but neither model should be mistaken for "a reliable model
for any financial expert" trying to bet on direction.

## Repository contents

- `garch_arima_ftse_banks.R` — full analysis script: data loading,
  returns transform, ADF stationarity test, ARMA order selection by AIC
  grid search, ARMA(3,3) fit and diagnostics, ARMA forecast, GARCH order
  selection by grid search, final GARCH(1,1) fit and diagnostics, GARCH
  forecast, and the ARMA-vs-GARCH comparison.
- `LICENSE` — MIT license.
- `.gitignore` — standard R/RStudio ignores.

## Tools / libraries

R, with `forecast`, `tseries`, `rugarch`, and `FinTS`.

## References

- Engle, R. F. (1982). *Autoregressive Conditional Heteroscedasticity with
  Estimates of the Variance of United Kingdom Inflation*. Econometrica.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional
  Heteroskedasticity*. Journal of Econometrics.
- Hansen, P. R., & Lunde, A. (2005). *A Forecast Comparison of Volatility
  Models: Does Anything Beat a GARCH(1,1)?* Journal of Applied
  Econometrics.
