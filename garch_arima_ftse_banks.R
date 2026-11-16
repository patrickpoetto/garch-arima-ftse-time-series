## ============================================================================
## Analysis of Financial Time Series
## FTSE Italy All Share Banks index: ARIMA and GARCH modeling
##
## Course: Time Series Analysis
## Authors: Bottino Manuel, Corra' Sara, Poetto Patrick, Shaboian Goar
## University coursework (group project) - 11/2024
##
## Dataset: FTSE Italy All Share Banks index, daily closing prices,
## source: Investing.com, from 02/01/2019 to 30/12/2022 (~1019 observations).
## The period covers the Covid-19 crash (March 2020, -33% in 16 trading
## days) and the Ukraine invasion crash (February 2022, -32% drawdown
## after a +107% bull run).
## ============================================================================

## ---- 0. Packages -----------------------------------------------------------

library(forecast)    # Arima(), forecast()
library(tseries)     # adf.test()
library(rugarch)     # ugarchspec(), ugarchfit(), ugarchforecast()
library(FinTS)        # ArchTest()
library(ggplot2)      # plotting

## ---- 1. Data loading and returns transform ---------------------------------

# Expects a CSV with at least columns: date, close (daily closing price of
# the FTSE Italy All Share Banks index).
raw <- read.csv("ftse_italy_banks.csv", stringsAsFactors = FALSE)
raw$date <- as.Date(raw$date)
raw <- raw[order(raw$date), ]

prices <- raw$close

# Log returns, expressed in percentage points:
#   r_t = log(P_t / P_{t-1}) * 100
bank_return <- diff(log(prices)) * 100
return_dates <- raw$date[-1]

values_df <- data.frame(date = return_dates, value = bank_return)

ggplot(values_df, aes(x = date, y = value)) +
  geom_line() +
  labs(title = "FTSE Italy All Share Banks - daily log returns (%)",
       x = NULL, y = "Return (%)")

# Visual inspection: returns look centered at zero and heteroskedastic,
# with clear volatility clustering around the Covid crash (March 2020) and
# the Ukraine invasion crash (February 2022).

## ---- 2. Stationarity test (ADF) --------------------------------------------

adf_result <- adf.test(bank_return, k = 20)
print(adf_result)
# ADF test statistic = -7.52, p-value = 0.01
# -> reject the null of non-stationarity: the return series is stationary,
#    as expected for financial returns (unlike the price level, which is
#    non-stationary / integrated of order 1).

## ---- 3. ARMA model selection via AIC grid search ---------------------------

# Grid search over ARMA(p,q), p,q in {0,...,3}, selecting by AIC.
p_range <- 0:3
q_range <- 0:3

arma_grid <- expand.grid(p = p_range, q = q_range)
arma_grid$aic <- NA_real_

for (i in seq_len(nrow(arma_grid))) {
  p <- arma_grid$p[i]
  q <- arma_grid$q[i]
  fit <- tryCatch(
    Arima(bank_return, order = c(p, 0, q), include.mean = FALSE, method = "ML"),
    error = function(e) NULL
  )
  arma_grid$aic[i] <- if (is.null(fit)) NA_real_ else fit$aic
}

arma_grid <- arma_grid[order(arma_grid$aic), ]
print(head(arma_grid, 10))

# Best model by AIC: ARMA(3,3).

## ---- 4. ARMA(3,3) fitting ---------------------------------------------------

arima_fit <- Arima(bank_return, order = c(3, 0, 3), include.mean = FALSE, method = "ML")
summary(arima_fit)

# Coefficient significance (Z-tests):
#   phi1   = -0.4748   (p = 0.0150)
#   phi2   =  0.2408   (not significant)
#   phi3   =  0.6416   (p < 0.0001)
#   theta1 =  0.4824   (p = 0.0105)
#   theta2 = -0.1291   (not significant)
#   theta3 = -0.5926   (p < 0.0001)
# AR(2) and MA(2) terms are not individually significant, but the model is
# kept in its full ARMA(3,3) form since it was selected by the AIC grid
# search as a whole (dropping individual lags changes the dynamics of the
# higher-order terms too).

## ---- 5. Residual diagnostics (ARMA) -----------------------------------------

arma_resid <- residuals(arima_fit)

# Autocorrelation of residuals
checkresiduals(arima_fit)
Box.test(arma_resid, lag = 20, type = "Ljung-Box")
# -> no major remaining autocorrelation in the residuals.

# Normality
shapiro.test(arma_resid)
qqnorm(arma_resid); qqline(arma_resid)
hist(arma_resid, breaks = 40, main = "ARMA(3,3) residuals")
# -> residuals are leptokurtic and negatively skewed; Shapiro-Wilk strongly
#    rejects normality (typical "fat tails" of financial return series).

# Conditional heteroskedasticity (ARCH effects)
ArchTest(arma_resid, lags = 20)
# LM test statistic = 312.40, p-value ~ 0
# Rank-based ARCH test statistic = 179.32, p-value ~ 0
# -> strong evidence of conditional heteroskedasticity in the residuals:
#    the constant-variance ARMA model is misspecified for the volatility
#    and a GARCH extension is warranted.

## ---- 6. ARMA(3,3) forecast ---------------------------------------------------

n_prediction <- 10
forecast_values <- forecast(arima_fit, h = n_prediction)

new_dates <- seq(max(values_df$date), by = "day", length.out = n_prediction + 1)

forecast_df <- rbind(
  values_df,
  data.frame(date = new_dates[-1], value = forecast_values$mean)
)

n_rows <- nrow(forecast_df)
range_forecasted <- (n_rows - n_prediction + 1):n_rows

forecast_df[range_forecasted, "lower_80"] <- as.vector(forecast_values$lower[, 1])
forecast_df[range_forecasted, "upper_80"] <- as.vector(forecast_values$upper[, 1])
forecast_df[range_forecasted, "lower_95"] <- as.vector(forecast_values$lower[, 2])
forecast_df[range_forecasted, "upper_95"] <- as.vector(forecast_values$upper[, 2])

ggplot(forecast_df[(n_rows - 60):n_rows, ], aes(x = date, y = value)) +
  geom_line() +
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95), alpha = 0.15) +
  geom_ribbon(aes(ymin = lower_80, ymax = upper_80), alpha = 0.25) +
  labs(title = "ARMA(3,3) 10-day-ahead return forecast",
       x = NULL, y = "Return (%)")

# IMPORTANT, honest result: when compared against the actual realised
# returns for early January 2023 (true out-of-sample data, not used in
# fitting), the ARMA(3,3) point forecast got MOST of the directional
# signs wrong. In-sample statistical adequacy did not translate into
# forecasting skill - see README for a discussion.

## ---- 7. GARCH extension: grid search over GARCH orders -----------------------

# Keep the ARMA(3,3) mean equation fixed and search over GARCH(p,q),
# p,q in {1,2}, under a normal conditional distribution.
possible_p <- possible_q <- 1:2
grid <- expand.grid(possible_p, possible_q)
colnames(grid) <- c("p", "q")

fit_garch <- function(data, pars, distr = "norm") {
  spec <- ugarchspec(
    variance.model = list(garchOrder = pars),
    mean.model = list(armaOrder = c(3, 3)),
    distribution.model = distr
  )
  fit <- ugarchfit(data = na.omit(data), spec = spec)
  return(fit)
}

garch_fits_norm <- apply(grid, 1, function(x) fit_garch(bank_return, x))

garch_grid_results <- data.frame(
  p   = grid$p,
  q   = grid$q,
  AIC = sapply(garch_fits_norm, function(f) infocriteria(f)[1]),
  BIC = sapply(garch_fits_norm, function(f) infocriteria(f)[2])
)
print(garch_grid_results[order(garch_grid_results$AIC), ])

# AIC/BIC comparison across GARCH(1,1), (2,1), (1,2), (2,2):
#   GARCH(1,1): AIC = 4104.92, BIC = 4154.18  <- best
# GARCH(1,1) wins, consistent with Hansen & Lunde (2005), "A forecast
# comparison of volatility models: does anything beat a GARCH(1,1)?".

## ---- 8. Final GARCH(1,1) fit --------------------------------------------------

spec <- ugarchspec(
  variance.model = list(garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(3, 3)),
  distribution.model = "norm"
)
garch_fit <- ugarchfit(data = bank_return, spec = spec)
show(garch_fit)

# All coefficients significant at alpha = 0.05:
#   mu     = -0.1314   phi1 = -0.6329   phi2 = -0.0737   phi3 = 0.6805
#   theta1 =  0.6720   theta2 = 0.1351  theta3 = -0.6467
#   omega  =  0.1985   alpha1 = 0.1747  beta1  = 0.7831
# alpha1 + beta1 = 0.9578 < 1 -> variance process is stationary with high
# volatility persistence, typical of equity/financial index returns.

## ---- 9. GARCH(1,1) diagnostics --------------------------------------------------

garch_std_resid <- residuals(garch_fit, standardize = TRUE)

# Weighted Ljung-Box tests on standardized residuals and squared
# standardized residuals, ARCH-LM test, and Pearson goodness-of-fit test
# are all reported directly by rugarch:
print(garch_fit@fit$matcoef)
# Use the built-in diagnostics report (Weighted Ljung-Box on residuals and
# squared residuals, ARCH-LM test, Pearson GoF test, sign-bias test):
show(garch_fit)   # see "Weighted Ljung-Box Test" / "Weighted ARCH LM Test"
                  # / "Adjusted Pearson Goodness-of-Fit Test" sections.

# Result: diagnostics mostly support the adequacy of the GARCH(1,1) fit -
# the conditional heteroskedasticity that was left unmodeled by the plain
# ARMA(3,3) is now captured by the GARCH variance equation.

## ---- 10. GARCH(1,1) forecast --------------------------------------------------

forc <- ugarchforecast(garch_fit, data = bank_return, n.ahead = 10)
print(forc)
plot(forc, which = 1)   # forecast of the series (mean)
plot(forc, which = 3)   # forecast of the conditional sigma (volatility)

## ---- 11. Final comparison: ARMA(3,3) vs GARCH(1,1) -----------------------------

comparison <- data.frame(
  metric = c("LogLik", "AIC", "BIC", "RMSE (in-sample)", "Forecasting RMSE"),
  ARMA_3_3   = c(-2197.95, 4409.90, 4444.38, 2.09617, 1.46580),
  GARCH_1_1  = c(-2042.46, 4104.92, 4154.18, 2.10958, 1.43816)
)
print(comparison)

# Conclusion:
# - GARCH(1,1) wins decisively on likelihood-based criteria (LogLik, AIC,
#   BIC) because it explicitly models the conditional heteroskedasticity
#   (volatility clustering) that the plain ARMA(3,3) ignores.
# - GARCH(1,1) also achieves a (slightly) lower out-of-sample forecasting
#   RMSE than ARMA(3,3), despite a marginally higher in-sample RMSE.
# - Overall, GARCH(1,1) is the preferred model for this return series,
#   consistent with the financial-econometrics literature (Engle 1982,
#   Bollerslev 1986, Hansen & Lunde 2005).
# - However, both models' point forecasts should be treated with caution:
#   the ARMA(3,3) forecast got most directional signs wrong against real
#   January 2023 data, a reminder that in-sample fit statistics do not
#   guarantee forecasting skill. See README.md for details.
