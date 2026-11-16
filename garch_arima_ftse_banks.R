## ============================================================================
## Analysis of Financial Time Series
## European bank-sector proxy for the FTSE Italy All Share Banks index:
## ARIMA and GARCH modeling
##
## Course: Time Series Analysis
## Authors: Bottino Manuel, Corra' Sara, Poetto Patrick, Shaboian Goar
## University coursework (group project) - 11/2024
##
## -----------------------------------------------------------------------
## DATA NOTE (read this first):
## The original dataset used for the coursework (FTSE Italy All Share
## Banks index, daily closing prices, sourced from Investing.com,
## 02/01/2019-30/12/2022) was lost and could not be recovered when this
## script was reconstructed for the portfolio. Investing.com has no
## free, scriptable historical-data API, so the exact original series
## could not be re-downloaded either.
##
## As a free, honestly-labeled substitute covering the same sector, the
## same four-year window, and almost exactly the same number of trading
## days, this script uses the daily closing price of the iShares STOXX
## Europe 600 Banks UCITS ETF (Yahoo Finance ticker "EXV1.DE", traded in
## Frankfurt, denominated in EUR). This ETF tracks a diversified basket
## of large European bank stocks (including major Italian banks such as
## Intesa Sanpaolo and UniCredit), so it reproduces the same economic
## story -- the Covid-19 crash and the 2022 energy/rate shock following
## the Ukraine invasion -- for the European banking sector as a whole.
## It is NOT the original FTSE Italy All Share Banks series and the
## numerical results below will not match the original coursework
## write-up; the modeling workflow (ADF test, ARMA/GARCH grid search,
## diagnostics, forecast evaluation) is unchanged and was re-run
## end-to-end on this substitute series. See README.md for details and
## for how to point the script back at the original CSV layout if that
## data is ever recovered.
##
## The price data (data/exv1_de_banks_etf.csv) was downloaded once via
## quantmod::getSymbols("EXV1.DE", src = "yahoo") and committed to the
## repo so the analysis below is fully reproducible offline:
##
##   library(quantmod)
##   x  <- getSymbols("EXV1.DE", src = "yahoo", from = "2019-01-01",
##                     to = "2023-01-20", auto.assign = FALSE)
##   df <- data.frame(date = format(index(x), "%Y-%m-%d"),
##                     close = as.numeric(Cl(x)))
##   write.csv(df, "data/exv1_de_banks_etf.csv", row.names = FALSE)
##
## The series runs 02/01/2019-30/12/2022 (1,017 daily closes, 1,016 log
## returns) for model fitting/selection, plus 14 further genuine trading
## days (02/01/2023-19/01/2023) held out and never used for fitting or
## order selection, used only in Section 6/10/11 to honestly evaluate
## forecast accuracy against real, realised future returns.
## -----------------------------------------------------------------------
## ============================================================================

## ---- 0. Packages -----------------------------------------------------------

library(forecast)    # Arima(), forecast()
library(tseries)      # adf.test()
library(rugarch)      # ugarchspec(), ugarchfit(), ugarchforecast()
library(FinTS)         # ArchTest()
library(ggplot2)       # plotting
library(xts)            # time-indexed series for rugarch (correct forecast dates)

## ---- 1. Data loading and returns transform ---------------------------------

# CSV columns: date, close (daily closing price of the EXV1.DE ETF, used as
# a substitute for the original FTSE Italy All Share Banks index - see the
# DATA NOTE above).
raw <- read.csv("data/exv1_de_banks_etf.csv", stringsAsFactors = FALSE)
raw$date <- as.Date(raw$date)
raw <- raw[order(raw$date), ]

prices <- raw$close

# Log returns, expressed in percentage points:
#   r_t = log(P_t / P_{t-1}) * 100
log_returns <- diff(log(prices)) * 100
return_dates <- raw$date[-1]

returns_df <- data.frame(date = return_dates, value = log_returns)

# In-sample window used for fitting and model selection (matches the
# original coursework's fitting window): 02/01/2019 - 30/12/2022.
cutoff_date <- as.Date("2022-12-30")
values_df   <- returns_df[returns_df$date <= cutoff_date, ]
bank_return <- values_df$value

# True out-of-sample returns: genuine future trading days, never used for
# fitting or order selection. Used only in Sections 6/10/11 to evaluate
# forecast accuracy honestly.
oos_df <- head(returns_df[returns_df$date > cutoff_date, ], 10)

# xts version of the in-sample return series, carrying real trading-day
# dates, so that rugarch's forecast output ("T0 = ...") shows the correct
# last-fitted date instead of a meaningless numeric-origin default.
bank_return_xts <- xts(bank_return, order.by = values_df$date)

cat(sprintf("In-sample observations (returns): %d (%s to %s)\n",
            nrow(values_df), min(values_df$date), max(values_df$date)))
cat(sprintf("Out-of-sample observations (returns): %d (%s to %s)\n",
            nrow(oos_df), min(oos_df$date), max(oos_df$date)))

ggplot(values_df, aes(x = date, y = value)) +
  geom_line() +
  labs(title = "European bank-sector ETF (EXV1.DE) - daily log returns (%)",
       x = NULL, y = "Return (%)")

# Visual inspection: returns look centered at zero and heteroskedastic,
# with clear volatility clustering around the Covid crash (March 2020) and
# the Ukraine invasion / energy-price shock (February-March 2022).

## ---- 2. Stationarity test (ADF) --------------------------------------------

adf_result <- adf.test(bank_return, k = 20)
print(adf_result)
cat(sprintf("ADF statistic = %.3f, p-value = %.4f\n",
            adf_result$statistic, adf_result$p.value))
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

# Best model by AIC (selected programmatically, not hardcoded, so the
# script stays correct if it is ever re-run on a different series):
best_p <- arma_grid$p[1]
best_q <- arma_grid$q[1]
cat(sprintf("Best ARMA order by AIC: ARMA(%d,%d)\n", best_p, best_q))

## ---- 4. Best ARMA(p,q) fitting ----------------------------------------------

arima_fit <- Arima(bank_return, order = c(best_p, 0, best_q),
                    include.mean = FALSE, method = "ML")
summary(arima_fit)

# Coefficient significance (Z-tests), printed from the actual fit:
arima_coefs <- arima_fit$coef
arima_se    <- sqrt(diag(arima_fit$var.coef))
arima_z     <- arima_coefs / arima_se
arima_pval  <- 2 * pnorm(-abs(arima_z))
print(data.frame(coef = arima_coefs, se = arima_se, z = arima_z, p_value = arima_pval))

## ---- 5. Residual diagnostics (ARMA) -----------------------------------------

arma_resid <- residuals(arima_fit)

# Autocorrelation of residuals
checkresiduals(arima_fit)
print(Box.test(arma_resid, lag = 20, type = "Ljung-Box"))

# Normality
print(shapiro.test(arma_resid))
qqnorm(arma_resid); qqline(arma_resid)
hist(arma_resid, breaks = 40, main = sprintf("ARMA(%d,%d) residuals", best_p, best_q))
# -> financial-return residuals are typically leptokurtic / fat-tailed;
#    Shapiro-Wilk commonly rejects normality here.

# Conditional heteroskedasticity (ARCH effects)
print(ArchTest(arma_resid, lags = 20))
# -> evidence of conditional heteroskedasticity in the residuals would mean
#    the constant-variance ARMA model is misspecified for the volatility
#    and a GARCH extension is warranted.

## ---- 6. ARMA forecast, evaluated against genuine future data ----------------

n_prediction <- nrow(oos_df)
forecast_values <- forecast(arima_fit, h = n_prediction)

# Use the REAL trading-day dates of the held-out period (not a naive
# calendar-day sequence, which would wrongly include weekends/holidays
# and misalign the forecast against the actual future trading days).
forecast_df <- rbind(
  values_df,
  data.frame(date = oos_df$date, value = forecast_values$mean)
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
  labs(title = sprintf("ARMA(%d,%d) %d-day-ahead return forecast", best_p, best_q, n_prediction),
       x = NULL, y = "Return (%)")

# Honest, genuine out-of-sample check: compare the point forecast against
# the REAL realised returns of the next n_prediction trading days (never
# used in fitting or order selection).
arma_oos_rmse <- sqrt(mean((forecast_values$mean - oos_df$value)^2))
arma_sign_hit_rate <- mean(sign(forecast_values$mean) == sign(oos_df$value))
cat(sprintf("ARMA(%d,%d) out-of-sample RMSE: %.5f | directional hit rate: %.1f%%\n",
            best_p, best_q, arma_oos_rmse, 100 * arma_sign_hit_rate))

## ---- 7. GARCH extension: grid search over GARCH orders -----------------------

# Keep the best ARMA mean equation fixed and search over GARCH(p,q),
# p,q in {1,2}, under a normal conditional distribution.
possible_p <- possible_q <- 1:2
grid <- expand.grid(possible_p, possible_q)
colnames(grid) <- c("p", "q")

fit_garch <- function(data, pars, arma_order, distr = "norm") {
  spec <- ugarchspec(
    variance.model = list(garchOrder = pars),
    mean.model = list(armaOrder = arma_order),
    distribution.model = distr
  )
  fit <- ugarchfit(data = na.omit(data), spec = spec)
  return(fit)
}

garch_fits_norm <- apply(grid, 1, function(x) fit_garch(bank_return_xts, as.numeric(x), c(best_p, best_q)))

garch_grid_results <- data.frame(
  p   = grid$p,
  q   = grid$q,
  AIC = sapply(garch_fits_norm, function(f) infocriteria(f)[1]),
  BIC = sapply(garch_fits_norm, function(f) infocriteria(f)[2])
)
garch_grid_results <- garch_grid_results[order(garch_grid_results$AIC), ]
print(garch_grid_results)

best_garch_p <- garch_grid_results$p[1]
best_garch_q <- garch_grid_results$q[1]
cat(sprintf("Best GARCH order by AIC: GARCH(%d,%d)\n", best_garch_p, best_garch_q))
# Hansen & Lunde (2005), "A forecast comparison of volatility models: does
# anything beat a GARCH(1,1)?" found GARCH(1,1) hard to beat for most
# financial series; this grid search checks that empirically rather than
# assuming it.

## ---- 8. Final GARCH(p,q) fit --------------------------------------------------

spec <- ugarchspec(
  variance.model = list(garchOrder = c(best_garch_p, best_garch_q)),
  mean.model = list(armaOrder = c(best_p, best_q)),
  distribution.model = "norm"
)
garch_fit <- ugarchfit(data = bank_return_xts, spec = spec)
show(garch_fit)

persistence <- sum(coef(garch_fit)[grepl("^(alpha|beta)", names(coef(garch_fit)))])
cat(sprintf("alpha+beta persistence = %.4f (stationary variance if < 1)\n", persistence))

## ---- 9. GARCH diagnostics --------------------------------------------------

garch_std_resid <- residuals(garch_fit, standardize = TRUE)

# Weighted Ljung-Box tests on standardized residuals and squared
# standardized residuals, ARCH-LM test, and Pearson goodness-of-fit test
# are all reported directly by rugarch:
print(garch_fit@fit$matcoef)
show(garch_fit)   # see "Weighted Ljung-Box Test" / "Weighted ARCH LM Test"
                  # / "Adjusted Pearson Goodness-of-Fit Test" sections.

## ---- 10. GARCH forecast, evaluated against genuine future data ---------------

forc <- ugarchforecast(garch_fit, data = bank_return_xts, n.ahead = n_prediction)
print(forc)
plot(forc, which = 1)   # forecast of the series (mean)
plot(forc, which = 3)   # forecast of the conditional sigma (volatility)

garch_point_fc <- as.numeric(fitted(forc))
garch_oos_rmse <- sqrt(mean((garch_point_fc - oos_df$value)^2))
garch_sign_hit_rate <- mean(sign(garch_point_fc) == sign(oos_df$value))
cat(sprintf("GARCH(%d,%d) out-of-sample RMSE: %.5f | directional hit rate: %.1f%%\n",
            best_garch_p, best_garch_q, garch_oos_rmse, 100 * garch_sign_hit_rate))

## ---- 11. Final comparison: ARMA vs GARCH -----------------------------

arma_insample_rmse  <- sqrt(mean(arma_resid^2))
garch_insample_rmse <- sqrt(mean(residuals(garch_fit)^2))

garch_loglik <- as.numeric(likelihood(garch_fit))
garch_ic     <- infocriteria(garch_fit)

comparison <- data.frame(
  metric = c("LogLik", "AIC", "BIC", "RMSE (in-sample)", "Forecasting RMSE",
             "Directional hit rate"),
  ARMA   = c(arima_fit$loglik, arima_fit$aic, BIC(arima_fit),
             arma_insample_rmse, arma_oos_rmse, arma_sign_hit_rate),
  GARCH  = c(garch_loglik, garch_ic[1] * length(bank_return), garch_ic[2] * length(bank_return),
             garch_insample_rmse, garch_oos_rmse, garch_sign_hit_rate)
)
colnames(comparison)[2] <- sprintf("ARMA_%d_%d", best_p, best_q)
colnames(comparison)[3] <- sprintf("GARCH_%d_%d", best_garch_p, best_garch_q)
print(comparison)

# Conclusion (evaluated honestly against genuine future trading days, never
# used in fitting or model selection - see README.md "Key results" for the
# actual numbers produced by this run):
# - The GARCH extension is expected to win decisively on likelihood-based
#   criteria (LogLik, AIC, BIC) because it explicitly models the conditional
#   heteroskedasticity (volatility clustering) that the plain ARMA mean
#   model ignores.
# - Whether GARCH also wins on out-of-sample point-forecast RMSE and
#   directional accuracy is an empirical question answered above, not
#   assumed; both numbers are printed above rather than hardcoded.
# - Regardless of the RMSE/hit-rate outcome, both models' point forecasts
#   should be treated with caution: a model that passes residual
#   diagnostics and minimizes AIC in-sample is not thereby guaranteed to
#   have forecasting skill. See README.md for a full discussion.
