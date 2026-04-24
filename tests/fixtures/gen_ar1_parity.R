#!/usr/bin/env Rscript
# Generate fixture: LMM with AR(1) residual correlation (nlme)
# Model: y ~ x, random = ~1|group, correlation = corAR1(form = ~time|group)
# Outputs: ar1_parity_data.csv, ar1_parity_r_results.json

library(nlme)
library(jsonlite)

set.seed(42)

n_groups <- 30
n_time   <- 10
n        <- n_groups * n_time

group <- rep(paste0("g", seq_len(n_groups)), each = n_time)
time  <- rep(seq_len(n_time), times = n_groups)

# Fixed effects
intercept <- 2.0
beta_x    <- -0.5

# Random intercept SD and residual SD
sigma_int <- 0.8
sigma_e   <- 0.6

# True AR(1) parameter
rho_true  <- 0.6

# Simulate random intercepts
b_int <- rnorm(n_groups, 0, sigma_int)
g_idx <- as.integer(factor(group, levels = paste0("g", seq_len(n_groups))))

# Continuous covariate
x <- rnorm(n)

# Simulate AR(1) residuals within each group
eps <- numeric(n)
for (i in seq_len(n_groups)) {
  idx <- ((i - 1) * n_time + 1):(i * n_time)
  e <- numeric(n_time)
  e[1] <- rnorm(1, 0, sigma_e)
  for (t in 2:n_time) {
    e[t] <- rho_true * e[t - 1] + rnorm(1, 0, sigma_e * sqrt(1 - rho_true^2))
  }
  eps[idx] <- e
}

y <- intercept + beta_x * x + b_int[g_idx] + eps

df <- data.frame(y = y, x = x, group = group, time = time)
write.csv(df, "tests/fixtures/ar1_parity_data.csv", row.names = FALSE)

# Fit with nlme
fit <- lme(
  fixed  = y ~ x,
  random = ~ 1 | group,
  correlation = corAR1(form = ~ time | group),
  data   = df,
  method = "REML"
)

# Extract results
fe <- fixef(fit)
vc <- VarCorr(fit)

# Variance components from VarCorr
# VarCorr returns a matrix with Variance and StdDev columns
# Row 1: (Intercept), Row 2: Residual
var_intercept <- as.numeric(vc["(Intercept)", "Variance"])
var_residual  <- as.numeric(vc["Residual", "Variance"])

# AR(1) rho
rho_hat <- as.numeric(coef(fit$modelStruct$corStruct, unconstrained = FALSE))

# Log-likelihood (REML)
ll <- as.numeric(logLik(fit))

# BLUPs (random effects)
re <- ranef(fit)
blups <- setNames(as.list(re[["(Intercept)"]]), rownames(re))

# Conditional residuals
resid_cond <- as.numeric(residuals(fit, type = "response"))

# AIC / BIC
aic_val <- AIC(fit)
bic_val <- BIC(fit)

results <- list(
  fe_params     = as.list(fe),
  var_intercept = var_intercept,
  var_residual  = var_residual,
  rho           = rho_hat,
  loglik        = ll,
  aic           = aic_val,
  bic           = bic_val,
  blups         = blups,
  resid_cond    = resid_cond,
  n_obs         = n,
  n_groups      = n_groups,
  rho_true      = rho_true
)

write_json(results, "tests/fixtures/ar1_parity_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Done. Wrote ar1_parity_data.csv and ar1_parity_r_results.json\n")
