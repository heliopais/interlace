#!/usr/bin/env Rscript
# Generate fixture: LMM with compound symmetry residual correlation (nlme)
# Model: y ~ x, random = ~1|group, correlation = corCompSymm(form = ~1|group)
# Outputs: cs_parity_data.csv, cs_parity_r_results.json
#
# NOTE: Random intercept + CS on the same grouping factor are not separately
# identifiable — any re-parameterisation along the manifold
#   var_resid*(1-rho) = const, var_int + var_resid*rho = const
# gives the same marginal covariance.  nlme typically absorbs CS into the
# random intercept (rho → 0).  The parity test therefore checks fixed effects,
# REML log-likelihood, and the marginal ICC rather than individual variance
# components.  A supplementary gls() fit (no random effects) validates rho
# estimation in isolation.

library(nlme)
library(jsonlite)

set.seed(123)

n_groups <- 30
n_time   <- 10
n        <- n_groups * n_time

group <- rep(paste0("g", seq_len(n_groups)), each = n_time)
time  <- rep(seq_len(n_time), times = n_groups)

# Fixed effects
intercept <- 3.0
beta_x    <- 0.7

# True parameters: CS-only (no random intercept in DGP)
sigma_e  <- 0.9
rho_true <- 0.4

# Continuous covariate
x <- rnorm(n)

# Simulate CS-correlated residuals within each group (no random intercept)
eps <- numeric(n)
for (i in seq_len(n_groups)) {
  idx <- ((i - 1) * n_time + 1):(i * n_time)
  R_g <- (1 - rho_true) * diag(n_time) + rho_true * matrix(1, n_time, n_time)
  L   <- t(chol(sigma_e^2 * R_g))
  eps[idx] <- L %*% rnorm(n_time)
}

y <- intercept + beta_x * x + eps

df <- data.frame(y = y, x = x, group = group, time = time)
write.csv(df, "tests/fixtures/cs_parity_data.csv", row.names = FALSE)

# --- Fit 1: lme with random intercept + corCompSymm (same as interlace) ---
fit_lme <- lme(
  fixed  = y ~ x,
  random = ~ 1 | group,
  correlation = corCompSymm(form = ~ 1 | group),
  data   = df,
  method = "REML"
)

fe_lme <- fixef(fit_lme)
vc_lme <- VarCorr(fit_lme)

var_intercept_lme <- as.numeric(vc_lme["(Intercept)", "Variance"])
var_residual_lme  <- as.numeric(vc_lme["Residual", "Variance"])
rho_lme <- as.numeric(coef(fit_lme$modelStruct$corStruct, unconstrained = FALSE))
ll_lme  <- as.numeric(logLik(fit_lme))

# Marginal ICC: proportion of total marginal variance due to exchangeable part
total_var_lme <- var_intercept_lme + var_residual_lme
marginal_icc_lme <- (var_intercept_lme + var_residual_lme * rho_lme) / total_var_lme

re_lme <- ranef(fit_lme)
blups_lme <- setNames(as.list(re_lme[["(Intercept)"]]), rownames(re_lme))
resid_lme <- as.numeric(residuals(fit_lme, type = "response"))

# --- Fit 2: gls with corCompSymm only (no random effects) ---
# This is the "clean" reference for rho — no identifiability issue.
fit_gls <- gls(
  model = y ~ x,
  correlation = corCompSymm(form = ~ 1 | group),
  data  = df,
  method = "REML"
)

fe_gls <- coef(fit_gls)
sigma_gls <- fit_gls$sigma
rho_gls   <- as.numeric(coef(fit_gls$modelStruct$corStruct, unconstrained = FALSE))
ll_gls    <- as.numeric(logLik(fit_gls))

results <- list(
  # lme fit (random intercept + CS — matches interlace model structure)
  lme = list(
    fe_params     = as.list(fe_lme),
    var_intercept = var_intercept_lme,
    var_residual  = var_residual_lme,
    rho           = rho_lme,
    loglik        = ll_lme,
    aic           = AIC(fit_lme),
    bic           = BIC(fit_lme),
    total_var     = total_var_lme,
    marginal_icc  = marginal_icc_lme,
    blups         = blups_lme,
    resid_cond    = resid_lme
  ),
  # gls fit (CS only — clean rho reference, no identifiability issue)
  gls = list(
    fe_params = as.list(fe_gls),
    sigma     = sigma_gls,
    rho       = rho_gls,
    loglik    = ll_gls
  ),
  # Simulation parameters
  n_obs    = n,
  n_groups = n_groups,
  rho_true = rho_true,
  sigma_e  = sigma_e
)

write_json(results, "tests/fixtures/cs_parity_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Done. Wrote cs_parity_data.csv and cs_parity_r_results.json\n")
cat(sprintf("lme: rho=%.6f, var_int=%.6f, var_resid=%.6f, ICC=%.4f\n",
            rho_lme, var_intercept_lme, var_residual_lme, marginal_icc_lme))
cat(sprintf("gls: rho=%.6f, sigma=%.6f\n", rho_gls, sigma_gls))
