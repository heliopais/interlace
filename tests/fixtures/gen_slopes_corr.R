#!/usr/bin/env Rscript
# Generate fixture: correlated random intercept + slope, single factor
# Model: y ~ x + (1 + x | g)
# Outputs: slopes_corr_data.csv, slopes_corr_r_results.json

library(lme4)
library(HLMdiag)
library(jsonlite)

set.seed(42)

n_groups <- 20
n_per    <- 30
n        <- n_groups * n_per

g   <- rep(paste0("g", seq_len(n_groups)), each = n_per)
x   <- rnorm(n)

intercept <- 1.5
beta_x    <- 0.6
sigma_int <- 0.8
sigma_slp <- 0.4
rho       <- 0.3
sigma_e   <- 0.5

# Correlated random effects
cov_mat <- matrix(
  c(sigma_int^2, rho * sigma_int * sigma_slp,
    rho * sigma_int * sigma_slp, sigma_slp^2),
  nrow = 2
)
L <- t(chol(cov_mat))
raw <- matrix(rnorm(n_groups * 2), ncol = 2)
re  <- raw %*% t(L)
b_int   <- re[, 1]
b_slope <- re[, 2]
g_idx   <- as.integer(factor(g, levels = paste0("g", seq_len(n_groups))))

y <- intercept + beta_x * x +
     b_int[g_idx] + b_slope[g_idx] * x +
     rnorm(n, 0, sigma_e)

df <- data.frame(y = y, x = x, g = g)
write.csv(df, "tests/fixtures/slopes_corr_data.csv", row.names = FALSE)

# Fit with lme4 REML
fit <- lmer(y ~ x + (1 + x | g), data = df, REML = TRUE)

# Fixed effects
fe <- fixef(fit)

# Variance components — full covariance matrix
vc_g   <- as.matrix(VarCorr(fit)$g)
vc_df  <- as.data.frame(VarCorr(fit))
var_resid <- vc_df$vcov[vc_df$grp == "Residual"]

# BLUPs: data frame with columns (Intercept) and x
re_g <- ranef(fit)$g
blups_g <- lapply(rownames(re_g), function(lv) {
  as.list(re_g[lv, , drop = FALSE])
})
names(blups_g) <- rownames(re_g)

# Conditional residuals
resid_cond <- residuals(fit, type = "response")

# Marginal residuals: y - X*beta
resid_marg <- as.numeric(df$y - model.matrix(fit) %*% fixef(fit))

# Leverage via full V matrix
{
  X_mat    <- model.matrix(fit)
  n_obs    <- nrow(X_mat)
  sigma2   <- sigma(fit)^2
  cov_beta <- as.matrix(vcov(fit))
  Lambda   <- getME(fit, "Lambda")
  Z_mat    <- t(as.matrix(getME(fit, "Zt")))
  D_mat    <- sigma2 * as.matrix(Lambda %*% t(Lambda))
  ZDZt     <- Z_mat %*% D_mat %*% t(Z_mat)
  V_mat    <- sigma2 * diag(n_obs) + ZDZt
  V_inv    <- solve(V_mat)
  H1_mat   <- X_mat %*% cov_beta %*% t(X_mat) %*% V_inv
  H2_mat   <- ZDZt %*% V_inv %*% (diag(n_obs) - H1_mat)
  lev_fixef    <- diag(H1_mat)
  lev_ranef    <- diag(H2_mat)
  lev_ranef_uc <- diag(ZDZt) / sigma2
}

# Cook's D and MDFFITS via HLMdiag
infl      <- hlm_influence(fit, level = 1)
cooksd    <- as.numeric(infl$cooksd)
mdffits_v <- as.numeric(infl$mdffits)

results <- list(
  fe_params  = as.list(fe),
  cov_g      = lapply(seq_len(nrow(vc_g)), function(i) as.list(vc_g[i, ])),
  var_resid  = var_resid,
  blups_g    = blups_g,
  resid_cond = as.numeric(resid_cond),
  resid_marg = resid_marg,
  leverage   = list(
    overall  = lev_fixef + lev_ranef,
    fixef    = lev_fixef,
    ranef    = lev_ranef,
    ranef.uc = lev_ranef_uc
  ),
  cooksd  = cooksd,
  mdffits = mdffits_v
)
# Add row names to cov_g
names(results$cov_g) <- rownames(vc_g)

write_json(results, "tests/fixtures/slopes_corr_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Done. Wrote slopes_corr_data.csv and slopes_corr_r_results.json\n")
