#!/usr/bin/env Rscript
# Generate fixture: correlated random intercept + slope, single factor
# Model: y ~ x + (1 + x | g)
# Outputs: slopes_corr_data.csv, slopes_corr_r_results.json

library(lme4)
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

results <- list(
  fe_params  = as.list(fe),
  cov_g      = lapply(seq_len(nrow(vc_g)), function(i) as.list(vc_g[i, ])),
  var_resid  = var_resid,
  blups_g    = blups_g,
  resid_cond = as.numeric(resid_cond)
)
# Add row names to cov_g
names(results$cov_g) <- rownames(vc_g)

write_json(results, "tests/fixtures/slopes_corr_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Done. Wrote slopes_corr_data.csv and slopes_corr_r_results.json\n")
