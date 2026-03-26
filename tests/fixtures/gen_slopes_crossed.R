#!/usr/bin/env Rscript
# Generate fixture: crossed factors — g1 has random slope, g2 is intercept-only
# Model: y ~ x + (1 + x | g1) + (1 | g2)
# Outputs: slopes_crossed_data.csv, slopes_crossed_r_results.json

library(lme4)
library(jsonlite)

set.seed(44)

n_g1  <- 15
n_g2  <- 8
n     <- 600

g1  <- sample(paste0("a", seq_len(n_g1)), n, replace = TRUE)
g2  <- sample(paste0("b", seq_len(n_g2)), n, replace = TRUE)
x   <- rnorm(n)

intercept <- 2.0
beta_x    <- 0.7
sigma_int <- 0.9
sigma_slp <- 0.35
rho       <- 0.2
sigma_g2  <- 0.5
sigma_e   <- 0.6

# Correlated RE for g1
cov_mat <- matrix(
  c(sigma_int^2, rho * sigma_int * sigma_slp,
    rho * sigma_int * sigma_slp, sigma_slp^2),
  nrow = 2
)
L   <- t(chol(cov_mat))
raw <- matrix(rnorm(n_g1 * 2), ncol = 2)
re1 <- raw %*% t(L)
b1_int   <- re1[, 1]
b1_slope <- re1[, 2]

# Intercept-only for g2
b2 <- rnorm(n_g2, 0, sigma_g2)

g1_idx <- as.integer(factor(g1, levels = paste0("a", seq_len(n_g1))))
g2_idx <- as.integer(factor(g2, levels = paste0("b", seq_len(n_g2))))

y <- intercept + beta_x * x +
     b1_int[g1_idx] + b1_slope[g1_idx] * x +
     b2[g2_idx] +
     rnorm(n, 0, sigma_e)

df <- data.frame(y = y, x = x, g1 = g1, g2 = g2)
write.csv(df, "tests/fixtures/slopes_crossed_data.csv", row.names = FALSE)

# Fit with lme4 REML
fit <- lmer(y ~ x + (1 + x | g1) + (1 | g2), data = df, REML = TRUE)

# Fixed effects
fe <- fixef(fit)

# Variance components
vc_g1   <- as.matrix(VarCorr(fit)$g1)
vc_df   <- as.data.frame(VarCorr(fit))
var_g2  <- vc_df$vcov[vc_df$grp == "g2"]
var_resid <- vc_df$vcov[vc_df$grp == "Residual"]

# BLUPs for g1 (intercept + slope)
re_g1 <- ranef(fit)$g1
blups_g1 <- lapply(rownames(re_g1), function(lv) {
  as.list(re_g1[lv, , drop = FALSE])
})
names(blups_g1) <- rownames(re_g1)

# BLUPs for g2 (intercept only)
re_g2 <- ranef(fit)$g2
blups_g2 <- as.list(re_g2[, 1])
names(blups_g2) <- rownames(re_g2)

# Conditional residuals
resid_cond <- residuals(fit, type = "response")

results <- list(
  fe_params  = as.list(fe),
  cov_g1     = lapply(seq_len(nrow(vc_g1)), function(i) as.list(vc_g1[i, ])),
  var_g2     = var_g2,
  var_resid  = var_resid,
  blups_g1   = blups_g1,
  blups_g2   = blups_g2,
  resid_cond = as.numeric(resid_cond)
)
names(results$cov_g1) <- rownames(vc_g1)

write_json(results, "tests/fixtures/slopes_crossed_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Done. Wrote slopes_crossed_data.csv and slopes_crossed_r_results.json\n")
