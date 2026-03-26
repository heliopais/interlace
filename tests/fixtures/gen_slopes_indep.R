#!/usr/bin/env Rscript
# Generate fixture: independent random intercept + slope, single factor
# Model: y ~ x + (1 + x || g)
# Outputs: slopes_indep_data.csv, slopes_indep_r_results.json

library(lme4)
library(jsonlite)

set.seed(43)

n_groups <- 20
n_per    <- 30
n        <- n_groups * n_per

g   <- rep(paste0("g", seq_len(n_groups)), each = n_per)
x   <- rnorm(n)

intercept <- 1.5
beta_x    <- 0.6
sigma_int <- 0.8
sigma_slp <- 0.4
sigma_e   <- 0.5

b_int   <- rnorm(n_groups, 0, sigma_int)
b_slope <- rnorm(n_groups, 0, sigma_slp)
g_idx   <- as.integer(factor(g, levels = paste0("g", seq_len(n_groups))))

y <- intercept + beta_x * x +
     b_int[g_idx] + b_slope[g_idx] * x +
     rnorm(n, 0, sigma_e)

df <- data.frame(y = y, x = x, g = g)
write.csv(df, "tests/fixtures/slopes_indep_data.csv", row.names = FALSE)

# Fit with lme4 REML — independent parameterisation
fit <- lmer(y ~ x + (1 + x || g), data = df, REML = TRUE)

# Fixed effects
fe <- fixef(fit)

# Variance components — two scalars (diagonal of covariance matrix)
vc_df     <- as.data.frame(VarCorr(fit))
var_int   <- vc_df$vcov[vc_df$grp == "g" & vc_df$var1 == "(Intercept)" & is.na(vc_df$var2)]
var_slp   <- vc_df$vcov[vc_df$grp == "g.1" & vc_df$var1 == "x" & is.na(vc_df$var2)]
var_resid <- vc_df$vcov[vc_df$grp == "Residual"]

# Build diagonal covariance matrix for consistent JSON structure
cov_g <- list(
  "(Intercept)" = list("(Intercept)" = var_int, "x" = 0.0),
  "x"           = list("(Intercept)" = 0.0,    "x" = var_slp)
)

# BLUPs: ranef returns two separate data frames for || models
re_list <- ranef(fit)
# re_list$g has (Intercept), re_list$g.1 has x
re_int  <- re_list$g
re_slp  <- re_list$`g.1`
blups_g <- lapply(rownames(re_int), function(lv) {
  list("(Intercept)" = re_int[lv, 1], "x" = re_slp[lv, 1])
})
names(blups_g) <- rownames(re_int)

# Conditional residuals
resid_cond <- residuals(fit, type = "response")

results <- list(
  fe_params  = as.list(fe),
  cov_g      = cov_g,
  var_resid  = var_resid,
  blups_g    = blups_g,
  resid_cond = as.numeric(resid_cond)
)

write_json(results, "tests/fixtures/slopes_indep_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Done. Wrote slopes_indep_data.csv and slopes_indep_r_results.json\n")
