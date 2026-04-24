#!/usr/bin/env Rscript
# Generate fixture: Cumulative link mixed model (ordinal::clmm)
# Model: rating ~ temp + contact + (1|judge), link = logit
# Dataset: wine (from ordinal package, 72 obs, 5 ordinal levels)
# Outputs: clmm_parity_data.csv, clmm_parity_r_results.json

library(ordinal)
library(jsonlite)

data(wine)

# --- Fit CLMM with logit link ---
fit <- clmm(rating ~ temp + contact + (1 | judge), data = wine, link = "logit")

# Extract thresholds (alpha)
thresholds <- fit$alpha
threshold_names <- names(thresholds)

# Extract fixed effects (beta)
betas <- fit$beta
beta_names <- names(betas)

# All coefficients and SEs
all_coefs <- coef(fit)
vcov_mat <- vcov(fit)
all_se <- sqrt(diag(vcov_mat))
# vcov(fit) returns SEs for thresholds + betas (no RE SD entry)

n_thresh <- length(thresholds)
n_beta <- length(betas)
threshold_se <- all_se[1:n_thresh]
beta_se <- all_se[(n_thresh + 1):(n_thresh + n_beta)]

# Variance components
vc <- VarCorr(fit)
# vc$judge is a matrix; attr(, "stddev") gives the SD
re_var <- as.numeric(vc$judge[1, 1])
re_sd <- attr(vc$judge, "stddev")

# Log-likelihood
ll <- as.numeric(logLik(fit))
npar <- attr(logLik(fit), "df")

# BLUPs (conditional modes)
re <- ranef(fit)
blups_judge <- setNames(
  as.list(re$judge[["(Intercept)"]]),
  rownames(re$judge)
)

# Condition number of Hessian (convergence quality)
cond_hess <- fit$cond.H

# --- Fit CLMM with probit link for multi-link validation ---
fit_probit <- clmm(rating ~ temp + contact + (1 | judge),
                    data = wine, link = "probit")
thresholds_probit <- fit_probit$alpha
betas_probit <- fit_probit$beta
ll_probit <- as.numeric(logLik(fit_probit))
vc_probit <- VarCorr(fit_probit)
re_var_probit <- as.numeric(vc_probit$judge[1, 1])

# --- Save data ---
wine_out <- data.frame(
  rating = as.integer(wine$rating),
  temp = as.character(wine$temp),
  contact = as.character(wine$contact),
  judge = as.character(wine$judge),
  bottle = as.character(wine$bottle)
)
write.csv(wine_out, "tests/fixtures/clmm_parity_data.csv", row.names = FALSE)

# --- Save results ---
results <- list(
  logit = list(
    thresholds = as.list(thresholds),
    threshold_se = as.list(setNames(threshold_se, threshold_names)),
    betas = as.list(betas),
    beta_se = as.list(setNames(beta_se, beta_names)),
    re_var_judge = re_var,
    re_sd_judge = as.numeric(re_sd),
    loglik = ll,
    npar = npar,
    blups_judge = blups_judge,
    cond_hess = cond_hess
  ),
  probit = list(
    thresholds = as.list(thresholds_probit),
    betas = as.list(betas_probit),
    re_var_judge = re_var_probit,
    loglik = ll_probit
  ),
  n_obs = nrow(wine),
  n_levels = length(levels(wine$rating))
)

write_json(results, "tests/fixtures/clmm_parity_r_results.json",
           digits = 12, auto_unbox = TRUE)

cat("Done. Wrote clmm_parity_data.csv and clmm_parity_r_results.json\n")
cat(sprintf("Logit:  thresholds = %s\n",
            paste(sprintf("%.4f", thresholds), collapse = ", ")))
cat(sprintf("        betas = %s\n",
            paste(sprintf("%.4f", betas), collapse = ", ")))
cat(sprintf("        RE var = %.4f (SD = %.4f)\n", re_var, re_sd))
cat(sprintf("        logLik = %.4f (df = %d)\n", ll, npar))
cat(sprintf("Probit: logLik = %.4f\n", ll_probit))
