#!/usr/bin/env Rscript
# Generate GLMM reference data from lme4::glmer() on the cbpp dataset.
#
# Usage:  Rscript tests/fixtures/gen_glmm_cbpp.R
#
# Produces:
#   tests/fixtures/glmm_cbpp_data.csv
#   tests/fixtures/glmm_cbpp_results.json

library(lme4)
library(jsonlite)

data(cbpp)

# Fit binomial GLMM (canonical lme4 example):
#   cbind(incidence, size - incidence) ~ period + (1 | herd)
fit <- glmer(
  cbind(incidence, size - incidence) ~ period + (1 | herd),
  data = cbpp,
  family = binomial
)

s <- summary(fit)

# --- Export data ---
# Add proportion column for easier use in Python
cbpp$proportion <- cbpp$incidence / cbpp$size
write.csv(cbpp, "tests/fixtures/glmm_cbpp_data.csv", row.names = FALSE)

# --- Export results ---
fe <- fixef(fit)
fe_se <- sqrt(diag(vcov(fit)))
vc <- as.data.frame(VarCorr(fit))
re <- ranef(fit)$herd

results <- list(
  fixed_effects = as.list(fe),
  fixed_effects_se = as.list(fe_se),
  variance_components = list(
    herd = vc$vcov[vc$grp == "herd"]
  ),
  random_effects_herd = as.list(re[, 1]),
  random_effects_herd_names = rownames(re),
  loglik = as.numeric(logLik(fit)),
  aic = AIC(fit),
  deviance = deviance(fit),
  nobs = nobs(fit),
  theta = getME(fit, "theta"),
  converged = (fit@optinfo$conv$opt == 0)
)

write(toJSON(results, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      "tests/fixtures/glmm_cbpp_results.json")

cat("Generated: glmm_cbpp_data.csv and glmm_cbpp_results.json\n")
