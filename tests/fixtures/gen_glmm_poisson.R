#!/usr/bin/env Rscript
# Generate Poisson GLMM reference data from lme4::glmer().
#
# Uses a simple simulated dataset: counts ~ x + (1 | group)
#
# Usage:  Rscript tests/fixtures/gen_glmm_poisson.R

library(lme4)
library(jsonlite)

set.seed(42)
n_groups <- 10
n_per_group <- 20
n <- n_groups * n_per_group

group <- rep(1:n_groups, each = n_per_group)
x <- rnorm(n)
# Random intercepts: SD = 0.5
b <- rnorm(n_groups, sd = 0.5)
eta <- 1.0 + 0.5 * x + b[group]  # intercept=1, slope=0.5
y <- rpois(n, lambda = exp(eta))

df <- data.frame(y = y, x = x, group = factor(group))
write.csv(df, "tests/fixtures/glmm_poisson_data.csv", row.names = FALSE)

fit <- glmer(y ~ x + (1 | group), data = df, family = poisson)
s <- summary(fit)

fe <- fixef(fit)
fe_se <- sqrt(diag(vcov(fit)))
vc <- as.data.frame(VarCorr(fit))
re <- ranef(fit)$group

results <- list(
  fixed_effects = as.list(fe),
  fixed_effects_se = as.list(fe_se),
  variance_components = list(
    group = vc$vcov[vc$grp == "group"]
  ),
  random_effects_group = as.list(re[, 1]),
  random_effects_group_names = rownames(re),
  loglik = as.numeric(logLik(fit)),
  aic = AIC(fit),
  nobs = nobs(fit),
  theta = getME(fit, "theta"),
  converged = (fit@optinfo$conv$opt == 0)
)

write(toJSON(results, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      "tests/fixtures/glmm_poisson_results.json")

cat("Generated: glmm_poisson_data.csv and glmm_poisson_results.json\n")
