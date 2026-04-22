#!/usr/bin/env Rscript
# Generate zero-inflated Poisson GLMM reference data from glmmTMB.
#
# Model: y ~ x + (1 | group), family = poisson, ziformula = ~1
#
# The data is generated from a zero-inflated Poisson mixture:
#   - With probability pi, Y = 0  (structural zero)
#   - With probability (1-pi), Y ~ Poisson(mu)
#
# Usage:  Rscript tests/fixtures/gen_glmm_zip.R

library(glmmTMB)
library(jsonlite)

set.seed(42)
n_groups <- 30
n_per_group <- 20
n <- n_groups * n_per_group
group <- rep(1:n_groups, each = n_per_group)
x <- rnorm(n)
u <- rnorm(n_groups, sd = 0.5)

# True parameters
beta0 <- 1.5        # intercept
beta1 <- 0.4        # slope
pi_true <- 0.3      # zero-inflation probability

# Generate Poisson counts
eta <- beta0 + beta1 * x + u[group]
mu <- exp(eta)
y_pois <- rpois(n, lambda = mu)

# Apply zero-inflation
zi <- rbinom(n, size = 1, prob = pi_true)
y <- ifelse(zi == 1, 0, y_pois)

df <- data.frame(y = y, x = x, group = factor(group))
write.csv(df, "tests/fixtures/zip_data.csv", row.names = FALSE)

# Fit ZIP GLMM
fit <- glmmTMB(y ~ x + (1 | group), data = df,
               family = poisson(), ziformula = ~1)
s <- summary(fit)

results <- list(
  fixed_effects = as.list(fixef(fit)$cond),
  fixed_effects_se = as.list(s$coefficients$cond[, "Std. Error"]),
  zi_params = as.list(fixef(fit)$zi),
  variance_components = list(
    group = VarCorr(fit)$cond$group[1, 1]
  ),
  loglik = as.numeric(logLik(fit)),
  aic = AIC(fit),
  nobs = nobs(fit)
)

write(toJSON(results, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      "tests/fixtures/zip_results.json")

cat("Generated: zip_data.csv + zip_results.json\n")
cat(sprintf("  Fixed effects: intercept=%.4f, x=%.4f\n",
            fixef(fit)$cond[["(Intercept)"]], fixef(fit)$cond[["x"]]))
cat(sprintf("  ZI intercept (logit): %.4f  => pi=%.4f\n",
            fixef(fit)$zi[["(Intercept)"]],
            plogis(fixef(fit)$zi[["(Intercept)"]])))
cat(sprintf("  RE variance (group): %.4f\n", VarCorr(fit)$cond$group[1, 1]))
cat(sprintf("  Log-likelihood: %.4f\n", as.numeric(logLik(fit))))
cat(sprintf("  AIC: %.4f\n", AIC(fit)))
