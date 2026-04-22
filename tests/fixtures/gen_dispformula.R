#!/usr/bin/env Rscript
# Generate dispformula reference data from glmmTMB.
#
# Two models:
#   1. Gaussian heteroscedastic: dispformula = ~z
#   2. NB2 with scalar dispformula: dispformula = ~1
#
# Usage:  Rscript tests/fixtures/gen_dispformula.R

library(glmmTMB)
library(jsonlite)

# ---- Model 1: Gaussian heteroscedastic ----
set.seed(123)
n_groups <- 30
n_per_group <- 20
n <- n_groups * n_per_group
group <- rep(1:n_groups, each = n_per_group)
x <- rnorm(n)
z <- runif(n, -1, 1)
u <- rnorm(n_groups, sd = 1.0)

# True dispersion: phi_i = exp(0.0 + 1.0 * z_i)
phi <- exp(0.0 + 1.0 * z)
eps <- rnorm(n) * sqrt(phi)
y_gauss <- 2.0 + 0.3 * x + u[group] + eps

df_gauss <- data.frame(y = y_gauss, x = x, z = z, group = factor(group))
write.csv(df_gauss, "tests/fixtures/dispformula_gaussian_data.csv",
          row.names = FALSE)

fit_gauss <- glmmTMB(y ~ x + (1 | group), data = df_gauss,
                     family = gaussian(), dispformula = ~z)
s_gauss <- summary(fit_gauss)

gauss_results <- list(
  fixed_effects = as.list(fixef(fit_gauss)$cond),
  disp_params = as.list(fixef(fit_gauss)$disp),
  variance_components = list(
    group = VarCorr(fit_gauss)$cond$group[1, 1]
  ),
  loglik = as.numeric(logLik(fit_gauss)),
  aic = AIC(fit_gauss),
  nobs = nobs(fit_gauss)
)

write(toJSON(gauss_results, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      "tests/fixtures/dispformula_gaussian_results.json")

cat("Generated: dispformula_gaussian_data.csv + results.json\n")

# ---- Model 2: NB2 with scalar dispformula ----
set.seed(77)
n_groups2 <- 30
n_per_group2 <- 20
n2 <- n_groups2 * n_per_group2
group2 <- rep(1:n_groups2, each = n_per_group2)
x2 <- rnorm(n2)
u2 <- rnorm(n_groups2, sd = 0.5)

eta2 <- 1.0 + 0.5 * x2 + u2[group2]
mu2 <- exp(eta2)
theta_true <- 2.0
lam2 <- rgamma(n2, shape = theta_true, rate = theta_true / mu2)
y_nb <- rpois(n2, lambda = lam2)

df_nb <- data.frame(y = y_nb, x = x2, group = factor(group2))
write.csv(df_nb, "tests/fixtures/dispformula_nb2_data.csv", row.names = FALSE)

fit_nb <- glmmTMB(y ~ x + (1 | group), data = df_nb,
                  family = nbinom2(), dispformula = ~1)
s_nb <- summary(fit_nb)

nb_results <- list(
  fixed_effects = as.list(fixef(fit_nb)$cond),
  disp_params = as.list(fixef(fit_nb)$disp),
  variance_components = list(
    group = VarCorr(fit_nb)$cond$group[1, 1]
  ),
  loglik = as.numeric(logLik(fit_nb)),
  aic = AIC(fit_nb),
  nobs = nobs(fit_nb)
)

write(toJSON(nb_results, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      "tests/fixtures/dispformula_nb2_results.json")

cat("Generated: dispformula_nb2_data.csv + results.json\n")
