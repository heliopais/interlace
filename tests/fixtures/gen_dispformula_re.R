#!/usr/bin/env Rscript
# Generate reference fits for dispformula with random effects on the
# dispersion side, matching the salary-models use case.
#
# Two models:
#   1. Single random intercept on disp:  dispformula = ~ (1|g_d)
#   2. Nested random intercepts on disp: dispformula = ~ (1|g1/g2)
#
# Usage:  Rscript tests/fixtures/gen_dispformula_re.R

library(glmmTMB)
library(jsonlite)

# ---- Model 1: single random intercept on disp side ----
set.seed(2025)
n_mean_groups <- 25
n_disp_groups <- 15
n_per_cell <- 8
n <- n_mean_groups * n_per_cell

# Mean-side grouping factor
g_mean <- factor(rep(1:n_mean_groups, each = n_per_cell))
# Disp-side grouping factor: spread across observations
g_disp <- factor(sample.int(n_disp_groups, n, replace = TRUE))

x <- rnorm(n)
# Mean-side random intercepts
u_mean <- rnorm(n_mean_groups, sd = 0.8)
# Disp-side random intercepts (on log_sigma scale)
u_disp <- rnorm(n_disp_groups, sd = 0.45)

# True dispersion: log sigma_i = 0.0 + u_disp[g_disp_i]
log_sigma <- 0.0 + u_disp[g_disp]
sigma <- exp(log_sigma)
eps <- rnorm(n) * sigma

y <- 1.5 + 0.4 * x + u_mean[g_mean] + eps

df <- data.frame(y = y, x = x, g_mean = g_mean, g_disp = g_disp)
write.csv(df, "tests/fixtures/dispformula_re_data.csv", row.names = FALSE)

fit1 <- glmmTMB(y ~ x + (1 | g_mean), data = df,
                family = gaussian(), dispformula = ~ (1 | g_disp))

s1 <- summary(fit1)
vc1 <- VarCorr(fit1)

res1 <- list(
  fixed_effects = as.list(fixef(fit1)$cond),
  disp_params   = as.list(fixef(fit1)$disp),
  variance_components = list(g_mean = vc1$cond$g_mean[1, 1]),
  disp_variance_components = list(g_disp = vc1$disp$g_disp[1, 1]),
  loglik = as.numeric(logLik(fit1)),
  aic    = AIC(fit1),
  nobs   = nobs(fit1)
)
write(toJSON(res1, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      "tests/fixtures/dispformula_re_results.json")
cat("Generated: dispformula_re_data.csv + results.json\n")

# ---- Model 2: nested random intercepts on disp side ----
set.seed(7)
n_lvl1 <- 8
n_lvl2_per_lvl1 <- 5
n_per_cell2 <- 12
n2 <- n_lvl1 * n_lvl2_per_lvl1 * n_per_cell2

g1 <- factor(rep(1:n_lvl1, each = n_lvl2_per_lvl1 * n_per_cell2))
g2_inner <- rep(1:n_lvl2_per_lvl1, each = n_per_cell2)
g2 <- factor(rep(g2_inner, n_lvl1))
g_mean2 <- factor(rep(1:(n_lvl1 * n_lvl2_per_lvl1), each = n_per_cell2))

x2 <- rnorm(n2)
u_mean2 <- rnorm(n_lvl1 * n_lvl2_per_lvl1, sd = 0.6)

# Disp-side: nested random intercepts on (g1/g2)
u_g1 <- rnorm(n_lvl1, sd = 0.35)
u_g1g2 <- rnorm(n_lvl1 * n_lvl2_per_lvl1, sd = 0.25)
# Map u_g1[g1] and u_g1g2[g_mean2]
log_sigma2 <- -0.05 + u_g1[as.integer(g1)] + u_g1g2[as.integer(g_mean2)]
sigma2 <- exp(log_sigma2)
y2 <- 2.0 + 0.3 * x2 + u_mean2[as.integer(g_mean2)] + rnorm(n2) * sigma2

df2 <- data.frame(y = y2, x = x2, g_mean = g_mean2, g1 = g1, g2 = g2)
write.csv(df2, "tests/fixtures/dispformula_nested_data.csv", row.names = FALSE)

fit2 <- glmmTMB(y ~ x + (1 | g_mean), data = df2,
                family = gaussian(), dispformula = ~ (1 | g1 / g2))

s2 <- summary(fit2)
vc2 <- VarCorr(fit2)

# glmmTMB names the nested factors g1 and g2:g1.
disp_vc <- vc2$disp
disp_vc_names <- names(disp_vc)
disp_vc_out <- list()
for (nm in disp_vc_names) {
  disp_vc_out[[nm]] <- disp_vc[[nm]][1, 1]
}

res2 <- list(
  fixed_effects = as.list(fixef(fit2)$cond),
  disp_params   = as.list(fixef(fit2)$disp),
  variance_components = list(g_mean = vc2$cond$g_mean[1, 1]),
  disp_variance_components = disp_vc_out,
  loglik = as.numeric(logLik(fit2)),
  aic    = AIC(fit2),
  nobs   = nobs(fit2)
)
write(toJSON(res2, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      "tests/fixtures/dispformula_nested_results.json")
cat("Generated: dispformula_nested_data.csv + results.json\n")
