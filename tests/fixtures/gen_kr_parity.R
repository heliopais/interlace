#!/usr/bin/env Rscript
# Generate Kenward-Roger reference values for testing interlace.
#
# Produces:
#   1. vcovAdj (KR-adjusted covariance matrix)
#   2. KR denominator DFs for each fixed effect
#   3. The unadjusted fe_cov for sanity-checking
#   4. dC (first derivatives of vcov w.r.t. variance parameters)
#
# Uses pbkrtest and lmerTest on a simple one-way random-intercept model
# with a synthetic dataset (seeded for reproducibility).
#
# Run from repo root:
#   Rscript tests/fixtures/gen_kr_parity.R

library(lme4)
library(pbkrtest)
library(lmerTest)
library(jsonlite)

set.seed(42)
out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# 1. Simple one-way: y ~ x + (1 | group), 15 groups x 8 obs
# ---------------------------------------------------------------------------
n_groups <- 15L
n_per <- 8L
n <- n_groups * n_per

group <- factor(rep(seq_len(n_groups), each = n_per))
x <- rnorm(n)
u <- rnorm(n_groups, sd = 1.5)
y <- 2.0 + 1.5 * x + u[group] + rnorm(n, sd = 0.6)

df1 <- data.frame(y = y, x = x, group = group)

fit1 <- lmer(y ~ x + (1 | group), data = df1, REML = TRUE)

# KR-adjusted covariance
kr1 <- vcovAdj(fit1)
kr1_mat <- as.matrix(kr1)

# KR denominator DFs (from lmerTest)
fit1_lt <- as(fit1, "lmerModLmerTest")
satt1 <- summary(fit1_lt, ddf = "Kenward-Roger")
kr_dfs1 <- coef(satt1)[, "df"]

# Unadjusted covariance
fe_cov1 <- as.matrix(vcov(fit1))

# Theta
theta1 <- getME(fit1, "theta")

# Fixed effects
beta1 <- fixef(fit1)

# Sigma
sigma1 <- sigma(fit1)

res1 <- list(
  model = "y ~ x + (1 | group)",
  description = "15 groups x 8 obs, seed=42, same RNG as Python test",
  n = n,
  n_groups = n_groups,
  theta = as.list(theta1),
  sigma = sigma1,
  beta = as.list(beta1),
  fe_cov = fe_cov1,
  kr_vcov_adj = kr1_mat,
  kr_dfs = as.list(kr_dfs1),
  # Store the raw data so Python can reproduce exactly
  y = y,
  x = x,
  group = as.integer(group)
)

write_json(res1,
           file.path(out_dir, "kr_parity_one_factor.json"),
           digits = 12, auto_unbox = TRUE, matrix = "columnmajor")
cat("One-factor KR parity fixture done\n")

# ---------------------------------------------------------------------------
# 2. Crossed two-factor: y ~ x + (1 | g1) + (1 | g2)
# ---------------------------------------------------------------------------
set.seed(99)
n2 <- 200L
g1 <- factor(sample(paste0("a", 1:10), n2, replace = TRUE))
g2 <- factor(sample(paste0("b", 1:6), n2, replace = TRUE))
x2 <- rnorm(n2)
u1 <- rnorm(10, sd = 1.0)
u2 <- rnorm(6, sd = 0.8)
y2 <- 3.0 + 2.0 * x2 + u1[as.integer(g1)] + u2[as.integer(g2)] + rnorm(n2, sd = 0.5)

df2 <- data.frame(y = y2, x = x2, g1 = g1, g2 = g2)

fit2 <- lmer(y ~ x + (1 | g1) + (1 | g2), data = df2, REML = TRUE)

kr2 <- vcovAdj(fit2)
kr2_mat <- as.matrix(kr2)

fit2_lt <- as(fit2, "lmerModLmerTest")
satt2 <- summary(fit2_lt, ddf = "Kenward-Roger")
kr_dfs2 <- coef(satt2)[, "df"]

fe_cov2 <- as.matrix(vcov(fit2))
theta2 <- getME(fit2, "theta")
beta2 <- fixef(fit2)
sigma2 <- sigma(fit2)

res2 <- list(
  model = "y ~ x + (1 | g1) + (1 | g2)",
  description = "Crossed 10 x 6 groups, n=200, seed=99",
  n = n2,
  theta = as.list(theta2),
  sigma = sigma2,
  beta = as.list(beta2),
  fe_cov = fe_cov2,
  kr_vcov_adj = kr2_mat,
  kr_dfs = as.list(kr_dfs2),
  y = y2,
  x = x2,
  g1 = as.character(g1),
  g2 = as.character(g2)
)

write_json(res2,
           file.path(out_dir, "kr_parity_two_factor.json"),
           digits = 12, auto_unbox = TRUE, matrix = "columnmajor")
cat("Two-factor KR parity fixture done\n")

cat("\nAll KR parity fixtures written to", out_dir, "\n")
