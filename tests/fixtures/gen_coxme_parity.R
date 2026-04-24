#!/usr/bin/env Rscript
# Generate coxme reference values for testing interlace.
#
# Produces:
#   1. Fixed-effect estimates (beta, SE, z, p)
#   2. Frailty variance
#   3. BLUPs
#   4. Raw data for Python reproducibility
#
# Requires: coxme, survival, jsonlite
#
# Run from repo root:
#   Rscript tests/fixtures/gen_coxme_parity.R

library(coxme)
library(survival)
library(jsonlite)

set.seed(42)
out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# Simulated shared frailty data
# ---------------------------------------------------------------------------
n_groups <- 30L
n_per <- 30L
n <- n_groups * n_per

group <- factor(rep(seq_len(n_groups), each = n_per))

x1 <- rnorm(n)
x2 <- rnorm(n)

frailty_sd <- 0.5
b <- rnorm(n_groups, sd = frailty_sd)

eta <- 0.5 * x1 - 0.3 * x2 + b[as.integer(group)]

# Event times: exponential baseline h0(t) = 1
u <- runif(n)
event_time <- -log(u) / exp(eta)

# Censoring
censor_rate <- 0.3
censor_time <- rexp(n, rate = censor_rate)

time <- pmin(event_time, censor_time)
event <- as.integer(event_time <= censor_time)

df <- data.frame(
  time = time,
  event = event,
  x1 = x1,
  x2 = x2,
  group = group
)

# ---------------------------------------------------------------------------
# Fit with coxme
# ---------------------------------------------------------------------------
fit <- coxme(Surv(time, event) ~ x1 + x2 + (1 | group), data = df)

# Extract results
beta <- fixef(fit)
vcov_mat <- as.matrix(vcov(fit))
se <- sqrt(diag(vcov_mat))
blups <- ranef(fit)$group

# Variance component
vcomp <- VarCorr(fit)$group

res <- list(
  description = "Shared frailty: 30 groups x 30 obs, seed=42",
  formula = "Surv(time, event) ~ x1 + x2 + (1|group)",
  n = n,
  n_groups = n_groups,
  n_events = sum(event),
  beta = as.list(beta),
  se = as.list(se),
  vcov = vcov_mat,
  frailty_variance = as.numeric(vcomp),
  blups = as.list(blups),
  # Raw data for Python
  time = time,
  event = event,
  x1 = x1,
  x2 = x2,
  group = as.integer(group)
)

write_json(res,
           file.path(out_dir, "coxme_parity.json"),
           digits = 12, auto_unbox = TRUE, matrix = "columnmajor")
cat("coxme parity fixture written to", file.path(out_dir, "coxme_parity.json"), "\n")
