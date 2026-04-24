#!/usr/bin/env Rscript
# Generate lme4 profile CI reference values for testing interlace.
# Produces CIs on BOTH the theta (relative Cholesky) scale and the
# SD/corr/sigma/beta scale.
#
# Run from repo root:
#   Rscript tests/fixtures/gen_profile_ci.R

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# Helper: theta-scale profile CI via deviance function
# ---------------------------------------------------------------------------
theta_profile_ci <- function(fit_ml, level = 0.95) {
  # Get the ML deviance function: devfun(theta) = -2 * logLik(theta)
  # where beta and sigma^2 are profiled out.
  devfun <- update(fit_ml, devFunOnly = TRUE)
  theta_hat <- getME(fit_ml, "theta")
  dev_hat <- devfun(theta_hat)
  chi2_crit <- qchisq(level, df = 1)

  n_theta <- length(theta_hat)
  result <- list()
  for (i in seq_len(n_theta)) {
    t_hat_i <- theta_hat[i]

    # 1D profile: fix all other thetas at MLEs, scan theta_i
    obj <- function(t) {
      tv <- theta_hat
      tv[i] <- t
      devfun(tv) - dev_hat - chi2_crit
    }

    # Lower bound
    lo <- tryCatch({
      uniroot(obj, c(1e-8, t_hat_i), tol = 1e-8)$root
    }, error = function(e) 0)  # boundary case

    # Upper bound: search up to 10x the MLE
    hi <- tryCatch({
      uniroot(obj, c(t_hat_i, max(t_hat_i * 10, 5)), tol = 1e-8)$root
    }, error = function(e) NA)

    result[[names(theta_hat)[i]]] <- list(
      estimate = t_hat_i,
      lower = lo,
      upper = hi
    )
  }
  result
}

# ---------------------------------------------------------------------------
# 1. Dyestuff  —  Yield ~ 1 + (1 | Batch)
# ---------------------------------------------------------------------------
data(Dyestuff)
fit_dye_ml <- lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = FALSE)
theta_ml_d <- getME(fit_dye_ml, "theta")

# SD-scale CIs (lme4 standard)
ci95_sd_d <- confint(fit_dye_ml, method = "profile", level = 0.95, oldNames = FALSE)

# Theta-scale CIs (our custom computation)
ci95_theta_d <- theta_profile_ci(fit_dye_ml, level = 0.95)

res_dye <- list(
  model = "Yield ~ 1 + (1 | Batch)",
  dataset = "Dyestuff",
  theta_ml = as.list(theta_ml_d),
  theta_names = names(theta_ml_d),
  ci95_sd_scale = list(
    lower = as.list(ci95_sd_d[, 1]),
    upper = as.list(ci95_sd_d[, 2]),
    rownames = rownames(ci95_sd_d)
  ),
  ci95_theta_scale = ci95_theta_d,
  sigma_ml = sigma(fit_dye_ml),
  logLik_ml = as.numeric(logLik(fit_dye_ml))
)

write_json(res_dye,
           file.path(out_dir, "lme4_profile_ci_dyestuff.json"),
           digits = 12, auto_unbox = TRUE)
cat("Dyestuff profile CI done\n")

# ---------------------------------------------------------------------------
# 2. sleepstudy  —  Reaction ~ Days + (Days | Subject)
# ---------------------------------------------------------------------------
data(sleepstudy)
fit_sleep_ml <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = FALSE)
theta_ml <- getME(fit_sleep_ml, "theta")

ci95_sd <- confint(fit_sleep_ml, method = "profile", level = 0.95, oldNames = FALSE)
ci95_theta <- theta_profile_ci(fit_sleep_ml, level = 0.95)

res_sleep <- list(
  model = "Reaction ~ Days + (Days | Subject)",
  dataset = "sleepstudy",
  theta_ml = as.list(theta_ml),
  theta_names = names(theta_ml),
  ci95_sd_scale = list(
    lower = as.list(ci95_sd[, 1]),
    upper = as.list(ci95_sd[, 2]),
    rownames = rownames(ci95_sd)
  ),
  ci95_theta_scale = ci95_theta,
  sigma_ml = sigma(fit_sleep_ml),
  logLik_ml = as.numeric(logLik(fit_sleep_ml))
)

write_json(res_sleep,
           file.path(out_dir, "lme4_profile_ci_sleepstudy.json"),
           digits = 12, auto_unbox = TRUE)
cat("sleepstudy profile CI done\n")

cat("\nAll profile CI fixtures written to", out_dir, "\n")
