#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Gelman & Hill Ch 15 models.
#
# Ch 15 examples in the book use BUGS, not lmer/glmer. We construct
# equivalent models that can be fit with glmer():
#
#   1. Poisson GLMM with overdispersion (observation-level RE) on roaches data
#   2. Negative binomial GLMM on roaches data
#
# Usage:  Rscript tests/fixtures/gen_gelman_hill_ch15.R

library(lme4)
library(jsonlite)
library(MASS)  # for glm.nb

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
extract_glmer <- function(fit, model_name) {
  fe <- fixef(fit)
  fe_se <- sqrt(diag(as.matrix(vcov(fit))))
  names(fe_se) <- names(fe)
  vc <- as.data.frame(VarCorr(fit))
  re <- ranef(fit)

  result <- list(
    model = model_name,
    formula = deparse(formula(fit)),
    nobs = nobs(fit),
    fixed_effects = as.list(fe),
    fixed_effects_se = as.list(fe_se),
    theta = as.numeric(getME(fit, "theta")),
    loglik = as.numeric(logLik(fit)),
    aic = AIC(fit),
    bic = BIC(fit),
    deviance = deviance(fit),
    converged = (fit@optinfo$conv$opt == 0)
  )

  # Variance components
  vc_list <- list()
  for (grp in unique(vc$grp)) {
    if (grp == "Residual") next
    sub <- vc[vc$grp == grp & is.na(vc$var2), ]
    for (i in seq_len(nrow(sub))) {
      vc_list[[grp]] <- sub$vcov[i]
    }
  }
  result$variance_components <- vc_list

  # Ngroups
  ngrps <- list()
  for (grp_name in names(re)) {
    ngrps[[grp_name]] <- nrow(re[[grp_name]])
  }
  result$ngroups <- ngrps

  result
}

# ---------------------------------------------------------------------------
# 1. Load roaches data
# ---------------------------------------------------------------------------
cat("--- Loading roaches data ---\n")
roach <- read.csv(
  "hlm/applied regression and multilevel modeling/ARM_Data/roaches/roachdata.csv"
)
# Columns: y (roach count), roach1 (pre-treatment), treatment, senior, exposure2

# Scale roach1 to avoid numerical issues (following the book)
roach$roach100 <- roach$roach1 / 100

# Add observation-level ID for overdispersion
roach$obs_id <- factor(1:nrow(roach))

write.csv(roach[, c("y", "roach100", "treatment", "senior", "exposure2", "obs_id")],
          file.path(out_dir, "gh_ch15_roaches_data.csv"), row.names = FALSE)
cat("Roaches data:", nrow(roach), "apartments\n")

# ---------------------------------------------------------------------------
# 2. Poisson GLMM with overdispersion (observation-level RE)
# ---------------------------------------------------------------------------
cat("--- Fitting Poisson with observation-level RE (overdispersion) ---\n")

# This is the model from Section 15.1: Poisson + observation-level RE = overdispersed Poisson
# y ~ roach100 + treatment + senior + (1 | obs_id), offset=log(exposure2), family=poisson
M_pois <- glmer(
  y ~ roach100 + treatment + senior + (1 | obs_id),
  data = roach,
  family = poisson,
  offset = log(exposure2)
)
cat("Poisson overdispersed:\n"); print(summary(M_pois))

res_pois <- extract_glmer(M_pois, "poisson_overdispersed")
res_pois$family <- "poisson"
res_pois$link <- "log"
write(toJSON(res_pois, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch15_poisson_results.json"))
cat("Poisson results written\n\n")

# ---------------------------------------------------------------------------
# 3. Standard Poisson GLMM (no overdispersion, for comparison)
# ---------------------------------------------------------------------------
cat("--- Fitting standard Poisson (no overdispersion) ---\n")
M_pois_std <- glm(
  y ~ roach100 + treatment + senior,
  data = roach,
  family = poisson,
  offset = log(exposure2)
)
cat("Standard Poisson GLM:\n"); print(summary(M_pois_std))

res_std <- list(
  model = "poisson_standard",
  formula = "y ~ roach100 + treatment + senior",
  nobs = nobs(M_pois_std),
  fixed_effects = as.list(coef(M_pois_std)),
  loglik = as.numeric(logLik(M_pois_std)),
  aic = AIC(M_pois_std),
  deviance = deviance(M_pois_std),
  dispersion = sum(residuals(M_pois_std, type = "pearson")^2) / M_pois_std$df.residual
)
write(toJSON(res_std, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch15_poisson_std_results.json"))
cat("Standard Poisson results written\n\n")

# ---------------------------------------------------------------------------
# 4. Negative binomial GLM (single-level, for comparison)
# ---------------------------------------------------------------------------
cat("--- Fitting negative binomial GLM ---\n")
M_nb <- glm.nb(
  y ~ roach100 + treatment + senior + offset(log(exposure2)),
  data = roach
)
cat("Negative binomial GLM:\n"); print(summary(M_nb))

res_nb <- list(
  model = "negbin_glm",
  formula = "y ~ roach100 + treatment + senior",
  nobs = nobs(M_nb),
  fixed_effects = as.list(coef(M_nb)),
  loglik = as.numeric(logLik(M_nb)),
  aic = AIC(M_nb),
  nb_theta = M_nb$theta,
  nb_theta_se = M_nb$SE.theta
)
write(toJSON(res_nb, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch15_negbin_results.json"))
cat("Negative binomial results written\n\n")

cat("All Ch 15 fixtures written to", out_dir, "\n")
