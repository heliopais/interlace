#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Statistical Rethinking ch12 reedfrogs.
#
# Models (binomial, logit link, varying intercept by tank):
#   M0: cbind(surv, density-surv) ~ 1                + (1 | tank)
#   M1: cbind(surv, density-surv) ~ pred             + (1 | tank)
#   M2: cbind(surv, density-surv) ~ size             + (1 | tank)
#   M3: cbind(surv, density-surv) ~ pred * size      + (1 | tank)
#
# Data source: McElreath rethinking::reedfrogs (canonical CSV vendored from
# rmcelreath/rethinking @ master, no rethinking dependency required).
#
# Usage:  Rscript tests/fixtures/gen_sr_12_reedfrogs.R

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

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
    family = "binomial",
    link = "logit",
    fixed_effects = as.list(fe),
    fixed_effects_se = as.list(fe_se),
    theta = as.numeric(getME(fit, "theta")),
    loglik = as.numeric(logLik(fit)),
    aic = AIC(fit),
    bic = BIC(fit),
    deviance = deviance(fit),
    converged = (fit@optinfo$conv$opt == 0)
  )

  vc_list <- list()
  for (grp in unique(vc$grp)) {
    if (grp == "Residual") next
    sub <- vc[vc$grp == grp & is.na(vc$var2), ]
    for (i in seq_len(nrow(sub))) {
      vc_list[[grp]] <- sub$vcov[i]
    }
  }
  result$variance_components <- vc_list

  for (grp_name in names(re)) {
    df <- re[[grp_name]]
    re_vals <- df[, 1]
    names(re_vals) <- rownames(df)
    result[[paste0("ranef_", grp_name)]] <- as.list(re_vals)
  }

  ngrps <- list()
  for (grp_name in names(re)) {
    ngrps[[grp_name]] <- nrow(re[[grp_name]])
  }
  result$ngroups <- ngrps

  result
}

# ---------------------------------------------------------------------------
# 1. Load data and prepare frame
# ---------------------------------------------------------------------------
cat("--- Loading reedfrogs ---\n")
df <- read.csv(file.path(out_dir, "sr_12_reedfrogs_data.csv"),
               stringsAsFactors = FALSE)
df$tank <- factor(seq_len(nrow(df)))
df$pred <- factor(df$pred, levels = c("no", "pred"))
df$size <- factor(df$size, levels = c("big", "small"))
df$fail <- df$density - df$surv
cat("Tanks:", nrow(df), "\n")

# ---------------------------------------------------------------------------
# 2. Fit four models
# ---------------------------------------------------------------------------
fit_and_dump <- function(formula_rhs, name) {
  cat(sprintf("--- Fitting %s ---\n", name))
  form <- as.formula(paste("cbind(surv, fail) ~", formula_rhs))
  # bobyqa: avoids the Nelder-Mead convergence warning on M0 (one-obs-per-tank
  # OLRE-like binomial GLMM produces a flat ridge that defeats the default).
  fit <- glmer(form, data = df, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
  print(summary(fit))
  res <- extract_glmer(fit, name)
  write(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE),
        file.path(out_dir, sprintf("sr_12_reedfrogs_%s_results.json", name)))
  cat(sprintf("%s written\n\n", name))
}

fit_and_dump("1 + (1 | tank)",                "M0")
fit_and_dump("pred + (1 | tank)",             "M1")
fit_and_dump("size + (1 | tank)",             "M2")
fit_and_dump("pred * size + (1 | tank)",      "M3")

cat("All SR ch12 reedfrogs fixtures written to", out_dir, "\n")
