#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Statistical Rethinking ch13 bangladesh.
#
# Models (Bernoulli, logit link, varying intercept by district):
#   M0: use_contraception ~ 1                                 + (1|district)
#   M1: use_contraception ~ urban                             + (1|district)
#   M2: use_contraception ~ urban + age_centered + living_children + (1|district)
#
# Data source: McElreath rethinking::bangladesh (canonical CSV vendored from
# rmcelreath/rethinking @ master). Note: column names normalised from
# `use.contraception` etc. to underscore form for python compatibility.
#
# Usage: Rscript tests/fixtures/gen_sr_13_bangladesh.R

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
# 1. Load + prepare frame
# ---------------------------------------------------------------------------
cat("--- Loading bangladesh ---\n")
df <- read.csv(file.path(out_dir, "sr_13_bangladesh_data.csv"),
               stringsAsFactors = FALSE)
df$district <- factor(df$district)
cat(sprintf("Observations: %d  districts: %d\n",
            nrow(df), nlevels(df$district)))

# ---------------------------------------------------------------------------
# 2. Fit three models
# ---------------------------------------------------------------------------
fit_and_dump <- function(formula_rhs, name) {
  cat(sprintf("--- Fitting %s ---\n", name))
  form <- as.formula(paste("use_contraception ~", formula_rhs))
  fit <- glmer(form, data = df, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
  print(summary(fit))
  res <- extract_glmer(fit, name)
  write(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE),
        file.path(out_dir, sprintf("sr_13_bangladesh_%s_results.json", name)))
  cat(sprintf("%s written\n\n", name))
}

fit_and_dump("1 + (1 | district)",                                              "M0")
fit_and_dump("urban + (1 | district)",                                          "M1")
fit_and_dump("urban + age_centered + living_children + (1 | district)",         "M2")

cat("All SR ch13 bangladesh fixtures written to", out_dir, "\n")
