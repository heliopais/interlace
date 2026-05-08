#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Statistical Rethinking ch14 bangladesh
# varying slopes — correlated random intercept and urban slope by district.
#
# Models (Bernoulli, logit link):
#   M0: use_contraception ~ urban + (1 + urban || district)   # uncorrelated
#   M1: use_contraception ~ urban + (1 + urban  | district)   # correlated (LKJ-style)
#
# Reuses the bangladesh CSV created by gen_sr_13_bangladesh.R.
#
# Usage: Rscript tests/fixtures/gen_sr_14_bangladesh_slopes.R

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

  # Variance terms (var1 == var2): per (group, term) variance
  var_blocks <- vc[is.na(vc$var2) & vc$grp != "Residual", ]
  vc_list <- list()
  for (i in seq_len(nrow(var_blocks))) {
    grp <- var_blocks$grp[i]
    term <- if (is.na(var_blocks$var1[i])) "(Intercept)" else var_blocks$var1[i]
    vc_list[[paste0(grp, "::", term)]] <- var_blocks$vcov[i]
  }
  result$variance_components <- vc_list

  # Correlations (var1 != var2): R reports each pair once
  cor_blocks <- vc[!is.na(vc$var2) & vc$grp != "Residual", ]
  cor_list <- list()
  for (i in seq_len(nrow(cor_blocks))) {
    key <- paste0(cor_blocks$grp[i], "::",
                  cor_blocks$var1[i], "::",
                  cor_blocks$var2[i])
    cor_list[[key]] <- list(
      cov = cor_blocks$vcov[i],
      cor = cor_blocks$sdcor[i]
    )
  }
  result$correlations <- cor_list

  # Random-effect BLUPs by group, by term column
  for (grp_name in names(re)) {
    df <- re[[grp_name]]
    block <- list()
    for (term in colnames(df)) {
      term_key <- if (term == "(Intercept)") "Intercept" else term
      vals <- df[[term]]
      names(vals) <- rownames(df)
      block[[term_key]] <- as.list(vals)
    }
    result[[paste0("ranef_", grp_name)]] <- block
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
# 2. Fit two models
# ---------------------------------------------------------------------------
fit_and_dump <- function(formula_text, name) {
  cat(sprintf("--- Fitting %s ---\n", name))
  form <- as.formula(formula_text)
  fit <- glmer(form, data = df, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
  print(summary(fit))
  res <- extract_glmer(fit, name)
  write(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE),
        file.path(out_dir, sprintf("sr_14_bangladesh_slopes_%s_results.json", name)))
  cat(sprintf("%s written\n\n", name))
}

fit_and_dump("use_contraception ~ urban + (1 + urban || district)", "M0")
fit_and_dump("use_contraception ~ urban + (1 + urban  | district)", "M1")

cat("All SR ch14 bangladesh-slopes fixtures written to", out_dir, "\n")
