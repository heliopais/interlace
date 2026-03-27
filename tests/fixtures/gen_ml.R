#!/usr/bin/env Rscript
# gen_ml.R — generate lme4 ML reference values for tests/test_ml_anova.py
#
# Usage: Rscript tests/fixtures/gen_ml.R
# Output: tests/fixtures/ml_r_results.json

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

data <- read.csv("tests/fixtures/two_re_data.csv")

# Fit with ML (REML=FALSE)
m_ml <- lmer(y ~ x + (1 | firm) + (1 | dept), data = data, REML = FALSE)

fe <- fixef(m_ml)
vc <- as.data.frame(VarCorr(m_ml))

results <- list(
  llf_ml     = as.numeric(logLik(m_ml)),
  aic_ml     = AIC(m_ml),
  bic_ml     = BIC(m_ml),
  fe_params  = as.list(fe),
  var_firm   = vc$vcov[vc$grp == "firm"],
  var_dept   = vc$vcov[vc$grp == "dept"],
  var_resid  = vc$vcov[vc$grp == "Residual"]
)

cat("ML log-likelihood:", results$llf_ml, "\n")
cat("Fixed effects:\n"); print(fe)

write(toJSON(results, auto_unbox = TRUE, digits = 10),
      "tests/fixtures/ml_r_results.json")
cat("Written tests/fixtures/ml_r_results.json\n")
