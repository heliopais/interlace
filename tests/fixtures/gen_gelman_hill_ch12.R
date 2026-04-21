#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Gelman & Hill Ch 12 radon models.
#
# Models:
#   M0: y ~ 1 + (1 | county)                 — varying intercept, no predictors
#   M1: y ~ x + (1 | county)                 — varying intercept, floor predictor
#   M2: y ~ x + u.full + (1 | county)        — varying intercept, floor + county uranium
#
# Outputs:
#   tests/fixtures/gh_ch12_radon_data.csv     — cleaned Minnesota radon data
#   tests/fixtures/gh_ch12_M0_results.json    — M0 reference values
#   tests/fixtures/gh_ch12_M1_results.json    — M1 reference values
#   tests/fixtures/gh_ch12_M2_results.json    — M2 reference values
#
# Usage:  Rscript tests/fixtures/gen_gelman_hill_ch12.R
#
# Requires: lme4, jsonlite

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# 1. Load and clean data (following the book's R code exactly)
# ---------------------------------------------------------------------------

# Read srrs2 (radon survey data)
srrs2 <- read.table(
  file.path("hlm/applied regression and multilevel modeling/Book_Codes/Ch.12/srrs2.dat"),
  header = TRUE, sep = ","
)

# Filter to Minnesota
mn <- srrs2$state == "MN"
radon <- srrs2$activity[mn]
log.radon <- log(ifelse(radon == 0, 0.1, radon))
floor <- srrs2$floor[mn]  # 0 = basement, 1 = first floor
n <- length(radon)
y <- log.radon
x <- floor

# Build county index (following book's approach)
county.name <- as.vector(srrs2$county[mn])
uniq <- unique(county.name)
J <- length(uniq)
county <- rep(NA, J)
for (i in 1:J) {
  county[county.name == uniq[i]] <- i
}

# Read county-level data for uranium predictor
cty <- read.table(
  file.path("hlm/applied regression and multilevel modeling/Book_Codes/Ch.12/cty.dat"),
  header = TRUE, sep = ","
)
srrs2.fips <- srrs2$stfips * 1000 + srrs2$cntyfips
usa.fips <- 1000 * cty[, "stfips"] + cty[, "ctfips"]
usa.rows <- match(unique(srrs2.fips[mn]), usa.fips)
uranium <- cty[usa.rows, "Uppm"]
u <- log(uranium)

# Expand county-level uranium to observation level
u.full <- u[county]

# Create a clean data frame for export
county.name.clean <- trimws(county.name)
radon_df <- data.frame(
  y = y,
  x = x,
  county = county,
  county_name = county.name.clean,
  u = u.full,
  log_radon = log.radon,
  radon = radon,
  stringsAsFactors = FALSE
)

write.csv(radon_df, file.path(out_dir, "gh_ch12_radon_data.csv"), row.names = FALSE)
cat("Wrote", nrow(radon_df), "observations,", J, "counties\n")

# Also export county-level data
county_df <- data.frame(
  county = 1:J,
  county_name = trimws(uniq),
  log_uranium = u,
  sample_size = as.vector(table(county)),
  stringsAsFactors = FALSE
)
write.csv(county_df, file.path(out_dir, "gh_ch12_county_data.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# Helper to extract results from an lmer fit
# ---------------------------------------------------------------------------
extract_lmer <- function(fit, model_name) {
  fe <- fixef(fit)
  fe_se <- sqrt(diag(as.matrix(vcov(fit))))
  names(fe_se) <- names(fe)
  vc <- as.data.frame(VarCorr(fit))
  re <- ranef(fit)$county
  re_se <- arm::se.ranef(fit)$county

  # Conditional coefficients (fixed + random)
  cc <- coef(fit)$county

  list(
    model = model_name,
    formula = deparse(formula(fit)),
    nobs = nobs(fit),
    ngroups = length(unique(county)),
    method = "REML",
    fixed_effects = as.list(fe),
    fixed_effects_se = as.list(fe_se),
    variance_components = list(
      county = vc$vcov[vc$grp == "county"],
      residual = vc$vcov[vc$grp == "Residual"]
    ),
    sigma = sigma(fit),
    theta = as.numeric(getME(fit, "theta")),
    loglik = as.numeric(logLik(fit)),
    REML_crit = as.numeric(REMLcrit(fit)),
    aic = AIC(fit),
    bic = BIC(fit),
    deviance = as.numeric(deviance(fit, REML = FALSE)),
    random_effects_county = as.list(re[, 1]),
    random_effects_county_names = rownames(re),
    random_effects_county_se = as.list(re_se[, 1]),
    conditional_coefs = lapply(as.list(cc), function(col) {
      v <- col
      names(v) <- rownames(cc)
      as.list(v)
    }),
    residuals = as.numeric(residuals(fit, type = "response")),
    fitted_values = as.numeric(fitted(fit)),
    converged = (fit@optinfo$conv$opt == 0)
  )
}

# ---------------------------------------------------------------------------
# 2. Fit M0: y ~ 1 + (1 | county)
# ---------------------------------------------------------------------------
M0 <- lmer(y ~ 1 + (1 | county))
cat("M0 fitted:\n")
print(summary(M0))

res_M0 <- extract_lmer(M0, "M0")
write(toJSON(res_M0, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch12_M0_results.json"))
cat("M0 results written\n\n")

# ---------------------------------------------------------------------------
# 3. Fit M1: y ~ x + (1 | county)
# ---------------------------------------------------------------------------
M1 <- lmer(y ~ x + (1 | county))
cat("M1 fitted:\n")
print(summary(M1))

res_M1 <- extract_lmer(M1, "M1")
write(toJSON(res_M1, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch12_M1_results.json"))
cat("M1 results written\n\n")

# ---------------------------------------------------------------------------
# 4. Fit M2: y ~ x + u.full + (1 | county)
# ---------------------------------------------------------------------------
M2 <- lmer(y ~ x + u.full + (1 | county))
cat("M2 fitted:\n")
print(summary(M2))

res_M2 <- extract_lmer(M2, "M2")
write(toJSON(res_M2, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch12_M2_results.json"))
cat("M2 results written\n\n")

# ---------------------------------------------------------------------------
# 5. Also fit with ML for model comparison (needed for exercises)
# ---------------------------------------------------------------------------
M1_ML <- lmer(y ~ x + (1 | county), REML = FALSE)
M2_ML <- lmer(y ~ x + u.full + (1 | county), REML = FALSE)

ml_comparison <- list(
  M1_ML = list(
    loglik = as.numeric(logLik(M1_ML)),
    aic = AIC(M1_ML),
    bic = BIC(M1_ML),
    deviance = as.numeric(deviance(M1_ML)),
    fixed_effects = as.list(fixef(M1_ML))
  ),
  M2_ML = list(
    loglik = as.numeric(logLik(M2_ML)),
    aic = AIC(M2_ML),
    bic = BIC(M2_ML),
    deviance = as.numeric(deviance(M2_ML)),
    fixed_effects = as.list(fixef(M2_ML))
  )
)
write(toJSON(ml_comparison, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch12_ML_comparison.json"))
cat("ML comparison results written\n\n")

cat("All Ch 12 fixtures written to", out_dir, "\n")
