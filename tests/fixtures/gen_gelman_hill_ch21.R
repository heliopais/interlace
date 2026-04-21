#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Gelman & Hill Ch 21.
#
# Ch 21 covers understanding and interpreting multilevel models.
# Key sections:
#   21.5: R-squared and explained variance
#   21.6: Summarizing the amount of partial pooling
#   21.7: Adding a predictor can increase residual variance
#
# Models (all on Minnesota radon data):
#   M_nox:   y ~ u.full + (1 | county)           — no floor predictor
#   M_floor: y ~ u.full + x + (1 | county)       — add floor
#   M_xbar:  y ~ u.full + x + x.mean + (1 | county) — add county mean floor
#
# Outputs:
#   tests/fixtures/gh_ch21_results.json
#
# Usage:  Rscript tests/fixtures/gen_gelman_hill_ch21.R
#
# Requires: lme4, jsonlite

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# Load radon data (same as Ch 12)
# ---------------------------------------------------------------------------
srrs2 <- read.table(
  "hlm/applied regression and multilevel modeling/Book_Codes/Ch.12/srrs2.dat",
  header = TRUE, sep = ","
)
mn <- srrs2$state == "MN"
radon <- srrs2$activity[mn]
log.radon <- log(ifelse(radon == 0, 0.1, radon))
y <- log.radon
x <- srrs2$floor[mn]
county.name <- as.vector(srrs2$county[mn])
uniq <- unique(county.name)
J <- length(uniq)
county <- rep(NA, J)
for (i in 1:J) county[county.name == uniq[i]] <- i

cty <- read.table(
  "hlm/applied regression and multilevel modeling/Book_Codes/Ch.12/cty.dat",
  header = TRUE, sep = ","
)
srrs2.fips <- srrs2$stfips * 1000 + srrs2$cntyfips
usa.fips <- 1000 * cty[, "stfips"] + cty[, "ctfips"]
usa.rows <- match(unique(srrs2.fips[mn]), usa.fips)
u <- log(cty[usa.rows, "Uppm"])
u.full <- u[county]

# County mean of floor (proportion first-floor measurements)
x.mean <- rep(NA, J)
for (j in 1:J) x.mean[j] <- mean(x[county == j])
x.mean.full <- x.mean[county]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
extract_model <- function(fit, model_name) {
  fe <- fixef(fit)
  fe_se <- sqrt(diag(as.matrix(vcov(fit))))
  names(fe_se) <- names(fe)
  vc <- as.data.frame(VarCorr(fit))
  re <- ranef(fit)$county[, 1]

  list(
    model = model_name,
    formula = deparse(formula(fit)),
    nobs = nobs(fit),
    fixed_effects = as.list(fe),
    fixed_effects_se = as.list(fe_se),
    variance_components = list(
      county = vc$vcov[vc$grp == "county"],
      residual = vc$vcov[vc$grp == "Residual"]
    ),
    sigma = sigma(fit),
    loglik = as.numeric(logLik(fit)),
    REML_crit = as.numeric(REMLcrit(fit)),
    aic = AIC(fit),
    bic = BIC(fit),
    converged = (fit@optinfo$conv$opt == 0),
    random_effects = as.list(re),
    random_effects_names = rownames(ranef(fit)$county),
    fitted_values = as.numeric(fitted(fit)),
    residuals = as.numeric(residuals(fit))
  )
}

# ---------------------------------------------------------------------------
# Model 1: y ~ u.full + (1 | county) — no floor predictor
# ---------------------------------------------------------------------------
cat("--- Model 1: no floor ---\n")
M_nox <- lmer(y ~ u.full + (1 | county))
print(summary(M_nox))

# ---------------------------------------------------------------------------
# Model 2: y ~ u.full + x + (1 | county) — add floor
# ---------------------------------------------------------------------------
cat("\n--- Model 2: add floor ---\n")
M_floor <- lmer(y ~ u.full + x + (1 | county))
print(summary(M_floor))

# ---------------------------------------------------------------------------
# Model 3: y ~ u.full + x + x.mean.full + (1 | county) — add county mean floor
# ---------------------------------------------------------------------------
cat("\n--- Model 3: add county mean floor ---\n")
M_xbar <- lmer(y ~ u.full + x + x.mean.full + (1 | county))
print(summary(M_xbar))

# ---------------------------------------------------------------------------
# Compile results
# ---------------------------------------------------------------------------
result <- list(
  description = "Ch 21: Partial pooling, R-squared, and the variance paradox",
  county_x_mean = as.list(x.mean),
  models = list(
    M_nox = extract_model(M_nox, "M_nox"),
    M_floor = extract_model(M_floor, "M_floor"),
    M_xbar = extract_model(M_xbar, "M_xbar")
  )
)

write(toJSON(result, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch21_results.json"))

cat("\nCh 21 fixtures written to", out_dir, "\n")
cat("\nVariance comparison (the paradox):\n")
cat(sprintf("  M_nox   (no floor):  sigma_county = %.4f, sigma_y = %.4f\n",
    sqrt(VarCorr(M_nox)$county[1]), sigma(M_nox)))
cat(sprintf("  M_floor (+ floor):   sigma_county = %.4f, sigma_y = %.4f\n",
    sqrt(VarCorr(M_floor)$county[1]), sigma(M_floor)))
cat(sprintf("  M_xbar  (+ x.mean): sigma_county = %.4f, sigma_y = %.4f\n",
    sqrt(VarCorr(M_xbar)$county[1]), sigma(M_xbar)))
cat("\nNote: Adding floor INCREASES sigma_county! This is the Section 21.7 paradox.\n")
