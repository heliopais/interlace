#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Gelman & Hill Ch 13 models.
#
# Models:
#   M3: y ~ x + (1 + x | county)                        — radon varying slopes
#   M4: y ~ x + u.full + x:u.full + (1 + x | county)   — radon with group predictors
#   earnings_M1: y ~ x + (1 + x | eth)                  — earnings by height/ethnicity
#   earnings_M2: y ~ x.centered + (1 + x.centered | eth)— centered version
#   pilots: y ~ 1 + (1 | group.id) + (1 | scenario.id)  — crossed non-nested
#
# Usage:  Rscript tests/fixtures/gen_gelman_hill_ch13.R

library(lme4)
library(jsonlite)
library(foreign)  # for read.dta

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
extract_lmer <- function(fit, model_name) {
  fe <- fixef(fit)
  fe_se <- sqrt(diag(as.matrix(vcov(fit))))
  names(fe_se) <- names(fe)
  vc <- as.data.frame(VarCorr(fit))
  re <- ranef(fit)

  result <- list(
    model = model_name,
    formula = deparse(formula(fit)),
    nobs = nobs(fit),
    method = ifelse(isREML(fit), "REML", "ML"),
    fixed_effects = as.list(fe),
    fixed_effects_se = as.list(fe_se),
    sigma = sigma(fit),
    theta = as.numeric(getME(fit, "theta")),
    loglik = as.numeric(logLik(fit)),
    REML_crit = as.numeric(REMLcrit(fit)),
    aic = AIC(fit),
    bic = BIC(fit),
    deviance = as.numeric(deviance(fit, REML = FALSE)),
    converged = (fit@optinfo$conv$opt == 0)
  )

  # Variance components
  vc_list <- list()
  for (grp in unique(vc$grp)) {
    if (grp == "Residual") {
      vc_list[["residual"]] <- vc$vcov[vc$grp == "Residual"][1]
    } else {
      sub <- vc[vc$grp == grp, ]
      # Variances
      var_rows <- sub[is.na(sub$var2), ]
      for (i in seq_len(nrow(var_rows))) {
        key <- paste0(grp, ".", var_rows$var1[i])
        vc_list[[key]] <- var_rows$vcov[i]
      }
      # Covariances/correlations
      cov_rows <- sub[!is.na(sub$var2), ]
      for (i in seq_len(nrow(cov_rows))) {
        key <- paste0(grp, ".", cov_rows$var1[i], ".", cov_rows$var2[i])
        vc_list[[key]] <- cov_rows$vcov[i]
      }
      # Full VarCorr matrix (only for groups with >1 RE term)
      vc_mat <- VarCorr(fit)[[grp]]
      if (nrow(vc_mat) > 1) {
        result[[paste0("cov_", grp)]] <- lapply(as.data.frame(vc_mat), as.list)
        result[[paste0("cor_", grp)]] <- attr(vc_mat, "correlation")[1, 2]
      }
    }
  }
  result$variance_components <- vc_list

  # Random effects per group
  for (grp_name in names(re)) {
    df <- re[[grp_name]]
    re_list <- lapply(as.list(df), function(col) {
      v <- col; names(v) <- rownames(df); as.list(v)
    })
    result[[paste0("ranef_", grp_name)]] <- re_list
  }

  # Conditional coefficients
  cc <- coef(fit)
  for (grp_name in names(cc)) {
    df <- cc[[grp_name]]
    cc_list <- lapply(as.list(df), function(col) {
      v <- col; names(v) <- rownames(df); as.list(v)
    })
    result[[paste0("coef_", grp_name)]] <- cc_list
  }

  # Residuals and fitted
  result$residuals <- as.numeric(residuals(fit, type = "response"))
  result$fitted_values <- as.numeric(fitted(fit))

  result
}

# ---------------------------------------------------------------------------
# 1. Radon data (reuse Ch 12 setup)
# ---------------------------------------------------------------------------
cat("--- Loading radon data ---\n")
srrs2 <- read.table(
  "hlm/applied regression and multilevel modeling/Book_Codes/Ch.12/srrs2.dat",
  header = TRUE, sep = ","
)
mn <- srrs2$state == "MN"
radon <- srrs2$activity[mn]
log.radon <- log(ifelse(radon == 0, 0.1, radon))
floor <- srrs2$floor[mn]
y <- log.radon
x <- floor
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
uranium <- cty[usa.rows, "Uppm"]
u <- log(uranium)
u.full <- u[county]

# ---------------------------------------------------------------------------
# 2. M3: y ~ x + (1 + x | county)
# ---------------------------------------------------------------------------
cat("--- Fitting M3 ---\n")
M3 <- lmer(y ~ x + (1 + x | county))
cat("M3:\n"); print(summary(M3))

res_M3 <- extract_lmer(M3, "M3")
# Add ngroups
res_M3$ngroups <- list(county = length(unique(county)))
write(toJSON(res_M3, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch13_M3_results.json"))
cat("M3 written\n\n")

# ---------------------------------------------------------------------------
# 3. M4: y ~ x + u.full + x:u.full + (1 + x | county)
# ---------------------------------------------------------------------------
cat("--- Fitting M4 ---\n")
M4 <- lmer(y ~ x + u.full + x:u.full + (1 + x | county))
cat("M4:\n"); print(summary(M4))

res_M4 <- extract_lmer(M4, "M4")
res_M4$ngroups <- list(county = length(unique(county)))
write(toJSON(res_M4, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch13_M4_results.json"))
cat("M4 written\n\n")

# ---------------------------------------------------------------------------
# 4. Pilots data — crossed non-nested
# ---------------------------------------------------------------------------
cat("--- Loading pilots data ---\n")
pilots <- read.table(
  "hlm/applied regression and multilevel modeling/Book_Codes/Ch.13/pilots.dat",
  header = TRUE
)

# Aggregate to success rates per group x scenario
group.names <- as.vector(unique(pilots$group))
scenario.names <- as.vector(unique(pilots$scenario))
n.group <- length(group.names)
n.scenario <- length(scenario.names)

successes <- failures <- group.id <- scenario.id <- NULL
for (j in 1:n.group) {
  for (k in 1:n.scenario) {
    ok <- pilots$group == group.names[j] & pilots$scenario == scenario.names[k]
    successes <- c(successes, sum(pilots$recovered[ok] == 1, na.rm = TRUE))
    failures <- c(failures, sum(pilots$recovered[ok] == 0, na.rm = TRUE))
    group.id <- c(group.id, j)
    scenario.id <- c(scenario.id, k)
  }
}
y_pilots <- successes / (successes + failures)

pilots_df <- data.frame(
  y = y_pilots,
  group_id = group.id,
  scenario_id = scenario.id,
  stringsAsFactors = FALSE
)

# Export pilot data
write.csv(pilots_df, file.path(out_dir, "gh_ch13_pilots_data.csv"), row.names = FALSE)

cat("--- Fitting Pilots model ---\n")
M_pilots <- lmer(y ~ 1 + (1 | group_id) + (1 | scenario_id), data = pilots_df)
cat("Pilots:\n"); print(summary(M_pilots))

res_pilots <- extract_lmer(M_pilots, "pilots")
res_pilots$ngroups <- list(
  group_id = length(unique(pilots_df$group_id)),
  scenario_id = length(unique(pilots_df$scenario_id))
)
write(toJSON(res_pilots, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch13_pilots_results.json"))
cat("Pilots written\n\n")

# ---------------------------------------------------------------------------
# 5. Earnings data — height by ethnicity
# ---------------------------------------------------------------------------
cat("--- Loading earnings data ---\n")
heights <- read.dta(
  "hlm/applied regression and multilevel modeling/ARM_Data/earnings/heights.dta"
)

# Clean following book code
age <- 90 - heights$yearbn
age[age < 18] <- NA
eth <- ifelse(heights$race == 2, 1,
        ifelse(heights$hisp == 1, 2,
          ifelse(heights$race == 1, 3, 4)))
ok <- !is.na(heights$earn + heights$height + heights$sex) &
      heights$earn > 0 & heights$yearbn > 25

earn_df <- data.frame(
  y = log(heights$earn[ok]),
  x = heights$height[ok],
  eth = eth[ok],
  stringsAsFactors = FALSE
)
earn_df$x_centered <- earn_df$x - mean(earn_df$x)

# Export earnings data
write.csv(earn_df, file.path(out_dir, "gh_ch13_earnings_data.csv"), row.names = FALSE)
cat("Earnings data:", nrow(earn_df), "obs,", length(unique(earn_df$eth)), "ethnicities\n")

# Model M1: uncentered
cat("--- Fitting Earnings M1 ---\n")
M_earn1 <- lmer(y ~ x + (1 + x | eth), data = earn_df)
cat("Earnings M1:\n"); print(summary(M_earn1))

res_earn1 <- extract_lmer(M_earn1, "earnings_M1")
res_earn1$ngroups <- list(eth = length(unique(earn_df$eth)))
write(toJSON(res_earn1, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch13_earnings_M1_results.json"))
cat("Earnings M1 written\n\n")

# Model M2: centered
cat("--- Fitting Earnings M2 ---\n")
M_earn2 <- lmer(y ~ x_centered + (1 + x_centered | eth), data = earn_df)
cat("Earnings M2:\n"); print(summary(M_earn2))

res_earn2 <- extract_lmer(M_earn2, "earnings_M2")
res_earn2$ngroups <- list(eth = length(unique(earn_df$eth)))
res_earn2$x_mean <- mean(earn_df$x)  # needed to reconstruct uncentered predictions
write(toJSON(res_earn2, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch13_earnings_M2_results.json"))
cat("Earnings M2 written\n\n")

cat("All Ch 13 fixtures written to", out_dir, "\n")
