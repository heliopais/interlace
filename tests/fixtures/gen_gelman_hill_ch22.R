#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Gelman & Hill Ch 22 ANOVA models.
#
# Ch 22 reframes ANOVA as a multilevel model. We generate:
#   1. Classical ANOVA results on the pilots data (two-way: treatment x scenario)
#   2. Multilevel lmer fits for comparison (reuses Ch 13 pilots model)
#   3. Radon no-predictor model: lmer vs county-as-fixed-effect
#
# Outputs:
#   tests/fixtures/gh_ch22_pilots_anova.json  — classical ANOVA + lmer comparison
#   tests/fixtures/gh_ch22_radon_anova.json   — radon one-way ANOVA + lmer comparison
#
# Usage:  Rscript tests/fixtures/gen_gelman_hill_ch22.R
#
# Requires: lme4, jsonlite

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# 1. Pilots data: classical two-way ANOVA vs crossed random effects
# ---------------------------------------------------------------------------

# Load pilots data (same cleaning as Ch 13)
pilots <- read.table(
  file.path("hlm/applied regression and multilevel modeling/Book_Codes/Ch.13/pilots.dat"),
  header = TRUE
)

group.names <- as.vector(unique(pilots$group))
scenario.names <- as.vector(unique(pilots$scenario))
n.group <- length(group.names)
n.scenario <- length(scenario.names)

successes <- NULL
failures <- NULL
group.id <- NULL
scenario.id <- NULL

for (j in 1:n.group) {
  for (k in 1:n.scenario) {
    ok <- pilots$group == group.names[j] & pilots$scenario == scenario.names[k]
    successes <- c(successes, sum(pilots$recovered[ok] == 1, na.rm = TRUE))
    failures <- c(failures, sum(pilots$recovered[ok] == 0, na.rm = TRUE))
    group.id <- c(group.id, j)
    scenario.id <- c(scenario.id, k)
  }
}

y <- successes / (successes + failures)
pilot_df <- data.frame(
  y = y,
  group_id = group.id,
  scenario_id = scenario.id,
  stringsAsFactors = FALSE
)

# Classical two-way ANOVA (fixed effects)
aov_fit <- aov(y ~ factor(group_id) + factor(scenario_id), data = pilot_df)
aov_summary <- summary(aov_fit)[[1]]

classical_anova <- list(
  source = rownames(aov_summary),
  df = as.numeric(aov_summary[, "Df"]),
  sum_sq = as.numeric(aov_summary[, "Sum Sq"]),
  mean_sq = as.numeric(aov_summary[, "Mean Sq"]),
  f_value = as.numeric(aov_summary[, "F value"]),
  p_value = as.numeric(aov_summary[, "Pr(>F)"])
)

# No-pooling (fixed effects) group means
group_means_nopooling <- tapply(y, pilot_df$group_id, mean)
scenario_means_nopooling <- tapply(y, pilot_df$scenario_id, mean)
grand_mean <- mean(y)

# Multilevel model (crossed random effects)
M_mlm <- lmer(y ~ 1 + (1 | group_id) + (1 | scenario_id), data = pilot_df)
fe <- fixef(M_mlm)
vc <- as.data.frame(VarCorr(M_mlm))
re_group <- ranef(M_mlm)$group_id[, 1]
re_scenario <- ranef(M_mlm)$scenario_id[, 1]

# Partial pooling estimates (intercept + RE)
group_means_pooling <- as.numeric(fe["(Intercept)"]) + re_group
scenario_means_pooling <- as.numeric(fe["(Intercept)"]) + re_scenario

pilots_result <- list(
  description = "Ch 22: Two-way ANOVA on pilots data — classical vs multilevel",
  nobs = nrow(pilot_df),
  n_groups = n.group,
  n_scenarios = n.scenario,
  grand_mean = grand_mean,
  classical_anova = classical_anova,
  no_pooling = list(
    group_means = as.list(as.numeric(group_means_nopooling)),
    scenario_means = as.list(as.numeric(scenario_means_nopooling))
  ),
  multilevel = list(
    intercept = as.numeric(fe["(Intercept)"]),
    sigma = sigma(M_mlm),
    variance_components = list(
      group_id = vc$vcov[vc$grp == "group_id"],
      scenario_id = vc$vcov[vc$grp == "scenario_id"],
      residual = vc$vcov[vc$grp == "Residual"]
    ),
    group_means_partial_pooling = as.list(as.numeric(group_means_pooling)),
    scenario_means_partial_pooling = as.list(as.numeric(scenario_means_pooling)),
    loglik = as.numeric(logLik(M_mlm)),
    REML_crit = as.numeric(REMLcrit(M_mlm)),
    theta = as.numeric(getME(M_mlm, "theta")),
    converged = (M_mlm@optinfo$conv$opt == 0)
  )
)

write(toJSON(pilots_result, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch22_pilots_anova.json"))
cat("Pilots ANOVA fixture written\n")

# ---------------------------------------------------------------------------
# 2. Radon data: one-way ANOVA (county) vs varying-intercept
# ---------------------------------------------------------------------------

# Load radon data (same cleaning as Ch 12)
srrs2 <- read.table(
  file.path("hlm/applied regression and multilevel modeling/Book_Codes/Ch.12/srrs2.dat"),
  header = TRUE, sep = ","
)

mn <- srrs2$state == "MN"
radon <- srrs2$activity[mn]
log.radon <- log(ifelse(radon == 0, 0.1, radon))
y_radon <- log.radon
county.name <- as.vector(srrs2$county[mn])
uniq <- unique(county.name)
J <- length(uniq)
county <- rep(NA, J)
for (i in 1:J) {
  county[county.name == uniq[i]] <- i
}

radon_df <- data.frame(
  y = y_radon,
  county = factor(county),
  stringsAsFactors = FALSE
)

# Classical one-way ANOVA (county as fixed)
aov_radon <- aov(y ~ county, data = radon_df)
aov_radon_summary <- summary(aov_radon)[[1]]

radon_classical <- list(
  source = rownames(aov_radon_summary),
  df = as.numeric(aov_radon_summary[, "Df"]),
  sum_sq = as.numeric(aov_radon_summary[, "Sum Sq"]),
  mean_sq = as.numeric(aov_radon_summary[, "Mean Sq"]),
  f_value = as.numeric(aov_radon_summary[, "F value"]),
  p_value = as.numeric(aov_radon_summary[, "Pr(>F)"])
)

# No-pooling county means
county_means_nopooling <- tapply(y_radon, county, mean)
county_n <- tapply(y_radon, county, length)

# Multilevel model
M_radon <- lmer(y ~ 1 + (1 | county), data = radon_df)
fe_radon <- fixef(M_radon)
vc_radon <- as.data.frame(VarCorr(M_radon))
re_radon <- ranef(M_radon)$county[, 1]
county_means_pooling <- as.numeric(fe_radon["(Intercept)"]) + re_radon

# Complete pooling (ignore county)
complete_pooling_mean <- mean(y_radon)

radon_result <- list(
  description = "Ch 22: One-way ANOVA on radon data — classical vs multilevel",
  nobs = nrow(radon_df),
  n_counties = J,
  complete_pooling_mean = complete_pooling_mean,
  classical_anova = radon_classical,
  no_pooling = list(
    county_means = as.list(as.numeric(county_means_nopooling)),
    county_n = as.list(as.numeric(county_n))
  ),
  multilevel = list(
    intercept = as.numeric(fe_radon["(Intercept)"]),
    sigma = sigma(M_radon),
    variance_components = list(
      county = vc_radon$vcov[vc_radon$grp == "county"],
      residual = vc_radon$vcov[vc_radon$grp == "Residual"]
    ),
    county_means_partial_pooling = as.list(as.numeric(county_means_pooling)),
    loglik = as.numeric(logLik(M_radon)),
    REML_crit = as.numeric(REMLcrit(M_radon)),
    theta = as.numeric(getME(M_radon, "theta")),
    converged = (M_radon@optinfo$conv$opt == 0)
  )
)

write(toJSON(radon_result, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch22_radon_anova.json"))
cat("Radon ANOVA fixture written\n")

cat("\nAll Ch 22 fixtures written to", out_dir, "\n")
