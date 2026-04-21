#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Gelman & Hill Ch 17 models.
#
# Ch 17 covers varying-intercept + varying-slope models fitted via BUGS.
# We generate lmer/glmer reference fits for:
#
#   1. Radon varying-slope model with correlation visualization data
#      (supplements Ch 13 M3 — same model, adds county-level slope summaries)
#
#   2. Election88 multilevel logistic regression (Section 17.4)
#      bush ~ female + black + female:black + v.prev.full +
#             (1 | state) + (1 | age) + (1 | edu)
#      Binomial GLMM with multiple crossed random effects.
#
# Outputs:
#   tests/fixtures/gh_ch17_radon_slopes.json     — slope summaries for visualization
#   tests/fixtures/gh_ch17_election_data.csv     — cleaned election88 data
#   tests/fixtures/gh_ch17_election_results.json — glmer reference values
#
# Usage:  Rscript tests/fixtures/gen_gelman_hill_ch17.R
#
# Requires: lme4, jsonlite, foreign

library(lme4)
library(jsonlite)
library(foreign)

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# 1. Radon varying-slope model — county-level slope summaries
# ---------------------------------------------------------------------------
cat("--- Radon varying-slopes (supplements Ch 13 M3) ---\n")

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

# County-level uranium
cty <- read.table(
  "hlm/applied regression and multilevel modeling/Book_Codes/Ch.12/cty.dat",
  header = TRUE, sep = ","
)
srrs2.fips <- srrs2$stfips * 1000 + srrs2$cntyfips
usa.fips <- 1000 * cty[, "stfips"] + cty[, "ctfips"]
usa.rows <- match(unique(srrs2.fips[mn]), usa.fips)
u <- log(cty[usa.rows, "Uppm"])
u.full <- u[county]

# Fit M3: y ~ x + (1 + x | county)
M3 <- lmer(y ~ x + (1 + x | county))
vc <- VarCorr(M3)
cc <- coef(M3)$county

slopes_result <- list(
  description = "Ch 17: County-level intercept-slope summaries from M3",
  formula = "y ~ x + (1 + x | county)",
  # Correlation between random intercept and slope
  cor_intercept_slope = attr(vc$county, "correlation")[1, 2],
  # Variance-covariance matrix
  vcov_matrix = list(
    intercept_var = as.numeric(vc$county[1, 1]),
    slope_var = as.numeric(vc$county[2, 2]),
    covariance = as.numeric(vc$county[1, 2])
  ),
  # County-level conditional coefficients
  county_intercepts = as.list(cc[, "(Intercept)"]),
  county_slopes = as.list(cc[, "x"]),
  county_names = trimws(uniq),
  county_n = as.list(as.numeric(table(county))),
  county_uranium = as.list(u)
)

write(toJSON(slopes_result, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch17_radon_slopes.json"))
cat("Radon slopes fixture written\n\n")

# ---------------------------------------------------------------------------
# 2. Election88 multilevel logistic regression (Section 17.4)
# ---------------------------------------------------------------------------
cat("--- Election88 multilevel logistic ---\n")

polls <- read.table(
  "hlm/applied regression and multilevel modeling/Book_Codes/Ch.17/polls.subset.dat",
  header = TRUE
)

# Region mapping (from book code)
data(state)
state.abbr <- c(state.abb[1:8], "DC", state.abb[9:50])
region <- c(3,4,4,3,4,4,1,1,5,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,
            1,4,1,3,2,2,3,4,1,1,3,2,3,3,4,1,3,4,1,2,4)

# Previous vote as state-level predictor
presvote <- read.dta(
  "hlm/applied regression and multilevel modeling/Book_Codes/Ch.17/presvote.dta"
)
v.prev <- presvote$g76_84pr

# Remove rows with missing bush response
ok <- !is.na(polls$bush)
election_df <- data.frame(
  bush = polls$bush[ok],
  state = polls$state[ok],
  female = polls$female[ok],
  black = polls$black[ok],
  age = polls$age[ok],
  edu = polls$edu[ok],
  region = region[polls$state[ok]],
  v_prev = v.prev[polls$state[ok]],
  state_abbr = state.abbr[polls$state[ok]],
  stringsAsFactors = FALSE
)

# Export data
write.csv(election_df, file.path(out_dir, "gh_ch17_election_data.csv"), row.names = FALSE)
cat("Election data:", nrow(election_df), "obs,",
    length(unique(election_df$state)), "states,",
    length(unique(election_df$age)), "age groups,",
    length(unique(election_df$edu)), "edu levels\n")

# Fit glmer: bush ~ female + black + female:black + v_prev +
#                    (1 | state) + (1 | age) + (1 | edu)
cat("Fitting glmer...\n")
M_election <- glmer(
  bush ~ female + black + female:black + v_prev +
    (1 | state) + (1 | age) + (1 | edu),
  data = election_df,
  family = binomial(link = "logit")
)
cat("Election model fitted\n")
print(summary(M_election))

fe <- fixef(M_election)
fe_se <- sqrt(diag(as.matrix(vcov(M_election))))
names(fe_se) <- names(fe)
vc <- as.data.frame(VarCorr(M_election))
re <- ranef(M_election)

election_result <- list(
  description = "Ch 17.4: Election88 multilevel logistic regression",
  formula = "bush ~ female + black + female:black + v_prev + (1|state) + (1|age) + (1|edu)",
  family = "binomial",
  link = "logit",
  nobs = nobs(M_election),
  ngroups = list(
    state = length(unique(election_df$state)),
    age = length(unique(election_df$age)),
    edu = length(unique(election_df$edu))
  ),
  fixed_effects = as.list(fe),
  fixed_effects_se = as.list(fe_se),
  variance_components = list(
    state = vc$vcov[vc$grp == "state"],
    age = vc$vcov[vc$grp == "age"],
    edu = vc$vcov[vc$grp == "edu"]
  ),
  theta = as.numeric(getME(M_election, "theta")),
  loglik = as.numeric(logLik(M_election)),
  aic = AIC(M_election),
  bic = BIC(M_election),
  deviance = as.numeric(deviance(M_election)),
  converged = (M_election@optinfo$conv$opt == 0),
  # Random effects
  ranef_state = as.list(re$state[, 1]),
  ranef_state_names = rownames(re$state),
  ranef_age = as.list(re$age[, 1]),
  ranef_age_names = rownames(re$age),
  ranef_edu = as.list(re$edu[, 1]),
  ranef_edu_names = rownames(re$edu),
  # State-level data for plotting
  state_abbrs = state.abbr,
  state_v_prev = as.list(v.prev),
  state_regions = as.list(region)
)

write(toJSON(election_result, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch17_election_results.json"))
cat("Election results written\n\n")

cat("All Ch 17 fixtures written to", out_dir, "\n")
