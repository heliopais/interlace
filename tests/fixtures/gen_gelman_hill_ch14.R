#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Gelman & Hill Ch 14 models.
#
# Models:
#   M1: y ~ black + female + (1 | state), family=binomial
#   M2: y ~ black + female + black:female + v.prev.full +
#       (1|age) + (1|edu) + (1|age.edu) + (1|state) + (1|region.full),
#       family=binomial
#
# Usage:  Rscript tests/fixtures/gen_gelman_hill_ch14.R

library(lme4)
library(jsonlite)
library(foreign)

out_dir <- "tests/fixtures"
data_dir <- "hlm/applied regression and multilevel modeling/ARM_Data/election88"

# ---------------------------------------------------------------------------
# Helper for GLMM results
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

  # Random effects
  for (grp_name in names(re)) {
    df <- re[[grp_name]]
    re_vals <- df[, 1]
    names(re_vals) <- rownames(df)
    result[[paste0("ranef_", grp_name)]] <- as.list(re_vals)
  }

  # Ngroups
  ngrps <- list()
  for (grp_name in names(re)) {
    ngrps[[grp_name]] <- nrow(re[[grp_name]])
  }
  result$ngroups <- ngrps

  result
}

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
cat("--- Loading election data ---\n")

# Region coding (book's definition)
region <- c(3,4,4,3,4,4,1,1,5,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,
            1,4,1,3,2,2,3,4,1,1,3,2,3,3,4,1,3,4,1,2,4)

# Read polls subset
polls <- read.table(file.path(data_dir, "polls.subset.dat"), header = TRUE)

# Remove undecided (NA in bush)
polls <- polls[!is.na(polls$bush), ]
cat("Poll respondents (non-NA):", nrow(polls), "\n")

# Define variables
y <- polls$bush
n <- length(y)
state <- polls$state
age <- polls$age
edu <- polls$edu
female <- polls$female
black <- polls$black

n.age <- max(age)
n.edu <- max(edu)
n.state <- max(state)

# Age-education interaction
age.edu <- n.edu * (age - 1) + edu

# Region expanded to observation level
region.full <- region[state]

# Previous vote
presvote <- read.dta(file.path(data_dir, "presvote.dta"))
v.prev <- presvote$g76_84pr
v.prev.full <- v.prev[state]

# Build clean data frame
election_df <- data.frame(
  y = y,
  black = black,
  female = female,
  state = state,
  age = age,
  edu = edu,
  age_edu = age.edu,
  region = region.full,
  v_prev = v.prev.full,
  stringsAsFactors = FALSE
)

write.csv(election_df, file.path(out_dir, "gh_ch14_election_data.csv"), row.names = FALSE)
cat("Wrote", nrow(election_df), "observations\n")

# Also export state-level data
state_df <- data.frame(
  state = 1:n.state,
  region = region,
  v_prev = v.prev,
  stringsAsFactors = FALSE
)
write.csv(state_df, file.path(out_dir, "gh_ch14_state_data.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 2. M1: y ~ black + female + (1 | state), binomial
# ---------------------------------------------------------------------------
cat("--- Fitting M1 ---\n")
M1 <- glmer(y ~ black + female + (1 | state),
            data = election_df, family = binomial)
cat("M1:\n"); print(summary(M1))

res_M1 <- extract_glmer(M1, "M1")
write(toJSON(res_M1, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch14_M1_results.json"))
cat("M1 written\n\n")

# ---------------------------------------------------------------------------
# 3. M2: Full model with 5 crossed random effects
# ---------------------------------------------------------------------------
cat("--- Fitting M2 ---\n")
# Convert grouping factors to factors for glmer
election_df$age_f <- factor(election_df$age)
election_df$edu_f <- factor(election_df$edu)
election_df$age_edu_f <- factor(election_df$age_edu)
election_df$state_f <- factor(election_df$state)
election_df$region_f <- factor(election_df$region)

M2 <- glmer(y ~ black + female + black:female + v_prev +
              (1 | age_f) + (1 | edu_f) + (1 | age_edu_f) +
              (1 | state_f) + (1 | region_f),
            data = election_df, family = binomial)
cat("M2:\n"); print(summary(M2))

res_M2 <- extract_glmer(M2, "M2")
write(toJSON(res_M2, auto_unbox = TRUE, digits = 10, pretty = TRUE),
      file.path(out_dir, "gh_ch14_M2_results.json"))
cat("M2 written\n\n")

cat("All Ch 14 fixtures written to", out_dir, "\n")
