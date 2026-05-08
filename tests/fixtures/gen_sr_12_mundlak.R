#!/usr/bin/env Rscript
# Generate Mundlak DGP + lme4 reference fixtures (interlace-x52r).
#
# DGP (McElreath, scripts/12_bonus_mundlak.r, seed 8672):
#   30 groups, 200 individuals. Each individual lives in one group.
#   Ug ~ Normal(1.5, 1)         # unobserved group confound
#   X  ~ Normal(Ug[g], 1)       # individual trait, correlated with group
#   Z  ~ Normal(0, 1) per group # observed group-level covariate
#   Y  ~ Bernoulli(invlogit(a0 + 1*X + Ug[g] + bZY*Z[g]))
#   a0 = -2, bZY = -0.5, true bxy = 1.
#
# Models written as fixtures (frequentist counterparts):
#   M0_naive: Y ~ X + Zg                                   (glm, no FE)
#   M0_fe   : Y ~ X + Zg + factor(g)                       (glm, fixed-effect dummies)
#   M1_re   : Y ~ X + Zg + (1 | g)                         (glmer, random intercept)
#   M2_mund : Y ~ X + Xbar + Zg + (1 | g)                  (glmer, Mundlak machine)
#
# Usage: Rscript tests/fixtures/gen_sr_12_mundlak.R

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# 1. DGP
# ---------------------------------------------------------------------------
set.seed(8672)
N_groups <- 30
N_id <- 200
a0 <- -2
bZY <- -0.5

g <- sample(1:N_groups, size = N_id, replace = TRUE)
Ug <- rnorm(N_groups, mean = 1.5, sd = 1)
X <- rnorm(N_id, mean = Ug[g], sd = 1)
Z <- rnorm(N_groups, mean = 0, sd = 1)
Zg <- Z[g]
Ug_obs <- Ug[g]
inv_logit <- function(x) 1 / (1 + exp(-x))
p <- inv_logit(a0 + 1 * X + Ug_obs + bZY * Zg)
Y <- rbinom(N_id, 1, p)

xbar <- sapply(1:N_groups, function(j) mean(X[g == j]))
Xbar <- xbar[g]

data_df <- data.frame(
  Y = Y, X = X, Zg = Zg, Xbar = Xbar, g = g
)
write.csv(data_df, file.path(out_dir, "sr_12_mundlak_data.csv"), row.names = FALSE)
cat(sprintf("Wrote %d obs in %d groups\n", nrow(data_df),
            length(unique(data_df$g))))

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
extract_glm <- function(fit, model_name) {
  coefs <- coef(summary(fit))
  fe <- coefs[, 1]
  fe_se <- coefs[, 2]
  list(
    model = model_name,
    formula = deparse(formula(fit)),
    nobs = nobs(fit),
    family = "binomial",
    link = "logit",
    fixed_effects = as.list(fe),
    fixed_effects_se = as.list(fe_se),
    loglik = as.numeric(logLik(fit))
  )
}

extract_glmer <- function(fit, model_name) {
  fe <- fixef(fit)
  fe_se <- sqrt(diag(as.matrix(vcov(fit))))
  names(fe_se) <- names(fe)
  vc <- as.data.frame(VarCorr(fit))
  re <- ranef(fit)

  vc_list <- list()
  for (grp in unique(vc$grp)) {
    if (grp == "Residual") next
    sub <- vc[vc$grp == grp & is.na(vc$var2), ]
    for (i in seq_len(nrow(sub))) {
      vc_list[[grp]] <- sub$vcov[i]
    }
  }
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
    variance_components = vc_list,
    converged = (fit@optinfo$conv$opt == 0)
  )
  for (grp_name in names(re)) {
    df <- re[[grp_name]]
    re_vals <- df[, 1]
    names(re_vals) <- rownames(df)
    result[[paste0("ranef_", grp_name)]] <- as.list(re_vals)
  }
  result
}

dump_json <- function(obj, name) {
  write(toJSON(obj, auto_unbox = TRUE, digits = 10, pretty = TRUE),
        file.path(out_dir, sprintf("sr_12_mundlak_%s_results.json", name)))
}

# ---------------------------------------------------------------------------
# 3. Fit four models
# ---------------------------------------------------------------------------
data_df$gf <- factor(data_df$g)

cat("--- M0_naive (glm) ---\n")
M0_naive <- glm(Y ~ X + Zg, data = data_df, family = binomial)
print(coef(summary(M0_naive)))
dump_json(extract_glm(M0_naive, "M0_naive"), "M0_naive")

cat("--- M0_fe (glm with as.factor(g)) ---\n")
M0_fe <- glm(Y ~ X + Zg + gf, data = data_df, family = binomial)
print(head(coef(summary(M0_fe)), 4))
dump_json(extract_glm(M0_fe, "M0_fe"), "M0_fe")

cat("--- M1_re (glmer, random intercept) ---\n")
M1_re <- glmer(Y ~ X + Zg + (1 | g), data = data_df, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
print(summary(M1_re))
dump_json(extract_glmer(M1_re, "M1_re"), "M1_re")

cat("--- M2_mund (glmer + Xbar = Mundlak machine) ---\n")
M2_mund <- glmer(Y ~ X + Xbar + Zg + (1 | g), data = data_df, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))
print(summary(M2_mund))
dump_json(extract_glmer(M2_mund, "M2_mund"), "M2_mund")

cat("\nTrue values: a0=-2, bxy=1, bzy=-0.5\n")
cat("All SR ch12 mundlak fixtures written to", out_dir, "\n")
