#!/usr/bin/env Rscript
# Generate 2-RE validation fixture: 2000 obs, 50 firms x 12 depts, y ~ x + (1|firm) + (1|dept)
# Outputs: two_re_data.csv, two_re_r_results.json

library(lme4)
library(jsonlite)

set.seed(123)

n_firms <- 50
n_depts <- 12
n <- 2000

firm_ids <- sample(paste0("f", seq_len(n_firms)), n, replace = TRUE)
dept_ids <- sample(paste0("d", seq_len(n_depts)), n, replace = TRUE)
x        <- rnorm(n)

# True params
intercept <- 3.0
beta_x    <- 0.8
sigma_u   <- 1.0   # firm SD
sigma_v   <- 0.6   # dept SD
sigma_e   <- 0.5   # residual SD

u_firm <- rnorm(n_firms, 0, sigma_u)
u_dept <- rnorm(n_depts, 0, sigma_v)
names(u_firm) <- paste0("f", seq_len(n_firms))
names(u_dept) <- paste0("d", seq_len(n_depts))

y <- intercept + beta_x * x + u_firm[firm_ids] + u_dept[dept_ids] + rnorm(n, 0, sigma_e)

df <- data.frame(y = y, x = x, firm = firm_ids, dept = dept_ids)
write.csv(df, "tests/fixtures/two_re_data.csv", row.names = FALSE)

# Fit with lme4 REML
fit <- lmer(y ~ x + (1 | firm) + (1 | dept), data = df, REML = TRUE)

# Extract fixed effects
fe   <- fixef(fit)

# Variance components
vc   <- as.data.frame(VarCorr(fit))
# vc has columns: grp, var1, var2, vcov, sdcor
var_firm  <- vc$vcov[vc$grp == "firm"]
var_dept  <- vc$vcov[vc$grp == "dept"]
var_resid <- vc$vcov[vc$grp == "Residual"]

# BLUPs
re   <- ranef(fit)
blups_firm <- re$firm[, 1]
names(blups_firm) <- rownames(re$firm)
blups_dept <- re$dept[, 1]
names(blups_dept) <- rownames(re$dept)

# Conditional residuals
resid_cond <- residuals(fit, type = "response")

results <- list(
  fe_params      = as.list(fe),
  var_firm       = var_firm,
  var_dept       = var_dept,
  var_resid      = var_resid,
  blups_firm     = as.list(blups_firm),
  blups_dept     = as.list(blups_dept),
  resid_cond     = as.numeric(resid_cond)
)

write_json(results, "tests/fixtures/two_re_r_results.json", digits = 12, auto_unbox = TRUE)
cat("Done. Wrote two_re_data.csv and two_re_r_results.json\n")
