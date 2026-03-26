#!/usr/bin/env Rscript
# Generate 3-RE validation fixture: 3000 obs, 50 firms x 12 depts x 8 regions
# Model: y ~ x1 + x2 + (1|firm) + (1|dept) + (1|region)
# Outputs: three_re_data.csv, three_re_r_results.json

library(lme4)
library(jsonlite)

set.seed(456)

n_firms   <- 50
n_depts   <- 12
n_regions <- 8
n         <- 3000

firm_ids   <- sample(paste0("f", seq_len(n_firms)),   n, replace = TRUE)
dept_ids   <- sample(paste0("d", seq_len(n_depts)),   n, replace = TRUE)
region_ids <- sample(paste0("r", seq_len(n_regions)), n, replace = TRUE)
x1 <- rnorm(n)
x2 <- rnorm(n)

# True params
intercept <- 1.5
beta_x1   <- 0.7
beta_x2   <- -0.4
sigma_u   <- 0.9   # firm SD
sigma_v   <- 0.5   # dept SD
sigma_w   <- 0.3   # region SD
sigma_e   <- 0.4   # residual SD

u_firm   <- rnorm(n_firms,   0, sigma_u); names(u_firm)   <- paste0("f", seq_len(n_firms))
u_dept   <- rnorm(n_depts,   0, sigma_v); names(u_dept)   <- paste0("d", seq_len(n_depts))
u_region <- rnorm(n_regions, 0, sigma_w); names(u_region) <- paste0("r", seq_len(n_regions))

y <- intercept + beta_x1 * x1 + beta_x2 * x2 +
     u_firm[firm_ids] + u_dept[dept_ids] + u_region[region_ids] +
     rnorm(n, 0, sigma_e)

df <- data.frame(y = y, x1 = x1, x2 = x2, firm = firm_ids, dept = dept_ids, region = region_ids)
write.csv(df, "tests/fixtures/three_re_data.csv", row.names = FALSE)

# Fit with lme4 REML
fit <- lmer(y ~ x1 + x2 + (1 | firm) + (1 | dept) + (1 | region), data = df, REML = TRUE)

fe <- fixef(fit)

vc <- as.data.frame(VarCorr(fit))
var_firm   <- vc$vcov[vc$grp == "firm"]
var_dept   <- vc$vcov[vc$grp == "dept"]
var_region <- vc$vcov[vc$grp == "region"]
var_resid  <- vc$vcov[vc$grp == "Residual"]

re <- ranef(fit)
blups_firm   <- re$firm[, 1];   names(blups_firm)   <- rownames(re$firm)
blups_dept   <- re$dept[, 1];   names(blups_dept)   <- rownames(re$dept)
blups_region <- re$region[, 1]; names(blups_region) <- rownames(re$region)

resid_cond <- as.numeric(residuals(fit, type = "response"))

results <- list(
  fe_params    = as.list(fe),
  var_firm     = var_firm,
  var_dept     = var_dept,
  var_region   = var_region,
  var_resid    = var_resid,
  blups_firm   = as.list(blups_firm),
  blups_dept   = as.list(blups_dept),
  blups_region = as.list(blups_region),
  resid_cond   = resid_cond
)

write_json(results, "tests/fixtures/three_re_r_results.json", digits = 12, auto_unbox = TRUE)
cat("Done. Wrote three_re_data.csv and three_re_r_results.json\n")
