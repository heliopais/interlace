#!/usr/bin/env Rscript
# Generate lme4 reference fixtures for Statistical Rethinking ch13 chimps.
#
# Models (Bernoulli, logit link, two crossed varying intercepts):
#   M0: pulled_left ~ 1                  + (1 | actor) + (1 | block)
#   M1: pulled_left ~ as.factor(tx)      + (1 | actor) + (1 | block)
#   M2: pulled_left ~ prosoc_left * cond + (1 | actor) + (1 | block)
#
# `tx` is McElreath's 4-level treatment factor: 1 + prosoc_left + 2*condition.
#
# Data source: McElreath rethinking::chimpanzees (canonical CSV vendored from
# rmcelreath/rethinking @ master).
#
# Usage: Rscript tests/fixtures/gen_sr_13_chimps.R

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

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

  vc_list <- list()
  for (grp in unique(vc$grp)) {
    if (grp == "Residual") next
    sub <- vc[vc$grp == grp & is.na(vc$var2), ]
    for (i in seq_len(nrow(sub))) {
      vc_list[[grp]] <- sub$vcov[i]
    }
  }
  result$variance_components <- vc_list

  for (grp_name in names(re)) {
    df <- re[[grp_name]]
    re_vals <- df[, 1]
    names(re_vals) <- rownames(df)
    result[[paste0("ranef_", grp_name)]] <- as.list(re_vals)
  }

  ngrps <- list()
  for (grp_name in names(re)) {
    ngrps[[grp_name]] <- nrow(re[[grp_name]])
  }
  result$ngroups <- ngrps

  result
}

# ---------------------------------------------------------------------------
# 1. Load + prepare frame
# ---------------------------------------------------------------------------
cat("--- Loading chimpanzees ---\n")
df <- read.csv(file.path(out_dir, "sr_13_chimpanzees_data.csv"),
               stringsAsFactors = FALSE)
df$actor <- factor(df$actor)
df$block <- factor(df$block)
# 4-level treatment: (prosoc_left, condition) in {(0,0),(1,0),(0,1),(1,1)}
df$tx <- factor(1 + df$prosoc_left + 2 * df$condition,
                levels = 1:4)
cat(sprintf("Observations: %d  actors: %d  blocks: %d\n",
            nrow(df), nlevels(df$actor), nlevels(df$block)))

# ---------------------------------------------------------------------------
# 2. Fit three models
# ---------------------------------------------------------------------------
fit_and_dump <- function(formula_rhs, name) {
  cat(sprintf("--- Fitting %s ---\n", name))
  form <- as.formula(paste("pulled_left ~", formula_rhs))
  fit <- glmer(form, data = df, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
  print(summary(fit))
  res <- extract_glmer(fit, name)
  write(toJSON(res, auto_unbox = TRUE, digits = 10, pretty = TRUE),
        file.path(out_dir, sprintf("sr_13_chimps_%s_results.json", name)))
  cat(sprintf("%s written\n\n", name))
}

fit_and_dump("1 + (1 | actor) + (1 | block)",                          "M0")
fit_and_dump("tx + (1 | actor) + (1 | block)",                         "M1")
fit_and_dump("prosoc_left * condition + (1 | actor) + (1 | block)",    "M2")

cat("All SR ch13 chimps fixtures written to", out_dir, "\n")
