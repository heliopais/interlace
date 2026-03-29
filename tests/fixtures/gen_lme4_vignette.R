#!/usr/bin/env Rscript
# Generate lme4 vignette parity fixtures using canonical lme4 datasets.
# Outputs: lme4_sleepstudy_{data,results}.{csv,json}
#          lme4_dyestuff_{data,results}.{csv,json}
#          lme4_pastes_{data,results}.{csv,json}
#
# Run from repo root:
#   Rscript tests/fixtures/gen_lme4_vignette.R

library(lme4)
library(jsonlite)

out_dir <- "tests/fixtures"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
extract_results <- function(fit) {
  fe  <- fixef(fit)
  vc  <- as.data.frame(VarCorr(fit))
  re  <- ranef(fit)
  list(
    fe_params  = as.list(fe),
    vc_table   = vc,          # full table; Python side picks what it needs
    ranef      = lapply(re, function(df) {
      # Convert each grouping factor's BLUP data.frame to a named list per column
      lapply(as.list(df), function(col) { v <- col; names(v) <- rownames(df); as.list(v) })
    }),
    resid_cond = as.numeric(residuals(fit, type = "response")),
    fitted     = as.numeric(fitted(fit))
  )
}

# ---------------------------------------------------------------------------
# 1. sleepstudy  —  Reaction ~ Days + (Days | Subject)
# ---------------------------------------------------------------------------
data(sleepstudy)

fit_sleep <- lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy, REML = TRUE)

write.csv(sleepstudy, file.path(out_dir, "lme4_sleepstudy_data.csv"), row.names = FALSE)

res_sleep <- extract_results(fit_sleep)

# Also store the full 2×2 covariance matrix for Subject RE
cov_subj  <- VarCorr(fit_sleep)$Subject
res_sleep$cov_subject <- lapply(as.data.frame(cov_subj), as.list)
res_sleep$cor_subject <- attr(cov_subj, "correlation")[1, 2]   # Days–Intercept correlation

write_json(res_sleep,
           file.path(out_dir, "lme4_sleepstudy_results.json"),
           digits = 12, auto_unbox = TRUE)
cat("sleepstudy done\n")

# ---------------------------------------------------------------------------
# 2. Dyestuff  —  Yield ~ 1 + (1 | Batch)
# ---------------------------------------------------------------------------
data(Dyestuff)

fit_dye <- lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = TRUE)

write.csv(Dyestuff, file.path(out_dir, "lme4_dyestuff_data.csv"), row.names = FALSE)

res_dye <- extract_results(fit_dye)
write_json(res_dye,
           file.path(out_dir, "lme4_dyestuff_results.json"),
           digits = 12, auto_unbox = TRUE)
cat("Dyestuff done\n")

# ---------------------------------------------------------------------------
# 3. Pastes  —  strength ~ 1 + (1 | batch/cask)
#    lme4 expands this as (1|batch) + (1|batch:cask)
# ---------------------------------------------------------------------------
data(Pastes)

fit_paste <- lmer(strength ~ 1 + (1 | batch/cask), data = Pastes, REML = TRUE)

write.csv(Pastes, file.path(out_dir, "lme4_pastes_data.csv"), row.names = FALSE)

res_paste <- extract_results(fit_paste)
write_json(res_paste,
           file.path(out_dir, "lme4_pastes_results.json"),
           digits = 12, auto_unbox = TRUE)
cat("Pastes done\n")

cat("\nAll fixtures written to", out_dir, "\n")
