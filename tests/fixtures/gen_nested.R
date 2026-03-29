#!/usr/bin/env Rscript
# Generate nested random effects parity fixtures.
# Outputs:
#   nested_pastes_data.csv + nested_pastes_r_results.json  (Pastes dataset)
#   nested_synthetic_data.csv + nested_synthetic_r_results.json  (synthetic)

library(lme4)
library(jsonlite)

# ---------------------------------------------------------------------------
# Helper: extract results from a nested lmer fit into the standard structure
# ---------------------------------------------------------------------------
extract_results <- function(fit) {
  fe     <- fixef(fit)
  vcov_f <- as.matrix(vcov(fit))

  vc <- as.data.frame(VarCorr(fit))
  # Build var_components named list from the vc table
  var_components <- list()
  for (i in seq_len(nrow(vc))) {
    grp <- vc$grp[i]
    if (grp == "Residual") {
      var_components[["residual"]] <- vc$vcov[i]
    } else {
      var_components[[grp]] <- vc$vcov[i]
    }
  }

  re      <- ranef(fit)
  ranef_l <- lapply(re, function(df) as.list(setNames(df[, 1], rownames(df))))

  list(
    fixef          = as.list(fe),
    vcov_fixef     = lapply(as.data.frame(vcov_f), as.list),
    ranef          = ranef_l,
    var_components = var_components,
    sigma          = sigma(fit),
    logLik         = as.numeric(logLik(fit))
  )
}

# ---------------------------------------------------------------------------
# 1. Pastes dataset
# ---------------------------------------------------------------------------
load("/Users/paishe01/repos/gpg/lme4/data/Pastes.rda")

# Add explicit batch:cask interaction column
Pastes$batch_cask <- paste(Pastes$batch, Pastes$cask, sep = ":")

write.csv(Pastes, "tests/fixtures/nested_pastes_data.csv", row.names = FALSE)
cat("Wrote nested_pastes_data.csv\n")

fit_pastes <- lmer(strength ~ 1 + (1 | batch) + (1 | batch:cask),
                   data = Pastes, REML = TRUE)
cat("Pastes model summary:\n")
print(summary(fit_pastes))

results_pastes <- extract_results(fit_pastes)
write_json(results_pastes, "tests/fixtures/nested_pastes_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Wrote nested_pastes_r_results.json\n")

# ---------------------------------------------------------------------------
# 2. Synthetic nested dataset: 500 obs, 10 top-level groups, 3 sub-groups each
# ---------------------------------------------------------------------------
set.seed(42)

n_top <- 10
n_sub <- 3
n     <- 500

top_ids <- sample(paste0("g", seq_len(n_top)), n, replace = TRUE)
# sub-group labels nested within top: "g1:s1", "g1:s2", etc.
sub_within <- sample(paste0("s", seq_len(n_sub)), n, replace = TRUE)
nested_ids <- paste(top_ids, sub_within, sep = ":")

intercept <- 5.0
sigma_top  <- 1.2
sigma_nest <- 0.8
sigma_e    <- 0.6

u_top  <- rnorm(n_top, 0, sigma_top)
names(u_top) <- paste0("g", seq_len(n_top))

all_nested <- unique(nested_ids)
u_nest <- rnorm(length(all_nested), 0, sigma_nest)
names(u_nest) <- all_nested

y <- intercept + u_top[top_ids] + u_nest[nested_ids] + rnorm(n, 0, sigma_e)

df_syn <- data.frame(
  y          = y,
  group      = top_ids,
  subgroup   = sub_within,
  group_sub  = nested_ids
)

write.csv(df_syn, "tests/fixtures/nested_synthetic_data.csv", row.names = FALSE)
cat("Wrote nested_synthetic_data.csv\n")

fit_syn <- lmer(y ~ 1 + (1 | group) + (1 | group:subgroup),
                data = df_syn, REML = TRUE)
cat("Synthetic model summary:\n")
print(summary(fit_syn))

results_syn <- extract_results(fit_syn)
write_json(results_syn, "tests/fixtures/nested_synthetic_r_results.json",
           digits = 12, auto_unbox = TRUE)
cat("Wrote nested_synthetic_r_results.json\n")
