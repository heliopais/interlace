# API Reference

A curated index of the `interlace` public API, grouped by task. Each entry
links to the full reference page for that function or class.

---

## Fitting

| Function / class | Description |
|---|---|
| {doc}`fit` | Main entry point — LMM via profiled REML |
| {doc}`glmer` | Generalised linear mixed model (GLMM) via Laplace |
| {doc}`clmm` | Cumulative link mixed model (ordinal outcomes) |
| {doc}`coxme` | Cox proportional hazards with Gaussian frailty |
| {doc}`ols` | Ordinary least squares with HC3 robust SEs |
| {doc}`quantreg` | Quantile regression (LP-based) |
| {doc}`correlation` | Residual correlation structures — `AR1`, `CompoundSymmetry` |
| {doc}`glmm_families` | GLMM family objects — Poisson, Binomial, NegBinomial, Gamma, ZI variants |

---

## Results and summaries

| Function / class | Description |
|---|---|
| {doc}`result` | `CrossedLMEResult` — the object returned by `fit()` |
| {doc}`summary` | `summary()` — formatted coefficient table |
| {doc}`varcorr` | `VarCorr()` — variance component table |
| {doc}`predict` | `predict()` — fitted values and out-of-sample predictions |

---

## Post-estimation

| Function / class | Description |
|---|---|
| {doc}`emmeans` | `emmeans()` — estimated marginal means |
| {doc}`contrast` | `contrast()` — custom contrasts between marginal means |
| {doc}`anova` | `anova()` — likelihood ratio test between nested models |
| {doc}`anova_table` | `Anova()` — type II / type III F-tests |
| {doc}`anova_type` | `anova_type2()` / `anova_type3()` — manual type II / III tables |

---

## Inference and uncertainty

| Function / class | Description |
|---|---|
| {doc}`kenward_roger` | Kenward-Roger denominator DFs and adjusted covariance |
| {doc}`profile_ci` | `profile_confint()` — profile likelihood confidence intervals |
| {doc}`convergence` | `isSingular()` — boundary / convergence diagnostics |
| {doc}`allfit` | `allFit()` — refit with all optimisers and compare |
| {doc}`cross_val` | `cross_val()` — group-aware cross-validation |

---

## Diagnostics

| Function / class | Description |
|---|---|
| {doc}`residuals` | `hlm_resid()` — residuals (marginal, conditional, standardised) |
| {doc}`augment` | `hlm_augment()` — residuals and fitted values as a DataFrame |
| {doc}`leverage` | `leverage()` — observation-level leverage |
| {doc}`influence` | `hlm_influence()` — Cook's D and DFFITS via exact deletion |
| {doc}`lmer_influence` | `lmer_influence_measures()` — group-level influence |
| {doc}`gradient` | Analytical REML gradient utilities |
| {doc}`plotting` | `plot_diagnostics()` — residual and QQ plots |

---

## Simulation and bootstrap

| Function / class | Description |
|---|---|
| {doc}`simulate` | `simulate()` / `bootMer()` — parametric bootstrap and power analysis |
