# Changelog

## v0.2.9 — 2026-04-03

### `glmer()` — generalised linear mixed models via Laplace approximation

interlace now supports **generalised linear mixed models** (GLMMs) through the
new `glmer()` function, targeting parity with R's `lme4::glmer()`.

```python
import interlace

# Binomial GLMM (proportion response with trial weights)
result = interlace.glmer(
    "incidence / size ~ period",
    data=df,
    family="binomial",
    groups="herd",
    weights=df["size"].values,
)

# Poisson GLMM (count response)
result = interlace.glmer(
    "count ~ x1 + x2",
    data=df,
    family="poisson",
    groups=["site", "year"],
)
```

**Supported families:** `"binomial"` (logit link), `"poisson"` (log link),
`"gaussian"` (identity link), or any custom object implementing the
`GLMMFamily` protocol.

**Estimation:** Laplace approximation of the marginal log-likelihood with
penalised iteratively reweighted least squares (PIRLS) for the inner loop
and L-BFGS-B or BOBYQA for the outer optimisation over variance parameters.

**Result object:** `GLMMResult` exposes `fe_params`, `fe_bse`,
`random_effects`, `variance_components`, `aic`, `bic`, `llf`, `converged`,
`fittedvalues`, and a `predict()` method with `type="response"` /
`type="link"` and `include_re` options.

**Validation:** fixed effects match R's `glmer()` within 1e-3 (absolute);
variance components within 5% (relative) on the CBPP and Poisson benchmark
datasets.

See the [GLMM quickstart](glmm-quickstart.md) and the
[glmer API reference](api/glmer.md) for full details.

---

## v0.2.8 — 2026-03-31

### Breaking changes

**`tau_gap()` removed** — `tau_gap()` is a domain-specific metric and has been
removed from `interlace.influence` and `interlace.__init__`. If you relied on
it, please use the equivalent function from the appropriate downstream package.

**Internal attributes renamed** — `_primary_group_col` and
`_secondary_group_cols` are now the canonical attribute names on `LMEResult`
for the primary and secondary grouping factors. Update any downstream code that
accessed the previous internal names directly.

### New public API

**`crossed_structures()` and `statsmodels_structures()`** — These helper
functions were promoted from private to public API. They extract the model
structures needed to dispatch leverage computation to the correct code path and
are part of the stable `interlace` namespace.

### `OLSResult` — new properties

`OLSResult` gains two properties that match the `statsmodels` OLS API:

- `df_resid` — residual degrees of freedom: `n − p`
- `mse_resid` — mean squared error of residuals: `RSS / df_resid`

### `quantreg_ker_se()` — corrected Gaussian kernel estimator

`quantreg_ker_se()` now correctly implements the Hendricks-Koenker Gaussian
kernel sandwich estimator, matching R's `summary.rq(se="ker")` within 2% on
test data. The previous implementation used a simpler sparsity estimator that
did not match R. The bandwidth computation, density estimation, and sandwich
covariance assembly have all been corrected.

### `emmeans()` — estimated marginal means

New top-level function for computing estimated marginal means (EMMs) for a
fitted LMM. Mirrors R's `emmeans` package: for each level combination of the
specified factor(s), the marginal mean of the fixed-effects predictor is
computed after averaging continuous covariates at their observed means.
Standard errors and degrees of freedom use the Satterthwaite approximation.

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])
emm = interlace.emmeans(result, specs="condition")
print(emm)
#   condition  estimate     SE      df   lower   upper  t.ratio  p.value
# 0   control     456.3   12.4    38.2   431.2   481.4     36.8    0.000
# 1      high     502.7   13.1    39.5   476.2   529.2     38.4    0.000
```

Crossed two-way grids and custom covariate values via `at=` are both
supported. See the [emmeans API page](api/emmeans.md) for full details.

### `contrast()` and `pairs()` — linear contrasts with p-value adjustment

`contrast()` applies named linear contrasts to an `emmeans()` result.
`pairs()` is a convenience wrapper for all pairwise differences with Tukey
HSD adjustment by default — matching R's `pairs()` ergonomics.

```python
# All pairwise, Tukey HSD
pw = interlace.pairs(emm)

# Treatment vs. control, Holm correction
tvc = interlace.contrast(emm, method="trt.vs.ctrl", adjust="holm")

# Custom contrast vectors
import numpy as np
custom = interlace.contrast(emm, method={
    "avg(low,high) - control": np.array([-1.0, 0.5, 0.5]),
})
```

Supported `adjust=` methods: `'none'`, `'bonferroni'`, `'holm'`, `'fdr'`,
`'tukey'`. See the [contrast API page](api/contrast.md).

### `Anova()` — car::Anova()-style F and chi-square tables

`car::Anova()`-style per-term ANOVA tables with Satterthwaite denominator
degrees of freedom. Supports Type II and Type III tests and both F and
Wald chi-square statistics.

```python
tbl = interlace.Anova(result)               # Type III F (default)
tbl2 = interlace.Anova(result, type="II")   # Type II F (ML refit internally)
tbl_chi = interlace.Anova(result, test="Chisq")

# Two-model LRT (delegates to interlace.anova)
interlace.Anova([m0, m1])
```

The lower-level `anova_type2()` and `anova_type3()` functions are also
exported for direct use. See the [Anova API page](api/anova_table.md).

### `isSingular()` / `result.is_singular` — boundary detection

Singular fits occur when one or more variance components collapse to zero,
placing the model at the boundary of the parameter space. This can inflate
fixed-effect standard errors and makes profile CIs unreliable.

`interlace.isSingular()` mirrors lme4's function of the same name:

```python
from interlace import isSingular

if isSingular(result):
    print("Warning: model is at a boundary — interpret SEs with caution")
```

The property shorthand and per-factor flags are also available on the result
object:

```python
result.is_singular          # True/False
result.boundary_flags       # {'school_id': False, 'student_id': True}
```

`fit()` now issues a `ConvergenceWarning` automatically when a boundary fit is
detected. See the [Variance Inference Guide](variance-inference.md) for
guidance on what to do when `is_singular` is `True`.

### `result.confint()` — profile likelihood CIs for variance parameters

Profile likelihood confidence intervals for the variance parameters (theta),
matching lme4's `confint(method="profile")`:

```python
ci = result.confint()            # 95 % by default
print(ci)
#                 estimate   2.5 %  97.5 %
# school_id          0.412   0.201   0.731
# student_id         0.638   0.389   1.042
# residual           1.000   1.000   1.000

ci_90 = result.confint(level=0.90)
```

CIs are reported on the theta (relative Cholesky factor) scale. For
intercept-only specs, `sigma_b ≈ theta * sqrt(sigma2)`. If the profile drops
to the boundary before reaching the lower quantile, the lower bound is set to
0. See the [Variance Inference Guide](variance-inference.md) for
interpretation guidance.

### `lmer_influence_measures()` — HLMdiag-parity all-in-one influence

New convenience function that combines case-deletion diagnostics, analytical
leverage, and DFBETAS into a single dict matching R's `HLMdiag::hlm_influence`
output exactly:

```python
measures = interlace.lmer_influence_measures(result, n_jobs=-1)
# Keys: 'cooks', 'hat', 'hat_overall', 'hat_fixef',
#       'dfbetas', 'dffits', 'residuals', 'sigma'
```

`n_jobs=-1` uses all available CPUs for the case-deletion loop; `n_jobs=1`
(default) runs sequentially. `hlm_influence()` also gains `n_jobs` and
`show_progress` parameters.

Pass `analytical=True` to use the **GLS-LOO Woodbury fast path** instead of
O(n) REML refits. This fixes variance components at the full-model estimates
and is thousands of times faster on large datasets while matching
`n_influential` counts exactly (n ≥ 200):

```python
measures = interlace.lmer_influence_measures(result, analytical=True)
# ~3700× faster vs case-deletion on n=2000
```

### `ols()` and `quantreg()` — statsmodels-free fitting

New formulaic-based fitting functions that replace `statsmodels.OLS` and
`statsmodels.QuantReg` without pulling in the statsmodels dependency.
Both accept any narwhals-compatible DataFrame (pandas, polars, …).

```python
# OLS with HC3 robust standard errors
ols_result = interlace.ols("score ~ age + education", data=df)
print(ols_result.params)
se_hc3 = ols_result.hc3_bse()        # HC3 SEs, matches statsmodels to 6 sig figs

# Quantile regression via HiGHS LP solver
qr = interlace.quantreg("score ~ age + education", data=df, tau=0.9)
print(qr.params)
se_ker = qr.ker_se()                  # Hall-Sheather kernel SEs
```

See the [ols API page](api/ols.md) and [quantreg API page](api/quantreg.md).

### `ols_influence_measures()` — single-QR OLS influence diagnostics

New function that computes hat diagonal, Cook's D, DFFITS, DFBETAS,
residuals, and sigma from a **single QR decomposition**, replacing the
previous two-pass approach (~10% faster at n=50,000, p=100):

```python
measures = interlace.ols_influence_measures(ols_result)
# Keys: 'hat', 'cooks', 'dffits', 'dfbetas', 'residuals', 'sigma'
```

### `summary().tables` — statsmodels compatibility

`SummaryResult` gains a `.tables` property returning `[info_df, fe_df]`.
`tables[1]` is a DataFrame with columns `Coef.`, `Std.Err.`, `z`, `P>|z|`,
`[0.025`, `0.975]` — matching `statsmodels.MixedLMResults.summary()` so that
downstream code written against statsmodels works without modification.

### statsmodels-compatible attribute aliases on `CrossedLMEResult`

`CrossedLMEResult` now exposes the following aliases, allowing code written
against statsmodels `MixedLMResults` to work without modification:

| Alias | Maps to |
|---|---|
| `params` | `fe_params` |
| `bse` | `fe_bse` |
| `tvalues` | `fe_tvalues` |
| `pvalues` | `fe_pvalues` |
| `llf_restricted` | `llf` (REML fits) / `None` (ML fits) |

### Analytical REML gradient (`use_gradient=True`)

`fit_reml()` gains a `use_gradient=True` keyword that passes an exact
analytical gradient to L-BFGS-B, reducing objective evaluations by ~3×
compared to the default finite-difference approximation. Gradient matches
finite differences to `rtol < 1e-3` on single- and two-factor crossed models.
See the [gradient API page](api/gradient.md).

---

## v0.2.7 — 2026-03-30

### `cross_val()` — group-aware cross-validation

New top-level function for evaluating out-of-group prediction error without
data leakage. Because observations within a group share a random effect,
splitting at the observation level inflates apparent accuracy; `cross_val`
splits at the *group* level instead.

```python
from interlace import cross_val

cv = cross_val(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups="school_id",           # grouping factor used for RE and for CV folds
    cv="logo",                    # leave-one-group-out (default)
    scoring="rmse",
)
print(f"LOGO CV RMSE: {cv.mean:.3f} ± {cv.std:.3f}")
```

`cv="kfold"` splits groups (not observations) into k folds for faster
estimates when there are many groups:

```python
cv_k = cross_val(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups="school_id",
    cv="kfold",
    k=5,
    scoring="mae",
)
```

Custom scoring functions are also accepted — any callable
`scorer(y_true, y_pred) -> float`. Pass `return_models=True` to inspect
per-fold fitted models, training groups, and predictions.
See the [Cross-Validation Guide](cross-validation.md) for the full workflow.

### `CrossedLMEResult.update()` — lme4-style model refitting

Refit a model with a modified formula, new data, or changed fit arguments
without retyping the full call:

```python
# Add a predictor
m2 = result.update(". ~ . + frequency")

# Remove a predictor
m1 = result.update(". ~ . - prior_gpa")

# Swap the dataset (e.g. sensitivity analysis on a filtered frame)
m_sens = result.update(data=df[df["school_size"] > 200])

# Combine both
m_refit = result.update(". ~ . + frequency", data=df_new, method="ML")
```

Dot notation follows lme4 convention: `.` on either side of `~` expands to
the corresponding part of the original formula. The original `result` object
is not modified; `update()` always returns a new `CrossedLMEResult`.

### `random_effects_se` and `random_effects_ci()` — uncertainty in BLUPs

BLUPs (Best Linear Unbiased Predictions) are point estimates. The new
`random_effects_se` property and `random_effects_ci()` method expose their
posterior standard errors and normal-approximation confidence intervals:

```python
# Per-group standard errors (same structure as random_effects)
se = result.random_effects_se
print(se["school_id"])          # pd.Series for intercept-only specs

# 95 % confidence intervals (default)
ci = result.random_effects_ci()
print(ci["school_id"])
#             lower    upper
# school_01   -8.4      2.1
# school_02    1.3      9.8
# ...

# Custom coverage
ci_90 = result.random_effects_ci(level=0.90)
```

For random-slope models, `random_effects_se` returns a DataFrame with one
column per random-effect term. `random_effects_ci()` uses a MultiIndex
column with `(term, "lower"/"upper")` pairs.
See the [Random Slopes Guide](random-slopes.md#uncertainty-in-blups) for a
caterpillar-plot example.

### `ols_dfbetas_qr()` — vectorised DFBETAS for OLS models

New function in `interlace.influence` for computing DFBETAS on an OLS model
via QR decomposition — no Python loops, fully vectorised:

```python
from interlace.influence import ols_dfbetas_qr
import statsmodels.formula.api as smf

ols = smf.ols("y ~ x1 + x2", data=df).fit()
dfbetas = ols_dfbetas_qr(ols)   # shape (n, p)

# Common rule-of-thumb threshold
import numpy as np
threshold = 2 / np.sqrt(len(df))
influential = np.any(np.abs(dfbetas) > threshold, axis=1)
print(f"{influential.sum()} influential observations")
```

Useful as a pre-check before fitting the mixed model, or for stage-1
diagnostics in two-stage workflows. See the
[Diagnostics Guide](diagnostics.ipynb) for a heatmap visualisation.

---

## v0.2.4 — 2026-03-27

### Nested random effects — `(1|g1/g2)` syntax

The `random` parameter now accepts lme4-style nested shorthand.
`(1|batch/cask)` expands automatically to `(1|batch) + (1|batch:cask)`, and
depth-3 nesting (`(1|a/b/c)`) is also supported.  No pre-derived interaction
column is needed; `interlace` derives it on the fly from the component columns.

```python
result = interlace.fit(
    "strength ~ 1",
    data=df,
    random=["(1|batch/cask)"],
)
print(result.variance_components)  # keys: 'batch', 'batch:cask', 'residual'
print(result.random_effects["batch:cask"])
```

Results match lme4 to the same tolerances as crossed designs
(fixed effects abs_diff < 1e-4, variance components rel_diff < 5%,
BLUP correlation > 0.99).

### `quantreg_ker_se` — Hall-Sheather kernel SE for quantile regression

New utility function `quantreg_ker_se(residuals, X, tau=0.5, hs=True)` that
computes quantile regression standard errors matching R's
`quantreg::summary.rq(se="ker", hs=TRUE)`.

```python
from interlace import quantreg_ker_se

fit = smf.quantreg("normalized ~ male", data=df).fit(q=0.5)
se = quantreg_ker_se(fit.resid, fit.model.exog, tau=0.5, hs=True)
```

`statsmodels` uses a different (narrower) bandwidth rule and underestimates the
SE by ~11% on typical payroll datasets relative to R.  This function reproduces
R's result exactly by implementing the Hall-Sheather bandwidth:

```
h = n^(-1/3) · z_{α/2}^(2/3) · [(1.5 · φ(Φ⁻¹(τ))²) / (2·Φ⁻¹(τ)² + 1)]^(1/3)
```

Set `hs=False` to use the Bofinger bandwidth instead
(`quantreg::summary.rq(se="ker", hs=FALSE)`).

### Fixes

- CHOLMOD compat: replaced deprecated `cholmod.analyze()` call with
  `cholmod.cholesky()` for `scikit-sparse` ≥ 0.5.0 (issue #8).

---

## v0.2.6 — 2026-03-27

### Satterthwaite denominator degrees of freedom

Fixed-effect t-statistics now carry Satterthwaite-approximated denominator
degrees of freedom, giving more accurate p-values for small or unbalanced
designs (particularly when the number of groups is modest):

```python
print(result.fe_df)
# Intercept         28.4
# hours_studied     41.7
# prior_gpa         38.2
```

These are also shown in `result.summary()` and match lme4's `summary.merMod`
output to within rounding.

### Fixes

- `predict()`: fixed column-alignment bug (GitHub #12) where predictions were
  incorrect when `newdata` columns arrived in a different order than the
  training frame.
- CHOLMOD: guarded against non-`Factor` return value from
  `cholmod.cholesky()` in newer `scikit-sparse` releases (GitHub #11).

---

## v0.2.5 — 2026-03-27

### ML fitting and `anova()` — likelihood-ratio tests as a first-class API

`interlace.fit()` now accepts `method="ML"` to fit by maximum likelihood
instead of REML. This is required for likelihood-ratio tests that compare
models with different fixed-effect structures.

The new `interlace.anova()` function automates the LRT, matching lme4's
`anova.merMod()` output format:

```python
import interlace

m0 = interlace.fit("score ~ 1",                    data=df, groups="school_id", method="ML")
m1 = interlace.fit("score ~ hours_studied",        data=df, groups="school_id", method="ML")
m2 = interlace.fit("score ~ hours_studied + gpa",  data=df, groups="school_id", method="ML")

print(interlace.anova(m0, m1))
#    Df   AIC   BIC  logLik  deviance  Chisq  Chi Df  Pr(>Chisq)
# 0   3  ...   ...    ...      ...     NaN     NaN        NaN
# 1   4  ...   ...    ...      ...    14.3     1.0      0.0002
```

`anova()` sorts by parameter count automatically and raises `ValueError` if
either model was fitted with REML.

### `summary()` and `VarCorr()` — lme4-style output

`result.summary()` now returns a `SummaryResult` with a `tables` property
compatible with statsmodels' `Summary.tables`, making it easy to embed in
notebooks or export to LaTeX.

`VarCorr()` returns variance-covariance components in the same format as R's
`as.data.frame(VarCorr(fit))`:

```python
from interlace import VarCorr

vc = VarCorr(result)
print(vc.as_dataframe())
#         grp           var1  var2    vcov    sdcor
# 0  school_id  (Intercept)  None   0.412    0.642
# 1   Residual         None  None   1.284    1.133
```

---

## v0.2.3 — 2026-03-27

### CHOLMOD sparse Cholesky

The REML objective now uses CHOLMOD's symbolic-then-numeric refactorisation
when `scikit-sparse` is installed.  CHOLMOD performs a single symbolic
analysis on the first iteration and reuses the sparsity pattern on every
subsequent step — substantially faster for large or deeply nested designs
where the KKT system has a stable non-zero structure.

```bash
pip install "interlace-lme[cholmod]"
```

No API change required — the fast path is activated automatically.

### `bootstrap_se` — cluster bootstrap for the median

`CrossedLMEResult` gains a `bootstrap_se()` method that computes a
cluster-bootstrap standard error for a scalar statistic of the response
(currently `"median"`):

```python
se = result.bootstrap_se(statistic="median", n_bootstrap=1000,
                          resample_level="group", seed=42)
```

`resample_level="group"` (default) resamples grouping-factor levels with
replacement and includes all observations from each sampled group, matching
R's `boot` cluster bootstrap.  `resample_level="observation"` resamples
individual rows.

### Random slopes in all diagnostics

`residuals.hlm_resid`, `leverage.leverage`, and all `influence` functions
now support models fitted with random slopes (lme4-style `(1 + x | g)`
specifications).

### `hlm_influence` performance

Design matrices `X`, `y`, and `Z` are pre-built once before the case-deletion
loop.  Per-observation formula re-parsing is eliminated, giving a meaningful
speedup on large datasets.

### Fixes

- `predict()`: fixed categorical column-ordering bug that produced wrong BLUPs
  when the newdata column order differed from the training frame.

---

## v0.2.2 — 2026-03-27

### Zero-pandas internals

Pandas has been eliminated from the diagnostics pipeline in two phases:

- **Phase 2** — `formula.py` and `sparse_z.py` now use
  [formulaic](https://matthewwardrop.github.io/formulaic/) instead of patsy,
  and [narwhals](https://narwhals-dev.github.io/narwhals/) for frame
  abstraction throughout.
- **Phase 3** — all diagnostics functions (`residuals`, `leverage`,
  `influence`, `augment`) operate on narwhals frames internally; polars
  DataFrames pass through without a pandas round-trip.

### Leverage fix for truly-crossed RE

For models with two or more grouping factors, the fixed-effects leverage
component now uses the OLS hat matrix `H_X = X(X'X)⁻¹X'` rather than the
full mixed-model hat approximation.  The previous calculation over-estimated
leverage in balanced crossed designs.

---

## v0.2.1 — 2026-03-26

### Random slopes

interlace now supports lme4-style random slopes in addition to random intercepts.

```python
# Correlated random intercept + slope
result = interlace.fit(
    "y ~ x",
    data=df,
    random=["(1 + x | group)"],
)

# Independent (uncorrelated) parameterisation
result = interlace.fit(
    "y ~ x",
    data=df,
    random=["(1 + x || group)"],
)
```

`result.random_effects["group"]` now returns a DataFrame with one column per
random effect term.  `result.varcov` exposes the full random-effect covariance
matrix.

Warm-start support: case-deletion refits initialise the optimizer from the
full-model `theta_hat`, reducing iterations for large models.

---

## v0.2.0 — 2026-03-26

### New feature: BOBYQA optimizer for R/Python diagnostic parity

interlace now ships an optional BOBYQA optimizer that significantly
improves parity between Python and R (HLMdiag) Cook's D flagging counts
in influence diagnostics.

#### Background

The residual gap documented in prior benchmarking (1.2–1.5× over-flagging
relative to R) traces to a single root cause: L-BFGS-B is gradient-based
and converges poorly near θ = 0 boundaries — the exact regime encountered
during case-deletion refits when a group shrinks to near-zero size.
R's lme4 uses **BOBYQA** (a gradient-free trust-region algorithm) by default,
which is inherently more robust at those boundaries.

#### What changed

**`interlace.fit()` — new `optimizer` parameter**

```python
interlace.fit("y ~ x", data=df, groups="firm", optimizer="bobyqa")
```

The `optimizer` keyword accepts `"lbfgsb"` (default, unchanged behaviour)
or `"bobyqa"`. BOBYQA requires the `bobyqa` optional extra (see
[Installation](installation.md)).

**Influence functions — `optimizer` parameter threaded through**

All case-deletion refit functions now accept `optimizer`:

```python
from interlace.influence import hlm_influence, cooks_distance, mdffits, tau_gap, n_influential

# Use BOBYQA for all refits — closer to R/HLMdiag flagging counts
infl = hlm_influence(model, optimizer="bobyqa")
cd   = cooks_distance(model, optimizer="bobyqa")
```

**Single-RE statsmodels routing**

When `optimizer="bobyqa"` is passed to any influence function and the
model is a statsmodels `MixedLMResults` with `_primary_group_col` set,
case-deletion refits are routed through interlace's own REML fitter
(instead of statsmodels). This ensures BOBYQA is used end-to-end rather
than only for crossed-RE models.

#### Parity improvement (benchmarks)

| Scenario | Before (L-BFGS-B) | After (BOBYQA) | R baseline |
|---|---|---|---|
| `large_scale` (single-RE, n=5000) | 259 flags | ~211 | 211 |
| `crossed_re` (3-RE, n=2000) | 327 flags | ~215 | 215 |
| `many_firms` (single-RE, n=2000) | 2000 | 2000 | 2000 ✓ |
| `many_covariates` | 93 | 93 | 92 ✓ |

The default `optimizer="lbfgsb"` is unchanged — no action required for
existing code.

---

## v0.1.1 — 2026-03-21

- Extended `Z` matrix and `Lambda_theta` to support random slopes
- Added `RandomEffectSpec` and `parse_random_effects` for lme4-style
  random parameter parsing
- Wired REML fit to use generalised `Z` and `Lambda` for random slopes

## v0.1.0 — initial release

- Profiled REML estimation for linear mixed models with crossed random
  intercepts
- Sparse `Z` matrix throughout — never materialised as dense
- Full diagnostics suite: residuals, leverage, Cook's D, MDFFITS,
  influence plots
- Compatible `CrossedLMEResult` object exposing the same attributes as
  `statsmodels.MixedLMResults`
- Validated against R's `lme4::lmer()` (fixed effects abs diff < 1e-4)
