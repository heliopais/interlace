# fit

The primary entry point for fitting linear mixed models with crossed random effects.
Accepts both random intercepts (via `groups`) and random slopes (via `random`).
Works with any narwhals-compatible DataFrame (pandas, polars, etc.).

```{eval-rst}
.. autofunction:: interlace.fit
```

## Key parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `formula` | `str` | Fixed-effects formula in Wilkinson notation (e.g. `"y ~ x + z"`) |
| `data` | `DataFrame` | Input data |
| `groups` | `str \| list[str]` | Column name(s) for crossed random intercepts (shorthand) |
| `random` | `list[str]` | lme4-style random-effect specs, e.g. `["(1 + x \| g)"]` |
| `method` | `"REML"` or `"ML"` | Estimator. Use `"ML"` for model comparison via LRT |
| `optimizer` | `"lbfgsb"` or `"bobyqa"` | Optimizer. `"bobyqa"` gives better R/lme4 parity |
| `weights` | `ndarray` | Observation-level prior weights, shape (n,) |
| `offset` | `ndarray` | Known term added to the linear predictor (not estimated) |
| `correlation` | `AR1 \| CompoundSymmetry` | Residual correlation structure for longitudinal data |
| `df_method` | `str` | `"satterthwaite"` (default) or `"kenward-roger"` for denominator DFs |
| `theta0` | `ndarray` | Initial variance parameters; defaults to ones |
| `dispformula` | `str` | Sub-model for observation-level dispersion; see Heteroscedastic LMM below |

## Examples

### Random intercepts (shorthand)

```python
import interlace

result = interlace.fit(
    "rt ~ condition",
    data=df,
    groups=["subject", "item"],  # crossed random intercepts
)

print(result.fe_params)           # fixed-effect coefficients
print(result.variance_components) # σ² per grouping factor
print(result.aic, result.bic)
```

### Random slopes

```python
result = interlace.fit(
    "rt ~ condition",
    data=df,
    random=[
        "(1 + condition | subject)",  # correlated intercept + slope
        "(1 | item)",                 # intercept only
    ],
)

# random_effects["subject"] is a DataFrame: one column per term
print(result.random_effects["subject"])
print(result.varcov)  # full random-effect covariance matrix
```

### Model comparison with ML

```python
# Fit with ML for likelihood ratio test
m1 = interlace.fit("y ~ x",      data=df, groups="g", method="ML")
m2 = interlace.fit("y ~ x + z",  data=df, groups="g", method="ML")

import scipy.stats
lrt_stat = 2 * (m2.llf - m1.llf)
p_value  = scipy.stats.chi2.sf(lrt_stat, df=1)
```

### Heteroscedastic LMM (`dispformula`)

Pass `dispformula=` to fit a Gaussian LMM whose residual SD varies across
observations. Mirrors R `glmmTMB(..., dispformula = ...)`:

```python
# σ_i = exp(δ_0 + δ_1 · z_i)  (heteroscedastic by covariate)
res = interlace.fit("y ~ x", data=df, groups="g", dispformula="~ z")
print(res.disp_params)         # log-σ coefficients (Intercept, z)
print(res.dispersion[:5])      # per-obs σ²

# Random intercepts on log σ (salary-models style):
res = interlace.fit(
    "y ~ x", data=df, groups="g_mean",
    dispformula="~ (1|g_disp)",
)
print(res.disp_variance_components)   # τ²_g_disp
print(res.disp_random_effects)        # BLUPs on log-σ scale
```

`fit` dispatches automatically:

* **FE-only dispformula** (`~1`, `~z`) → joint Laplace via the GLMM Laplace
  machinery. Parity with `glmmTMB` is ~1e-6 on FE and disp coefficients.
* **Random effects on dispformula side** (`~ (1|g)`, `~ (1|g/h)`) → joint
  Laplace over both random-effect blocks (β, b, u_d). For nested designs
  (the salary-models target), disp variance components match `glmmTMB`
  within ~1%; mean-side FE within 1e-3.

The reported `disp_method` attribute on the result distinguishes the
paths: `"joint_laplace"` (default) or `"bca"`. Pass
`dispformula_method="bca"` to opt in to the legacy Block-Coordinate Ascent
path (faster but biased disp varcomps by 15–80%; kept available for
cross-comparison).

## See also

- {doc}`result` — attributes on the returned `CrossedLMEResult`
- {doc}`predict` — generating predictions from a fitted model
- {doc}`correlation` — AR(1) and compound symmetry for longitudinal data
- {doc}`kenward_roger` — Kenward-Roger denominator degrees of freedom
- {doc}`glmer` — generalised linear mixed models (non-normal outcomes)
- {doc}`clmm` — ordinal regression with random effects
- {doc}`coxme` — Cox frailty models
- [Random Slopes Guide](../random-slopes.md) — when and how to use `random=`
- [Model Comparison Guide](../model-comparison.md) — LRT workflow with `method="ML"`
