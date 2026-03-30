# Variance Inference Guide

After fitting a mixed model the natural questions are: *are my variance
components reliably estimated?* and *how uncertain are they?* This page
covers three tools for answering those questions:

- `is_singular` / `isSingular()` — detect boundary fits before reporting
- `result.confint()` — profile likelihood confidence intervals for variance parameters
- `VarCorr()` — R-style variance-covariance summary

---

## Boundary and singular fits

A **singular fit** occurs when one or more variance components are estimated
at exactly zero — the boundary of the parameter space. This is not a crash or
an error; it means the data contain little or no evidence for that grouping
factor's contribution. But boundary fits have practical consequences:

- Standard errors for fixed effects may be inflated or poorly calibrated
- Profile CIs for the collapsed component will hit zero on the lower bound
- The model may be over-parameterised for the available data

### Detecting a singular fit

```python
import interlace
from interlace import isSingular

result = interlace.fit("score ~ hours_studied", data=df,
                       groups=["student_id", "school_id"])

# Function form (mirrors lme4)
if isSingular(result):
    print("Singular fit — check variance_components")

# Property shorthand
print(result.is_singular)          # True or False

# Per-factor flags
print(result.boundary_flags)
# {'student_id': False, 'school_id': True}
```

`fit()` issues a `ConvergenceWarning` automatically when a boundary fit is
detected, so you will usually see the warning before you check manually.

### What to do

| Situation | Action |
|-----------|--------|
| Near-zero variance for a grouping factor | Consider dropping it; compare AIC with `anova()` |
| Pre-registered random effect | Report the boundary result and flag the limitation |
| Convergence warning but non-zero variance | Try BOBYQA: `optimizer="bobyqa"` |
| Random slope variance collapsed | Try the independent (`||`) parameterisation |

The tolerance for declaring a component "effectively zero" is `tol=1e-4`
(the lme4 default), matching the diagonal of the relative covariance factor
Lambda_theta. Pass a custom tolerance to `isSingular()` if needed:

```python
isSingular(result, tol=1e-5)   # stricter
```

---

## Profile likelihood CIs for variance parameters

Wald confidence intervals (estimate ± 1.96 SE) assume asymptotic normality,
which is a poor approximation for variance components — especially near the
boundary or with few groups. Profile likelihood CIs are more accurate because
they follow the actual curvature of the log-likelihood surface.

`result.confint()` computes these by fixing each variance parameter in turn,
profiling out the others, and finding the points where
`2 × (L_max − L(θ)) = χ²(level, df=1)`.

### Basic usage

```python
ci = result.confint()           # 95 % by default
print(ci)
#                 estimate   2.5 %  97.5 %
# school_id          0.412   0.201   0.731
# student_id         0.638   0.389   1.042
# residual           1.000   1.000   1.000
```

```python
ci_90 = result.confint(level=0.90)
```

### Interpreting the theta scale

CIs are reported on the **theta** scale — the diagonal of the relative
Cholesky factor Lambda_theta. For an intercept-only random effect:

```
sigma_b ≈ theta × sqrt(sigma2_hat)
```

where `sigma2_hat` is `result.scale`. To convert:

```python
sigma2 = result.scale
ci_sd = ci[["2.5 %", "97.5 %"]] * sigma2 ** 0.5
print(ci_sd)   # CIs on the standard-deviation scale
```

### Boundary lower bounds

If the profile drops to its target before theta reaches zero, the lower bound
is reported as `0.0`. This is expected for singular or near-singular fits and
means the data are compatible with the variance component being zero.

### When to use profile vs Wald CIs

| | Profile | Wald |
|--|---------|------|
| Accuracy for variance params | Good | Poor near boundary |
| Speed | Slower (1D optimisation per param) | Instant |
| Appropriate for | Final reporting, small n_groups | Quick checks, large samples |

---

## `VarCorr()` — R-style variance-covariance summary

`VarCorr()` returns the variance components in the same format as R's
`as.data.frame(VarCorr(fit))` — useful for direct comparison with lme4 output
or for embedding in a tidy report.

```python
from interlace import VarCorr

vc = VarCorr(result)
df_vc = vc.as_dataframe()
print(df_vc)
#         grp           var1   var2    vcov    sdcor
# 0  school_id  (Intercept)   None   0.412    0.642
# 1   Residual          None  None   1.284    1.133
```

Columns:
- `grp` — grouping factor name (or `"Residual"`)
- `var1`, `var2` — term names; `var2` is non-null only for covariance entries in random-slope models
- `vcov` — variance (diagonal) or covariance (off-diagonal)
- `sdcor` — standard deviation (diagonal) or correlation (off-diagonal)

### Random slopes

For a model with correlated random slopes, `as_dataframe()` includes both the
variance and covariance rows:

```python
result_slopes = interlace.fit(
    "rt ~ condition",
    data=df,
    random=["(1 + condition | subject)"],
)
print(VarCorr(result_slopes).as_dataframe())
#        grp           var1       var2    vcov   sdcor
# 0  subject  (Intercept)       None  45.20   6.723
# 1  subject  (Intercept)  condition  -8.40  -0.307
# 2  subject    condition       None   3.10   1.761
# 3  Residual        None       None  12.80   3.578
```

### `VarCorr` vs `variance_components`

| | `VarCorr().as_dataframe()` | `result.variance_components` |
|--|---------------------------|------------------------------|
| Format | Long-form DataFrame (R-compatible) | Dict keyed by group name |
| Covariances | Included | Not directly |
| SDs / correlations | Included as `sdcor` | Not directly |
| Best for | Reporting, R parity | Programmatic access |

---

## Putting it together

A complete post-fit variance diagnostic workflow:

```python
import interlace
from interlace import isSingular, VarCorr

result = interlace.fit(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups=["student_id", "school_id"],
)

# 1. Check for boundary
if isSingular(result):
    print("Singular:", result.boundary_flags)

# 2. Inspect variance components in R-compatible format
print(VarCorr(result).as_dataframe())

# 3. Profile CIs (skip if singular — lower bound will be 0)
if not result.is_singular:
    ci = result.confint()
    print(ci)
```

---

## See also

- [Random Slopes Guide](random-slopes.md) — `random_effects_se` and BLUP uncertainty
- [Model Comparison Guide](model-comparison.md) — `anova()` and LRT for model selection
- [FAQ](faq.md#how-do-i-know-if-my-model-has-a-boundarysingular-fit) — quick answers on boundary fits and variance CIs
- {doc}`api/convergence` — `isSingular` API reference
- {doc}`api/profile_ci` — `profile_confint` API reference
