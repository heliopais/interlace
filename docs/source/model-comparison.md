# Model Comparison

Comparing mixed models lets you test whether adding predictors or expanding the
random-effect structure improves fit beyond what you'd expect by chance. This page
covers the likelihood ratio test (LRT) workflow and when to use ML vs REML.

---

## REML vs ML: which to use

| Goal | Estimator | Why |
|------|-----------|-----|
| **Final parameter estimates** (fixed effects, variance components) | `method="REML"` *(default)* | Unbiased variance estimates |
| **Comparing models with different fixed effects** | `method="ML"` | REML likelihoods are not comparable when the fixed-effect design differs |
| **Comparing models with the same fixed effects, different random structure** | Either (REML preferred) | REML likelihoods *are* comparable here; use REML for unbiased variance estimates |

**Practical workflow:**

1. Use `method="ML"` to select fixed effects (test whether adding a predictor helps)
2. Refit the winning model with `method="REML"` (default) to get the final estimates

---

## Likelihood ratio test (LRT)

The LRT compares two nested models. The test statistic is:

```
χ² = 2 × (log-likelihood_full − log-likelihood_reduced)
```

Under the null hypothesis (reduced model is adequate), this follows a χ² distribution
with degrees of freedom equal to the number of added parameters.

### Comparing fixed-effect structures

```python
import interlace
import scipy.stats

# Both models must use method="ML"
m_reduced = interlace.fit(
    "rt ~ 1",           # intercept only
    data=df,
    groups=["subject", "item"],
    method="ML",
)

m_full = interlace.fit(
    "rt ~ condition",   # + condition effect
    data=df,
    groups=["subject", "item"],
    method="ML",
)

lrt_stat = 2 * (m_full.llf - m_reduced.llf)
df_diff  = 1  # one added parameter (condition coefficient)
p_value  = scipy.stats.chi2.sf(lrt_stat, df=df_diff)

print(f"LRT χ²({df_diff}) = {lrt_stat:.3f}, p = {p_value:.4f}")
# LRT χ²(1) = 18.42, p = 0.0000
```

### Comparing random-effect structures

When adding a random slope, the number of added parameters depends on the
parameterisation:

| Change | Added parameters |
|--------|-----------------|
| Add intercept-only term | 1 (variance) |
| Add correlated slope `(1 + x \| g)` | 2 (slope variance + intercept-slope covariance) |
| Add independent slope `(1 + x \|\| g)` | 1 (slope variance only, covariance = 0) |

```python
m_intercept = interlace.fit(
    "rt ~ condition",
    data=df,
    groups=["subject", "item"],
    method="ML",
)

m_slopes = interlace.fit(
    "rt ~ condition",
    data=df,
    random=["(1 + condition | subject)", "(1 | item)"],
    method="ML",
)

lrt_stat = 2 * (m_slopes.llf - m_intercept.llf)
p_value  = scipy.stats.chi2.sf(lrt_stat, df=2)  # 2 extra params
print(f"LRT χ²(2) = {lrt_stat:.3f}, p = {p_value:.4f}")
```

---

## AIC and BIC

For non-nested comparisons (e.g. two different fixed-effect structures where neither
is a special case of the other), use information criteria:

```python
print(f"AIC: {m_reduced.aic:.1f} vs {m_full.aic:.1f}")
print(f"BIC: {m_reduced.bic:.1f} vs {m_full.bic:.1f}")
```

Lower AIC/BIC is better. BIC penalises complexity more heavily and tends to favour
simpler models. Differences < 2 are not meaningful; differences > 10 are strong.

---

## Step-by-step workflow

A practical model-building workflow for a typical analysis:

```python
import interlace
import scipy.stats

# Step 1: Fit baseline with ML
m0 = interlace.fit("rt ~ 1", data=df, groups=["subject", "item"], method="ML")

# Step 2: Add predictors one at a time, test with LRT
m1 = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"], method="ML")
lrt = 2 * (m1.llf - m0.llf)
print(f"Adding condition: χ²(1) = {lrt:.2f}, p = {scipy.stats.chi2.sf(lrt, 1):.4f}")

m2 = interlace.fit("rt ~ condition + frequency", data=df, groups=["subject", "item"], method="ML")
lrt = 2 * (m2.llf - m1.llf)
print(f"Adding frequency: χ²(1) = {lrt:.2f}, p = {scipy.stats.chi2.sf(lrt, 1):.4f}")

# Step 3: Refit winning model with REML for final estimates
m_final = interlace.fit("rt ~ condition + frequency", data=df, groups=["subject", "item"])
# method="REML" is the default

print(m_final.fe_params)
print(m_final.variance_components)
```

---

## Iterative refinement with `update()`

Repeatedly calling `interlace.fit()` with slightly different formulas is
verbose. The `update()` method reruns the fit with only the parts you want to
change — formula, data, or any keyword argument — while inheriting the rest
from the original call.

### Dot notation

A `.` in the new formula expands to the corresponding part of the original:

```python
# Original model
m0 = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"], method="ML")

# Add a predictor: . ~ . + frequency
m1 = m0.update(". ~ . + frequency")

# Remove a predictor: . ~ . - condition
m_reduced = m0.update(". ~ . - condition")

# Replace the response (LHS)
m_alt = m0.update("log_rt ~ .")
```

The original `m0` is unchanged; `update()` always returns a new
`CrossedLMEResult`.

### Changing the dataset

Pass `data=` to refit on a different frame — useful for sensitivity analyses or
rolling-window designs:

```python
# Refit on a filtered subset
m_large = m0.update(data=df[df["school_size"] > 200])

# Change data and formula together
m_sens = m0.update(". ~ . + frequency", data=df_filtered)
```

### Overriding fit arguments

Any keyword accepted by `interlace.fit()` can be overridden:

```python
# Switch from ML to REML for final estimates
m_reml = m1.update(method="REML")

# Switch optimizer
m_bobyqa = m0.update(optimizer="bobyqa")
```

### Typical workflow

```python
import interlace, scipy.stats

# 1. Build models incrementally with ML
m0 = interlace.fit("rt ~ 1",         data=df, groups=["subject", "item"], method="ML")
m1 = m0.update(". ~ . + condition")
m2 = m1.update(". ~ . + frequency")

# 2. Test each step with LRT
for reduced, full, name in [(m0, m1, "condition"), (m1, m2, "frequency")]:
    stat = 2 * (full.llf - reduced.llf)
    p    = scipy.stats.chi2.sf(stat, df=1)
    print(f"{name}: χ²(1) = {stat:.2f}, p = {p:.4f}")

# 3. Refit winner with REML for final estimates
m_final = m2.update(method="REML")
print(m_final.summary())
```

---

## Notes and caveats

- **LRT p-values for variance components are conservative.** The null hypothesis puts
  the parameter on the boundary of the parameter space (variance ≥ 0), so the
  chi-squared approximation is anti-conservative. A rule of thumb: halve the p-value,
  or use a mixture distribution. For fixed effects, the approximation is accurate.

- **Do not compare REML likelihoods across different fixed-effect structures.** REML
  integrates out the fixed effects, so the likelihood depends on the fixed-effect
  design matrix — two models with different fixed effects have incomparable REML
  likelihoods.

- **AIC/BIC with REML**: `result.aic` and `result.bic` are computed from the REML
  log-likelihood when `method="REML"`. Use them only for models with identical fixed
  effects.

---

## See also

- [Concepts](concepts.md) — REML vs ML background
- [Random Slopes Guide](random-slopes.md) — testing slope variance with LRT
- {doc}`api/fit` — `method` and `optimizer` parameters
