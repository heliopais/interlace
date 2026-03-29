# fit

The primary entry point for fitting linear mixed models with crossed random effects.
Accepts both random intercepts (via `groups`) and random slopes (via `random`).
Works with any pandas DataFrame.

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

## See also

- {doc}`result` — attributes on the returned `CrossedLMEResult`
- {doc}`predict` — generating predictions from a fitted model
- [Random Slopes Guide](../random-slopes.md) — when and how to use `random=`
- [Model Comparison Guide](../model-comparison.md) — LRT workflow with `method="ML"`
