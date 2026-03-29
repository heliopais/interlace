# Random Slopes Guide

Random slopes let each group have its own relationship between a predictor and the
outcome — rather than a single population slope shared across all groups. This page
explains when to use them, how to specify them, and how to interpret the results.

---

## When to add a random slope

Start with random intercepts only (`groups=`). Consider adding a random slope when:

- **Theory suggests it**: subjects or groups plausibly differ in *how much* a
  predictor affects them (not just their baseline)
- **Model comparison favours it**: a likelihood ratio test or AIC comparison of
  intercept-only vs slope model is significant
- **Residual plots show group-by-predictor interaction**: conditional residuals
  fan out systematically with the predictor value

**Rule of thumb**: if you have fewer than ~10 observations per group for the
predictor of interest, a random slope may be poorly identified. Check that the
model converges and that the slope variance is not estimated near zero.

---

## Syntax

### Correlated intercept + slope

The default: the intercept and slope for each group are modelled as jointly Normal,
allowing them to correlate (e.g. groups with a high baseline also tend to have a
steeper slope).

```python
import interlace

result = interlace.fit(
    "rt ~ condition",
    data=df,
    random=["(1 + condition | subject)"],
)
```

Equivalent lme4 R syntax: `rt ~ condition + (1 + condition | subject)`

### Independent (uncorrelated) parameterisation

Use `||` to force the intercept and slope variance to be estimated independently,
with covariance fixed to zero. This is more parsimonious and useful when correlation
cannot be estimated reliably (few groups, sparse data per group).

```python
result = interlace.fit(
    "rt ~ condition",
    data=df,
    random=["(1 + condition || subject)"],
)
```

Equivalent lme4 R syntax: `rt ~ condition + (1 + condition || subject)`

### Mixing slope and intercept-only terms

Use `random=` for all terms when combining random slopes for one factor with
random intercepts for another:

```python
result = interlace.fit(
    "rt ~ condition",
    data=df,
    random=[
        "(1 + condition | subject)",  # by-subject slope
        "(1 | item)",                 # item intercept only
    ],
)
```

---

## Interpreting results

### BLUPs (random_effects)

When a model includes random slopes, `result.random_effects[group]` returns a
**DataFrame** rather than a Series — one column per random effect term:

```python
print(result.random_effects["subject"])
#             (Intercept)  condition
# subject_01       -12.3       0.42
# subject_02         8.7      -0.31
# subject_03         2.1       0.05
# ...
```

The `(Intercept)` column is the by-subject deviation from the grand mean. The
`condition` column is the by-subject deviation from the population slope for
`condition`. A subject with `condition = 0.42` responds *more strongly* to condition
than average.

### Variance components and covariance matrix

```python
# Variance of random intercepts and slopes
print(result.variance_components)
# {'subject': {'(Intercept)': 45.2, 'condition': 3.1}, 'residual': 12.8}

# Full covariance matrix (intercept-slope correlation included)
print(result.varcov)
#                 (Intercept)  condition
# (Intercept)         45.2       -8.4
# condition            -8.4        3.1
```

The off-diagonal of `varcov` gives the **intercept-slope covariance**. A negative
value means groups with a higher baseline tend to show a smaller (or negative) slope.

### Comparing to the intercept-only model

Fit both models with `method="ML"` and compare:

```python
m_intercept = interlace.fit("rt ~ condition", data=df,
                             groups=["subject", "item"], method="ML")

m_slopes = interlace.fit("rt ~ condition", data=df,
                          random=["(1 + condition | subject)", "(1 | item)"],
                          method="ML")

import scipy.stats
# Correlated slope adds 2 parameters (slope variance + covariance)
lrt = 2 * (m_slopes.llf - m_intercept.llf)
p   = scipy.stats.chi2.sf(lrt, df=2)
print(f"LRT χ²(2) = {lrt:.2f}, p = {p:.4f}")
```

If p < 0.05, the slope variance is statistically meaningful. See the
[Model Comparison Guide](model-comparison.md) for the full LRT workflow.

---

## Common issues

### Singular fit / convergence warning

If the optimizer warns about a singular fit or near-zero slope variance, the random
slope may be over-parameterised for your data:

- Try the independent (`||`) parameterisation first
- Check that you have enough observations per group (≥ 5–10 per group for the
  predictor)
- Compare AIC with the intercept-only model; if they are similar, prefer the simpler
  model

### Switching from `groups=` to `random=`

```python
# Before
result = interlace.fit("y ~ x", data=df, groups=["g1", "g2"])

# After — equivalent intercept-only model using random=
result = interlace.fit(
    "y ~ x",
    data=df,
    random=["(1 | g1)", "(1 | g2)"],
)
```

Both produce identical results. Use whichever is clearer for your use case.

---

## Uncertainty in BLUPs

BLUPs are point estimates — each group's deviation from the population mean. The
`random_effects_se` property and `random_effects_ci()` method expose the posterior
standard errors and normal-approximation confidence intervals for those estimates.

### Standard errors

```python
se = result.random_effects_se
# Intercept-only model → pd.Series indexed by group
print(se["subject"])
# subject_01    4.32
# subject_02    3.87
# subject_03    5.01
# ...

# Random-slope model → pd.DataFrame, one column per term
print(se["subject"])
#             (Intercept)  condition
# subject_01         4.32       0.61
# subject_02         3.87       0.54
```

### Confidence intervals

```python
# 95 % CIs (default)
ci = result.random_effects_ci()
print(ci["subject"])
#             lower    upper
# subject_01  -20.8     -3.8
# subject_02   -2.1     17.3
# ...

# 90 % CIs
ci_90 = result.random_effects_ci(level=0.90)
```

For random-slope models `random_effects_ci()` returns a DataFrame with a
MultiIndex column: `(term, "lower")` and `(term, "upper")` pairs.

### Caterpillar plot

A caterpillar plot orders groups by their BLUP and overlays the CI — a quick
visual check for which groups stand out from the population mean:

```python
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_errorbar, geom_hline, coord_flip, theme_bw

blups = result.random_effects["subject"]
ci    = result.random_effects_ci()["subject"]

caterpillar = pd.DataFrame({
    "group":  blups.index,
    "blup":   blups.values,
    "lower":  ci["lower"].values,
    "upper":  ci["upper"].values,
}).sort_values("blup").assign(rank=lambda d: range(len(d)))

(
    ggplot(caterpillar, aes(x="rank", y="blup"))
    + geom_errorbar(aes(ymin="lower", ymax="upper"), width=0.3, alpha=0.5)
    + geom_point(size=2)
    + geom_hline(yintercept=0, linetype="dashed", color="grey")
    + coord_flip()
    + theme_bw()
)
```

Groups whose CI excludes zero differ reliably from the population mean. Note that
these are normal-approximation CIs — treat them as indicative rather than exact
for small group counts or boundary variance estimates.

---

## See also

- [Quickstart](quickstart.md) — `groups` vs `random` parameter overview
- [Model Comparison Guide](model-comparison.md) — LRT for testing slope variance
- [Concepts](concepts.md) — statistical background on random effects and variance components
- {doc}`api/fit` — full `fit()` parameter reference
