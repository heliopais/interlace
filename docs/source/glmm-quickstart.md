# GLMM quickstart

This page walks through fitting generalised linear mixed models (GLMMs) with
`interlace.glmer()`. For linear mixed models (continuous outcomes), see the
[LMM quickstart](quickstart.md).

---

## When to use a GLMM

Use `glmer()` instead of `fit()` when your outcome is:

- **Binary** (0/1, success/failure) or a **proportion** (successes / trials) — use `family="binomial"`
- **Count data** (events, errors, frequencies) — use `family="poisson"`
- **Overdispersed counts** (variance > mean) — use `family=NegativeBinomial2Family(theta=...)`

For continuous, approximately normal outcomes, use `interlace.fit()`.

---

## Binomial GLMM: disease incidence

A classic example: modelling disease incidence across cattle herds over
multiple periods. The response is a proportion (number infected / herd size),
so we use a binomial family with trial-count weights.

```python
import interlace
import pandas as pd

# Each row: one herd in one period
df = pd.DataFrame({
    "incidence": [2, 3, 4, 0, 3, 1, ...],  # number infected
    "size":      [14, 12, 9, 5, 22, 18, ...],  # herd size (trials)
    "period":    [1, 2, 3, 4, 1, 2, ...],
    "herd":      [1, 1, 1, 1, 2, 2, ...],
})

result = interlace.glmer(
    formula="incidence / size ~ period",
    data=df,
    family="binomial",
    groups="herd",
    weights=df["size"].values,  # trial counts
)
```

### Inspect results

```python
# Fixed effects (log-odds scale)
print(result.fe_params)

# Between-herd variance
print(result.variance_components)

# Model fit
print(f"AIC: {result.aic:.1f}  BIC: {result.bic:.1f}")

# Did it converge?
print(result.converged)
```

### Predict

```python
# Predicted probabilities (response scale)
probs = result.predict(newdata=df_new)

# Log-odds (link scale)
logits = result.predict(newdata=df_new, type="link")

# Population-level only (ignore herd-specific effects)
probs_marginal = result.predict(newdata=df_new, include_re=False)
```

---

## Poisson GLMM: count data

For count outcomes, use `family="poisson"`. No weights are needed.

```python
result = interlace.glmer(
    formula="count ~ x1 + x2",
    data=df,
    family="poisson",
    groups=["site", "year"],  # crossed random intercepts
)

# Fixed effects (log-rate scale)
print(result.fe_params)

# BLUPs
print(result.random_effects["site"])
print(result.random_effects["year"])
```

---

## Negative Binomial GLMM: overdispersed counts

When count data has more variance than a Poisson model predicts
(overdispersion), use the negative binomial family. The shape parameter `theta`
controls overdispersion — smaller values mean more overdispersion.

```python
from interlace import NegativeBinomial2Family

result = interlace.glmer(
    formula="count ~ x1 + x2",
    data=df,
    family=NegativeBinomial2Family(theta=2.0),
    groups="site",
)

# Fixed effects (log-rate scale)
print(result.fe_params)

# Compare with Poisson to assess overdispersion
result_pois = interlace.glmer(
    formula="count ~ x1 + x2",
    data=df,
    family="poisson",
    groups="site",
)
print(f"NB AIC: {result.aic:.1f}  Poisson AIC: {result_pois.aic:.1f}")
```

---

## Using `random=` for lme4-style syntax

Like `fit()`, `glmer()` supports both the `groups` shorthand and the full
lme4-style `random` parameter:

```python
# These are equivalent:
result = interlace.glmer(..., groups="herd")
result = interlace.glmer(..., random=["(1 | herd)"])
```

---

## Choosing an optimizer

The default optimizer is L-BFGS-B. For models that struggle to converge,
try BOBYQA (gradient-free):

```python
result = interlace.glmer(
    formula="incidence / size ~ period",
    data=df,
    family="binomial",
    groups="herd",
    weights=df["size"].values,
    optimizer="bobyqa",
)
```

---

## Adaptive Gauss-Hermite quadrature (AGQ)

The default Laplace approximation (`nAGQ=1`) is fast but can be inaccurate for
models with few observations per group (e.g. sparse binary data). Setting
`nAGQ > 1` uses adaptive Gauss-Hermite quadrature for a more accurate
marginal-likelihood integral, matching `lme4::glmer(nAGQ = ...)`.

```python
result = interlace.glmer(
    formula="incidence / size ~ period",
    data=df,
    family="binomial",
    groups="herd",
    weights=df["size"].values,
    nAGQ=25,
)
```

**Constraints:** `nAGQ > 1` requires a single grouping factor with a random
intercept only (no random slopes or crossed random effects). This matches
lme4's restriction.

---

## Comparison with R

| interlace | R (lme4) |
|-----------|----------|
| `interlace.glmer(formula, data, family="binomial", groups="herd", weights=w)` | `glmer(cbind(incidence, size - incidence) ~ period + (1\|herd), data, family=binomial)` |
| `interlace.glmer(formula, data, family="poisson", groups="site")` | `glmer(count ~ x + (1\|site), data, family=poisson)` |
| `interlace.glmer(..., family=NegativeBinomial2Family(theta=2.0), ...)` | `glmer.nb(count ~ x + (1\|site), data)` |
| `interlace.glmer(..., nAGQ=25)` | `glmer(..., nAGQ=25)` |
| `result.fe_params` | `fixef(fit)` |
| `result.variance_components` | `as.data.frame(VarCorr(fit))` |
| `result.predict(newdata)` | `predict(fit, newdata, type="response")` |

---

## Next steps

- See {doc}`api/glmer` for the full API reference and all parameters
- See {doc}`quickstart` for linear mixed models
- See {doc}`api/fit` for the `fit()` API reference
