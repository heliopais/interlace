# CLMM quickstart

This page walks through fitting cumulative link mixed models (CLMMs) with
`interlace.clmm()`. CLMMs are for **ordinal outcomes** — ratings, Likert
scales, severity grades — where the response has a natural order but the
distances between levels are unknown.

For continuous outcomes, see the [LMM quickstart](quickstart.md).
For binary or count outcomes, see the [GLMM quickstart](glmm-quickstart.md).

---

## When to use a CLMM

Use `clmm()` when your outcome is:

- **Ordinal** — categories with a meaningful order but no meaningful spacing
  (e.g. 1-star to 5-star ratings, "strongly disagree" to "strongly agree",
  pain severity "none / mild / moderate / severe")
- **Grouped** — observations are clustered within subjects, judges, raters,
  or other grouping factors

If you treat ordinal responses as continuous and fit a standard LMM, you
assume equal spacing between levels and normality of residuals — both wrong
for Likert-type data. If you collapse to binary (e.g. "high vs low"), you
throw away information. A CLMM respects the ordinal nature directly.

---

## The model

A CLMM models cumulative probabilities via a link function:

```
link(P(Y <= k | b)) = alpha_k - x'beta - z'b,   k = 1,...,K-1
```

- **alpha_1 < alpha_2 < ... < alpha_{K-1}** are threshold (cut-point) parameters
  separating the K ordinal categories
- **beta** are fixed-effect coefficients (the formula intercept is absorbed
  into the thresholds)
- **b ~ N(0, sigma^2)** are random effects (per-group deviations)

The **proportional odds assumption** (with logit link) means that each
predictor shifts the entire cumulative probability curve by the same amount,
regardless of which threshold you look at.

---

## Basic example: wine tasting

The classic `wine` dataset from R's `ordinal` package: 9 judges rate wines
on a 1-5 bitterness scale. The predictors are serving temperature (`temp`:
cold/warm) and whether the judge had contact with the wine before tasting
(`contact`: yes/no).

```python
import interlace
import pandas as pd

# Load data — 72 observations, 5 ordinal levels, 9 judges
df = pd.read_csv("wine.csv")

result = interlace.clmm(
    formula="rating ~ temp + contact",
    data=df,
    groups="judge",
    link="logit",       # default — proportional odds model
)
```

### Inspect results

```python
# Threshold estimates (cut points between ordinal levels)
print(result.thresholds)
# {'1|2': -1.34, '2|3': 1.25, '3|4': 3.47, '4|5': 5.01}

# Fixed-effect coefficients
print(result.fe_params)
# tempwarm     2.32
# contactyes   1.53

# Standard errors
print(result.fe_bse)

# Between-judge variance
print(result.variance_components)
# {'judge': 1.13}

# Model fit
print(f"AIC: {result.aic:.1f}  BIC: {result.bic:.1f}")
print(f"Converged: {result.converged}")
```

### Interpreting thresholds

Thresholds are the boundaries between adjacent ordinal categories on the
latent scale. With a logit link:

- `alpha['1|2'] = -1.34` means that at the population mean (all predictors
  at reference level, no random effect), the log-odds of rating <= 1 vs > 1
  is -1.34
- The gap between thresholds reflects how "spread out" the categories are on
  the latent scale

You rarely interpret thresholds directly — the fixed effects and predictions
are more useful. But threshold ordering (`alpha_1 < alpha_2 < ...`) is
verified automatically; a violation would indicate a model specification
problem.

### Interpreting fixed effects

With the logit link, fixed-effect coefficients are **log cumulative odds
ratios** — the same interpretation as in standard proportional-odds logistic
regression:

- `tempwarm = 2.32`: warm temperature increases the log-odds of being in a
  higher rating category by 2.32 (odds ratio = exp(2.32) = 10.2)
- `contactyes = 1.53`: prior contact increases the log-odds of a higher
  rating by 1.53 (odds ratio = exp(1.53) = 4.6)

```python
import numpy as np

# Odds ratios with 95% CI
ci = result.confint()
print("Odds ratios:")
for name in result.fe_params.index:
    est = result.fe_params[name]
    row = ci.loc[name]
    print(f"  {name}: OR = {np.exp(est):.2f} "
          f"[{np.exp(row.iloc[0]):.2f}, {np.exp(row.iloc[1]):.2f}]")
```

---

## Prediction

`predict()` supports three output types:

### Category probabilities

```python
# P(Y=k) for each observation, shape (n, K)
probs = result.predict(newdata=df, type="prob")
print(probs[:3])
# [[0.12, 0.34, 0.28, 0.18, 0.08],
#  [0.05, 0.18, 0.32, 0.29, 0.16],
#  ...]
```

Rows sum to 1. Each column is the probability of the corresponding ordinal
level.

### Cumulative probabilities

```python
# P(Y<=k) for k=1,...,K-1, shape (n, K-1)
cum = result.predict(newdata=df, type="cum.prob")
```

These are monotonically non-decreasing across columns (by construction).

### Linear predictor

```python
# x'beta + z'b (without thresholds), shape (n,)
eta = result.predict(newdata=df, type="linear.predictor")
```

Useful for computing your own probability transformations or for
model-checking.

### Unseen groups

Unseen group levels at prediction time automatically receive a BLUP of zero
(population-level prediction only):

```python
new_data = pd.DataFrame({
    "temp": ["cold", "warm"],
    "contact": ["no", "yes"],
    "judge": ["new_judge_A", "new_judge_B"],
})
probs = result.predict(new_data, type="prob")
# Uses fixed effects only — no judge-specific adjustment
```

---

## Link functions

The `link` parameter controls the shape of the cumulative probability curve:

| Link | When to use |
|------|-------------|
| `"logit"` *(default)* | Most common. Proportional odds model. Coefficients are log cumulative odds ratios. |
| `"probit"` | When you assume the latent variable is normally distributed. Coefficients don't have a simple odds-ratio interpretation. |
| `"cloglog"` | Complementary log-log. For asymmetric data or grouped survival/discrete-time hazard models. |

```python
# Probit link
result_probit = interlace.clmm(
    "rating ~ temp + contact",
    data=df,
    groups="judge",
    link="probit",
)
```

In practice, logit and probit give very similar results. Use logit unless you
have a specific reason for probit (e.g. a normality assumption on the latent
variable) or cloglog (e.g. discrete-time survival).

---

## Crossed random effects

Like `fit()` and `glmer()`, `clmm()` supports crossed random effects. If
wines are rated by judges *and* each wine bottle is a separate source of
variability:

```python
result = interlace.clmm(
    "rating ~ temp + contact",
    data=df,
    random=["(1|judge)", "(1|bottle)"],
    link="logit",
)

print(result.variance_components)
# {'judge': 1.13, 'bottle': 0.31}

print(result.random_effects["judge"])
print(result.random_effects["bottle"])
```

---

## Confidence intervals

`confint()` returns Wald confidence intervals for both thresholds and fixed
effects:

```python
ci = result.confint(level=0.95)
print(ci)
#              2.5 %   97.5 %
# 1|2         -2.58   -0.10
# 2|3          0.00    2.50
# 3|4          2.11    4.83
# 4|5          3.54    6.48
# tempwarm     0.90    3.74
# contactyes   0.17    2.89
```

---

## Summary

`summary()` prints a human-readable overview matching the style of R's
`ordinal::clmm()` output:

```python
print(result.summary())
```

---

## Comparison with R

| interlace | R (ordinal) |
|-----------|-------------|
| `interlace.clmm(formula, data, groups="judge")` | `clmm(rating ~ temp + contact + (1\|judge), data)` |
| `result.thresholds` | `result$alpha` |
| `result.fe_params` | `result$beta` |
| `result.variance_components` | `VarCorr(result)` |
| `result.predict(newdata, type="prob")` | `predict(result, newdata, type="prob")` |
| `result.predict(newdata, type="cum.prob")` | `predict(result, newdata, type="cum.prob")` |
| `result.confint()` | `confint(result)` |

**Parity:** thresholds and fixed effects match R's `ordinal::clmm()` within
1e-2 (absolute), variance components within 5% (relative), and BLUPs
correlate > 0.99 on the wine benchmark dataset.

---

## CLMM vs GLMM: which to use

| Your outcome | Use |
|---|---|
| Ordered categories (ratings, Likert scales, severity) | `clmm()` |
| Binary (0/1, success/failure) | `glmer(family="binomial")` |
| Counts (events, frequencies) | `glmer(family="poisson")` |
| Continuous proportions in (0,1) | `glmer(family=BetaFamily(...))` |
| Continuous, approximately normal | `fit()` |

A common mistake is to treat ordinal responses as continuous and use `fit()`.
This works mechanically but violates distributional assumptions and can
produce misleading p-values — especially with few ordinal levels (3-5) or
skewed category distributions.

---

## Next steps

- See {doc}`api/clmm` for the full API reference and all parameters
- See {doc}`api/coxme` for Cox frailty models (survival data with random effects)
- See {doc}`glmm-quickstart` for binomial, Poisson, and other GLMMs
- See {doc}`quickstart` for linear mixed models
- See {doc}`comparison` for a feature comparison across libraries
