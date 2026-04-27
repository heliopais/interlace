# Cox frailty quickstart

This page walks through fitting Cox proportional hazards models with
Gaussian frailty (shared frailty models) using `interlace.coxme()`.
These models are for **time-to-event data** where observations are
clustered within groups — patients within hospitals, recurrent events
within subjects, survival times within treatment centres.

For continuous outcomes, see the [LMM quickstart](quickstart.md).
For binary or count outcomes, see the [GLMM quickstart](glmm-quickstart.md).
For ordinal outcomes, see the [CLMM quickstart](clmm-quickstart.md).

---

## When to use a Cox frailty model

Use `coxme()` when:

- Your outcome is **time-to-event** (survival time, duration, time-to-failure)
  with possible **right censoring**
- Observations are **clustered** within groups (centres, subjects, families)
  and you want to account for unobserved group-level heterogeneity
- You would otherwise fit a standard Cox PH model (`coxph`) but need random
  effects for the grouping structure

The "frailty" is the survival analysis term for a random effect — it captures
unobserved group-level factors that make some groups more or less prone to
the event.

---

## The model

A Cox frailty model extends the standard Cox PH model with a group-level
random effect:

```
h(t | x_i, b_j) = h_0(t) * exp(x_i' beta + b_j)
```

- **h_0(t)** is the unspecified baseline hazard (no distributional assumption)
- **beta** are log hazard ratios for the fixed-effect covariates
- **b_j ~ N(0, sigma^2)** are the group-level frailties

Groups with b_j > 0 have higher hazard (faster event occurrence); groups
with b_j < 0 have lower hazard (slower event occurrence).

---

## Basic example: multi-centre clinical trial

Patients are treated at different hospitals. We want to estimate the
treatment effect while accounting for hospital-level variability in
patient outcomes.

```python
import interlace
import pandas as pd

result = interlace.coxme(
    formula="Surv(time, event) ~ treatment + age",
    data=df,
    groups="hospital",
)
```

The `Surv(time, event)` syntax mirrors R — `time` is the follow-up
duration and `event` is an indicator (1 = event occurred, 0 = censored).

### Inspect results

```python
# Log hazard ratios
print(result.fe_params)
# treatment    -0.42
# age           0.03

# Standard errors and p-values
print(result.fe_bse)
print(result.fe_pvalues)

# 95% confidence intervals
print(result.fe_conf_int)
#              2.5 %  97.5 %
# treatment   -0.71   -0.13
# age          0.01    0.05

# Frailty variance (between-hospital heterogeneity)
print(result.variance_components)
# {'hospital': 0.18}

# Concordance (C-statistic)
print(f"C-index: {result.concordance:.3f}")

# Model fit
print(f"AIC: {result.aic:.1f}  BIC: {result.bic:.1f}")
```

### Interpreting fixed effects

Coefficients are **log hazard ratios**. To get hazard ratios:

```python
import numpy as np

print("Hazard ratios:")
for name in result.fe_params.index:
    beta = result.fe_params[name]
    ci = result.fe_conf_int.loc[name]
    print(f"  {name}: HR = {np.exp(beta):.3f} "
          f"[{np.exp(ci['2.5 %']):.3f}, {np.exp(ci['97.5 %']):.3f}]")
# treatment: HR = 0.657 [0.492, 0.878]  — treatment reduces hazard by 34%
# age:       HR = 1.030 [1.010, 1.051]  — each year increases hazard by 3%
```

An HR < 1 means the covariate is protective (reduces hazard); HR > 1 means
it increases hazard.

### Interpreting frailty

The frailty variance (`result.variance_components`) measures how much
hospitals differ in their baseline event rates after adjusting for
covariates:

- **Large sigma^2** — substantial unexplained hospital-level variation;
  patients at some hospitals do systematically better or worse
- **Small sigma^2 (near zero)** — hospitals are similar after adjusting
  for patient characteristics; a standard Cox PH model may suffice

Individual hospital frailties (BLUPs) are available in
`result.random_effects`:

```python
print(result.random_effects["hospital"])
# hosp_01    0.32    — higher-than-average hazard
# hosp_02   -0.18    — lower-than-average hazard
# ...
```

---

## Prediction

`predict()` supports two output types:

### Linear predictor

```python
# x'beta + z'b (in-sample)
lp = result.predict()

# On new data
lp_new = result.predict(newdata=df_new)

# Fixed effects only (no frailty — for new hospitals)
lp_marginal = result.predict(newdata=df_new, include_re=False)
```

### Hazard ratios

```python
# exp(x'beta + z'b)
hr = result.predict(type="risk")
hr_new = result.predict(newdata=df_new, type="risk")
```

An HR of 2.0 means that observation has twice the instantaneous hazard
of the baseline.

### Unseen groups

Unseen group levels at prediction time receive a frailty of zero
(population-average prediction):

```python
new_data = pd.DataFrame({
    "time": [12.0, 24.0],
    "event": [1, 0],
    "treatment": [1, 0],
    "age": [55, 62],
    "hospital": ["new_hosp_A", "new_hosp_B"],
})
hr = result.predict(newdata=new_data, type="risk")
```

---

## Baseline hazard

The Breslow estimate of the cumulative baseline hazard is available as a
DataFrame:

```python
print(result.baseline_hazard.head())
#    time    hazard
# 0  0.5     0.012
# 1  1.0     0.031
# 2  1.5     0.055
# ...
```

Combined with the linear predictor, this gives the full survival curve for
any observation.

---

## Crossed frailty

Like `fit()` and `glmer()`, `coxme()` supports crossed random effects.
If patients are treated at hospitals by different surgeons, both hospital
and surgeon are independent sources of frailty:

```python
result = interlace.coxme(
    "Surv(time, event) ~ treatment + age",
    data=df,
    groups=["hospital", "surgeon"],
)

print(result.variance_components)
# {'hospital': 0.18, 'surgeon': 0.07}

print(result.random_effects["hospital"])
print(result.random_effects["surgeon"])
```

---

## Concordance (C-statistic)

The concordance index (C-statistic, Harrell's C) measures the model's
discriminative ability — the probability that for a random pair of
observations, the one with the higher predicted risk experiences the event
first:

- **C = 0.5** — no better than chance
- **C = 0.7-0.8** — acceptable discrimination
- **C > 0.8** — strong discrimination

```python
print(f"C-index: {result.concordance:.3f}")
```

---

## Summary

`summary()` prints a human-readable overview:

```python
print(result.summary())
```

---

## Comparison with R

| interlace | R (coxme) |
|-----------|-----------|
| `interlace.coxme("Surv(t,e) ~ x", data, groups="g")` | `coxme(Surv(t,e) ~ x + (1\|g), data)` |
| `result.fe_params` | `fixef(fit)` |
| `result.fe_bse` | `sqrt(diag(vcov(fit)))` |
| `result.random_effects["g"]` | `ranef(fit)$g` |
| `result.variance_components` | `VarCorr(fit)` |
| `result.concordance` | `concordance(fit)$concordance` |
| `result.predict(type="risk")` | `predict(fit, type="risk")` |
| `result.baseline_hazard` | `basehaz(fit)` |

**Parity:** fixed effects match R's `coxme::coxme()` within 1e-3
(absolute), variance components within 10% (relative). Standard errors
use the exact Breslow Hessian (not diagonal approximation), matching R
to < 1%.

---

## When to use coxme vs other models

| Your data | Use |
|---|---|
| Time-to-event with grouped observations | `coxme()` |
| Time-to-event without grouping | Standard Cox PH (e.g. `lifelines.CoxPHFitter`) |
| Ordinal outcomes (ratings, severity) | `clmm()` |
| Binary outcomes (success/failure) | `glmer(family="binomial")` |
| Continuous outcomes | `fit()` |

---

## Next steps

- See {doc}`api/coxme` for the full API reference and all parameters
- See {doc}`clmm-quickstart` for ordinal regression with random effects
- See {doc}`glmm-quickstart` for binomial, Poisson, and other GLMMs
- See {doc}`quickstart` for linear mixed models
- See {doc}`comparison` for a feature comparison across libraries
