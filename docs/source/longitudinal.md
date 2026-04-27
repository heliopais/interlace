# Longitudinal Data Guide

Longitudinal (repeated-measures) data have multiple observations per subject
over time. A standard random-intercept LMM accounts for between-subject
differences, but assumes that within-subject residuals are independent. When
measurements close in time are more correlated than measurements far apart,
that assumption fails — and the fix is a **residual correlation structure**.

`interlace` provides two structures via the `correlation` parameter on
`fit()`: **AR(1)** and **compound symmetry (CS)**. Both match R's `nlme`
package (`corAR1` and `corCompSymm`).

---

## When do you need a correlation structure?

A random intercept already captures *some* within-subject dependence — all
observations from the same subject share a common shift. But it implies that
any two observations from the same subject are *equally* correlated,
regardless of how far apart in time they are.

Signs that you need more:

- **Residual autocorrelation** — after fitting a random-intercept model,
  plot the residuals by time within subject. If early residuals predict later
  ones, AR(1) may help.
- **Model comparison favours it** — AIC/BIC drops when you add a correlation
  structure.
- **Domain knowledge** — the outcome has temporal inertia (e.g. blood
  pressure, growth curves, daily mood ratings).

If within-subject observations are exchangeable (e.g. test items within a
session, raters scoring the same stimulus), a random intercept is usually
sufficient.

---

## AR(1) — first-order autoregressive

### What it models

AR(1) assumes that residuals within a group decay in correlation
exponentially with the time gap:

```
Cor(e_i, e_j) = rho^|t_i - t_j|
```

- **rho close to 1** — strong serial dependence; nearby measurements are
  highly correlated
- **rho close to 0** — no serial dependence; residuals are nearly
  independent
- rho is always estimated in (-1, 1)

This is equivalent to a continuous-time AR(1) (Ornstein-Uhlenbeck) process,
so it handles **unequally spaced** time points naturally.

### Example

```python
import interlace
from interlace import AR1

result = interlace.fit(
    "y ~ time + treatment",
    data=df,
    groups="subject",
    correlation=AR1(time="time"),
)

print(result.correlation_params)
# {'rho': 0.72}

print(result.variance_components)
print(result.fe_params)
```

### When to use AR(1)

- Measurements are taken at **multiple time points** per subject
- The correlation between measurements **decays** with temporal distance
- Data may have **unequal spacing** between time points (AR(1) handles
  this; CS does not distinguish time gaps)

---

## Compound symmetry (exchangeable)

### What it models

Compound symmetry (CS) assumes that all pairs of residuals within a group
share the **same** correlation, regardless of time gap:

```
Cor(e_i, e_j) = rho   for all i != j within the same group
```

The within-group covariance matrix is:

```
R_g = sigma^2 * [(1 - rho) I + rho J]
```

where J is the all-ones matrix.

### Example

```python
from interlace import CompoundSymmetry

result = interlace.fit(
    "y ~ time + treatment",
    data=df,
    groups="subject",
    correlation=CompoundSymmetry(time="time"),
)

print(result.correlation_params)
# {'rho': 0.45}
```

### When to use CS

- Repeated measures are approximately **exchangeable** — the within-subject
  correlation does not depend on how far apart measurements are
- You want to model a **residual intraclass correlation** beyond what the
  random intercept captures
- As a reference model to compare against AR(1) via AIC/BIC

---

## AR(1) vs compound symmetry vs random slopes

Three approaches to within-subject dependence, in increasing complexity:

| Approach | What it captures | When to use |
|---|---|---|
| Random intercept only (`groups=`) | Subjects differ in their baseline | Default starting point |
| Compound symmetry (`CompoundSymmetry`) | Subjects have equal residual correlation for all measurement pairs | Exchangeable repeated measures |
| AR(1) (`AR1`) | Residual correlation decays with time gap | Temporal dependence |
| Random slope (`random=["(1+time\|subject)"]`) | Subjects differ in their trajectory over time | Subject-specific growth curves |

These are not mutually exclusive — you can combine a random intercept with
AR(1) correlation:

```python
# Random intercept + AR(1) residual correlation
result = interlace.fit(
    "y ~ time + treatment",
    data=df,
    groups="subject",
    correlation=AR1(time="time"),
)
```

The random intercept captures between-subject variability; the AR(1)
captures within-subject serial dependence in the residuals.

### Choosing between them

**Step 1:** Fit a random-intercept model (no correlation structure).

**Step 2:** Check residual plots for autocorrelation. If residuals within
subjects show temporal patterns, proceed.

**Step 3:** Fit AR(1) and CS models and compare:

```python
from interlace import AR1, CompoundSymmetry

m_base = interlace.fit("y ~ time + treatment", data=df, groups="subject")
m_ar1  = interlace.fit("y ~ time + treatment", data=df, groups="subject",
                        correlation=AR1(time="time"))
m_cs   = interlace.fit("y ~ time + treatment", data=df, groups="subject",
                        correlation=CompoundSymmetry(time="time"))

print(f"Base AIC: {m_base.aic:.1f}")
print(f"AR(1) AIC: {m_ar1.aic:.1f}")
print(f"CS AIC:   {m_cs.aic:.1f}")
```

Pick the model with the lowest AIC. If AR(1) and CS give similar AIC, prefer
the simpler model (CS has the same number of parameters as AR(1), so compare
directly).

**Step 4:** If neither helps much, consider a random slope instead — it
models between-subject differences in *trajectory*, which is a different
(and sometimes more important) source of dependence.

---

## How it works internally

The correlation structure operates by **whitening** the data before the
standard profiled-REML machinery runs. For a given rho:

1. Within each group, compute the Cholesky factor L of the correlation
   matrix R_g
2. Transform y, X, and Z: multiply by L^{-1} within each group
3. Run profiled REML on the whitened data (residuals are now independent)

The correlation parameter rho is estimated jointly with the variance
parameters (theta) in the outer optimisation loop.

---

## Comparison with R (nlme)

| interlace | R (nlme) |
|-----------|----------|
| `fit(..., correlation=AR1(time="time"))` | `lme(..., correlation=corAR1(form=~time\|subject))` |
| `fit(..., correlation=CompoundSymmetry(time="time"))` | `lme(..., correlation=corCompSymm(form=~time\|subject))` |
| `result.correlation_params["rho"]` | `coef(fit$modelStruct$corStruct, unconstrained=FALSE)` |

**Parity:** fixed effects and variance components match R's `nlme::lme()`
within 1e-3 (absolute) and 5% (relative). The correlation parameter (rho)
matches within 2% (relative) on the AR(1) and CS benchmark datasets.

---

## Tips

- **Sort your data by time within subject** before fitting. `interlace`
  does this internally, but sorting yourself makes residual plots easier
  to interpret.
- **AR(1) with unequal spacing** works correctly — the time gaps between
  consecutive observations are computed from the `time` column, not assumed
  to be 1.
- **Negative rho** is allowed but rare in practice. A negative AR(1)
  parameter means consecutive measurements tend to alternate above and
  below the mean.
- **CS + random intercept** — compound symmetry adds residual correlation
  *on top of* the random intercept. If both are positive, the total
  within-subject correlation is ICC + rho*(1 - ICC), which can be
  substantial.

---

## See also

- {doc}`api/correlation` — full API reference for AR1 and CompoundSymmetry
- {doc}`quickstart` — basic `fit()` workflow
- {doc}`random-slopes` — random slopes as an alternative to correlation structures
- {doc}`model-comparison` — comparing models with different structures via AIC/LRT
- {doc}`api/fit` — the `correlation` parameter on `fit()`
