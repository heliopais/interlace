# Interpreting model results

After calling `interlace.fit()`, you receive a `CrossedLMEResult` object. This page
explains what each part of the output means and how to use it to draw conclusions from
your model.

Throughout this page we use a running example: exam scores modelled as a function of
hours studied and prior GPA, with students and schools as crossed grouping factors.

```python
import interlace

result = interlace.fit(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups=["student_id", "school_id"],
)
```

---

## Fixed effects

Fixed effects capture the population-level relationships between predictors and the
outcome, controlling for the grouping structure.

```python
result.fe_params      # coefficient estimates (pd.Series)
result.fe_bse         # standard errors
result.fe_pvalues     # two-sided p-values (normal approximation)
result.fe_conf_int    # 95% confidence intervals (pd.DataFrame, columns [0.025, 0.975])
```

### Reading the coefficients

`fe_params` is a `pandas.Series` indexed by term name:

```python
print(result.fe_params)
# Intercept         55.12
# hours_studied      0.81
# prior_gpa          1.23
```

Each coefficient is the expected change in the outcome per unit increase in the
predictor, **holding grouping structure constant** — i.e., for a student in the same
school compared to themselves at a different value of the predictor. This is the
within-group effect.

### Standard errors and inference

Standard errors in `fe_bse` account for the clustering: they are derived from the
fixed-effect covariance matrix `fe_cov = scale * (X'Ω⁻¹X)⁻¹`, where Ω encodes the
random effect structure. This is why they are typically larger than OLS standard errors
on the same data — the effective sample size is reduced by the grouping.

```python
import pandas as pd

summary = pd.DataFrame({
    "estimate": result.fe_params,
    "se":       result.fe_bse,
    "p":        result.fe_pvalues,
    "ci_lo":    result.fe_conf_int.iloc[:, 0],
    "ci_hi":    result.fe_conf_int.iloc[:, 1],
})
print(summary)
```

P-values use a normal approximation (z-test), which is appropriate for large samples.
For small samples with few groups, consider using likelihood-ratio tests or
parametric bootstrap instead.

### Comparing models with different fixed structures

To test whether a predictor improves fit, refit with `method="ML"` (not REML) and
compare log-likelihoods:

```python
m1 = interlace.fit("score ~ hours_studied",           data=df, groups=["student_id", "school_id"], method="ML")
m2 = interlace.fit("score ~ hours_studied + prior_gpa", data=df, groups=["student_id", "school_id"], method="ML")

lr_stat = 2 * (m2.llf - m1.llf)
# chi-squared with df = difference in number of parameters
```

Use REML (the default) for your final reported estimates. Use ML only for likelihood
ratio tests comparing fixed structures. See [Concepts: REML](concepts.md#reml-restricted-maximum-likelihood)
for the reason.

---

## Variance components

Variance components quantify how much variability in the outcome is attributable to
each grouping factor and to residual within-group noise.

```python
result.variance_components   # dict {group_col: σ²_group}
result.scale                 # σ²_residual (within-group variance)
```

### Reading variance components

```python
print(result.variance_components)
# {'student_id': 12.4, 'school_id': 5.8}

print(result.scale)
# 38.2
```

`student_id: 12.4` means the between-student variance is 12.4 points² — students
differ from each other by a standard deviation of roughly √12.4 ≈ 3.5 points, after
accounting for hours studied and prior GPA.

`school_id: 5.8` means schools vary with σ ≈ 2.4 points.

`scale = 38.2` is the remaining within-student-within-school variance (σ ≈ 6.2 points).

### Intraclass correlation (ICC)

The ICC tells you what fraction of total variance each grouping factor explains:

```python
total = sum(result.variance_components.values()) + result.scale

icc = {g: v / total for g, v in result.variance_components.items()}
print(icc)
# {'student_id': 0.22, 'school_id': 0.10}
```

An ICC of 0.22 for students means 22% of the total score variability is due to
stable student-level differences (ability, motivation, etc.) not captured by the
predictors. An ICC below ~0.05 suggests the grouping factor contributes little and
may not need a random intercept.

### What if a variance component is near zero?

A variance component close to zero (sometimes reported as exactly 0 at the boundary of
the parameter space) means the corresponding grouping factor explains almost no
variance in the outcome after conditioning on the fixed effects. This is not an error —
it is a valid result indicating that the groups in that factor are essentially
homogeneous. You may choose to drop that grouping factor and refit, or retain it for
theoretical reasons.

---

## Model fit

```python
result.aic        # Akaike Information Criterion (lower is better)
result.bic        # Bayesian Information Criterion (lower is better)
result.llf        # log-likelihood at optimum
result.converged  # True if the optimiser found a solution
```

### AIC and BIC

AIC and BIC are used to compare models with the **same fixed structure** fitted with
REML, or models with **different fixed structures** fitted with ML. Both penalise
model complexity; BIC penalises more heavily for large samples.

A difference of ≥ 4 AIC points between models is often treated as meaningful; a
difference of 1–3 is weak evidence. These are heuristics — use them alongside
subject-matter reasoning, not as automatic decision rules.

### Checking convergence

```python
if not result.converged:
    print("Warning: optimiser did not converge — estimates may be unreliable")
```

Non-convergence typically occurs with very small datasets, degenerate variance
(one group per level), or near-zero variance components. See the
[FAQ](faq.md#convergence) for troubleshooting steps.

---

## Random effects (BLUPs)

BLUPs (Best Linear Unbiased Predictors) are the estimated deviations of each group
from the population mean. They are accessible per grouping factor:

```python
result.random_effects          # dict {group_col: pd.Series or pd.DataFrame}

student_blups = result.random_effects["student_id"]
school_blups  = result.random_effects["school_id"]
```

### Reading BLUPs

For a random intercepts model, each BLUP is a scalar per group level:

```python
print(student_blups.sort_values())
# s14   -4.21
# s07   -3.88
# ...
# s22    3.14
# s03    4.67
```

A BLUP of +4.67 for student `s03` means that, after accounting for `hours_studied`,
`prior_gpa`, and the school-level effect, `s03` scores about 4.7 points above the
average student.

For random slopes models, `random_effects[group]` is a `pd.DataFrame` with one column
per term (intercept + slope).

### Shrinkage

BLUPs are **shrunk toward zero** relative to the group's raw deviation. Groups with
fewer observations or groups whose variance component is small are shrunk more. This
is intentional: a student with only one observation should not be assigned an extreme
intercept — the BLUP borrows strength from the population distribution.

You can observe shrinkage by comparing a BLUP to the raw group mean residual:

```python
# Raw group mean residual (no shrinkage)
raw = df.assign(resid=result.resid).groupby("student_id")["resid"].mean()

# Shrinkage is visible: BLUPs are pulled toward zero
import matplotlib.pyplot as plt
plt.scatter(raw, student_blups.reindex(raw.index))
plt.axline((0, 0), slope=1, linestyle="--")
plt.xlabel("raw group mean residual"); plt.ylabel("BLUP")
```

### BLUPs vs fixed group dummies

BLUPs are not the same as including a fixed dummy variable for each group. Dummies
use only the data from that group; BLUPs partially pool across groups. For groups with
very many observations, BLUPs converge to the dummy-variable estimate. For small or
sparse groups, BLUPs are much more stable.

---

## Predictions

`result.predict()` returns fitted values, optionally including random effects:

```python
# Conditional predictions: uses BLUPs (default)
y_hat = result.predict()

# Marginal predictions: ignores random effects, uses only fixed effects
y_hat_marginal = result.predict(include_re=False)
```

### Conditional vs marginal predictions

| | Conditional (`include_re=True`) | Marginal (`include_re=False`) |
|---|---|---|
| Includes BLUPs | Yes | No |
| Use case | In-sample fit, known groups | New groups, population-level |
| Unseen group levels | BLUP = 0 (shrinks to mean) | Identical to marginal |

**Conditional predictions** are appropriate when you know which group an observation
belongs to and want the best estimate for that specific group. They are used to
compute residuals (`result.resid = observed − conditional prediction`).

**Marginal predictions** give the expected outcome for a hypothetical average group
— useful for reporting population-level effects or when the new data includes groups
not seen during training.

### Predicting on new data

```python
df_new = pd.DataFrame({
    "hours_studied": [5.0, 8.0],
    "prior_gpa":     [3.2, 3.8],
    "student_id":    ["s01", "s_new"],   # s_new has no BLUP → shrinks to 0
    "school_id":     ["sch1", "sch2"],
})

preds = result.predict(newdata=df_new)
```

Unseen group levels automatically receive a BLUP of zero — the population mean for
that factor. This is the correct Bayesian-optimal behaviour under the model's
assumptions.

---

## Bootstrap standard error

For scalar statistics of the outcome (currently: median), `bootstrap_se()` computes
a cluster-bootstrap standard error that respects the grouping structure:

```python
se = result.bootstrap_se(statistic="median", n_bootstrap=1000, seed=42)
print(f"Median SE (cluster bootstrap): {se:.4f}")
```

### Why cluster bootstrap?

Ordinary bootstrap resamples individual observations, which underestimates the SE when
group variance is substantial — it can resample multiple observations from the same
group, inflating the apparent precision. Cluster bootstrap resamples **group levels**
with replacement, then includes all observations from the sampled groups. This
preserves the within-group correlation and gives a valid SE.

Use `resample_level="observation"` only if you have verified that the ICC is
negligible.

---

## Quick reference

| Attribute | Type | What it tells you |
|---|---|---|
| `fe_params` | `pd.Series` | Fixed-effect coefficient estimates |
| `fe_bse` | `pd.Series` | Standard errors of fixed effects |
| `fe_pvalues` | `pd.Series` | Two-sided p-values (z-test) |
| `fe_conf_int` | `pd.DataFrame` | 95% confidence intervals |
| `fe_cov` | `np.ndarray` | Fixed-effect covariance matrix |
| `variance_components` | `dict` | Between-group variance per factor |
| `scale` | `float` | Within-group (residual) variance σ² |
| `random_effects` | `dict` | BLUPs per grouping factor |
| `fittedvalues` | `np.ndarray` | Conditional fitted values |
| `resid` | `np.ndarray` | Conditional residuals |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `llf` | `float` | Log-likelihood at optimum |
| `converged` | `bool` | Whether the optimiser converged |
| `nobs` | `int` | Number of observations |
| `ngroups` | `dict` | Number of levels per grouping factor |

---

## Where to go next

- [Diagnostics guide](diagnostics.md) — residuals, leverage, Cook's D, influence plots
- [API: CrossedLMEResult](api/result.md) — full attribute and method reference
- [API: Prediction](api/predict.md) — `predict()` parameter details
- [Concepts](concepts.md) — background on REML, BLUPs, and variance components
