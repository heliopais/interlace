# FAQ & Troubleshooting

Common questions, error messages, and tuning guidance for `interlace`.

---

## Data format requirements

**What format does my data need to be in?**

`interlace.fit()` accepts a **pandas DataFrame** in long (tidy) format — one row per observation.

- The response variable and all predictors must be **numeric** columns (int or float).
- Grouping columns (`groups` / `random`) must be **string or categorical** columns that identify group membership. Integer IDs work too.
- Column names must be valid Python identifiers and must not clash with formula operators. Avoid names containing `~`, `+`, `*`, `:`, `|`, or spaces.

**Missing values**

Rows with `NaN` in the response, any predictor, or any grouping column are **dropped silently** before fitting. Check `result.nobs` against `len(df)` to confirm how many rows were used.

```python
model = interlace.fit("y ~ x", data=df, groups="firm")
print(f"Used {model.nobs} of {len(df)} rows")
```

**Minimum group sizes**

There is no hard minimum, but groups with only one observation cannot contribute information about within-group variance. Random effects for singleton groups are estimated by shrinkage toward zero; treat their BLUPs with caution.

---

## Common error messages

### `ConvergenceWarning: optimizer did not converge`

The REML optimiser hit its iteration limit without finding a stable solution. See [Convergence troubleshooting](#convergence-troubleshooting) below.

### `LinAlgError: singular matrix` or `Matrix is not positive definite`

The model matrix or the estimated covariance matrix is singular. Common causes:

- **Perfect multicollinearity** — a predictor is an exact linear combination of others (e.g. dummy variables that sum to 1). Drop one category or use `pd.get_dummies(..., drop_first=True)`.
- **Near-zero variance component** — a random effect explains essentially no variance. The Hessian becomes ill-conditioned at the boundary. Try removing the offending grouping factor or see [Numerical stability](#numerical-stability).

### `ValueError: groups column 'X' not found in data`

The string passed to `groups=` does not match any column name. Check for trailing spaces or case differences:

```python
print(df.columns.tolist())  # verify exact column names
```

### `ValueError: formula refers to unknown variables`

A variable in the formula is not a column in the DataFrame. Remember that the formula uses Python-style names — check that column names are not reserved words and contain no special characters.

### `KeyError` when accessing `result.random_effects["group"]`

The key must exactly match the column name passed to `groups`. If you passed a list, each element becomes a separate key:

```python
result = interlace.fit("y ~ x", data=df, groups=["student_id", "school_id"])
blups = result.random_effects["student_id"]  # correct
```

---

## Convergence troubleshooting

If the optimiser does not converge, try the following steps in order.

### 1. Rescale your predictors

The REML surface is sensitive to the scale of the design matrix. Centre and standardise continuous predictors before fitting:

```python
from sklearn.preprocessing import StandardScaler

df["x_scaled"] = StandardScaler().fit_transform(df[["x"]])
model = interlace.fit("y ~ x_scaled", data=df, groups="firm")
```

### 2. Increase the iteration limit

Pass `maxiter` to allow more optimisation steps:

```python
model = interlace.fit("y ~ x", data=df, groups="firm", maxiter=500)
```

The default is 100 for the scipy L-BFGS-B backend and 200 for BOBYQA.

### 3. Switch optimisers

The default L-BFGS-B optimiser uses gradients and can struggle near variance-component boundaries. BOBYQA is gradient-free and often more robust in these cases:

```python
# Requires: pip install "interlace-lme[bobyqa]"
model = interlace.fit("y ~ x", data=df, groups="firm", optimizer="bobyqa")
```

### 4. Check your model structure

A model that does not converge is often overparameterised. Ask:

- Do you have enough groups? As a rough guide, at least 5–10 groups per random-effect term.
- Are all random slopes warranted? If a random slope variance is near zero, remove it.
- Are crossed groupings truly crossed, or is one nested in the other?

---

## Model comparison

**Can I compare two models to test if a predictor improves fit?**

Yes — fit both models with `method="ML"` and use a likelihood ratio test (LRT).
Do not compare models fitted with REML when the fixed-effect structures differ.
See the [Model Comparison Guide](model-comparison.md) for the full workflow.

---

## Solver choice: CHOLMOD vs default

`interlace` uses a dense Cholesky factorisation of the random-effect covariance blocks by default. For large models this can be slow.

**When to use CHOLMOD:**

- You have many groups (e.g. > 1 000 firms) or many random-effect levels.
- Fitting takes more than a few seconds on your machine.
- You have a sparse crossing structure (most combinations of grouping levels are unobserved).

**Installing CHOLMOD:**

```bash
pip install "interlace-lme[cholmod]"
```

No code change is required. `interlace` detects the library at runtime and automatically uses the sparse path: symbolic analysis runs once, and the sparsity pattern is reused on every REML iteration.

**When to stick with the default:**

- Small datasets (< 500 observations, < 50 groups).
- Fully-balanced designs where the covariance matrix is dense — CHOLMOD's sparsity savings disappear.
- Environments where compiling SuiteSparse is difficult (e.g. restricted CI images).

---

## Optimizer choice: BOBYQA vs default

`interlace` defaults to scipy's L-BFGS-B, a gradient-based optimizer. BOBYQA (Bound Optimisation BY Quadratic Approximation) is an alternative gradient-free optimizer that more closely matches lme4's internal algorithm.

**When to use BOBYQA:**

- The default optimizer does not converge or produces a `ConvergenceWarning`.
- You need close numerical parity with lme4 (e.g. for reproducing R results or using `hlm_influence` diagnostics).
- Your model has variance components near the boundary (zero), where gradients become unreliable.

**Installing BOBYQA:**

```bash
pip install "interlace-lme[bobyqa]"
```

**Using BOBYQA:**

```python
model = interlace.fit("y ~ x", data=df, groups="firm", optimizer="bobyqa")

# Also pass it to influence functions for consistent results
from interlace.influence import hlm_influence
infl = hlm_influence(model, optimizer="bobyqa")
```

**When to stick with the default:**

- Fast convergence is the priority and you are not comparing to lme4 output.
- You have not installed the `bobyqa` extra (the package is optional).

---

## Numerical stability

### Near-zero variance components

If a random-effect variance is estimated at or near zero, the model is at a boundary of the parameter space. This is not an error — it means the grouping factor explains very little variance — but it can cause:

- Inflated condition numbers and near-singular Hessians.
- Inaccurate standard errors for fixed effects.
- Convergence warnings from the gradient-based optimizer.

**What to do:**

1. Check `result.variance_components` — if a component is `< 1e-4` relative to `result.scale`, consider dropping that grouping factor.
2. Switch to BOBYQA, which handles boundary cases more gracefully.
3. If the near-zero component is meaningful (e.g. a pre-registered random effect), report the boundary result and note the limitation.

### Ill-conditioned fixed-effect design

If your fixed-effect matrix is ill-conditioned (detected by a `LinAlgWarning`), standardise all continuous predictors and ensure categorical variables are not perfectly collinear.

```python
# Check condition number of the design matrix
import numpy as np
import patsy

_, X = patsy.dmatrices("y ~ x1 + x2", data=df)
print(np.linalg.cond(X))  # values > 1e6 indicate near-singularity
```
