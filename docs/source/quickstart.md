# Quickstart

## Build a dataset

interlace works with any pandas DataFrame. For this example, we generate synthetic
exam scores where students are nested within schools — a classic crossed random effects
structure.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

# Crossed structure: 30 students × 10 schools
student_ids = [f"s{i}" for i in rng.integers(1, 31, n)]
school_ids  = [f"sch{i}" for i in rng.integers(1, 11, n)]

df = pd.DataFrame({
    "score":         60 + rng.normal(0, 8, n),
    "hours_studied": rng.uniform(0, 10, n),
    "prior_gpa":     rng.uniform(2.0, 4.0, n),
    "student_id":    student_ids,
    "school_id":     school_ids,
})
```

## Fit the model

Pass a Patsy-style fixed-effects formula and the grouping column names:

```python
from interlace import fit

result = fit(
    formula="score ~ hours_studied + prior_gpa",
    data=df,
    groups=["student_id", "school_id"],  # crossed random intercepts
)
```

`groups` accepts a single string (one random intercept) or a list (crossed intercepts).
The first entry is the primary grouping factor.

## Inspect results

```python
# Fixed-effect coefficients
print(result.fe_params)
# Intercept         55.12
# hours_studied      0.81
# prior_gpa          1.23

# Standard errors and p-values
print(result.fe_bse)
print(result.fe_pvalues)

# Variance components per grouping factor
print(result.variance_components)
# {'student_id': 12.4, 'school_id': 5.8}

# Residual variance σ²
print(result.scale)

# Model fit
print(result.aic, result.bic)
```

## Access random effects

```python
# BLUPs (Best Linear Unbiased Predictors) for each factor
student_blups = result.random_effects["student_id"]
school_blups  = result.random_effects["school_id"]

print(student_blups.head())
# s1    1.23
# s2   -0.87
# ...
```

## Predict on new data

```python
df_new = pd.DataFrame({
    "hours_studied": [5.0, 8.0],
    "prior_gpa":     [3.2, 3.8],
    "student_id":    ["s1", "s_new"],  # s_new is unseen → shrinks to 0
    "school_id":     ["sch1", "sch2"],
})

preds = result.predict(newdata=df_new)
print(preds)
```

Unseen group levels automatically shrink to the population mean (BLUP = 0).

## Next steps

- See {doc}`examples` for a full walkthrough of diagnostics and plots
- See the {doc}`api/fit` reference for all `fit()` parameters
- See {doc}`api/result` for the complete list of `CrossedLMEResult` attributes
