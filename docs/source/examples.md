# Examples

All examples use the same synthetic exam scores dataset. Run this setup block first:

```python
import numpy as np
import pandas as pd
from interlace import fit

rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "score":         60 + rng.normal(0, 8, n),
    "hours_studied": rng.uniform(0, 10, n),
    "prior_gpa":     rng.uniform(2.0, 4.0, n),
    "student_id":    [f"s{i}"   for i in rng.integers(1, 31, n)],
    "school_id":     [f"sch{i}" for i in rng.integers(1, 11, n)],
})

result = fit(
    formula="score ~ hours_studied + prior_gpa",
    data=df,
    groups=["student_id", "school_id"],
)
```

---

## Single grouping factor

When you only have one source of clustering, pass a single column name:

```python
result_single = fit(
    formula="score ~ hours_studied + prior_gpa",
    data=df,
    groups="student_id",
)

print(result_single.variance_components)
# {'student_id': 14.2}
```

---

## Residuals

`hlm_resid` returns a DataFrame with `.resid` and `.fitted` columns alongside the
original data. Use `type="marginal"` to ignore random effects, or
`type="conditional"` to subtract predicted BLUPs.

```python
from interlace import hlm_resid

# Marginal residuals: y - Xβ
marginal = hlm_resid(result, type="marginal")
print(marginal[[".resid", ".fitted"]].describe())

# Conditional residuals: y - Xβ - Zû
conditional = hlm_resid(result, type="conditional")

# Standardised
std_resid = hlm_resid(result, type="conditional", standardized=True)

# Group-level random effects
school_re = hlm_resid(result, level="school_id")
print(school_re.head())
```

---

## Leverage

The hat-matrix diagonal is decomposed into fixed-effect and random-effect components
following Demidenko & Stukel (2005) and Nobre & Singer (2007).

```python
from interlace import leverage

lev = leverage(result)
print(lev.columns)
# ['overall', 'fixef', 'ranef', 'ranef.uc']

# High-leverage observations
high_lev = lev[lev["overall"] > 2 * lev["overall"].mean()]
print(f"{len(high_lev)} high-leverage observations")
```

---

## Influence diagnostics

`hlm_influence` fits the model *n* times with one observation (or group) deleted,
computing Cook's D, MDFFITS, COVTRACE, COVRATIO, and RVC for each deletion.

```python
from interlace import hlm_influence

# Observation-level influence
infl = hlm_influence(result, level=1)
print(infl.columns)
# ['index', 'cooksd', 'mdffits', 'covtrace', 'covratio',
#  'rvc.var_student_id', 'rvc.var_school_id', 'rvc.error_var']

# Group-level influence (delete one school at a time)
school_infl = hlm_influence(result, level="school_id")
```

### Cook's distance and MDFFITS

```python
from interlace import cooks_distance, mdffits

cd  = cooks_distance(result)   # np.ndarray, shape (n,)
mdf = mdffits(result)          # np.ndarray, shape (n,)

print(f"Max Cook's D: {cd.max():.4f}")
```

### Count and measure influential observations

```python
from interlace import n_influential, tau_gap

# Count observations exceeding the 4/n heuristic threshold
n_inf = n_influential(result)
print(f"{n_inf} influential observations (Cook's D > 4/n)")

# Change in random-effects std devs after removing influential observations
gap = tau_gap(result)
print(gap)
# {'student_id': 0.31, 'school_id': 0.12}
```

---

## Combined augmented DataFrame

`hlm_augment` is a convenience wrapper that returns a single DataFrame containing the
original data, conditional residuals, fitted values, and all influence statistics.
Useful for exploratory analysis or downstream filtering.

```python
from interlace import hlm_augment

aug = hlm_augment(result)
print(aug.columns.tolist())
# ['score', 'hours_studied', 'prior_gpa', 'student_id', 'school_id',
#  '.resid', '.fitted', 'index', 'cooksd', 'mdffits', ...]

# Find the most influential observations
aug.nlargest(5, "cooksd")[["student_id", "school_id", "score", "cooksd"]]
```

Skip the influence refit loop (faster, residuals only):

```python
aug_fast = hlm_augment(result, include_influence=False)
```

---

## Prediction on new data

```python
df_new = pd.DataFrame({
    "hours_studied": [3.0, 7.0, 5.0],
    "prior_gpa":     [2.5, 3.8, 3.1],
    "student_id":    ["s1", "s2", "s_new"],  # s_new is unseen
    "school_id":     ["sch1", "sch1", "sch_new"],
})

# Conditional prediction (known BLUPs applied, unknown → 0)
y_hat = result.predict(newdata=df_new)

# Fixed-effects only (population-level)
y_fe = result.predict(newdata=df_new, include_re=False)
```

---

## Plotting

All plots return `plotnine.ggplot` objects and can be further customised with
standard plotnine layers.

### Residual plots

```python
from interlace import hlm_resid, plot_resid

resid_df = hlm_resid(result, type="conditional")

plot_resid(resid_df, type="resid_vs_fitted")   # residuals vs fitted values
plot_resid(resid_df, type="qq")                # Normal Q-Q plot
```

### Influence index plot

```python
from interlace import hlm_influence, plot_influence

infl = hlm_influence(result, level=1)

plot_influence(infl, diag="cooksd")    # Cook's D by observation index
plot_influence(infl, diag="mdffits")
```

### Ranked dotplot with outlier labels

`dotplot_diag` ranks observations by the chosen diagnostic and labels any that
exceed 3 × IQR above Q3.

```python
from interlace import dotplot_diag

dotplot_diag(infl, diag="cooksd", cutoff="internal", name="index")

# With a manual threshold
dotplot_diag(infl, diag="cooksd", cutoff=4 / len(df))
```
