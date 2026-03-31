# ols

Ordinary least squares fitting with formulaic design matrices and HC3 robust standard errors.
Statsmodels-free drop-in replacement for `statsmodels.OLS`.
Accepts any narwhals-compatible DataFrame (pandas, polars, …).

```{eval-rst}
.. autofunction:: interlace.ols
```

## OLSResult attributes

| Attribute | Type | Description |
|---|---|---|
| `params` | `pd.Series` | Named coefficient estimates; supports `.get(name)` and `.values` |
| `resid` | `np.ndarray` (n,) | Residuals `y − X β̂` |
| `fittedvalues` | `np.ndarray` (n,) | Fitted values `X β̂` |
| `normalized_cov_params` | `np.ndarray` (p, p) | `(XᵀX)⁻¹` — unscaled covariance matrix |
| `model.exog` | `np.ndarray` (n, p) | Design matrix |
| `model.endog` | `np.ndarray` (n,) | Response vector |
| `model.exog_names` | `list[str]` | Column names from the formula |
| `model.formula` | `str` | Original formula string |
| `model.data.frame` | native frame | Original data as passed by the caller |
| `df_resid` | `float` | Residual degrees of freedom: `n − p`, matching statsmodels |
| `mse_resid` | `float` | Mean squared error of residuals: `RSS / df_resid`, matching statsmodels |

## OLSResult methods

### `hc3_bse()`

HC3 heteroskedasticity-consistent standard errors, matching statsmodels to 6 significant figures.

```
h_ii    = diag(X (XᵀX)⁻¹ Xᵀ)
d_i     = eᵢ² / (1 − hᵢᵢ)²
cov_HC3 = (XᵀX)⁻¹ (Xᵀ diag(d) X) (XᵀX)⁻¹
se_HC3  = sqrt(diag(cov_HC3))
```

Returns `np.ndarray` of shape `(p,)`.

### `predict(data)`

Re-evaluates the RHS formula on new data via the stored formulaic `ModelSpec`.
Accepts any narwhals-compatible frame.
Returns `np.ndarray` of shape `(n_new,)`.

## Examples

### Basic fit

```python
import interlace

result = interlace.ols("score ~ age + education", data=df)

print(result.params)
# Intercept    50.0
# age            1.5
# education      3.2
# dtype: float64
```

### HC3 robust standard errors

```python
se = result.hc3_bse()
import pandas as pd
print(pd.Series(se, index=result.params.index))
```

### Prediction on new data

```python
import pandas as pd

new_df = pd.DataFrame({"age": [30, 40], "education": [16, 18]})
preds = result.predict(new_df)
```

### Polars input

```python
import polars as pl

df_pl = pl.from_pandas(df)
result = interlace.ols("score ~ age + education", data=df_pl)
```

## See also

- {doc}`quantreg` — quantile regression fitting
- {doc}`influence` — influence diagnostics (uses `ols()` internally)
