# quantreg

Quantile regression fitting and kernel standard errors.
`quantreg()` solves the standard LP formulation via the HiGHS interior-point solver.
`quantreg_ker_se()` computes kernel-based SEs matching R's `summary.rq(se="ker")`.
Accepts any narwhals-compatible DataFrame (pandas, polars, …).

## quantreg

```{eval-rst}
.. autofunction:: interlace.quantreg
```

### QuantRegResult attributes

| Attribute | Type | Description |
|---|---|---|
| `params` | `pd.Series` | Named coefficient estimates |
| `resid` | `np.ndarray` (n,) | Residuals `y − X β̂` |
| `fittedvalues` | `np.ndarray` (n,) | Fitted values `X β̂` |
| `tau` | `float` | Quantile level used for fitting |

### QuantRegResult methods

**`ker_se(hs=True)`** — Kernel standard errors for the coefficients.
Delegates to `quantreg_ker_se()` using the stored residuals and design matrix.
`hs=True` (default) uses the Hall-Sheather bandwidth; `hs=False` uses Bofinger.
Returns `np.ndarray` of shape `(p,)`.

**`predict(data)`** — Predict on new data by re-evaluating the RHS formula.
Accepts any narwhals-compatible frame.
Returns `np.ndarray` of shape `(n_new,)`.

### Examples

```python
import interlace

# Median regression (tau=0.5)
result = interlace.quantreg("salary ~ experience + education", data=df)
print(result.params)
print(result.tau)   # 0.5

# 90th percentile regression
result_90 = interlace.quantreg("salary ~ experience + education", data=df, tau=0.9)

# Kernel standard errors
se = result.ker_se()           # Hall-Sheather bandwidth
se_bof = result.ker_se(hs=False)  # Bofinger bandwidth

# Prediction
import pandas as pd
new_df = pd.DataFrame({"experience": [5, 10], "education": [16, 18]})
preds = result.predict(new_df)
```

---

## quantreg_ker_se / ols_dfbetas_qr

Low-level utilities used internally by the influence diagnostics pipeline.
Exposed publicly for users who need to compute kernel-based standard errors
or QR-based DFBETAS on their own quantile regression fits.

```{eval-rst}
.. autofunction:: interlace.quantreg_ker_se
```

```{eval-rst}
.. autofunction:: interlace.ols_dfbetas_qr
```

These functions replicate the behaviour of R's `quantreg::summary.rq(se="ker")`
and the DFBETAS diagnostic from `car::dfbetas()`. They are used internally by
`hlm_influence()` and `lmer_influence_measures()`.

## See also

- {doc}`ols` — OLS fitting with HC3 robust standard errors
- {doc}`influence` — high-level influence diagnostics
- {doc}`augment` — combined augmented DataFrame with `.cooksd` and `.mdffits`
