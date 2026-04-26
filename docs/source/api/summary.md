# summary and VarCorr

Human-readable summaries and variance-covariance tables for fitted mixed models.

## summary()

`result.summary()` returns a `SummaryResult` that renders an lme4-style
text summary matching `summary.merMod()` in R. The output includes scaled
residuals, random-effects variance table, fixed-effects coefficients with
Satterthwaite DFs and p-values, and convergence status.

```python
import interlace

result = interlace.fit("score ~ hours + gpa", data=df, groups=["student", "school"])
print(result.summary())
```

### statsmodels compatibility

`SummaryResult` has a `.tables` property returning `[info_df, fe_df]`:

- `tables[0]` — model info (method, nobs, log-likelihood, AIC, BIC)
- `tables[1]` — fixed-effects table with columns `Coef.`, `Std.Err.`, `z`,
  `P>|z|`, `[0.025`, `0.975]` — matching `statsmodels.MixedLMResults.summary()`

```python
s = result.summary()
fe_table = s.tables[1]  # DataFrame, compatible with statsmodels consumers
```

---

## VarCorr()

```{eval-rst}
.. autofunction:: interlace.VarCorr
```

Returns variance-covariance components in the same format as R's
`as.data.frame(VarCorr(fit))`.

### Example

```python
from interlace import VarCorr

vc = VarCorr(result)
print(vc.as_dataframe())
#         grp           var1  var2    vcov    sdcor
# 0  school_id  (Intercept)  None   0.412    0.642
# 1   Residual         None  None   1.284    1.133
```

### Columns

| Column | Description |
|--------|-------------|
| `grp` | Grouping factor name, or `"Residual"` |
| `var1` | Random-effect term name (e.g. `"(Intercept)"`) |
| `var2` | Second term for correlations; NaN for variances |
| `vcov` | Variance or covariance value |
| `sdcor` | Standard deviation (diagonal) or correlation (off-diagonal) |

For random-slope models, off-diagonal entries show the correlation between
the intercept and slope random effects.

## See also

- {doc}`result` — `CrossedLMEResult` attributes
- {doc}`fit` — fitting the model
- {doc}`varcorr` — additional VarCorr documentation
