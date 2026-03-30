# VarCorr

Return variance-covariance components from a fitted mixed model.
Mirrors `lme4::VarCorr()` and returns a structured result object with
an `as_dataframe()` method for tidy output.

```{eval-rst}
.. autofunction:: interlace.VarCorr
```

## VarCorrResult

```{eval-rst}
.. autoclass:: interlace.summary.VarCorrResult
   :members:
   :undoc-members:
```

## Example

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

vc = interlace.VarCorr(result)
print(vc.as_dataframe())
#       grp          var1   var2  vcov     sdcor
# 0  subject  (Intercept)   None  0.45    0.671
# 1     item  (Intercept)   None  0.12    0.346
# 2 Residual         None   None  0.38    0.616
```

## See also

- {doc}`result` — `CrossedLMEResult.variance_components` dict
- {doc}`simulate` — parametric simulation using variance components
