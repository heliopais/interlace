# cross_val

Cross-validate a mixed model with group-aware splits. Supports
leave-one-group-out (LOGO) and k-fold strategies, with pluggable
scoring metrics.

## cross_val

```{eval-rst}
.. autofunction:: interlace.cross_val
```

## CVResult

```{eval-rst}
.. autoclass:: interlace.CVResult
   :members:
   :undoc-members:
```

## Example

```python
import interlace

# Leave-one-group-out cross-validation
cv = interlace.cross_val(
    "rt ~ condition",
    data=df,
    groups="subject",
    cv="logo",
    scoring="rmse",
)
print(cv.mean_score)   # mean RMSE across folds
print(cv.scores)       # per-fold scores

# k-fold with MAE
cv_k = interlace.cross_val(
    "rt ~ condition",
    data=df,
    groups="subject",
    cv="kfold",
    k=5,
    scoring="mae",
)
print(cv_k.mean_score)
```

## See also

- [Cross-Validation Guide](../cross-validation.md) — methodology and interpretation
- {doc}`fit` — underlying fitting function
