# Augment

Combine residuals, predictions, and influence diagnostics into a single tidy DataFrame —
one row per observation in the original data. Useful for plotting and downstream
analysis. Works with both `CrossedLMEResult` and `statsmodels.MixedLMResults`.

```{eval-rst}
.. autofunction:: interlace.hlm_augment
```

## Returned columns

| Column | Description |
|--------|-------------|
| `.fitted` | Conditional fitted values (fixed + random effects) |
| `.resid` | Conditional residuals (observed − fitted) |
| `.leverage` | Total hat-matrix diagonal (`fixef` + `ranef`) |
| `.cooksd` | Cook's distance (case-deletion influence on fixed effects) |
| `.mdffits` | MDFFITS (scale-free Cook's D) |

All original columns from the input DataFrame are preserved.

## Example

```python
import interlace
import matplotlib.pyplot as plt

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

aug = interlace.hlm_augment(result)
print(aug.columns.tolist())
# ['subject', 'item', 'condition', 'rt',
#  '.fitted', '.resid', '.leverage', '.cooksd', '.mdffits']

# Residual vs fitted plot
aug.plot.scatter(x=".fitted", y=".resid", alpha=0.4)
plt.axhline(0, linestyle="--", color="grey")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# Flag influential observations
influential = aug[aug[".cooksd"] > 4 / len(aug)]
print(f"{len(influential)} influential observations")
print(influential[["subject", "item", ".cooksd"]].sort_values(".cooksd", ascending=False))
```

## See also

- {doc}`residuals` — compute residuals only
- {doc}`leverage` — compute leverage only
- {doc}`influence` — full influence diagnostics with additional metrics
