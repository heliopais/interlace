# Residuals

Extract marginal or conditional residuals from a fitted model.
Works with both `CrossedLMEResult` and `statsmodels.MixedLMResults`.

```{eval-rst}
.. autofunction:: interlace.hlm_resid
```

## Residual types

| Type | Formula | Interpretation |
|------|---------|----------------|
| **Conditional** | `y − Xβ̂ − Zb̂` | Within-group unexplained variation; accounts for BLUPs |
| **Marginal** | `y − Xβ̂` | Between- and within-group unexplained variation; ignores BLUPs |

Use **conditional residuals** to check within-group model fit (normality, homoscedasticity).
Use **marginal residuals** to assess the overall fit including random-effect structure.

## Example

```python
import interlace
import matplotlib.pyplot as plt
import numpy as np

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

resid_df = interlace.hlm_resid(result)
print(resid_df.columns.tolist())
# ['resid.conditional', 'resid.marginal', 'fitted.conditional', 'fitted.marginal']

# Check normality of conditional residuals
resid_df["resid.conditional"].plot.hist(bins=30, edgecolor="white")
plt.xlabel("Conditional residual")
plt.title("Distribution of conditional residuals")
plt.show()

# Residual vs fitted (conditional)
resid_df.plot.scatter(x="fitted.conditional", y="resid.conditional", alpha=0.4)
plt.axhline(0, linestyle="--", color="grey")
plt.title("Residuals vs Fitted (conditional)")
plt.show()
```

## See also

- {doc}`augment` — append residuals + influence diagnostics to the original DataFrame
- {doc}`leverage` — hat-matrix diagnostics
- {doc}`influence` — Cook's distance and MDFFITS
