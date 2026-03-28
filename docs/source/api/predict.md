# Prediction

Generate predictions from a fitted `CrossedLMEResult`. The recommended interface is
`result.predict(newdata=...)` — this page documents the underlying function and its
parameters. Works with both `CrossedLMEResult` and `statsmodels.MixedLMResults`.

```{eval-rst}
.. autofunction:: interlace.predict.predict
```

## Conditional vs marginal predictions

| Type | Includes BLUPs? | Use when |
|------|-----------------|----------|
| **Conditional** | Yes | Predicting for known groups already in the training data |
| **Marginal** | No (fixed effects only) | Predicting for new, unseen groups |

Unseen group levels automatically receive a BLUP of zero (pure population mean) —
no special handling required.

## Examples

### Predicting for known groups

```python
import pandas as pd

df_new = pd.DataFrame({
    "hours_studied": [5.0, 8.0],
    "prior_gpa":     [3.2, 3.8],
    "student_id":    ["s1", "s2"],   # known groups — BLUPs applied
    "school_id":     ["sch1", "sch1"],
})

preds = result.predict(newdata=df_new)
print(preds)  # conditional predictions including group offsets
```

### Predicting for new (unseen) groups

```python
df_new = pd.DataFrame({
    "hours_studied": [6.0],
    "prior_gpa":     [3.5],
    "student_id":    ["s_new"],  # unseen → BLUP shrinks to 0
    "school_id":     ["sch_new"],
})

preds = result.predict(newdata=df_new)
# Returns marginal prediction (population average for these covariate values)
```

### Marginal predictions (fixed effects only)

```python
# Set all group columns to a sentinel unseen value, or omit them if supported
preds_marginal = result.predict(newdata=df_new, use_re=False)
```

## See also

- {doc}`result` — `CrossedLMEResult` attributes
- {doc}`augment` — append predictions and residuals to the original DataFrame
