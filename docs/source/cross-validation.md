# Cross-Validation Guide

Cross-validation estimates how well a fitted model generalises to unseen data.
For mixed models, the key challenge is **data leakage**: observations within the
same group share a random effect, so splitting at the observation level lets the
test set "peek" at training-set information. `interlace.cross_val()` avoids this
by splitting at the **group level**.

---

## Why group-level CV?

Suppose you split randomly and a school appears in both training and test sets.
The model's BLUP for that school is informed by its training observations, giving
an optimistic error estimate on the test observations from the same school.
Group-level CV holds out entire groups, giving a realistic picture of prediction
error for *new, unseen groups*.

---

## Leave-one-group-out (LOGO)

The default strategy, `cv="logo"`, removes one group at a time, fits the model on
the remaining data, and predicts on the held-out group. Repeat for every group.

```python
from interlace import cross_val

cv = cross_val(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups="school_id",    # grouping factor used for both RE and CV folds
    cv="logo",             # default — can be omitted
    scoring="rmse",        # default
)

print(f"LOGO RMSE: {cv.mean:.3f} ± {cv.std:.3f}")
print(f"Per-fold scores: {cv.scores}")
```

LOGO is exact but can be slow when there are many groups, because it fits one
model per unique group level.

---

## K-fold by groups

`cv="kfold"` randomly partitions the unique group labels into `k` folds (default
`k=5`). Each fold holds out all observations from roughly `n_groups / k` groups.
This is faster than LOGO while still preventing leakage.

```python
cv_k = cross_val(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups="school_id",
    cv="kfold",
    k=5,
    scoring="rmse",
)

print(f"5-fold CV RMSE: {cv_k.mean:.3f} ± {cv_k.std:.3f}")
```

The fold assignment uses a fixed seed (42) for reproducibility.

---

## Scoring

### Built-in metrics

| `scoring=` | Metric |
|------------|--------|
| `"rmse"` *(default)* | Root mean squared error |
| `"mae"` | Mean absolute error |

### Custom scoring function

Pass any callable with the signature `scorer(y_true, y_pred) -> float`:

```python
import numpy as np

def mape(y_true, y_pred):
    """Mean absolute percentage error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

cv = cross_val(
    "score ~ hours_studied",
    data=df,
    groups="school_id",
    scoring=mape,
)
print(f"MAPE: {cv.mean:.1f}%")
```

---

## Passing fit arguments

Extra keyword arguments are forwarded to `interlace.fit()`. Use this to select
a different optimizer, set the random-effect structure, or switch to ML:

```python
cv = cross_val(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups="school_id",
    cv="logo",
    random=["(1 + hours_studied | school_id)"],   # random slopes
    optimizer="bobyqa",
)
```

---

## Inspecting per-fold models

Set `return_models=True` to store the fitted model and prediction details for
each fold. Each element of `cv.fold_results` is a dict with keys:

- `"model"` — the fitted `CrossedLMEResult` for that fold
- `"train_groups"` — group labels used for training
- `"test_groups"` — group labels held out
- `"y_true"` — observed response in the test set
- `"y_pred"` — model predictions on the test set

```python
cv = cross_val(
    "score ~ hours_studied + prior_gpa",
    data=df,
    groups="school_id",
    cv="logo",
    return_models=True,
)

for fold in cv.fold_results:
    test_g  = fold["test_groups"]
    rmse    = np.sqrt(np.mean((fold["y_true"] - fold["y_pred"]) ** 2))
    print(f"Held-out group: {test_g}, RMSE: {rmse:.3f}")
```

---

## `CVResult` reference

`cross_val()` returns a `CVResult` dataclass:

| Attribute / property | Type | Description |
|----------------------|------|-------------|
| `scores` | `np.ndarray` | Per-fold score values (length = number of folds) |
| `mean` | `float` | Mean of `scores` |
| `std` | `float` | Standard deviation of `scores` (ddof=1) |
| `fold_results` | `list[dict] \| None` | Only populated when `return_models=True` |

---

## Tips

- **Prefer LOGO when groups are few** (< 20): each group gets exactly one held-out
  fold, and the variance estimate is more informative.
- **Prefer kfold when groups are many** (≥ 50): LOGO becomes expensive; 5- or
  10-fold CV is a good trade-off.
- **Interpret `cv.std` carefully**: with LOGO, fold scores are correlated (each fold
  removes one group from a common pool), so `std` underestimates true uncertainty.
- **Use `return_models=True` sparingly**: storing one fitted model per fold can
  consume substantial memory for large datasets.

---

## See also

- [Model Comparison Guide](model-comparison.md) — LRT and AIC for nested models
- [Quickstart](quickstart.md) — basic `fit()` workflow
- {doc}`api/cross_val` — full `cross_val()` parameter reference
