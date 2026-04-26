# Correlation structures

Residual correlation structures for linear mixed models with repeated
measures or longitudinal data. Pass a correlation structure to
`interlace.fit()` via the `correlation` parameter.

The key operation is *whitening*: transforming (y, X, Z) so the residual
covariance becomes identity, allowing the standard profiled-REML machinery
to operate on the transformed data. The correlation parameter (rho) is
estimated jointly with the variance parameters.

## AR(1) — first-order autoregressive

Within each group, residuals separated by time gap dt have correlation
rho^dt (continuous-time AR(1) / Ornstein-Uhlenbeck process).

```{eval-rst}
.. autoclass:: interlace.AR1
   :members:
   :undoc-members:
```

### Example

```python
import interlace
from interlace import AR1

result = interlace.fit(
    formula="y ~ time + treatment",
    data=df,
    groups="subject",
    correlation=AR1(time="time"),
)

# Estimated correlation parameter
print(result.correlation_params)
# {'rho': 0.72}
```

### When to use

AR(1) is appropriate when observations within a group are ordered in time
and the correlation decays with increasing temporal distance. Common in
longitudinal studies with equally or unequally spaced measurements.

---

## CompoundSymmetry — exchangeable correlation

All pairs of residuals within a group share the same correlation rho. The
within-group correlation matrix is R_g = (1-rho)I + rho J, where J is the
all-ones matrix.

```{eval-rst}
.. autoclass:: interlace.CompoundSymmetry
   :members:
   :undoc-members:
```

### Example

```python
from interlace import CompoundSymmetry

result = interlace.fit(
    formula="y ~ time + treatment",
    data=df,
    groups="subject",
    correlation=CompoundSymmetry(time="time"),
)
```

### When to use

Compound symmetry (exchangeable correlation) is appropriate when the
within-group correlation does not depend on temporal distance — e.g. when
repeated measures are interchangeable or when a random intercept alone
does not adequately capture the residual dependence.

---

## Constructor parameters

Both `AR1` and `CompoundSymmetry` accept a single parameter:

| Parameter | Type | Description |
|-----------|------|-------------|
| `time` | `str` | Name of the column in the data containing the time/order variable. For AR(1), used to compute time gaps. For CS, used only for the CorStruct interface. |

## Estimated parameters

After fitting, `result.correlation_params` is a dict containing:

| Key | Description |
|-----|-------------|
| `rho` | Estimated within-group correlation parameter |

## Comparison with R

| interlace | R (nlme) |
|-----------|----------|
| `fit(..., correlation=AR1(time="time"))` | `lme(..., correlation=corAR1(form=~time\|subject))` |
| `fit(..., correlation=CompoundSymmetry(time="time"))` | `lme(..., correlation=corCompSymm(form=~time\|subject))` |
| `result.correlation_params["rho"]` | `coef(fit$modelStruct$corStruct, unconstrained=FALSE)` |

## See also

- {doc}`fit` — the `correlation` parameter on `fit()`
- {doc}`result` — `CrossedLMEResult` attributes including `correlation_params`
