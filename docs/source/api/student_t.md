# Student-t (robust) family

Linear mixed models with Student-t residuals for robustness against
heavy-tailed errors or outliers. Estimated via EM with latent scale
variables (Lange, Little & Taylor 1989): each EM iteration reduces to a
weighted Gaussian LMM, reusing the profiled-REML machinery.

```{eval-rst}
.. autofunction:: interlace.student_t.student_t_fit
```

```{eval-rst}
.. autoclass:: interlace.student_t.StudentTResult
   :members:
```

## When to use

- Residual heavy tails / outliers that would distort Gaussian REML.
- Compensation, lifetime, or count-like outcomes after transformation,
  where a small number of observations have disproportionate influence.
- Replacing a Bayesian Student-t LMM (e.g. cmdstanpy) with a faster
  frequentist point-estimator while keeping the robustness property.

## Quickstart

```python
import interlace

result = interlace.fit(
    "y ~ x1 + x2",
    data=df,
    groups=["g1", "g2"],     # crossed random intercepts
    family="student_t",      # robust residuals
)
print(result.nu)             # estimated degrees of freedom
print(result.fe_params)
print(result.variance_components)
```

Equivalently via the explicit entry point:

```python
from interlace.student_t import student_t_fit

result = student_t_fit(
    formula="y ~ x1 + x2",
    data=df,
    groups=["g1", "g2"],
    nu=None,                 # None → estimate; pass a number > 2 to fix
    weights=df["w"].values,  # optional observation weights
)
```

## Notes

- `nu` must satisfy `nu > 2` (variance is undefined otherwise).
- `nu` is weakly identified above ~10; the likelihood becomes flat and
  the estimator approaches the Gaussian LMM. Use `nu_max` to cap.
- Observation `weights` enter multiplicatively on the log-likelihood,
  identical in meaning to {func}`interlace.fit`.
