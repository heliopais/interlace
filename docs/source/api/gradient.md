# Analytical REML gradient

The `use_gradient=True` parameter on `fit_reml()` enables an exact analytical gradient
for the L-BFGS-B optimizer, reducing the number of objective evaluations by approximately
3× compared to the default finite-difference approximation.

This is an **advanced / internal parameter** exposed via `fit_reml()`. Users of the
high-level `interlace.fit()` API do not need to set it directly.

## Parameter

`use_gradient` is a keyword argument on `fit_reml()`:

```python
from interlace.profiled_reml import fit_reml

result = fit_reml(y, X, Z, q_sizes, use_gradient=True)
```

When `True`, the analytical gradient is passed as `jac=` to
`scipy.optimize.minimize(..., method="L-BFGS-B")`.

## How it works

The gradient of the profiled REML deviance with respect to each variance parameter
`θₖ` is decomposed into three terms via the chain rule:

```
d obj / d θₖ = tr(A11⁻¹ · dA11ₖ)          [log|A11| term]
             + tr(MX⁻¹  · dMXₖ)            [log|MX|  term]
             + (n−p)/yPy · d(yPy)/d θₖ     [log(yPy) term]
```

The `yPy` gradient simplifies to:

```
2 · f[idx_k] @ (ZtZ[idx_k,:] λ_f − Zt_resid[idx_k])
```

where `f = c1 − C_X β̂` is the BLUP residual in random-effect space.

The gradient matches finite differences to `rtol < 1e-3` on single- and two-factor
crossed intercept models.

## When to use it

| Scenario | Recommendation |
|---|---|
| Large datasets, tight convergence needed | `use_gradient=True` — ~3× fewer objective evaluations |
| `optimizer="bobyqa"` | Not applicable — BOBYQA is gradient-free |
| Debugging / numerical instability | Use default (`False`) — finite differences are more robust near boundaries |

## Caveats

- Only supported with `optimizer="lbfgsb"` (the default). Passing `use_gradient=True`
  with `optimizer="bobyqa"` has no effect.
- Gradient accuracy degrades near variance-parameter boundaries (θ close to 0). For
  models where variance components are expected to be near zero, `optimizer="bobyqa"`
  may be more reliable regardless.

## See also

- {doc}`fit` — high-level `interlace.fit()` API
- {doc}`../performance` — general performance guidance
