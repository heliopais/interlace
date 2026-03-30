# simulate / bootMer

Parametric simulation and bootstrap for fitted mixed models.
`simulate()` draws response vectors from the conditional model;
`bootMer()` repeats that process to build a bootstrap distribution over any
user-defined statistic.

## simulate

```{eval-rst}
.. autofunction:: interlace.simulate
```

## bootMer

```{eval-rst}
.. autofunction:: interlace.bootMer
```

## BootResult

```{eval-rst}
.. autoclass:: interlace.BootResult
   :members:
   :undoc-members:
```

## Example

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

# Draw 100 response vectors from the fitted model
sims = interlace.simulate(result, n=100, seed=42)
print(sims.shape)  # (n_obs, 100)

# Parametric bootstrap: distribution of fixed-effect coefficients
boot = interlace.bootMer(result, B=500, seed=0)
print(boot.estimates.shape)          # (500, n_params)
ci = boot.ci(method="perc", level=0.95)
print(ci)                            # (n_params, 2) lower/upper bounds
```

## See also

- {doc}`result` — `CrossedLMEResult.bootstrap_se()` for a scalar cluster bootstrap
- [Variance Inference Guide](../variance-inference.md) — when to use bootstrap CIs
