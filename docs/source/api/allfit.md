# allFit

Refit a model with all available optimizers and compare convergence.
Useful for diagnosing whether results are optimizer-sensitive and for
identifying which optimizer gives the best (lowest) log-likelihood.

## allFit

```{eval-rst}
.. autofunction:: interlace.allFit
```

## AllFitResult

```{eval-rst}
.. autoclass:: interlace.AllFitResult
   :members:
   :undoc-members:
```

## Example

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

fits = interlace.allFit(result)
print(fits.summary())
# optimizer   llf        converged  possible_issue
# lbfgsb      -1234.56   True       False
# bobyqa      -1234.56   True       False

# Access individual fits
bobyqa_fit = fits.results["bobyqa"]
print(bobyqa_fit.fe_params)
```

## See also

- {doc}`fit` — primary fitting function
- [Convergence Guide](../variance-inference.md) — diagnosing singular fits
