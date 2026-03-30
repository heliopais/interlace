# isSingular

Detect whether a fitted model is at or near the boundary of the parameter
space — i.e., whether any variance component has collapsed to zero.
Mirrors `lme4::isSingular()`.

```{eval-rst}
.. autofunction:: interlace.isSingular
```

## Example

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

if interlace.isSingular(result):
    print("Warning: one or more variance components are at the boundary.")
    print(result.boundary_flags)  # {'subject': False, 'item': True}
```

## Notes

`fit()` automatically issues a `ConvergenceWarning` when `isSingular()` returns
`True`. You can silence the warning if you expect a boundary fit (e.g., in
simulations) and check it explicitly with this function instead.

## See also

- {doc}`result` — `CrossedLMEResult.boundary_flags` attribute
- [Variance Inference Guide](../variance-inference.md) — interpreting singular fits
