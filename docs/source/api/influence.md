# Influence diagnostics

Exact-deletion influence analysis following Demidenko & Stukel (2005).
All functions accept both `CrossedLMEResult` and `statsmodels.MixedLMResults`.

## hlm_influence

```{eval-rst}
.. autofunction:: interlace.hlm_influence
```

### Returned columns

| Column | Description |
|---|---|
| `cooksd` | Cook's distance |
| `mdffits` | MDFFITS (Measures of Difference in Fixed Effects) |
| `covtrace` | COVTRACE (trace of V⁻¹Vᵢ) − p |
| `covratio` | COVRATIO (det(Vᵢ) / det(V)) |
| `rvc.<name>` | Relative variance change per variance component |

## Convenience wrappers

```{eval-rst}
.. autofunction:: interlace.cooks_distance

.. autofunction:: interlace.mdffits

.. autofunction:: interlace.n_influential

.. autofunction:: interlace.tau_gap
```

## Example

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

# Full influence frame
infl = interlace.hlm_influence(result)
print(infl.columns.tolist())
# ['cooksd', 'mdffits', 'covtrace', 'covratio', 'rvc.subject', 'rvc.item']

# Flag influential observations (Cook's D threshold: 4/n)
n = result.nobs
flagged = infl[infl["cooksd"] > 4 / n]
print(f"{len(flagged)} influential observations by Cook's D")

# Convenience: count influential at a given threshold
print(interlace.n_influential(result, threshold=4 / n))

# Better parity with R/HLMdiag (requires bobyqa extra)
infl_bobyqa = interlace.hlm_influence(result, optimizer="bobyqa")
```

## See also

- {doc}`leverage` — hat-matrix diagnostics (influence on fit, not estimates)
- {doc}`augment` — append Cook's D + leverage to the original DataFrame in one call
- {doc}`residuals` — residual diagnostics
