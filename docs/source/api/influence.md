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
