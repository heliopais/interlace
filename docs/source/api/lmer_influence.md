# lmer_influence_measures

Combined influence measures matching R's `gpgap::compute_influence_lmer` / `HLMdiag::hlm_influence`.
Returns Cook's D, leverage, DFBETAS, and MDFFITS for every observation.

```{eval-rst}
.. autofunction:: interlace.lmer_influence_measures
```

## Returned keys

| Key | Description |
|---|---|
| `cooks` | Cook's distance per observation (case-deletion or GLS-LOO, see below) |
| `hat` | Leverage used for threshold flagging (`hat_overall` for single-RE, `hat_fixef` for crossed multi-RE) |
| `hat_overall` | Full leverage H₁ + H₂ |
| `hat_fixef` | Fixed-effects-only leverage H₁ |
| `dfbetas` | DFBETAS matrix, shape `(n, p)` — one column per fixed-effect coefficient |
| `dffits` | MDFFITS per observation |
| `residuals` | Conditional residuals |
| `sigma` | Residual standard deviation √(scale) |

## `analytical=True`: GLS-LOO Woodbury fast path

By default (`analytical=False`) Cook's D and MDFFITS are computed by O(n) REML refits
(case-deletion), which is exact but slow for large datasets.

With `analytical=True`, Cook's D is computed via the **GLS-LOO Woodbury identity**,
fixing variance components at the full-model estimates:

```
β̂₍₋ᵢ₎ = β̂ − V_β · tᵢ · εᵢ / (pᵢᵢ · (1 − h̃ᵢ))
D_i    = εᵢ² · h̃ᵢ / (p · pᵢᵢ · (1 − h̃ᵢ)²)
```

This is **thousands of times faster** than O(n) refits and matches `n_influential`
counts exactly on large datasets (n ≥ 200):

| Dataset | `analytical=False` | `analytical=True` | Speedup |
|---|---|---|---|
| crossed, n=2000 | 12.65 s | 0.003 s | ~3700× |
| large scale, n=1000 | 7.48 s | 0.000 s | ~15000× |

**Requirements**: the model must be a `CrossedLMEResult` (not statsmodels path), and must
have been fitted with the default L-BFGS-B optimizer so that `_A11` and `_W` are populated
on the result object.

## Examples

### Case-deletion (exact, default)

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])
measures = interlace.lmer_influence_measures(result)

# Flag observations with Cook's D > 4/n
n = result.nobs
flagged = measures["cooks"] > 4 / n
print(f"{flagged.sum()} influential observations")
```

### GLS-LOO Woodbury (fast, recommended for n > 200)

```python
measures = interlace.lmer_influence_measures(result, analytical=True)
```

### Parallel case-deletion refits

```python
# -1 = use all available CPUs
measures = interlace.lmer_influence_measures(result, n_jobs=-1, show_progress=True)
```

### Accessing DFBETAS

```python
import pandas as pd

dfbetas = pd.DataFrame(
    measures["dfbetas"],
    columns=result.fe_params.index,
)
print(dfbetas.abs().max())   # largest DFBETAS per coefficient
```

## See also

- {doc}`influence` — `hlm_influence()` for the lower-level case-deletion frame
- {doc}`leverage` — hat-matrix diagnostics
- {doc}`augment` — append Cook's D and leverage to the original DataFrame
