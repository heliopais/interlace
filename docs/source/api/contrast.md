# contrast / pairs

Linear contrasts and pairwise comparisons on estimated marginal means.

`contrast()` applies a set of linear contrasts to an {doc}`emmeans` result.
`pairs()` is a convenience wrapper for all pairwise differences with Tukey HSD adjustment,
matching R's `pairs()` ergonomics.

```{eval-rst}
.. autofunction:: interlace.contrast

.. autofunction:: interlace.pairs
```

## Built-in contrast methods

| `method` | Description | Rows produced |
|---|---|---|
| `'pairwise'` | All pairwise differences `i − j` (alphabetical order) | `n*(n-1)//2` |
| `'trt.vs.ctrl'` | Each non-first level minus the first (control) level | `n-1` |
| list of `np.ndarray` | Custom contrast vectors of length `n` (number of EMM rows) | one per vector |
| dict `str → np.ndarray` | Same as list, using the dict keys as contrast names | one per entry |

## p-value adjustment methods

| `adjust` | Method |
|---|---|
| `'none'` | No adjustment (raw p-values) |
| `'bonferroni'` | Bonferroni correction (`p * n`) |
| `'holm'` | Step-down Bonferroni (Holm 1979) |
| `'fdr'` | Benjamini-Hochberg false discovery rate |
| `'tukey'` | Tukey HSD via the studentized range distribution |

## Returned columns

| Column | Description |
|---|---|
| `contrast` | Name of the contrast (e.g. `"high - control"`) |
| `estimate` | Contrast estimate |
| `SE` | Standard error |
| `df` | Satterthwaite denominator degrees of freedom |
| `t.ratio` | t-statistic (`estimate / SE`) |
| `p.value` | Adjusted p-value |

## Examples

### Pairwise comparisons (no adjustment)

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])
emm = interlace.emmeans(result, specs="condition")

pw = interlace.contrast(emm, method="pairwise")
print(pw)
#         contrast  estimate    SE    df  t.ratio  p.value
# 0  high - control    46.4  8.3  41.2     5.59   0.0001
```

### All pairwise with Tukey HSD via `pairs()`

```python
pw_tukey = interlace.pairs(emm)           # adjust='tukey' by default
pw_bonf  = interlace.pairs(emm, adjust="bonferroni")
```

### Treatment vs. control

```python
# First level (alphabetically) is treated as control
tvc = interlace.contrast(emm, method="trt.vs.ctrl", adjust="holm")
```

### Custom contrast vectors

```python
import numpy as np

# 3-level factor: [control, low, high]
# Custom: average of treatments vs. control
custom = interlace.contrast(emm, method={
    "avg(low,high) - control": np.array([-1.0, 0.5, 0.5]),
})
```

## See also

- {doc}`emmeans` — computing the EMMs that `contrast` operates on
- {doc}`anova_table` — omnibus F-tests for fixed effects
