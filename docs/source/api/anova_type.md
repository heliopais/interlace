# anova_type2 / anova_type3

Low-level functions for Type II and Type III ANOVA F-tables.
These are the computational back-ends for {doc}`anova_table`. Most users should
call `Anova()` directly; these functions are useful when you need finer control or
want to inspect the underlying tables without the `car::Anova()` wrapper.

```{eval-rst}
.. autofunction:: interlace.anova_type3

.. autofunction:: interlace.anova_type2
```

## Returned columns

Both functions return a DataFrame with columns:

| Column | Description |
|---|---|
| `term` | Fixed-effect term name (intercept excluded) |
| `df1` | Numerator degrees of freedom |
| `df2` | Satterthwaite denominator degrees of freedom (harmonic mean over columns) |
| `F` | Wald F-statistic |
| `Pr(>F)` | p-value from the F distribution |

## How they work

### `anova_type3`

For each non-intercept term, builds a hypothesis matrix `L` (rows = term columns,
columns = all FE columns), then computes the Wald F-statistic:

```
F = (1/df1) · (Lβ̂)ᵀ (L V_β Lᵀ)⁻¹ (Lβ̂)
```

where `V_β` is the REML fixed-effects covariance matrix.
Denominator DF is the harmonic mean of the Satterthwaite DFs for the individual
coefficients belonging to the term.

### `anova_type2`

Internally refits the model with `method="ML"` (REML FE covariances are not appropriate
for direct FE hypothesis tests), then delegates to `anova_type3` on the ML result.
For **additive models** (no interactions) the tests are identical to Type III because the
hypothesis matrices coincide; the distinction matters when interactions are present.

## Examples

```python
import interlace

result = interlace.fit("rt ~ condition + load", data=df, groups=["subject", "item"])

# Type III (no ML refit needed — uses existing REML covariance)
tbl3 = interlace.anova_type3(result)

# Type II (triggers internal ML refit)
tbl2 = interlace.anova_type2(result)

print(tbl3)
#        term  df1       df2      F   Pr(>F)
# 0  condition    1   38.2   31.2   0.0000
# 1       load    1   41.5    8.9   0.0048
```

## See also

- {doc}`anova_table` — `Anova()` wrapper with `type=` and `test=` parameters
- {doc}`fit` — `method="ML"` parameter
