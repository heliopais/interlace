# Anova

`car::Anova()`-style ANOVA table for a fitted linear mixed model.
Produces per-term F- or chi-square statistics with Satterthwaite denominator degrees of
freedom. Mirrors the interface of `car::Anova()` in R.

```{eval-rst}
.. autofunction:: interlace.Anova
```

## Returned columns

### `test='F'` (default)

| Column | Description |
|---|---|
| `term` | Fixed-effect term name |
| `df1` | Numerator degrees of freedom (number of columns for this term) |
| `df2` | Satterthwaite denominator degrees of freedom |
| `F` | Wald F-statistic |
| `Pr(>F)` | p-value from the F distribution |

### `test='Chisq'`

| Column | Description |
|---|---|
| `term` | Fixed-effect term name |
| `df` | Degrees of freedom (`df1`) |
| `Chisq` | Wald chi-square statistic (`F * df1`) |
| `Pr(>Chisq)` | p-value from the chi-square distribution |

## Choosing between Type II and Type III

| Type | Use when |
|---|---|
| `type='III'` (default) | Interpreting main effects in the presence of interactions; matches `lmerTest::anova()` Type III output |
| `type='II'` | Additive models (no interactions), or when you want hierarchical marginal tests; requires ML refit internally |

For **additive models** (no interactions) the two types produce identical results because the
hypothesis matrices coincide. The distinction matters only when interactions are present.

## Examples

### Basic Type III F-table

```python
import interlace

result = interlace.fit("rt ~ condition + load", data=df, groups=["subject", "item"])

tbl = interlace.Anova(result)
print(tbl)
#        term  df1       df2      F   Pr(>F)
# 0  condition    1   38.2   31.2   0.0000
# 1       load    1   41.5    8.9   0.0048
```

### Type II

```python
tbl2 = interlace.Anova(result, type="II")
```

### Chi-square statistics

```python
tbl_chisq = interlace.Anova(result, test="Chisq")
#        term  df  Chisq  Pr(>Chisq)
# 0  condition   1   31.2      0.0000
# 1       load   1    8.9      0.0028
```

### LRT comparison (two-model form)

When a list of two models is passed, `Anova()` delegates to the LRT-based
{doc}`anova` function (both models must use `method="ML"`):

```python
m0 = interlace.fit("rt ~ 1",         data=df, groups=["subject", "item"], method="ML")
m1 = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"], method="ML")

interlace.Anova([m0, m1])   # LRT table, same as interlace.anova(m0, m1)
```

## See also

- {doc}`anova` — likelihood-ratio test for two nested models
- {doc}`emmeans` — marginal means and contrasts after fitting
- {doc}`fit` — fitting the model
