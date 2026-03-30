# emmeans

Estimated marginal means (EMMs) for a fitted linear mixed model.
Mirrors R's [`emmeans`](https://cran.r-project.org/package=emmeans) package: for each level combination of the
specified factor(s), the marginal mean of the fixed-effects predictor is computed after
averaging continuous covariates at their observed means. Standard errors and degrees of
freedom use the Satterthwaite approximation.

Reference: Lenth (2016) *Least-Squares Means: The R Package lsmeans*, J. Stat. Softw. 69(1).

```{eval-rst}
.. autofunction:: interlace.emmeans
```

## Returned columns

| Column | Description |
|---|---|
| `<specs>` | One column per specs factor showing the level combination |
| `estimate` | Marginal mean (EMM) |
| `SE` | Standard error |
| `df` | Satterthwaite denominator degrees of freedom |
| `lower` | Lower bound of the confidence interval |
| `upper` | Upper bound of the confidence interval |
| `t.ratio` | t-statistic (`estimate / SE`) |
| `p.value` | Two-tailed p-value |

The confidence interval width is controlled by the `level` parameter (default 0.95).

## Examples

### Single factor

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

emm = interlace.emmeans(result, specs="condition")
print(emm)
#   condition  estimate        SE        df      lower      upper  t.ratio  p.value
# 0   control    456.3   12.4  38.2  431.2  481.4    36.8    0.000
# 1      high    502.7   13.1  39.5  476.2  529.2    38.4    0.000
```

### Two-way grid

```python
emm2 = interlace.emmeans(result, specs=["condition", "group"])
# One row per condition × group combination
```

### Custom covariate values with `at=`

```python
# Centre estimates at x = 0 instead of the training mean
emm_at = interlace.emmeans(result, specs="condition", at={"x": 0.0})
```

### Narrower confidence interval

```python
emm_90 = interlace.emmeans(result, specs="condition", level=0.90)
```

## See also

- {doc}`contrast` — pairwise or custom contrasts on an EMM result
- {doc}`fit` — fitting the model passed to `emmeans`
- {doc}`result` — `CrossedLMEResult` attributes
