# Kenward-Roger degrees of freedom

Kenward-Roger (KR) small-sample correction for fixed-effect inference in
linear mixed models. Provides bias-adjusted covariance and denominator
degrees of freedom that are more accurate than Satterthwaite for small or
unbalanced designs.

Matches R's `lmerTest::summary(ddf = "Kenward-Roger")` output.

## Usage

Pass `df_method="kenward-roger"` to `interlace.fit()`:

```python
import interlace

result = interlace.fit(
    formula="score ~ treatment + age",
    data=df,
    groups=["subject", "site"],
    df_method="kenward-roger",
)

# KR-adjusted denominator DFs
print(result.fe_df)

# p-values use t-distribution with KR DFs
print(result.fe_pvalues)
```

## How it works

The KR correction operates in the **un-profiled variance-component
parameterisation** (sigma^2_1, ..., sigma^2_k, sigma^2_resid), unlike
Satterthwaite which uses the profiled theta parameterisation. This
includes sigma^2_resid as a free parameter, giving more accurate DFs for
coefficients whose uncertainty depends on the residual variance.

Two outputs:

1. **Bias-adjusted FE covariance** (C_adj) — for moderate-to-large samples
   the adjustment is negligible (< 0.01% of C)
2. **Denominator DFs** per coefficient — computed via Satterthwaite's formula
   in the un-profiled parameterisation: nu = 2 * C_jj^2 / (g' W g)

## When to use KR vs Satterthwaite

| | Satterthwaite | Kenward-Roger |
|---|---|---|
| Speed | Faster (uses profiled theta) | Slower (numerical Hessian in vc space) |
| Accuracy (large n) | Equivalent | Equivalent |
| Accuracy (small n) | Good | Better for unbalanced designs |
| Random slopes | Supported | Intercept-only specs only |
| Default | Yes (`df_method="satterthwaite"`) | No |

**Use KR** when you have small sample sizes (< 50 groups), unbalanced
designs, or when reviewers require KR DFs. **Use Satterthwaite** (the
default) for larger datasets or models with random slopes.

## Limitations

- Currently only supports random-intercept specs (`n_terms=1`). Models with
  random slopes will raise `NotImplementedError`.
- Computationally more expensive than Satterthwaite due to numerical
  differentiation in the un-profiled parameterisation.

## References

- Kenward, M.G. & Roger, J.H. (1997). Small sample inference for fixed
  effects from restricted maximum likelihood. *Biometrics*, 53(3), 983-997.
- Halekoh, U. & Hojsgaard, S. (2014). A Kenward-Roger Approximation and
  Parametric Bootstrap Methods for Tests in Linear Mixed Models — The R
  Package pbkrtest. *J. Stat. Softw.* 59(9), 1-30.

## Comparison with R

| interlace | R (lmerTest / pbkrtest) |
|-----------|------------------------|
| `fit(..., df_method="kenward-roger")` | `lmer(...) %>% summary(ddf="Kenward-Roger")` |
| `result.fe_df` | `summary(fit, ddf="Kenward-Roger")$coefficients[,"df"]` |

## See also

- {doc}`fit` — the `df_method` parameter
- {doc}`result` — `CrossedLMEResult` attributes including `fe_df`
- {doc}`anova_type` — per-term F-tests also use `df_method`
