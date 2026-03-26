# For R / lme4 users

If you use `lme4::lmer()` in R, `interlace` is the closest Python equivalent for models
with crossed random intercepts. This page maps lme4 concepts and syntax to `interlace`,
notes where the two differ, and points to the reference literature they share.

## Formula syntax

lme4 encodes random effects inside the model formula using `(term | group)` notation.
`interlace` separates fixed-effect formula from grouping factors, following the
`statsmodels.MixedLM` convention:

| lme4 (R) | interlace (Python) | Notes |
|---|---|---|
| `y ~ x + (1\|g)` | `formula="y ~ x", groups="g"` | Single random intercept |
| `y ~ x + (1\|g1) + (1\|g2)` | `formula="y ~ x", groups=["g1", "g2"]` | **Crossed** random intercepts |
| `y ~ x + (1\|g1/g2)` | — | Nested designs: not yet supported |
| `y ~ x + (x\|g)` | — | Random slopes: not yet supported |

Only **crossed random intercepts** are implemented. Random slopes, correlated random
effects, and nested shorthand (`/`) are outside the current scope.

## Side-by-side example

The following example fits the same model in both languages, using a synthetic dataset
of reading times with subject and item as crossed grouping factors — the canonical
setup in psycholinguistics and item response theory.

**R (lme4)**

```r
library(lme4)

fm <- lmer(
  rt ~ condition + (1 | subject) + (1 | item),
  data   = df,
  REML   = TRUE
)

summary(fm)
fixef(fm)           # fixed-effect coefficients
ranef(fm)           # conditional modes (BLUPs)
VarCorr(fm)         # variance components
```

**Python (interlace)**

```python
import interlace

result = interlace.fit(
    "rt ~ condition",
    data   = df,
    groups = ["subject", "item"],
    method = "REML",          # default
)

print(result.fe_params)           # fixed-effect coefficients
print(result.random_effects)      # BLUPs, keyed by grouping factor
print(result.variance_components) # sigma^2 per grouping factor + residual
```

### Comparing output

`result.fe_params` is a `pandas.Series` with the same ordering as `fixef(fm)`.
`result.random_effects` is a dict of `{group_col: pandas.Series}`, equivalent to
`ranef(fm)`. Variance components match `as.data.frame(VarCorr(fm))$vcov` (the variance,
not the standard deviation).

## Estimation

Both lme4 and interlace use **profiled REML** by default, with the same
Lambda-theta parameterisation described in Bates et al. (2015). The sparse Cholesky
factor is the core computational primitive in both implementations — lme4 uses the
Eigen C++ library; interlace uses `scipy.sparse.linalg`.

Fixed-effect estimates agree to within 1e-4 (absolute) and variance components to
within 5% (relative) on standard benchmarks.

## Diagnostics

lme4 delegates post-fit diagnostics to companion packages. `interlace` bundles an
equivalent suite directly, inspired by the R package
[HLMdiag](https://cran.r-project.org/package=HLMdiag) (Loy & Hofmann, 2014):

| HLMdiag / lme4 (R) | interlace (Python) |
|---|---|
| `hlm_resid(fm)` | `interlace.hlm_resid(result)` |
| `leverage(fm)` | `interlace.leverage(result)` |
| `hlm_influence(fm)` | `interlace.hlm_influence(result)` |
| `cooks.distance(fm)` | `interlace.cooks_distance(result)` |
| `mdffits(fm)` | `interlace.mdffits(result)` |
| `dotplot_diag(...)` | `interlace.dotplot_diag(result, ...)` |

See [Examples](examples.md) for full diagnostic workflows.

## References

**Primary citation for lme4:**

Douglas Bates, Martin Mächler, Ben Bolker, Steve Walker (2015).
*Fitting Linear Mixed-Effects Models Using lme4.*
Journal of Statistical Software, 67(1), 1–48.
[doi:10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)

**Crossed random effects in item-response models:**

Harold Doran, Douglas Bates, Paul Bliese, Maritza Dowling (2007).
*Estimating the Multilevel Rasch Model: With the lme4 Package.*
Journal of Statistical Software, 20(2), 1–18.
[doi:10.18637/jss.v020.i02](https://doi.org/10.18637/jss.v020.i02)

**HLMdiag diagnostics:**

Adam Loy, Heike Hofmann (2014).
*HLMdiag: A Suite of Diagnostics for Hierarchical Linear Models in R.*
Journal of Statistical Software, 56(5), 1–28.
[doi:10.18637/jss.v056.i05](https://doi.org/10.18637/jss.v056.i05)
