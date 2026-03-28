# For R / lme4 users

If you use `lme4::lmer()` in R, `interlace` is the closest Python equivalent for models
with crossed random intercepts. This page maps lme4 concepts and syntax to `interlace`,
notes where the two differ, and points to the reference literature they share.

## Formula syntax

lme4 encodes random effects inside the model formula using `(term | group)` notation.
`interlace` separates fixed-effect formula from grouping factors for simple random
intercepts, and accepts lme4-style `(term | group)` notation via the `random` parameter
for random slopes:

| lme4 (R) | interlace (Python) | Notes |
|---|---|---|
| `y ~ x + (1\|g)` | `formula="y ~ x", groups="g"` | Single random intercept |
| `y ~ x + (1\|g1) + (1\|g2)` | `formula="y ~ x", groups=["g1", "g2"]` | **Crossed** random intercepts |
| `y ~ x + (x\|g)` | `formula="y ~ x", random=["(1 + x \| g)"]` | Correlated random intercept + slope *(v0.2.1+)* |
| `y ~ x + (x\|\|g)` | `formula="y ~ x", random=["(1 + x \|\| g)"]` | Independent (uncorrelated) parameterisation *(v0.2.1+)* |
| `y ~ x + (1\|g1/g2)` | — | Nested designs: not yet supported |

**`groups` vs `random`:** use `groups=` for random intercepts only (shorter syntax); use
`random=` when you need random slopes or want to mix intercept-only and slope terms
across grouping factors. See [Quickstart](quickstart.md) for a side-by-side example.

## Side-by-side examples

### Crossed random intercepts

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

### Random slopes (v0.2.1+)

When subjects differ not just in their baseline RT but also in how much condition
affects them, add a by-subject random slope:

**R (lme4)**

```r
fm_slopes <- lmer(
  rt ~ condition + (1 + condition | subject) + (1 | item),
  data = df,
  REML = TRUE
)

ranef(fm_slopes)$subject  # DataFrame: columns (Intercept) + condition
```

**Python (interlace)**

```python
result_slopes = interlace.fit(
    "rt ~ condition",
    data   = df,
    random = [
        "(1 + condition | subject)",  # correlated intercept + slope
        "(1 | item)",                 # intercept only
    ],
)

# random_effects["subject"] is now a DataFrame, one column per term
print(result_slopes.random_effects["subject"])
#             (Intercept)  condition
# subject_01       -12.3       0.42
# subject_02         8.7      -0.31
# ...

# Full random-effect covariance matrix
print(result_slopes.varcov)
```

For the independent (uncorrelated) parameterisation — equivalent to lme4's `||` — use
`"(1 + condition || subject)"`. See the [Random Slopes Guide](random-slopes.md) for
a full walkthrough including interpretation and when to use each parameterisation.

### Comparing output

`result.fe_params` is a `pandas.Series` with the same ordering as `fixef(fm)`.
`result.random_effects` is a dict of `{group_col: pandas.Series}`, equivalent to
`ranef(fm)`. Variance components match `as.data.frame(VarCorr(fm))$vcov` (the variance,
not the standard deviation).

## Estimation

Both lme4 and interlace use **profiled REML** by default, with the same
Lambda-theta parameterisation described in [Bates et al. (2015)](https://doi.org/10.18637/jss.v067.i01). The sparse Cholesky
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

See the [Diagnostics notebook](diagnostics.ipynb) for full diagnostic workflows,
and {doc}`api/influence` for the function reference.

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
