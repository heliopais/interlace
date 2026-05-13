# glmer

Fit a generalised linear mixed model (GLMM) via Laplace approximation or
adaptive Gauss-Hermite quadrature (AGQ) with penalised iteratively reweighted
least squares (PIRLS) — targeting parity with R's
[`lme4::glmer()`](https://lme4.r-lib.org/reference/glmer.html).

```{eval-rst}
.. autofunction:: interlace.glmer
```

## Key parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `formula` | `str` | Fixed-effects formula in Wilkinson notation (e.g. `"y ~ x + z"`) |
| `data` | `DataFrame` | Input data (pandas, polars, or any narwhals-compatible frame) |
| `family` | `str \| GLMMFamily` | Response distribution: `"binomial"`, `"poisson"`, `"gaussian"`, or `"negativebinomial"` |
| `groups` | `str \| list[str]` | Column name(s) for random intercepts (shorthand) |
| `random` | `list[str]` | lme4-style random-effect specs, e.g. `["(1 \| herd)"]` |
| `weights` | `ndarray` | Prior weights; for binomial proportion responses, pass trial counts |
| `optimizer` | `"lbfgsb"` or `"bobyqa"` | Optimizer. `"bobyqa"` is gradient-free and more robust |
| `theta0` | `ndarray` | Initial variance parameters; defaults to ones |
| `nAGQ` | `int` | Number of adaptive Gauss-Hermite quadrature points (default `1` = Laplace) |
| `offset` | `ndarray` | Known term added to the linear predictor (e.g. `np.log(exposure)` for Poisson rates) |
| `dispformula` | `str` | Formula for the dispersion sub-model with log link (e.g. `"~1"`, `"~ z"`) |

## Supported families

### String shorthand

| Family | Link function | Use case |
|--------|--------------|----------|
| `"binomial"` | logit | Binary outcomes, proportions (pass trial counts via `weights`) |
| `"poisson"` | log | Count data (events, errors, frequencies) |
| `"gaussian"` | identity | Continuous outcomes (equivalent to `fit()` with identity link) |
| `"negativebinomial"` | log | Overdispersed count data (variance = mu + mu^2/theta) |

### Extended families (pass instances)

| Family class | Link | Use case |
|-------------|------|----------|
| `NegativeBinomial1Family(alpha=)` | log | Overdispersed counts (linear variance) |
| `BetaFamily(phi=)` | logit | Continuous proportions in (0, 1) |
| `GammaFamily(link=)` | log / inverse | Positive continuous data |
| `ZeroInflatedPoissonFamily(pi=)` | log | Counts with excess zeros |
| `ZeroInflatedNB2Family(pi=, theta=)` | log | Overdispersed counts with excess zeros |
| `HurdlePoissonFamily()` | log | Counts with structural zeros |
| `ZeroOneInflatedBetaFamily(pi0=, pi1=, phi=)` | logit | Proportions with exact 0s and 1s |

See {doc}`glmm_families` for constructor parameters, examples, and the
`GLMMFamily` protocol for building custom families.

## Examples

### Binomial GLMM (proportion response)

The classic CBPP dataset: disease incidence across cattle herds.

```python
import interlace
import pandas as pd

df = pd.read_csv("cbpp.csv")

result = interlace.glmer(
    formula="incidence / size ~ period",
    data=df,
    family="binomial",
    groups="herd",
    weights=df["size"].values,
)

print(result.fe_params)       # fixed-effect log-odds
print(result.variance_components)  # between-herd variance
print(result.aic, result.bic)
```

### Poisson GLMM (count response)

```python
result = interlace.glmer(
    formula="count ~ x1 + x2",
    data=df,
    family="poisson",
    groups=["site", "year"],
)

print(result.fe_params)       # fixed-effect log-rates
print(result.random_effects["site"])
```

### Negative Binomial GLMM (overdispersed counts)

For count data with overdispersion (variance > mean), use the negative binomial
family. Pass a `NegativeBinomial2Family` instance with the shape parameter
`theta`:

```python
from interlace import NegativeBinomial2Family

result = interlace.glmer(
    formula="count ~ x1 + x2",
    data=df,
    family=NegativeBinomial2Family(theta=2.0),
    groups="site",
)

print(result.fe_params)       # fixed-effect log-rates
print(result.variance_components)
```

You can also use the string shorthand `"negativebinomial"` which defaults to
`theta=1.0`:

```python
result = interlace.glmer(
    formula="count ~ x1 + x2",
    data=df,
    family="negativebinomial",
    groups="site",
)
```

### Using `random=` for lme4-style syntax

```python
result = interlace.glmer(
    formula="incidence / size ~ period",
    data=df,
    family="binomial",
    random=["(1 | herd)"],
    weights=df["size"].values,
)
```

### Adaptive Gauss-Hermite quadrature (AGQ)

For models with a single scalar random intercept, set `nAGQ > 1` for a more
accurate marginal-likelihood approximation than Laplace. This mirrors
`lme4::glmer(nAGQ = ...)`.

```python
result = interlace.glmer(
    formula="incidence / size ~ period",
    data=df,
    family="binomial",
    groups="herd",
    weights=df["size"].values,
    nAGQ=25,  # 25 quadrature points
)
```

`nAGQ=1` (the default) uses the Laplace approximation. Higher values are more
accurate but slower. `nAGQ > 1` requires exactly one grouping factor with a
random intercept only (no random slopes). See the
{doc}`/glmm-quickstart` AGQ section for guidance on when Laplace is sufficient
vs when AGQ improves estimation.

### Dispersion modelling (`dispformula`)

By default, the dispersion (scale) parameter is fixed at 1.0 for binomial and
Poisson families.  For Gaussian GLMMs, you can estimate the residual variance
or model heteroscedasticity using `dispformula`.

**Scalar dispersion** (`~1`): estimates a single residual variance
$\phi = \exp(\delta_0)$.

```python
result = interlace.glmer(
    formula="y ~ x1 + x2",
    data=df,
    family="gaussian",
    groups="site",
    dispformula="~1",
)

# Estimated residual variance
import numpy as np
phi_hat = np.exp(result.disp_params["Intercept"])
print(f"Residual variance: {phi_hat:.3f}")
```

**Covariate-dependent dispersion** (`~ z`): models observation-level variance as
$\phi_i = \exp(\delta_0 + \delta_1 z_i)$ (heteroscedastic model).

```python
result = interlace.glmer(
    formula="y ~ x1 + x2",
    data=df,
    family="gaussian",
    groups="site",
    dispformula="~ z",
)

print(result.disp_params)   # log-scale dispersion coefficients
print(result.dispersion)    # per-observation phi values
```

The dispersion sub-model coefficients are on the **log scale** (log link ensures
$\phi > 0$).  They are stored in `result.disp_params` as a `pd.Series`.  The
per-observation dispersion values are in `result.dispersion`.

For Gaussian families, the reported coefficients match `glmmTMB`'s convention:
`disp_params` is on the **log-σ scale** (so $\sigma_i = \exp(W_i \delta)$).
For families like NB2 where the dispersion has a different natural scale
(e.g., $\log \theta$), the coefficients are on that scale instead.

`dispformula` cannot be combined with `nAGQ > 1`.

#### Random effects on the dispersion side

`dispformula` accepts lme4-style random effect terms, mirroring
`glmmTMB(..., dispformula = ~ (1|g))`:

```python
result = interlace.fit(
    formula="y ~ x",
    data=df,
    groups="g_mean",
    dispformula="~ (1|g_disp)",          # random intercepts on log σ
)
# Nested shorthand also supported:
result = interlace.fit(
    formula="y ~ x", data=df, groups="g_mean",
    dispformula="~ (1|g1/g2)",
)

print(result.disp_variance_components)   # τ²_g_disp etc.
print(result.disp_random_effects)        # BLUPs on log-σ scale
```

When the dispformula contains random effects, `interlace.fit` switches from
the joint-Laplace path (FE-only dispformula) to a **Block-Coordinate Ascent
(BCA)** algorithm: the mean block is a weighted LMM with weights
$1/\sigma_i^2$, and the dispersion block is a Gamma GLMM with log link on
squared residuals. BCA converges to the joint mode but uses block-wise
Laplace approximations for each side's variance components, so its
dispersion-side variance components are biased vs the true joint MLE
(typically within 10–20% in well-conditioned cases). Mean-side fixed effects
match `glmmTMB` tightly. A future release will add a fully joint Laplace
path for tighter parity (see issue `interlace-c803`).

### Prediction

```python
# Response-scale predictions (probabilities for binomial, counts for Poisson)
preds = result.predict(newdata=df_new)

# Linear predictor (log-odds for binomial, log-rate for Poisson)
eta = result.predict(newdata=df_new, type="link")

# Population-level only (no random effects)
preds_marginal = result.predict(newdata=df_new, include_re=False)
```

## Result object

`glmer()` returns a `GLMMResult` with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `fe_params` | `pd.Series` | Fixed-effect coefficients |
| `fe_bse` | `pd.Series` | Fixed-effect standard errors |
| `random_effects` | `dict` | BLUPs per grouping factor |
| `variance_components` | `dict` | Variance estimates per grouping factor |
| `theta` | `ndarray` | Variance parameters (Cholesky factors) |
| `converged` | `bool` | Whether both PIRLS and outer optimizer converged |
| `nobs` | `int` | Number of observations |
| `llf` | `float` | Log-likelihood (Laplace or AGQ, depending on `nAGQ`) |
| `aic` | `float` | Akaike information criterion |
| `bic` | `float` | Bayesian information criterion |
| `family` | `GLMMFamily` | Family object used for fitting |
| `ngroups` | `dict` | Number of levels per grouping factor |
| `scale` | `float` | Dispersion parameter (1.0 for binomial and Poisson) |
| `fittedvalues` | `ndarray` | Fitted values on the response scale |
| `disp_params` | `pd.Series \| None` | Dispersion formula coefficients (log scale); `None` when no `dispformula` |
| `dispersion` | `ndarray \| None` | Per-observation dispersion values; `None` when no `dispformula` |

## Custom families

Any object implementing the `GLMMFamily` protocol can be passed as the
`family` parameter:

```python
from interlace import GLMMFamily

class MyFamily:
    name: str = "custom"

    def link(self, mu):       ...  # eta = g(mu)
    def linkinv(self, eta):   ...  # mu = g^{-1}(eta)
    def mu_eta(self, eta):    ...  # d(mu)/d(eta)
    def variance(self, mu):   ...  # Var(Y | mu)
    def dev_resids(self, y, mu, wt): ...  # deviance residuals
```

## See also

- {doc}`fit` — linear mixed models (`lmer`-equivalent)
- {doc}`result` — attributes on `CrossedLMEResult` (LMM result)
- {doc}`predict` — generating predictions from a fitted model
