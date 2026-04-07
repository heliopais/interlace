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
| `family` | `str \| GLMMFamily` | Response distribution: `"binomial"`, `"poisson"`, or `"gaussian"` |
| `groups` | `str \| list[str]` | Column name(s) for random intercepts (shorthand) |
| `random` | `list[str]` | lme4-style random-effect specs, e.g. `["(1 \| herd)"]` |
| `weights` | `ndarray` | Prior weights; for binomial proportion responses, pass trial counts |
| `optimizer` | `"lbfgsb"` or `"bobyqa"` | Optimizer. `"bobyqa"` is gradient-free and more robust |
| `theta0` | `ndarray` | Initial variance parameters; defaults to ones |
| `nAGQ` | `int` | Number of adaptive Gauss-Hermite quadrature points (default `1` = Laplace) |

## Supported families

| Family | Link function | Use case |
|--------|--------------|----------|
| `"binomial"` | logit | Binary outcomes, proportions (pass trial counts via `weights`) |
| `"poisson"` | log | Count data (events, errors, frequencies) |
| `"gaussian"` | identity | Continuous outcomes (equivalent to `fit()` with identity link) |

Custom families can be passed as any object implementing the `GLMMFamily`
protocol (see below).

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
random intercept only (no random slopes).

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
