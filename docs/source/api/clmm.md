# clmm

Fit a cumulative link mixed model (CLMM) with random effects via Laplace
approximation — targeting parity with R's
[`ordinal::clmm()`](https://cran.r-project.org/package=ordinal).

The model is:

    link(P(Y <= k | b)) = alpha_k - x'beta - z'b,   k = 1,...,K-1

where alpha_1 < ... < alpha_{K-1} are threshold parameters, beta are
fixed-effect coefficients (no intercept — absorbed into thresholds), and
b ~ N(0, sigma^2 Lambda Lambda') are random effects.

```{eval-rst}
.. autofunction:: interlace.clmm
```

## Key parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `formula` | `str` | Fixed-effects formula, e.g. `"rating ~ temp + contact"`. Intercept is suppressed (thresholds absorb it). |
| `data` | `DataFrame` | Input data (pandas, polars, or narwhals-compatible) |
| `groups` | `str \| list[str]` | Column name(s) for random intercepts |
| `random` | `list[str]` | lme4-style random-effect specs (takes precedence over `groups`) |
| `link` | `str` | Link function: `"logit"` (default), `"probit"`, or `"cloglog"` |
| `optimizer` | `str` | `"lbfgsb"` (default) or `"bobyqa"` |
| `theta0` | `ndarray` | Initial variance parameters; defaults to ones |

## Supported link functions

| Link | CDF | Use case |
|------|-----|----------|
| `"logit"` | Logistic | Proportional odds (default, most common) |
| `"probit"` | Normal | When normality of latent variable is assumed |
| `"cloglog"` | Complementary log-log | Asymmetric, for grouped survival data |

## Examples

### Basic ordinal regression with random intercepts

```python
import interlace

result = interlace.clmm(
    formula="rating ~ temp + contact",
    data=df,
    groups="judge",
)

# Threshold estimates (cut points between ordinal levels)
print(result.thresholds)
# {'1|2': -1.34, '2|3': 1.25, '3|4': 3.47, '4|5': 5.01}

# Fixed-effect coefficients
print(result.fe_params)

# Between-judge variance
print(result.variance_components)
```

### Prediction

```python
# Category probabilities P(Y=k), shape (n, K)
probs = result.predict(newdata=df_new, type="prob")

# Cumulative probabilities P(Y<=k), shape (n, K-1)
cum = result.predict(newdata=df_new, type="cum.prob")

# Linear predictor x'beta + z'b, shape (n,)
eta = result.predict(newdata=df_new, type="linear.predictor")
```

### Confidence intervals

```python
ci = result.confint(level=0.95)
# DataFrame with lower/upper for both thresholds and fixed effects
```

## Result object

`clmm()` returns a `CLMMResult` with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `thresholds` | `dict[str, float]` | Threshold estimates, keyed by label (e.g. `"1\|2"`) |
| `threshold_bse` | `dict[str, float]` | Standard errors of thresholds |
| `fe_params` | `pd.Series` | Fixed-effect coefficients |
| `fe_bse` | `pd.Series` | Standard errors of fixed effects |
| `random_effects` | `dict` | BLUPs per grouping factor |
| `variance_components` | `dict` | Variance estimate per grouping factor |
| `theta` | `ndarray` | Raw variance parameters |
| `converged` | `bool` | Whether the optimizer converged |
| `nobs` | `int` | Number of observations |
| `llf` | `float` | Laplace-approximated log-likelihood |
| `aic` | `float` | Akaike information criterion |
| `bic` | `float` | Bayesian information criterion |
| `ngroups` | `dict` | Number of levels per grouping factor |
| `link` | `str` | Link function name |

### Methods

| Method | Description |
|--------|-------------|
| `predict(newdata, type)` | Predict probabilities, cumulative probabilities, or linear predictor |
| `confint(level)` | Wald confidence intervals for thresholds and fixed effects |
| `summary()` | Human-readable summary |

## Comparison with R

| interlace | R (ordinal) |
|-----------|-------------|
| `interlace.clmm(formula, data, groups="judge")` | `clmm(rating ~ temp + contact + (1\|judge), data)` |
| `result.thresholds` | `result$alpha` |
| `result.fe_params` | `result$beta` |
| `result.variance_components` | `VarCorr(result)` |
| `result.predict(newdata, type="prob")` | `predict(result, newdata, type="prob")` |
| `result.confint()` | `confint(result)` |

## See also

- {doc}`glmer` — generalised linear mixed models (continuous/binary/count)
- {doc}`fit` — linear mixed models
- {doc}`result` — `CrossedLMEResult` attributes
