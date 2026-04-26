# coxme

Fit a Cox proportional hazards model with Gaussian frailty (shared frailty
model) via penalised partial likelihood — targeting parity with R's
[`coxme::coxme()`](https://cran.r-project.org/package=coxme).

The model is:

    h(t | x_i, b_j) = h_0(t) exp(x_i' beta + z_i' b_j)

where b_j ~ N(0, sigma^2 I) for shared frailty. Estimation uses Laplace-
approximated integrated partial likelihood with an inner Newton-Raphson loop
and outer Brent / L-BFGS-B optimisation over variance parameters.

```{eval-rst}
.. autofunction:: interlace.coxme
```

## Key parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `formula` | `str` | Survival formula: `"Surv(time, event) ~ x1 + x2"` |
| `data` | `DataFrame` | Input data (pandas, polars, or narwhals-compatible) |
| `groups` | `str \| list[str]` | Column name(s) for shared frailty grouping |
| `random` | `list[str]` | lme4-style random-effect specs (takes precedence over `groups`) |
| `optimizer` | `str` | `"lbfgsb"` (default) |
| `theta0` | `ndarray` | Initial variance parameters; defaults to ones |

## Examples

### Shared frailty model

```python
import interlace

result = interlace.coxme(
    formula="Surv(time, status) ~ age + treatment",
    data=df,
    groups="centre",
)

# Fixed effects (log hazard ratios)
print(result.fe_params)

# Standard errors and p-values
print(result.fe_bse)
print(result.fe_pvalues)

# Frailty variance
print(result.variance_components)

# Concordance (C-statistic)
print(result.concordance)
```

### Crossed frailty

```python
result = interlace.coxme(
    formula="Surv(time, status) ~ treatment",
    data=df,
    groups=["centre", "surgeon"],
)
```

### Prediction

```python
# Linear predictor (x'beta + z'b)
lp = result.predict()                           # in-sample
lp_new = result.predict(newdata=df_new)          # new data

# Hazard ratios: exp(linear predictor)
hr = result.predict(type="risk")

# Fixed effects only (no frailty)
lp_marginal = result.predict(newdata=df_new, include_re=False)
```

### Summary

```python
print(result.summary())
```

## Result object

`coxme()` returns a `CoxmeResult` with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `fe_params` | `pd.Series` | Log hazard ratio estimates |
| `fe_bse` | `pd.Series` | Standard errors |
| `fe_pvalues` | `pd.Series` | Two-sided Wald p-values (z-test) |
| `fe_conf_int` | `pd.DataFrame` | 95% confidence intervals |
| `random_effects` | `dict` | BLUPs per grouping factor |
| `variance_components` | `dict` | Frailty variance per grouping factor |
| `theta` | `ndarray` | Raw variance parameters |
| `converged` | `bool` | Whether the optimizer converged |
| `nobs` | `int` | Number of observations |
| `n_events` | `int` | Number of events (non-censored) |
| `ngroups` | `dict` | Number of levels per grouping factor |
| `llf` | `float` | Integrated partial log-likelihood |
| `aic` | `float` | Akaike information criterion |
| `bic` | `float` | Bayesian information criterion |
| `concordance` | `float` | Harrell's C-statistic |
| `baseline_hazard` | `pd.DataFrame` | Breslow baseline cumulative hazard |

### Methods

| Method | Description |
|--------|-------------|
| `predict(newdata, include_re, type)` | Linear predictor or hazard ratio |
| `summary()` | Human-readable summary |

## Comparison with R

| interlace | R (coxme) |
|-----------|-----------|
| `interlace.coxme("Surv(t,e) ~ x", data, groups="g")` | `coxme(Surv(t,e) ~ x + (1\|g), data)` |
| `result.fe_params` | `fixef(fit)` |
| `result.random_effects["g"]` | `ranef(fit)$g` |
| `result.variance_components` | `VarCorr(fit)` |
| `result.concordance` | `concordance(fit)$concordance` |
| `result.predict(type="risk")` | `predict(fit, type="risk")` |

## See also

- {doc}`fit` — linear mixed models
- {doc}`glmer` — generalised linear mixed models
- {doc}`clmm` — cumulative link mixed models (ordinal)
