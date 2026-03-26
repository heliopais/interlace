# interlace

**[Documentation](https://heliopais.github.io/interlace/)**

Pure-Python profiled REML estimation for linear mixed models with **crossed random intercepts**, validated to match R's `lme4::lmer()`.

Designed as a drop-in replacement for `statsmodels.MixedLM` in diagnostics pipelines that require crossed grouping factors (e.g. `(1|worker) + (1|company)`), which statsmodels does not support.

## Installation

```bash
pip install interlace
```

Requires Python ≥ 3.13.

## Quick start

```python
import pandas as pd
from interlace import fit

result = fit(
    formula="score ~ hours_studied + prior_gpa",
    data=df,
    groups=["student_id", "school_id"],   # crossed random intercepts
)

print(result.fe_params)          # fixed-effect coefficients
print(result.variance_components) # per-factor variance components
print(result.scale)              # residual variance σ²
```

`groups` accepts a single string (one random intercept) or a list (crossed intercepts). The first entry is the primary grouping factor.

## API

### Fitting

```python
from interlace import fit

result = fit(formula, data, groups, method="REML")
```

Returns a `CrossedLMEResult` with the following attributes:

| Attribute | Description |
|---|---|
| `fe_params` | Fixed-effect coefficients (Series) |
| `fe_bse` | Standard errors |
| `fe_pvalues` | Wald p-values |
| `fe_conf_int` | 95% confidence intervals |
| `random_effects` | Dict of BLUPs per grouping factor |
| `variance_components` | Dict of variance estimates per grouping factor |
| `scale` | Residual variance σ² |
| `fittedvalues` | Conditional fitted values (Xβ + Zû) |
| `resid` | Conditional residuals |
| `llf`, `aic`, `bic` | Log-likelihood and information criteria |

### Prediction

```python
# In-sample (uses BLUPs)
result.predict()

# New data (unseen group levels shrink to zero)
result.predict(newdata=df_new)

# Fixed effects only
result.predict(newdata=df_new, include_re=False)
```

### Residuals

```python
from interlace import hlm_resid

resid_df = hlm_resid(result, type="conditional")  # or "marginal"
# Returns DataFrame with .resid, .fitted, and original data columns
```

### Leverage

```python
from interlace import leverage

lev = leverage(result)  # array of hat-matrix diagonal values
```

### Influence diagnostics

```python
from interlace import hlm_influence, cooks_distance, mdffits, n_influential, tau_gap

infl = hlm_influence(result, level=1)   # Cook's D, MDFFITS, COVTRACE, COVRATIO, RVC per obs

# Scalar summaries
n = n_influential(result)   # count of high-influence observations
gap = tau_gap(result)        # gap statistic between influential and non-influential groups
```

### Combined augment

```python
from interlace import hlm_augment

aug = hlm_augment(result)
# DataFrame: original data + conditional residuals + influence statistics
```

### Plotting

```python
from interlace import plot_resid, plot_influence, dotplot_diag

plot_resid(resid_df, type="resid_vs_fitted")  # or "qq"
plot_influence(infl_df, measure="cooks_d")
dotplot_diag(infl_df, variable="cooks_d", cutoff="internal")
```

All plots return `plotnine.ggplot` objects.

## statsmodels compatibility

`CrossedLMEResult` exposes the same interface as `statsmodels.MixedLMResults` so it can be used as a drop-in in downstream code that accesses `fe_params`, `resid`, `scale`, `fittedvalues`, `random_effects`, `predict()`, and `model.exog / model.groups / model.data.frame`.

`hlm_resid`, `hlm_influence`, and `hlm_augment` all accept either a `CrossedLMEResult` or a statsmodels `MixedLMResults` object.

## Parity with lme4

Results are validated against R's `lme4::lmer()` to the following tolerances:

| Metric | Tolerance |
|---|---|
| Fixed effects | abs diff < 1e-4 |
| Variance components | rel diff < 5% |
| BLUP correlation | > 0.99 |
| Conditional residual correlation | > 0.999 |

## Development

```bash
make install      # create venv and install all dev deps via uv
make test         # run pytest
make lint         # ruff format + ruff check --fix
make typecheck    # mypy
make check        # lint + typecheck + test (full CI gate)
```

## Attribution

- **[lme4](https://github.com/lme4/lme4)** — the reference implementation for mixed-effects models in R; interlace targets parity with `lme4::lmer()` and uses its output as the validation benchmark.
- **[HLMdiag](https://github.com/aloy/HLMdiag)** — the R package whose diagnostics API (`hlm_resid`, `hlm_influence`, `hlm_augment`, `dotplot_diag`) interlace replicates in Python.
