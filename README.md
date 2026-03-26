# interlace

<p align="center">
  <img src="docs/source/_static/interlace_logo.png" alt="interlace" width="220">
</p>

**[Documentation](https://heliopais.github.io/interlace/)**

Pure-Python profiled REML estimation for linear mixed models with **crossed random intercepts**, validated to match R's `lme4::lmer()`.

Designed as a drop-in replacement for `statsmodels.MixedLM` in diagnostics pipelines that require crossed grouping factors (e.g. `(1|worker) + (1|company)`), which statsmodels does not support.

**Scope:** interlace fits models with random intercepts only — it does not support random slopes, generalised outcomes (GLMM), or nested random effects with `/` syntax. For those cases, use R's `lme4` directly or a Python GLMM library.

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

See the **[full API reference](https://heliopais.github.io/interlace/)** for prediction, residuals, leverage, influence diagnostics, augmentation, and plotting.

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

## Contributing

Bug reports, documentation fixes, and new features are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started. To open an issue or ask a question, use the [GitHub issue tracker](https://github.com/heliopais/interlace/issues).

## Attribution

- **[lme4](https://github.com/lme4/lme4)** — the reference implementation for mixed-effects models in R; interlace targets parity with `lme4::lmer()` and uses its output as the validation benchmark.
- **[HLMdiag](https://github.com/aloy/HLMdiag)** — the R package whose diagnostics API (`hlm_resid`, `hlm_influence`, `hlm_augment`, `dotplot_diag`) interlace replicates in Python.
