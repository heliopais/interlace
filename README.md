# interlace

<p align="center">
  <img src="docs/source/_static/interlace_logo.png" alt="interlace" width="220">
</p>

**[Documentation](https://heliopais.github.io/interlace/)**

Pure-Python mixed-effects modelling library — linear (LMM), generalised (GLMM), ordinal (CLMM), and survival (Cox frailty) — validated against R's `lme4`, `glmer`, `ordinal`, and `coxme`.

Designed as a drop-in replacement for `statsmodels.MixedLM` in diagnostics pipelines that require crossed grouping factors (e.g. `(1|worker) + (1|company)`), which statsmodels does not support.

## Installation

```bash
pip install interlace-lme
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

## Model types

### Linear mixed models (LMM)

```python
from interlace import fit

result = fit(
    formula="score ~ hours_studied + prior_gpa",
    data=df,
    groups=["student_id", "school_id"],
    method="REML",           # or "ML" for likelihood-ratio tests
    df_method="satterthwaite", # or "kenward-roger"
)
```

Supports crossed and nested random intercepts, random slopes (`(1 + x | g)`), observation-level weights, AR(1) and compound-symmetry residual correlation structures, and Satterthwaite or Kenward-Roger denominator degrees of freedom.

### Generalised linear mixed models (GLMM)

```python
from interlace import glmer

result = glmer(
    formula="incidence / size ~ period",
    data=df,
    family="binomial",       # or "poisson", "gaussian", "negativebinomial"
    groups="herd",
    weights=df["size"].values,
)
```

Families: binomial, Poisson, Gaussian, NB1, NB2, Beta, Gamma, zero-inflated Poisson/NB2, hurdle Poisson, zero-one-inflated Beta. Supports Laplace approximation and adaptive Gauss-Hermite quadrature.

### Cumulative link mixed models (CLMM)

```python
from interlace import clmm

result = clmm(
    formula="rating ~ condition",
    data=df,
    groups="subject",
)
```

Ordinal regression with random effects, matching R's `ordinal::clmm()`.

### Cox frailty models

```python
from interlace import coxme

result = coxme(
    formula="Surv(time, event) ~ treatment + age",
    data=df,
    groups="centre",
)
```

Cox proportional hazards with Gaussian frailty, matching R's `coxme::coxme()`.

## Diagnostics

```python
from interlace import hlm_resid, leverage, hlm_influence, hlm_augment

resid_df = hlm_resid(result, type="conditional")  # or "marginal"
lev = leverage(result)                              # hat-matrix diagonal
infl = hlm_influence(result, level=1)               # Cook's D, MDFFITS, etc.
aug = hlm_augment(result)                           # data + residuals + influence

# Plotting (all return plotnine.ggplot objects)
from interlace import plot_resid, plot_influence, dotplot_diag

plot_resid(resid_df, type="resid_vs_fitted")
plot_influence(infl, measure="cooks_d")
dotplot_diag(infl, variable="cooks_d", cutoff="internal")
```

## statsmodels compatibility

`CrossedLMEResult` exposes the same interface as `statsmodels.MixedLMResults` so it can be used as a drop-in in downstream code that accesses `fe_params`, `resid`, `scale`, `fittedvalues`, `random_effects`, `predict()`, and `model.exog / model.groups / model.data.frame`.

`hlm_resid`, `hlm_influence`, and `hlm_augment` all accept either a `CrossedLMEResult` or a statsmodels `MixedLMResults` object.

## Parity with R

Results are validated against R reference implementations to the following tolerances:

| Model type | R reference | Fixed effects | Variance components |
|---|---|---|---|
| LMM | `lme4::lmer()` | abs diff < 1e-4 | rel diff < 5% |
| GLMM | `lme4::glmer()` | abs diff < 1e-3 | rel diff < 5% |
| CLMM | `ordinal::clmm()` | abs diff < 1e-3 | rel diff < 5% |
| Cox frailty | `coxme::coxme()` | abs diff < 1e-3 | rel diff < 10% |
| AR(1) / CS | `nlme::lme()` | abs diff < 1e-3 | rel diff < 5% |

## Contributing

Bug reports, documentation fixes, and new features are welcome — see [CONTRIBUTING.md](docs/source/contributing.md) for how to get started. To open an issue or ask a question, use the [GitHub issue tracker](https://github.com/heliopais/interlace/issues).

## Attribution

- **[lme4](https://github.com/lme4/lme4)** — the reference implementation for mixed-effects models in R; interlace targets parity with `lme4::lmer()` and `lme4::glmer()`.
- **[ordinal](https://github.com/runehaubo/ordinal)** — R package for cumulative link models; interlace's `clmm()` targets parity with `ordinal::clmm()`.
- **[coxme](https://cran.r-project.org/package=coxme)** — R package for Cox models with random effects; interlace's `coxme()` targets parity with `coxme::coxme()`.
- **[nlme](https://cran.r-project.org/package=nlme)** — R package for linear mixed models with correlation structures; AR(1) and CS validation benchmarks.
- **[HLMdiag](https://github.com/aloy/HLMdiag)** — the R package whose diagnostics API (`hlm_resid`, `hlm_influence`, `hlm_augment`, `dotplot_diag`) interlace replicates in Python.
