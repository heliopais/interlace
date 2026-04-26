# interlace

```{image} _static/interlace.png
:alt: interlace
:align: center
:width: 480px
```

[![PyPI](https://img.shields.io/github/v/tag/heliopais/interlace?label=version)](https://github.com/heliopais/interlace/releases)
[![CI](https://github.com/heliopais/interlace/actions/workflows/ci.yml/badge.svg)](https://github.com/heliopais/interlace/actions/workflows/ci.yml)
[![Docs](https://github.com/heliopais/interlace/actions/workflows/docs.yml/badge.svg)](https://github.com/heliopais/interlace/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](https://github.com/heliopais/interlace/blob/main/LICENSE)

**interlace** is a pure-Python mixed-effects modelling library — targeting
parity with R's [`lme4`](https://lme4.r-lib.org/),
[`ordinal`](https://github.com/runehaubo/ordinal),
[`coxme`](https://cran.r-project.org/package=coxme), and
[`nlme`](https://cran.r-project.org/package=nlme), and designed as a drop-in
replacement for
[`statsmodels.MixedLM`](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html)
in production pipelines.

It supports **linear** (LMM), **generalised** (GLMM), **cumulative link** (CLMM),
and **Cox frailty** mixed models with crossed or nested random effects.

## Why interlace?

[`statsmodels.MixedLM`](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html) is built around a single grouping factor. When a model has two
independent sources of variance — say, `subject` *and* `item` — there is no native
syntax for crossed random intercepts, and the available workarounds produce estimates
that diverge from REML. `interlace` fills that gap, implementing the same profiled REML
and sparse Cholesky machinery as R's [`lme4::lmer()`](https://lme4.r-lib.org/reference/lmer.html).

- **Coming from Python / statsmodels?** See [For Python users](why-python.md) for a
  side-by-side comparison and explanation of the limitation.
- **Coming from R / lme4?** See [For R users](why-r.md) for formula syntax mapping and
  shared references.

## Key features

- **LMM** — crossed / nested random intercepts and slopes, REML or ML, Satterthwaite or Kenward-Roger DFs
- **GLMM** — Laplace or adaptive Gauss-Hermite quadrature; 10+ families including binomial, Poisson, NB1/NB2, Beta, Gamma, zero-inflated, hurdle, and ZOIB
- **CLMM** — ordinal regression with random effects, matching R's `ordinal::clmm()`
- **Cox frailty** — Cox PH with Gaussian frailty, matching R's `coxme::coxme()`
- **Correlation structures** — AR(1) and compound symmetry for longitudinal data
- Sparse throughout — Z is never materialised as a dense matrix
- Full suite of diagnostics: residuals, leverage, Cook's D, MDFFITS, influence plots
- Compatible result object exposing the same attributes as `statsmodels.MixedLMResults`
- Validated against R reference implementations to tight tolerances

## Get started

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Installation & quickstart
:link: installation
:link-type: doc

Install interlace and fit your first crossed random-intercepts model.
:::

:::{grid-item-card} GLMM quickstart
:link: glmm-quickstart
:link-type: doc

Fit GLMMs with `glmer()` — binomial, Poisson, NB, Beta, and more.
:::

:::{grid-item-card} Examples
:link: examples
:link-type: doc

Worked notebooks: simulation, diagnostics, and comparison with lme4.
:::

:::{grid-item-card} API reference
:link: api/augment
:link-type: doc

Full documentation for every public function and result object.
:::

:::{grid-item-card} Background
:link: why-python
:link-type: doc

Why interlace exists, and how it compares to statsmodels and lme4.
:::
::::
