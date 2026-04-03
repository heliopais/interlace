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

**interlace** is a pure-Python implementation of profiled [restricted maximum likelihood
(REML)](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood) estimation for linear mixed models with **crossed random effects** — targeting
parity with R's [`lme4::lmer()`](https://lme4.r-lib.org/reference/lmer.html) and
designed as a drop-in replacement for
[`statsmodels.MixedLM`](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html)
in production pipelines. It also supports **generalised linear mixed models** (GLMMs)
via Laplace approximation, matching R's
[`lme4::glmer()`](https://lme4.r-lib.org/reference/glmer.html) for binomial and
Poisson families.

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

- Fit models with multiple crossed grouping factors, e.g. `(1|subject) + (1|item)`
- Random slopes via lme4-style notation: `(1 + x | g)` and `(1 + x || g)`
- **GLMMs** via Laplace approximation: binomial (logistic), Poisson, and custom families
- Sparse throughout — Z is never materialised as a dense matrix
- Full suite of diagnostics: residuals, leverage, Cook's D, MDFFITS, influence plots
- Compatible result object exposing the same attributes as `statsmodels.MixedLMResults`
- Validated against R's [`lme4::lmer()`](https://lme4.r-lib.org/reference/lmer.html) and [`lme4::glmer()`](https://lme4.r-lib.org/reference/glmer.html) to tight tolerances

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

Fit binomial and Poisson GLMMs with `glmer()`.
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
