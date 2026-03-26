# interlace

[![PyPI](https://img.shields.io/github/v/tag/heliopais/interlace?label=version)](https://github.com/heliopais/interlace/releases)
[![CI](https://github.com/heliopais/interlace/actions/workflows/ci.yml/badge.svg)](https://github.com/heliopais/interlace/actions/workflows/ci.yml)
[![Docs](https://github.com/heliopais/interlace/actions/workflows/docs.yml/badge.svg)](https://github.com/heliopais/interlace/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](https://github.com/heliopais/interlace/blob/main/LICENSE)

**interlace** is a pure-Python implementation of profiled REML estimation for linear mixed
models with **crossed random intercepts** — targeting parity with R's `lme4::lmer()` and
designed as a drop-in replacement for `statsmodels.MixedLM` in production pipelines.

## Why interlace?

`statsmodels.MixedLM` is built around a single grouping factor. When a model has two
independent sources of variance — say, `subject` *and* `item` — there is no native
syntax for crossed random intercepts, and the available workarounds produce estimates
that diverge from REML. `interlace` fills that gap, implementing the same profiled
restricted maximum likelihood (REML) and sparse Cholesky machinery as R's `lme4::lmer()`.

- **Coming from Python / statsmodels?** See [For Python users](why-python.md) for a
  side-by-side comparison and explanation of the limitation.
- **Coming from R / lme4?** See [For R users](why-r.md) for formula syntax mapping and
  shared references.

## Key features

- Fit models with multiple crossed grouping factors, e.g. `(1|subject) + (1|item)`
- Sparse throughout — Z is never materialised as a dense matrix
- Full suite of HLM diagnostics: residuals, leverage, Cook's D, MDFFITS, influence plots
- Compatible result object exposing the same attributes as `statsmodels.MixedLMResults`
- Validated against R's `lme4::lmer()` to tight tolerances (fixed effects abs diff < 1e-4)
