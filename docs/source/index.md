# interlace

[![PyPI](https://img.shields.io/pypi/v/interlace-lme?label=PyPI&color=blue)](https://pypi.org/project/interlace-lme/)
[![CI](https://github.com/heliopais/interlace/actions/workflows/ci.yml/badge.svg)](https://github.com/heliopais/interlace/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://heliopais.github.io/interlace/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](https://github.com/heliopais/interlace/blob/main/LICENSE)

**interlace** is a pure-Python implementation of profiled REML estimation for linear mixed
models with **crossed random intercepts** — targeting parity with R's `lme4::lmer()` and
designed as a drop-in replacement for `statsmodels.MixedLM` in production pipelines.

## Key features

- Fit models with multiple crossed grouping factors, e.g. `(1|worker) + (1|company)`
- Sparse throughout — Z is never materialised as a dense matrix
- Full suite of HLM diagnostics: residuals, leverage, Cook's D, MDFFITS, influence plots
- Compatible result object exposing the same attributes as `statsmodels.MixedLMResults`
- Validated against R's `lme4::lmer()` to tight tolerances (fixed effects abs diff < 1e-4)
