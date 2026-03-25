# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install      # create venv and install all dev deps via uv
make test         # run pytest
make lint         # ruff format + ruff check --fix (auto-fixes in place, then reports unfixable issues)
make typecheck    # mypy
make check        # lint + typecheck + test (full CI gate)
make build        # build wheel and sdist
make clean        # remove dist/, caches

# Run a single test file or test function
uv run pytest tests/test_foo.py
uv run pytest tests/test_foo.py::test_bar
```

## Architecture

`interlace` is a pure-Python implementation of profiled REML estimation for linear mixed models with **crossed random intercepts**, targeting parity with R's `lme4::lmer()`. It is designed as a drop-in replacement for `statsmodels.MixedLM` in the downstream `gpgap` diagnostics pipeline.

### Planned module layout (`src/interlace/`)

| Module | Responsibility |
|---|---|
| `__init__.py` | Public API: `fit(formula, data, method)` |
| `formula.py` | Parse lme4-style `(1\|g)` and pipe-style `y ~ x \| g1+g2` formulas; build X via `formulaic`; extract grouping factor arrays |
| `sparse_z.py` | Build per-factor `scipy.sparse.csc_matrix` indicator matrices; hstack into joint Z |
| `profiled_reml.py` | Lambda_theta parameterisation; sparse Cholesky (splu / optional sksparse.cholmod); profiled REML objective; L-BFGS-B optimiser |
| `result.py` | `CrossedLMEResult` dataclass and `ModelInfo` dataclass |
| `predict.py` | BLUP-based prediction; unseen group levels shrink to zero |
| `compat.py` | Attribute mapping so `CrossedLMEResult` is a drop-in for `gpgap` code that accesses `statsmodels` result attributes |

### Key design constraints

- **No R dependency.** Reuse statsmodels functionality wherever possible — prefer importing from statsmodels over reimplementing equivalent logic.
- **Formula syntax mirrors statsmodels `MixedLM.from_formula()`**: `formula="y ~ x1 + x2"` for fixed effects (standard patsy/formulaic syntax), `groups="group_col"` as a separate parameter, optional `re_formula` for random effects structure.
- **`CrossedLMEResult` must be statsmodels-compatible** for `gpgap`: it exposes `fe_params`, `resid`, `scale`, `fittedvalues`, `random_effects`, `predict(newdata)`, `_gpgap_group_col`, `_gpgap_vc_cols`, and a `model` attribute with `exog`, `groups`, `data.frame`, `endog_names`, `formula`.
- **Sparse throughout**: Z is never materialised as a dense matrix. The Cholesky solve operates on `Z'Z + I` in sparse form.
- **Validation target**: results must match R lme4 to tight tolerances (fixed effects abs_diff < 1e-4, variance components rel_diff < 5%, BLUP correlation > 0.99, conditional residual correlation > 0.999).

### Development workflow: TDD

Always write the test first, see it fail, then write the minimum implementation to make it pass.

1. Pick a task from `bd ready`
2. Write a failing test in `tests/` that captures the acceptance criteria
3. Run `make test` — confirm it fails for the right reason
4. Implement the minimum code to pass the test
5. Run `make check` before marking the task done

Never write implementation code that does not have a corresponding failing test driving it.

### Issue tracker

Issues are tracked with `bd` (beads). Run `bd list` to see open tasks, `bd ready` for unblocked ones, `bd epic status` for epic-level progress.
