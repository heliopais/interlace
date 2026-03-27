# Changelog

## v0.2.4 — 2026-03-27

### `quantreg_ker_se` — Hall-Sheather kernel SE for quantile regression

New utility function `quantreg_ker_se(residuals, X, tau=0.5, hs=True)` that
computes quantile regression standard errors matching R's
`quantreg::summary.rq(se="ker", hs=TRUE)`.

```python
from interlace import quantreg_ker_se

fit = smf.quantreg("normalized ~ male", data=df).fit(q=0.5)
se = quantreg_ker_se(fit.resid, fit.model.exog, tau=0.5, hs=True)
```

`statsmodels` uses a different (narrower) bandwidth rule and underestimates the
SE by ~11% on typical payroll datasets relative to R.  This function reproduces
R's result exactly by implementing the Hall-Sheather bandwidth:

```
h = n^(-1/3) · z_{α/2}^(2/3) · [(1.5 · φ(Φ⁻¹(τ))²) / (2·Φ⁻¹(τ)² + 1)]^(1/3)
```

Set `hs=False` to use the Bofinger bandwidth instead
(`quantreg::summary.rq(se="ker", hs=FALSE)`).

### Fixes

- CHOLMOD compat: replaced deprecated `cholmod.analyze()` call with
  `cholmod.cholesky()` for `scikit-sparse` ≥ 0.5.0 (issue #8).

---

## v0.2.3 — 2026-03-27

### CHOLMOD sparse Cholesky

The REML objective now uses CHOLMOD's symbolic-then-numeric refactorisation
when `scikit-sparse` is installed.  CHOLMOD performs a single symbolic
analysis on the first iteration and reuses the sparsity pattern on every
subsequent step — substantially faster for large or deeply nested designs
where the KKT system has a stable non-zero structure.

```bash
pip install "interlace-lme[cholmod]"
```

No API change required — the fast path is activated automatically.

### `bootstrap_se` — cluster bootstrap for the median

`CrossedLMEResult` gains a `bootstrap_se()` method that computes a
cluster-bootstrap standard error for a scalar statistic of the response
(currently `"median"`):

```python
se = result.bootstrap_se(statistic="median", n_bootstrap=1000,
                          resample_level="group", seed=42)
```

`resample_level="group"` (default) resamples grouping-factor levels with
replacement and includes all observations from each sampled group, matching
R's `boot` cluster bootstrap.  `resample_level="observation"` resamples
individual rows.

### Random slopes in all diagnostics

`residuals.hlm_resid`, `leverage.leverage`, and all `influence` functions
now support models fitted with random slopes (lme4-style `(1 + x | g)`
specifications).

### `hlm_influence` performance

Design matrices `X`, `y`, and `Z` are pre-built once before the case-deletion
loop.  Per-observation formula re-parsing is eliminated, giving a meaningful
speedup on large datasets.

### Fixes

- `predict()`: fixed categorical column-ordering bug that produced wrong BLUPs
  when the newdata column order differed from the training frame.

---

## v0.2.2 — 2026-03-27

### Zero-pandas internals

Pandas has been eliminated from the diagnostics pipeline in two phases:

- **Phase 2** — `formula.py` and `sparse_z.py` now use
  [formulaic](https://matthewwardrop.github.io/formulaic/) instead of patsy,
  and [narwhals](https://narwhals-dev.github.io/narwhals/) for frame
  abstraction throughout.
- **Phase 3** — all diagnostics functions (`residuals`, `leverage`,
  `influence`, `augment`) operate on narwhals frames internally; polars
  DataFrames pass through without a pandas round-trip.

### Leverage fix for truly-crossed RE

For models with two or more grouping factors, the fixed-effects leverage
component now uses the OLS hat matrix `H_X = X(X'X)⁻¹X'` rather than the
full mixed-model hat approximation.  The previous calculation over-estimated
leverage in balanced crossed designs.

---

## v0.2.1 — 2026-03-26

### Random slopes

interlace now supports lme4-style random slopes in addition to random intercepts.

```python
# Correlated random intercept + slope
result = interlace.fit(
    "y ~ x",
    data=df,
    random=["(1 + x | group)"],
)

# Independent (uncorrelated) parameterisation
result = interlace.fit(
    "y ~ x",
    data=df,
    random=["(1 + x || group)"],
)
```

`result.random_effects["group"]` now returns a DataFrame with one column per
random effect term.  `result.varcov` exposes the full random-effect covariance
matrix.

Warm-start support: case-deletion refits initialise the optimizer from the
full-model `theta_hat`, reducing iterations for large models.

---

## v0.2.0 — 2026-03-26

### New feature: BOBYQA optimizer for R/Python diagnostic parity

interlace now ships an optional BOBYQA optimizer that significantly
improves parity between Python and R (HLMdiag) Cook's D flagging counts
in influence diagnostics.

#### Background

The residual gap documented in prior benchmarking (1.2–1.5× over-flagging
relative to R) traces to a single root cause: L-BFGS-B is gradient-based
and converges poorly near θ = 0 boundaries — the exact regime encountered
during case-deletion refits when a group shrinks to near-zero size.
R's lme4 uses **BOBYQA** (a gradient-free trust-region algorithm) by default,
which is inherently more robust at those boundaries.

#### What changed

**`interlace.fit()` — new `optimizer` parameter**

```python
interlace.fit("y ~ x", data=df, groups="firm", optimizer="bobyqa")
```

The `optimizer` keyword accepts `"lbfgsb"` (default, unchanged behaviour)
or `"bobyqa"`. BOBYQA requires the `bobyqa` optional extra (see
[Installation](installation.md)).

**Influence functions — `optimizer` parameter threaded through**

All case-deletion refit functions now accept `optimizer`:

```python
from interlace.influence import hlm_influence, cooks_distance, mdffits, tau_gap, n_influential

# Use BOBYQA for all refits — closer to R/HLMdiag flagging counts
infl = hlm_influence(model, optimizer="bobyqa")
cd   = cooks_distance(model, optimizer="bobyqa")
```

**Single-RE statsmodels routing**

When `optimizer="bobyqa"` is passed to any influence function and the
model is a statsmodels `MixedLMResults` with `_gpgap_group_col` set,
case-deletion refits are routed through interlace's own REML fitter
(instead of statsmodels). This ensures BOBYQA is used end-to-end rather
than only for crossed-RE models.

#### Parity improvement (benchmarks)

| Scenario | Before (L-BFGS-B) | After (BOBYQA) | R baseline |
|---|---|---|---|
| `large_scale` (single-RE, n=5000) | 259 flags | ~211 | 211 |
| `crossed_re` (3-RE, n=2000) | 327 flags | ~215 | 215 |
| `many_firms` (single-RE, n=2000) | 2000 | 2000 | 2000 ✓ |
| `many_covariates` | 93 | 93 | 92 ✓ |

The default `optimizer="lbfgsb"` is unchanged — no action required for
existing code.

---

## v0.1.1 — 2026-03-21

- Extended `Z` matrix and `Lambda_theta` to support random slopes
- Added `RandomEffectSpec` and `parse_random_effects` for lme4-style
  random parameter parsing
- Wired REML fit to use generalised `Z` and `Lambda` for random slopes

## v0.1.0 — initial release

- Profiled REML estimation for linear mixed models with crossed random
  intercepts
- Sparse `Z` matrix throughout — never materialised as dense
- Full diagnostics suite: residuals, leverage, Cook's D, MDFFITS,
  influence plots
- Compatible `CrossedLMEResult` object exposing the same attributes as
  `statsmodels.MixedLMResults`
- Validated against R's `lme4::lmer()` (fixed effects abs diff < 1e-4)
