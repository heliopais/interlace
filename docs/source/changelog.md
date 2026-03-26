# Changelog

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
