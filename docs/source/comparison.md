# Feature comparison

This page compares `interlace` with `statsmodels.MixedLM` (Python) and `lme4::lmer()`
(R) across the features that most often determine library choice.

---

## At a glance

| Feature | interlace | statsmodels MixedLM | lme4 (R) |
|---------|:---------:|:-------------------:|:--------:|
| **Random effect structures** | | | |
| Single random intercept | ✓ | ✓ | ✓ |
| Crossed random intercepts | ✓ | Workaround¹ | ✓ |
| Nested random effects | ✗ | ✓ | ✓ |
| Random slopes | ✗ | ✓ | ✓ |
| Correlated random effects | ✗ | ✓ | ✓ |
| Generalised LMM (Poisson, binomial) | ✗ | ✗² | ✓ (`glmer`) |
| **Estimation** | | | |
| REML (default) | ✓ | ✓ | ✓ |
| ML (for LRT / model comparison) | ✓ | ✓ | ✓ |
| Profiled REML | ✓ | ✗ | ✓ |
| **Optimizers** | | | |
| Default optimizer | L-BFGS-B | L-BFGS-B | BOBYQA |
| BOBYQA (gradient-free) | ✓ (optional extra) | ✗ | ✓ |
| Sparse Cholesky | ✓ (optional extra) | ✗ | ✓ (always) |
| **Formula syntax** | | | |
| Wilkinson formula for fixed effects | ✓ | ✓ | ✓ |
| Grouping via `groups=` argument | ✓ | ✓ | — |
| lme4 `(1\|g)` notation | ✓ (`random=`) | ✗ | ✓ |
| **Diagnostics** | | | |
| Residuals (marginal + conditional) | ✓ | Marginal only | Via HLMdiag |
| Leverage decomposition (H1/H2) | ✓ | ✗ | Via HLMdiag |
| Cook's distance | ✓ | ✗ | Via HLMdiag |
| MDFFITS | ✓ | ✗ | Via HLMdiag |
| COVTRACE / COVRATIO | ✓ | ✗ | Via HLMdiag |
| Relative variance change (RVC) | ✓ | ✗ | Via HLMdiag |
| Augmented data frame | ✓ (`hlm_augment`) | ✗ | Via HLMdiag |
| Cluster-bootstrap SE | ✓ | ✗ | Via `bootMer` |
| **Output** | | | |
| Fixed-effect coefficients | ✓ | ✓ | ✓ |
| Variance components | ✓ | ✓ | ✓ |
| BLUPs / conditional modes | ✓ | ✓ | ✓ |
| AIC / BIC | ✓ | ✓ | ✓ |
| **Numerical parity with lme4** | ✓³ | ✗ | — |
| **Language** | Python | Python | R |

¹ See [statsmodels workaround](#crossed-random-effects-statsmodels-workaround) below.
² statsmodels has separate GLM support; mixed GLMs require a different approach.
³ Fixed effects within 1e-4 (absolute); variance components within 5% (relative) on standard benchmarks.

---

## Notes on key rows

### Crossed random effects — statsmodels workaround

`statsmodels.MixedLM` accepts only a single grouping column. To approximate crossed
random effects you must pass the whole dataset as one group and add variance components
via `vc_formula`. This is documented in the statsmodels manual but produces estimates
that are **not numerically equivalent** to lme4 REML — particularly in unbalanced
designs. See [For Python / statsmodels users](why-python.md) for a worked example.

### Profiled REML

In profiled REML, the fixed effects are eliminated analytically so the optimiser only
searches over variance parameters. This reduces the dimension of the search space and
produces tighter convergence. Both `interlace` and `lme4` use this parameterisation.
`statsmodels.MixedLM` optimises over all parameters jointly.

### Optimizers

`interlace` defaults to scipy's L-BFGS-B. BOBYQA (gradient-free, matches lme4's
default) is available via `pip install "interlace-lme[bobyqa]"`. BOBYQA is more robust
near variance-component boundaries and gives closer numerical parity with lme4. See
[FAQ: Optimizer choice](faq.md#optimizer-choice-bobyqa-vs-default).

### Sparse Cholesky

`lme4` always uses Eigen's sparse Cholesky factorisation. `interlace` ships a dense
path by default and a sparse path when `scikit-sparse` is installed
(`pip install "interlace-lme[cholmod]"`). For small-to-medium models the dense path is
fast enough; for models with thousands of group levels the sparse path can be
substantially faster. See [FAQ: Solver choice](faq.md#solver-choice-cholmod-vs-default).

### Diagnostics

`lme4` has no built-in diagnostics — users rely on the R package
[HLMdiag](https://cran.r-project.org/package=HLMdiag) (Loy & Hofmann, 2014).
`interlace` bundles an equivalent suite directly so no additional package is needed.
See [For R / lme4 users](why-r.md#diagnostics) for the function-by-function mapping.

### Random slopes and nested designs

`interlace` is scoped to **crossed random intercepts**. For random slopes, nested
hierarchies, or generalised outcomes, use `statsmodels.MixedLM` or `lme4`.

---

## When to use each

| Situation | Recommended library |
|-----------|-------------------|
| Crossed random intercepts in Python | **interlace** |
| Need lme4 numerical parity from Python | **interlace** |
| Need built-in diagnostics in Python | **interlace** |
| Random slopes or nested designs in Python | statsmodels MixedLM |
| GLMM (Poisson, binomial) | lme4 (R) or statsmodels GLM |
| Full lme4 feature set | lme4 (R) |

---

## Further reading

- [For Python / statsmodels users](why-python.md) — detailed prose comparison and the statsmodels workaround
- [For R / lme4 users](why-r.md) — formula syntax mapping and shared references
- [FAQ](faq.md) — solver and optimizer guidance, convergence troubleshooting
- [Diagnostics guide](diagnostics.ipynb) — using the built-in diagnostic suite
