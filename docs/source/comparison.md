# Feature comparison

This page compares `interlace` with `statsmodels.MixedLM` (Python),
`linearmodels.RandomEffects` (Python), and `lme4::lmer()` (R) across the features
that most often determine library choice.

---

## At a glance

| Feature | interlace | statsmodels MixedLM | linearmodels RE | lme4 (R) |
|---------|:---------:|:-------------------:|:---------------:|:--------:|
| **Random effect structures** | | | | |
| Single random intercept | ✓ | ✓ | One-way entity⁴ | ✓ |
| Crossed random intercepts | ✓ | Workaround¹ | ✗ | ✓ |
| Nested random effects | ✓ *(v0.2.4+)* | ✓ | ✗ | ✓ |
| Random slopes | ✓ *(v0.2.1+)* | ✓ | ✗ | ✓ |
| Correlated random effects | ✓ *(v0.2.1+)* | ✓ | ✗ | ✓ |
| Generalised LMM (10+ families) | ✓ (`glmer`) | ✗² | ✗ | ✓ (`glmer`) |
| Ordinal mixed models (CLMM) | ✓ (`clmm`) | ✗ | ✗ | Via `ordinal` |
| Cox frailty (survival) | ✓ (`coxme`) | ✗ | ✗ | Via `coxme` |
| Residual correlation (AR1, CS) | ✓ *(v0.3.0+)* | ✗ | ✗ | Via `nlme` |
| **Estimation** | | | | |
| REML (default) | ✓ | ✓ | ✗ (Swamy-Arora⁵) | ✓ |
| ML (for LRT / model comparison) | ✓ | ✓ | ✗ | ✓ |
| Profiled REML | ✓ | ✗ | ✗ | ✓ |
| IV / 2SLS fixed-effects estimator | ✗ | ✗ | ✓ | ✗ |
| Cluster-robust / HAC standard errors | ✗ | ✗ | ✓ (4 types) | ✗ |
| **Optimizers** | | | | |
| Default optimizer | L-BFGS-B | L-BFGS-B | Closed form | BOBYQA |
| BOBYQA (gradient-free) | ✓ (optional extra) | ✗ | — | ✓ |
| Sparse Cholesky | ✓ (optional extra) | ✗ | — | ✓ (always) |
| **Formula syntax** | | | | |
| Wilkinson formula for fixed effects | ✓ | ✓ | ✓ (formulaic) | ✓ |
| Grouping via `groups=` argument | ✓ | ✓ | Multi-index DF | — |
| lme4 `(1\|g)` notation | ✓ (`random=`) | ✗ | ✗ | ✓ |
| **Diagnostics** | | | | |
| Residuals (marginal + conditional) | ✓ | Marginal only | Residuals only | Via HLMdiag |
| Leverage decomposition (H1/H2) | ✓ | ✗ | ✗ | Via HLMdiag |
| Cook's distance | ✓ | ✗ | ✗ | Via HLMdiag |
| MDFFITS | ✓ | ✗ | ✗ | Via HLMdiag |
| COVTRACE / COVRATIO | ✓ | ✗ | ✗ | Via HLMdiag |
| Relative variance change (RVC) | ✓ | ✗ | ✗ | Via HLMdiag |
| Augmented data frame | ✓ (`hlm_augment`) | ✗ | ✗ | Via HLMdiag |
| Cluster-bootstrap SE | ✓ | ✗ | ✗ | Via `bootMer` |
| **Output** | | | | |
| Fixed-effect coefficients | ✓ | ✓ | ✓ | ✓ |
| Variance components | ✓ | ✓ | ✗ (θ only) | ✓ |
| BLUPs / conditional modes | ✓ | ✓ | ✗ | ✓ |
| Estimated marginal means (`emmeans`) | ✓ *(v0.2.8+)* | ✗ | ✗ | Via `emmeans` |
| Satterthwaite DFs | ✓ | ✗ | ✗ | Via `lmerTest` |
| Kenward-Roger DFs | ✓ *(v0.3.0+)* | ✗ | ✗ | Via `lmerTest`/`pbkrtest` |
| AIC / BIC | ✓ | ✓ | ✓ | ✓ |
| **Numerical parity with lme4** | ✓³ | ✗ | ✗ | — |
| **Language** | Python | Python | Python | R |

¹ See [statsmodels workaround](#crossed-random-effects-statsmodels-workaround) below.
² statsmodels has separate GLM support; mixed GLMs require a different approach.
³ Fixed effects within 1e-4 (absolute); variance components within 5% (relative) on standard benchmarks.
⁴ One-way entity-level random intercept only, using the Swamy-Arora estimator. Requires a multi-index (entity, time) DataFrame.
⁵ Uses a method-of-moments estimator, not ML or REML. Variance component is not directly comparable to lme4/interlace output.

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

### linearmodels — panel econometrics, not LMM

[`linearmodels`](https://bashtage.github.io/linearmodels/) is a panel-data and
instrumental-variables econometrics library, not a general-purpose linear mixed model
package. The distinction matters:

| Dimension | interlace | linearmodels |
|---|---|---|
| **Data structure** | Flat DataFrame with grouping columns | Multi-index (entity, time) DataFrame |
| **Random effect** | Crossed/nested intercepts and slopes via REML | One-way entity intercept (Swamy-Arora) |
| **Estimation** | Profiled REML or ML | Closed-form moments estimator |
| **Primary use cases** | Psycholinguistics, social science, medicine | Macroeconomics, finance, policy evaluation |
| **IV / 2SLS** | ✗ | ✓ (`IV2SLS`, `IVLIML`, `IVGMM`) |
| **System estimators** | ✗ | ✓ (`SUR`, `IV3SLS`) |
| **Clustered / HAC SEs** | ✗ | ✓ (4 covariance types) |

Use `linearmodels.RandomEffects` when:
- Your data has a two-dimensional panel structure (entities × time periods).
- You need IV/2SLS or system estimators alongside the RE model.
- You need cluster-robust or Driscoll-Kraay HAC standard errors.
- The econometric one-way RE assumption (individual effect uncorrelated with regressors) is acceptable.

Use `interlace` (or `lme4`) when:
- Your data has **crossed** random effects (e.g. subjects × stimuli).
- You need REML variance components that are numerically equivalent to lme4.
- You need random slopes, nested designs, or profile-likelihood CIs on variance parameters.
- You need a built-in diagnostics suite (Cook's D, leverage, MDFFITS, …).

### Diagnostics

`lme4` has no built-in diagnostics — users rely on the R package
[HLMdiag](https://cran.r-project.org/package=HLMdiag) (Loy & Hofmann, 2014).
`interlace` bundles an equivalent suite directly so no additional package is needed.
See [For R / lme4 users](why-r.md#diagnostics) for the function-by-function mapping.

### Random slopes and nested designs

`interlace` supports random slopes (v0.2.1+) and nested random effects (v0.2.4+)
via the `random=` parameter using lme4-style `(term | group)` notation.
For generalised outcomes, use `interlace.glmer()`. For ordinal outcomes, use
`interlace.clmm()`. For survival data, use `interlace.coxme()`.

---

## When to use each

| Situation | Recommended library |
|-----------|-------------------|
| Crossed random intercepts in Python | **interlace** |
| Need lme4 numerical parity from Python | **interlace** |
| Need built-in diagnostics in Python | **interlace** |
| Random slopes or nested designs in Python | **interlace** *(v0.2.1+ / v0.2.4+)* or statsmodels MixedLM |
| One-way entity RE with panel (entity × time) data | **linearmodels** |
| Need IV/2SLS, SUR, or IV3SLS alongside RE | **linearmodels** |
| Need clustered or Driscoll-Kraay HAC SEs | **linearmodels** |
| GLMM (Poisson, binomial, Beta, ...) in Python | **interlace** (`glmer()`) |
| Ordinal regression with random effects in Python | **interlace** (`clmm()`) |
| Survival with frailty in Python | **interlace** (`coxme()`) |
| Longitudinal data with AR(1)/CS correlation | **interlace** (`fit(correlation=...)`) |
| Full lme4 feature set | lme4 (R) |

---

## Further reading

- [For Python / statsmodels users](why-python.md) — detailed prose comparison and the statsmodels workaround
- [For R / lme4 users](why-r.md) — formula syntax mapping and shared references
- [FAQ](faq.md) — solver and optimizer guidance, convergence troubleshooting
- [Diagnostics guide](diagnostics.ipynb) — using the built-in diagnostic suite
- [linearmodels documentation](https://bashtage.github.io/linearmodels/) — panel RE, IV, SUR, and GMM estimators
