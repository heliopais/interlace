# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-04-24

## User Preferences

<!-- How the user likes things done. Code style, tools, patterns, communication. -->

## Key Learnings

- **Project:** interlace-lme
- **Description:** <p align="center">
- ZI models (ZIP, ZINB2) need custom PIRLS working weights: the score and negative Hessian of the full mixture log-likelihood w.r.t. eta, not the count-component approximation. For y=0, the posterior probability r0 = (1-pi)*f(0)/P0 modulates the contribution. The negative Hessian for zeros can go negative when mu*(1-r0) > 1; floor it at 1% of the count-component Hessian.

- Cox PH frailty uses Laplace-approximated integrated partial likelihood (IPL). The correct formula is: IPL = PLL + 0.5*log|Σ⁻¹| - 0.5*log|H_bb + Σ⁻¹|. The 0.5*log|Σ⁻¹| term (prior normalisation) is essential — without it, the optimizer overestimates frailty variance by ~7x because there's no penalty for model complexity.
- Cox PH has no intercept (absorbed into baseline hazard). When building the design matrix, use `~ 0 + rhs` with formulaic. The `Surv(time, event) ~ ...` LHS syntax is parsed separately from the RHS design matrix.
- The coxme module reuses the Lambda parameterisation and sparse Z infrastructure from profiled_reml. The penalty term is -0.5*||Λ⁻¹b||² (same structure as GLMM PIRLS), but working weights come from the Breslow partial likelihood instead of a GLM family.

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

- [2026-04-22] When adding a new GLMMFamily variant (e.g., ZI), update ALL family-name dispatch points: `_clamp_mu`, `_glm_start`, `_conditional_loglik`, `_zi_pirls_weights` (PIRLS loop, final log|A|, SE computation, profiled objective), and any `family.name == ...` checks.
- [2026-04-22] `np.empty_like(y)` inherits y's dtype. For integer count data, this creates int64 arrays that silently truncate float log-likelihood values. Always use `np.empty_like(y, dtype=np.float64)` when allocating arrays for log-likelihood computations.
- [2026-04-22] Phase 2 (Nelder-Mead joint theta+beta) must be SKIPPED for ZI families. The ZI-adjusted PIRLS has warm-start-dependent local optima, making the profiled objective non-deterministic. Nelder-Mead then reports a `res.fun` that doesn't match `res.x`. Phase 1 alone gives near-exact parity with glmmTMB for ZI models.
- [2026-04-23] Hurdle model y=0 observations have score=0 and neg_hess=0 w.r.t. the count-component eta (they carry no information about count mean mu). The 1e-10 floor in `_zi_pirls_weights` handles the 0/0 division gracefully — these observations effectively drop out of PIRLS, which is statistically correct.

- [2026-04-23] GammaFamily supports both log and inverse links via a `link` parameter in the constructor. The inverse link requires special care: data must keep eta > 0, and mu_eta needs a floor to avoid overflow in 1/eta^2.
- [2026-04-23] NB1 deviance residuals can be negative (unlike NB2/Poisson) because r = mu/alpha is observation-dependent, so the saturated model at mu=y does not maximise the NB1 log-likelihood. This is mathematically correct, not a bug. Don't write tests asserting non-negative deviance for NB1.

- [2026-04-24] The Laplace IPL for Cox frailty MUST include the prior normalisation term 0.5*log|Σ⁻¹|. Without it, the optimizer dramatically overestimates frailty variance because the objective doesn't penalise unnecessary model complexity. This was the first bug caught by the parameter-recovery test.
- [2026-04-24] For 1D theta optimization (shared frailty), use `minimize_scalar` (Brent's method), NOT `L-BFGS-B`. L-BFGS-B with numerical gradients can get stuck at the initial value (theta=1.0) because the finite-difference step is unreliable for this objective. Brent's method is derivative-free and 4x faster (14s vs 58s).
- [2026-04-24] The full sparse Newton step for the inner Cox solve is LESS reliable than the diagonal approximation. The augmented (p+q)x(p+q) system is poorly conditioned when p << q (e.g., 2 covariates + 50 groups), causing the optimizer to converge to worse points. Stick with diagonal inner step + full-system Schur complement SEs.

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->
- [2026-04-24] Cox frailty (coxme) uses penalized partial likelihood + Laplace IPL (matching R's coxme package), not EM or full Bayesian MCMC. Chose PPL because: (1) same algorithmic structure as GLMM PIRLS, enabling code reuse; (2) matches the R reference implementation; (3) fast convergence for shared frailty. The diagonal Hessian approximation in the inner Newton step trades some convergence speed for O(n) per-iteration cost.
