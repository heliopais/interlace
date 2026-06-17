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

- When a residual correlation structure (AR1, CS) is present, ALL post-estimation quantities (beta, BLUPs, residuals, fe_cov) must be computed from whitened data (y_w, X_w, Z_w). The theta values from the optimizer are defined relative to the whitened cross-products — using them with unwhitened data gives subtly wrong results (variance components still match but BLUPs and residuals diverge).

- Kenward-Roger DFs match R's lmerTest when computed as Satterthwaite in the **un-profiled variance-component parameterization** (σ²_1, ..., σ²_k, σ²_resid). The profiled theta parameterization (used by standard Satterthwaite) gives dramatically different slope DFs (~906 vs 104) because σ²_resid is profiled out. The KR bias correction (vcovAdj) is negligible (<1e-5 relative) for moderate samples because the REML projection matrix P annihilates X from both sides (X'PV_rPX = 0).

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

- [2026-04-24] KR bias correction: do NOT use Σ W_rs * dC_r C⁻¹ dC_s — this gives a correction ~10,000x too large. The KR97 formula involves X'PV_rPX which vanishes because X'P = 0. The correction is zero/negligible; just use C_adj = fe_cov.
- [2026-04-24] KR DFs: do NOT use the full moment-matching formula (rho, m = 4+3/(rho-1)) for per-coefficient t-tests. lmerTest uses Satterthwaite ν = 2C_jj²/(g'Wg) in the un-profiled vc parameterization, which gives the correct answer. The full KR formula is for multi-DF F-tests (anova, q>1).

- [2026-04-22] When adding a new GLMMFamily variant (e.g., ZI), update ALL family-name dispatch points: `_clamp_mu`, `_glm_start`, `_conditional_loglik`, `_zi_pirls_weights` (PIRLS loop, final log|A|, SE computation, profiled objective), and any `family.name == ...` checks.
- [2026-04-22] `np.empty_like(y)` inherits y's dtype. For integer count data, this creates int64 arrays that silently truncate float log-likelihood values. Always use `np.empty_like(y, dtype=np.float64)` when allocating arrays for log-likelihood computations.
- [2026-04-22] Phase 2 (Nelder-Mead joint theta+beta) must be SKIPPED for ZI families. The ZI-adjusted PIRLS has warm-start-dependent local optima, making the profiled objective non-deterministic. Nelder-Mead then reports a `res.fun` that doesn't match `res.x`. Phase 1 alone gives near-exact parity with glmmTMB for ZI models.
- [2026-04-23] Hurdle model y=0 observations have score=0 and neg_hess=0 w.r.t. the count-component eta (they carry no information about count mean mu). The 1e-10 floor in `_zi_pirls_weights` handles the 0/0 division gracefully — these observations effectively drop out of PIRLS, which is statistically correct.

- [2026-04-23] GammaFamily supports both log and inverse links via a `link` parameter in the constructor. The inverse link requires special care: data must keep eta > 0, and mu_eta needs a floor to avoid overflow in 1/eta^2.
- [2026-04-23] NB1 deviance residuals can be negative (unlike NB2/Poisson) because r = mu/alpha is observation-dependent, so the saturated model at mu=y does not maximise the NB1 log-likelihood. This is mathematically correct, not a bug. Don't write tests asserting non-negative deviance for NB1.

- [2026-04-24] The Laplace IPL for Cox frailty MUST include the prior normalisation term 0.5*log|Σ⁻¹|. Without it, the optimizer dramatically overestimates frailty variance because the objective doesn't penalise unnecessary model complexity. This was the first bug caught by the parameter-recovery test.
- [2026-04-24] For 1D theta optimization (shared frailty), use `minimize_scalar` (Brent's method), NOT `L-BFGS-B`. L-BFGS-B with numerical gradients can get stuck at the initial value (theta=1.0) because the finite-difference step is unreliable for this objective. Brent's method is derivative-free and 4x faster (14s vs 58s).
- [2026-04-24] The full sparse Newton step for the inner Cox solve is LESS reliable than the diagonal approximation. The augmented (p+q)x(p+q) system is poorly conditioned when p << q (e.g., 2 covariates + 50 groups), causing the optimizer to converge to worse points. Stick with diagonal inner step for the Newton iteration.
- [2026-04-24] Cox SE computation MUST use the exact Breslow Hessian products (`_breslow_info_products`), not `diag(w)`. The Cox Hessian is -H = Σ_k [diag(a_k) - a_k a_k']; using only diag(w) drops the rank-1 outer products, overestimating information by ~13% and underestimating SEs by ~7%. The exact computation uses backward cumulative weighted sums: O(n*(p+q)²) storage, O(n*d*(p+q)) time. The diagonal approximation is fine for the inner Newton iteration but NOT for SE computation.
- [2026-04-24] In fit(), when a correlation structure is present, the post-estimation code MUST whiten (y, X, Z) before computing cross-products for beta/BLUPs. The optimizer's theta is defined on the whitened system; using it with raw cross-products gives wrong BLUPs (corr=0.98 vs R instead of 1.0). Residuals/fitted values use the original (unwhitened) data: resid = y - X*beta - Z*b.

- [2026-04-24] CLMM (cumulative link mixed model): the design matrix X must be built WITH an intercept (`~ rhs`) then the Intercept column dropped. Using `~ 0 + rhs` gives ALL dummy levels for the first factor instead of treatment contrasts. The thresholds absorb the intercept role.
- [2026-04-24] CLMM threshold parameterisation: use increments (alpha_1 free, alpha_k = alpha_1 + Σ exp(log_delta_j)) to enforce strict ordering during unconstrained optimization.
- [2026-04-24] CLMM SEs MUST include theta (variance parameter) uncertainty. Conditioning on theta gives SEs that are consistently ~4-6% too narrow. The fix: numerical Hessian of the full profiled Laplace LL w.r.t. (alpha, beta, theta), then extract the (alpha, beta) block from the inverse. This gives <0.1% parity with R's ordinal::clmm.

- [2026-05-08] `scipy.sparse.linalg.spsolve(A, B)` silently squeezes a (q, 1) 2D dense rhs to a (q,) 1D output, breaking 2D shape contracts. Any wrapper that promises (q, p) output for 2D rhs must reshape: `if rhs.ndim == 2 and out.ndim == 1: out = out.reshape(rhs.shape[0], -1)`. This bit `_sparse_solve` in profiled_reml.py for intercept-only X (p=1) + crossed RE in `reml_gradient`.
- [2026-05-08] The current `reml_gradient` is correct (passes check_grad) but **slower than forward-difference** at q ≥ ~80 because it computes a dense `A11_inv = lu.solve(np.eye(q))` per call (O(q³)). Obj-call count drops 50–75% but wall time is flat-to-negative on multi-factor diagonal fits. Don't flip `use_gradient=True` as the default until the gradient body uses selective inverse / Hutchinson estimators (tracked as interlace-mxzk). Counter-intuitive: an analytic gradient that's mathematically correct can still lose to FD if its per-call cost dominates.

- [2026-06-13] Phase B.0 of interlace-f0x1: no C-speed selected inverse is reachable as a dep on macOS/Linux. `sksparse v0.5.0`'s `Factor.inv()` returns the full P^T L^-T L^-1 P inverse (dense q×q), not a selected one. libcholmod 5.3.4 (SuiteSparse 7.12.2) does NOT export `cholmod_spinv` / sparseinv / takahashi — Tim Davis ships `sparseinv` separately under `SuiteSparse/MATLAB_Tools/sparseinv` and it is not compiled into any installed library. `scipy.sparse.linalg` has no Takahashi routine. Remaining paths: Numba (issue's planned B.1) or vendor `sparseinv.c` via cffi (out of issue scope).

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->
- [2026-04-24] Cox frailty (coxme) uses penalized partial likelihood + Laplace IPL (matching R's coxme package), not EM or full Bayesian MCMC. Chose PPL because: (1) same algorithmic structure as GLMM PIRLS, enabling code reuse; (2) matches the R reference implementation; (3) fast convergence for shared frailty. The diagonal Hessian approximation in the inner Newton step trades some convergence speed for O(n) per-iteration cost.
- [2026-04-24] CLMM uses a separate module (clmm.py) rather than shoehorning into GLMMFamily/glmm_laplace. Rationale: ordinal models have threshold parameters, no intercept, and different PIRLS working quantities (score/Hessian from cumulative probabilities, not mean/variance). Shares Lambda/sparse Z infrastructure from profiled_reml and the Schur complement solve pattern from glmm_laplace.
