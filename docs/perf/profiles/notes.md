# Profile notes — Sleepstudy & CBPP

Profile artefacts for issue **interlace-1yyz** (parent epic
**interlace-81ru**: close the 7-11x gap vs lme4).

## Setup

- `python 3.14.2`, `scipy 1.17.1`, `numpy 2.4.3`, interlace `main` (commit
  `e68cbd0`).
- Profiled via `cProfile` (call counts, %time) and call-counter
  monkey-patch (iteration counts). `py-spy` flamegraphs not produced
  on this host (macOS without root). Equivalent SVGs generated via
  `flameprof` direct from the `.prof` dumps — open
  `*.svg` next to this file.
- Re-run with `uv run python docs/perf/profile_fits.py`.

## Wall timings (n_reps=5, after 1 warmup, no profiler attached)

| dataset    | config                       | median (s) | min (s) |
|---         |---                           |---:        |---:     |
| sleepstudy | auto (CHOLMOD)               | 0.124      | 0.112   |
| sleepstudy | superlu (manuscript baseline)| 0.163      | 0.138   |
| cbpp       | auto                         | 0.682      | 0.534   |

Reference targets (lme4, manuscript): Sleepstudy 0.009s (≈14× over our
SuperLU figure here, 8× over CHOLMOD), CBPP 0.038s (≈18× over our
median). Note: this host shows numbers above the manuscript's because
the timer here runs each fit cold; the manuscript runs the JIT-warm
case averaged over 10 reps. Ratios within this host are still
informative.

## Iteration / call counts (one fit)

| dataset    | reml_obj | _pirls | _laplace_obj_profiled | spsolve | splu |
|---         |---:      |---:    |---:                   |---:     |---: |
| sleepstudy/auto    |  92 |  0 |   0 |   65 |  25 |
| sleepstudy/superlu |  92 |  0 |   0 |  249 | 117 |
| cbpp/auto          |   0 | 19 | 403 | 1220 | 422 |

- L-BFGS-B drives ~92 outer objective calls on Sleepstudy. With three
  thetas and finite-difference gradient at 3 evals/call, that's
  consistent with ~23 outer iterations.
- CBPP runs a **two-phase optimiser**: Phase 1 = L-BFGS-B over θ alone
  (PIRLS profiled out — gives 19 `_pirls` calls), Phase 2 = Nelder-Mead
  over (θ, β) jointly using `_laplace_objective_profiled` (403 evals).
  This Phase-2 dominates CBPP wall time and is **not in the epic
  scope** — see §"New findings" below.

## Hypothesis verdicts vs epic interlace-81ru body

For each Tier-0/1 hypothesis listed under "## Scope" in the epic:

### 1. `glmm_laplace.py:895/1668` — column-loop `spsolve` refactorises A on every column

- **Verdict: PARTIAL / SECONDARY.**
- CBPP: 1220 `spsolve` calls → `_superlu.gssv` self-time **0.036 s = 2.5 % of fit**. Per-call solve is cheap at q=15.
- Multi-RHS solve still saves the redundant pattern work (`get_index_dtype`, copy, etc.) inside `spla.spsolve`, but absolute upside on CBPP is in the **3-5 %** range, not the dominant gap.
- Worth doing for code quality + larger-q correctness, **not** the lever to chase first.

### 2. `profiled_reml.py:226 sparse_chol_logdet` — SuperLU instead of CHOLMOD

- **Verdict: PARTIAL.**
- Sleepstudy SuperLU vs CHOLMOD: 0.163 → 0.124 s wall (**~24 % saving**, single fit).
- `_superlu.gssv` + `gstrf` self-time on SuperLU path is only ~5 % combined; the win comes from CHOLMOD's lower per-call overhead (1 factor + 1 solve vs SciPy's repeated `splu` build).
- Promoting CHOLMOD to a hard dep (epic task `interlace-98ip`) buys ~25 % on Sleepstudy, **0 %** on CBPP (GLMM path uses `spla.spsolve` directly, never the `sparse_chol_logdet` route — see `glmm_laplace.py:895,1066,1668`).

### 3. L-BFGS-B without analytic gradient

- **Verdict: CONFIRMED for Sleepstudy (DOMINANT). REFUTED for CBPP.**
- Sleepstudy SuperLU: `approx_derivative` + `_dense_difference` cumtime **0.205 s of 0.381 s = 54 %** of the cProfile-instrumented fit; on CHOLMOD, **0.247 s of 0.640 s ≈ 39 %**. With 23 outer iters × 3 thetas + 1 = 92 obj evals, of which ~69 are forward-difference gradient probes, **3 of every 4 obj calls are gradient probes**.
- **Important caveat**: `reml_gradient` at `profiled_reml.py:452` raises *"reml_gradient is only implemented for the diagonal (intercept-only) path"*. Sleepstudy is random-slopes — there is no analytic gradient available today. Task `interlace-jjm7` ("default analytic gradient on") is therefore **larger than its title implies**: it requires extending `reml_gradient` to the general non-diagonal Λ. Either that, or switch this case to a derivative-free method (BOBYQA, which `fit_reml` already supports) — likely a much smaller change.
- CBPP runs **Nelder-Mead** in Phase 2 (no gradient at all), so this knob has zero effect there.

### 4. Λ / A11 / Z'WZ rebuilt as fresh CSC matrices on every objective call

- **Verdict: CONFIRMED — DOMINANT secondary cost on Sleepstudy, DOMINANT primary cost on CBPP.**
- Sleepstudy: `make_lambda` 124 calls / 0.149 s cumtime, `_build_A11` 124 calls / 0.099 s cumtime, `kron` 0.075 s, `block_diag` 0.037 s, `_compressed.__init__` 0.211 s cumtime. Sparse construction collectively: **~35-40 % of fit**.
- CBPP: `_compressed.__init__` 22 851 calls / **0.604 s cumtime = 42 %**, `get_index_dtype` 0.249 s = 17 %, `_matmul_sparse` 0.309 s = 21 %, `diags` 0.310 s, `tocsc/tocsr` collectively ~0.26 s. Sparse construction collectively: **~50-60 %** of fit.
- Pattern across θ is fixed; only values change. Caching the structural
  CSC pattern and mutating `data` in place (epic task `interlace-7bxn`)
  is the **single biggest pure-Python win available for both datasets**.

### 5. Sparse machinery overhead dominates at small q

- **Verdict: CONFIRMED.**
- For both Sleepstudy (q≈36) and CBPP (q=15), the bulk of self-time is
  `scipy.sparse._compressed.__init__`, `_sputils.get_index_dtype`,
  `_compressed.check_format`, `_compressed.prune` — all metadata
  housekeeping inside SciPy's sparse format machinery, not numerical
  work. Tier-3 task `interlace-72oc` (small-q dense fast path for A11)
  is well-motivated.

## New findings (not in the epic body)

### A. CBPP Phase-2 Nelder-Mead is **93 % of fit time**

- `glmm_laplace.py:1548` runs a Phase-2 Nelder-Mead over (θ, β) with
  `maxiter=2000`, `adaptive=True`, `xatol=fatol=1e-7`. Profile shows
  **403 `_laplace_objective_profiled` evals** vs 19 PIRLS (Phase 1).
  Nelder-Mead cumtime = 1.340 s of the 1.442 s instrumented fit.
- Phase-2 is documented to "avoid PIRLS multimodality with many
  observation-level random effects" — but CBPP has just q=15 and a
  trivially unimodal log-likelihood. **Skipping Phase 2 when Phase 1
  converged tightly, or tightening the maxiter/atol budget, looks
  like the highest-ROI single change for CBPP.** Worth a new
  child issue under `interlace-81ru`.

### B. CHOLMOD does not affect the GLMM/CBPP path at all

- All sparse solves in `glmm_laplace.py` use `spla.spsolve` directly,
  never the `_try_cholmod()`-aware `sparse_chol_logdet`. Task
  `interlace-98ip` (CHOLMOD as hard dep) and `interlace-w1sl` (reuse
  symbolic factor) need to **also touch `glmm_laplace.py`** to benefit
  CBPP. Currently they only help LMM.

### C. `_superlu.gstrf` is called 422× for CBPP

- 422 LU **factorisations** during a single CBPP fit. Reusing the
  symbolic factor (epic task `interlace-w1sl`) and/or batching RHS
  (`interlace-t3er`) could cut this dramatically — the per-call solve
  is cheap, but the factorise-then-solve cycle is repeated for every
  PIRLS step inside every Phase-2 evaluation.

## Ranked recommendations for ordering Tier-1 work

| Rank | Task                                    | Sleepstudy gain | CBPP gain | Rationale |
|---:  | ---                                     | ---             | ---       | --- |
| 1    | `interlace-7bxn` cache sparse patterns  | ~25-35 %        | ~30-50 %  | Largest confirmed share of self-time on both fits, no numerical risk. |
| 2    | **NEW**: cap Phase-2 Nelder-Mead on CBPP| —               | ~50-80 %  | 93 % of CBPP wall is Phase-2; budget is over-tuned for the easy multimodal case. |
| 3    | `interlace-jjm7` analytic gradient      | ~30-50 %        | 0 %       | But requires extending `reml_gradient` to non-diagonal Λ, or switching to BOBYQA. |
| 4    | `interlace-98ip` CHOLMOD hard dep       | ~20-25 %        | 0 %       | LMM only — confirm CBPP path still uses `spla.spsolve`. |
| 5    | `interlace-w1sl` reuse symbolic factor  | small           | small     | Only worthwhile *combined* with #1 + #4. |
| 6    | `interlace-t3er` batch multi-RHS spsolve| —               | ~3-5 %    | Smaller than expected; do for code quality. |
| 7    | `interlace-72oc` small-q dense fast path| small           | ~5-10 %   | After #1+#5: dense LAPACK on A11 may help at q≤20. |
| 8    | `interlace-tfrg` outer optimiser tols   | small           | small     | Not measured to be limiting in current data. |
