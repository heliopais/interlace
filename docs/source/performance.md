# Performance and scalability

This page covers the computational characteristics of `interlace`: time and memory
complexity, practical size limits, and guidance on when the optional CHOLMOD and BOBYQA
extras improve performance.

For installation instructions and the decision of *whether* to install these extras,
see [FAQ: solver choice](faq.md#solver-choice-cholmod-vs-default) and
[FAQ: optimizer choice](faq.md#optimizer-choice-bobyqa-vs-default).

---

## How fitting works

`interlace.fit()` uses **profiled REML**: the fixed effects are eliminated
analytically, reducing the optimisation to a search over variance parameters only. For
a model with `K` grouping factors the search space has `K + 1` parameters (one
variance per factor plus the residual variance), regardless of how many observations or
group levels the data contain.

Each evaluation of the REML criterion requires:

1. Forming the random-effect design matrix `Z` (sparse, `n × Σᵢ gᵢ` where `gᵢ` is the
   number of levels in factor `i`).
2. Factorising the updated covariance matrix — this is the dominant cost.
3. Solving a linear system to recover the fixed effects and BLUPs.

The number of REML evaluations is typically 30–150 for the default L-BFGS-B optimizer
and 50–200 for BOBYQA.

---

## Time complexity

| Step | Dense path (default) | Sparse path (CHOLMOD) |
|------|---------------------|-----------------------|
| Covariance factorisation | O(G³) per iteration | O(G · nnz) per iteration¹ |
| Fixed-effect solve | O(p² · n) | O(p² · n) |
| BLUP recovery | O(G · p) | O(G · p) |

`G = Σᵢ gᵢ` is the total number of random-effect levels across all grouping factors.
`nnz` is the number of non-zeros in the sparse factor.

**Key implication:** fitting time grows cubically with the total number of group levels
on the dense path. Doubling the number of groups (e.g. from 500 to 1 000) increases
factorisation cost roughly 8×. CHOLMOD reduces this to near-linear in the non-zero
structure of the problem.

¹ After symbolic analysis (done once), each numeric refactorisation reuses the sparsity
pattern, so the per-iteration cost is proportional to the fill-in of the factor, not G³.

---

## Memory complexity

| Object | Size |
|--------|------|
| Random-effect design matrix `Z` | O(n · K) — sparse, one non-zero per observation per factor |
| Dense covariance matrix | O(G²) |
| Sparse Cholesky factor | O(G · fill-in) — typically O(G log G) for crossed designs |

For most practical models `Z` is very sparse and fits easily in memory. The dense
covariance matrix becomes the bottleneck around G ≈ 5 000 group levels (≈ 200 MB for
float64); CHOLMOD reduces this significantly.

---

## Practical size limits

The table below gives indicative fitting times measured on a laptop-class machine
(Apple M-series, single thread). Actual times vary with observation count, number of
predictors, and convergence behaviour.

| Observations (n) | Total group levels (G) | Dense path | CHOLMOD path |
|:----------------:|:----------------------:|:----------:|:------------:|
| 1 000 | 50 | < 1 s | < 1 s |
| 10 000 | 200 | 1–3 s | 1–2 s |
| 50 000 | 500 | 5–15 s | 3–8 s |
| 100 000 | 1 000 | 30–90 s | 8–20 s |
| 500 000 | 5 000 | minutes | 30–90 s |
| 1 000 000 | 10 000 | not practical | 2–5 min |

These are rough orders of magnitude. Models with more fixed-effect predictors or more
variance components (more grouping factors) will be slower; well-conditioned models with
few iterations will be faster.

**Rule of thumb:** if G > 500, install CHOLMOD. If n > 100 000, profile your specific
model before deploying in a pipeline.

---

## When CHOLMOD helps most

CHOLMOD's advantage comes from exploiting the **sparsity of the crossed random-effect
structure**. The benefit is largest when:

- **G is large** — hundreds or thousands of group levels. Below ~200 total levels the
  overhead of sparse bookkeeping can outweigh the savings.
- **The design is unbalanced or sparse** — many (group level i, group level j)
  combinations are unobserved. A fully balanced design (every combination observed
  equally) produces a denser factor and narrows the gap.
- **You are refitting the same model structure repeatedly** — e.g. a bootstrap loop or
  a grid search over fixed-effect specifications. CHOLMOD amortises the symbolic
  analysis cost across fits.

CHOLMOD provides little benefit when:

- G < 200 (dense path is fast enough).
- The design is perfectly balanced (dense factor, minimal sparsity gain).
- The environment makes compiling SuiteSparse difficult (e.g. restricted CI images).

---

## When BOBYQA helps most

BOBYQA (gradient-free) does not directly speed up fitting — it typically uses *more*
REML evaluations than L-BFGS-B. Its advantage is **robustness**, not raw speed:

- Models where variance components approach zero (the gradient becomes unreliable near
  the boundary of the parameter space).
- Models that fail to converge with L-BFGS-B, requiring retries or manual rescaling.
- Matching lme4's default algorithm for numerical parity.

If L-BFGS-B converges cleanly, it is faster. Switch to BOBYQA when convergence is the
problem, not fitting time.

---

## Reducing fitting time in practice

### 1. Standardise predictors

The REML surface is more spherical when predictors are on similar scales. Standardising
continuous predictors typically reduces iteration counts by 30–50%:

```python
from sklearn.preprocessing import StandardScaler

df["x_scaled"] = StandardScaler().fit_transform(df[["x"]])
model = interlace.fit("y ~ x_scaled", data=df, groups="firm")
```

### 2. Install CHOLMOD for large models

```bash
pip install "interlace-lme[cholmod]"
```

No code change required — `interlace` detects and uses CHOLMOD automatically.

### 3. Reduce the number of variance parameters

Each additional grouping factor adds one variance parameter and increases G. If a
factor has a near-zero variance component (check `result.variance_components`), consider
dropping it:

```python
# Check whether a grouping factor contributes
for g, v in result.variance_components.items():
    icc = v / (sum(result.variance_components.values()) + result.scale)
    print(f"{g}: ICC = {icc:.3f}")
# If ICC < 0.01, the factor may not be worth the computational cost
```

### 4. Fit with ML for model comparison, then refit with REML

When comparing fixed-effect structures via likelihood-ratio tests, fit with `method="ML"`
(faster for multiple fits) and do a final `method="REML"` fit for reported estimates:

```python
# Fast comparison fits
m1 = interlace.fit("y ~ x1",      data=df, groups="g", method="ML")
m2 = interlace.fit("y ~ x1 + x2", data=df, groups="g", method="ML")

# Final reported model
best = interlace.fit("y ~ x1 + x2", data=df, groups="g", method="REML")
```

### 5. Profile before optimising

Use Python's `time` module or `cProfile` to find where fitting time goes before
installing extras or restructuring data:

```python
import time

t0 = time.perf_counter()
result = interlace.fit("y ~ x", data=df, groups=["g1", "g2"])
print(f"Fit took {time.perf_counter() - t0:.2f}s")
```

---

## Known limitations

- **Single-threaded** — `interlace` does not currently parallelise the REML
  optimisation. Parallelism is available at the model level: fitting multiple
  independent models concurrently with `concurrent.futures` or similar.
- **In-memory only** — the full dataset must fit in RAM. For very large datasets,
  consider pre-aggregating or sampling before fitting.
- **Two or three grouping factors** — while the API accepts any number of `groups`,
  practical fitting with more than three crossed factors is largely untested and may be
  slow or numerically unstable.
- **Balanced vs unbalanced** — highly unbalanced designs (some groups with 1–2
  observations) increase the condition number of the covariance matrix and may require
  BOBYQA or predictor rescaling for reliable convergence.

---

## Where to go next

- [FAQ](faq.md) — convergence troubleshooting, solver and optimizer switching
- [Installation](installation.md) — installing CHOLMOD and BOBYQA extras
- [Feature comparison](comparison.md) — how interlace compares to statsmodels and lme4 on supported structures
