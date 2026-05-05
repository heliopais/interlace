# Simulation and Bootstrap Guide

`interlace` provides two functions for simulation-based inference on fitted
mixed models:

- **`simulate()`** — draw new response vectors from the fitted model,
  using the same random-effect structure and parameter estimates
- **`bootMer()`** — run a full parametric bootstrap: simulate, refit, and
  collect any statistic you define

Together these cover three common workflows: **bootstrap standard errors and
confidence intervals**, **power analysis**, and **simulation-based model
checking** (posterior predictive checks).

---

## `simulate()` — drawing from the fitted model

`simulate()` draws response vectors from the fitted conditional model:

```
y* = X β + Z u* + ε*
```

where `u* ~ N(0, σ² ΛΛᵀ)` and `ε* ~ N(0, σ² Iₙ)`.

```python
import interlace
import numpy as np

result = interlace.fit("score ~ time + treatment", data=df, groups="subject")

# Draw 500 simulated response vectors (shape: n × 500)
y_sim = interlace.simulate(result, nsim=500, seed=42)
```

The return value is an `(n, nsim)` array — each column is one draw from the
model's predictive distribution. A single draw (`nsim=1`) gives an `(n, 1)`
array.

### When to use `simulate()` directly

`simulate()` is the building block for the two higher-level workflows below,
but it is also useful on its own when you want full control over the
simulated draws — for example, to compute a custom summary across replicates
without the overhead of refitting the model.

---

## Bootstrap standard errors and CIs with `bootMer()`

`bootMer()` runs a parametric bootstrap: for each of `B` replicates it
simulates `y*`, refits the model on the simulated data, and records a
statistic vector. The default statistic collects
`[fixed effects..., σ, θ...]`.

```python
boot = interlace.bootMer(result, B=500, seed=42, show_progress=True)

# Boot SEs for each parameter
se = boot.estimates.std(axis=0)
print(se)

# 95% percentile CIs (shape: p × 2, columns are [lower, upper])
ci = boot.ci(level=0.95)
print(ci)
```

`boot.estimates` has shape `(B, p)` where `p` is the length of the statistic
vector.

### Custom statistic

Pass any callable `(CrossedLMEResult) → np.ndarray` as the `statistic`
argument:

```python
def fixed_effects_only(r):
    return np.asarray(r.fe_params)

boot = interlace.bootMer(result, statistic=fixed_effects_only, B=500, seed=42)

# Columns: one per fixed-effect coefficient
print(boot.estimates.shape)       # (500, p)
print(boot.ci())                  # (p, 2)
```

### Bootstrap CIs vs Wald CIs

Bootstrap CIs are asymmetry-aware: if the sampling distribution of a
variance component is right-skewed, the percentile CI will correctly be
wider on the upper end. Wald CIs from `result.conf_int()` assume
approximate normality, which is reasonable for fixed effects but can be
misleading for variance components and their ratios.

Use `bootMer()` when:

- Reporting CIs for **variance components or ICC** (right-skewed sampling
  distributions)
- The sample is small and normality of the fixed-effect estimators is
  in doubt
- You want a **custom statistic** with no closed-form SE (e.g. a ratio of
  coefficients, a predicted difference between two groups)

---

## Power analysis

The parametric bootstrap is the standard approach for power analysis in
mixed models, because closed-form power formulas assume balanced designs
and ignore the variability introduced by estimating random effects.

The recipe:

1. Specify the data-generating model (effect sizes and variance components).
2. Simulate a dataset from that model.
3. Fit the model and test the effect of interest.
4. Repeat many times and count how often the test is significant.

```python
import numpy as np
import pandas as pd
import interlace

rng = np.random.default_rng(2024)

def power_lmm(
    n_subjects: int,
    n_obs: int,
    beta_treatment: float,
    sigma_subject: float = 1.0,
    sigma_resid: float = 1.0,
    B: int = 500,
    alpha: float = 0.05,
) -> float:
    """Estimated power for a two-group treatment effect in a random-intercept LMM."""
    n = n_subjects * n_obs
    significant = 0

    for _ in range(B):
        # Simulate data from the assumed model
        subject = np.repeat(np.arange(n_subjects), n_obs)
        treatment = (subject % 2).astype(float)          # half treated
        u = rng.normal(0, sigma_subject, n_subjects)     # random intercepts
        eps = rng.normal(0, sigma_resid, n)
        y = beta_treatment * treatment + u[subject] + eps

        df = pd.DataFrame({"y": y, "treatment": treatment, "subject": subject})
        result = interlace.fit("y ~ treatment", data=df, groups="subject")

        # Test: is the treatment effect significant at level alpha?
        pval = result.pvalues["treatment"]
        if pval < alpha:
            significant += 1

    return significant / B


power = power_lmm(
    n_subjects=40,
    n_obs=5,
    beta_treatment=0.5,
)
print(f"Estimated power: {power:.2f}")
```

### Varying sample size

Wrap the function in a grid search to find the required `n`:

```python
results = {}
for n_subjects in [20, 30, 40, 60, 80]:
    results[n_subjects] = power_lmm(n_subjects=n_subjects, n_obs=5,
                                    beta_treatment=0.5, B=200)
    print(f"n={n_subjects}: power={results[n_subjects]:.2f}")
```

### Tips for power analysis

- **Use realistic variance components.** Power is very sensitive to the
  ICC (ratio of between-subject variance to total variance). If you have
  pilot data, plug its variance component estimates in rather than
  guessing.
- **Match your analysis model to your data-generating model.** If the true
  model has random slopes but you power for random intercepts only, your
  power estimate will be optimistic.
- **Report the assumed effect size and variance.** Power results are only
  interpretable relative to the assumed parameters.

---

## Simulation-based model checking

A posterior predictive check asks: do the data look like they could have
come from the fitted model? This is a useful complement to residual plots,
especially for catching systematic misfit (e.g. overdispersion, heavy
tails, floor effects).

### Checking distributional assumptions

```python
import matplotlib.pyplot as plt

result = interlace.fit("score ~ time + treatment", data=df, groups="subject")

# Draw 200 simulated response vectors
y_sim = interlace.simulate(result, nsim=200, seed=0)

# Overlay observed and simulated densities
fig, ax = plt.subplots()
for j in range(y_sim.shape[1]):
    ax.hist(y_sim[:, j], bins=30, alpha=0.02, color="steelblue", density=True)
ax.hist(df["score"], bins=30, alpha=0.7, color="black", density=True, label="Observed")
ax.set_xlabel("score")
ax.legend()
plt.show()
```

If the observed histogram consistently falls in the tails of the simulated
distribution, the model is misspecified in some way (e.g., skewed
residuals, outliers, incorrect link function for a GLMM).

### Checking group-level summaries

```python
# Compare observed vs simulated group means
observed_group_means = df.groupby("subject")["score"].mean().values

sim_group_means = np.array([
    df.assign(score=y_sim[:, j]).groupby("subject")["score"].mean().values
    for j in range(y_sim.shape[1])
])  # shape: (nsim, n_subjects)

# 95% predictive interval for each group's mean
lo = np.percentile(sim_group_means, 2.5, axis=0)
hi = np.percentile(sim_group_means, 97.5, axis=0)

# How many observed group means fall outside the interval?
outside = np.mean((observed_group_means < lo) | (observed_group_means > hi))
print(f"Fraction of groups outside 95% PI: {outside:.2f}")
```

A well-calibrated model should have roughly 5% of group means outside the
95% predictive interval. Substantially more than 5% suggests the random
effect variance is underestimated.

### Checking a scalar summary statistic (T-test)

For a single summary statistic `T(y)`, you can compute a simulation-based
p-value:

```python
# Example: kurtosis of the residuals
from scipy.stats import kurtosis

y_obs = np.asarray(df["score"])
T_obs = kurtosis(y_obs - result.fittedvalues)

T_sim = np.array([
    kurtosis(y_sim[:, j] - result.fittedvalues)
    for j in range(y_sim.shape[1])
])

p_ppc = np.mean(np.abs(T_sim) >= np.abs(T_obs))
print(f"PPC p-value (kurtosis): {p_ppc:.3f}")
```

A p-value near 0 or 1 indicates the observed statistic is extreme relative
to what the model predicts.

---

## Tips

- **Seed everything.** Pass an integer `seed` to both `simulate()` and
  `bootMer()` for reproducible results.
- **Progress bar.** `bootMer(..., show_progress=True)` uses `tqdm` if
  installed. Install it with `pip install tqdm`.
- **Speed.** Each `bootMer` replicate refits the model from scratch. For
  large datasets or many replicates, budget accordingly — 500 replicates
  on a model that takes 0.5 s to fit takes ~4 min.
- **Percentile CIs are not bias-corrected.** `BootResult.ci()` uses the
  raw percentile method. For small `B` or heavily skewed statistics, the
  BCa (bias-corrected accelerated) method is more accurate, but requires
  larger `B` to be stable.

---

## See also

- {doc}`api/fit` — the `fit()` function and its parameters
- {doc}`variance-inference` — profile CIs and Wald CIs for variance
  components
- {doc}`model-comparison` — AIC/BIC and likelihood ratio tests
