# For Python / statsmodels users

`statsmodels` is an excellent statistical library. Its `MixedLM` class handles many
mixed-model use cases well — random intercepts, random slopes, and nested designs.
This page explains the one structural gap that `interlace` was built to fill:
**crossed random effects**.

## Nested vs. crossed grouping factors

In a simple mixed model, observations are grouped by a single factor and groups are
independent of each other. For example, students clustered within schools:

```
school 1 → students A, B, C
school 2 → students D, E, F
```

In a **crossed** design, every level of one factor appears with every level of another.
A classic example from psycholinguistics: subjects responding to items in a
reading-time experiment, where each subject sees a subset of items and each item is
seen by a subset of subjects:

```
          item 1   item 2   item 3
subject A    ✓        ✓
subject B             ✓        ✓
subject C    ✓                  ✓
```

Both `subject` and `item` are genuine sources of variance. A model that ignores either
one produces biased estimates for fixed effects and inflated residuals.

## What statsmodels can do

`statsmodels.MixedLM` handles nested designs naturally:

```python
import statsmodels.formula.api as smf

# One grouping factor — works perfectly
result = smf.mixedlm(
    "rt ~ condition",
    data=df,
    groups=df["subject"],
).fit()
```

This is appropriate when each observation belongs to exactly one group and groups are
independent.

## Where statsmodels hits a wall

`MixedLM` is architecturally group-based: it partitions the data into disjoint groups
along a single axis. The `groups` parameter accepts **a single 1-D array**. There is no
syntax for two independent grouping factors:

```python
# This raises an error — groups must be a single column
result = smf.mixedlm(
    "rt ~ condition",
    data=df,
    groups=df[["subject", "item"]],   # TypeError
).fit()
```

The statsmodels documentation explicitly suggests this approach ([`mixed_linear.rst`](https://www.statsmodels.org/stable/mixed_linear.html)):

> *"To include crossed random effects in a model, it is necessary to treat the entire
> dataset as a single group. The variance components arguments to the model can then be
> used to define models with various combinations of crossed and non-crossed random
> effects."*

In practice this means passing a constant `groups` vector and using `vc_formula` to
add variance components for each factor:

```python
import statsmodels.formula.api as smf

# Workaround: single group = the whole dataset
df["intercept"] = 1
result = smf.mixedlm(
    "rt ~ condition",
    data=df,
    groups=df["intercept"],
    vc_formula={"subject": "0 + C(subject)", "item": "0 + C(item)"},
).fit()
```

This runs without error, but there is a catch. The `vc_formula` path fits an
**independent variance components** model rather than a true mixed model with crossed
random intercepts. The optimiser works in a different parameterisation, convergence
diagnostics differ, and the resulting variance estimates are generally **not equivalent**
to the REML estimates from `lme4::lmer()` — especially in unbalanced designs.

## How interlace fills the gap

`interlace` implements profiled REML for crossed random intercepts from the ground up,
using the same sparse Cholesky parameterisation as `lme4`. The API is intentionally
close to `statsmodels`:

```python
import interlace

result = interlace.fit(
    "rt ~ condition",
    data=df,
    groups=["subject", "item"],   # both grouping factors, simultaneously
)

print(result.fe_params)           # fixed effects
print(result.variance_components) # sigma^2_subject, sigma^2_item, sigma^2_residual
print(result.random_effects)      # BLUPs for every subject and item level
```

The result object exposes the same attributes as `statsmodels.MixedLMResults`
(`fe_params`, `resid`, `fittedvalues`, `scale`, `random_effects`, `predict()`), so
it is a drop-in replacement in existing pipelines.

## Validation against lme4

Fixed-effect estimates from `interlace` match `lme4::lmer()` to within 1e-4 (absolute)
and variance components to within 5% (relative) on standard benchmarks. See
[Contributing](contributing.md#validation-against-lme4) for the full tolerance table
and the test suite that enforces it.

## What interlace does not cover

`interlace` targets **crossed random effects** for linear (Gaussian) outcomes.
If your model needs any of the following, `statsmodels.MixedLM` or another library
is the right choice:

- Generalised linear mixed models (Poisson, binomial outcomes)
- Nested designs with many levels of hierarchy

Random slopes (`(1 + x | g)` and `(1 + x || g)`) are supported as of v0.2.1 via
the `random=` parameter. See the [Random Slopes Guide](random-slopes.md) for syntax
and examples.

For a full feature comparison table, see [Comparison](comparison.md).
For a full mapping of lme4 formula syntax, see [For R / lme4 users](why-r.md).
