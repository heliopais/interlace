# Concepts: Linear mixed models

This page introduces the statistical ideas behind `interlace` — linear mixed models,
random effects, variance components, REML estimation, and BLUPs. No prior experience
with mixed models is assumed, though familiarity with ordinary linear regression helps.

---

## Why not ordinary regression?

Ordinary least squares (OLS) regression assumes that observations are **independent**
of each other. That assumption breaks down whenever data have a **grouping structure**:
students from the same school tend to be more similar to each other than to students
from a different school; items in a questionnaire elicit systematically different
response times regardless of condition; measurements from the same patient on different
days share a common baseline.

Ignoring this structure has two consequences:

1. **Standard errors are too small** — the model treats 100 observations from 10
   students as if they were 100 independent data points rather than 10 × 10 correlated
   ones. Significance tests become anti-conservative.
2. **Variance is misattributed** — variability that belongs to between-group differences
   leaks into the residual, inflating `σ²` and distorting effect estimates.

Mixed models address both problems by explicitly modelling the grouping structure.

---

## Fixed effects and random effects

A linear mixed model partitions the model into two parts:

**Fixed effects** capture the population-level relationships you care about — the
intercept, treatment conditions, continuous predictors. These are the same as
coefficients in ordinary regression.

**Random effects** capture the deviation of each group from the population mean. Rather
than estimating a separate intercept for every school (which would require many
parameters and overfit small groups), the model assumes group intercepts are drawn from
a common Normal distribution with mean zero and variance `σ²_group`. The variance
`σ²_group` is estimated from the data; individual group offsets are predicted
conditionally.

**Rule of thumb for choosing:** a factor is fixed if you want to make inferences about
its specific levels (treatment A vs treatment B); it is random if the levels are a
sample from a larger population and you want to generalise beyond the observed groups
(these 30 schools, these 50 items).

---

## Variance components

A mixed model with two crossed grouping factors decomposes the total variance in the
outcome into additive pieces:

```
σ²_total  =  σ²_group1  +  σ²_group2  +  σ²_residual
```

Each `σ²` is called a **variance component**:

| Component | Meaning |
|---|---|
| `σ²_group1` | How much groups in factor 1 differ from each other |
| `σ²_group2` | How much groups in factor 2 differ from each other |
| `σ²_residual` | Residual within-group variability |

Large `σ²_group` relative to `σ²_residual` means the grouping structure explains a
substantial fraction of the total variance — the groups are very different from each
other. The **intraclass correlation coefficient (ICC)** summarises this:

```
ICC_g  =  σ²_group / σ²_total
```

An ICC of 0.3 for schools means 30% of the variance in scores is attributable to
school-level differences.

In `interlace`, variance components are returned as `result.variance_components` — a
dict of `{group_col: σ²}`. The residual variance is `result.scale`.

---

## Nested vs crossed designs

The grouping structure of your data determines which type of model you need.

**Nested (hierarchical) design:** every level of factor B exists within exactly one
level of factor A. Students are nested within schools: each student belongs to one and
only one school.

```
school 1 → students A, B, C
school 2 → students D, E, F
school 3 → students G, H, I
```

Groups at the lower level are *not* shared across groups at the higher level. The
standard tool for this in Python is `statsmodels.MixedLM`.

**Crossed design:** every level of factor A appears with multiple levels of factor B,
and vice versa. Subjects responding to items in a reading-time study: each subject sees
many items, and each item is seen by many subjects.

```
           item 1  item 2  item 3  item 4
subject A    ✓       ✓              ✓
subject B    ✓               ✓      ✓
subject C            ✓       ✓
subject D    ✓       ✓       ✓
```

Both `subject` and `item` independently shift the outcome. Fitting a model with only
one of them leaves the other's variance in the residual, biasing inference. Neither
factor is nested inside the other — this is the defining feature of a **crossed**
design.

`interlace` targets crossed designs. For nested designs, `statsmodels.MixedLM` is the
right tool. See [For Python users](why-python.md) for a side-by-side comparison.

---

## REML: restricted maximum likelihood

Mixed models have two estimators in common use: **maximum likelihood (ML)** and
**restricted maximum likelihood (REML)**.

Plain ML treats fixed-effect coefficients as known when estimating variance components.
In practice they are estimated, and ignoring that estimation uncertainty causes ML to
**systematically underestimate variance components** — more so with smaller samples and
more predictors.

REML corrects for this by maximising a likelihood that integrates out the fixed effects
first (the "restricted" part), separating the estimation of variance from the estimation
of coefficients. The result is unbiased variance component estimates.

**When to use REML vs ML:**

| Situation | Estimator |
|---|---|
| Comparing models with the **same fixed effects** (different random structures) | REML |
| Comparing models with **different fixed effects** | ML (REML likelihoods are not comparable across different fixed structures) |
| Reporting final parameter estimates | REML |

`interlace` uses profiled REML by default — the same algorithm as R's `lme4::lmer()`.
Profiled REML collapses the optimisation to a low-dimensional problem over variance
parameters, which is faster and more numerically stable than optimising all parameters
jointly.

---

## BLUPs: best linear unbiased predictors

Once the model is fitted, we want to estimate the group-level deviations — how much
each school, each subject, each item shifts the outcome relative to the population mean.

These estimates are called **BLUPs** (Best Linear Unbiased Predictors), or sometimes
**conditional modes** or **empirical Bayes estimates**. They are not the same as
including a fixed dummy variable for each group:

- A dummy variable for each group gives an **unregularised** estimate — it uses only
  the data from that group.
- A BLUP **shrinks** the estimate toward zero in proportion to the group's sample size
  and `σ²_group`. Small or sparse groups shrink more; large, consistent groups shrink
  less.

This shrinkage is why BLUPs are useful for **prediction**: a new subject with only one
observation should not be assigned an extreme intercept. The BLUP for that subject
pulls toward the population mean, which produces better out-of-sample predictions.

In `interlace`:

```python
# Dict of {group_col → Series of BLUPs, indexed by group level}
result.random_effects["subject"]  # e.g. Series: s1→1.3, s2→-0.8, ...
```

An **unseen group level** at prediction time automatically receives a BLUP of zero
(pure population mean), as implemented in `result.predict(newdata=...)`.

---

## Glossary

| Term | Definition |
|---|---|
| **Fixed effect** | A population-level coefficient; inference focuses on the specific levels |
| **Random effect** | A group-level deviation assumed drawn from a Normal distribution |
| **Variance component** | The variance `σ²` of a random effect distribution |
| **ICC** | Intraclass correlation: proportion of total variance due to grouping |
| **Nested design** | Groups at one level exist entirely within a single group at the next level |
| **Crossed design** | Each level of factor A co-occurs with multiple levels of factor B |
| **REML** | Restricted maximum likelihood: unbiased estimator for variance components |
| **ML** | Maximum likelihood: biased for variance components but comparable across fixed structures |
| **Profiled REML** | REML where the optimisation is collapsed to variance parameters only |
| **BLUP** | Best Linear Unbiased Predictor: shrinkage estimate of a group's deviation |
| **Shrinkage** | Pulling individual group estimates toward the population mean |
| **Sparse Cholesky** | Matrix factorisation used internally for efficient REML computation |

---

## Where to go next

- [For Python / statsmodels users](why-python.md) — code-level comparison and when to use `interlace` vs `statsmodels`
- [For R / lme4 users](why-r.md) — formula syntax mapping and shared references
- [Quickstart](quickstart.md) — fit your first crossed random-intercepts model
- [Interpreting results](interpreting-results.md) — what to do with the output once the model is fitted
