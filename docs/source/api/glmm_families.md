# GLMM families

All response-distribution families available for `interlace.glmer()`. Each
family defines a link function, inverse link, variance function, and
deviance residuals — mirroring R's `family()` interface.

For the basic families (`"binomial"`, `"poisson"`, `"gaussian"`,
`"negativebinomial"`), pass a string to `glmer()`. For extended families,
pass an instance directly.

## GLMMFamily protocol

```{eval-rst}
.. autoclass:: interlace.GLMMFamily
   :members:
```

Any object implementing these five methods can be passed as the `family`
parameter to `glmer()`.

---

## Standard families (string shorthand)

These can be passed as strings to `glmer(family=...)`:

| String | Family | Link | Variance |
|--------|--------|------|----------|
| `"binomial"` | Binomial | logit | mu(1-mu) |
| `"poisson"` | Poisson | log | mu |
| `"gaussian"` | Gaussian | identity | 1 |
| `"negativebinomial"` | NB2 (theta=1) | log | mu + mu^2/theta |

---

## Extended families (pass instances)

### NegativeBinomial2Family

Quadratic mean-variance relationship: Var(Y) = mu + mu^2/theta.

```python
from interlace import NegativeBinomial2Family

result = interlace.glmer(
    ..., family=NegativeBinomial2Family(theta=2.0), ...
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | 1.0 | Shape parameter. Smaller = more overdispersion. |

### NegativeBinomial1Family

Linear mean-variance relationship: Var(Y) = mu * (1 + alpha).

```python
from interlace import NegativeBinomial1Family

result = interlace.glmer(
    ..., family=NegativeBinomial1Family(alpha=2.0), ...
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.0 | Overdispersion parameter. Must be positive. |

### BetaFamily

For continuous proportions (0, 1). Uses logit link with variance
mu(1-mu)/(1+phi).

```python
from interlace import BetaFamily

result = interlace.glmer(
    ..., family=BetaFamily(phi=5.0), ...
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phi` | 1.0 | Precision parameter. Larger = less variance. |

### GammaFamily

For positive continuous data. Supports log and inverse links.

```python
from interlace import GammaFamily

result = interlace.glmer(
    ..., family=GammaFamily(link="log"), ...
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `link` | `"log"` | Link function: `"log"` or `"inverse"`. |

### ZeroInflatedPoissonFamily

Mixture model: structural zeros with probability pi, Poisson counts
otherwise.

```python
from interlace import ZeroInflatedPoissonFamily

result = interlace.glmer(
    ..., family=ZeroInflatedPoissonFamily(pi=0.2), ...
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pi` | 0.1 | Zero-inflation probability. |

### ZeroInflatedNB2Family

Mixture model: structural zeros with probability pi, NB2 counts otherwise.

```python
from interlace import ZeroInflatedNB2Family

result = interlace.glmer(
    ..., family=ZeroInflatedNB2Family(pi=0.2, theta=2.0), ...
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pi` | 0.1 | Zero-inflation probability. |
| `theta` | 1.0 | NB2 shape parameter. |

### HurdlePoissonFamily

Two-part model: y=0 observations carry no count information (score and
Hessian are zero). The count component is Poisson.

```python
from interlace import HurdlePoissonFamily

result = interlace.glmer(
    ..., family=HurdlePoissonFamily(), ...
)
```

### ZeroOneInflatedBetaFamily

For proportions with exact 0s and/or 1s. Three-component mixture: point
mass at 0, point mass at 1, and Beta distribution on (0, 1).

```python
from interlace import ZeroOneInflatedBetaFamily

result = interlace.glmer(
    ..., family=ZeroOneInflatedBetaFamily(pi0=0.1, pi1=0.05, phi=5.0), ...
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pi0` | 0.1 | Probability of structural zero. |
| `pi1` | 0.05 | Probability of structural one. |
| `phi` | 1.0 | Beta precision parameter. |

---

## Summary table

| Family | Link | Variance | Use case |
|--------|------|----------|----------|
| Binomial | logit | mu(1-mu) | Binary, proportions |
| Poisson | log | mu | Counts |
| Gaussian | identity | 1 | Continuous (same as `fit()`) |
| NB2 | log | mu + mu^2/theta | Overdispersed counts (quadratic) |
| NB1 | log | mu(1+alpha) | Overdispersed counts (linear) |
| Beta | logit | mu(1-mu)/(1+phi) | Continuous proportions in (0,1) |
| Gamma | log/inverse | mu^2 | Positive continuous |
| ZIP | log | mu + pi*mu^2 | Counts with excess zeros |
| ZINB2 | log | NB2 + zero-inflation | Overdispersed counts with excess zeros |
| Hurdle Poisson | log | Poisson (y>0 only) | Counts with structural zeros |
| ZOIB | logit | Beta + point masses | Proportions with 0s and 1s |

## See also

- {doc}`glmer` — fitting GLMMs with these families
