"""GLMMFamily protocol and concrete family implementations for GLMM support.

Mirrors R's ``family()`` interface: link, linkinv, variance, dev_resids, mu_eta.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Upper bound for exp() to avoid overflow (exp(709) ≈ 8.2e307).
_EXP_MAX = 709.0
# Clamp for log(x) to avoid -inf.
_LOG_EPS = 1e-300


@runtime_checkable
class GLMMFamily(Protocol):
    """Protocol that all GLMM families must satisfy."""

    name: str

    def link(self, mu: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def linkinv(self, eta: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def mu_eta(self, eta: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def variance(self, mu: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


class BinomialFamily:
    """Binomial family with logit link."""

    name: str = "binomial"

    def link(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Logit: log(mu / (1 - mu))."""
        return np.log(mu / (1.0 - mu))

    def linkinv(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse logit (expit), numerically stable."""
        from scipy.special import expit

        return np.asarray(expit(eta), dtype=np.float64)

    def mu_eta(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        """d(linkinv)/d(eta) = mu * (1 - mu)."""
        mu = self.linkinv(eta)
        return mu * (1.0 - mu)

    def variance(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Var(Y) = mu * (1 - mu)."""
        return mu * (1.0 - mu)

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Unit deviance: 2 * wt * [y*log(y/mu) + (1-y)*log((1-y)/(1-mu))].

        Uses the 0*log(0) = 0 convention.
        """
        d = np.zeros_like(y, dtype=np.float64)
        pos = y > 0
        neg = y < 1
        d[pos] += y[pos] * np.log(y[pos] / mu[pos])
        d[neg] += (1.0 - y[neg]) * np.log((1.0 - y[neg]) / (1.0 - mu[neg]))
        return 2.0 * wt * d


class PoissonFamily:
    """Poisson family with log link."""

    name: str = "poisson"

    def link(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log link."""
        return np.log(mu)

    def linkinv(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        """exp(eta), clamped to avoid overflow."""
        return np.exp(np.clip(eta, -_EXP_MAX, _EXP_MAX))

    def mu_eta(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        """d(exp(eta))/d(eta) = exp(eta)."""
        return self.linkinv(eta)

    def variance(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Var(Y) = mu."""
        return mu.copy()

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Unit deviance: 2 * wt * [y*log(y/mu) - (y - mu)].

        Uses the 0*log(0) = 0 convention.
        """
        d = -(y - mu)  # the -(y-mu) term is always present
        pos = y > 0
        d[pos] += y[pos] * np.log(y[pos] / mu[pos])
        return 2.0 * wt * d


class GaussianFamily:
    """Gaussian family with identity link."""

    name: str = "gaussian"

    def link(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        return mu.copy()

    def linkinv(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        return eta.copy()

    def mu_eta(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(eta)

    def variance(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(mu)

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """wt * (y - mu)^2."""
        return wt * (y - mu) ** 2


# ---------------------------------------------------------------------------
# Resolver: string | GLMMFamily → GLMMFamily
# ---------------------------------------------------------------------------

_FAMILIES: dict[str, type[BinomialFamily | PoissonFamily | GaussianFamily]] = {
    "binomial": BinomialFamily,
    "poisson": PoissonFamily,
    "gaussian": GaussianFamily,
}


def resolve_family(family: str | GLMMFamily) -> GLMMFamily:
    """Convert a family name string to a ``GLMMFamily`` instance.

    If *family* is already a ``GLMMFamily`` instance, return it as-is.
    """
    if isinstance(family, str):
        cls = _FAMILIES.get(family.lower())
        if cls is None:
            raise ValueError(
                f"Unknown family '{family}'. Choose from: {sorted(_FAMILIES)}"
            )
        return cls()
    return family
