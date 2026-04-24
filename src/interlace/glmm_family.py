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
    """Protocol that all GLMM families must satisfy.

    All concrete family classes (e.g. :class:`BinomialFamily`,
    :class:`BetaFamily`) must implement the five methods below.  The protocol
    is ``runtime_checkable``, so ``isinstance(obj, GLMMFamily)`` works at
    runtime.

    Attributes
    ----------
    name : str
        Short string identifier for the family, e.g. ``"binomial"``,
        ``"beta"``.

    Methods
    -------
    link(mu)
        Apply the link function: map mean mu to the linear predictor eta.
    linkinv(eta)
        Apply the inverse link: map linear predictor eta to mean mu.
    mu_eta(eta)
        Derivative d(mu)/d(eta) of the inverse link.
    variance(mu)
        Variance function V(mu) used by PIRLS.
    dev_resids(y, mu, wt)
        Weighted unit deviance residuals.

    Examples
    --------
    >>> from interlace import BetaFamily
    >>> from interlace.glmm_family import GLMMFamily
    >>> fam = BetaFamily(phi=5.0)
    >>> isinstance(fam, GLMMFamily)
    True
    """

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


class NegativeBinomial2Family:
    """Negative Binomial (NB2) family with log link.

    The NB2 parameterisation uses variance function V(mu) = mu + mu^2 / theta,
    where *theta* (also called *size* or *k*) is the shape parameter controlling
    overdispersion.  As theta → ∞ the distribution converges to Poisson.

    Parameters
    ----------
    theta:
        Shape (overdispersion) parameter.  Must be positive.
        Default is 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from interlace.glmm_family import NegativeBinomial2Family
    >>> fam = NegativeBinomial2Family(theta=2.0)
    >>> fam.variance(np.array([1.0]))
    array([1.5])
    """

    name: str = "negativebinomial"

    def __init__(self, theta: float = 1.0) -> None:
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.theta = theta

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
        """Var(Y) = mu + mu^2 / theta."""
        return mu + mu**2 / self.theta

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """NB2 unit deviance residuals.

        d_i = 2 * wt * [y*log(y/mu) - (y + theta)*log((y + theta)/(mu + theta))]

        Uses the 0*log(0) = 0 convention.
        """
        theta = self.theta
        d = np.zeros_like(y, dtype=np.float64)
        pos = y > 0
        d[pos] += y[pos] * np.log(y[pos] / mu[pos])
        d -= (y + theta) * np.log((y + theta) / (mu + theta))
        return 2.0 * wt * d


class NegativeBinomial1Family:
    """Negative Binomial (NB1) family with log link.

    The NB1 parameterisation uses a **linear** mean-variance relationship:
    V(mu) = mu * (1 + alpha), where *alpha* > 0 is the overdispersion
    parameter.  As alpha → 0 the distribution converges to Poisson.

    Internally the NB1 is parameterised as NB(r, p) with
    observation-dependent r = mu / alpha and p = 1 / (1 + alpha).

    Parameters
    ----------
    alpha:
        Overdispersion parameter.  Must be positive.  Default is 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from interlace.glmm_family import NegativeBinomial1Family
    >>> fam = NegativeBinomial1Family(alpha=2.0)
    >>> fam.variance(np.array([1.0]))
    array([3.])
    """

    name: str = "negativebinomial1"

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

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
        """Var(Y) = mu * (1 + alpha)."""
        return mu * (1.0 + self.alpha)

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """NB1 unit deviance residuals.

        d_i = 2 * (ll_sat_i - ll_fit_i) where both use NB1 log-pmf.

        For y > 0:
          d_i = 2 * [lgamma(y + y/a) - lgamma(y/a)
                     - lgamma(y + mu/a) + lgamma(mu/a)
                     - (y - mu)/a * log(1 + a)]
        For y = 0:
          d_i = 2 * (mu/a) * log(1 + a)

        Uses the 0*log(0) = 0 convention.
        """
        from scipy.special import gammaln

        alpha = self.alpha
        d = np.zeros_like(y, dtype=np.float64)
        pos = y > 0
        if np.any(pos):
            yp, mp = y[pos], mu[pos]
            d[pos] = (
                gammaln(yp + yp / alpha)
                - gammaln(yp / alpha)
                - gammaln(yp + mp / alpha)
                + gammaln(mp / alpha)
                - (yp - mp) / alpha * np.log(1.0 + alpha)
            )
        zero = y == 0
        if np.any(zero):
            d[zero] = mu[zero] / alpha * np.log(1.0 + alpha)
        return 2.0 * wt * d


class ZeroInflatedNB2Family:
    """Zero-inflated Negative Binomial (NB2) family with log link.

    A mixture model: with probability *pi* the observation is a structural
    zero, and with probability *(1 - pi)* it follows NB2(mu, theta).

    For PIRLS purposes the link, variance, and mu_eta operate on the
    **count component** only (identical to :class:`NegativeBinomial2Family`).
    The zero-inflation probability is stored but handled separately in the
    likelihood (see ``_conditional_loglik``).

    Parameters
    ----------
    theta:
        Shape (overdispersion) parameter for the NB2 count component.
        Must be positive.  Default is 1.0.
    pi:
        Zero-inflation probability.  Must be in [0, 1).  Default is 0.0
        (no zero-inflation, equivalent to plain NB2).

    Examples
    --------
    >>> from interlace.glmm_family import ZeroInflatedNB2Family
    >>> fam = ZeroInflatedNB2Family(theta=2.0, pi=0.3)
    >>> fam.pi
    0.3
    >>> fam.theta
    2.0
    """

    name: str = "zeroinflated_negativebinomial"

    def __init__(self, theta: float = 1.0, pi: float = 0.0) -> None:
        if theta <= 0:
            raise ValueError("theta must be positive")
        if pi < 0 or pi >= 1:
            raise ValueError("pi must be in [0, 1)")
        self.theta = theta
        self.pi = pi

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
        """Var(Y) = mu + mu^2 / theta (count component)."""
        return mu + mu**2 / self.theta

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """NB2 unit deviance residuals (count component).

        d_i = 2 * wt * [y*log(y/mu) - (y + theta)*log((y + theta)/(mu + theta))]

        Uses the 0*log(0) = 0 convention.
        """
        theta = self.theta
        d = np.zeros_like(y, dtype=np.float64)
        pos = y > 0
        d[pos] += y[pos] * np.log(y[pos] / mu[pos])
        d -= (y + theta) * np.log((y + theta) / (mu + theta))
        return 2.0 * wt * d


class ZeroInflatedPoissonFamily:
    """Zero-inflated Poisson family with log link.

    A mixture model: with probability *pi* the observation is a structural
    zero, and with probability *(1 - pi)* it follows Poisson(mu).

    For PIRLS purposes the link, variance, and mu_eta operate on the
    **count component** only (identical to :class:`PoissonFamily`).
    The zero-inflation probability is stored but handled separately in the
    likelihood (see ``_conditional_loglik``).

    Parameters
    ----------
    pi:
        Zero-inflation probability.  Must be in [0, 1).  Default is 0.0
        (no zero-inflation, equivalent to plain Poisson).

    Examples
    --------
    >>> from interlace.glmm_family import ZeroInflatedPoissonFamily
    >>> fam = ZeroInflatedPoissonFamily(pi=0.2)
    >>> fam.pi
    0.2
    """

    name: str = "zeroinflated_poisson"

    def __init__(self, pi: float = 0.0) -> None:
        if pi < 0 or pi >= 1:
            raise ValueError("pi must be in [0, 1)")
        self.pi = pi

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
        """Var(Y) = mu (count component)."""
        return mu.copy()

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Poisson unit deviance residuals (count component).

        d_i = 2 * wt * [y*log(y/mu) - (y - mu)]

        Uses the 0*log(0) = 0 convention.
        """
        d = -(y - mu)
        pos = y > 0
        d[pos] += y[pos] * np.log(y[pos] / mu[pos])
        return 2.0 * wt * d


class BetaFamily:
    """Beta family with logit link.

    The Beta distribution is parameterised by mean *mu* in (0, 1) and precision
    *phi* > 0.  The shape parameters are a = mu * phi, b = (1 - mu) * phi,
    giving variance V(mu) = mu * (1 - mu) / (1 + phi).

    Parameters
    ----------
    phi:
        Precision parameter.  Must be positive.  Default is 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from interlace.glmm_family import BetaFamily
    >>> fam = BetaFamily(phi=5.0)
    >>> fam.variance(np.array([0.5]))
    array([0.04166667])
    """

    name: str = "beta"

    def __init__(self, phi: float = 1.0) -> None:
        if phi <= 0:
            raise ValueError("phi must be positive")
        self.phi = phi

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
        """Var(Y) = mu * (1 - mu) / (1 + phi)."""
        return mu * (1.0 - mu) / (1.0 + self.phi)

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Beta unit deviance residuals.

        d_i = 2 * wt * [log f(y; y, phi) - log f(y; mu, phi)]

        where f is the Beta density with a = mean * phi, b = (1 - mean) * phi.
        The lgamma(phi) terms cancel in the saturated - fitted difference.
        """
        from scipy.special import gammaln

        phi = self.phi
        # Saturated model: mean = y
        a_sat = y * phi
        b_sat = (1.0 - y) * phi
        # Fitted model: mean = mu
        a_fit = mu * phi
        b_fit = (1.0 - mu) * phi

        # log f(y; mean, phi) = lgamma(phi) - lgamma(a) - lgamma(b)
        #                       + (a-1)*log(y) + (b-1)*log(1-y)
        # The lgamma(phi) and the -log(y) -log(1-y) terms cancel in the diff.
        ll_sat = (
            -gammaln(a_sat)
            - gammaln(b_sat)
            + a_sat * np.log(y)
            + b_sat * np.log(1.0 - y)
        )
        ll_fit = (
            -gammaln(a_fit)
            - gammaln(b_fit)
            + a_fit * np.log(y)
            + b_fit * np.log(1.0 - y)
        )

        return np.asarray(2.0 * wt * (ll_sat - ll_fit), dtype=np.float64)


class ZeroOneInflatedBetaFamily:
    """Zero/one-inflated Beta family with logit link.

    A mixture model:
      - With probability *p0*, Y = 0 (point mass at zero)
      - With probability *p1*, Y = 1 (point mass at one)
      - With probability *(1 - p0 - p1)*, Y ~ Beta(mu, phi)

    For PIRLS purposes the link, variance, and mu_eta operate on the
    **Beta component** only (identical to :class:`BetaFamily`).
    The inflation probabilities are stored but handled separately in the
    likelihood (see ``_conditional_loglik``).

    Parameters
    ----------
    phi:
        Precision parameter for the Beta component.  Must be positive.
        Default is 1.0.
    p0:
        Zero-inflation probability.  Must be in [0, 1).  Default is 0.0.
    p1:
        One-inflation probability.  Must be in [0, 1).  Default is 0.0.

    Examples
    --------
    >>> from interlace.glmm_family import ZeroOneInflatedBetaFamily
    >>> fam = ZeroOneInflatedBetaFamily(phi=5.0, p0=0.1, p1=0.05)
    >>> fam.p0, fam.p1, fam.phi
    (0.1, 0.05, 5.0)
    """

    name: str = "zerooneinflated_beta"

    def __init__(self, phi: float = 1.0, p0: float = 0.0, p1: float = 0.0) -> None:
        if phi <= 0:
            raise ValueError("phi must be positive")
        if p0 < 0 or p0 >= 1:
            raise ValueError("p0 must be in [0, 1)")
        if p1 < 0 or p1 >= 1:
            raise ValueError("p1 must be in [0, 1)")
        if p0 + p1 >= 1:
            raise ValueError("p0 + p1 must be < 1")
        self.phi = phi
        self.p0 = p0
        self.p1 = p1

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
        """Var(Y) = mu * (1 - mu) / (1 + phi)  (Beta component)."""
        return mu * (1.0 - mu) / (1.0 + self.phi)

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Beta unit deviance residuals (Beta component).

        Identical to :meth:`BetaFamily.dev_resids`.
        """
        from scipy.special import gammaln

        phi = self.phi
        a_sat = y * phi
        b_sat = (1.0 - y) * phi
        a_fit = mu * phi
        b_fit = (1.0 - mu) * phi

        ll_sat = (
            -gammaln(a_sat)
            - gammaln(b_sat)
            + a_sat * np.log(y)
            + b_sat * np.log(1.0 - y)
        )
        ll_fit = (
            -gammaln(a_fit)
            - gammaln(b_fit)
            + a_fit * np.log(y)
            + b_fit * np.log(1.0 - y)
        )

        return np.asarray(2.0 * wt * (ll_sat - ll_fit), dtype=np.float64)


class GammaFamily:
    """Gamma family with log link (default) or inverse link.

    The Gamma distribution is parameterised by mean *mu* > 0 and shape
    parameter *shape* > 0.  The variance function is V(mu) = mu^2,
    making it suitable for positive continuous responses with variance
    proportional to the square of the mean.

    Parameters
    ----------
    link:
        Link function.  ``"log"`` (default) or ``"inverse"``.
    shape:
        Shape parameter (also called *k* or *alpha*).  Must be positive.
        Default is 1.0.  As shape → ∞ the distribution concentrates
        around the mean.

    Examples
    --------
    >>> import numpy as np
    >>> from interlace.glmm_family import GammaFamily
    >>> fam = GammaFamily(link="log", shape=5.0)
    >>> fam.variance(np.array([2.0]))
    array([4.])
    """

    name: str = "gamma"

    def __init__(self, link: str = "log", shape: float = 1.0) -> None:
        if link not in ("log", "inverse"):
            raise ValueError("link must be 'log' or 'inverse'")
        if shape <= 0:
            raise ValueError("shape must be positive")
        self._link = link
        self.shape = shape

    def link(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._link == "log":
            return np.log(mu)
        # inverse link: eta = 1/mu
        return 1.0 / mu

    def linkinv(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._link == "log":
            return np.exp(np.clip(eta, -_EXP_MAX, _EXP_MAX))
        # inverse link: mu = 1/eta, eta must be > 0
        return 1.0 / np.maximum(eta, _LOG_EPS)

    def mu_eta(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._link == "log":
            return self.linkinv(eta)
        # inverse link: d(1/eta)/d(eta) = -1/eta^2
        eta_safe = np.maximum(eta, _LOG_EPS)
        return -1.0 / eta_safe**2

    def variance(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Var(Y) = mu^2 (up to dispersion)."""
        return mu**2

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Gamma unit deviance: 2 * wt * [-log(y/mu) + (y - mu)/mu]."""
        return 2.0 * wt * (-np.log(y / mu) + (y - mu) / mu)


class HurdlePoissonFamily:
    """Hurdle (truncated) Poisson family with log link.

    A hurdle model separates the zero/non-zero process:
      - With probability *pi* the observation is a structural zero.
      - With probability *(1 - pi)* it follows a zero-truncated Poisson(mu).

    For PIRLS purposes the link, variance, and mu_eta operate on the
    **count component** mean (identical to :class:`PoissonFamily`).
    The hurdle probability is stored but handled separately in the
    likelihood and PIRLS working weights.

    Parameters
    ----------
    pi:
        Structural-zero probability.  Must be in [0, 1).  Default is 0.0.

    Examples
    --------
    >>> from interlace.glmm_family import HurdlePoissonFamily
    >>> fam = HurdlePoissonFamily(pi=0.3)
    >>> fam.pi
    0.3
    """

    name: str = "hurdle_poisson"

    def __init__(self, pi: float = 0.0) -> None:
        if pi < 0 or pi >= 1:
            raise ValueError("pi must be in [0, 1)")
        self.pi = pi

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
        """Var(Y) = mu (count component)."""
        return mu.copy()

    def dev_resids(
        self,
        y: NDArray[np.float64],
        mu: NDArray[np.float64],
        wt: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Poisson unit deviance residuals (count component).

        d_i = 2 * wt * [y*log(y/mu) - (y - mu)]

        Uses the 0*log(0) = 0 convention.
        """
        d = -(y - mu)
        pos = y > 0
        d[pos] += y[pos] * np.log(y[pos] / mu[pos])
        return 2.0 * wt * d


# ---------------------------------------------------------------------------
# Resolver: string | GLMMFamily → GLMMFamily
# ---------------------------------------------------------------------------

_FAMILIES: dict[
    str,
    type[
        BinomialFamily
        | PoissonFamily
        | GaussianFamily
        | NegativeBinomial2Family
        | NegativeBinomial1Family
        | ZeroInflatedNB2Family
        | ZeroInflatedPoissonFamily
        | BetaFamily
        | ZeroOneInflatedBetaFamily
        | GammaFamily
        | HurdlePoissonFamily
    ],
] = {
    "binomial": BinomialFamily,
    "poisson": PoissonFamily,
    "gaussian": GaussianFamily,
    "negativebinomial": NegativeBinomial2Family,
    "negativebinomial1": NegativeBinomial1Family,
    "zeroinflated_negativebinomial": ZeroInflatedNB2Family,
    "zeroinflated_poisson": ZeroInflatedPoissonFamily,
    "beta": BetaFamily,
    "zerooneinflated_beta": ZeroOneInflatedBetaFamily,
    "gamma": GammaFamily,
    "hurdle_poisson": HurdlePoissonFamily,
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
