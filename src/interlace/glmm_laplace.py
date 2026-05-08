"""GLMM estimation via Laplace approximation (PIRLS).

Implements the penalized iteratively reweighted least squares (PIRLS)
algorithm from Bates et al. (2015) for generalized linear mixed models.

The Laplace approximation to the marginal log-likelihood is:

    log p(y|theta) ≈ log p(y|u_hat,beta_hat) - 0.5*u_hat'*Lambda'*Lambda*u_hat
                     - 0.5*log|L_theta|^2

where u_hat, beta_hat are found by the inner PIRLS loop and L_theta is
the Cholesky factor of the penalized system.

References
----------
Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015).
Fitting Linear Mixed-Effects Models Using lme4.
Journal of Statistical Software, 67(1), 1-48.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from interlace import profiled_reml as _profiled_reml
from interlace.formula import (
    groups_to_random_effects,
    parse_formula,
    parse_random_effects,
)
from interlace.glmm_family import (
    GaussianFamily,
    GLMMFamily,
    HurdlePoissonFamily,
    NegativeBinomial2Family,
    resolve_family,
)
from interlace.profiled_reml import (
    LambdaBuilder,
    _build_theta_bounds,
    _init_chol_factor,
    make_lambda,
    n_theta_for_spec,
    sparse_chol_logdet,
)
from interlace.sparse_z import build_joint_z_from_specs, group_array

if TYPE_CHECKING:
    from interlace.formula import RandomEffectSpec


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GLMMResult:
    """Result container for a fitted GLMM.

    Attributes
    ----------
    fe_params : pd.Series
        Fixed-effect coefficient estimates, indexed by predictor name.
    fe_bse : pd.Series
        Standard errors of the fixed-effect estimates.
    random_effects : dict[str, Any]
        Mapping from group label to BLUP vector (one entry per grouping factor).
    variance_components : dict[str, float]
        Estimated variance components for each random-effect grouping factor.
    theta : np.ndarray
        Raw variance-component parameter vector passed to the optimiser
        (relative Cholesky factors in the Lambda_theta parameterisation).
    converged : bool
        ``True`` if the optimiser reported successful convergence.
    nobs : int
        Number of observations used to fit the model.
    llf : float
        Laplace-approximated log-likelihood at the optimum.
    aic : float
        Akaike information criterion: ``-2 * llf + 2 * k``.
    bic : float
        Bayesian information criterion: ``-2 * llf + k * log(n)``.
    family : GLMMFamily
        Family object used for the conditional distribution.
    ngroups : dict[str, int]
        Number of levels for each grouping factor.
    scale : float
        Dispersion parameter (1.0 for binomial and Poisson families;
        estimated for Gaussian).
    fittedvalues : np.ndarray
        In-sample fitted values on the response scale, shape ``(n,)``.
    disp_params : pd.Series or None
        Dispersion sub-model coefficient estimates (log link), or ``None``
        when no ``dispformula`` was supplied.
    dispersion : np.ndarray or None
        Per-observation dispersion values ``exp(X_d @ delta)``, or ``None``
        when no ``dispformula`` was supplied.

    Examples
    --------
    >>> import numpy as np, pandas as pd, interlace
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"y": rng.binomial(1, 0.6, 200),
    ...                    "x": rng.normal(size=200),
    ...                    "g": rng.integers(0, 20, 200)})
    >>> result = interlace.glmer("y ~ x", df, family="binomial", groups="g")
    >>> result.fe_params
    Intercept    ...
    x            ...
    dtype: float64
    >>> result.converged
    True
    """

    fe_params: pd.Series
    fe_bse: pd.Series
    random_effects: dict[str, Any]
    variance_components: dict[str, float]
    theta: np.ndarray
    converged: bool
    nobs: int
    llf: float
    aic: float
    bic: float
    family: GLMMFamily
    ngroups: dict[str, int]
    scale: float  # dispersion (1.0 for binomial/Poisson)
    fittedvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    _formula: str = ""
    _group_cols: list[str] | None = None
    _eta: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # linear predictor (link scale)
    disp_params: pd.Series | None = None  # dispersion formula coefficients (log link)
    dispersion: np.ndarray | None = None  # per-observation dispersion values

    def predict(
        self,
        newdata: Any = None,
        *,
        type: str = "response",
        include_re: bool = True,
    ) -> np.ndarray:
        """Predict from a fitted GLMM.

        Parameters
        ----------
        newdata :
            DataFrame to predict on. If ``None``, returns in-sample
            predictions.
        type :
            ``"response"`` (default) returns predictions on the response
            scale (mu). ``"link"`` returns predictions on the linear
            predictor scale (eta).
        include_re :
            If ``True`` (default), include BLUPs for known group levels.
            If ``False``, return population-level (fixed-effects only)
            predictions.

        Returns
        -------
        np.ndarray of shape (n_obs,)
        """
        if type not in ("response", "link"):
            msg = f"type must be 'response' or 'link', got {type!r}"
            raise ValueError(msg)

        if newdata is None:
            eta = self._eta
            if not include_re:
                # Recompute FE-only eta from stored formula
                eta = self._fe_only_eta()
        else:
            eta = self._predict_newdata(newdata, include_re)

        if type == "link":
            return np.asarray(eta)
        return np.asarray(self.family.linkinv(eta))

    def _fe_only_eta(self) -> np.ndarray:
        """Return fixed-effects-only linear predictor for in-sample data."""
        # fittedvalues = linkinv(eta), eta = X@beta + Z@u
        # FE-only: X@beta = eta - Z@u
        # Since we don't store X separately, reconstruct from
        # eta and RE contributions. But simpler: we can use
        # the formula to rebuild X from stored data. However we
        # don't store the data. Instead, compute from eta and BLUPs.
        # For in-sample, the simplest approach: not supported without
        # storing X. Return eta minus RE contribution.
        #
        # Actually, we can approximate: fittedvalues on response scale
        # is linkinv(eta), and we have eta stored. We just need to
        # subtract the RE contribution. But we don't have Z stored.
        # For now, raise if data isn't available.
        msg = (
            "In-sample include_re=False requires newdata. "
            "Pass the original data explicitly."
        )
        raise ValueError(msg)

    def _predict_newdata(self, newdata: Any, include_re: bool) -> np.ndarray:
        """Compute linear predictor for new data."""
        import formulaic
        import narwhals as nw

        nw_new = nw.from_native(newdata, eager_only=True)

        # Build X from formula
        fe_formula = self._formula.split("~", 1)[1].strip()
        X_mm = formulaic.model_matrix(fe_formula, nw_new)
        mm_cols = list(X_mm.columns)
        mm_arr = np.asarray(X_mm)

        # Reorder/pad columns to match fe_params order
        fe_cols = list(self.fe_params.index)
        if mm_cols != fe_cols:
            n_obs = mm_arr.shape[0]
            col_lookup = {c: mm_arr[:, i] for i, c in enumerate(mm_cols)}
            mm_arr = np.column_stack(
                [col_lookup.get(c, np.zeros(n_obs)) for c in fe_cols]
            )

        eta = mm_arr @ np.asarray(self.fe_params)

        if not include_re or self._group_cols is None:
            return np.asarray(eta)

        # Add BLUP contributions
        for col in self._group_cols:
            if col not in nw_new.columns:
                continue
            blup_re = self.random_effects.get(col)
            if blup_re is None:
                continue
            col_vals = nw_new[col].to_numpy()

            if isinstance(blup_re, pd.DataFrame):
                predictors = list(blup_re.columns[1:])
                contrib = np.zeros(len(col_vals))
                for i, level in enumerate(col_vals):
                    if level not in blup_re.index:
                        continue
                    blup_vec = blup_re.loc[level].to_numpy(dtype=float)
                    z_row = np.array(
                        [1.0] + [float(nw_new[p].to_numpy()[i]) for p in predictors]
                    )
                    contrib[i] = blup_vec @ z_row
                eta = eta + contrib
            elif isinstance(blup_re, pd.Series):
                lookup = blup_re.to_dict()
                contrib = np.array([lookup.get(v, 0.0) for v in col_vals], dtype=float)
                eta = eta + contrib

        return np.asarray(eta)

    def summary(self) -> _GLMMSummary:
        """Return a human-readable summary of the fitted GLMM."""
        return _GLMMSummary(self)


class _GLMMSummary:
    """Human-readable summary of a fitted GLMM."""

    def __init__(self, result: GLMMResult) -> None:
        self._result = result

    def __str__(self) -> str:
        return self._render()

    def __repr__(self) -> str:
        return self._render()

    def _render(self) -> str:
        r = self._result
        lines: list[str] = []

        lines.append("Generalized linear mixed model fit by Laplace")
        lines.append(f"Family: {r.family.name}")
        lines.append(f"Formula: {r._formula}")
        lines.append("")

        # Fixed effects
        lines.append("Fixed effects:")
        fe_arr = np.asarray(r.fe_params)
        bse_arr = np.asarray(r.fe_bse)
        z_arr = fe_arr / bse_arr
        names = list(r.fe_params.index)
        header = f"  {'':20} {'Estimate':>12} {'Std. Error':>12} {'z value':>10}"
        lines.append(header)
        for name, est, se, zv in zip(names, fe_arr, bse_arr, z_arr, strict=True):
            lines.append(f"  {name:<20} {est:>12.4f} {se:>12.4f} {zv:>10.4f}")
        lines.append("")

        # Random effects
        lines.append("Random effects:")
        for grp, vc in r.variance_components.items():
            n_grp = r.ngroups[grp]
            sd = np.sqrt(vc)
            lines.append(f"  {grp:<15} Var: {vc:.6f}  SD: {sd:.6f}  (n={n_grp})")
        lines.append("")

        # Dispersion model coefficients
        if r.disp_params is not None:
            lines.append("Dispersion model coefficients (log link):")
            dp = r.disp_params
            dp_names = list(dp.index)
            dp_vals = np.asarray(dp)
            header_d = f"  {'':20} {'Estimate':>12}"
            lines.append(header_d)
            for name, val in zip(dp_names, dp_vals, strict=True):
                lines.append(f"  {name:<20} {val:>12.4f}")
            lines.append("")

        # Model fit
        groups_str = "; ".join(f"{g}: {n}" for g, n in r.ngroups.items())
        lines.append(f"Number of obs: {r.nobs}, groups: {groups_str}")
        lines.append(f"AIC: {r.aic:.2f}  BIC: {r.bic:.2f}  logLik: {r.llf:.2f}")
        status = "converged" if r.converged else "DID NOT CONVERGE"
        lines.append(f"Optimizer: {status}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PIRLS inner loop
# ---------------------------------------------------------------------------

_PIRLS_MAXITER = 100
_PIRLS_TOL = 1e-8
# Below this q, the dense PIRLS kernel is faster than the sparse one (the
# overhead of scipy.sparse construction dwarfs the actual factorisation work
# at small q).  Tracked under interlace-72oc; tune empirically against the
# CBPP / Sleepstudy bench.
_DENSE_Q_THRESHOLD = 200
_MU_EPS = 1e-10  # clamp mu away from 0 boundary


class _CholmodHandle:
    """Wrapper around an sksparse.cholmod factor with version-agnostic API.

    Exposes:
      - ``refactor(A)``: numeric-only refactorisation reusing the symbolic
        analysis from construction.
      - ``solve(b)``: solve A x = b using the most recently refactored A.
      - ``logdet()``: log|A|.

    Created by :func:`_make_cholmod_handle` once per fit; ``refactor``
    is called per inner objective evaluation.
    """

    __slots__ = ("factor", "api")

    def __init__(self, factor: Any, api: str) -> None:
        self.factor = factor
        self.api = api

    def refactor(self, A: sp.csc_matrix) -> None:
        if self.api == "new":
            self.factor.factorize(A)
        else:
            self.factor.cholesky(A)

    def solve(self, b: np.ndarray) -> np.ndarray:
        x = self.factor.solve(b, "A") if self.api == "new" else self.factor.solve_A(b)
        x = np.asarray(x)
        return x.squeeze() if b.ndim == 1 else x

    def logdet(self) -> float:
        return float(self.factor.logdet())


def _make_cholmod_handle(A0: sp.csc_matrix) -> _CholmodHandle | None:
    """Try to construct a CHOLMOD factor for the given A0 sparsity pattern.

    A0 should be a representative ``(WZs.T @ WZs) + I_q`` with full
    pattern (use theta=ones, W=ones).  Returns ``None`` if sksparse is
    unavailable or the factorisation fails.
    """
    # Look up via module attribute so tests can monkeypatch the
    # profiled_reml._try_cholmod symbol to simulate sksparse absence.
    cholmod = _profiled_reml._try_cholmod()
    if cholmod is None:
        return None
    factor, api = _init_chol_factor(cholmod, A0)
    if factor is None or api is None:
        return None
    return _CholmodHandle(factor, api)


def _scale_columns_csc(Z: sp.csc_matrix, col_factors: np.ndarray) -> sp.csc_matrix:
    """Right-multiply Z by diag(col_factors) without sparse matmul.

    Equivalent to ``Z @ sp.diags(col_factors, format='csc')``.  For a CSC
    matrix, slot ``s`` in column ``c`` is multiplied by ``col_factors[c]``.
    Reuses ``Z.indices`` / ``Z.indptr`` (only ``data`` is freshly allocated).
    """
    col_for_slot = np.repeat(np.arange(Z.shape[1]), np.diff(Z.indptr))
    out = sp.csc_matrix(
        (Z.data * col_factors[col_for_slot], Z.indices, Z.indptr),
        shape=Z.shape,
        copy=False,
    )
    out.indices = Z.indices
    out.indptr = Z.indptr
    return out


def _scale_rows_csc(M: sp.csc_matrix, row_factors: np.ndarray) -> sp.csc_matrix:
    """Left-multiply M by diag(row_factors) without sparse matmul.

    Equivalent to ``sp.diags(row_factors, format='csc') @ M``.  For a CSC
    matrix, ``indices`` are row indices, so we just scale ``.data`` by
    ``row_factors[indices]``.  Reuses ``M.indices`` / ``M.indptr``.
    """
    out = sp.csc_matrix(
        (M.data * row_factors[M.indices], M.indices, M.indptr),
        shape=M.shape,
        copy=False,
    )
    out.indices = M.indices
    out.indptr = M.indptr
    return out


def _clamp_mu(mu: np.ndarray, family: GLMMFamily) -> np.ndarray:
    """Clamp mu to valid range for the family."""
    if family.name in ("binomial", "beta"):
        return np.asarray(np.clip(mu, _MU_EPS, 1.0 - _MU_EPS))
    if family.name in (
        "poisson",
        "negativebinomial",
        "negativebinomial1",
        "zeroinflated_negativebinomial",
        "zeroinflated_poisson",
        "hurdle_poisson",
        "gamma",
    ):
        return np.asarray(np.maximum(mu, _MU_EPS))
    return mu


def _conditional_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
    family: GLMMFamily,
    phi: np.ndarray | None = None,
) -> float:
    """Compute the conditional log-likelihood sum_i log p(y_i | mu_i).

    For binomial (proportion y, trial count wt):
        ll_i = log(C(n_i, k_i)) + k_i*log(mu_i) + (n_i - k_i)*log(1 - mu_i)
        where k_i = y_i * n_i, n_i = wt_i.

    For Poisson (count y, wt=1):
        ll_i = y_i*log(mu_i) - mu_i - log(y_i!)

    For Gaussian with dispersion phi_i:
        ll_i = -0.5 * [log(phi_i) + (y_i - mu_i)^2 / phi_i] + const

    Parameters
    ----------
    phi : Per-observation dispersion vector.  ``None`` means dispersion = 1.
    """
    from scipy.special import gammaln

    if family.name == "binomial":
        n_trials = weights
        k = y * n_trials  # successes
        # log(C(n, k)) + k*log(mu) + (n-k)*log(1-mu)
        log_binom = gammaln(n_trials + 1) - gammaln(k + 1) - gammaln(n_trials - k + 1)
        mu_safe = _clamp_mu(mu, family)
        ll = log_binom + k * np.log(mu_safe) + (n_trials - k) * np.log(1.0 - mu_safe)
        return float(np.sum(ll))
    elif family.name == "poisson":
        mu_safe = np.maximum(mu, _MU_EPS)
        ll = y * np.log(mu_safe) - mu_safe - gammaln(y + 1)
        return float(np.sum(weights * ll))
    elif family.name == "negativebinomial":
        assert isinstance(family, NegativeBinomial2Family)
        # When phi is provided for NB2, it carries per-observation theta
        # (the shape/overdispersion parameter), matching glmmTMB convention.
        theta = phi if phi is not None else np.full_like(y, family.theta)
        mu_safe = np.maximum(mu, _MU_EPS)
        # NB2 log-likelihood:
        # ll_i = lgamma(y+theta) - lgamma(theta) - lgamma(y+1)
        #        + theta*log(theta) - theta*log(mu+theta)
        #        + y*log(mu) - y*log(mu+theta)
        ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta)
            - theta * np.log(mu_safe + theta)
            + y * np.log(mu_safe)
            - y * np.log(mu_safe + theta)
        )
        return float(np.sum(weights * ll))
    elif family.name == "negativebinomial1":
        from interlace.glmm_family import NegativeBinomial1Family

        assert isinstance(family, NegativeBinomial1Family)
        # When phi is provided, it overrides alpha per-observation.
        alpha = phi if phi is not None else np.full_like(y, family.alpha)
        mu_safe = np.maximum(mu, _MU_EPS)
        # NB1: r = mu/alpha (observation-dependent), p = 1/(1+alpha)
        r = mu_safe / alpha
        p = 1.0 / (1.0 + alpha)
        ll = (
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1)
            + r * np.log(p)
            + y * np.log(1.0 - p)
        )
        return float(np.sum(weights * ll))
    elif family.name == "gaussian":
        n = len(y)
        if phi is not None:
            # Heteroscedastic: -0.5 * sum[log(phi_i) + (y-mu)^2/phi_i] + const
            ll = -0.5 * np.sum(
                np.log(phi) + weights * (y - mu) ** 2 / phi
            ) - 0.5 * n * np.log(2.0 * np.pi)
        else:
            # Homoscedastic (phi = 1)
            ll = -0.5 * np.sum(weights * (y - mu) ** 2) - 0.5 * n * np.log(2.0 * np.pi)
        return float(ll)
    elif family.name == "zeroinflated_negativebinomial":
        from interlace.glmm_family import ZeroInflatedNB2Family

        assert isinstance(family, ZeroInflatedNB2Family)
        theta = phi if phi is not None else np.full_like(y, family.theta)
        pi = family.pi
        mu_safe = np.maximum(mu, _MU_EPS)

        # NB2 log-pmf for all observations
        nb2_ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (mu_safe + theta))
            + y * np.log(mu_safe / (mu_safe + theta))
        )

        if pi == 0.0:
            # No zero-inflation: identical to NB2
            ll = nb2_ll
        else:
            ll = np.empty_like(y, dtype=np.float64)
            zero = y == 0
            pos = ~zero
            # y=0: log[pi + (1-pi) * f_NB2(0|mu, theta)]
            ll[zero] = np.log(pi + (1 - pi) * np.exp(nb2_ll[zero]))
            # y>0: log(1-pi) + log f_NB2(y|mu, theta)
            ll[pos] = np.log(1 - pi) + nb2_ll[pos]

        return float(np.sum(weights * ll))
    elif family.name == "zeroinflated_poisson":
        from interlace.glmm_family import ZeroInflatedPoissonFamily

        assert isinstance(family, ZeroInflatedPoissonFamily)
        pi = family.pi
        mu_safe = np.maximum(mu, _MU_EPS)

        # Poisson log-pmf for all observations
        pois_ll = y * np.log(mu_safe) - mu_safe - gammaln(y + 1)

        if pi == 0.0:
            # No zero-inflation: identical to Poisson
            ll = pois_ll
        else:
            ll = np.empty_like(y, dtype=np.float64)
            zero = y == 0
            pos = ~zero
            # y=0: log[pi + (1-pi) * exp(-mu)]
            ll[zero] = np.log(pi + (1 - pi) * np.exp(-mu_safe[zero]))
            # y>0: log(1-pi) + log f_Pois(y|mu)
            ll[pos] = np.log(1 - pi) + pois_ll[pos]

        return float(np.sum(weights * ll))
    elif family.name == "hurdle_poisson":
        assert isinstance(family, HurdlePoissonFamily)
        pi = family.pi
        mu_safe = np.maximum(mu, _MU_EPS)

        # Zero-truncated Poisson log-pmf for all observations:
        # log f_trunc(y|mu) = y*log(mu) - mu - lgamma(y+1) - log(1-exp(-mu))
        pois_ll = y * np.log(mu_safe) - mu_safe - gammaln(y + 1)
        log_q = np.log(np.maximum(1.0 - np.exp(-mu_safe), _MU_EPS))
        trunc_ll = pois_ll - log_q

        if pi == 0.0:
            # No structural zeros: pure truncated Poisson
            ll = trunc_ll
        else:
            ll = np.empty_like(y, dtype=np.float64)
            zero = y == 0
            pos = ~zero
            # y=0: log(pi)
            ll[zero] = np.log(pi)
            # y>0: log(1-pi) + log f_trunc(y|mu)
            ll[pos] = np.log(1 - pi) + trunc_ll[pos]

        return float(np.sum(weights * ll))
    elif family.name == "gamma":
        from interlace.glmm_family import GammaFamily

        assert isinstance(family, GammaFamily)
        shape = phi if phi is not None else np.full_like(y, family.shape)
        mu_safe = np.maximum(mu, _MU_EPS)
        # Gamma log-pdf: (shape-1)*log(y) - shape*y/mu - shape*log(mu)
        #                + shape*log(shape) - lgamma(shape)
        ll = (
            (shape - 1.0) * np.log(y)
            - shape * y / mu_safe
            - shape * np.log(mu_safe)
            + shape * np.log(shape)
            - gammaln(shape)
        )
        return float(np.sum(weights * ll))
    elif family.name == "beta":
        from interlace.glmm_family import BetaFamily

        assert isinstance(family, BetaFamily)
        precision = phi if phi is not None else np.full_like(y, family.phi)
        mu_safe = np.clip(mu, _MU_EPS, 1.0 - _MU_EPS)
        y_safe = np.clip(y, _MU_EPS, 1.0 - _MU_EPS)
        a = mu_safe * precision
        b = (1.0 - mu_safe) * precision
        ll = (
            gammaln(precision)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * np.log(y_safe)
            + (b - 1.0) * np.log(1.0 - y_safe)
        )
        return float(np.sum(weights * ll))
    elif family.name == "zerooneinflated_beta":
        from interlace.glmm_family import ZeroOneInflatedBetaFamily

        assert isinstance(family, ZeroOneInflatedBetaFamily)
        precision = phi if phi is not None else np.full_like(y, family.phi)
        p0 = family.p0
        p1 = family.p1
        mu_safe = np.clip(mu, _MU_EPS, 1.0 - _MU_EPS)
        y_safe = np.clip(y, _MU_EPS, 1.0 - _MU_EPS)

        # Beta log-pdf for all observations
        a = mu_safe * precision
        b = (1.0 - mu_safe) * precision
        beta_ll = (
            gammaln(precision)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * np.log(y_safe)
            + (b - 1.0) * np.log(1.0 - y_safe)
        )

        if p0 == 0.0 and p1 == 0.0:
            # No inflation: identical to Beta
            ll = beta_ll
        else:
            ll = np.empty_like(y, dtype=np.float64)
            p_beta = 1.0 - p0 - p1
            zero = y == 0.0
            one = y == 1.0
            interior = ~zero & ~one
            # y=0: log[p0 + p_beta * f_Beta(0+|mu, phi)]
            if np.any(zero):
                ll[zero] = np.log(p0 + p_beta * np.exp(beta_ll[zero]))
            # y=1: log[p1 + p_beta * f_Beta(1-|mu, phi)]
            if np.any(one):
                ll[one] = np.log(p1 + p_beta * np.exp(beta_ll[one]))
            # 0 < y < 1: log(p_beta) + log f_Beta(y|mu, phi)
            if np.any(interior):
                ll[interior] = np.log(p_beta) + beta_ll[interior]

        return float(np.sum(weights * ll))
    else:
        # Fallback: use -0.5 * deviance (no normalizing constant)
        dev = float(np.sum(family.dev_resids(y, mu, weights)))
        return -0.5 * dev


def _glm_start(
    y: np.ndarray,
    X: np.ndarray,
    family: GLMMFamily,
    weights: np.ndarray,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Compute starting beta from a fixed-effects-only GLM (IRLS).

    Runs a few IRLS iterations without random effects to get a reasonable
    starting point for PIRLS.
    """
    n, p = X.shape
    _off = offset if offset is not None else np.zeros(n)

    # Initialize mu from y, with safety clamps
    if family.name == "binomial":
        mu = np.clip(y, 0.01, 0.99)
    elif family.name in (
        "poisson",
        "negativebinomial",
        "negativebinomial1",
        "zeroinflated_negativebinomial",
        "zeroinflated_poisson",
        "hurdle_poisson",
        "gamma",
    ):
        mu = np.maximum(y, 0.1)
    else:
        mu = y.copy()

    beta = np.zeros(p)
    for _ in range(25):
        eta = family.link(mu)
        mu_eta_val = family.mu_eta(eta)
        var_mu = family.variance(mu)
        w = weights * mu_eta_val**2 / var_mu
        # Working residual on the link scale, excluding offset
        z_w = (eta - _off) + (y - mu) / mu_eta_val

        WX = np.sqrt(w)[:, None] * X
        Wz = np.sqrt(w) * z_w
        try:
            beta_new = la.solve(WX.T @ WX, WX.T @ Wz, assume_a="pos")
        except la.LinAlgError:
            break
        eta = X @ beta_new + _off
        mu = family.linkinv(eta)
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    return beta


# ---------------------------------------------------------------------------
# ZI-adjusted PIRLS working quantities
# ---------------------------------------------------------------------------

_ZI_FAMILIES = frozenset(
    {"zeroinflated_negativebinomial", "zeroinflated_poisson", "hurdle_poisson"}
)


def _zi_pirls_weights(
    y: np.ndarray,
    mu: np.ndarray,
    weights: np.ndarray,
    family: GLMMFamily,
    offset: np.ndarray,
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ZI-adjusted PIRLS working weights and working response.

    For zero-inflated families, the standard PIRLS working weights
    (derived from count-component variance) are wrong because they ignore
    the zero-inflation mixture.  This function computes the score and
    negative observed Hessian of the *full* ZI conditional log-likelihood
    w.r.t. eta, then converts to (w, z_w) for the PIRLS linear system.

    Returns
    -------
    w : Working weights (n,), always positive.
    z_w : Working response (n,).
    """
    from interlace.glmm_family import ZeroInflatedNB2Family, ZeroInflatedPoissonFamily

    n = len(y)
    score_unit = np.zeros(n)
    neg_hess_unit = np.zeros(n)

    zero = y == 0
    pos = ~zero

    if isinstance(family, HurdlePoissonFamily):
        pi = family.pi

        # --- Positive observations: truncated Poisson score/hessian ---
        # Score: dl/deta = y - mu + mu*P0/q  where P0=exp(-mu), q=1-P0
        # Neg Hessian: mu*(1 - P0/q*(1 - mu/q))
        if np.any(pos):
            mu_p = mu[pos]
            y_p = y[pos]
            P0_p = np.exp(-mu_p)
            q_p = np.maximum(1.0 - P0_p, _MU_EPS)
            trunc_corr = mu_p * P0_p / q_p
            score_unit[pos] = y_p - mu_p + trunc_corr
            neg_hess_unit[pos] = mu_p - trunc_corr * (1.0 - mu_p / q_p)

        # --- Zero observations: no dependence on eta ---
        # score = 0, neg_hess = 0 (floored below to 1e-10)

    elif isinstance(family, ZeroInflatedNB2Family):
        theta = family.theta
        pi = family.pi

        # --- Positive observations: NB2 score/hessian (log link) ---
        if np.any(pos):
            mu_p = mu[pos]
            y_p = y[pos]
            score_unit[pos] = y_p - mu_p * (y_p + theta) / (mu_p + theta)
            neg_hess_unit[pos] = (y_p + theta) * theta * mu_p / (mu_p + theta) ** 2

        # --- Zero observations ---
        if np.any(zero):
            mu_z = mu[zero]
            if pi > 0:
                f0 = (theta / (mu_z + theta)) ** theta
                P0 = pi + (1 - pi) * f0
                r0 = (1 - pi) * f0 / P0

                score_unit[zero] = -r0 * theta * mu_z / (mu_z + theta)

                raw_hess = (
                    theta**2 * mu_z / (mu_z + theta) ** 2 * r0 * (1 - mu_z * (1 - r0))
                )
                # Floor: count-component Hessian at y=0, scaled down
                floor = theta**2 * mu_z / (mu_z + theta) ** 2 * 0.01
                neg_hess_unit[zero] = np.maximum(raw_hess, floor)
            else:
                # pi=0: pure NB2 at y=0
                mu_z = mu[zero]
                score_unit[zero] = -mu_z * theta / (mu_z + theta)
                neg_hess_unit[zero] = theta**2 * mu_z / (mu_z + theta) ** 2

    elif isinstance(family, ZeroInflatedPoissonFamily):
        pi = family.pi

        # --- Positive observations: Poisson score/hessian ---
        if np.any(pos):
            score_unit[pos] = y[pos] - mu[pos]
            neg_hess_unit[pos] = mu[pos]

        # --- Zero observations ---
        if np.any(zero):
            mu_z = mu[zero]
            if pi > 0:
                f0 = np.exp(-mu_z)
                P0 = pi + (1 - pi) * f0
                r0 = (1 - pi) * f0 / P0

                score_unit[zero] = -r0 * mu_z

                raw_hess = r0 * mu_z * (1 - mu_z * (1 - r0))
                floor = mu_z * 0.01
                neg_hess_unit[zero] = np.maximum(raw_hess, floor)
            else:
                # pi=0: pure Poisson at y=0
                score_unit[zero] = -mu[zero]
                neg_hess_unit[zero] = mu[zero]
    else:
        raise ValueError(f"_zi_pirls_weights called with non-ZI family: {family.name}")

    # Ensure positivity for numerical stability
    neg_hess_unit = np.maximum(neg_hess_unit, 1e-10)

    w = weights * neg_hess_unit
    z_w = (eta - offset) + score_unit / neg_hess_unit

    return w, z_w


def _pirls(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    u0: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    lambda_builder: LambdaBuilder | None = None,
    cholmod_handle: _CholmodHandle | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """PIRLS dispatcher: dense fast path for q <= _DENSE_Q_THRESHOLD,
    otherwise the original sparse implementation (interlace-72oc)."""
    if Z.shape[1] <= _DENSE_Q_THRESHOLD:
        return _pirls_dense(
            y,
            X,
            Z,
            family,
            theta,
            specs,
            n_levels,
            weights,
            u0=u0,
            beta0=beta0,
            offset=offset,
            phi=phi,
            lambda_builder=lambda_builder,
        )
    return _pirls_sparse(
        y,
        X,
        Z,
        family,
        theta,
        specs,
        n_levels,
        weights,
        u0=u0,
        beta0=beta0,
        offset=offset,
        phi=phi,
        lambda_builder=lambda_builder,
        cholmod_handle=cholmod_handle,
    )


def _pirls_sparse(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    u0: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    lambda_builder: LambdaBuilder | None = None,
    cholmod_handle: _CholmodHandle | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Run PIRLS to find conditional modes (u_hat, beta_hat).

    Parameters
    ----------
    y : Response vector (n,). For binomial, this is proportion (0-1).
    X : Fixed-effects design matrix (n, p).
    Z : Random-effects design matrix (n, q).
    family : GLMMFamily instance.
    theta : Variance parameters.
    specs : Random effect specifications.
    n_levels : Number of levels per grouping factor.
    weights : Prior weights (n,). For binomial, this is the trial count.
    u0 : Initial random effects (q,). Defaults to zeros.
    beta0 : Initial fixed effects (p,). Defaults to zeros.
    offset : Offset vector (n,). Added to the linear predictor.
    phi : Per-observation dispersion vector (n,).  ``None`` means 1.

    Returns
    -------
    beta_hat : Fixed effects (p,).
    u_hat : Conditional modes of random effects (q,).
    mu_hat : Fitted means on response scale (n,).
    laplace_loglik : Laplace-approximated log-likelihood.
    converged : Whether PIRLS converged.
    """
    n, p = X.shape
    q = Z.shape[1]

    if lambda_builder is not None:
        Lambda = lambda_builder.update(theta)
        if lambda_builder.is_diagonal:
            Z_star = _scale_columns_csc(Z, lambda_builder.diag(theta))
        else:
            Z_star = (Z @ Lambda).tocsc()
    else:
        Lambda = make_lambda(theta, specs, n_levels)
        Z_star = (Z @ Lambda).tocsc()
    I_q = sp.eye(q, format="csc")

    u = np.zeros(q) if u0 is None else u0.copy()
    # Initialize beta from a GLM fit (no random effects) when not warm-starting.
    beta = (
        _glm_start(y, X, family, weights, offset=offset)
        if beta0 is None
        else beta0.copy()
    )

    _off = offset if offset is not None else np.zeros(n)

    converged = False

    for _iteration in range(_PIRLS_MAXITER):
        # Current linear predictor and mean
        eta = X @ beta + Z @ u + _off
        mu = family.linkinv(eta)

        # Clamp mu to avoid numerical issues
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)

        # Working weights and working residual
        if family.name in _ZI_FAMILIES:
            # ZI families need score/Hessian of the full mixture likelihood.
            w, z_w = _zi_pirls_weights(y, mu, weights, family, _off, eta)
        else:
            mu_eta_val = family.mu_eta(eta)  # d(mu)/d(eta)

            # W = diag(weights * mu_eta^2 / var) — the IRLS weight matrix
            # For NB2 with dispformula, phi carries per-obs theta → recompute var.
            # For other families, phi scales the variance multiplicatively.
            if phi is not None and family.name == "negativebinomial":
                var_mu = mu + mu**2 / phi  # NB2 variance with per-obs theta
                denom = var_mu
            else:
                var_mu = family.variance(mu)
                denom = var_mu if phi is None else phi * var_mu
            w = weights * mu_eta_val**2 / denom  # (n,)
            # Working residual (offset excluded so we solve for X@beta + Z@u)
            z_w = (eta - _off) + (y - mu) / mu_eta_val  # (n,)

        # Penalized weighted least squares via Lambda parameterisation.
        # Let v = Lambda^{-1} u so the penalty term becomes v'v.

        sqrtW = np.sqrt(w)  # (n,)
        # Scale everything by sqrt(W)
        WX = sqrtW[:, None] * X  # (n, p)
        Wz = sqrtW * z_w  # (n,)

        # Z_star is fixed across iterations within this PIRLS call (Lambda
        # depends on theta, not on iter); only the W-scaled version changes.
        WZs = _scale_rows_csc(Z_star, sqrtW)  # (n, q)

        XtWX = WX.T @ WX  # (p, p)
        ZstWX = np.asarray(
            (WZs.T @ WX).toarray() if sp.issparse(WZs.T @ WX) else WZs.T @ WX
        )  # (q, p)
        ZstWZs = (WZs.T @ WZs).tocsc()  # (q, q)
        XtWz = WX.T @ Wz  # (p,)
        ZstWz = np.asarray(WZs.T @ Wz).squeeze()  # (q,)

        # A = Z_star'WZ_star + I
        A = (ZstWZs + I_q).tocsc()

        # Schur complement solve for beta, then back-substitute for v.
        # CHOLMOD reuses the symbolic factor across PIRLS iters and does
        # multi-RHS in one call; SuperLU falls through to per-column spsolve.
        if cholmod_handle is not None:
            cholmod_handle.refactor(A)
            A_inv_ZstWX = np.asarray(cholmod_handle.solve(ZstWX))  # (q, p)
            A_inv_ZstWz = cholmod_handle.solve(ZstWz)  # (q,)
        else:
            A_inv_ZstWX = np.column_stack(
                [spla.spsolve(A, ZstWX[:, j]) for j in range(p)]
            )  # (q, p)
            A_inv_ZstWz = spla.spsolve(A, ZstWz)  # (q,)

        schur = XtWX - ZstWX.T @ A_inv_ZstWX  # (p, p)
        rhs_beta = XtWz - ZstWX.T @ A_inv_ZstWz  # (p,)

        try:
            beta_new = la.solve(schur, rhs_beta, assume_a="pos")
        except la.LinAlgError:
            beta_new = la.lstsq(schur, rhs_beta)[0]

        v_new = A_inv_ZstWz - A_inv_ZstWX @ beta_new  # (q,)
        u_new = np.asarray(Lambda @ v_new).squeeze()  # (q,)

        # Step-halving: limit the maximum change per iteration to prevent
        # overshooting in Poisson/binomial models with extreme predictions.
        max_step = 5.0
        delta_beta_raw = beta_new - beta
        delta_u_raw = u_new - u
        max_delta = max(np.max(np.abs(delta_beta_raw)), np.max(np.abs(delta_u_raw)))
        if max_delta > max_step:
            scale = max_step / max_delta
            beta_new = beta + scale * delta_beta_raw
            u_new = u + scale * delta_u_raw

        # Check convergence
        delta_beta = np.max(np.abs(beta_new - beta))
        delta_u = np.max(np.abs(u_new - u))
        max_change = max(delta_beta, delta_u)

        beta = beta_new
        u = u_new

        if max_change < _PIRLS_TOL:
            converged = True
            break

    # Final values
    eta = X @ beta + Z @ u + _off
    mu = family.linkinv(eta)
    if not isinstance(family, GaussianFamily):
        mu = _clamp_mu(mu, family)

    # Use v from the last PIRLS iteration (Lambda^{-1} u)
    v_final = v_new if converged or _iteration > 0 else np.zeros(q)  # noqa: F821
    penalty = float(v_final @ v_final)

    # Recompute A at final values for log|A|
    if family.name in _ZI_FAMILIES:
        w, _z_w_final = _zi_pirls_weights(y, mu, weights, family, _off, eta)
    else:
        mu_eta_val = family.mu_eta(eta)
        if phi is not None and family.name == "negativebinomial":
            var_mu = mu + mu**2 / phi
            denom_final = var_mu
        else:
            var_mu = family.variance(mu)
            denom_final = var_mu if phi is None else phi * var_mu
        w = weights * mu_eta_val**2 / denom_final
    WZs_final = _scale_rows_csc(Z_star, np.sqrt(w))
    A_final = ((WZs_final.T @ WZs_final) + I_q).tocsc()
    if cholmod_handle is not None:
        cholmod_handle.refactor(A_final)
        log_det_A = cholmod_handle.logdet()
    else:
        log_det_A = sparse_chol_logdet(A_final)

    # Laplace log-likelihood:
    # ll = log p(y|u_hat, beta_hat) - 0.5*v'v - 0.5*log|A|
    # where log p(y|...) is the conditional log-likelihood (not deviance).
    cond_ll = _conditional_loglik(y, mu, weights, family, phi=phi)

    laplace_ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A

    return beta, u, mu, laplace_ll, converged


def _pirls_dense(
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    u0: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    lambda_builder: LambdaBuilder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Dense PIRLS for small q (interlace-72oc).

    Mirrors :func:`_pirls_sparse` line-for-line but keeps Z, Z*, A, and the
    Cholesky factor as plain numpy arrays.  At small q the scipy.sparse
    construction overhead dwarfs the actual factorisation work.
    """
    n, p = X.shape
    q = Z.shape[1]
    Z_dense = np.asarray(Z.toarray()) if sp.issparse(Z) else np.asarray(Z)

    if lambda_builder is not None and lambda_builder.is_diagonal:
        lambda_diag = lambda_builder.diag(theta)
        Z_star_dense = Z_dense * lambda_diag[None, :]
        Lambda_dense_full = None
    else:
        if lambda_builder is not None:
            Lambda_dense_full = np.asarray(lambda_builder.update(theta).toarray())
        else:
            Lambda_dense_full = np.asarray(
                make_lambda(theta, specs, n_levels).toarray()
            )
        Z_star_dense = Z_dense @ Lambda_dense_full

    u = np.zeros(q) if u0 is None else u0.copy()
    beta = (
        _glm_start(y, X, family, weights, offset=offset)
        if beta0 is None
        else beta0.copy()
    )

    _off = offset if offset is not None else np.zeros(n)
    eye_q = np.eye(q)
    converged = False
    v_new = np.zeros(q)

    for _iteration in range(_PIRLS_MAXITER):
        eta = X @ beta + Z_dense @ u + _off
        mu = family.linkinv(eta)
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)

        if family.name in _ZI_FAMILIES:
            w, z_w = _zi_pirls_weights(y, mu, weights, family, _off, eta)
        else:
            mu_eta_val = family.mu_eta(eta)
            if phi is not None and family.name == "negativebinomial":
                var_mu = mu + mu**2 / phi
                denom = var_mu
            else:
                var_mu = family.variance(mu)
                denom = var_mu if phi is None else phi * var_mu
            w = weights * mu_eta_val**2 / denom
            z_w = (eta - _off) + (y - mu) / mu_eta_val

        sqrtW = np.sqrt(w)
        WX = sqrtW[:, None] * X
        Wz = sqrtW * z_w
        WZs = sqrtW[:, None] * Z_star_dense

        XtWX = WX.T @ WX
        ZstWX = WZs.T @ WX
        ZstWZs = WZs.T @ WZs
        XtWz = WX.T @ Wz
        ZstWz = WZs.T @ Wz

        A = ZstWZs + eye_q
        cho = la.cho_factor(A, lower=True)
        A_inv_ZstWX = la.cho_solve(cho, ZstWX)
        A_inv_ZstWz = la.cho_solve(cho, ZstWz)

        schur = XtWX - ZstWX.T @ A_inv_ZstWX
        rhs_beta = XtWz - ZstWX.T @ A_inv_ZstWz

        try:
            beta_new = la.solve(schur, rhs_beta, assume_a="pos")
        except la.LinAlgError:
            beta_new = la.lstsq(schur, rhs_beta)[0]

        v_new = A_inv_ZstWz - A_inv_ZstWX @ beta_new
        if Lambda_dense_full is not None:
            u_new = Lambda_dense_full @ v_new
        else:
            u_new = lambda_diag * v_new

        # Step-halving (mirrors sparse path).
        max_step = 5.0
        delta_beta_raw = beta_new - beta
        delta_u_raw = u_new - u
        max_delta = max(np.max(np.abs(delta_beta_raw)), np.max(np.abs(delta_u_raw)))
        if max_delta > max_step:
            scale = max_step / max_delta
            beta_new = beta + scale * delta_beta_raw
            u_new = u + scale * delta_u_raw

        delta_beta = float(np.max(np.abs(beta_new - beta)))
        delta_u = float(np.max(np.abs(u_new - u)))
        max_change = max(delta_beta, delta_u)

        beta = beta_new
        u = u_new

        if max_change < _PIRLS_TOL:
            converged = True
            break

    eta = X @ beta + Z_dense @ u + _off
    mu = family.linkinv(eta)
    if not isinstance(family, GaussianFamily):
        mu = _clamp_mu(mu, family)

    v_final = v_new if converged or _iteration > 0 else np.zeros(q)  # noqa: F821
    penalty = float(v_final @ v_final)

    if family.name in _ZI_FAMILIES:
        w, _ = _zi_pirls_weights(y, mu, weights, family, _off, eta)
    else:
        mu_eta_val = family.mu_eta(eta)
        if phi is not None and family.name == "negativebinomial":
            var_mu = mu + mu**2 / phi
            denom_final = var_mu
        else:
            var_mu = family.variance(mu)
            denom_final = var_mu if phi is None else phi * var_mu
        w = weights * mu_eta_val**2 / denom_final

    sqrtW_f = np.sqrt(w)
    WZs_final = sqrtW_f[:, None] * Z_star_dense
    A_final = WZs_final.T @ WZs_final + eye_q
    L_final = la.cholesky(A_final, lower=True)
    log_det_A = 2.0 * float(np.sum(np.log(np.diag(L_final))))

    cond_ll = _conditional_loglik(y, mu, weights, family, phi=phi)
    laplace_ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A

    return beta, u, mu, laplace_ll, converged


# ---------------------------------------------------------------------------
# Outer optimisation over theta
# ---------------------------------------------------------------------------


def _laplace_objective(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    warm: dict[str, np.ndarray | None],
    offset: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    lambda_builder: LambdaBuilder | None = None,
    cholmod_handle: _CholmodHandle | None = None,
) -> float:
    """Negative Laplace log-likelihood (to minimize over theta)."""
    beta, u, _mu, ll, _conv = _pirls(
        y,
        X,
        Z,
        family,
        theta,
        specs,
        n_levels,
        weights,
        u0=warm.get("u"),
        beta0=warm.get("beta"),
        offset=offset,
        phi=phi,
        lambda_builder=lambda_builder,
        cholmod_handle=cholmod_handle,
    )
    # Warm-start next call
    warm["u"] = u
    warm["beta"] = beta

    if not np.isfinite(ll):
        return 1e20
    return -ll


def _laplace_objective_profiled(
    theta_beta: np.ndarray,
    n_theta: int,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    warm: dict[str, np.ndarray | None],
    offset: np.ndarray | None = None,
    lambda_builder: LambdaBuilder | None = None,
    cholmod_handle: _CholmodHandle | None = None,
) -> float:
    """Negative Laplace log-likelihood optimising (theta, beta) jointly.

    Dispatches to a dense PIRLS kernel for small q (interlace-72oc), where
    the constant overhead of scipy.sparse construction dominates the actual
    factorisation work.  Above the threshold the original sparse path runs.
    """
    if Z.shape[1] <= _DENSE_Q_THRESHOLD:
        return _laplace_objective_profiled_dense(
            theta_beta,
            n_theta,
            y,
            X,
            Z,
            family,
            specs,
            n_levels,
            weights,
            warm,
            offset=offset,
            lambda_builder=lambda_builder,
        )
    return _laplace_objective_profiled_sparse(
        theta_beta,
        n_theta,
        y,
        X,
        Z,
        family,
        specs,
        n_levels,
        weights,
        warm,
        offset=offset,
        lambda_builder=lambda_builder,
        cholmod_handle=cholmod_handle,
    )


def _laplace_objective_profiled_sparse(
    theta_beta: np.ndarray,
    n_theta: int,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    warm: dict[str, np.ndarray | None],
    offset: np.ndarray | None = None,
    lambda_builder: LambdaBuilder | None = None,
    cholmod_handle: _CholmodHandle | None = None,
) -> float:
    """Sparse PIRLS kernel.  Same as the original ``_laplace_objective_profiled``
    body before interlace-72oc; used at q > _DENSE_Q_THRESHOLD."""
    theta = theta_beta[:n_theta]
    beta_fixed = theta_beta[n_theta:]

    n = len(y)
    q = Z.shape[1]
    if lambda_builder is not None:
        Lambda = lambda_builder.update(theta)
        if lambda_builder.is_diagonal:
            Z_star = _scale_columns_csc(Z, lambda_builder.diag(theta))
        else:
            Z_star = (Z @ Lambda).tocsc()
    else:
        Lambda = make_lambda(theta, specs, n_levels)
        Z_star = (Z @ Lambda).tocsc()
    I_q = sp.eye(q, format="csc")
    _off = offset if offset is not None else np.zeros(n)

    # Run PIRLS for u only, with beta fixed
    u_cached = warm.get("u")
    u = np.zeros(q) if u_cached is None else u_cached.copy()

    for _iteration in range(_PIRLS_MAXITER):
        eta = X @ beta_fixed + Z @ u + _off
        mu = family.linkinv(eta)
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)

        if family.name in _ZI_FAMILIES:
            w, z_w_prof = _zi_pirls_weights(y, mu, weights, family, _off, eta)
            wtres = z_w_prof - (eta - _off)  # score/neg_hess
        else:
            mu_eta_val = family.mu_eta(eta)
            var_mu = family.variance(mu)
            w = weights * mu_eta_val**2 / var_mu
            wtres = (y - mu) / mu_eta_val

        # Solve for v only: (Z_*'WZ_* + I) v = Z_*'W * residual
        # where residual = (y - mu) / mu_eta + Z_*' v_old (incremental)
        ZstWr = np.asarray(
            Z_star.T @ (w * (Z_star @ (u / Lambda.diagonal()) + wtres))
        ).squeeze()  # noqa: E501
        sqrtW = np.sqrt(w)
        WZs = _scale_rows_csc(Z_star, sqrtW)
        A = ((WZs.T @ WZs) + I_q).tocsc()
        if cholmod_handle is not None:
            cholmod_handle.refactor(A)
            v_new = cholmod_handle.solve(ZstWr)
        else:
            v_new = spla.spsolve(A, ZstWr)
        u_new = np.asarray(Lambda @ v_new).squeeze()

        delta_u = np.max(np.abs(u_new - u))
        u = u_new

        if delta_u < _PIRLS_TOL:
            break

    warm["u"] = u

    # Compute Laplace log-likelihood
    eta = X @ beta_fixed + Z @ u + _off
    mu = family.linkinv(eta)
    if not isinstance(family, GaussianFamily):
        mu = _clamp_mu(mu, family)

    cond_ll = _conditional_loglik(y, mu, weights, family)
    penalty = float(v_new @ v_new)

    if family.name in _ZI_FAMILIES:
        w, _ = _zi_pirls_weights(y, mu, weights, family, _off, eta)
    else:
        mu_eta_val = family.mu_eta(eta)
        var_mu = family.variance(mu)
        w = weights * mu_eta_val**2 / var_mu
    WZs_final = _scale_rows_csc(Z_star, np.sqrt(w))
    A_final = ((WZs_final.T @ WZs_final) + I_q).tocsc()
    if cholmod_handle is not None:
        cholmod_handle.refactor(A_final)
        log_det_A = cholmod_handle.logdet()
    else:
        log_det_A = sparse_chol_logdet(A_final)

    ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A

    if not np.isfinite(ll):
        return 1e20
    return -ll


def _laplace_objective_profiled_dense(
    theta_beta: np.ndarray,
    n_theta: int,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    warm: dict[str, np.ndarray | None],
    offset: np.ndarray | None = None,
    lambda_builder: LambdaBuilder | None = None,
) -> float:
    """Dense PIRLS kernel for small q (interlace-72oc).

    Mirrors :func:`_laplace_objective_profiled_sparse` line-for-line but
    keeps Z, Z*, and A as plain numpy arrays.  At q in {15, 36, 50} the
    sparse construction overhead in the inner loop dwarfs the actual
    factorisation work; going dense skips ~80% of wall time on CBPP.

    A dense ``Z_dense`` view is cached in ``warm['Z_dense']`` so it is
    materialised once per fit, not once per objective evaluation.
    """
    theta = theta_beta[:n_theta]
    beta_fixed = theta_beta[n_theta:]

    n = len(y)
    q = Z.shape[1]

    Z_dense = warm.get("Z_dense")
    if Z_dense is None:
        Z_dense = np.asarray(Z.toarray()) if sp.issparse(Z) else np.asarray(Z)
        warm["Z_dense"] = Z_dense

    # Build dense Lambda factor (for "u = Lambda @ v" / "v = Lambda^{-1} u").
    # Diagonal Lambda → just a (q,) vector; correlated → (q, q) lower-triangular.
    if lambda_builder is not None and lambda_builder.is_diagonal:
        lambda_diag = lambda_builder.diag(theta)
        Z_star_dense = Z_dense * lambda_diag[None, :]
        Lambda_diag_for_v = lambda_diag  # used to recover v = u / lambda_diag
        Lambda_dense_full = None
    else:
        if lambda_builder is not None:
            Lambda_dense_full = np.asarray(lambda_builder.update(theta).toarray())
        else:
            Lambda_dense_full = np.asarray(
                make_lambda(theta, specs, n_levels).toarray()
            )
        Z_star_dense = Z_dense @ Lambda_dense_full
        # Mirrors sparse code's ``Z_star @ (u / Lambda.diagonal())``: that
        # expression is only mathematically correct when Lambda is diagonal,
        # which is the path users hit at small q (CBPP, intercept-only).  For
        # correlated slopes at small q we keep parity with the sparse path,
        # which uses ``.diagonal()`` even though it's an approximation; tests
        # exercise the diagonal branch.
        Lambda_diag_for_v = np.diag(Lambda_dense_full)

    _off = offset if offset is not None else np.zeros(n)
    u_cached = warm.get("u")
    u = np.zeros(q) if u_cached is None else u_cached.copy()

    eye_q = np.eye(q)
    v_new = np.zeros(q)
    w = None

    # PIRLS for u only (beta is fixed by the outer optimiser in this routine).
    for _iteration in range(_PIRLS_MAXITER):
        eta = X @ beta_fixed + Z_dense @ u + _off
        mu = family.linkinv(eta)
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)

        if family.name in _ZI_FAMILIES:
            w, z_w_prof = _zi_pirls_weights(y, mu, weights, family, _off, eta)
            wtres = z_w_prof - (eta - _off)
        else:
            mu_eta_val = family.mu_eta(eta)
            var_mu = family.variance(mu)
            w = weights * mu_eta_val**2 / var_mu
            wtres = (y - mu) / mu_eta_val

        # Same expression as the sparse kernel:
        # ZstWr = Z_*' W (Z_* (u / Lambda.diag) + wtres)
        ZstWr = Z_star_dense.T @ (w * (Z_star_dense @ (u / Lambda_diag_for_v) + wtres))

        sqrtW = np.sqrt(w)
        WZs = sqrtW[:, None] * Z_star_dense
        A = WZs.T @ WZs + eye_q
        cho = la.cho_factor(A, lower=True)
        v_new = la.cho_solve(cho, ZstWr)
        if Lambda_dense_full is not None:
            u_new = Lambda_dense_full @ v_new
        else:
            u_new = lambda_diag * v_new

        delta_u = float(np.max(np.abs(u_new - u)))
        u = u_new

        if delta_u < _PIRLS_TOL:
            break

    warm["u"] = u

    eta = X @ beta_fixed + Z_dense @ u + _off
    mu = family.linkinv(eta)
    if not isinstance(family, GaussianFamily):
        mu = _clamp_mu(mu, family)

    cond_ll = _conditional_loglik(y, mu, weights, family)
    penalty = float(v_new @ v_new)

    if family.name in _ZI_FAMILIES:
        w, _ = _zi_pirls_weights(y, mu, weights, family, _off, eta)
    else:
        mu_eta_val = family.mu_eta(eta)
        var_mu = family.variance(mu)
        w = weights * mu_eta_val**2 / var_mu

    sqrtW_f = np.sqrt(w)
    WZs_final = sqrtW_f[:, None] * Z_star_dense
    A_final = WZs_final.T @ WZs_final + eye_q
    L_final = la.cholesky(A_final, lower=True)
    log_det_A = 2.0 * float(np.sum(np.log(np.diag(L_final))))

    ll = cond_ll - 0.5 * penalty - 0.5 * log_det_A
    if not np.isfinite(ll):
        return 1e20
    return -ll


# ---------------------------------------------------------------------------
# AGQ (Adaptive Gauss-Hermite Quadrature) objective
# ---------------------------------------------------------------------------


def _agq_loglik(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    family: GLMMFamily,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    weights: np.ndarray,
    nAGQ: int,
    group_indices: list[np.ndarray],
    warm: dict[str, np.ndarray | None],
    offset: np.ndarray | None = None,
) -> float:
    """Compute the AGQ-approximated marginal log-likelihood.

    For each group i the marginal contribution is estimated by adaptive
    Gauss-Hermite quadrature over the scalar random intercept u_i, adapting
    the quadrature nodes to the conditional mode and curvature found by PIRLS.

    Parameters
    ----------
    theta : Variance parameters.
    y, X, Z : Model matrices.
    family : GLMMFamily instance.
    specs : Random effect specifications (must be single scalar intercept).
    n_levels : Number of levels per grouping factor.
    weights : Prior weights.
    nAGQ : Number of GH quadrature points.
    group_indices : List of arrays, one per group level, each containing
        the row indices of observations belonging to that level.
    warm : Warm-start dict for PIRLS.
    offset : Offset vector (n,). Added to the linear predictor.

    Returns
    -------
    Negative log-likelihood (for minimisation).
    """
    from numpy.polynomial.hermite import hermgauss

    # Step 1: PIRLS to find conditional modes and working quantities
    beta, u, _mu, _laplace_ll, _conv = _pirls(
        y,
        X,
        Z,
        family,
        theta,
        specs,
        n_levels,
        weights,
        u0=warm.get("u"),
        beta0=warm.get("beta"),
        offset=offset,
    )
    warm["u"] = u
    warm["beta"] = beta

    _off = offset if offset is not None else np.zeros(len(y))

    sigma_u = float(theta[0])  # theta parameterises SD: sigma_u = theta * sigma
    # For GLMM dispersion is 1 (binomial/Poisson), so var_u = theta^2
    var_u = sigma_u**2

    if var_u < 1e-20:
        # Degenerate: no random effects, fall back to conditional ll
        eta = X @ beta + _off
        mu = family.linkinv(eta)
        if not isinstance(family, GaussianFamily):
            mu = _clamp_mu(mu, family)
        cond_ll = _conditional_loglik(y, mu, weights, family)
        return -cond_ll if np.isfinite(cond_ll) else 1e20

    # Step 2: Compute conditional precision per group from PIRLS working weights
    eta_hat = X @ beta + Z @ u + _off
    mu_hat = family.linkinv(eta_hat)
    if not isinstance(family, GaussianFamily):
        mu_hat = _clamp_mu(mu_hat, family)

    mu_eta_hat = family.mu_eta(eta_hat)
    var_mu_hat = family.variance(mu_hat)
    w_hat = weights * mu_eta_hat**2 / var_mu_hat  # IRLS weights at mode

    # GH nodes and weights
    gh_z, gh_w = hermgauss(nAGQ)  # ∫ exp(-t²) f(t) dt ≈ Σ w_k f(z_k)

    q = n_levels[0]  # number of groups
    total_ll = 0.0

    for i in range(q):
        idx = group_indices[i]
        u_hat_i = float(u[i])

        # Conditional precision: h_i = Σ_j w_{ij} + 1/var_u
        h_i = float(np.sum(w_hat[idx])) + 1.0 / var_u
        sigma_c_i = 1.0 / np.sqrt(h_i)  # conditional SD

        # Pre-extract group data
        y_i = y[idx]
        X_i = X[idx]
        w_i = weights[idx]
        off_i = _off[idx]

        # For each GH node, compute log-integrand
        # u_k = u_hat_i + sqrt(2) * sigma_c_i * z_k
        log_integrands = np.empty(nAGQ)
        for k in range(nAGQ):
            u_ik = u_hat_i + np.sqrt(2.0) * sigma_c_i * gh_z[k]

            # Linear predictor for group i at this u
            eta_ik = X_i @ beta + u_ik + off_i
            mu_ik = family.linkinv(eta_ik)
            if not isinstance(family, GaussianFamily):
                mu_ik = _clamp_mu(mu_ik, family)

            # Conditional log-likelihood for group i
            cll_ik = _conditional_loglik(y_i, mu_ik, w_i, family)

            # Log prior: log N(u_ik; 0, var_u)
            log_prior_ik = -0.5 * u_ik**2 / var_u

            # g(u_ik) = cond_ll + log_prior (unnormalised)
            g_ik = cll_ik + log_prior_ik

            # Integrand for GH: exp(g(u_ik) + z_k^2) * w_k
            # We work on log scale: log(w_k) + g_ik + z_k^2
            log_integrands[k] = np.log(gh_w[k]) + g_ik + gh_z[k] ** 2

        # log L_i = log(sqrt(2) * sigma_c_i) + logsumexp(log_integrands)
        max_li = np.max(log_integrands)
        log_Li = (
            0.5 * np.log(2.0)
            + np.log(sigma_c_i)
            + max_li
            + np.log(np.sum(np.exp(log_integrands - max_li)))
        )
        total_ll += log_Li

    # Subtract the log-prior normalising constant: -0.5*q*log(2*pi*var_u)
    # (the N(0, var_u) density has normalisation 1/sqrt(2*pi*var_u) per group)
    total_ll -= 0.5 * q * np.log(2.0 * np.pi * var_u)

    if not np.isfinite(total_ll):
        return 1e20
    return float(-total_ll)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_glmm(
    formula: str,
    data: Any,
    family: str | GLMMFamily,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    weights: np.ndarray | None = None,
    optimizer: str = "lbfgsb",
    theta0: np.ndarray | None = None,
    nAGQ: int = 1,
    offset: np.ndarray | None = None,
    dispformula: str | None = None,
) -> GLMMResult:
    """Fit a generalized linear mixed model.

    Parameters
    ----------
    formula :
        Fixed-effects formula, e.g. ``"y ~ x1 + x2"``.
    data :
        DataFrame (pandas, polars, or any narwhals-compatible frame).
    family :
        Family name (``"binomial"``, ``"poisson"``, ``"gaussian"``) or
        a :class:`GLMMFamily` instance.
    groups :
        Column name(s) for random intercepts.
    random :
        lme4-style random effect specs (takes precedence over ``groups``).
    weights :
        Prior weights. For binomial proportion response, pass trial counts.
        Defaults to ones.
    optimizer :
        ``"lbfgsb"`` (default) or ``"bobyqa"``.
    theta0 :
        Initial theta. Defaults to ones.
    nAGQ :
        Number of adaptive Gauss-Hermite quadrature points.  ``1`` (default)
        uses the Laplace approximation.  Values ``> 1`` use AGQ for a more
        accurate marginal-likelihood integral; this requires a single scalar
        random intercept (one grouping factor, intercept only).
    offset :
        Offset vector, shape ``(n,)``.  A known term added to the linear
        predictor that is not estimated.  Common use: ``np.log(exposure)``
        in Poisson rate models.  Defaults to zero.
    dispformula :
        Formula for the dispersion sub-model with a log link, e.g.
        ``"~1"`` for scalar dispersion or ``"~ z"`` for covariate-dependent
        dispersion.  The dispersion for observation *i* is
        ``phi_i = exp(X_d[i] @ delta)``.  ``None`` (default) fixes
        dispersion at 1.0.  Cannot be combined with ``nAGQ > 1``.

    Returns
    -------
    GLMMResult
        Fitted GLMM result containing fixed-effect estimates (``fe_params``,
        ``fe_bse``), BLUPs (``random_effects``), variance components
        (``variance_components``), in-sample fitted values (``fittedvalues``),
        log-likelihood (``llf``), information criteria (``aic``, ``bic``),
        and convergence status (``converged``).

    Examples
    --------
    >>> import numpy as np, pandas as pd, interlace
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"y": rng.binomial(10, 0.4, 200),
    ...                    "n": np.full(200, 10),
    ...                    "x": rng.normal(size=200),
    ...                    "g": rng.integers(0, 20, 200)})
    >>> result = interlace.glmer("cbind(y, n-y) ~ x", df,
    ...                          family="binomial", groups="g")
    >>> result.fe_params
    Intercept    ...
    x            ...
    dtype: float64
    """
    import formulaic
    import narwhals as nw

    fam = resolve_family(family)
    nw_data = nw.from_native(data, eager_only=True)

    # --- Build random effect specs ---
    if random is not None:
        specs = parse_random_effects(random)
    elif groups is not None:
        specs = groups_to_random_effects(groups)
    else:
        raise ValueError("Either 'groups' or 'random' must be provided.")

    group_cols = [s.group for s in specs]

    # --- Validate nAGQ ---
    if nAGQ < 1:
        raise ValueError("nAGQ must be >= 1.")
    if nAGQ > 1:
        if len(specs) > 1:
            msg = "nAGQ > 1 is only supported with a single grouping factor."
            raise ValueError(msg)
        if specs[0].n_terms > 1:
            msg = (
                "nAGQ > 1 is only supported for scalar random intercepts "
                "(no random slopes)."
            )
            raise ValueError(msg)

    # --- Validate dispformula + nAGQ ---
    if dispformula is not None and nAGQ > 1:
        msg = "dispformula cannot be combined with nAGQ > 1."
        raise ValueError(msg)

    # --- Parse formula, build X ---
    parsed = parse_formula(formula, data, groups=group_cols[0])
    y = parsed.y
    X = parsed.X
    term_names = parsed.term_names
    n, p = X.shape

    # --- Build dispersion design matrix ---
    if dispformula is not None:
        disp_formula_rhs = dispformula.lstrip("~").strip()
        if not disp_formula_rhs:
            disp_formula_rhs = "1"
        X_d_mm = formulaic.model_matrix("~ " + disp_formula_rhs, nw_data)
        disp_term_names = list(X_d_mm.columns)
        X_d = np.asarray(X_d_mm, dtype=np.float64)
        n_disp = X_d.shape[1]
    else:
        X_d = None
        disp_term_names = None
        n_disp = 0

    # --- Build Z ---
    Z = build_joint_z_from_specs(specs, data)
    n_levels_list = [
        int(np.unique(group_array(spec, nw_data)).shape[0]) for spec in specs
    ]

    # --- Weights ---
    if weights is None:
        weights_arr = np.ones(n)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)

    # --- Offset ---
    if offset is not None:
        offset_arr = np.asarray(offset, dtype=np.float64)
        if offset_arr.shape != (n,):
            msg = f"offset length ({offset_arr.size}) must match data length ({n})."
            raise ValueError(msg)
    else:
        offset_arr = np.zeros(n)

    # --- Theta setup ---
    n_theta = sum(n_theta_for_spec(s.n_terms, s.correlated) for s in specs)
    theta_bounds = _build_theta_bounds(specs)

    if theta0 is None:
        theta0 = np.ones(n_theta)

    # --- Optimize ---
    warm: dict[str, np.ndarray | None] = {"u": None, "beta": None}

    # Cache the structural sparse pattern of Lambda_theta once.  Reused by
    # _pirls and _laplace_objective_profiled across all optimiser evals.
    lambda_builder = LambdaBuilder(specs, n_levels_list)

    # Init CHOLMOD factor for inner-loop A solves (when sksparse is
    # available).  A's pattern is invariant across theta and W: it equals
    # pattern(Z_star.T @ Z_star) ∪ I_q.  Build a representative A0 with
    # theta=ones to capture the full pattern, then refactor numerically
    # per inner objective evaluation.
    q_rand = lambda_builder.q
    Lambda_init = lambda_builder.update(np.ones(lambda_builder._n_theta))
    Z_star_init = (Z @ Lambda_init).tocsc()
    A0 = ((Z_star_init.T @ Z_star_init) + sp.eye(q_rand, format="csc")).tocsc()
    cholmod_handle = _make_cholmod_handle(A0)

    # Precompute group indices for AGQ
    if nAGQ > 1:
        group_codes = group_array(specs[0], nw_data)
        group_uniques = sorted(np.unique(group_codes).tolist())
        group_indices = [np.where(group_codes == lvl)[0] for lvl in group_uniques]

    if X_d is not None:
        # Joint optimization over [theta | delta].
        # delta are the dispersion regression coefficients (log link, unbounded).
        delta0 = np.zeros(n_disp)
        params0 = np.concatenate([theta0, delta0])
        joint_bounds = list(theta_bounds) + [(None, None)] * n_disp

        def joint_obj(params: np.ndarray) -> float:
            theta = params[:n_theta]
            delta = params[n_theta:]
            phi = np.exp(X_d @ delta)
            return _laplace_objective(
                theta,
                y,
                X,
                Z,
                fam,
                specs,
                n_levels_list,
                weights_arr,
                warm,
                offset=offset_arr,
                phi=phi,
                lambda_builder=lambda_builder,
                cholmod_handle=cholmod_handle,
            )

        lower_bounds = np.array(
            [lo if lo is not None else -np.inf for lo, _ in joint_bounds]
        )

        if optimizer == "bobyqa":
            import pybobyqa

            upper = np.array(
                [hi if hi is not None else np.inf for _, hi in joint_bounds]
            )
            soln = pybobyqa.solve(joint_obj, params0, bounds=(lower_bounds, upper))
            params_hat = soln.x
            opt_converged = soln.msg == "Success: rho has reached rhoend"
        else:
            res = opt.minimize(
                joint_obj,
                params0,
                method="L-BFGS-B",
                bounds=joint_bounds,
            )
            params_hat = res.x
            opt_converged = bool(res.success)

        theta_hat = params_hat[:n_theta]
        delta_hat = params_hat[n_theta:]
        phi_hat = np.exp(X_d @ delta_hat)
    else:
        # No dispformula — original optimization path
        delta_hat = None
        phi_hat = None

        def obj(theta: np.ndarray) -> float:
            if nAGQ > 1:
                return _agq_loglik(
                    theta,
                    y,
                    X,
                    Z,
                    fam,
                    specs,
                    n_levels_list,
                    weights_arr,
                    nAGQ,
                    group_indices,
                    warm,
                    offset=offset_arr,
                )
            return _laplace_objective(
                theta,
                y,
                X,
                Z,
                fam,
                specs,
                n_levels_list,
                weights_arr,
                warm,
                offset=offset_arr,
                lambda_builder=lambda_builder,
                cholmod_handle=cholmod_handle,
            )

        lower_bounds = np.array(
            [lo if lo is not None else -np.inf for lo, _ in theta_bounds]
        )

        if optimizer == "bobyqa":
            import pybobyqa

            upper = np.array(
                [hi if hi is not None else np.inf for _, hi in theta_bounds]
            )
            soln = pybobyqa.solve(obj, theta0, bounds=(lower_bounds, upper))
            theta_hat = soln.x
            opt_converged = soln.msg == "Success: rho has reached rhoend"
        else:
            res = opt.minimize(
                obj,
                theta0,
                method="L-BFGS-B",
                bounds=theta_bounds,
            )
            theta_hat = res.x
            opt_converged = bool(res.success)

    # --- Phase 2: refine (theta, beta) jointly (lme4 nAGQ=1 style) ---
    # After finding theta_hat, re-optimise (theta, beta) jointly using
    # Nelder-Mead.  PIRLS in this phase only solves for u, with beta
    # supplied from the outer optimiser.  This avoids the PIRLS multimodality
    # issue that can occur with many observation-level random effects.
    if nAGQ <= 1 and phi_hat is None and fam.name not in _ZI_FAMILIES:
        # Get beta from the phase 1 warm cache
        beta_phase1 = warm.get("beta")
        if beta_phase1 is None:
            beta_phase1 = np.zeros(p)
        params_phase2 = np.concatenate([theta_hat, beta_phase1])
        warm_phase2: dict[str, np.ndarray | None] = {"u": warm.get("u")}

        res2 = opt.minimize(
            _laplace_objective_profiled,
            params_phase2,
            args=(
                n_theta,
                y,
                X,
                Z,
                fam,
                specs,
                n_levels_list,
                weights_arr,
                warm_phase2,
                offset_arr,
                lambda_builder,
                cholmod_handle,
            ),
            method="Nelder-Mead",
            # Tolerances aligned with lme4 stage-2 Nelder_Mead defaults
            # (R/optimizer.R:27-32, FtolAbs=1e-5, per-dim xt=xst*5e-4=1e-5,
            # fixed simplex algorithm).  maxiter is a defensive cap; lme4's
            # equivalent (maxfun=10000) is even larger.
            options={
                "xatol": 1e-5,
                "fatol": 1e-5,
                "maxiter": 2000,
                "adaptive": False,
            },
        )
        # Accept phase 2 if it improved the objective
        phase1_ll = -(obj(theta_hat) if callable(obj) else np.inf)
        phase2_ll = -res2.fun
        _phase2_accepted = False
        if phase2_ll > phase1_ll + 0.01:
            theta_hat = res2.x[:n_theta]
            warm["beta"] = res2.x[n_theta:]
            warm["u"] = warm_phase2.get("u")
            opt_converged = opt_converged and bool(res2.success)
            _phase2_accepted = True
    else:
        _phase2_accepted = False

    # --- Final PIRLS at optimum ---
    if _phase2_accepted:
        # Phase 2 found a better optimum by profiling beta out of PIRLS.
        # Run the profiled objective one final time to get clean (beta, u, ll).
        _laplace_objective_profiled(
            np.concatenate([theta_hat, warm["beta"]]),
            n_theta,
            y,
            X,
            Z,
            fam,
            specs,
            n_levels_list,
            weights_arr,
            warm_phase2,
            offset_arr,
            lambda_builder,
            cholmod_handle,
        )
        beta_hat = warm["beta"].copy()  # type: ignore[union-attr]
        u_hat = warm_phase2["u"].copy()  # type: ignore[union-attr]
        _off_final = offset_arr
        eta_final = X @ beta_hat + Z @ u_hat + _off_final
        mu_hat = fam.linkinv(eta_final)
        if not isinstance(fam, GaussianFamily):
            mu_hat = _clamp_mu(mu_hat, fam)
        laplace_llf = phase2_ll
        pirls_converged = True
    else:
        beta_hat, u_hat, mu_hat, laplace_llf, pirls_converged = _pirls(
            y,
            X,
            Z,
            fam,
            theta_hat,
            specs,
            n_levels_list,
            weights_arr,
            u0=warm.get("u"),
            beta0=warm.get("beta"),
            offset=offset_arr,
            phi=phi_hat,
            lambda_builder=lambda_builder,
            cholmod_handle=cholmod_handle,
        )
    converged = opt_converged and pirls_converged

    # For AGQ, recompute the final log-likelihood using AGQ at theta_hat
    if nAGQ > 1:
        warm_final: dict[str, np.ndarray | None] = {"u": u_hat, "beta": beta_hat}
        llf = -_agq_loglik(
            theta_hat,
            y,
            X,
            Z,
            fam,
            specs,
            n_levels_list,
            weights_arr,
            nAGQ,
            group_indices,
            warm_final,
            offset=offset_arr,
        )
    else:
        llf = laplace_llf

    # --- Fixed effects standard errors ---
    # From the Hessian of the penalized log-likelihood w.r.t. beta
    eta = X @ beta_hat + Z @ u_hat + offset_arr
    if fam.name in _ZI_FAMILIES:
        w, _z_w_se = _zi_pirls_weights(y, mu_hat, weights_arr, fam, offset_arr, eta)
    else:
        mu_eta_val = fam.mu_eta(eta)
        if phi_hat is not None and fam.name == "negativebinomial":
            var_mu = mu_hat + mu_hat**2 / phi_hat
            denom_se = var_mu
        else:
            var_mu = fam.variance(mu_hat)
            denom_se = var_mu if phi_hat is None else phi_hat * var_mu
        w = weights_arr * mu_eta_val**2 / denom_se
    WX = np.sqrt(w)[:, None] * X
    XtWX = WX.T @ WX

    Lambda = make_lambda(theta_hat, specs, n_levels_list)
    Z_star = Z @ Lambda
    WZs = sp.diags(np.sqrt(w), format="csc") @ Z_star
    ZstWZs = (WZs.T @ WZs).tocsc()
    q = Z.shape[1]
    A = (ZstWZs + sp.eye(q, format="csc")).tocsc()
    ZstWX = np.asarray(
        (WZs.T @ WX).toarray() if sp.issparse(WZs.T @ WX) else WZs.T @ WX
    )
    A_inv_ZstWX = np.column_stack([spla.spsolve(A, ZstWX[:, j]) for j in range(p)])
    schur = XtWX - ZstWX.T @ A_inv_ZstWX  # Marginal precision of beta

    try:
        fe_cov = la.inv(schur)
    except la.LinAlgError:
        fe_cov = np.linalg.pinv(schur)

    fe_bse_arr = np.sqrt(np.maximum(np.diag(fe_cov), 0.0))

    # --- Package results ---
    fe_params = pd.Series(beta_hat, index=term_names)
    fe_bse = pd.Series(fe_bse_arr, index=term_names)

    # --- Random effects per spec ---
    random_effects: dict[str, Any] = {}
    variance_components: dict[str, float] = {}
    ngroups: dict[str, int] = {}
    sigma2 = 1.0  # dispersion fixed at 1 for binomial/Poisson

    theta_idx = 0
    blup_offset = 0
    for spec, q_j in zip(specs, n_levels_list, strict=True):
        n_theta_j = n_theta_for_spec(spec.n_terms, spec.correlated)
        n_blups_j = spec.n_terms * q_j
        blup_block = u_hat[blup_offset : blup_offset + n_blups_j]
        uniques = sorted(np.unique(group_array(spec, nw_data)).tolist())

        if spec.n_terms == 1:
            random_effects[spec.group] = pd.Series(
                blup_block,
                index=uniques,
                name=spec.group,
            )
            theta_j0 = theta_hat[theta_idx]
            variance_components[spec.group] = float(sigma2 * theta_j0**2)
        else:
            term_names_j = (["(Intercept)"] if spec.intercept else []) + list(
                spec.predictors
            )
            theta_j = theta_hat[theta_idx : theta_idx + n_theta_j]
            re_mat = blup_block.reshape(spec.n_terms, q_j).T
            random_effects[spec.group] = pd.DataFrame(
                re_mat,
                index=uniques,
                columns=term_names_j,
            )
            p_j = spec.n_terms
            if spec.correlated:
                L_j = np.zeros((p_j, p_j))
                idx = 0
                for row in range(p_j):
                    for col in range(row + 1):
                        L_j[row, col] = theta_j[idx]
                        idx += 1
                cov_mat = sigma2 * L_j @ L_j.T
            else:
                cov_mat = np.diag(sigma2 * theta_j**2)
            variance_components[spec.group] = float(cov_mat[0, 0])

        ngroups[spec.group] = q_j
        theta_idx += n_theta_j
        blup_offset += n_blups_j

    # --- Dispersion results ---
    if delta_hat is not None:
        disp_params = pd.Series(delta_hat, index=disp_term_names)
    else:
        disp_params = None

    # --- Information criteria ---
    nparams = p + n_theta + n_disp
    aic = -2.0 * llf + 2.0 * nparams
    bic = -2.0 * llf + np.log(n) * nparams

    # --- Fitted values (response scale) and linear predictor ---
    eta_hat = X @ beta_hat + Z @ u_hat + offset_arr
    fittedvalues = np.asarray(fam.linkinv(eta_hat))

    return GLMMResult(
        fe_params=fe_params,
        fe_bse=fe_bse,
        random_effects=random_effects,
        variance_components=variance_components,
        theta=theta_hat,
        converged=converged,
        nobs=n,
        llf=float(llf),
        aic=float(aic),
        bic=float(bic),
        family=fam,
        ngroups=ngroups,
        scale=sigma2,
        fittedvalues=fittedvalues,
        _formula=formula,
        _group_cols=group_cols,
        _eta=np.asarray(eta_hat),
        disp_params=disp_params,
        dispersion=phi_hat,
    )
