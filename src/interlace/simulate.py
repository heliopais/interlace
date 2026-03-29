"""Parametric simulation and bootstrap for CrossedLMEResult.

Implements:
- simulate(): draw response vectors from the fitted mixed model
- bootMer(): parametric bootstrap with arbitrary user-defined statistic
- BootResult: container for bootstrap estimates with CI computation
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult


# ---------------------------------------------------------------------------
# BootResult
# ---------------------------------------------------------------------------


@dataclass
class BootResult:
    """Container for parametric bootstrap estimates.

    Attributes
    ----------
    estimates:
        Array of shape ``(B, p)`` where ``B`` is the number of bootstrap
        replicates and ``p`` is the length of the statistic vector.
    """

    estimates: np.ndarray

    def ci(
        self,
        method: str = "perc",
        level: float = 0.95,
    ) -> np.ndarray:
        """Compute bootstrap confidence intervals.

        Parameters
        ----------
        method:
            ``"perc"`` (percentile) is currently supported.
        level:
            Confidence level, e.g. ``0.95`` for a 95% CI.

        Returns
        -------
        np.ndarray of shape ``(p, 2)`` with columns ``[lower, upper]``.
        """
        if method != "perc":
            raise ValueError(
                f"method={method!r} is not supported; only 'perc' is available"
            )
        alpha = 1.0 - level
        lo = 100.0 * (alpha / 2.0)
        hi = 100.0 * (1.0 - alpha / 2.0)
        lower = np.percentile(self.estimates, lo, axis=0)
        upper = np.percentile(self.estimates, hi, axis=0)
        return np.column_stack([lower, upper])


# ---------------------------------------------------------------------------
# simulate()
# ---------------------------------------------------------------------------


def simulate(
    result: CrossedLMEResult,
    nsim: int = 1,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw response vectors from the fitted conditional model.

    Simulates from:

    .. math::

        y^* = X\\beta + Z u^* + \\varepsilon^*

    where :math:`u^* \\sim N(0,\\, \\sigma^2 \\Lambda\\Lambda^\\top)` and
    :math:`\\varepsilon^* \\sim N(0,\\, \\sigma^2 I_n)`.

    Equivalently: :math:`u^* = \\sigma \\Lambda z_u` with
    :math:`z_u \\sim N(0, I_q)`.

    Parameters
    ----------
    result:
        Fitted :class:`~interlace.result.CrossedLMEResult`.
    nsim:
        Number of response vectors to simulate.
    seed:
        Seed for the random number generator (int, Generator, or None).

    Returns
    -------
    np.ndarray of shape ``(n, nsim)`` — each column is one simulated
    response vector.
    """
    from interlace.profiled_reml import make_lambda

    rng = np.random.default_rng(seed)

    X = result.model.exog  # (n, p)
    Z = result._Z  # (n, q) sparse csc_matrix
    beta = np.asarray(result.fe_params)
    sigma = float(np.sqrt(result.scale))

    Lambda = make_lambda(result.theta, result._random_specs, result._n_levels)
    q = Lambda.shape[0]
    n = X.shape[0]

    # Fixed-effect mean: (n,)
    mu = X @ beta

    # Random effect draws: u* = sigma * Lambda @ z_u,  z_u ~ N(0, I_q)
    z_u = rng.standard_normal((q, nsim))  # (q, nsim)
    u_star = sigma * Lambda.dot(sp.csc_matrix(z_u))
    u_star_dense = u_star.toarray() if sp.issparse(u_star) else np.asarray(u_star)

    re_contrib = Z.dot(u_star_dense)  # (n, nsim)

    # Residual noise
    eps = sigma * rng.standard_normal((n, nsim))  # (n, nsim)

    return np.asarray(mu[:, np.newaxis] + re_contrib + eps)


# ---------------------------------------------------------------------------
# Default statistic for bootMer
# ---------------------------------------------------------------------------


def _default_statistic(result: CrossedLMEResult) -> np.ndarray:
    """Collect fixed effects, sqrt(sigma^2), and theta into a single vector."""
    fe = np.asarray(result.fe_params).ravel()
    sigma = np.array([float(np.sqrt(result.scale))])
    theta = np.asarray(result.theta).ravel()
    return np.concatenate([fe, sigma, theta])


# ---------------------------------------------------------------------------
# bootMer()
# ---------------------------------------------------------------------------


def _replace_col_in_frame(frame, col_name: str, values: np.ndarray):  # type: ignore[no-untyped-def]
    """Replace *col_name* in *frame* with *values*, returning the same native type."""
    import pandas as pd

    if isinstance(frame, pd.DataFrame):
        return frame.assign(**{col_name: values})

    try:
        import polars as pl

        if isinstance(frame, pl.DataFrame):
            return frame.with_columns(pl.Series(col_name, values))
    except ImportError:
        pass

    # Generic fallback: convert to pandas, replace, return pandas
    import narwhals as nw
    import pandas as pd

    pdf = nw.from_native(frame, eager_only=True).to_pandas()
    return pdf.assign(**{col_name: values})


def bootMer(
    result: CrossedLMEResult,
    statistic: Callable[..., np.ndarray] | None = None,
    B: int = 500,
    seed: int | np.random.Generator | None = None,
    show_progress: bool = False,
) -> BootResult:
    """Parametric bootstrap for a fitted mixed model.

    For each of ``B`` iterations:

    1. Simulate a new response vector ``y*`` from the fitted model.
    2. Refit the model on ``(X, Z, y*)``.
    3. Collect ``statistic(refit)`` into ``estimates``.

    Parameters
    ----------
    result:
        Fitted :class:`~interlace.result.CrossedLMEResult`.
    statistic:
        Callable ``(CrossedLMEResult) → np.ndarray`` (1-D).
        Defaults to :func:`_default_statistic` which collects
        ``[fe_params..., sqrt(sigma^2), theta...]``.
    B:
        Number of bootstrap replicates.
    seed:
        Seed for the random number generator.
    show_progress:
        If True and ``tqdm`` is installed, display a progress bar.

    Returns
    -------
    BootResult
        ``.estimates`` has shape ``(B, p)``.
    """
    from interlace import fit as _fit

    if statistic is None:
        statistic = _default_statistic

    rng = np.random.default_rng(seed)

    # Pre-generate seeds for each replicate (ensures reproducibility even if
    # the loop is parallelised in the future)
    rep_seeds = rng.integers(0, 2**31, size=B)

    # Reconstruct random= spec strings for refitting
    random_strs = [_spec_to_str(s) for s in result._random_specs]

    # Original data (native frame) and response column name
    native_data = result.model.data.frame
    endog_name = result.model.endog_names
    formula = result.model.formula
    method = result.method

    _iter: range | object = range(B)
    if show_progress:
        try:
            from tqdm import tqdm

            _iter = tqdm(range(B), desc="bootMer")
        except ImportError:
            pass

    estimates: list[np.ndarray] = []
    for i in _iter:  # type: ignore[union-attr]
        y_star = simulate(result, nsim=1, seed=int(rep_seeds[i])).ravel()
        data_b = _replace_col_in_frame(native_data, endog_name, y_star)

        refit = _fit(formula, data_b, random=random_strs, method=method)
        estimates.append(np.asarray(statistic(refit)).ravel())

    return BootResult(estimates=np.array(estimates))


# ---------------------------------------------------------------------------
# Helper: RandomEffectSpec → lme4-style string
# ---------------------------------------------------------------------------


def _spec_to_str(spec) -> str:  # type: ignore[no-untyped-def]
    """Convert a RandomEffectSpec back to an lme4-style string for refitting."""
    terms: list[str] = []
    if spec.intercept:
        terms.append("1")
    terms.extend(spec.predictors)
    effects = " + ".join(terms) if terms else "0"
    pipe = "|" if spec.correlated else "||"
    return f"({effects} {pipe} {spec.group})"
