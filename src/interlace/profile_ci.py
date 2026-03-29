"""Profile likelihood confidence intervals for variance components.

For each variance parameter theta_i, scans the 1D profile log-likelihood
holding all other thetas fixed at their ML estimates and finds the two
values where:

    2 * (L_max - L(theta_i)) = chi2(level, df=1)

using a bracket-then-Brent search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.optimize as _opt
import scipy.stats as _stats

from interlace.profiled_reml import _precompute, fit_ml, profile_loglik

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Bracket helpers
# ---------------------------------------------------------------------------


def _bracket_lower(
    f: Any,
    theta_hat_i: float,
    min_val: float = 1e-8,
    max_steps: int = 60,
) -> tuple[float, float] | tuple[None, None]:
    """Find [a, b] with f(a) < 0 <= f(b), searching left from theta_hat_i.

    Returns ``(None, None)`` if the lower boundary is hit (CI lower = 0).
    """
    # f(theta_hat_i) > 0 by construction
    b = theta_hat_i
    a = theta_hat_i
    # Start with 20% steps, doubling each iteration
    step = max(theta_hat_i * 0.2, 0.02)
    for _ in range(max_steps):
        a_new = max(a - step, min_val)
        val = f(a_new)
        if val < 0:
            return a_new, b  # found bracket
        b = a
        a = a_new
        step *= 1.5
        if a <= min_val:
            # Boundary hit: L is still above target at theta → 0
            return None, None
    return None, None


def _bracket_upper(
    f: Any,
    theta_hat_i: float,
    max_steps: int = 60,
) -> tuple[float, float]:
    """Find [a, b] with f(a) >= 0 > f(b), searching right from theta_hat_i.

    Raises ``RuntimeError`` if the upper bracket is not found.
    """
    a = theta_hat_i
    b = theta_hat_i
    step = max(theta_hat_i * 0.2, 0.02)
    for _ in range(max_steps):
        b_new = b + step
        if f(b_new) < 0:
            return a, b_new
        a = b
        b = b_new
        step *= 1.5
    msg = f"Could not find upper bracket for theta (last tried {b_new:.4f})"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Row label helpers
# ---------------------------------------------------------------------------


def _theta_labels(specs: list[Any], n_levels: list[int]) -> list[str]:
    """Return a human-readable label for each theta component."""
    labels: list[str] = []
    for spec in specs:
        group = spec.group
        p = spec.n_terms  # number of RE terms

        if p == 1:
            # Intercept-only or single predictor
            term = "(Intercept)" if spec.intercept else spec.predictors[0]
            labels.append(f"{group}.{term}")
        elif spec.correlated:
            # Lower-triangular Cholesky: p*(p+1)/2 parameters
            terms: list[str] = []
            if spec.intercept:
                terms.append("(Intercept)")
            terms.extend(spec.predictors)
            # Row-major lower-tri order: (0,0), (1,0), (1,1), (2,0), ...
            for row in range(p):
                for col in range(row + 1):
                    labels.append(f"{group}.L[{terms[row]},{terms[col]}]")
        else:
            # Diagonal (independent): one theta per term
            terms = []
            if spec.intercept:
                terms.append("(Intercept)")
            terms.extend(spec.predictors)
            for t in terms:
                labels.append(f"{group}.{t}")
    return labels


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def profile_confint(
    result: Any,
    level: float = 0.95,
) -> Any:
    """Compute profile likelihood CIs for variance parameters (theta).

    For each theta_i, fixes all other thetas at their ML estimates and
    finds the CI endpoints where:

        2 * (L_max - L(theta_i)) = chi2(level, df=1)

    using a geometric bracket search followed by Brent's method.

    Parameters
    ----------
    result:
        A fitted :class:`~interlace.result.CrossedLMEResult`.
    level:
        Nominal coverage probability (default 0.95).

    Returns
    -------
    pd.DataFrame
        Rows: one per theta parameter (named from the model's random-effect
        specs).  Columns: ``['estimate', lo_col, hi_col]`` where the
        percentage columns are named from *level*, e.g. ``'2.5 %'`` and
        ``'97.5 %'`` for ``level=0.95``.

    Notes
    -----
    Parameters are reported on the theta (relative Cholesky factor) scale.
    For intercept-only specs, ``sigma_b ≈ theta * sqrt(sigma2_hat)``.

    If the profile drops below the target before theta reaches 0, the lower
    bound is set to 0 (boundary case).
    """
    import pandas as _pd

    y = result.model.endog
    X = result.model.exog
    Z = result._Z
    specs = result._random_specs
    n_levels = result._n_levels

    # Always use ML (not REML) for profile likelihood
    ml_fit = fit_ml(y, X, Z, q_sizes=[], specs=specs, n_levels=n_levels)
    theta_hat = ml_fit.theta
    llf_max = ml_fit.llf

    chi2_crit = float(_stats.chi2.ppf(level, df=1))
    target = llf_max - chi2_crit / 2.0

    # Precompute cross-products once
    cache = _precompute(y, X, Z)

    def _profile(theta: np.ndarray) -> float:
        return profile_loglik(theta, y, X, Z, [], cache, specs=specs, n_levels=n_levels)

    n_theta = len(theta_hat)
    estimates = []
    lowers = []
    uppers = []

    for i in range(n_theta):
        theta_i_hat = float(theta_hat[i])

        def f(t: float, _i: int = i) -> float:
            theta = theta_hat.copy()
            theta[_i] = t
            return _profile(theta) - target

        # --- Lower bound ---
        lo_bracket = _bracket_lower(f, theta_i_hat)
        if lo_bracket[0] is None:
            # Profile never drops below target as theta_i → 0
            lower = 0.0
        else:
            a_lo, b_lo = lo_bracket
            lower = float(_opt.brentq(f, a_lo, b_lo, xtol=1e-6, rtol=1e-6))

        # --- Upper bound ---
        a_hi, b_hi = _bracket_upper(f, theta_i_hat)
        upper = float(_opt.brentq(f, a_hi, b_hi, xtol=1e-6, rtol=1e-6))

        estimates.append(theta_i_hat)
        lowers.append(lower)
        uppers.append(upper)

    # Column names from level
    lo_pct = 100.0 * (1.0 - level) / 2.0
    hi_pct = 100.0 - lo_pct
    lo_col = f"{lo_pct:.1f} %"
    hi_col = f"{hi_pct:.1f} %"

    labels = _theta_labels(specs, n_levels)

    return _pd.DataFrame(
        {
            "estimate": estimates,
            lo_col: lowers,
            hi_col: uppers,
        },
        index=labels,
    )
