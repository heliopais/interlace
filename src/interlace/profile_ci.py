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

from interlace.profiled_reml import (
    _precompute,
    _sigma2_at_theta,
    fit_ml,
    profile_loglik,
)

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

    Returns ``(None, None)`` if the lower boundary is hit (CI lower = min_val).
    For unconstrained parameters (off-diagonal Cholesky), pass min_val=-1e6.
    """
    # f(theta_hat_i) > 0 by construction
    b = theta_hat_i
    a = theta_hat_i
    # Start with 20% steps, doubling each iteration
    step = max(abs(theta_hat_i) * 0.2, 0.02)
    for _ in range(max_steps):
        a_new = max(a - step, min_val)
        val = f(a_new)
        if val < 0:
            return a_new, b  # found bracket
        b = a
        a = a_new
        step *= 1.5
        if a <= min_val:
            # Boundary hit: profile still above target at min_val
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
# Natural-scale transform helpers
# ---------------------------------------------------------------------------


def _build_natural_transforms(specs: list[Any]) -> list[dict[str, Any]]:
    """Build a list of transform descriptors, one per theta, in theta order.

    Each descriptor has:
      'label'       : natural-scale row name (e.g. 'sd_(Intercept)|Group')
      'is_diagonal' : True for diagonal Cholesky elements (SD-type)
      'companions'  : for 'sd': list of off-diagonal theta indices in same row;
                      for 'cor': [diag_theta_index] of the same row
    """
    transforms: list[dict[str, Any]] = []
    theta_idx = 0

    for spec in specs:
        group = spec.group
        p = spec.n_terms
        terms: list[str] = []
        if spec.intercept:
            terms.append("(Intercept)")
        terms.extend(spec.predictors)

        if not spec.correlated or p == 1:
            # Diagonal spec: one theta per term, always diagonal
            for term in terms:
                transforms.append(
                    {
                        "label": f"sd_{term}|{group}",
                        "is_diagonal": True,
                        "companions": [],
                    }
                )
                theta_idx += 1
        else:
            # Correlated spec: lower-triangular Cholesky, p*(p+1)//2 thetas
            # First pass: record global theta index for each (row, col) position
            pos_to_tidx: dict[tuple[int, int], int] = {}
            k = 0
            for row in range(p):
                for col in range(row + 1):
                    pos_to_tidx[(row, col)] = theta_idx + k
                    k += 1

            # Second pass: build transform descriptors in emission order
            for row in range(p):
                for col in range(row + 1):
                    if row == col:
                        # Diagonal: sd_j = sigma * sqrt(off_diag_sq_sum + theta_jj^2)
                        offdiag = [pos_to_tidx[(row, c)] for c in range(row)]
                        transforms.append(
                            {
                                "label": f"sd_{terms[row]}|{group}",
                                "is_diagonal": True,
                                "companions": offdiag,
                            }
                        )
                    else:
                        # Off-diagonal: cor = theta / sqrt(theta^2 + theta_row_diag^2)
                        diag_tidx = pos_to_tidx[(row, row)]
                        transforms.append(
                            {
                                "label": f"cor_{terms[row]}.{terms[col]}|{group}",
                                "is_diagonal": False,
                                "companions": [diag_tidx],
                            }
                        )

            theta_idx += p * (p + 1) // 2

    return transforms


def _theta_to_natural(
    theta_val: float,
    info: dict[str, Any],
    theta_hat: np.ndarray,
    sigma_ml: float,
) -> float:
    """Transform a single theta value (or CI bound) to the natural scale."""
    if info["is_diagonal"]:
        # SD = sigma * sqrt(sum_of_offdiag_companions^2 + theta^2)
        companion_sq = sum(theta_hat[j] ** 2 for j in info["companions"])
        return sigma_ml * float(np.sqrt(companion_sq + theta_val**2))
    else:
        # Correlation = theta / sqrt(theta^2 + diag_companion^2)
        theta_diag = float(theta_hat[info["companions"][0]])
        denom = float(np.sqrt(theta_val**2 + theta_diag**2))
        return theta_val / denom if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def profile_confint(
    result: Any,
    level: float = 0.95,
    scale: str = "natural",
) -> Any:
    """Compute profile likelihood CIs for variance parameters.

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
    scale:
        ``'natural'`` (default) reports SDs and correlations, matching
        lme4's ``confint(m, method='profile')`` output.  Row labels take the
        form ``sd_TERM|GROUP`` and ``cor_TERM_k.TERM_j|GROUP``.

        ``'theta'`` reports raw relative Cholesky-factor entries.

    Returns
    -------
    pd.DataFrame
        Rows indexed by parameter name.  Columns: ``['estimate', lo_col,
        hi_col]`` where the percentage columns are named from *level*, e.g.
        ``'2.5 %'`` and ``'97.5 %'`` for ``level=0.95``.

    Notes
    -----
    For off-diagonal entries (correlation-type parameters), the profile
    search is unconstrained (theta can go negative), enabling negative
    correlation CI lower bounds.  Diagonal entries are constrained to
    theta > 0.

    If the profile never drops below the target as theta → 0 (boundary
    case), the lower bound is set to 0 (theta scale) or 0.0 (SD scale).
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

    # For natural scale: build transform descriptors per theta
    nat_transforms = _build_natural_transforms(specs) if scale == "natural" else None
    sigma_ml = float(np.sqrt(ml_fit.sigma2)) if scale == "natural" else 0.0

    n_theta = len(theta_hat)
    estimates = []
    lowers = []
    uppers = []

    for i in range(n_theta):
        theta_i_hat = float(theta_hat[i])

        # Off-diagonal (correlation-type) thetas are unconstrained
        is_diagonal = nat_transforms is None or nat_transforms[i]["is_diagonal"]
        lower_min = 1e-8 if is_diagonal else -1e6

        def f(t: float, _i: int = i) -> float:
            theta = theta_hat.copy()
            theta[_i] = t
            return _profile(theta) - target

        # --- Lower bound ---
        lo_bracket = _bracket_lower(f, theta_i_hat, min_val=lower_min)
        if lo_bracket[0] is None:
            lower_theta = lower_min if lower_min < 0 else 0.0
        else:
            a_lo, b_lo = lo_bracket
            lower_theta = float(_opt.brentq(f, a_lo, b_lo, xtol=1e-6, rtol=1e-6))

        # --- Upper bound ---
        a_hi, b_hi = _bracket_upper(f, theta_i_hat)
        upper_theta = float(_opt.brentq(f, a_hi, b_hi, xtol=1e-6, rtol=1e-6))

        if scale == "natural":
            info = nat_transforms[i]  # type: ignore[index]
            # Use sigma at each boundary (not sigma_ML) for accurate SD transform.
            # Correlation transform is sigma-free so sigma_ml is fine there.
            if info["is_diagonal"]:
                theta_lo_vec = theta_hat.copy()
                theta_lo_vec[i] = lower_theta
                theta_hi_vec = theta_hat.copy()
                theta_hi_vec[i] = upper_theta
                sigma_lo = float(
                    np.sqrt(
                        _sigma2_at_theta(
                            theta_lo_vec, y, X, Z, cache, specs=specs, n_levels=n_levels
                        )
                    )
                )
                sigma_hi = float(
                    np.sqrt(
                        _sigma2_at_theta(
                            theta_hi_vec, y, X, Z, cache, specs=specs, n_levels=n_levels
                        )
                    )
                )
            else:
                sigma_lo = sigma_hi = sigma_ml  # sigma cancels in correlation formula
            estimates.append(_theta_to_natural(theta_i_hat, info, theta_hat, sigma_ml))
            lowers.append(_theta_to_natural(lower_theta, info, theta_hat, sigma_lo))
            uppers.append(_theta_to_natural(upper_theta, info, theta_hat, sigma_hi))
        else:
            estimates.append(theta_i_hat)
            lowers.append(lower_theta)
            uppers.append(upper_theta)

    # Column names from level
    lo_pct = 100.0 * (1.0 - level) / 2.0
    hi_pct = 100.0 - lo_pct
    lo_col = f"{lo_pct:.1f} %"
    hi_col = f"{hi_pct:.1f} %"

    if scale == "natural":
        labels = [t["label"] for t in nat_transforms]  # type: ignore[union-attr]
    else:
        labels = _theta_labels(specs, n_levels)

    return _pd.DataFrame(
        {
            "estimate": estimates,
            lo_col: lowers,
            hi_col: uppers,
        },
        index=labels,
    )
