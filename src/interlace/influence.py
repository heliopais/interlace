"""Influence diagnostics for fitted linear mixed models via exact deletion.

Supports both ``CrossedLMEResult`` and statsmodels ``MixedLMResults``.

References
----------
Demidenko, E., & Stukel, T. A. (2005). Influence analysis for linear
mixed-effects models. Statistics in Medicine, 24(6), 893–909.
"""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import numpy as np
from tqdm import tqdm

from interlace._frame import filter_rows as _filter_rows
from interlace.result import CrossedLMEResult


def _is_crossed(model: Any) -> bool:
    return isinstance(model, CrossedLMEResult)


def _require_pandas() -> Any:
    """Import and return pandas, raising a helpful error if not installed."""
    try:
        import pandas as pd

        return pd
    except ImportError as exc:
        raise ImportError(
            "The statsmodels compat path requires pandas. "
            "Install it with: pip install interlace-lme[pandas]"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers to extract a unified parameter set from either model type
# ---------------------------------------------------------------------------


def _vc_to_scalars(vc: Any, col: str) -> tuple[list[float], list[str]]:
    """Flatten a variance component into (values, names) for RVC computation.

    For scalar VCs (intercept-only) returns a single ``(var_col, [value])`` pair.
    For matrix VCs (random slopes) returns one entry per diagonal element,
    named ``var_col_term`` — e.g. ``var_g_(Intercept)`` and ``var_g_x``.
    """
    vc_arr = np.asarray(vc)
    if vc_arr.ndim == 0:
        return [float(vc_arr)], [f"var_{col}"]
    diag = np.diag(vc_arr) if vc_arr.ndim == 2 else vc_arr
    terms: list[str]
    if hasattr(vc, "index"):
        terms = list(vc.index)
    else:
        terms = [str(i) for i in range(len(diag))]
    return list(diag.astype(float)), [f"var_{col}_{t}" for t in terms]


def _full_params(
    model: Any,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Return ``(beta, V, V_inv, theta, theta_names, p)`` from either model type."""
    if _is_crossed(model):
        beta = model.fe_params
        p = len(np.asarray(beta))
        V = model.fe_cov
        group_cols = [model._gpgap_group_col] + model._gpgap_vc_cols
        theta_vals: list[float] = []
        theta_names_list: list[str] = []
        for col in group_cols:
            vals, names = _vc_to_scalars(model.variance_components[col], col)
            theta_vals.extend(vals)
            theta_names_list.extend(names)
        theta_vals.append(model.scale)
        theta_names_list.append("error_var")
        theta = np.array(theta_vals)
        theta_names = theta_names_list
    else:
        pd = _require_pandas()
        p = model.k_fe
        beta = model.fe_params
        V = model.cov_params().iloc[:p, :p].values
        re_vars = np.diag(model.cov_re.values)
        re_names = [f"var_{name}" for name in model.cov_re.index]
        theta = np.append(re_vars, model.scale)
        theta_names = list(re_names) + ["error_var"]
        del pd  # only imported for the statsmodels path

    V_inv = np.linalg.inv(V)
    return beta, V, V_inv, theta, theta_names, p


def _refit(model: Any, data_i: Any) -> Any:
    """Refit the model on the reduced dataset *data_i*.

    Returns a lightweight namespace with ``fe_params``, ``fe_cov``,
    ``variance_components`` / ``cov_re``, and ``scale``.
    """
    if _is_crossed(model):
        import interlace
        from interlace.formula import spec_to_str

        specs = getattr(model, "_random_specs", [])
        has_slopes = any(s.n_terms > 1 for s in specs)
        if has_slopes:
            random_strs = [spec_to_str(s) for s in specs]
            return interlace.fit(model.model.formula, data_i, random=random_strs)
        else:
            group_cols = [model._gpgap_group_col] + model._gpgap_vc_cols
            groups_arg = group_cols[0] if len(group_cols) == 1 else group_cols
            return interlace.fit(model.model.formula, data_i, groups=groups_arg)
    else:
        _require_pandas()
        model_i = model.model.__class__.from_formula(
            model.model.formula,
            data=data_i,
            groups=np.asarray(data_i[model._gpgap_group_col])
            if hasattr(model, "_gpgap_group_col")
            else model.model.groups[data_i.index],
        )
        return model_i.fit(reml=model.method == "REML")


def _reduced_params(
    model_i: Any,
    p: int,
    theta_names: list[str],
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Extract ``(beta_i, V_i, theta_i)`` from a refitted model."""
    if _is_crossed(model_i):
        beta_i = model_i.fe_params
        Vi = model_i.fe_cov
        group_cols = [model_i._gpgap_group_col] + model_i._gpgap_vc_cols
        theta_vals_i: list[float] = []
        for col in group_cols:
            vals, _ = _vc_to_scalars(model_i.variance_components[col], col)
            theta_vals_i.extend(vals)
        theta_vals_i.append(model_i.scale)
        theta_i = np.array(theta_vals_i)
    else:
        beta_i = model_i.fe_params
        Vi = model_i.cov_params().iloc[:p, :p].values
        re_vars_i = np.diag(model_i.cov_re.values)
        theta_i = np.append(re_vars_i, model_i.scale)

    return beta_i, Vi, theta_i


def _refit_groups_arg(model: Any) -> Any:
    """Return the groups argument string/list for interlace.fit refits."""
    if not _is_crossed(model):
        return None
    group_cols = [model._gpgap_group_col] + model._gpgap_vc_cols
    return group_cols[0] if len(group_cols) == 1 else group_cols


def _refit_matrices_crossed(
    y_i: np.ndarray,
    X_i: np.ndarray,
    Z_i: Any,
    specs: list[Any],
    n_levels: list[int],
    theta0: np.ndarray | None,
    optimizer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit REML on pre-built arrays; return ``(beta_i, Vi, theta_i)``.

    Bypasses formula parsing by working directly with numpy/sparse arrays.
    ``theta_i`` is in variance units matching :func:`_full_params` output
    (i.e. ``sigma2 * L @ L.T`` diagonal for each spec, then ``sigma2``).
    """
    import scipy.sparse as _sp

    from interlace.profiled_reml import (
        _build_A11,
        _precompute,
        _sparse_solve,
        fit_reml,
        make_lambda,
        n_theta_for_spec,
    )

    reml_i = fit_reml(
        y_i,
        X_i,
        Z_i,
        q_sizes=[],
        specs=specs,
        n_levels=n_levels,
        optimizer=optimizer,
        theta0=theta0,
    )
    sigma2_i = reml_i.sigma2

    # Recover fe_cov = sigma2 * (X'Ω⁻¹X)⁻¹ at optimum
    cache_i = _precompute(y_i, X_i, Z_i)
    Lambda_i = make_lambda(reml_i.theta, specs, n_levels)
    ZtZ_i = _sp.csc_matrix(cache_i["ZtZ"])
    ZtX_i = np.asarray(cache_i["ZtX"])
    A11_i = _build_A11(ZtZ_i, Lambda_i)
    lZtX_i = np.asarray(Lambda_i.T @ ZtX_i)
    C_X_i = _sparse_solve(A11_i, lZtX_i)
    XtX_i = np.asarray(cache_i["XtX"])
    MX_i = XtX_i - lZtX_i.T @ C_X_i
    Vi = sigma2_i * np.linalg.inv(MX_i)

    # Extract theta_i in variance units (one entry per VC diagonal, then sigma2)
    theta_vals_i: list[float] = []
    theta_raw_idx = 0
    for spec in specs:
        n_theta_j = n_theta_for_spec(spec.n_terms, spec.correlated)
        theta_j = reml_i.theta[theta_raw_idx : theta_raw_idx + n_theta_j]
        if spec.n_terms == 1:
            theta_vals_i.append(sigma2_i * float(theta_j[0] ** 2))
        elif spec.correlated:
            p_j = spec.n_terms
            L_j = np.zeros((p_j, p_j))
            idx = 0
            for row in range(p_j):
                for col in range(row + 1):
                    L_j[row, col] = theta_j[idx]
                    idx += 1
            cov_mat = sigma2_i * L_j @ L_j.T
            theta_vals_i.extend(np.diag(cov_mat).tolist())
        else:
            theta_vals_i.extend((sigma2_i * theta_j**2).tolist())
        theta_raw_idx += n_theta_j
    theta_vals_i.append(sigma2_i)  # error_var

    return np.asarray(reml_i.beta), Vi, np.array(theta_vals_i)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def hlm_influence(
    model: Any,
    level: int | str = 1,
    vc_formula: Any = None,
    optimizer: str = "lbfgsb",
) -> Any:
    """Calculate multiple influence diagnostics via exact deletion.

    Parameters
    ----------
    model:
        A ``CrossedLMEResult`` or statsmodels ``MixedLMResults`` object.
    level:
        ``1`` for observation-level; a group column name for group-level
        deletion.
    vc_formula:
        Variance-components formula passed through to statsmodels refits
        (3-level models only; ignored for ``CrossedLMEResult``).
    optimizer:
        Optimizer used for each case-deletion refit.  ``"lbfgsb"``
        (default) uses L-BFGS-B via scipy.  ``"bobyqa"`` uses
        ``pybobyqa`` and routes single-RE statsmodels refits through
        interlace REML, which is more robust near variance-parameter
        boundaries and reduces the Cook's D gap relative to R/HLMdiag.
        Requires the ``bobyqa`` optional extra when set to ``"bobyqa"``.

    Returns
    -------
    Native DataFrame (pandas, polars, …) in the same type as the model input.
        Columns: ``cooksd``, ``mdffits``, ``covtrace``, ``covratio``,
        ``rvc.<name>`` for each variance component.
    """
    if optimizer not in ("lbfgsb", "bobyqa"):
        msg = f"optimizer must be 'lbfgsb' or 'bobyqa', got {optimizer!r}"
        raise ValueError(msg)

    # Guard statsmodels path: requires pandas.
    if not _is_crossed(model):
        _require_pandas()

    beta, V, V_inv, theta, theta_names, p = _full_params(model)
    det_V = np.linalg.det(V)

    native_frame = model.model.data.frame

    # Choose the working frame: prefer the cached pandas frame when available
    # (ensures index-based operations work for the statsmodels compat path).
    # For CrossedLMEResult with polars input and no pandas installed, the
    # native frame is used directly.
    _pandas_frame = getattr(model.model.data, "_pandas_frame", None)
    data_src = _pandas_frame if _pandas_frame is not None else native_frame
    nw_data = nw.from_native(data_src, eager_only=True)
    n_rows = len(nw_data)

    groups = (
        nw_data[model._gpgap_group_col].to_numpy()
        if _is_crossed(model)
        else model.model.groups
    )

    if level == 1:
        units = list(range(n_rows))
        n_units = n_rows
        desc = "Cook's D (obs)"
    else:
        level_col = level if isinstance(level, str) else None
        if level_col is not None:
            unit_vals = nw_data[level_col].unique().to_numpy()
        else:
            unit_vals = np.unique(groups)
        units = list(unit_vals)
        n_units = len(units)
        desc = f"Cook's D (level={level})"

    cooks_d = np.full(n_units, np.nan)
    mdffits_val = np.full(n_units, np.nan)
    covtrace_val = np.full(n_units, np.nan)
    covratio_val = np.full(n_units, np.nan)
    rvc_val = np.full((n_units, len(theta)), np.nan)

    data_native = nw.to_native(nw_data)

    # Pre-build design matrices once for CrossedLMEResult — avoids re-parsing
    # the formula on every case-deletion refit (GitHub issue #7).
    _cc: dict[str, Any] | None = None
    if _is_crossed(model):
        from interlace.sparse_z import build_joint_z_from_specs as _build_z

        _cc_specs = getattr(model, "_random_specs", [])
        _cc_group_cols = [model._gpgap_group_col] + model._gpgap_vc_cols
        _cc_n_levels = [model.ngroups[col] for col in _cc_group_cols]
        _cc = {
            "specs": _cc_specs,
            "n_levels": _cc_n_levels,
            "X": model.model.exog,
            "y": model.model.endog,
            "Z": _build_z(_cc_specs, data_src),
            "theta0": model.theta,
        }

    for i, unit in tqdm(enumerate(units), total=n_units, desc=desc, disable=True):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if _cc is not None:
                    # Crossed path: slice pre-built arrays — no formula parsing.
                    if level == 1:
                        row_mask = np.ones(n_rows, dtype=bool)
                        row_mask[i] = False
                    else:
                        lc = level if isinstance(level, str) else None
                        row_mask = (
                            nw_data[lc].to_numpy() != unit
                            if lc is not None
                            else groups != unit
                        )
                    beta_i, Vi, theta_i = _refit_matrices_crossed(
                        _cc["y"][row_mask],
                        _cc["X"][row_mask],
                        _cc["Z"][row_mask].tocsc(),
                        _cc["specs"],
                        _cc["n_levels"],
                        theta0=_cc["theta0"],
                        optimizer=optimizer,
                    )
                elif optimizer != "lbfgsb" and hasattr(model, "_gpgap_group_col"):
                    # statsmodels bobyqa path — re-route through interlace.
                    if level == 1:
                        nw_before = nw_data[:i]
                        nw_after = nw_data[i + 1 :]
                        if i == 0:
                            data_i = nw.to_native(nw_after)
                        elif i == n_rows - 1:
                            data_i = nw.to_native(nw_before)
                        else:
                            data_i = nw.to_native(nw.concat([nw_before, nw_after]))
                    else:
                        level_col = level if isinstance(level, str) else None
                        if level_col is not None:
                            keep_mask = nw_data[level_col].to_numpy() != unit
                        else:
                            keep_mask = groups != unit
                        data_i = _filter_rows(data_native, keep_mask)
                    import interlace

                    model_i = interlace.fit(
                        model.model.formula,
                        data_i,
                        groups=model._gpgap_group_col,
                        optimizer=optimizer,
                    )
                    beta_i, Vi, theta_i = _reduced_params(model_i, p, theta_names)
                else:
                    # Pure statsmodels path — requires pandas (already checked above).
                    if level == 1:
                        nw_before = nw_data[:i]
                        nw_after = nw_data[i + 1 :]
                        if i == 0:
                            data_i = nw.to_native(nw_after)
                        elif i == n_rows - 1:
                            data_i = nw.to_native(nw_before)
                        else:
                            data_i = nw.to_native(nw.concat([nw_before, nw_after]))
                        groups_i = np.delete(model.model.groups, i)
                    else:
                        level_col = level if isinstance(level, str) else None
                        if level_col is not None:
                            keep_mask = nw_data[level_col].to_numpy() != unit
                            groups_i = model.model.groups[
                                data_native[level_col] != unit
                            ]
                        else:
                            keep_mask = groups != unit
                            groups_i = model.model.groups[groups != unit]
                        data_i = _filter_rows(data_native, keep_mask)
                    model_i_obj = model.model.__class__.from_formula(
                        model.model.formula,
                        data=data_i,
                        groups=groups_i,
                        vc_formula=vc_formula,
                    )
                    model_i = model_i_obj.fit(reml=model.method == "REML")
                    beta_i, Vi, theta_i = _reduced_params(model_i, p, theta_names)

            diff = np.asarray(beta) - np.asarray(beta_i)

            # Cook's Distance
            cooks_d[i] = (1 / p) * float(diff @ V_inv @ diff)

            # MDFFITS
            Vi_inv = np.linalg.inv(Vi)
            mdffits_val[i] = (1 / p) * float(diff @ Vi_inv @ diff)

            # COVTRACE
            covtrace_val[i] = float(np.trace(V_inv @ Vi)) - p

            # COVRATIO
            covratio_val[i] = float(np.linalg.det(Vi)) / det_V

            # RVC
            rvc_val[i] = theta_i / theta

        except Exception:  # noqa: BLE001
            pass  # leave as nan

    res: dict[str, Any] = {
        level if isinstance(level, str) else "index": units,
        "cooksd": cooks_d,
        "mdffits": mdffits_val,
        "covtrace": covtrace_val,
        "covratio": covratio_val,
    }
    for j, name in enumerate(theta_names):
        res[f"rvc.{name}"] = rvc_val[:, j]

    native_ns = nw.get_native_namespace(native_frame)
    return native_ns.DataFrame(res)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def cooks_distance(model: Any, optimizer: str = "lbfgsb") -> np.ndarray:
    """Return Cook's distance for each observation."""
    result = hlm_influence(model, level=1, optimizer=optimizer)
    # Support both pandas (.values) and polars (.to_numpy()) result frames
    col = result["cooksd"]
    return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col.values)


def mdffits(model: Any, optimizer: str = "lbfgsb") -> np.ndarray:
    """Return MDFFITS for each observation."""
    result = hlm_influence(model, level=1, optimizer=optimizer)
    col = result["mdffits"]
    return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col.values)


# ---------------------------------------------------------------------------
# n_influential and tau_gap
# ---------------------------------------------------------------------------


def n_influential(
    model: Any, threshold: float | None = None, optimizer: str = "lbfgsb"
) -> int:
    """Count observations whose Cook's distance exceeds *threshold*.

    Parameters
    ----------
    model:
        A ``CrossedLMEResult`` or statsmodels ``MixedLMResults`` object.
    threshold:
        Cut-off value.  Defaults to ``4 / n`` where ``n`` is the number of
        observations (the standard heuristic).
    optimizer:
        Optimizer used for case-deletion refits.  See :func:`hlm_influence`.

    Returns
    -------
    int
    """
    n = model.nobs if hasattr(model, "nobs") else model.model.nobs
    if threshold is None:
        threshold = 4.0 / n
    cd = cooks_distance(model, optimizer=optimizer)
    return int(np.sum(cd > threshold))


def tau_gap(
    model: Any, threshold: float | None = None, optimizer: str = "lbfgsb"
) -> dict[str, float]:
    """Absolute difference in random-effects std devs after removing influential obs.

    Refits the model excluding all observations where Cook's D > *threshold*,
    then returns ``|τ_full − τ_reduced|`` for each variance component, where
    ``τ = sqrt(variance_component)``.

    Parameters
    ----------
    model:
        A ``CrossedLMEResult`` or statsmodels ``MixedLMResults`` object.
    threshold:
        Cook's D cut-off.  Defaults to ``4 / n``.
    optimizer:
        Optimizer used for the reduced-data refit and for Cook's D
        computation.  See :func:`hlm_influence` for details.

    Returns
    -------
    dict[str, float]
        Keys match the random-effects factor names; values are ``|Δτ|``.
    """
    n = model.nobs if hasattr(model, "nobs") else model.model.nobs
    if threshold is None:
        threshold = 4.0 / n

    cd = cooks_distance(model, optimizer=optimizer)
    influential_mask = cd > threshold

    native_frame = model.model.data.frame
    _pandas_frame = getattr(model.model.data, "_pandas_frame", None)
    data_src = _pandas_frame if _pandas_frame is not None else native_frame
    data_reduced = _filter_rows(data_src, ~influential_mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _is_crossed(model):
            import interlace

            groups_arg = _refit_groups_arg(model)
            model_reduced = interlace.fit(
                model.model.formula,
                data_reduced,
                groups=groups_arg,
                optimizer=optimizer,
            )
            vc_full = model.variance_components
            vc_reduced = model_reduced.variance_components
        elif optimizer != "lbfgsb" and hasattr(model, "_gpgap_group_col"):
            import interlace

            model_reduced = interlace.fit(
                model.model.formula,
                data_reduced,
                groups=model._gpgap_group_col,
                optimizer=optimizer,
            )
            full_names = list(model.cov_re.index)
            vc_full = {
                name: float(model.cov_re.iloc[i, i])
                for i, name in enumerate(full_names)
            }
            vc_reduced = model_reduced.variance_components
        else:
            # statsmodels path — requires pandas
            _require_pandas()
            groups_reduced_arr = model.model.groups[~influential_mask]
            model_i_obj = model.model.__class__.from_formula(
                model.model.formula, data=data_reduced, groups=groups_reduced_arr
            )
            model_reduced = model_i_obj.fit(reml=model.method == "REML")
            full_names = list(model.cov_re.index)
            vc_full = {
                name: float(model.cov_re.iloc[i, i])
                for i, name in enumerate(full_names)
            }
            vc_reduced = {
                full_names[i]: float(model_reduced.cov_re.iloc[i, i])
                for i in range(len(full_names))
            }

    gaps = {}
    for factor in vc_full:
        vc_f = vc_full[factor]
        vc_r = vc_reduced.get(factor, 0.0)
        # variance_components values may be floats or numpy arrays (covariance matrix)
        tau_f = np.sqrt(
            max(float(vc_f) if np.ndim(vc_f) == 0 else float(np.trace(vc_f)), 0.0)
        )  # noqa: E501
        tau_r = np.sqrt(
            max(float(vc_r) if np.ndim(vc_r) == 0 else float(np.trace(vc_r)), 0.0)
        )  # noqa: E501
        gaps[factor] = float(abs(tau_f - tau_r))

    return gaps


# ---------------------------------------------------------------------------
# OLS influence — vectorised QR-based DFBETAS
# ---------------------------------------------------------------------------


def ols_dfbetas_qr(model: Any) -> np.ndarray:
    """Compute DFBETAS for an OLS model via QR decomposition (no Python loops).

    Implements the exact closed-form formula using the Sherman-Morrison-Woodbury
    identity and thin QR decomposition, matching R's ``influence.measures()``
    convention (LOO sigma in the denominator).

    For a design matrix X = QR (thin QR) with residuals e and MSE s²:

    - Hat diagonal: hᵢ = ‖Qᵢ‖²
    - LOO sigma²: s²ᵢ = (s²(n−p) − eᵢ²/(1−hᵢ)) / (n−p−1)
    - C = R⁻¹Qᵀ  (p×n), the "influence matrix" (X'X)⁻¹Xᵀ
    - se_coef[j] = ‖row j of R⁻¹‖ = √(diag[(X'X)⁻¹]ⱼ)
    - DFBETAS[i,j] = C[j,i] · eᵢ / ((1−hᵢ) · sᵢ · se_coef[j])

    Parameters
    ----------
    model :
        A fitted statsmodels ``RegressionResultsWrapper`` (OLS).

    Returns
    -------
    np.ndarray of shape (n, p)
        DFBETAS matrix, one row per observation, one column per parameter.

    References
    ----------
    Belsley, Kuh & Welsch (1980). *Regression Diagnostics*. Wiley.
    R's ``stats::dfbetas.lm`` / ``stats::influence.measures``.
    """
    X = np.asarray(model.model.exog)
    e = np.asarray(model.resid)
    n, p = X.shape
    df_resid = int(model.df_resid)  # n - p
    mse = float(model.mse_resid)

    # Thin QR decomposition
    Q, R = np.linalg.qr(X, mode="reduced")  # Q: (n,p), R: (p,p)

    # Hat diagonal
    h = np.einsum("ij,ij->i", Q, Q)  # (n,) — faster than (Q**2).sum(axis=1)

    # LOO sigma squared (clamped to avoid numerical negatives near h=1)
    loo_var = (mse * df_resid - e**2 / np.maximum(1 - h, 1e-10)) / (df_resid - 1)
    loo_sigma = np.sqrt(np.maximum(loo_var, 0.0))  # (n,)

    # Influence matrix C = R⁻¹ Qᵀ  (p×n) = (X'X)⁻¹ Xᵀ
    R_inv = np.linalg.solve(R, np.eye(p))  # (p,p)
    C = R_inv @ Q.T  # (p,n)

    # se_coef[j] = sqrt(diag[(X'X)⁻¹]_j) = ‖R_inv[j,:]‖
    se_coef = np.sqrt(np.einsum("ij,ij->i", R_inv, R_inv))  # (p,)

    # Scaled residuals for numerator
    scale = e / np.maximum(1 - h, 1e-10)  # (n,)

    # Numerator: (n, p)
    numerator = (C * scale[np.newaxis, :]).T

    # Denominator: (n, p)
    denominator = loo_sigma[:, np.newaxis] * se_coef[np.newaxis, :]

    return np.asarray(numerator / np.maximum(denominator, 1e-300))
