"""Influence diagnostics for fitted linear mixed models via exact deletion.

Supports both ``CrossedLMEResult`` and statsmodels ``MixedLMResults``.

References
----------
Demidenko, E., & Stukel, T. A. (2005). Influence analysis for linear
mixed-effects models. Statistics in Medicine, 24(6), 893–909.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import narwhals as nw
import numpy as np
from tqdm import tqdm

from interlace._frame import filter_rows as _filter_rows
from interlace.result import CrossedLMEResult


def _is_crossed(model: Any) -> bool:
    return isinstance(model, CrossedLMEResult)


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
        p = model.k_fe
        beta = model.fe_params
        V = model.cov_params().iloc[:p, :p].values
        re_vars = np.diag(model.cov_re.values)
        re_names = [f"var_{name}" for name in model.cov_re.index]
        theta = np.append(re_vars, model.scale)
        theta_names = list(re_names) + ["error_var"]

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
    tight: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit REML on pre-built arrays; return ``(beta_i, Vi, theta_i)``.

    Bypasses formula parsing by working directly with numpy/sparse arrays.
    ``theta_i`` is in variance units matching :func:`_full_params` output
    (i.e. ``sigma2 * L @ L.T`` diagonal for each spec, then ``sigma2``).
    """
    from interlace.profiled_reml import (
        fit_reml,
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
        tight=tight,
    )
    sigma2_i = reml_i.sigma2

    # fe_cov = sigma2 * (X'Ω⁻¹X)^{-1} is pre-computed inside fit_reml
    if reml_i.fe_cov is None:
        msg = "fit_reml did not return fe_cov; this should never happen"
        raise RuntimeError(msg)
    Vi: np.ndarray = reml_i.fe_cov

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
# Module-level worker for multiprocessing (must be picklable)
# ---------------------------------------------------------------------------


def _refit_unit_worker(
    payload: dict[str, Any],
) -> tuple[int, float, float, float, float, np.ndarray]:
    """Case-deletion refit for a single unit; designed for ProcessPoolExecutor.

    Returns ``(unit_idx, cooks_d, mdffits, covtrace, covratio, rvc)`` where
    each scalar is ``nan`` if the refit fails.
    """
    i: int = payload["i"]
    theta: np.ndarray = payload["theta"]
    n_theta = len(theta)

    try:
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            beta_i, Vi, theta_i = _refit_matrices_crossed(
                payload["y_i"],
                payload["X_i"],
                payload["Z_i"],
                payload["specs"],
                payload["n_levels"],
                theta0=payload["theta0"],
                optimizer=payload["optimizer"],
                tight=False,
            )

        p: int = payload["p"]
        beta: np.ndarray = payload["beta"]
        V_inv: np.ndarray = payload["V_inv"]
        det_V: float = payload["det_V"]

        diff = beta - np.asarray(beta_i)
        cooks_d_i = (1.0 / p) * float(diff @ V_inv @ diff)
        Vi_inv = np.linalg.inv(Vi)
        mdffits_i = (1.0 / p) * float(diff @ Vi_inv @ diff)
        covtrace_i = float(np.trace(V_inv @ Vi)) - p
        covratio_i = float(np.linalg.det(Vi)) / det_V
        rvc_i: np.ndarray = theta_i / theta
        return (i, cooks_d_i, mdffits_i, covtrace_i, covratio_i, rvc_i)

    except Exception:  # noqa: BLE001
        return (i, np.nan, np.nan, np.nan, np.nan, np.full(n_theta, np.nan))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def hlm_influence(
    model: Any,
    level: int | str = 1,
    vc_formula: Any = None,
    optimizer: str = "lbfgsb",
    n_jobs: int = 1,
    show_progress: bool = False,
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
    n_jobs:
        Number of parallel worker processes for case-deletion refits.
        ``1`` (default) runs sequentially.  ``-1`` uses all available
        CPUs (``os.cpu_count()``).  Values > 1 are used as-is.
        Parallelism is only applied on the ``CrossedLMEResult`` path;
        statsmodels refits always run sequentially.  On Linux, workers
        are forked (fast startup); on macOS/Windows, they are spawned
        (slower startup — parallelism helps mainly when n ≳ 500).
    show_progress:
        Show a tqdm progress bar.  Default: ``False``.

    Returns
    -------
    Native DataFrame (pandas, polars, …) in the same type as the model input.
        Columns: ``cooksd``, ``mdffits``, ``covtrace``, ``covratio``,
        ``rvc.<name>`` for each variance component.
    """
    if optimizer not in ("lbfgsb", "bobyqa"):
        msg = f"optimizer must be 'lbfgsb' or 'bobyqa', got {optimizer!r}"
        raise ValueError(msg)

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

    # ------------------------------------------------------------------
    # Parallel path: CrossedLMEResult + n_jobs != 1
    # ------------------------------------------------------------------
    use_parallel = _cc is not None and n_jobs != 1

    if use_parallel:
        import concurrent.futures
        import multiprocessing

        assert _cc is not None  # guaranteed by use_parallel condition
        # Build one payload dict per unit (slicing is cheap — arrays are views)
        payloads: list[dict[str, Any]] = []
        for i, unit in enumerate(units):
            if level == 1:
                row_mask = np.ones(n_rows, dtype=bool)
                row_mask[i] = False
            else:
                lc = level if isinstance(level, str) else None
                row_mask = (
                    nw_data[lc].to_numpy() != unit if lc is not None else groups != unit
                )
            payloads.append(
                {
                    "i": i,
                    "y_i": _cc["y"][row_mask],
                    "X_i": _cc["X"][row_mask],
                    "Z_i": _cc["Z"][row_mask].tocsc(),
                    "specs": _cc["specs"],
                    "n_levels": _cc["n_levels"],
                    "theta0": _cc["theta0"],
                    "optimizer": optimizer,
                    "p": p,
                    "beta": np.asarray(beta),
                    "V_inv": V_inv,
                    "det_V": det_V,
                    "theta": theta,
                }
            )

        actual_workers = os.cpu_count() or 1 if n_jobs == -1 else max(1, n_jobs)
        # On Linux, fork is safe and fast (no reimport of numpy/scipy per worker).
        # On macOS/Windows, spawn is required to avoid fork-related crashes with
        # multi-threaded BLAS.
        mp_ctx_name = "fork" if sys.platform == "linux" else "spawn"
        ctx = multiprocessing.get_context(mp_ctx_name)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=actual_workers, mp_context=ctx
        ) as pool:
            results_iter = pool.map(_refit_unit_worker, payloads)
            for r in tqdm(
                results_iter, total=n_units, desc=desc, disable=not show_progress
            ):
                i_r, cd, md, ct, cr, rvc_i = r
                cooks_d[i_r] = cd
                mdffits_val[i_r] = md
                covtrace_val[i_r] = ct
                covratio_val[i_r] = cr
                rvc_val[i_r] = rvc_i

    else:
        # ------------------------------------------------------------------
        # Sequential path (original loop)
        # ------------------------------------------------------------------
        for i, unit in tqdm(
            enumerate(units), total=n_units, desc=desc, disable=not show_progress
        ):
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
                            tight=False,
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
                        # Pure statsmodels path — requires pandas (already checked).
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
            # statsmodels path
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
# Combined lmer influence measures (matches R's compute_influence_lmer)
# ---------------------------------------------------------------------------


def lmer_influence_measures(
    model: Any,
    optimizer: str = "lbfgsb",
    show_progress: bool = False,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Compute all influence measures for an lmer model.

    Matches R's HLMdiag convention.
    This combines case-deletion Cook's D / mdffits (via :func:`hlm_influence`)
    with analytical leverage and DFBETAS — exactly mirroring R's
    ``gpgap::compute_influence_lmer`` which calls ``HLMdiag::hlm_influence``.

    Parameters
    ----------
    model:
        A ``CrossedLMEResult`` or statsmodels ``MixedLMResults`` object.
    optimizer:
        Optimizer for case-deletion refits.  See :func:`hlm_influence`.
    show_progress:
        Show a tqdm progress bar during case-deletion refits.  Useful for
        large datasets (n > 500).
    n_jobs:
        Number of parallel worker processes.  ``1`` = sequential (default),
        ``-1`` = all CPUs.  See :func:`hlm_influence` for details.

    Returns
    -------
    dict with keys:
        ``cooks``   — Cook's D via case-deletion (matches HLMdiag ``cooksd``)
        ``hat``     — leverage used for threshold flagging (``overall`` for
                      single-RE, ``fixef`` for crossed multi-RE — mirrors R)
        ``hat_overall``  — full leverage H1+H2
        ``hat_fixef``    — fixed-effects leverage H1 only
        ``dfbetas`` — DFBETAS matrix (analytical, same formula as R)
        ``dffits``  — mdffits via case-deletion (matches HLMdiag ``mdffits``)
        ``residuals``  — conditional residuals
        ``sigma``      — residual standard deviation sqrt(scale)

    Notes
    -----
    Cook's D uses the Demidenko & Stukel (2005) case-deletion formula:

        D_i = (1/p) (β̂ − β̂₍₋ᵢ₎)ᵀ V_β⁻¹ (β̂ − β̂₍₋ᵢ₎)

    mdffits (returned as ``dffits``) uses the case-deletion covariance:

        MDFFITS_i = (1/p) (β̂ − β̂₍₋ᵢ₎)ᵀ V_β₍₋ᵢ₎⁻¹ (β̂ − β̂₍₋ᵢ₎)

    Both require O(n) model refits.  For large datasets consider setting
    ``show_progress=True`` to monitor progress.

    DFBETAS is computed analytically using the fixed-effects design matrix
    and conditional residuals, matching R's implementation.

    Leverage flagging uses ``hat_overall`` (H1+H2) for single-RE models and
    ``hat_fixef`` (H1 only) for crossed multi-RE models, exactly as R does
    when HLMdiag cannot compute overall leverage for crossed random effects.
    """
    from interlace.leverage import leverage as _leverage

    # --- Case-deletion Cook's D and mdffits (exact match to R/HLMdiag) ---
    infl_df = hlm_influence(
        model,
        level=1,
        optimizer=optimizer,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    def _col(df: Any, name: str) -> np.ndarray:
        col = df[name]
        return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col.values)

    cooks_arr = _col(infl_df, "cooksd")
    mdffits_arr = _col(infl_df, "mdffits")

    # --- Leverage: overall (H1+H2) and fixef (H1) ---
    lev_df = _leverage(model, level=1)

    def _lev_col(df: Any, name: str) -> np.ndarray:
        col = df[name]
        return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col.values)

    hat_overall = _lev_col(lev_df, "overall")
    hat_fixef = _lev_col(lev_df, "fixef")

    # R uses leverage.overall for single-RE and falls back to hat_fixef for
    # crossed multi-RE (HLMdiag can't compute overall leverage for crossed RE).
    truly_crossed = _is_crossed(model) and len(getattr(model, "_gpgap_vc_cols", [])) > 0
    hat_for_flag = hat_fixef if truly_crossed else hat_overall

    # --- DFBETAS: analytical from fixed-effects design matrix (same as R) ---
    X = np.asarray(model.model.exog)
    XtX_inv = np.linalg.pinv(X.T @ X)
    residuals = np.asarray(model.resid)
    sigma = float(np.sqrt(model.scale))

    hat_clamped = np.minimum(hat_fixef, 1.0 - 1e-10)
    scaling = residuals / (sigma * np.sqrt(1.0 - hat_clamped))
    dfbetas_mat = (XtX_inv @ (X * scaling[:, None]).T).T

    return {
        "cooks": cooks_arr,
        "hat": hat_for_flag,
        "hat_overall": hat_overall,
        "hat_fixef": hat_fixef,
        "dfbetas": dfbetas_mat,
        "dffits": mdffits_arr,
        "residuals": residuals,
        "sigma": sigma,
    }


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
