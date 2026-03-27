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


def _full_params(
    model: Any,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Return ``(beta, V, V_inv, theta, theta_names, p)`` from either model type."""
    if _is_crossed(model):
        beta = model.fe_params
        p = len(np.asarray(beta))
        V = model.fe_cov
        group_cols = [model._gpgap_group_col] + model._gpgap_vc_cols
        theta = np.array(
            [model.variance_components[col] for col in group_cols] + [model.scale]
        )
        theta_names = [f"var_{col}" for col in group_cols] + ["error_var"]
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
        theta_i = np.array(
            [model_i.variance_components[col] for col in group_cols] + [model_i.scale]
        )
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

    for i, unit in tqdm(enumerate(units), total=n_units, desc=desc, disable=True):
        if level == 1:
            # Drop row i: concat slices before and after
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
                level_vals = nw_data[level_col].to_numpy()
                keep_mask = level_vals != unit
            else:
                keep_mask = groups != unit
            data_i = _filter_rows(data_native, keep_mask)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if _is_crossed(model):
                    import interlace

                    groups_arg = _refit_groups_arg(model)
                    model_i = interlace.fit(
                        model.model.formula,
                        data_i,
                        groups=groups_arg,
                        optimizer=optimizer,
                        theta0=model.theta,
                    )
                elif optimizer != "lbfgsb" and hasattr(model, "_gpgap_group_col"):
                    import interlace

                    model_i = interlace.fit(
                        model.model.formula,
                        data_i,
                        groups=model._gpgap_group_col,
                        optimizer=optimizer,
                    )
                else:
                    # statsmodels path — requires pandas (already checked above)
                    model_class = model.model.__class__
                    groups_col = model.model.groups
                    if level == 1:
                        groups_i = np.delete(groups_col, i)
                    else:
                        if level_col is not None:
                            groups_i = groups_col[data_native[level_col] != unit]
                        else:
                            groups_i = groups_col[groups_col != unit]
                    model_i_obj = model_class.from_formula(
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
