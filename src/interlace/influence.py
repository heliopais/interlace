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

import numpy as np
import pandas as pd
from tqdm import tqdm

from interlace.result import CrossedLMEResult


def _is_crossed(model: Any) -> bool:
    return isinstance(model, CrossedLMEResult)


# ---------------------------------------------------------------------------
# Helpers to extract a unified parameter set from either model type
# ---------------------------------------------------------------------------


def _full_params(
    model: Any,
) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Return ``(beta, V, V_inv, theta, theta_names, p)`` from either model type."""
    if _is_crossed(model):
        beta = model.fe_params
        p = len(beta)
        V = model.fe_cov
        group_cols = [model._gpgap_group_col] + model._gpgap_vc_cols
        theta = np.array(
            [model.variance_components[col] for col in group_cols] + [model.scale]
        )
        theta_names = [f"var_{col}" for col in group_cols] + ["error_var"]
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


def _refit(model: Any, data_i: pd.DataFrame) -> Any:
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
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
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
) -> pd.DataFrame:
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
    pd.DataFrame
        Columns: ``cooksd``, ``mdffits``, ``covtrace``, ``covratio``,
        ``rvc.<name>`` for each variance component.
    """
    if optimizer not in ("lbfgsb", "bobyqa"):
        msg = f"optimizer must be 'lbfgsb' or 'bobyqa', got {optimizer!r}"
        raise ValueError(msg)
    beta, V, V_inv, theta, theta_names, p = _full_params(model)
    det_V = np.linalg.det(V)

    data = model.model.data.frame.reset_index(drop=True)
    groups = (
        np.asarray(data[model._gpgap_group_col])
        if _is_crossed(model)
        else model.model.groups
    )

    if level == 1:
        units = data.index.tolist()
        n_units = len(units)
        desc = "Cook's D (obs)"
    else:
        level_col = level if isinstance(level, str) else None
        if level_col is not None:
            unit_vals = data[level_col].unique()
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

    for i, unit in tqdm(enumerate(units), total=n_units, desc=desc, disable=True):
        if level == 1:
            data_i = data.drop(unit).reset_index(drop=True)
        else:
            if level_col is not None:
                mask = data[level_col] == unit
            else:
                mask = pd.Series(groups == unit, index=data.index)
            data_i = data[~mask].reset_index(drop=True)

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
                    model_class = model.model.__class__
                    groups_col = model.model.groups
                    if level == 1:
                        groups_i = np.delete(groups_col, i)
                    else:
                        if level_col is not None:
                            groups_i = groups_col[data[level_col] != unit]
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

            diff = (beta - beta_i).values

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

    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def cooks_distance(model: Any, optimizer: str = "lbfgsb") -> np.ndarray:
    """Return Cook's distance for each observation."""
    return np.asarray(
        hlm_influence(model, level=1, optimizer=optimizer)["cooksd"].values
    )


def mdffits(model: Any, optimizer: str = "lbfgsb") -> np.ndarray:
    """Return MDFFITS for each observation."""
    return np.asarray(
        hlm_influence(model, level=1, optimizer=optimizer)["mdffits"].values
    )


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

    data = model.model.data.frame.reset_index(drop=True)
    data_reduced = data[~influential_mask].reset_index(drop=True)

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
            # Use the original groups array aligned to the reset_index data
            # Refit by passing the groups column values from reduced data
            # so that cov_re retains the same positional structure.
            groups_reduced_arr = model.model.groups[~influential_mask]
            model_i_obj = model.model.__class__.from_formula(
                model.model.formula, data=data_reduced, groups=groups_reduced_arr
            )
            model_reduced = model_i_obj.fit(reml=model.method == "REML")
            # Use positional indexing to avoid label-mismatch when groups
            # were passed as an array (the refitted cov_re may have a
            # different index name than the original).
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
        tau_f = np.sqrt(max(vc_full[factor], 0.0))
        tau_r = np.sqrt(max(vc_reduced.get(factor, 0.0), 0.0))
        gaps[factor] = float(abs(tau_f - tau_r))

    return gaps
