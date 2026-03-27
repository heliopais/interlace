"""Leverage diagnostics for fitted linear mixed models.

Supports both ``CrossedLMEResult`` and statsmodels ``MixedLMResults``.

References
----------
Demidenko, E., & Stukel, T. A. (2005). Influence analysis for linear
mixed-effects models. Statistics in Medicine, 24(6), 893–909.

Nobre, J. S., & Singer, J. M. (2007). Residual analysis for linear
mixed models. Biometrical Journal, 49(6), 863–875.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import scipy.linalg as la

from interlace.result import CrossedLMEResult


def _is_crossed(model: Any) -> bool:
    return isinstance(model, CrossedLMEResult)


def _crossed_structures(
    model: CrossedLMEResult,
) -> tuple[np.ndarray, list[Any], list[np.ndarray], np.ndarray]:
    """Extract (groups, group_labels, exog_re_li, D) from a CrossedLMEResult.

    Builds the per-primary-group Z_i matrices using the full joint random-effects
    structure so that V_i = Z_i D Z_i' + σ²I is correct for random intercepts
    and slopes.
    """
    native_frame = model.model.data.frame
    nw_data = nw.from_native(native_frame, eager_only=True)

    specs = getattr(model, "_random_specs", [])
    if not specs:
        # Fallback for legacy results without _random_specs
        from interlace.formula import groups_to_random_effects

        specs = groups_to_random_effects(
            [model._gpgap_group_col] + model._gpgap_vc_cols
        )

    primary_col = specs[0].group
    groups = nw_data[primary_col].to_numpy()
    group_labels = sorted(np.unique(groups).tolist())

    # Build D = blkdiag(cov_j ⊗ I_{q_j}) for each spec.
    # Z columns are term-first: [q_j intercept cols, q_j slope cols, ...]
    # so the matching covariance block is cov_j ⊗ I_{q_j}.
    vc_blocks = []
    for spec in specs:
        cov_j = np.asarray(model.variance_components[spec.group])
        q_j = model.ngroups[spec.group]
        if cov_j.ndim == 0:
            vc_blocks.append(float(cov_j) * np.eye(q_j))
        else:
            vc_blocks.append(np.kron(cov_j, np.eye(q_j)))
    D = la.block_diag(*vc_blocks)

    # Column offsets: each spec contributes n_terms_j * q_j columns
    col_widths = [spec.n_terms * model.ngroups[spec.group] for spec in specs]
    col_offsets = np.cumsum([0] + col_widths)
    q_total = int(col_offsets[-1])

    # Precompute per-spec: group codes and term value arrays (intercept=1, slope=x)
    spec_data = []
    for spec in specs:
        arr = nw_data[spec.group].to_numpy()
        _, codes = np.unique(arr, return_inverse=True)
        q_j = model.ngroups[spec.group]
        term_values: list[np.ndarray] = []
        if spec.intercept:
            term_values.append(np.ones(len(arr)))
        for pred in spec.predictors:
            term_values.append(nw_data[pred].to_numpy().astype(float))
        spec_data.append((codes, q_j, term_values))

    # Build Z_i for each level of the primary group
    exog_re_li = []
    for gval in group_labels:
        obs_idx = np.where(groups == gval)[0]
        n_i = len(obs_idx)
        Zi = np.zeros((n_i, q_total))
        for j_spec, (codes, q_j, term_values) in enumerate(spec_data):
            col_start = int(col_offsets[j_spec])
            for t_idx, tvals in enumerate(term_values):
                term_start = col_start + t_idx * q_j
                for k, obs in enumerate(obs_idx):
                    Zi[k, term_start + codes[obs]] = tvals[obs]
        exog_re_li.append(Zi)

    return groups, group_labels, exog_re_li, D


def _statsmodels_structures(model: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Extract leverage structures from a statsmodels MixedLMResults object."""
    cov_fe = model.cov_params().iloc[: model.k_fe, : model.k_fe].values
    D = model.cov_re.values
    groups = model.model.groups
    group_labels = model.model.group_labels
    exog_re_li = model.model.exog_re_li
    return groups, group_labels, exog_re_li, D, cov_fe


def leverage(model: Any, level: int = 1) -> Any:  # noqa: ARG001
    """Calculate observation-level leverage for a fitted linear mixed model.

    Parameters
    ----------
    model:
        A ``CrossedLMEResult`` or statsmodels ``MixedLMResults`` object.
    level:
        Reserved for future group-level leverage; currently only ``1``
        (observation level) is supported.

    Returns
    -------
    Native DataFrame (pandas, polars, …) matching the model input type.
        Columns: ``overall`` (H1+H2), ``fixef`` (H1), ``ranef`` (H2),
        ``ranef.uc`` (unconfounded H2, Nobre & Singer).
    """
    native_frame = model.model.data.frame
    X = model.model.exog
    n = X.shape[0]
    scale = model.scale

    truly_crossed = _is_crossed(model) and len(model._gpgap_vc_cols) > 0

    if truly_crossed:
        # For crossed RE (≥2 grouping factors), V is NOT block-diagonal by the
        # primary group, so the block-diagonal GLS hat gives trace(H1) != p.
        # Return the OLS hat X(X'X)^-1 X' as fixef — consistent with HLMdiag's
        # documented limitation for crossed random effects.
        XtX_inv = np.linalg.pinv(X.T @ X)
        h_ols = np.sum((X @ XtX_inv) * X, axis=1)
        result_dict: dict[str, Any] = {
            "overall": h_ols,
            "fixef": h_ols,
            "ranef": np.zeros(n),
            "ranef.uc": np.zeros(n),
        }
        native_ns = nw.get_native_namespace(native_frame)
        return native_ns.DataFrame(result_dict)

    if _is_crossed(model):
        cov_fe = model.fe_cov
        groups, group_labels, exog_re_li, D = _crossed_structures(model)
    else:
        groups, group_labels, exog_re_li, D, cov_fe = _statsmodels_structures(model)

    h1 = np.zeros(n)
    h2 = np.zeros(n)
    h2_uc = np.zeros(n)

    for i, gval in enumerate(group_labels):
        idx = np.where(groups == gval)[0]
        Xi = X[idx, :]
        Zi = exog_re_li[i]

        # V_i = Z_i D Z_i' + σ² I_{n_i}
        Vi = Zi @ D @ Zi.T + scale * np.eye(len(idx))
        Vi_inv = np.linalg.inv(Vi)

        # H1_i = X_i (X'Ω⁻¹X)⁻¹ X_i' V_i⁻¹  (fixed-effect leverage)
        H1_i = Xi @ cov_fe @ Xi.T @ Vi_inv
        h1[idx] = np.diagonal(H1_i)

        # H2_i = Z_i D Z_i' V_i⁻¹ (I - H1_i)  (Demidenko & Stukel)
        ZDZt = Zi @ D @ Zi.T
        H2_i = ZDZt @ Vi_inv @ (np.eye(len(idx)) - H1_i)
        h2[idx] = np.diagonal(H2_i)

        # H2_uc_i = Z_i D Z_i' / σ²  (Nobre & Singer unconfounded)
        h2_uc[idx] = np.diagonal(ZDZt / scale)

    result_dict = {"overall": h1 + h2, "fixef": h1, "ranef": h2, "ranef.uc": h2_uc}
    native_ns = nw.get_native_namespace(native_frame)
    return native_ns.DataFrame(result_dict)
