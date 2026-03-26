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

import numpy as np
import pandas as pd
import scipy.linalg as la

from interlace.result import CrossedLMEResult


def _is_crossed(model: Any) -> bool:
    return isinstance(model, CrossedLMEResult)


def _crossed_structures(
    model: CrossedLMEResult,
) -> tuple[np.ndarray, list[Any], list[np.ndarray], np.ndarray]:
    """Extract (groups, group_labels, exog_re_li, D, fe_cov, scale) from a
    CrossedLMEResult.

    Builds the per-primary-group Z_i matrices using the full joint random-effects
    structure so that V_i = Z_i D Z_i' + σ²I is correct for any number of
    crossed random intercepts.
    """
    data = model.model.data.frame
    primary_col = model._gpgap_group_col
    vc_cols = model._gpgap_vc_cols
    all_group_cols = [primary_col] + vc_cols

    groups = np.asarray(data[primary_col])
    group_labels = sorted(np.unique(groups))

    # Block-diagonal D: blkdiag(var_j * I_{q_j}, ...)
    vc_blocks = [
        model.variance_components[col] * np.eye(model.ngroups[col])
        for col in all_group_cols
    ]
    D = la.block_diag(*vc_blocks)

    # Column offsets into the joint Z matrix
    q_offsets = np.cumsum([0] + [model.ngroups[col] for col in all_group_cols])
    q_total = int(q_offsets[-1])

    # Integer codes for each factor (sorted, matching build_joint_z)
    codes_per_col = [pd.factorize(data[col], sort=True)[0] for col in all_group_cols]

    # Build Z_i for each level of the primary group
    exog_re_li = []
    for gval in group_labels:
        obs_idx = np.where(groups == gval)[0]
        n_i = len(obs_idx)
        Zi = np.zeros((n_i, q_total))
        for j_col, codes in enumerate(codes_per_col):
            col_start = int(q_offsets[j_col])
            for k, obs in enumerate(obs_idx):
                Zi[k, col_start + codes[obs]] = 1.0
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


def leverage(model: Any, level: int = 1) -> pd.DataFrame:  # noqa: ARG001
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
    pd.DataFrame
        Columns: ``overall`` (H1+H2), ``fixef`` (H1), ``ranef`` (H2),
        ``ranef.uc`` (unconfounded H2, Nobre & Singer).
    """
    X = model.model.exog
    n = X.shape[0]
    scale = model.scale

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

    return pd.DataFrame(
        {"overall": h1 + h2, "fixef": h1, "ranef": h2, "ranef.uc": h2_uc}
    )
