"""Parity test: interlace vs statsmodels MixedLM for a single random-intercept model.

Acceptance criteria (from CLAUDE.md):
  - Fixed effects abs_diff < 1e-4
  - Variance components rel_diff < 5%
  - BLUP correlation > 0.99
  - Conditional residual correlation > 0.999
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace


@pytest.fixture(scope="module")
def single_re_data() -> pd.DataFrame:
    """Synthetic dataset: y ~ x + (1|group), 20 groups, 10 obs each."""
    rng = np.random.default_rng(42)
    n_groups = 20
    n_per_group = 10
    n = n_groups * n_per_group

    group_ids = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(n)

    # True params
    intercept = 2.0
    beta_x = 1.5
    sigma_u = 1.2  # group SD
    sigma_e = 0.8  # residual SD

    u = rng.normal(0, sigma_u, size=n_groups)
    eps = rng.normal(0, sigma_e, size=n)

    y = intercept + beta_x * x + u[group_ids] + eps

    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


def test_fixed_effects_match(single_re_data):
    df = single_re_data

    sm = MixedLM.from_formula("y ~ x", groups="group", data=df).fit(reml=True)
    il = interlace.fit("y ~ x", data=df, groups="group")

    for name in sm.fe_params.index:
        diff = abs(il.fe_params[name] - sm.fe_params[name])
        assert diff < 1e-4, (
            f"Fixed effect '{name}' diff={diff:.2e} "
            f"(interlace={il.fe_params[name]:.6f}, sm={sm.fe_params[name]:.6f})"
        )


def test_variance_components_match(single_re_data):
    df = single_re_data

    sm = MixedLM.from_formula("y ~ x", groups="group", data=df).fit(reml=True)
    il = interlace.fit("y ~ x", data=df, groups="group")

    # Residual variance
    rel_diff_scale = abs(il.scale - sm.scale) / sm.scale
    assert rel_diff_scale < 0.05, (
        f"Residual variance rel_diff={rel_diff_scale:.2%} "
        f"(interlace={il.scale:.6f}, sm={sm.scale:.6f})"
    )

    # Random intercept variance: statsmodels stores in cov_re (1×1 matrix)
    sm_var_u = float(sm.cov_re.iloc[0, 0])
    il_var_u = il.variance_components["group"]
    rel_diff_u = abs(il_var_u - sm_var_u) / sm_var_u
    assert rel_diff_u < 0.05, (
        f"Group variance rel_diff={rel_diff_u:.2%} "
        f"(interlace={il_var_u:.6f}, sm={sm_var_u:.6f})"
    )


def test_blups_correlated(single_re_data):
    df = single_re_data

    sm = MixedLM.from_formula("y ~ x", groups="group", data=df).fit(reml=True)
    il = interlace.fit("y ~ x", data=df, groups="group")

    # Align BLUPs by group label
    group_labels = sorted(df["group"].unique())
    sm_blups = np.array([sm.random_effects[g].iloc[0] for g in group_labels])
    il_blups = np.array([il.random_effects["group"][g] for g in group_labels])

    corr = np.corrcoef(sm_blups, il_blups)[0, 1]
    assert corr > 0.99, f"BLUP correlation={corr:.4f} < 0.99"


def test_residuals_correlated(single_re_data):
    df = single_re_data

    sm = MixedLM.from_formula("y ~ x", groups="group", data=df).fit(reml=True)
    il = interlace.fit("y ~ x", data=df, groups="group")

    corr = np.corrcoef(il.resid, sm.resid)[0, 1]
    assert corr > 0.999, f"Residual correlation={corr:.6f} < 0.999"
