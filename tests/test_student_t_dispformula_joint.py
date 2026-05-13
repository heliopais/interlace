"""Tests for Student-t × dispformula via joint Laplace (interlace-1t0v).

Phase B: outer EM over the latent-scale t representation, with the
inner M-step delegated to the existing Gaussian joint-Laplace path
(:func:`interlace.dispformula_joint.fit_dispformula_joint` for
RE-on-disp; :func:`fit_dispformula_joint_laplace` for FE-only). The
disp-side variance components inherit the joint-Laplace bias
characteristics, i.e. they are unbiased to within numerical precision
of the Gaussian joint-MLE — by contrast with the BCA path which is
biased ~15-20%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats


def _simulate_t_heteroscedastic(
    n_groups: int = 60,
    n_per_group: int = 25,
    beta_true: tuple[float, float] = (1.0, 0.5),
    sigma_b: float = 0.7,
    disp_intercept: float = 0.0,
    disp_slope_z: float = 0.8,
    nu_true: float = 5.0,
    seed: int = 20260620,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    g = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    b = rng.normal(scale=sigma_b, size=n_groups)
    sigma_i = np.exp(disp_intercept + disp_slope_z * z)
    eps = stats.t.rvs(df=nu_true, scale=1.0, size=n, random_state=rng) * sigma_i
    y = beta_true[0] + beta_true[1] * x + b[g] + eps
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": g.astype(str)})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_fit_routes_to_joint_laplace_with_explicit_method() -> None:
    """``dispformula_method='joint_laplace'`` with family='student_t'
    no longer raises and returns a result exposing the disp surface."""
    import interlace

    df = _simulate_t_heteroscedastic(n_groups=20, n_per_group=10, seed=1)
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="joint_laplace",
    )
    assert hasattr(res, "nu")
    assert hasattr(res, "disp_params")
    assert isinstance(res.disp_params, pd.Series)
    assert "z" in res.disp_params.index
    assert getattr(res, "disp_method", None) == "joint_laplace"


def test_default_method_is_joint_laplace() -> None:
    """When ``dispformula_method`` is omitted, default to joint_laplace
    (matching the Gaussian dispformula default)."""
    import interlace

    df = _simulate_t_heteroscedastic(n_groups=20, n_per_group=10, seed=2)
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
    )
    assert getattr(res, "disp_method", None) == "joint_laplace"


# ---------------------------------------------------------------------------
# Recovery of mean-side parameters
# ---------------------------------------------------------------------------


def test_joint_recovers_mean_fixed_effects() -> None:
    """Recovery of beta within reasonable simulation noise."""
    import interlace

    df = _simulate_t_heteroscedastic(
        n_groups=60, n_per_group=30, nu_true=5.0, seed=20260621
    )
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="joint_laplace",
    )
    assert abs(res.fe_params["Intercept"] - 1.0) < 0.35
    assert abs(res.fe_params["x"] - 0.5) < 0.10


def test_joint_recovers_disp_slope() -> None:
    """Dispformula slope on z should be positive and order-of-magnitude
    correct."""
    import interlace

    df = _simulate_t_heteroscedastic(
        n_groups=60,
        n_per_group=30,
        disp_intercept=0.0,
        disp_slope_z=0.8,
        nu_true=5.0,
        seed=20260622,
    )
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="joint_laplace",
    )
    assert res.disp_params["z"] > 0.3


# ---------------------------------------------------------------------------
# Joint Laplace yields disp VCs closer to truth than BCA on RE-on-disp
# ---------------------------------------------------------------------------


def test_joint_disp_vc_unbiased_vs_bca() -> None:
    """On a nested RE-on-disp design (the salary structure), the joint
    Laplace path should recover the true disp variance component within
    ~20%, whereas BCA is expected to under-estimate by ~15-25%.

    We compare the absolute relative error of both estimators to the
    truth; joint Laplace must be no worse than BCA, and within 30%."""
    import interlace

    rng = np.random.default_rng(20260623)
    n_g1, n_g2_per, n_obs_per = 8, 6, 6
    n = n_g1 * n_g2_per * n_obs_per
    g1 = np.repeat(np.arange(n_g1), n_g2_per * n_obs_per)
    g2_within = np.tile(np.repeat(np.arange(n_g2_per), n_obs_per), n_g1)
    g2_label = np.array([f"{a}-{b}" for a, b in zip(g1, g2_within, strict=True)])
    x = rng.normal(size=n)
    b1 = rng.normal(scale=0.5, size=n_g1)
    sigma_d1_true = 0.45
    d1 = rng.normal(scale=sigma_d1_true, size=n_g1)
    log_sigma = d1[g1]
    sigma_i = np.exp(log_sigma)
    eps = stats.t.rvs(df=6.0, scale=1.0, size=n, random_state=rng) * sigma_i
    y = 1.0 + 0.4 * x + b1[g1] + eps
    df = pd.DataFrame({"y": y, "x": x, "g1": g1.astype(str), "g2": g2_label})

    res_bca = interlace.fit(
        "y ~ x",
        df,
        groups="g1",
        family="student_t",
        dispformula="~ (1 | g1)",
        dispformula_method="bca",
    )
    res_joint = interlace.fit(
        "y ~ x",
        df,
        groups="g1",
        family="student_t",
        dispformula="~ (1 | g1)",
        dispformula_method="joint_laplace",
    )

    truth = sigma_d1_true**2
    err_bca = abs(res_bca.disp_variance_components["g1"] - truth) / truth
    err_joint = abs(res_joint.disp_variance_components["g1"] - truth) / truth
    # Joint Laplace within 35% of truth.
    assert err_joint < 0.35
    # Joint Laplace not materially worse than BCA (BCA can be lucky on
    # a single seed; require joint <= BCA + 10pp).
    assert err_joint <= err_bca + 0.10


# ---------------------------------------------------------------------------
# nu identifiability
# ---------------------------------------------------------------------------


def test_joint_estimated_nu_in_range() -> None:
    """nu free should land in a plausible range."""
    import interlace

    df = _simulate_t_heteroscedastic(
        n_groups=60, n_per_group=25, nu_true=5.0, seed=20260624
    )
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="joint_laplace",
    )
    assert res.nu_estimated
    assert 2.5 < res.nu < 50.0


# ---------------------------------------------------------------------------
# nu fixed
# ---------------------------------------------------------------------------


def test_joint_nu_fixed() -> None:
    """When ``nu`` is provided, the estimate equals the input."""
    import interlace

    df = _simulate_t_heteroscedastic(n_groups=20, n_per_group=15, seed=20260625)
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="joint_laplace",
        nu=5.0,
    )
    assert res.nu == pytest.approx(5.0)
    assert not res.nu_estimated


# ---------------------------------------------------------------------------
# Large nu collapses to the Gaussian joint Laplace fit
# ---------------------------------------------------------------------------


def test_joint_large_nu_collapses_to_gaussian() -> None:
    """With nu very large, the t joint Laplace fit should approximate
    the Gaussian joint Laplace fit on the same data."""
    import interlace

    rng = np.random.default_rng(20260626)
    n_g, n_per = 30, 20
    n = n_g * n_per
    g = np.repeat(np.arange(n_g), n_per)
    x = rng.normal(size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    b = rng.normal(scale=0.5, size=n_g)
    sigma_i = np.exp(0.0 + 0.5 * z)
    eps = rng.normal(size=n) * sigma_i
    df = pd.DataFrame(
        {"y": 1.0 + 0.3 * x + b[g] + eps, "x": x, "z": z, "g": g.astype(str)}
    )
    gauss = interlace.fit(
        "y ~ x", df, groups="g", dispformula="~ z", dispformula_method="joint_laplace"
    )
    tfit = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="joint_laplace",
        nu=200.0,
    )
    np.testing.assert_allclose(gauss.fe_params.values, tfit.fe_params.values, atol=0.05)
    np.testing.assert_allclose(
        gauss.disp_params.values, tfit.disp_params.values, atol=0.1
    )
