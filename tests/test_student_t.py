"""Tests for Student-t (robust) family for LMM (interlace-8tna).

EM-with-latent-scale formulation:
    tau_i ~ Gamma(nu/2, nu/2)
    y_i | tau_i ~ N(x_i' beta + z_i' b, sigma^2 / tau_i)
Marginally each residual is Student-t with scale sigma and df nu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats


def _simulate_student_t_lmm(
    n_groups: int = 40,
    n_per_group: int = 12,
    beta_true: tuple[float, float] = (1.0, 0.5),
    sigma_b: float = 0.7,
    sigma_e: float = 0.3,
    nu_true: float = 4.0,
    seed: int = 20260512,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    g = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=n)
    b = rng.normal(scale=sigma_b, size=n_groups)
    # Student-t residuals via scipy: scale = sigma_e, df = nu_true
    eps = stats.t.rvs(df=nu_true, scale=sigma_e, size=n, random_state=rng)
    y = beta_true[0] + beta_true[1] * x + b[g] + eps
    return pd.DataFrame({"y": y, "x": x, "g": g})


def test_student_t_fixed_nu_recovers_params() -> None:
    """Test 1: with nu fixed at truth, recover beta and variance components
    within 3-SE simulation noise."""
    from interlace.student_t import student_t_fit

    df = _simulate_student_t_lmm(
        n_groups=200, n_per_group=20, nu_true=4.0, seed=20260512
    )
    res = student_t_fit(
        formula="y ~ x",
        data=df,
        groups="g",
        nu=4.0,  # fixed at truth
        max_iter=200,
        tol=1e-6,
    )

    # SE(Intercept) ~ sigma_b/sqrt(n_g) = 0.7/sqrt(200) ~ 0.05; allow 3 SE.
    assert abs(res.fe_params["Intercept"] - 1.0) < 0.15
    # SE(slope) ~ 0.01 with t-tail variance inflation; allow 0.03.
    assert abs(res.fe_params["x"] - 0.5) < 0.03

    # Group-intercept variance: rel std ~ sqrt(2/n_g) ~ 10%; allow 25%.
    vc = float(res.variance_components["g"])
    assert abs(vc - 0.49) / 0.49 < 0.25

    # Residual scale recovered to within ~10%.
    assert abs(res.sigma - 0.3) / 0.3 < 0.10

    # nu is exposed and equals the fixed input.
    assert res.nu == pytest.approx(4.0)
    assert res.converged


def test_student_t_estimable_nu() -> None:
    """Test 2: with nu free, recover nu within ~50% on n~3000.

    nu is weakly identified — likelihood is flat in nu past ~10. We use a
    large simulation to give the estimator a chance, and tolerate factor-2
    error on nu (the regime that matters is nu < 10).
    """
    from interlace.student_t import student_t_fit

    df = _simulate_student_t_lmm(
        n_groups=200, n_per_group=15, nu_true=5.0, seed=20260513
    )
    res = student_t_fit(
        formula="y ~ x",
        data=df,
        groups="g",
        nu=None,  # estimate
        max_iter=300,
        tol=1e-6,
    )

    # nu is weakly identified; allow factor-2 tolerance.
    assert 2.5 < res.nu < 10.0
    # SE(Intercept) ~ 0.7/sqrt(200) ~ 0.05; allow 3 SE.
    assert abs(res.fe_params["Intercept"] - 1.0) < 0.15
    # SE(slope) ~ 0.012; allow 0.04.
    assert abs(res.fe_params["x"] - 0.5) < 0.04
    assert res.nu_estimated


def test_student_t_weights_noop() -> None:
    """Test 3a: weights=ones is identical to weights=None (no-op)."""
    from interlace.student_t import student_t_fit

    df = _simulate_student_t_lmm(n_groups=30, n_per_group=10, seed=20260514)
    res_none = student_t_fit(formula="y ~ x", data=df, groups="g", nu=4.0, max_iter=200)
    res_ones = student_t_fit(
        formula="y ~ x",
        data=df,
        groups="g",
        nu=4.0,
        weights=np.ones(len(df)),
        max_iter=200,
    )
    np.testing.assert_allclose(
        res_none.fe_params.values, res_ones.fe_params.values, atol=1e-6
    )
    assert res_none.sigma == pytest.approx(res_ones.sigma, rel=1e-5)


def test_student_t_weights_change_estimate() -> None:
    """Test 3b: non-trivial weights move the estimate (weights are used)."""
    from interlace.student_t import student_t_fit

    df = _simulate_student_t_lmm(n_groups=30, n_per_group=10, seed=20260514)
    res_unw = student_t_fit(formula="y ~ x", data=df, groups="g", nu=4.0, max_iter=200)
    # Heavy down-weight first half — must produce different beta.
    w = np.where(np.arange(len(df)) < len(df) // 2, 0.1, 10.0)
    res_w = student_t_fit(
        formula="y ~ x", data=df, groups="g", nu=4.0, weights=w, max_iter=200
    )
    # At least one fixed-effect coefficient should differ by > 0.01.
    diff = np.abs(res_unw.fe_params.values - res_w.fe_params.values)
    assert diff.max() > 0.01


def test_student_t_nu_must_exceed_two() -> None:
    """nu <= 2 makes variance undefined; reject."""
    from interlace.student_t import student_t_fit

    df = _simulate_student_t_lmm(n_groups=10, n_per_group=5, seed=1)
    with pytest.raises(ValueError, match="nu"):
        student_t_fit(formula="y ~ x", data=df, groups="g", nu=2.0)


def test_fit_family_student_t_routes_to_student_t_fit() -> None:
    """Public API: ``fit(..., family='student_t')`` delegates to student_t_fit."""
    import interlace
    from interlace.student_t import StudentTResult

    df = _simulate_student_t_lmm(n_groups=20, n_per_group=10, seed=1)
    res = interlace.fit("y ~ x", df, groups="g", family="student_t")
    assert isinstance(res, StudentTResult)
    assert res.nu_estimated


def test_fit_family_invalid_raises() -> None:
    import interlace

    df = _simulate_student_t_lmm(n_groups=5, n_per_group=5, seed=1)
    with pytest.raises(ValueError, match="family"):
        interlace.fit("y ~ x", df, groups="g", family="poisson")


def test_student_t_large_nu_collapses_to_gaussian() -> None:
    """With nu very large, Student-t LMM should match Gaussian LMM fit."""
    import interlace
    from interlace.student_t import student_t_fit

    rng = np.random.default_rng(42)
    n_g, n_per = 20, 10
    n = n_g * n_per
    g = np.repeat(np.arange(n_g), n_per)
    x = rng.normal(size=n)
    b = rng.normal(scale=0.5, size=n_g)
    eps = rng.normal(scale=0.3, size=n)  # actually Gaussian
    df = pd.DataFrame({"y": 1.0 + 0.5 * x + b[g] + eps, "x": x, "g": g})

    gauss = interlace.fit("y ~ x", df, groups="g")
    tfit = student_t_fit(formula="y ~ x", data=df, groups="g", nu=200.0, max_iter=100)
    np.testing.assert_allclose(gauss.fe_params.values, tfit.fe_params.values, atol=5e-3)
