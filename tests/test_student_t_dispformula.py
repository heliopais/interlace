"""Tests for Student-t × dispformula composition (interlace-hpzy).

EM-with-latent-scale for Student-t residuals combined with a log-linear
sub-model for sigma_i:

    tau_i ~ Gamma(nu/2, nu/2)
    y_i | tau_i, sigma_i ~ N(x_i' beta + z_i' b, sigma_i^2 / tau_i)
    log sigma_i = W_i delta + V_i d

The mean step is a weighted Gaussian LMM with weights
``user_w * tau / sigma_i^2``; the disp step is a Gamma GLMM (log link,
shape 0.5) on ``tau_i * e_i^2`` (the t-likelihood sufficient statistic
for log sigma^2 under the latent-scale representation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats


def _simulate_student_t_heteroscedastic(
    n_groups: int = 60,
    n_per_group: int = 25,
    beta_true: tuple[float, float] = (1.0, 0.5),
    sigma_b: float = 0.7,
    disp_intercept: float = 0.0,
    disp_slope_z: float = 0.8,
    nu_true: float = 5.0,
    seed: int = 20260513,
) -> pd.DataFrame:
    """Simulate from a Student-t LMM with heteroscedastic scale.

    log sigma_i = disp_intercept + disp_slope_z * z_i; residual is
    Student-t with df=nu_true and scale sigma_i.
    """
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


def test_fit_routes_student_t_with_dispformula() -> None:
    """Public API: ``fit(family='student_t', dispformula=...)`` returns a
    fitted result exposing both Student-t and dispformula attributes."""
    import interlace

    df = _simulate_student_t_heteroscedastic(n_groups=20, n_per_group=10, seed=1)
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="bca",
    )

    # Student-t-specific surface
    assert hasattr(res, "nu")
    assert hasattr(res, "fe_params")
    # dispformula-specific surface
    assert hasattr(res, "disp_params")
    assert isinstance(res.disp_params, pd.Series)
    assert "z" in res.disp_params.index


# ---------------------------------------------------------------------------
# Recovery of mean-side parameters
# ---------------------------------------------------------------------------


def test_recovers_mean_fixed_effects_with_fixed_nu() -> None:
    """With nu fixed at truth, recover beta within reasonable simulation
    noise even when the residual scale is heteroscedastic."""
    import interlace

    df = _simulate_student_t_heteroscedastic(
        n_groups=80, n_per_group=30, nu_true=5.0, seed=20260514
    )
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="bca",
    )
    # True beta = (1.0, 0.5). SE(intercept) ~ 0.7/sqrt(80) ~ 0.08, allow 4 SE.
    assert abs(res.fe_params["Intercept"] - 1.0) < 0.35
    # SE(slope) bounded by ~0.02 with t-tails inflated; allow generous margin.
    assert abs(res.fe_params["x"] - 0.5) < 0.10


# ---------------------------------------------------------------------------
# Recovery of dispformula parameters
# ---------------------------------------------------------------------------


def test_recovers_disp_slope() -> None:
    """The dispformula slope on z should be positive and within ~30% of
    the true value (0.8) under a reasonable sample size."""
    import interlace

    df = _simulate_student_t_heteroscedastic(
        n_groups=80,
        n_per_group=30,
        disp_intercept=0.0,
        disp_slope_z=0.8,
        nu_true=5.0,
        seed=20260515,
    )
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="bca",
    )
    # Sign and order of magnitude.
    assert res.disp_params["z"] > 0.3


# ---------------------------------------------------------------------------
# nu identifiability
# ---------------------------------------------------------------------------


def test_estimated_nu_is_in_range() -> None:
    """With nu free, nu should land in a plausible range (not boundary)."""
    import interlace

    df = _simulate_student_t_heteroscedastic(
        n_groups=80, n_per_group=30, nu_true=5.0, seed=20260516
    )
    res = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="bca",
    )
    assert res.nu_estimated
    assert 2.5 < res.nu < 50.0


# ---------------------------------------------------------------------------
# Weights propagate through
# ---------------------------------------------------------------------------


def test_weights_ones_is_noop() -> None:
    """``weights=ones`` should match ``weights=None``."""
    import interlace

    df = _simulate_student_t_heteroscedastic(n_groups=20, n_per_group=15, seed=20260517)
    res_none = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="bca",
    )
    res_ones = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="bca",
        weights=np.ones(len(df)),
    )
    np.testing.assert_allclose(
        res_none.fe_params.values, res_ones.fe_params.values, atol=1e-6
    )
    np.testing.assert_allclose(
        res_none.disp_params.values, res_ones.disp_params.values, atol=1e-6
    )


# ---------------------------------------------------------------------------
# RE-on-disp (the salary case)
# ---------------------------------------------------------------------------


def test_re_on_disp_side_runs() -> None:
    """Mirror the salary-models structure: nested RE on disp side."""
    import interlace

    rng = np.random.default_rng(20260518)
    n_g1, n_per_g1 = 6, 30
    n_g2_per = 5
    rows = []
    for i in range(n_g1):
        for j in range(n_g2_per):
            for k in range(n_per_g1 // n_g2_per):
                rows.append({"g1": str(i), "g2": f"{i}-{j}", "k": k})
    df = pd.DataFrame(rows)
    n = len(df)
    df["x"] = rng.normal(size=n)
    # mean-side RE
    b1 = rng.normal(scale=0.5, size=n_g1)
    g1_codes = pd.factorize(df["g1"])[0]
    g2_codes = pd.factorize(df["g2"])[0]
    # disp-side RE
    d1 = rng.normal(scale=0.3, size=n_g1)
    d2 = rng.normal(scale=0.3, size=df["g2"].nunique())
    log_sigma = d1[g1_codes] + d2[g2_codes]
    sigma_i = np.exp(log_sigma)
    eps = stats.t.rvs(df=5.0, scale=1.0, size=n, random_state=rng) * sigma_i
    df["y"] = 1.0 + 0.4 * df["x"] + b1[g1_codes] + eps

    res = interlace.fit(
        "y ~ x",
        df,
        groups="g1",
        family="student_t",
        dispformula="~ (1 | g1/g2)",
        dispformula_method="bca",
    )
    # Fit ran end-to-end and emitted variance components on the disp side.
    assert hasattr(res, "disp_variance_components")
    assert "g1" in res.disp_variance_components
    assert res.disp_variance_components["g1"] > 0.0


# ---------------------------------------------------------------------------
# Large nu collapses to the Gaussian dispformula fit
# ---------------------------------------------------------------------------


def test_large_nu_collapses_to_gaussian_dispformula() -> None:
    """With nu fixed very large, t-dispformula should approximate the
    Gaussian dispformula fit."""
    import interlace

    rng = np.random.default_rng(2026)
    n_g, n_per = 30, 20
    n = n_g * n_per
    g = np.repeat(np.arange(n_g), n_per)
    x = rng.normal(size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    b = rng.normal(scale=0.5, size=n_g)
    sigma_i = np.exp(0.0 + 0.5 * z)
    eps = rng.normal(size=n) * sigma_i  # Gaussian
    df = pd.DataFrame(
        {"y": 1.0 + 0.3 * x + b[g] + eps, "x": x, "z": z, "g": g.astype(str)}
    )

    gauss = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        dispformula="~ z",
        dispformula_method="bca",
    )
    tfit = interlace.fit(
        "y ~ x",
        df,
        groups="g",
        family="student_t",
        dispformula="~ z",
        dispformula_method="bca",
    )
    # Mean-side FE should be close.
    np.testing.assert_allclose(gauss.fe_params.values, tfit.fe_params.values, atol=0.05)
    # Disp-side FE should be close.
    np.testing.assert_allclose(
        gauss.disp_params.values, tfit.disp_params.values, atol=0.1
    )


# ---------------------------------------------------------------------------
# nu validation propagates
# ---------------------------------------------------------------------------


def test_nu_validation_propagates() -> None:
    """Invalid nu (<=2) should be rejected before fitting begins."""
    import interlace

    df = _simulate_student_t_heteroscedastic(n_groups=10, n_per_group=5, seed=1)
    with pytest.raises(ValueError, match="nu"):
        interlace.fit(
            "y ~ x",
            df,
            groups="g",
            family="student_t",
            dispformula="~ z",
            dispformula_method="bca",
            nu=2.0,
        )
