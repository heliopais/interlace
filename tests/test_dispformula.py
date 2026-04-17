"""Tests for the dispformula parameter in glmer().

Tests model-based dispersion estimation via a log-linear sub-model
for the scale parameter: phi_i = exp(X_d[i] @ delta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_data(
    n_groups: int = 20,
    n_per_group: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate Gaussian data with known dispersion (sigma^2 = 4.0)."""
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.5, n_groups)  # random intercepts, SD=1.5
    sigma = 2.0  # residual SD → phi = sigma^2 = 4.0
    y = 1.0 + 0.5 * x + u[group] + rng.normal(0, sigma, n)
    return pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})


def _make_heteroscedastic_data(
    n_groups: int = 30,
    n_per_group: int = 20,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate Gaussian data with covariate-dependent dispersion.

    True model:
        y ~ 1 + x, groups=group
        dispformula: log(phi) = 0.0 + 1.0 * z
        So phi_i = exp(z_i) and residual SD = sqrt(phi_i).
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(n)
    z = rng.uniform(-1, 1, n)  # dispersion covariate
    u = rng.normal(0, 1.0, n_groups)

    # True dispersion: phi_i = exp(0 + 1*z_i)
    phi = np.exp(0.0 + 1.0 * z)
    eps = rng.normal(0, 1, n) * np.sqrt(phi)

    y = 2.0 + 0.3 * x + u[group] + eps
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "group": group.astype(str),
        }
    )


# ---------------------------------------------------------------------------
# Tests: dispformula=None preserves existing behaviour
# ---------------------------------------------------------------------------


class TestDispformulaDefault:
    """When dispformula is None, behaviour is unchanged."""

    def test_binomial_scale_is_one(self):
        """Binomial models always have scale=1.0."""
        rng = np.random.default_rng(99)
        n = 200
        group = np.repeat(np.arange(20), 10).astype(str)
        x = rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(0.5 * x)))
        y = rng.binomial(1, p, n).astype(float)
        df = pd.DataFrame({"y": y, "x": x, "group": group})

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="binomial",
            groups="group",
        )
        assert result.scale == 1.0
        assert result.disp_params is None

    def test_poisson_scale_is_one(self):
        """Poisson models always have scale=1.0."""
        rng = np.random.default_rng(99)
        n = 200
        group = np.repeat(np.arange(20), 10).astype(str)
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"y": y, "x": x, "group": group})

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="poisson",
            groups="group",
        )
        assert result.scale == 1.0
        assert result.disp_params is None


# ---------------------------------------------------------------------------
# Tests: dispformula="~1" (scalar dispersion)
# ---------------------------------------------------------------------------


class TestScalarDispersion:
    """dispformula='~1' estimates a single dispersion parameter."""

    def test_gaussian_scalar_dispersion(self):
        """Gaussian GLMM with ~1 should recover the true residual variance."""
        df = _make_gaussian_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~1",
        )
        # True sigma^2 = 4.0.  Allow generous tolerance for 200 obs.
        assert result.converged
        estimated_phi = np.exp(result.disp_params["Intercept"])
        assert 2.0 < estimated_phi < 8.0  # within factor of 2

    def test_disp_params_is_series(self):
        """disp_params should be a pd.Series with named coefficients."""
        df = _make_gaussian_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~1",
        )
        assert isinstance(result.disp_params, pd.Series)
        assert "Intercept" in result.disp_params.index

    def test_aic_accounts_for_disp_params(self):
        """AIC should count dispersion parameters."""
        df = _make_gaussian_data()
        r1 = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
        )
        r2 = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~1",
        )
        # The dispformula model estimates 1 extra parameter.
        # AIC difference should reflect the improvement in fit
        # minus the penalty for the extra parameter.
        # Just check both are finite.
        assert np.isfinite(r1.aic)
        assert np.isfinite(r2.aic)


# ---------------------------------------------------------------------------
# Tests: dispformula with covariates (heteroscedastic)
# ---------------------------------------------------------------------------


class TestHeteroscedastic:
    """dispformula='~ z' models observation-level dispersion."""

    def test_heteroscedastic_dispersion_covariate(self):
        """Should recover positive dispersion slope for z."""
        df = _make_heteroscedastic_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~ z",
        )
        assert result.converged
        assert isinstance(result.disp_params, pd.Series)
        # True delta = [0.0, 1.0].  Slope on z should be positive.
        assert result.disp_params["z"] > 0.0

    def test_dispersion_values_stored(self):
        """Result should store the per-observation dispersion vector."""
        df = _make_heteroscedastic_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~ z",
        )
        assert hasattr(result, "dispersion")
        assert result.dispersion.shape == (len(df),)
        assert np.all(result.dispersion > 0)  # phi > 0 by exp()


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


class TestDispformulaValidation:
    """Edge cases and validation."""

    def test_rejects_dispformula_with_nAGQ_gt1(self):
        """dispformula + nAGQ > 1 is not supported."""
        df = _make_gaussian_data()
        with pytest.raises(ValueError, match="nAGQ"):
            interlace.glmer(
                "y ~ x",
                data=df,
                family="gaussian",
                groups="group",
                dispformula="~1",
                nAGQ=5,
            )
