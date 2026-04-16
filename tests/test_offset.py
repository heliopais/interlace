"""Tests for offset parameter in fit() and glmer()."""

import numpy as np
import pandas as pd
import pytest

import interlace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lmm_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n_groups, n_per = 10, 20
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    x = rng.normal(size=n)
    b = rng.normal(scale=0.5, size=n_groups)
    y = 1.0 + 0.5 * x + b[group] + rng.normal(scale=1.0, size=n)
    offset = rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x": x, "group": group, "offset": offset})


def _poisson_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    n_groups, n_per = 15, 30
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    x = rng.normal(size=n)
    b = rng.normal(scale=0.3, size=n_groups)
    exposure = rng.uniform(1, 10, size=n)
    log_mu = 0.5 + 0.3 * x + b[group] + np.log(exposure)
    y = rng.poisson(np.exp(log_mu))
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "group": group,
            "exposure": exposure,
            "log_exposure": np.log(exposure),
        }
    )


# ---------------------------------------------------------------------------
# LMM offset tests
# ---------------------------------------------------------------------------


class TestLMMOffset:
    """Tests for offset in fit()."""

    def test_fit_accepts_offset(self):
        df = _lmm_dataset()
        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            offset=df["offset"].values,
        )
        assert result.converged

    def test_zero_offset_matches_no_offset(self):
        df = _lmm_dataset()
        result_no = interlace.fit(formula="y ~ x", data=df, groups="group")
        result_zero = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            offset=np.zeros(len(df)),
        )
        np.testing.assert_allclose(
            result_zero.fe_params.values,
            result_no.fe_params.values,
            atol=1e-8,
        )

    def test_offset_shifts_intercept(self):
        """A constant offset should shift the intercept by -offset."""
        df = _lmm_dataset()
        c = 3.0
        result_no = interlace.fit(formula="y ~ x", data=df, groups="group")
        result_off = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            offset=np.full(len(df), c),
        )
        # Intercept should decrease by c; slope unchanged
        np.testing.assert_allclose(
            result_off.fe_params.values[0],
            result_no.fe_params.values[0] - c,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result_off.fe_params.values[1],
            result_no.fe_params.values[1],
            atol=1e-6,
        )

    def test_offset_equivalent_to_known_covariate(self):
        """Using a variable as offset should match including it with coef fixed to 1.

        We test this by constructing y_adj = y - offset, fitting y_adj ~ x,
        and verifying we get the same fixed effects as fitting y ~ x with offset.
        """
        df = _lmm_dataset()
        off = df["offset"].values

        result_off = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            offset=off,
        )

        df_adj = df.copy()
        df_adj["y_adj"] = df["y"] - off
        result_adj = interlace.fit(
            formula="y_adj ~ x",
            data=df_adj,
            groups="group",
        )

        np.testing.assert_allclose(
            result_off.fe_params.values,
            result_adj.fe_params.values,
            atol=1e-6,
        )

    def test_offset_validation(self):
        df = _lmm_dataset()
        with pytest.raises(ValueError, match="length"):
            interlace.fit(
                formula="y ~ x",
                data=df,
                groups="group",
                offset=np.ones(5),
            )

    def test_fittedvalues_include_offset(self):
        """Fitted values should include the offset term."""
        df = _lmm_dataset()
        off = df["offset"].values
        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            offset=off,
        )
        # fittedvalues = X @ beta + Z @ blups + offset
        # Residuals should be y - fittedvalues (which already includes offset)
        resid = df["y"].values - result.fittedvalues
        np.testing.assert_allclose(resid, result.resid, atol=1e-8)


# ---------------------------------------------------------------------------
# GLMM offset tests
# ---------------------------------------------------------------------------


class TestGLMMOffset:
    """Tests for offset in glmer()."""

    def test_glmer_accepts_offset(self):
        df = _poisson_dataset()
        result = interlace.glmer(
            formula="y ~ x",
            data=df,
            family="poisson",
            groups="group",
            offset=df["log_exposure"].values,
        )
        assert result.converged

    def test_zero_offset_matches_no_offset(self):
        df = _poisson_dataset()
        result_no = interlace.glmer(
            formula="y ~ x",
            data=df,
            family="poisson",
            groups="group",
        )
        result_zero = interlace.glmer(
            formula="y ~ x",
            data=df,
            family="poisson",
            groups="group",
            offset=np.zeros(len(df)),
        )
        np.testing.assert_allclose(
            result_zero.fe_params.values,
            result_no.fe_params.values,
            atol=1e-6,
        )

    def test_offset_changes_estimates(self):
        """Non-zero offset should produce different estimates."""
        df = _poisson_dataset()
        result_no = interlace.glmer(
            formula="y ~ x",
            data=df,
            family="poisson",
            groups="group",
        )
        result_off = interlace.glmer(
            formula="y ~ x",
            data=df,
            family="poisson",
            groups="group",
            offset=df["log_exposure"].values,
        )
        assert not np.allclose(
            result_off.fe_params.values,
            result_no.fe_params.values,
            atol=1e-4,
        )

    def test_poisson_offset_rate_model(self):
        """With log(exposure) offset, Poisson GLMM estimates log-rates."""
        rng = np.random.default_rng(42)
        n_groups, n_per = 20, 50
        n = n_groups * n_per
        group = np.repeat(np.arange(n_groups), n_per)
        exposure = rng.uniform(1, 10, size=n)

        # True rate model: rate = exp(0.5), no covariates besides intercept
        true_log_rate = 0.5
        b = rng.normal(scale=0.2, size=n_groups)
        mu = exposure * np.exp(true_log_rate + b[group])
        y = rng.poisson(mu)

        df = pd.DataFrame(
            {
                "y": y,
                "group": group,
                "log_exposure": np.log(exposure),
            }
        )

        result = interlace.glmer(
            formula="y ~ 1",
            data=df,
            family="poisson",
            groups="group",
            offset=df["log_exposure"].values,
        )

        # Intercept should recover the true log-rate
        np.testing.assert_allclose(
            result.fe_params.values[0],
            true_log_rate,
            atol=0.15,
        )

    def test_offset_validation(self):
        df = _poisson_dataset()
        with pytest.raises(ValueError, match="length"):
            interlace.glmer(
                formula="y ~ x",
                data=df,
                family="poisson",
                groups="group",
                offset=np.ones(5),
            )
