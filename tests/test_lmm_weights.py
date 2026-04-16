"""Tests for observation-level weights in LMM fit()."""

import numpy as np
import pandas as pd
import pytest

import interlace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weighted_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """Create a dataset where some observations should count more than others.

    Groups of 20 obs each, 10 groups. Weights vary across observations.
    """
    n_groups = 10
    n_per_group = 20
    n = n_groups * n_per_group

    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=n)

    # True parameters
    beta0, beta1 = 2.0, 0.5
    sigma_b = 1.0
    sigma_e = 1.5

    b = rng.normal(scale=sigma_b, size=n_groups)
    y = beta0 + beta1 * x + b[group] + rng.normal(scale=sigma_e, size=n)

    # Weights: some observations are more precise (higher weight)
    weights = rng.uniform(0.5, 3.0, size=n)

    return pd.DataFrame({"y": y, "x": x, "group": group, "weights": weights})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeightsParameter:
    """Test that the weights parameter is accepted and affects the fit."""

    def test_fit_accepts_weights(self):
        """fit() should accept a weights parameter without error."""
        rng = np.random.default_rng(42)
        df = _make_weighted_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            weights=df["weights"].values,
        )

        assert result.converged
        assert hasattr(result, "fe_params")

    def test_unit_weights_match_unweighted(self):
        """Weights of all ones should give identical results to no weights."""
        rng = np.random.default_rng(42)
        df = _make_weighted_dataset(rng)

        result_no_wt = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
        )
        result_unit_wt = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            weights=np.ones(len(df)),
        )

        np.testing.assert_allclose(
            result_unit_wt.fe_params.values,
            result_no_wt.fe_params.values,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result_unit_wt.scale,
            result_no_wt.scale,
            rtol=1e-6,
        )

    def test_weights_change_estimates(self):
        """Non-uniform weights should produce different estimates than unweighted."""
        rng = np.random.default_rng(42)
        df = _make_weighted_dataset(rng)

        result_no_wt = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
        )
        result_wt = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            weights=df["weights"].values,
        )

        # Estimates should differ (not be identical)
        assert not np.allclose(
            result_wt.fe_params.values,
            result_no_wt.fe_params.values,
            atol=1e-6,
        )

    def test_double_weight_equals_duplicated_obs(self):
        """Doubling an observation's weight should be close to duplicating it.

        Not exact because duplication changes n (and thus REML df = n-p),
        but the cross-products are identical so fixed effects should be close.
        """
        rng = np.random.default_rng(42)
        n = 100
        group = np.repeat(np.arange(10), 10)
        x = rng.normal(size=n)
        b = rng.normal(scale=0.5, size=10)
        y = 1.0 + 0.5 * x + b[group] + rng.normal(scale=1.0, size=n)

        df = pd.DataFrame({"y": y, "x": x, "group": group})

        # Approach 1: weight the first 10 obs by 2
        weights = np.ones(n)
        weights[:10] = 2.0
        result_wt = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            weights=weights,
        )

        # Approach 2: duplicate the first 10 obs
        df_dup = pd.concat([df, df.iloc[:10]], ignore_index=True)
        result_dup = interlace.fit(
            formula="y ~ x",
            data=df_dup,
            groups="group",
        )

        # Not exactly equal because duplication changes n (REML df),
        # but should be close (~1% for fixed effects).
        np.testing.assert_allclose(
            result_wt.fe_params.values,
            result_dup.fe_params.values,
            rtol=0.01,
        )

    def test_weights_validation(self):
        """Weights must be positive and the right length."""
        rng = np.random.default_rng(42)
        df = _make_weighted_dataset(rng)

        # Wrong length
        with pytest.raises(ValueError, match="length"):
            interlace.fit(
                formula="y ~ x",
                data=df,
                groups="group",
                weights=np.ones(5),
            )

        # Negative weights
        with pytest.raises(ValueError, match="positive"):
            interlace.fit(
                formula="y ~ x",
                data=df,
                groups="group",
                weights=-np.ones(len(df)),
            )

    def test_weights_with_ml(self):
        """Weights should also work with method='ML'."""
        rng = np.random.default_rng(42)
        df = _make_weighted_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            groups="group",
            method="ML",
            weights=df["weights"].values,
        )

        assert result.converged

    def test_weights_with_random_slopes(self):
        """Weights should work with random slopes, not just intercepts."""
        rng = np.random.default_rng(42)
        df = _make_weighted_dataset(rng)

        result = interlace.fit(
            formula="y ~ x",
            data=df,
            random=["(1 + x | group)"],
            weights=df["weights"].values,
        )

        assert result.converged
