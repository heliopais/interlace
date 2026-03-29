"""Tests for simulate() and bootMer().

Acceptance criteria (interlace-s85):
  - simulate(nsim=1000) shape, reproducibility, mean ≈ fitted values, variance check
  - bootMer(B=200) returns BootResult with correct shape
  - bootMer CI has correct structure and contains the point estimate
  - bootMer is seeded and reproducible
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from interlace import fit
from interlace.simulate import BootResult, bootMer, simulate


@pytest.fixture(scope="module")
def simple_df() -> pd.DataFrame:
    """Single grouping factor; clean data for fast testing."""
    rng = np.random.default_rng(99)
    n_groups, n_per = 15, 8
    n = n_groups * n_per
    groups = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    b = rng.normal(0, 1.0, size=n_groups)
    y = 3.0 + 0.7 * x + b[groups] + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "grp": groups.astype(str)})


@pytest.fixture(scope="module")
def result(simple_df: pd.DataFrame):
    return fit("y ~ x", simple_df, groups="grp")


# ---------------------------------------------------------------------------
# simulate() — shape and type
# ---------------------------------------------------------------------------


class TestSimulateShape:
    def test_single_sim_shape(self, result) -> None:
        sims = simulate(result, nsim=1, seed=0)
        assert sims.shape == (result.nobs, 1)

    def test_multi_sim_shape(self, result) -> None:
        sims = simulate(result, nsim=50, seed=0)
        assert sims.shape == (result.nobs, 50)

    def test_returns_ndarray(self, result) -> None:
        sims = simulate(result, nsim=5, seed=0)
        assert isinstance(sims, np.ndarray)


# ---------------------------------------------------------------------------
# simulate() — reproducibility
# ---------------------------------------------------------------------------


class TestSimulateReproducibility:
    def test_same_seed_same_output(self, result) -> None:
        s1 = simulate(result, nsim=10, seed=42)
        s2 = simulate(result, nsim=10, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self, result) -> None:
        s1 = simulate(result, nsim=10, seed=1)
        s2 = simulate(result, nsim=10, seed=2)
        assert not np.allclose(s1, s2)

    def test_no_seed_differs_across_calls(self, result) -> None:
        s1 = simulate(result, nsim=10)
        s2 = simulate(result, nsim=10)
        # Extremely unlikely to be identical without a seed
        assert not np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# simulate() — statistical properties
# ---------------------------------------------------------------------------


class TestSimulateStatistics:
    def test_mean_approx_marginal_mean(self, result) -> None:
        """Column-mean of many simulations ≈ X @ beta (marginal mean, not conditional).

        simulate() draws from the marginal model (integrating out random effects),
        so E[y*] = X @ beta, not X @ beta + Z @ BLUP.
        """
        sims = simulate(result, nsim=2000, seed=7)
        sim_mean = sims.mean(axis=1)
        marginal_mean = result.model.exog @ np.asarray(result.fe_params)
        np.testing.assert_allclose(
            sim_mean,
            marginal_mean,
            atol=0.15,  # within 0.15 on average (stochastic; generous tolerance)
        )

    def test_variance_exceeds_residual_scale(self, result) -> None:
        """Marginal variance of simulations > sigma^2 (RE adds extra variance)."""
        sims = simulate(result, nsim=2000, seed=7)
        sim_var = sims.var(axis=1).mean()
        assert sim_var > result.scale

    def test_variance_roughly_matches_theory(self, result) -> None:
        """Marginal variance ≈ sigma^2 * (1 + sum(theta^2)) for intercept-only RE."""
        sims = simulate(result, nsim=3000, seed=7)
        sim_var = sims.var(axis=1).mean()
        # For single intercept-only RE: Var(y_i) = sigma^2 * (1 + theta^2)
        expected = result.scale * (1 + float(result.theta[0]) ** 2)
        # Allow 30% relative tolerance (sampling variance in the estimate)
        assert abs(sim_var - expected) / expected < 0.30


# ---------------------------------------------------------------------------
# bootMer() — basic structure
# ---------------------------------------------------------------------------


class TestBootMerStructure:
    def test_returns_boot_result(self, result) -> None:
        br = bootMer(result, B=10, seed=0)
        assert isinstance(br, BootResult)

    def test_estimates_shape_default_statistic(self, result) -> None:
        """Default statistic: [fe_params..., sqrt(sigma^2), theta...]."""
        br = bootMer(result, B=20, seed=0)
        n_fe = len(np.asarray(result.fe_params))
        n_theta = len(result.theta)
        expected_cols = n_fe + 1 + n_theta  # fe + sigma + theta
        assert br.estimates.shape == (20, expected_cols)

    def test_custom_statistic(self, result) -> None:
        br = bootMer(result, statistic=lambda r: np.asarray(r.fe_params), B=10, seed=0)
        n_fe = len(np.asarray(result.fe_params))
        assert br.estimates.shape == (10, n_fe)

    def test_scalar_statistic(self, result) -> None:
        br = bootMer(
            result, statistic=lambda r: np.array([float(r.scale)]), B=10, seed=0
        )
        assert br.estimates.shape == (10, 1)


# ---------------------------------------------------------------------------
# bootMer() — reproducibility
# ---------------------------------------------------------------------------


class TestBootMerReproducibility:
    def test_same_seed_same_estimates(self, result) -> None:
        br1 = bootMer(result, B=15, seed=5)
        br2 = bootMer(result, B=15, seed=5)
        np.testing.assert_array_equal(br1.estimates, br2.estimates)

    def test_different_seeds_differ(self, result) -> None:
        br1 = bootMer(result, B=15, seed=1)
        br2 = bootMer(result, B=15, seed=2)
        assert not np.allclose(br1.estimates, br2.estimates)


# ---------------------------------------------------------------------------
# BootResult.ci() — structure and validity
# ---------------------------------------------------------------------------


class TestBootResultCI:
    @pytest.fixture(scope="class")
    def boot_result(self, result) -> BootResult:
        return bootMer(
            result, statistic=lambda r: np.asarray(r.fe_params), B=200, seed=42
        )

    def test_ci_shape(self, boot_result, result) -> None:
        ci = boot_result.ci(method="perc")
        n_fe = len(np.asarray(result.fe_params))
        assert ci.shape == (n_fe, 2)

    def test_ci_lower_less_than_upper(self, boot_result) -> None:
        ci = boot_result.ci(method="perc")
        assert np.all(ci[:, 0] < ci[:, 1])

    def test_ci_95_wider_than_80(self, boot_result) -> None:
        ci_95 = boot_result.ci(method="perc", level=0.95)
        ci_80 = boot_result.ci(method="perc", level=0.80)
        widths_95 = ci_95[:, 1] - ci_95[:, 0]
        widths_80 = ci_80[:, 1] - ci_80[:, 0]
        assert np.all(widths_95 >= widths_80)

    def test_ci_contains_point_estimate(self, boot_result, result) -> None:
        """95% CI should contain the original fitted FE params."""
        ci = boot_result.ci(method="perc", level=0.95)
        fe = np.asarray(result.fe_params)
        assert np.all((fe >= ci[:, 0]) & (fe <= ci[:, 1]))

    def test_invalid_method_raises(self, boot_result) -> None:
        with pytest.raises(ValueError, match="method"):
            boot_result.ci(method="studentized")


# ---------------------------------------------------------------------------
# CrossedLMEResult method delegation
# ---------------------------------------------------------------------------


class TestResultMethodDelegation:
    def test_simulate_method_on_result(self, result) -> None:
        sims = result.simulate(nsim=5, seed=0)
        assert sims.shape == (result.nobs, 5)

    def test_bootMer_method_on_result(self, result) -> None:
        br = result.bootMer(B=5, seed=0)
        assert isinstance(br, BootResult)
