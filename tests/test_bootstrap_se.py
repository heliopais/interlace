"""Tests for CrossedLMEResult.bootstrap_se().

Acceptance criteria (GitHub issue #3):
  - Returns a float SE
  - resample_level="group" and resample_level="observation" produce different SEs
  - Seeding gives reproducible results
  - Invalid statistic or resample_level raises ValueError
  - Group-level SE > observation-level SE when group variance is substantial
    (group bootstrap correctly accounts for clustering)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from interlace import fit


@pytest.fixture(scope="module")
def grouped_df() -> pd.DataFrame:
    """20 firms, 10 obs each; substantial group variance."""
    rng = np.random.default_rng(42)
    n_groups, n_per = 20, 10
    n = n_groups * n_per
    groups = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    b = rng.normal(0, 1.5, size=n_groups)  # large group SD
    y = 2.0 + 0.5 * x + b[groups] + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "firm": groups.astype(str)})


@pytest.fixture(scope="module")
def result(grouped_df: pd.DataFrame):
    return fit("y ~ x", grouped_df, groups="firm")


class TestBootstrapSEBasic:
    def test_returns_float(self, result) -> None:
        se = result.bootstrap_se(statistic="median", n_bootstrap=100, seed=0)
        assert isinstance(se, float)

    def test_positive(self, result) -> None:
        se = result.bootstrap_se(statistic="median", n_bootstrap=100, seed=0)
        assert se > 0

    def test_reproducible_with_seed(self, result) -> None:
        se1 = result.bootstrap_se(statistic="median", n_bootstrap=200, seed=7)
        se2 = result.bootstrap_se(statistic="median", n_bootstrap=200, seed=7)
        assert se1 == se2

    def test_different_seeds_differ(self, result) -> None:
        se1 = result.bootstrap_se(statistic="median", n_bootstrap=200, seed=1)
        se2 = result.bootstrap_se(statistic="median", n_bootstrap=200, seed=2)
        # Very unlikely to be identical with different seeds
        assert se1 != se2


class TestBootstrapSEResampleLevel:
    def test_group_and_observation_differ(self, result) -> None:
        se_group = result.bootstrap_se(
            statistic="median", n_bootstrap=500, resample_level="group", seed=42
        )
        se_obs = result.bootstrap_se(
            statistic="median", n_bootstrap=500, resample_level="observation", seed=42
        )
        assert se_group != se_obs

    def test_group_se_larger_than_obs_se(self, result) -> None:
        """Group bootstrap must be larger because it correctly accounts for clustering.

        With large group variance (sigma_b=1.5 vs sigma_e=0.5), group-level
        resampling preserves the clustering structure and produces a larger SE.
        Observation-level resampling ignores clustering and underestimates SE.
        """
        se_group = result.bootstrap_se(
            statistic="median", n_bootstrap=1000, resample_level="group", seed=42
        )
        se_obs = result.bootstrap_se(
            statistic="median", n_bootstrap=1000, resample_level="observation", seed=42
        )
        assert se_group > se_obs

    def test_default_resample_level_is_group(self, result) -> None:
        se_default = result.bootstrap_se(statistic="median", n_bootstrap=200, seed=42)
        se_group = result.bootstrap_se(
            statistic="median", n_bootstrap=200, resample_level="group", seed=42
        )
        assert se_default == se_group


class TestBootstrapSEValidation:
    def test_invalid_statistic_raises(self, result) -> None:
        with pytest.raises(ValueError, match="statistic"):
            result.bootstrap_se(statistic="mean_absolute", n_bootstrap=10, seed=0)

    def test_invalid_resample_level_raises(self, result) -> None:
        with pytest.raises(ValueError, match="resample_level"):
            result.bootstrap_se(
                statistic="median",
                n_bootstrap=10,
                resample_level="cluster",
                seed=0,
            )
