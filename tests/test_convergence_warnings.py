"""Tests for convergence boundary warnings and CrossedLMEResult properties.

Covers:
- interlace-90l: is_singular property and boundary_flags on CrossedLMEResult
- interlace-o0g: fit() emits a UserWarning when model is singular
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace import isSingular

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_spec(n_terms: int, correlated: bool) -> MagicMock:
    spec = MagicMock()
    spec.n_terms = n_terms
    spec.correlated = correlated
    return spec


def _well_identified_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_groups, n_per = 20, 10
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 2.0, n_groups)
    y = 1.0 + u[np.repeat(np.arange(n_groups), n_per)] + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


# ---------------------------------------------------------------------------
# interlace-90l: is_singular property
# ---------------------------------------------------------------------------


class TestIsSingularProperty:
    def test_is_singular_false_on_well_identified_model(self):
        df = _well_identified_data()
        result = interlace.fit("y ~ x", data=df, groups="g")
        assert result.is_singular is False

    def test_is_singular_true_after_zeroing_theta(self):
        df = _well_identified_data()
        result = interlace.fit("y ~ x", data=df, groups="g")
        result.theta = np.zeros_like(result.theta)
        assert result.is_singular is True

    def test_is_singular_matches_isSingular_function(self):
        df = _well_identified_data()
        result = interlace.fit("y ~ x", data=df, groups="g")
        assert result.is_singular == isSingular(result)

    def test_is_singular_custom_tol(self):
        df = _well_identified_data()
        result = interlace.fit("y ~ x", data=df, groups="g")
        # With an absurdly large tol everything looks singular
        assert result.is_singular is False
        # With tol larger than the fitted theta the model appears singular
        large_tol = float(result.theta[0]) + 1.0
        assert isSingular(result, tol=large_tol) is True


# ---------------------------------------------------------------------------
# interlace-90l: boundary_flags property
# ---------------------------------------------------------------------------


class TestBoundaryFlags:
    def test_boundary_flags_keys_match_group_names(self):
        df = _well_identified_data()
        result = interlace.fit("y ~ x", data=df, groups="g")
        flags = result.boundary_flags
        assert isinstance(flags, dict)
        assert set(flags.keys()) == set(result.ngroups.keys())

    def test_boundary_flags_false_on_well_identified(self):
        df = _well_identified_data()
        result = interlace.fit("y ~ x", data=df, groups="g")
        assert all(v is False for v in result.boundary_flags.values())

    def test_boundary_flags_true_when_theta_zeroed(self):
        df = _well_identified_data()
        result = interlace.fit("y ~ x", data=df, groups="g")
        result.theta = np.zeros_like(result.theta)
        assert all(v is True for v in result.boundary_flags.values())

    def test_boundary_flags_crossed_re_second_at_boundary(self):
        """For two crossed REs, only the one whose theta is zero is flagged."""
        rng = np.random.default_rng(7)
        n_g1, n_g2, n_per = 15, 10, 5
        n = n_g1 * n_per
        g1 = np.repeat(np.arange(n_g1), n_per).astype(str)
        g2 = np.tile(np.arange(n_g2), n_g1 * n_per // n_g2 + 1)[:n].astype(str)
        u1 = rng.normal(0, 2.0, n_g1)
        y = u1[np.repeat(np.arange(n_g1), n_per)] + 0.3 * rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "g1": g1, "g2": g2})
        result = interlace.fit("y ~ 1", data=df, groups=["g1", "g2"])
        flags = result.boundary_flags
        # g1 has real variance — should not be flagged
        assert flags["g1"] is False
        # g2 may or may not be flagged depending on the data, but the flag
        # must match what isSingular reports for that spec individually
        theta_idx = 0
        for spec in result._random_specs:
            from interlace.profiled_reml import n_theta_for_spec

            n_theta_j = n_theta_for_spec(spec.n_terms, spec.correlated)
            theta_j = result.theta[theta_idx : theta_idx + n_theta_j]
            theta_idx += n_theta_j
            expected = bool(abs(theta_j[0]) < 1e-4)
            assert flags[spec.group] == expected


# ---------------------------------------------------------------------------
# interlace-o0g: fit() emits warning when model is singular
# ---------------------------------------------------------------------------


class TestFitSingularWarning:
    def test_no_warning_for_well_identified_model(self):
        df = _well_identified_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise
            interlace.fit("y ~ x", data=df, groups="g")

    def test_warning_message_contains_singular(self):
        """A model that converges to the boundary should emit a warning."""
        # Pure noise with many groups → optimizer collapses RE variance to 0
        rng = np.random.default_rng(99)
        n = 200
        g = np.repeat(np.arange(100), 2).astype(str)  # 100 groups, 2 obs each
        y = rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "g": g})
        with pytest.warns(UserWarning, match="singular"):
            result = interlace.fit("y ~ 1", data=df, groups="g")
        assert result.is_singular is True

    def test_warning_can_be_suppressed(self):
        rng = np.random.default_rng(99)
        n = 200
        g = np.repeat(np.arange(100), 2).astype(str)
        y = rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "g": g})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = interlace.fit("y ~ 1", data=df, groups="g")
        assert result.is_singular is True
