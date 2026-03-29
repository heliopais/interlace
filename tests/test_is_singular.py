"""Tests for isSingular() boundary detection.

Acceptance criteria:
- Returns True when any variance component is at the boundary (theta diagonal ≈ 0)
- Returns False for well-identified models
- Respects the tol parameter
- Works for single RE, crossed RE, correlated slopes, independent slopes
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace import isSingular

# ---------------------------------------------------------------------------
# Helpers — build a minimal CrossedLMEResult-like mock with only the fields
# that isSingular() needs: theta, _random_specs, _n_levels
# ---------------------------------------------------------------------------


def _mock_spec(n_terms: int, correlated: bool) -> MagicMock:
    spec = MagicMock()
    spec.n_terms = n_terms
    spec.correlated = correlated
    return spec


def _mock_result(theta: list[float], specs_meta: list[tuple[int, bool]]) -> MagicMock:
    result = MagicMock()
    result.theta = np.array(theta, dtype=float)
    result._random_specs = [_mock_spec(n, c) for n, c in specs_meta]
    return result


# ---------------------------------------------------------------------------
# Single random intercept (p_j == 1)
# ---------------------------------------------------------------------------


def test_single_re_at_boundary():
    """theta = [0.0] → singular."""
    r = _mock_result([0.0], [(1, False)])
    assert isSingular(r) is True


def test_single_re_near_boundary():
    """theta = [5e-5] < default tol 1e-4 → singular."""
    r = _mock_result([5e-5], [(1, False)])
    assert isSingular(r) is True


def test_single_re_not_singular():
    """theta = [1.2] → not singular."""
    r = _mock_result([1.2], [(1, False)])
    assert isSingular(r) is False


def test_single_re_tol_respected():
    """theta = [1e-3] with tol=0.01 → singular; with tol=1e-4 → not."""
    r = _mock_result([1e-3], [(1, False)])
    assert isSingular(r, tol=0.01) is True
    assert isSingular(r, tol=1e-4) is False


# ---------------------------------------------------------------------------
# Crossed random intercepts (two specs, each p_j == 1)
# ---------------------------------------------------------------------------


def test_crossed_re_first_at_boundary():
    r = _mock_result([0.0, 1.0], [(1, False), (1, False)])
    assert isSingular(r) is True


def test_crossed_re_second_at_boundary():
    r = _mock_result([1.0, 0.0], [(1, False), (1, False)])
    assert isSingular(r) is True


def test_crossed_re_neither_singular():
    r = _mock_result([1.0, 0.8], [(1, False), (1, False)])
    assert isSingular(r) is False


# ---------------------------------------------------------------------------
# Correlated random slopes (p_j > 1, correlated=True)
# Theta layout for p_j=2: [L[0,0], L[1,0], L[1,1]]
# Diagonal positions: 0 and 2
# ---------------------------------------------------------------------------


def test_correlated_slopes_diagonal_zero():
    """L[0,0] = 0 → intercept variance is 0 → singular."""
    # theta = [L00, L10, L11]
    r = _mock_result([0.0, 0.5, 1.0], [(2, True)])
    assert isSingular(r) is True


def test_correlated_slopes_second_diagonal_zero():
    """L[1,1] = 0 → slope variance is 0 → singular."""
    r = _mock_result([1.0, 0.3, 0.0], [(2, True)])
    assert isSingular(r) is True


def test_correlated_slopes_not_singular():
    """Both diagonals nonzero → not singular (off-diagonal can be anything)."""
    r = _mock_result([1.0, 0.5, 0.8], [(2, True)])
    assert isSingular(r) is False


def test_correlated_slopes_p3_diagonal_positions():
    """p_j=3: theta has 6 entries, diagonal at positions 0, 2, 5."""
    # Not singular — all diagonals nonzero
    theta_ok = [1.0, 0.1, 0.9, 0.2, 0.3, 0.7]
    r = _mock_result(theta_ok, [(3, True)])
    assert isSingular(r) is False

    # Singular — position 5 (L[2,2]) is 0
    theta_bad = [1.0, 0.1, 0.9, 0.2, 0.3, 0.0]
    r2 = _mock_result(theta_bad, [(3, True)])
    assert isSingular(r2) is True


# ---------------------------------------------------------------------------
# Independent random slopes (p_j > 1, correlated=False)
# Theta layout: [theta_0, theta_1, ...] one per term
# ---------------------------------------------------------------------------


def test_independent_slopes_one_zero():
    """Second theta = 0 → singular."""
    r = _mock_result([1.0, 0.0], [(2, False)])
    assert isSingular(r) is True


def test_independent_slopes_not_singular():
    r = _mock_result([1.0, 0.5], [(2, False)])
    assert isSingular(r) is False


# ---------------------------------------------------------------------------
# Integration test — fit a real model where one grouping factor has negligible
# between-group variance, so the optimizer pushes theta → 0
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def singular_data() -> pd.DataFrame:
    """Dataset where group2 has essentially zero between-group variance."""
    rng = np.random.default_rng(0)
    n_groups = 30
    n_per = 5
    n = n_groups * n_per
    group1 = np.repeat(np.arange(n_groups), n_per).astype(str)
    # group2 is nearly random noise — each obs in its own tiny group
    group2 = np.tile(np.arange(n_per), n_groups).astype(str)
    x = rng.standard_normal(n)
    # y has real group1 effect but no group2 effect
    u1 = rng.normal(0, 2.0, n_groups)
    eps = rng.normal(0, 0.5, n)
    y = 1.0 + 0.8 * x + u1[np.repeat(np.arange(n_groups), n_per)] + eps
    return pd.DataFrame({"y": y, "x": x, "group1": group1, "group2": group2})


def test_real_fit_not_singular(singular_data):
    """Model with a genuine random effect should not be singular."""
    result = interlace.fit("y ~ x", data=singular_data, groups="group1")
    assert isSingular(result) is False


def test_real_fit_forced_zero_theta():
    """Zeroing theta on a real result makes isSingular return True."""
    rng = np.random.default_rng(1)
    n_groups = 20
    n_per = 10
    n = n_groups * n_per
    g = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 2.0, n_groups)  # strong group effect
    y = 1.0 + u[np.repeat(np.arange(n_groups), n_per)] + 0.3 * rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "x": x, "g": g})
    result = interlace.fit("y ~ x", data=df, groups="g")
    # Genuine group variance → not singular
    assert isSingular(result) is False
    # Force theta to zero → singular
    result.theta = np.zeros_like(result.theta)
    assert isSingular(result) is True
