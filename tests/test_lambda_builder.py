"""Tests for LambdaBuilder — cached-pattern Lambda construction.

Builder caches the structural sparse pattern of Lambda_theta (indices,
indptr) once and overwrites .data per call.  Numerical output must match
make_lambda() bitwise; the pattern arrays must be reused across calls.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from interlace.formula import RandomEffectSpec
from interlace.profiled_reml import LambdaBuilder, make_lambda


def _intercept(group: str) -> RandomEffectSpec:
    return RandomEffectSpec(
        group=group, predictors=[], intercept=True, correlated=True
    )


def _slope(group: str, predictor: str, correlated: bool = True) -> RandomEffectSpec:
    return RandomEffectSpec(
        group=group, predictors=[predictor], intercept=True, correlated=correlated
    )


class TestLambdaBuilderEquivalence:
    """Builder.update(theta) must equal make_lambda(theta, specs, n_levels)."""

    def test_single_intercept(self) -> None:
        specs = [_intercept("g")]
        n_levels = [10]
        theta = np.array([2.5])
        builder = LambdaBuilder(specs, n_levels)
        L_built = builder.update(theta)
        L_expected = make_lambda(theta, specs, n_levels)
        np.testing.assert_array_equal(L_built.toarray(), L_expected.toarray())

    def test_two_intercepts(self) -> None:
        specs = [_intercept("g1"), _intercept("g2")]
        n_levels = [4, 7]
        theta = np.array([0.8, 1.3])
        builder = LambdaBuilder(specs, n_levels)
        L_built = builder.update(theta)
        L_expected = make_lambda(theta, specs, n_levels)
        np.testing.assert_array_equal(L_built.toarray(), L_expected.toarray())

    def test_random_slopes_correlated(self) -> None:
        # Sleepstudy-shaped: (1 + Days | Subject), p=2 correlated
        specs = [_slope("Subject", "Days", correlated=True)]
        n_levels = [18]
        theta = np.array([1.5, 0.3, 0.8])  # L = [[1.5, 0], [0.3, 0.8]]
        builder = LambdaBuilder(specs, n_levels)
        L_built = builder.update(theta)
        L_expected = make_lambda(theta, specs, n_levels)
        np.testing.assert_array_equal(L_built.toarray(), L_expected.toarray())

    def test_random_slopes_independent(self) -> None:
        # (1 + x || Subject) — independent intercept and slope variances
        specs = [_slope("Subject", "x", correlated=False)]
        n_levels = [6]
        theta = np.array([1.1, 0.4])
        builder = LambdaBuilder(specs, n_levels)
        L_built = builder.update(theta)
        L_expected = make_lambda(theta, specs, n_levels)
        np.testing.assert_array_equal(L_built.toarray(), L_expected.toarray())

    def test_mixed_specs(self) -> None:
        specs = [
            _slope("g1", "x", correlated=True),  # 3 thetas, p=2
            _intercept("g2"),  # 1 theta, p=1
        ]
        n_levels = [5, 8]
        theta = np.array([1.0, 0.2, 0.7, 0.5])
        builder = LambdaBuilder(specs, n_levels)
        L_built = builder.update(theta)
        L_expected = make_lambda(theta, specs, n_levels)
        np.testing.assert_array_equal(L_built.toarray(), L_expected.toarray())

    def test_returns_csc(self) -> None:
        builder = LambdaBuilder([_intercept("g")], [3])
        L = builder.update(np.array([1.0]))
        assert isinstance(L, sp.csc_matrix)

    def test_negative_theta_off_diagonal(self) -> None:
        # Correlated case: off-diagonal L entries can legitimately be negative
        specs = [_slope("g", "x", correlated=True)]
        theta = np.array([1.2, -0.4, 0.9])  # negative L[1,0]
        builder = LambdaBuilder(specs, [4])
        L_built = builder.update(theta)
        L_expected = make_lambda(theta, specs, [4])
        np.testing.assert_array_equal(L_built.toarray(), L_expected.toarray())


class TestLambdaBuilderPatternReuse:
    """The whole point of the class: indices/indptr cached across calls."""

    def test_indices_array_reused_across_calls(self) -> None:
        builder = LambdaBuilder([_slope("g", "x", correlated=True)], [5])
        L1 = builder.update(np.array([1.0, 0.2, 0.8]))
        L2 = builder.update(np.array([2.0, -0.1, 1.3]))
        # Same structural arrays — not just equal, but the same object
        assert L1.indices is L2.indices
        assert L1.indptr is L2.indptr

    def test_only_data_changes_between_calls(self) -> None:
        builder = LambdaBuilder([_intercept("g")], [4])
        L1 = builder.update(np.array([1.0]))
        data1 = L1.data.copy()
        L2 = builder.update(np.array([3.5]))
        # Pattern unchanged
        np.testing.assert_array_equal(L1.indices, L2.indices)
        np.testing.assert_array_equal(L1.indptr, L2.indptr)
        # Values changed
        assert not np.array_equal(data1, L2.data)

    def test_repeated_updates_match_make_lambda(self) -> None:
        # Drive the builder through many theta values and confirm parity.
        rng = np.random.default_rng(42)
        specs = [
            _slope("g1", "x", correlated=True),
            _slope("g2", "y", correlated=False),
        ]
        n_levels = [12, 7]
        builder = LambdaBuilder(specs, n_levels)
        for _ in range(20):
            # 3 thetas (correlated p=2) + 2 thetas (independent p=2) = 5
            theta = rng.uniform(0.1, 2.0, size=5)
            np.testing.assert_array_equal(
                builder.update(theta).toarray(),
                make_lambda(theta, specs, n_levels).toarray(),
            )


class TestLambdaBuilderShape:
    def test_total_size(self) -> None:
        specs = [
            _slope("g1", "x", correlated=True),  # p=2, q=3 → 6
            _intercept("g2"),  # p=1, q=4 → 4
        ]
        builder = LambdaBuilder(specs, [3, 4])
        L = builder.update(np.array([1.0, 0.5, 0.8, 1.2]))
        assert L.shape == (10, 10)

    def test_empty_specs_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError, IndexError)):
            LambdaBuilder([], [])
