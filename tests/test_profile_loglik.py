"""Tests for profile_loglik in profiled_reml.py.

The profile log-likelihood L(theta) is obtained by maximising the ML
log-likelihood over beta and sigma^2 analytically, leaving a function only
of the variance parameters theta.  It satisfies:

    profile_loglik(theta) = -n/2 * (1 + log(2*pi/n)) - ml_objective(theta) / 2

which equals fit_ml().llf at the MLE theta_hat.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from interlace.profiled_reml import fit_ml, ml_objective, profile_loglik
from interlace.sparse_z import build_indicator_matrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def single_re_data(rng: np.random.Generator) -> dict:
    """200 obs, 1 covariate, 10 groups — simple intercept-only RE."""
    n, q = 200, 10
    sigma2 = 2.0
    sigma2_b = 1.0
    theta_true = np.sqrt(sigma2_b / sigma2)

    codes = np.repeat(np.arange(q), n // q)
    b = rng.normal(scale=np.sqrt(sigma2_b), size=q)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta = np.array([1.5, 0.8])
    y = X @ beta + b[codes] + rng.normal(scale=np.sqrt(sigma2), size=n)
    Z = build_indicator_matrix(codes, q)

    return {
        "y": y,
        "X": X,
        "Z": Z,
        "q_sizes": [q],
        "theta_true": np.array([theta_true]),
    }


@pytest.fixture()
def two_re_data(rng: np.random.Generator) -> dict:
    """400 obs, 2 crossed grouping factors (20 x 10 levels)."""
    n, q1, q2 = 400, 20, 10
    sigma2 = 1.5

    codes1 = np.tile(np.arange(q1), n // q1)
    codes2 = np.repeat(np.arange(q2), n // q2)
    b1 = rng.normal(scale=1.0, size=q1)
    b2 = rng.normal(scale=0.7, size=q2)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta = np.array([2.0, -0.5])
    y = X @ beta + b1[codes1] + b2[codes2] + rng.normal(scale=np.sqrt(sigma2), size=n)

    Z1 = build_indicator_matrix(codes1, q1)
    Z2 = build_indicator_matrix(codes2, q2)
    Z = sp.hstack([Z1, Z2], format="csc")

    return {
        "y": y,
        "X": X,
        "Z": Z,
        "q_sizes": [q1, q2],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfileLoglik:
    def test_at_optimum_equals_fit_ml_llf(self, single_re_data: dict) -> None:
        """profile_loglik(theta_hat) must equal fit_ml(...).llf to float precision."""
        d = single_re_data
        result = fit_ml(d["y"], d["X"], d["Z"], d["q_sizes"])
        theta_hat = result.theta

        ll = profile_loglik(theta_hat, d["y"], d["X"], d["Z"], d["q_sizes"])

        np.testing.assert_allclose(ll, result.llf, rtol=1e-10)

    def test_maximized_at_mle(self, single_re_data: dict) -> None:
        """theta_hat should yield the maximum of profile_loglik."""
        d = single_re_data
        result = fit_ml(d["y"], d["X"], d["Z"], d["q_sizes"])
        theta_hat = result.theta
        ll_hat = profile_loglik(theta_hat, d["y"], d["X"], d["Z"], d["q_sizes"])

        rng = np.random.default_rng(0)
        for _ in range(5):
            theta_perturbed = theta_hat + rng.uniform(0.1, 0.5, size=theta_hat.shape)
            ll_perturbed = profile_loglik(
                theta_perturbed, d["y"], d["X"], d["Z"], d["q_sizes"]
            )
            assert ll_hat >= ll_perturbed, (
                f"profile_loglik at MLE ({ll_hat:.4f}) < profile_loglik at "
                f"perturbed theta ({ll_perturbed:.4f})"
            )

    def test_monotone_with_ml_deviance(self, single_re_data: dict) -> None:
        """profile_loglik(theta) == -n/2*(1+log(2pi/n)) - ml_objective(theta)/2."""
        d = single_re_data
        n = len(d["y"])
        constant = -n / 2.0 * (1.0 + np.log(2.0 * np.pi / n))

        for theta in [np.array([0.5]), np.array([1.0]), np.array([2.0])]:
            deviance = ml_objective(theta, d["y"], d["X"], d["Z"], d["q_sizes"])
            expected = constant - deviance / 2.0
            actual = profile_loglik(theta, d["y"], d["X"], d["Z"], d["q_sizes"])
            np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_returns_neg_inf_when_ml_objective_is_inf(
        self, single_re_data: dict
    ) -> None:
        """When ml_objective returns inf (e.g. singular A11), return -inf."""
        from unittest.mock import patch

        d = single_re_data
        theta = np.array([1.0])
        with patch("interlace.profiled_reml.ml_objective", return_value=np.inf):
            ll = profile_loglik(theta, d["y"], d["X"], d["Z"], d["q_sizes"])
        assert ll == -np.inf

    def test_two_re_at_optimum(self, two_re_data: dict) -> None:
        """Works for two crossed random effects."""
        d = two_re_data
        result = fit_ml(d["y"], d["X"], d["Z"], d["q_sizes"])
        theta_hat = result.theta

        ll = profile_loglik(theta_hat, d["y"], d["X"], d["Z"], d["q_sizes"])

        np.testing.assert_allclose(ll, result.llf, rtol=1e-10)

    def test_uses_precomputed_cache(self, single_re_data: dict) -> None:
        """Passing _cache avoids redundant cross-product computation."""
        from interlace.profiled_reml import _precompute

        d = single_re_data
        cache = _precompute(d["y"], d["X"], d["Z"])
        theta = np.array([1.0])

        ll_cached = profile_loglik(theta, d["y"], d["X"], d["Z"], d["q_sizes"], cache)
        ll_nocache = profile_loglik(theta, d["y"], d["X"], d["Z"], d["q_sizes"])

        np.testing.assert_allclose(ll_cached, ll_nocache, rtol=1e-12)
