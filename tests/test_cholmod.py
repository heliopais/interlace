"""Tests for the CHOLMOD sparse Cholesky path in profiled_reml.

When sksparse is installed, CHOLMOD is used for the sparse Cholesky factorisation
in the REML objective: symbolic analysis once, cheap numeric refactorisation per
call.  When sksparse is absent, the code falls back to SuperLU.

Test strategy:
- Tests NOT guarded by importorskip: verify _try_cholmod() returns None without
  sksparse and that fit_reml still works via the SuperLU fallback (always run).
- Tests guarded by pytest.importorskip("sksparse"): verify that the CHOLMOD path
  produces numerically identical results to the SuperLU path.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from interlace.profiled_reml import (
    _build_A11,
    _precompute,
    _try_cholmod,
    fit_reml,
    make_lambda_diag,
    reml_objective,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_re_data():
    rng = np.random.default_rng(7)
    n, q = 160, 8
    group_codes = np.repeat(np.arange(q), n // q)
    b = rng.normal(0, 1.0, q)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ [2.0, 0.5] + b[group_codes] + rng.normal(0, 0.7, n)

    from interlace.sparse_z import build_indicator_matrix

    Z = build_indicator_matrix(group_codes, q)
    return {"y": y, "X": X, "Z": Z, "q_sizes": [q]}


# ---------------------------------------------------------------------------
# Always-run tests: _try_cholmod() and SuperLU fallback
# ---------------------------------------------------------------------------


def test_try_cholmod_returns_module_or_none():
    """_try_cholmod() must return the cholmod module or None — never raise."""
    result = _try_cholmod()
    assert result is None or hasattr(result, "cholesky")


def test_fit_reml_superlu_fallback_converges(single_re_data):
    """fit_reml must converge via SuperLU when CHOLMOD is disabled."""
    d = single_re_data
    with patch("interlace.profiled_reml._try_cholmod", return_value=None):
        result = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    assert result.converged
    assert result.theta[0] > 0
    assert np.isfinite(result.llf)


def test_reml_objective_superlu_fallback_finite(single_re_data):
    """reml_objective must return a finite value without a chol_factor in cache."""
    d = single_re_data
    cache = _precompute(d["y"], d["X"], d["Z"])
    # Ensure no chol_factor in cache (simulates SuperLU path)
    assert "chol_factor" not in cache
    val = reml_objective(np.ones(1), d["y"], d["X"], d["Z"], d["q_sizes"], cache)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# CHOLMOD-specific tests (skipped when sksparse is not installed)
# ---------------------------------------------------------------------------

_cholmod_available = pytest.mark.skipif(
    _try_cholmod() is None, reason="sksparse not installed"
)


@_cholmod_available
def test_try_cholmod_returns_cholmod_module():
    cholmod = _try_cholmod()
    assert cholmod is not None
    assert hasattr(cholmod, "cholesky")


@_cholmod_available
def test_fit_reml_cholmod_converges(single_re_data):
    d = single_re_data
    result = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    assert result.converged
    assert result.theta[0] > 0
    assert np.isfinite(result.llf)


@_cholmod_available
def test_fit_reml_cholmod_matches_superlu_beta(single_re_data):
    """CHOLMOD and SuperLU paths must agree on fixed-effect estimates."""
    d = single_re_data
    result_cholmod = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    with patch("interlace.profiled_reml._try_cholmod", return_value=None):
        result_superlu = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    np.testing.assert_allclose(result_cholmod.beta, result_superlu.beta, rtol=1e-5)


@_cholmod_available
def test_fit_reml_cholmod_matches_superlu_theta(single_re_data):
    d = single_re_data
    result_cholmod = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    with patch("interlace.profiled_reml._try_cholmod", return_value=None):
        result_superlu = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    np.testing.assert_allclose(result_cholmod.theta, result_superlu.theta, rtol=1e-4)


@_cholmod_available
def test_fit_reml_cholmod_matches_superlu_llf(single_re_data):
    d = single_re_data
    result_cholmod = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    with patch("interlace.profiled_reml._try_cholmod", return_value=None):
        result_superlu = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
    np.testing.assert_allclose(result_cholmod.llf, result_superlu.llf, rtol=1e-5)


@_cholmod_available
def test_reml_objective_cholmod_matches_superlu(single_re_data):
    """A single objective evaluation via CHOLMOD must match SuperLU."""
    d = single_re_data
    theta = np.array([1.2])

    # SuperLU path (no chol_factor in cache)
    cache_su = _precompute(d["y"], d["X"], d["Z"])
    val_su = reml_objective(theta, d["y"], d["X"], d["Z"], d["q_sizes"], cache_su)

    # CHOLMOD path (chol_factor populated by fit_reml mechanism)
    cache_ch = _precompute(d["y"], d["X"], d["Z"])
    cholmod = _try_cholmod()
    lambda_diag = make_lambda_diag(theta, d["q_sizes"])
    A11_0 = _build_A11(cache_ch["ZtZ"], lambda_diag)
    chol_factor = cholmod.cholesky(A11_0)
    cache_ch["chol_factor"] = chol_factor

    val_ch = reml_objective(theta, d["y"], d["X"], d["Z"], d["q_sizes"], cache_ch)

    np.testing.assert_allclose(val_ch, val_su, rtol=1e-10)
