"""Tests for the small-q dense fast path in ``profiled_reml`` (interlace-72oc Step 2).

Mirrors the GLMM dense kernel tests: parity is checked at three layers --
direct sparse-vs-dense kernel calls, dispatcher routing, and end-to-end
``fit_reml`` parity against threshold=0 forced-sparse.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from interlace import profiled_reml
from interlace.profiled_reml import (
    _build_A11,
    _init_chol_factor,
    _precompute,
    _try_cholmod,
    fit_reml,
    make_lambda_diag,
)


@pytest.fixture()
def small_q_dataset(rng: np.random.Generator | None = None) -> dict:
    """Small intercept-only LMM (q=12, n=200, 1 factor)."""
    rng = np.random.default_rng(0)
    n, q = 200, 10
    sigma2 = 2.0
    sigma2_b = 1.0
    g = np.repeat(np.arange(q), n // q)
    b = rng.normal(scale=np.sqrt(sigma2_b), size=q)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ [1.5, 0.8] + b[g] + rng.normal(scale=np.sqrt(sigma2), size=n)
    rows = np.arange(n)
    Z = sp.csc_matrix((np.ones(n), (rows, g)), shape=(n, q))
    return {
        "y": y,
        "X": X,
        "Z": Z,
        "q_sizes": [q],
        "theta_test": np.array([np.sqrt(sigma2_b / sigma2) * 1.2]),
    }


class TestRemlObjectiveKernelParity:
    def test_sparse_dense_match_single_factor(self, small_q_dataset: dict) -> None:
        d = small_q_dataset
        cache_sp = _precompute(d["y"], d["X"], d["Z"])
        # Prime cholmod factor for fair comparison (mirrors fit_reml).
        cholmod = _try_cholmod()
        if cholmod is not None:
            A11_init = _build_A11(
                sp.csc_matrix(cache_sp["ZtZ"]),
                make_lambda_diag(d["theta_test"], d["q_sizes"]),
            )
            factor, api = _init_chol_factor(cholmod, A11_init)
            cache_sp["chol_factor"] = factor
            cache_sp["chol_api"] = api
        cache_de = _precompute(d["y"], d["X"], d["Z"])

        v_sp = profiled_reml.reml_objective_sparse(
            d["theta_test"],
            d["y"],
            d["X"],
            d["Z"],
            d["q_sizes"],
            _cache=cache_sp,
        )
        v_de = profiled_reml.reml_objective_dense(
            d["theta_test"],
            d["y"],
            d["X"],
            d["Z"],
            d["q_sizes"],
            _cache=cache_de,
        )
        np.testing.assert_allclose(v_de, v_sp, rtol=1e-10, atol=1e-12)


class TestRemlGradientKernelParity:
    def test_sparse_dense_match_single_factor(self, small_q_dataset: dict) -> None:
        d = small_q_dataset
        cache_sp = _precompute(d["y"], d["X"], d["Z"])
        cholmod = _try_cholmod()
        if cholmod is not None:
            A11_init = _build_A11(
                sp.csc_matrix(cache_sp["ZtZ"]),
                make_lambda_diag(d["theta_test"], d["q_sizes"]),
            )
            factor, api = _init_chol_factor(cholmod, A11_init)
            cache_sp["chol_factor"] = factor
            cache_sp["chol_api"] = api
        cache_de = _precompute(d["y"], d["X"], d["Z"])

        g_sp = profiled_reml.reml_gradient_sparse(
            d["theta_test"],
            d["y"],
            d["X"],
            d["Z"],
            d["q_sizes"],
            _cache=cache_sp,
        )
        g_de = profiled_reml.reml_gradient_dense(
            d["theta_test"],
            d["y"],
            d["X"],
            d["Z"],
            d["q_sizes"],
            _cache=cache_de,
        )
        np.testing.assert_allclose(g_de, g_sp, rtol=1e-10, atol=1e-12)


class TestDispatcherSelectsByQ:
    def test_objective_dispatches_dense_at_small_q(
        self, small_q_dataset: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {"sparse": 0, "dense": 0}
        orig_sp = profiled_reml.reml_objective_sparse
        orig_de = profiled_reml.reml_objective_dense

        def spy_sp(*a, **kw):
            called["sparse"] += 1
            return orig_sp(*a, **kw)

        def spy_de(*a, **kw):
            called["dense"] += 1
            return orig_de(*a, **kw)

        monkeypatch.setattr(profiled_reml, "reml_objective_sparse", spy_sp)
        monkeypatch.setattr(profiled_reml, "reml_objective_dense", spy_de)
        d = small_q_dataset
        profiled_reml.reml_objective(
            d["theta_test"],
            d["y"],
            d["X"],
            d["Z"],
            d["q_sizes"],
        )
        assert called["dense"] == 1
        assert called["sparse"] == 0

    def test_objective_dispatches_sparse_when_threshold_zero(
        self, small_q_dataset: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(profiled_reml, "_DENSE_Q_THRESHOLD", 0)
        called = {"sparse": 0, "dense": 0}
        orig_sp = profiled_reml.reml_objective_sparse
        orig_de = profiled_reml.reml_objective_dense

        def spy_sp(*a, **kw):
            called["sparse"] += 1
            return orig_sp(*a, **kw)

        def spy_de(*a, **kw):
            called["dense"] += 1
            return orig_de(*a, **kw)

        monkeypatch.setattr(profiled_reml, "reml_objective_sparse", spy_sp)
        monkeypatch.setattr(profiled_reml, "reml_objective_dense", spy_de)
        d = small_q_dataset
        profiled_reml.reml_objective(
            d["theta_test"],
            d["y"],
            d["X"],
            d["Z"],
            d["q_sizes"],
        )
        assert called["sparse"] == 1
        assert called["dense"] == 0


class TestEndToEndFitRemlParity:
    def test_fit_reml_dense_matches_sparse(
        self, small_q_dataset: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = small_q_dataset
        r_dense = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
        monkeypatch.setattr(profiled_reml, "_DENSE_Q_THRESHOLD", 0)
        r_sparse = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"])
        np.testing.assert_allclose(r_dense.beta, r_sparse.beta, rtol=1e-6)
        np.testing.assert_allclose(r_dense.theta, r_sparse.theta, rtol=1e-6)
        np.testing.assert_allclose(r_dense.sigma2, r_sparse.sigma2, rtol=1e-6)
        assert abs(r_dense.llf - r_sparse.llf) < 1e-6
