"""Tests for the small-q dense fast path in ``glmm_laplace`` (interlace-72oc).

Parity is tested at two layers:

  1. ``_laplace_objective_profiled_dense`` vs ``_laplace_objective_profiled_sparse``
     called directly with identical inputs captured from a real CBPP fit.
  2. End-to-end ``fit_glmm`` on CBPP (q=15, dispatched to dense) matches the
     theta/beta/llf returned when the threshold is forced to zero (sparse).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace import glmm_laplace

FIXTURES = Path("tests/fixtures")


@pytest.fixture()
def cbpp_data() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "glmm_cbpp_data.csv")
    df["period"] = df["period"].astype(str)
    df["herd"] = df["herd"].astype(str)
    return df


def _capture_first_kernel_args(cbpp_df: pd.DataFrame) -> tuple:
    """Run a CBPP fit with the dispatcher patched to record the first call's
    arguments, then return them.  Both kernels can be called with these args
    to compare results.
    """
    captured: dict = {}
    orig = glmm_laplace._laplace_objective_profiled

    def spy(*args, **kwargs):
        if not captured:
            captured["args"] = args
            captured["kwargs"] = kwargs
        return orig(*args, **kwargs)

    glmm_laplace._laplace_objective_profiled = spy
    try:
        interlace.glmer(
            "proportion ~ period",
            data=cbpp_df,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_df["size"], dtype=float),
        )
    finally:
        glmm_laplace._laplace_objective_profiled = orig
    return captured["args"], captured["kwargs"]


class TestLaplaceObjectiveProfiledKernelParity:
    """Dense kernel result must match sparse kernel at the same inputs."""

    def test_kernels_match_on_real_cbpp_call(self, cbpp_data: pd.DataFrame) -> None:
        args, kwargs = _capture_first_kernel_args(cbpp_data)
        # args are: theta_beta, n_theta, y, X, Z, family, specs, n_levels,
        #          weights, warm, offset, lambda_builder, cholmod_handle
        # We need to clone warm (mutated by the kernel) and drop cholmod_handle
        # for the dense call.
        theta_beta = args[0].copy()
        warm_sp: dict = {"u": None}
        warm_de: dict = {"u": None}

        ll_sp = glmm_laplace._laplace_objective_profiled_sparse(
            theta_beta,
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            warm_sp,
            offset=args[10],
            lambda_builder=args[11],
            cholmod_handle=args[12],
        )
        ll_de = glmm_laplace._laplace_objective_profiled_dense(
            theta_beta,
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            warm_de,
            offset=args[10],
            lambda_builder=args[11],
        )
        np.testing.assert_allclose(ll_de, ll_sp, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(warm_de["u"], warm_sp["u"], rtol=1e-8, atol=1e-10)

    def test_kernels_match_with_warm_start(self, cbpp_data: pd.DataFrame) -> None:
        """After a non-trivial warm-start u, both kernels must agree."""
        args, _ = _capture_first_kernel_args(cbpp_data)
        q = args[4].shape[1]
        rng = np.random.default_rng(0)
        u_warm = rng.standard_normal(q) * 0.3
        # Perturb theta_beta a bit too
        theta_beta = args[0] + rng.standard_normal(args[0].shape) * 0.05

        warm_sp: dict = {"u": u_warm.copy()}
        warm_de: dict = {"u": u_warm.copy()}
        ll_sp = glmm_laplace._laplace_objective_profiled_sparse(
            theta_beta,
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            warm_sp,
            offset=args[10],
            lambda_builder=args[11],
            cholmod_handle=args[12],
        )
        ll_de = glmm_laplace._laplace_objective_profiled_dense(
            theta_beta,
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            warm_de,
            offset=args[10],
            lambda_builder=args[11],
        )
        np.testing.assert_allclose(ll_de, ll_sp, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(warm_de["u"], warm_sp["u"], rtol=1e-8, atol=1e-10)


class TestDispatcherSelectsByQ:
    def test_dispatches_to_dense_at_small_q(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, int] = {"sparse": 0, "dense": 0}
        orig_sparse = glmm_laplace._laplace_objective_profiled_sparse
        orig_dense = glmm_laplace._laplace_objective_profiled_dense

        def spy_sparse(*a, **kw):
            called["sparse"] += 1
            return orig_sparse(*a, **kw)

        def spy_dense(*a, **kw):
            called["dense"] += 1
            return orig_dense(*a, **kw)

        monkeypatch.setattr(
            glmm_laplace, "_laplace_objective_profiled_sparse", spy_sparse
        )
        monkeypatch.setattr(
            glmm_laplace, "_laplace_objective_profiled_dense", spy_dense
        )
        interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        assert called["dense"] > 0, "dense kernel should be hit at q=15 (CBPP)"
        assert called["sparse"] == 0, (
            "sparse kernel should be skipped at q below threshold; "
            f"saw {called['sparse']} call(s)"
        )

    def test_dispatches_to_sparse_when_threshold_zero(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(glmm_laplace, "_DENSE_Q_THRESHOLD", 0)
        called: dict[str, int] = {"sparse": 0, "dense": 0}
        orig_sparse = glmm_laplace._laplace_objective_profiled_sparse
        orig_dense = glmm_laplace._laplace_objective_profiled_dense

        def spy_sparse(*a, **kw):
            called["sparse"] += 1
            return orig_sparse(*a, **kw)

        def spy_dense(*a, **kw):
            called["dense"] += 1
            return orig_dense(*a, **kw)

        monkeypatch.setattr(
            glmm_laplace, "_laplace_objective_profiled_sparse", spy_sparse
        )
        monkeypatch.setattr(
            glmm_laplace, "_laplace_objective_profiled_dense", spy_dense
        )
        interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        assert called["sparse"] > 0
        assert called["dense"] == 0


class TestEndToEndCBPPParity:
    def test_cbpp_theta_beta_llf_match(
        self, cbpp_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        r_dense = interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        monkeypatch.setattr(glmm_laplace, "_DENSE_Q_THRESHOLD", 0)
        r_sparse = interlace.glmer(
            "proportion ~ period",
            data=cbpp_data,
            family="binomial",
            groups="herd",
            weights=np.array(cbpp_data["size"], dtype=float),
        )
        np.testing.assert_allclose(
            r_dense.fe_params, r_sparse.fe_params, rtol=1e-4, atol=1e-5
        )
        assert abs(r_dense.llf - r_sparse.llf) < 1e-3
