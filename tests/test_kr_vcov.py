"""Tests for Kenward-Roger variance-covariance second derivatives.

Acceptance criteria:
  - kr_vcov_derivs returns a KRDerivatives dataclass
  - dC shape is (k, p, p), d2C shape is (k, k, p, p), Phi shape is (k, k)
  - d2C is symmetric: d2C[i,j] == d2C[j,i] for all i,j
  - diag(dC[i]) matches Satterthwaite gradient (consistency check)
  - d2C matches naive numerical second derivative of dC (accuracy check)
  - Phi is symmetric positive definite
  - Works for both 1-factor and 2-factor (crossed) models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace.kr_vcov import KRDerivatives, kr_vcov_derivs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_factor_result():
    """Single-factor model: y ~ x + (1|group), 15 groups x 8 obs."""
    rng = np.random.default_rng(42)
    n_groups, n_per = 15, 8
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per).astype(str)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.5, size=n_groups)
    y = (
        2.0
        + 1.5 * x
        + u[np.repeat(np.arange(n_groups), n_per)]
        + rng.normal(0, 0.6, size=n)
    )
    df = pd.DataFrame({"y": y, "x": x, "group": group})
    return interlace.fit("y ~ x", data=df, groups="group")


@pytest.fixture(scope="module")
def two_factor_result():
    """Crossed two-factor model: y ~ x + (1|g1) + (1|g2), 10 x 6 groups."""
    rng = np.random.default_rng(99)
    n = 200
    g1 = rng.choice([f"a{i}" for i in range(10)], size=n)
    g2 = rng.choice([f"b{j}" for j in range(6)], size=n)
    x = rng.standard_normal(n)
    u1 = rng.normal(0, 1.0, size=10)
    u2 = rng.normal(0, 0.8, size=6)
    u1_map = {f"a{i}": u1[i] for i in range(10)}
    u2_map = {f"b{j}": u2[j] for j in range(6)}
    y = (
        3.0
        + 2.0 * x
        + np.array([u1_map[g] for g in g1])
        + np.array([u2_map[g] for g in g2])
        + rng.normal(0, 0.5, size=n)
    )
    df = pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2})
    return interlace.fit("y ~ x", data=df, random=["(1|g1)", "(1|g2)"])


# ---------------------------------------------------------------------------
# Shape and type
# ---------------------------------------------------------------------------


class TestKRDerivativesShape:
    def test_returns_dataclass(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        assert isinstance(kr, KRDerivatives)

    def test_one_factor_shapes(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        k = len(one_factor_result.theta)
        p = one_factor_result.model.exog.shape[1]
        assert kr.dC.shape == (k, p, p)
        assert kr.d2C.shape == (k, k, p, p)
        assert kr.Phi.shape == (k, k)
        assert kr.fe_cov.shape == (p, p)

    def test_two_factor_shapes(self, two_factor_result):
        kr = kr_vcov_derivs(two_factor_result)
        k = len(two_factor_result.theta)
        p = two_factor_result.model.exog.shape[1]
        assert k == 2
        assert kr.dC.shape == (2, p, p)
        assert kr.d2C.shape == (2, 2, p, p)
        assert kr.Phi.shape == (2, 2)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


class TestSymmetry:
    def test_d2C_symmetric_in_theta_indices(self, two_factor_result):
        kr = kr_vcov_derivs(two_factor_result)
        np.testing.assert_allclose(kr.d2C[0, 1], kr.d2C[1, 0], atol=1e-10)

    def test_d2C_each_slice_is_symmetric_matrix(self, two_factor_result):
        kr = kr_vcov_derivs(two_factor_result)
        k = kr.d2C.shape[0]
        for i in range(k):
            for j in range(k):
                np.testing.assert_allclose(
                    kr.d2C[i, j],
                    kr.d2C[i, j].T,
                    atol=1e-10,
                    err_msg=f"d2C[{i},{j}] is not a symmetric p x p matrix",
                )

    def test_dC_each_slice_is_symmetric_matrix(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        for i in range(kr.dC.shape[0]):
            np.testing.assert_allclose(
                kr.dC[i],
                kr.dC[i].T,
                atol=1e-10,
                err_msg=f"dC[{i}] is not a symmetric p x p matrix",
            )

    def test_Phi_symmetric(self, two_factor_result):
        kr = kr_vcov_derivs(two_factor_result)
        np.testing.assert_allclose(kr.Phi, kr.Phi.T, atol=1e-10)

    def test_Phi_positive_definite(self, two_factor_result):
        kr = kr_vcov_derivs(two_factor_result)
        eigvals = np.linalg.eigvalsh(kr.Phi)
        assert np.all(eigvals > 0), (
            f"Phi not positive definite: eigenvalues = {eigvals}"
        )


# ---------------------------------------------------------------------------
# Consistency with Satterthwaite
# ---------------------------------------------------------------------------


class TestSatterthwaiteConsistency:
    def test_dC_diagonal_matches_satterthwaite_gradient(self, one_factor_result):
        """The diagonal of dC[i] should match the Satterthwaite gradient for θ_i."""
        import scipy.linalg as la
        import scipy.sparse as sp

        from interlace.profiled_reml import (
            _build_A11,
            _precompute,
            _sparse_solve,
            make_lambda,
        )

        result = one_factor_result
        y = result.model.endog
        X = result.model.exog
        Z = result._Z
        theta_hat = result.theta
        specs = result._random_specs
        n_levels = result._n_levels
        n, p = X.shape
        k = len(theta_hat)

        cache = _precompute(y, X, Z)
        ZtX = np.asarray(cache["ZtX"])
        Zty = np.asarray(cache["Zty"])
        XtX = np.asarray(cache["XtX"])
        Xty = np.asarray(cache["Xty"])
        yty = float(cache["yty"])

        def _fe_cov_diag(theta):
            Lambda = make_lambda(theta, specs, n_levels)
            A11 = _build_A11(sp.csc_matrix(cache["ZtZ"]), Lambda)
            lZtX = np.asarray(Lambda.T @ ZtX)
            lZty = np.asarray(Lambda.T @ Zty).squeeze()
            C_X = _sparse_solve(A11, lZtX)
            c1 = _sparse_solve(A11, lZty)
            MX = XtX - lZtX.T @ C_X
            rhs = Xty - lZtX.T @ c1
            beta = la.solve(MX, rhs, assume_a="pos")
            yPy = yty - lZty @ c1 - rhs @ beta
            sigma2 = yPy / (n - p)
            return sigma2 * np.diag(np.linalg.inv(MX))

        # Satterthwaite-style gradient of diag(C)
        h = 1e-4
        satt_grad = np.zeros((k, p))
        for i in range(k):
            tp = theta_hat.copy()
            tp[i] += h
            tm = theta_hat.copy()
            tm[i] -= h
            satt_grad[i] = (_fe_cov_diag(tp) - _fe_cov_diag(tm)) / (2.0 * h)

        # KR dC diagonal
        kr = kr_vcov_derivs(result)
        for i in range(k):
            kr_diag = np.diag(kr.dC[i])
            np.testing.assert_allclose(
                kr_diag,
                satt_grad[i],
                rtol=1e-4,
                err_msg=f"dC[{i}] diagonal does not match Satterthwaite gradient",
            )


# ---------------------------------------------------------------------------
# Numerical accuracy: d2C matches derivative-of-derivative
# ---------------------------------------------------------------------------


class TestNumericalAccuracy:
    def test_d2C_matches_derivative_of_dC(self, one_factor_result):
        """d2C[i,j] should match d(dC_i)/dθ_j computed by differencing dC."""
        import scipy.linalg as la
        import scipy.sparse as sp

        from interlace.profiled_reml import (
            _build_A11,
            _precompute,
            _sparse_solve,
            make_lambda,
        )

        result = one_factor_result
        y = result.model.endog
        X = result.model.exog
        Z = result._Z
        theta_hat = result.theta
        specs = result._random_specs
        n_levels = result._n_levels
        n, p = X.shape
        k = len(theta_hat)

        cache = _precompute(y, X, Z)
        ZtX = np.asarray(cache["ZtX"])
        Zty = np.asarray(cache["Zty"])
        XtX = np.asarray(cache["XtX"])
        Xty = np.asarray(cache["Xty"])
        yty = float(cache["yty"])

        def _fe_cov_full(theta):
            Lambda = make_lambda(theta, specs, n_levels)
            A11 = _build_A11(sp.csc_matrix(cache["ZtZ"]), Lambda)
            lZtX = np.asarray(Lambda.T @ ZtX)
            lZty = np.asarray(Lambda.T @ Zty).squeeze()
            C_X = _sparse_solve(A11, lZtX)
            c1 = _sparse_solve(A11, lZty)
            MX = XtX - lZtX.T @ C_X
            rhs = Xty - lZtX.T @ c1
            beta = la.solve(MX, rhs, assume_a="pos")
            yPy = yty - lZty @ c1 - rhs @ beta
            sigma2 = yPy / (n - p)
            return sigma2 * np.linalg.inv(MX)

        # Compute dC at theta +/- h_j, then difference to get d2C
        h_outer = 1e-3
        h_inner = 1e-4
        naive_d2C = np.zeros((k, k, p, p))
        for i in range(k):
            for j in range(k):
                # dC_i(theta + h_j ej) via central diff in theta_i
                tp_j = theta_hat.copy()
                tp_j[j] += h_outer
                tm_j = theta_hat.copy()
                tm_j[j] -= h_outer

                def _dC_i_at(theta_base, idx=i):
                    tp_i = theta_base.copy()
                    tp_i[idx] += h_inner
                    tm_i = theta_base.copy()
                    tm_i[idx] -= h_inner
                    return (_fe_cov_full(tp_i) - _fe_cov_full(tm_i)) / (2.0 * h_inner)

                naive_d2C[i, j] = (_dC_i_at(tp_j) - _dC_i_at(tm_j)) / (2.0 * h_outer)

        kr = kr_vcov_derivs(result)
        np.testing.assert_allclose(
            kr.d2C,
            naive_d2C,
            atol=1e-4,
            rtol=0.05,
            err_msg="d2C does not match naive derivative-of-derivative",
        )

    def test_fe_cov_matches_result(self, one_factor_result):
        """KRDerivatives.fe_cov should match the result's fe_cov."""
        kr = kr_vcov_derivs(one_factor_result)
        np.testing.assert_allclose(kr.fe_cov, one_factor_result.fe_cov, rtol=1e-10)


# ---------------------------------------------------------------------------
# Finite and non-degenerate values
# ---------------------------------------------------------------------------


class TestFiniteValues:
    def test_dC_all_finite(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        assert np.all(np.isfinite(kr.dC)), "dC contains non-finite values"

    def test_d2C_all_finite(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        assert np.all(np.isfinite(kr.d2C)), "d2C contains non-finite values"

    def test_Phi_all_finite(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        assert np.all(np.isfinite(kr.Phi)), "Phi contains non-finite values"

    def test_dC_not_all_zero(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        assert np.any(np.abs(kr.dC) > 1e-12), "dC is all zeros"

    def test_d2C_not_all_zero(self, one_factor_result):
        kr = kr_vcov_derivs(one_factor_result)
        assert np.any(np.abs(kr.d2C) > 1e-12), "d2C is all zeros"
