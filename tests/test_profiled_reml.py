"""Tests for profiled_reml.py — Lambda parameterisation, sparse Cholesky,
REML objective, and L-BFGS-B optimiser."""

import numpy as np
import pytest
import scipy.sparse as sp

from interlace.profiled_reml import (
    REMLResult,
    fit_reml,
    make_lambda_diag,
    reml_objective,
    sparse_chol_logdet,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def single_re_dataset(rng: np.random.Generator) -> dict:
    """200 observations, 1 covariate, 10 groups, known variance components."""
    n, q, p = 200, 10, 2  # p includes intercept
    sigma2 = 2.0
    sigma2_b = 1.0  # RE variance
    theta_true = np.sqrt(sigma2_b / sigma2)  # = sqrt(0.5)

    group_codes = np.repeat(np.arange(q), n // q)
    b_true = rng.normal(scale=np.sqrt(sigma2_b), size=q)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([1.5, 0.8])
    y = X @ beta_true + b_true[group_codes] + rng.normal(scale=np.sqrt(sigma2), size=n)

    return {
        "y": y,
        "X": X,
        "group_codes": group_codes,
        "q_sizes": [q],
        "n": n,
        "p": p,
        "theta_true": np.array([theta_true]),
        "beta_true": beta_true,
        "sigma2_true": sigma2,
    }


# ---------------------------------------------------------------------------
# make_lambda_diag
# ---------------------------------------------------------------------------


class TestMakeLambdaDiag:
    def test_single_factor(self) -> None:
        diag = make_lambda_diag(np.array([2.0]), [3])
        np.testing.assert_array_equal(diag, [2.0, 2.0, 2.0])

    def test_two_factors(self) -> None:
        diag = make_lambda_diag(np.array([3.0, 0.5]), [2, 4])
        np.testing.assert_array_equal(diag, [3.0, 3.0, 0.5, 0.5, 0.5, 0.5])

    def test_length(self) -> None:
        diag = make_lambda_diag(np.array([1.0, 2.0, 3.0]), [5, 3, 7])
        assert len(diag) == 15

    def test_zero_theta(self) -> None:
        diag = make_lambda_diag(np.array([0.0]), [4])
        np.testing.assert_array_equal(diag, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# sparse_chol_logdet
# ---------------------------------------------------------------------------


class TestSparseCholLogdet:
    def test_identity_logdet(self) -> None:
        M = sp.eye(5, format="csc")
        logdet = sparse_chol_logdet(M)
        assert abs(logdet) < 1e-12

    def test_diagonal_matrix(self) -> None:
        diag_vals = np.array([2.0, 4.0, 8.0])
        M = sp.diags(diag_vals, format="csc")
        logdet = sparse_chol_logdet(M)
        expected = np.sum(np.log(diag_vals))
        assert abs(logdet - expected) < 1e-10

    def test_spd_matrix(self) -> None:
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        M = sp.csc_matrix(A)
        logdet = sparse_chol_logdet(M)
        expected = np.linalg.slogdet(A)[1]
        assert abs(logdet - expected) < 1e-10


# ---------------------------------------------------------------------------
# reml_objective
# ---------------------------------------------------------------------------


class TestRemlObjective:
    def test_returns_scalar(self, single_re_dataset: dict) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        val = reml_objective(d["theta_true"], d["y"], d["X"], Z, d["q_sizes"])
        assert np.isscalar(val) or val.ndim == 0

    def test_finite_at_truth(self, single_re_dataset: dict) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        val = reml_objective(d["theta_true"], d["y"], d["X"], Z, d["q_sizes"])
        assert np.isfinite(val)

    def test_objective_increases_away_from_optimum(
        self, single_re_dataset: dict
    ) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])

        def obj(t: float) -> float:
            return reml_objective(np.array([t]), d["y"], d["X"], Z, d["q_sizes"])

        # Evaluate on a grid; minimum should be near true theta
        thetas = np.linspace(0.05, 3.0, 30)
        vals = [obj(t) for t in thetas]
        min_idx = int(np.argmin(vals))
        # Minimum is in the interior, not at the boundary
        assert 0 < min_idx < len(thetas) - 1


# ---------------------------------------------------------------------------
# fit_reml
# ---------------------------------------------------------------------------


class TestFitReml:
    def test_returns_reml_result(self, single_re_dataset: dict) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        assert isinstance(result, REMLResult)

    def test_converged(self, single_re_dataset: dict) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        assert result.converged

    def test_beta_close_to_ols(self, single_re_dataset: dict) -> None:
        """Fixed effects should be close to OLS estimates (large sample)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        beta_ols = np.linalg.lstsq(d["X"], d["y"], rcond=None)[0]
        np.testing.assert_allclose(result.beta, beta_ols, atol=0.3)

    def test_theta_positive(self, single_re_dataset: dict) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        assert np.all(result.theta >= 0)

    def test_sigma2_positive(self, single_re_dataset: dict) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        assert result.sigma2 > 0

    def test_matches_statsmodels_single_re(self, single_re_dataset: dict) -> None:
        """REML estimates must match statsmodels MixedLM for single RE."""
        import pandas as pd
        import statsmodels.formula.api as smf

        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])

        df = pd.DataFrame(
            {
                "y": d["y"],
                "x1": d["X"][:, 1],
                "group": d["group_codes"].astype(str),
            }
        )
        sm_res = smf.mixedlm("y ~ x1", df, groups=df["group"]).fit(
            reml=True, method="lbfgs"
        )

        # Fixed effects
        np.testing.assert_allclose(
            result.beta,
            [sm_res.fe_params["Intercept"], sm_res.fe_params["x1"]],
            rtol=1e-3,
        )
        # Residual variance
        assert abs(result.sigma2 - sm_res.scale) / sm_res.scale < 0.01

    def test_llf_finite(self, single_re_dataset: dict) -> None:
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        assert np.isfinite(result.llf)

    def test_aic_bic_ordering(self, single_re_dataset: dict) -> None:
        """BIC >= AIC for datasets with n > e (which is always true here)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        # BIC penalises more than AIC when n > e^2 ~ 7.4
        assert result.bic >= result.aic


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_Z(codes: np.ndarray, n_levels: int) -> sp.csc_matrix:
    from interlace.sparse_z import build_indicator_matrix

    return build_indicator_matrix(codes, n_levels)
