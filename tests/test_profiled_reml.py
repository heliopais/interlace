"""Tests for profiled_reml.py — Lambda parameterisation, sparse Cholesky,
REML objective, and L-BFGS-B optimiser."""

import numpy as np
import pytest
import scipy.sparse as sp

from interlace.formula import RandomEffectSpec
from interlace.profiled_reml import (
    REMLResult,
    fit_reml,
    make_lambda,
    make_lambda_diag,
    n_theta_for_spec,
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

        if abs(sm_res.fe_params["Intercept"]) < 0.1:
            pytest.skip("statsmodels converged to degenerate solution on this platform")

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


class TestFitRemlWithSlopes:
    """fit_reml wired to use generalised Z and Lambda via specs."""

    @pytest.fixture()
    def slope_dataset(self) -> dict:
        """200 obs, 10 groups, independent random intercept + slope."""
        rng = np.random.default_rng(99)
        n, q = 200, 10
        group_codes = np.repeat(np.arange(q), n // q)
        x = rng.normal(size=n)
        b_int = rng.normal(scale=0.8, size=q)
        b_slope = rng.normal(scale=0.4, size=q)
        X = np.column_stack([np.ones(n), x])
        y = X @ np.array([1.0, 0.5]) + b_int[group_codes] + b_slope[group_codes] * x
        y += rng.normal(scale=1.0, size=n)
        import pandas as pd

        from interlace.formula import RandomEffectSpec
        from interlace.sparse_z import build_joint_z_from_specs

        df = pd.DataFrame({"g": group_codes.astype(str), "x": x})
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=True
        )
        Z = build_joint_z_from_specs([spec], df)
        return {"y": y, "X": X, "Z": Z, "spec": spec, "q": q}

    def test_converges_correlated_slope(self, slope_dataset: dict) -> None:

        d = slope_dataset
        spec = d["spec"]
        result = fit_reml(
            d["y"],
            d["X"],
            d["Z"],
            q_sizes=[],
            specs=[spec],
            n_levels=[d["q"]],
        )
        assert result.converged

    def test_finite_llf_correlated_slope(self, slope_dataset: dict) -> None:
        d = slope_dataset
        result = fit_reml(
            d["y"],
            d["X"],
            d["Z"],
            q_sizes=[],
            specs=[d["spec"]],
            n_levels=[d["q"]],
        )
        assert np.isfinite(result.llf)

    def test_theta_length_matches_specs(self, slope_dataset: dict) -> None:
        d = slope_dataset
        spec = d["spec"]
        result = fit_reml(
            d["y"],
            d["X"],
            d["Z"],
            q_sizes=[],
            specs=[spec],
            n_levels=[d["q"]],
        )
        # correlated intercept+slope: 3 theta params (l11, l21, l22)
        assert len(result.theta) == 3

    def test_specs_stored_on_result(self, slope_dataset: dict) -> None:
        d = slope_dataset
        spec = d["spec"]
        result = fit_reml(
            d["y"],
            d["X"],
            d["Z"],
            q_sizes=[],
            specs=[spec],
            n_levels=[d["q"]],
        )
        assert result.specs is not None
        assert result.n_levels == [d["q"]]

    def test_backward_compat_intercept_only_specs_matches_original(
        self, single_re_dataset: dict
    ) -> None:
        # fit_reml with intercept-only specs gives same result as plain q_sizes path
        from interlace.formula import RandomEffectSpec

        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result_old = fit_reml(d["y"], d["X"], Z, d["q_sizes"])

        spec = RandomEffectSpec(
            group="g", predictors=[], intercept=True, correlated=True
        )
        result_new = fit_reml(
            d["y"],
            d["X"],
            Z,
            q_sizes=[],
            specs=[spec],
            n_levels=[d["q_sizes"][0]],
        )
        np.testing.assert_allclose(result_old.beta, result_new.beta, rtol=1e-5)
        np.testing.assert_allclose(result_old.sigma2, result_new.sigma2, rtol=1e-5)


def _make_Z(codes: np.ndarray, n_levels: int) -> sp.csc_matrix:
    from interlace.sparse_z import build_indicator_matrix

    return build_indicator_matrix(codes, n_levels)


# ---------------------------------------------------------------------------
# n_theta_for_spec
# ---------------------------------------------------------------------------


class TestNThetaForSpec:
    def test_intercept_only(self) -> None:
        assert n_theta_for_spec(1, correlated=True) == 1

    def test_intercept_only_independent(self) -> None:
        assert n_theta_for_spec(1, correlated=False) == 1

    def test_correlated_two_terms(self) -> None:
        assert n_theta_for_spec(2, correlated=True) == 3  # l11, l21, l22

    def test_independent_two_terms(self) -> None:
        assert n_theta_for_spec(2, correlated=False) == 2

    def test_correlated_three_terms(self) -> None:
        assert n_theta_for_spec(3, correlated=True) == 6

    def test_independent_three_terms(self) -> None:
        assert n_theta_for_spec(3, correlated=False) == 3


# ---------------------------------------------------------------------------
# make_lambda
# ---------------------------------------------------------------------------


class TestMakeLambda:
    def test_intercept_only_is_diagonal(self) -> None:
        spec = RandomEffectSpec(
            group="g", predictors=[], intercept=True, correlated=True
        )
        theta = np.array([2.0])
        L = make_lambda(theta, [spec], n_levels=[4])
        assert L.shape == (4, 4)
        dense = L.toarray()
        np.testing.assert_allclose(dense, 2.0 * np.eye(4))

    def test_intercept_only_matches_make_lambda_diag(self) -> None:
        specs = [
            RandomEffectSpec(
                group="g1", predictors=[], intercept=True, correlated=True
            ),
            RandomEffectSpec(
                group="g2", predictors=[], intercept=True, correlated=True
            ),
        ]
        theta = np.array([3.0, 0.5])
        L = make_lambda(theta, specs, n_levels=[2, 4])
        diag_expected = make_lambda_diag(theta, [2, 4])
        np.testing.assert_allclose(L.diagonal(), diag_expected)

    def test_correlated_two_terms_structure(self) -> None:
        # L_j = [[l11, 0], [l21, l22]] ⊗ I_3
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=True
        )
        l11, l21, l22 = 1.5, 0.3, 0.8
        theta = np.array([l11, l21, l22])
        q = 3
        L = make_lambda(theta, [spec], n_levels=[q])
        assert L.shape == (6, 6)
        dense = L.toarray()
        # Top-left 3×3 block: l11 * I_3
        np.testing.assert_allclose(dense[:3, :3], l11 * np.eye(3))
        # Top-right 3×3 block: zeros
        np.testing.assert_allclose(dense[:3, 3:], np.zeros((3, 3)))
        # Bottom-left 3×3 block: l21 * I_3
        np.testing.assert_allclose(dense[3:, :3], l21 * np.eye(3))
        # Bottom-right 3×3 block: l22 * I_3
        np.testing.assert_allclose(dense[3:, 3:], l22 * np.eye(3))

    def test_independent_two_terms_is_block_diagonal(self) -> None:
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=False
        )
        t0, t1 = 1.2, 0.6
        theta = np.array([t0, t1])
        q = 3
        L = make_lambda(theta, [spec], n_levels=[q])
        assert L.shape == (6, 6)
        dense = L.toarray()
        # Top-left 3×3: t0 * I_3
        np.testing.assert_allclose(dense[:3, :3], t0 * np.eye(3))
        # Bottom-right 3×3: t1 * I_3
        np.testing.assert_allclose(dense[3:, 3:], t1 * np.eye(3))
        # Off-diagonal blocks: zeros
        np.testing.assert_allclose(dense[:3, 3:], np.zeros((3, 3)))
        np.testing.assert_allclose(dense[3:, :3], np.zeros((3, 3)))

    def test_total_size(self) -> None:
        specs = [
            RandomEffectSpec(
                group="g1", predictors=["x"], intercept=True, correlated=True
            ),
            RandomEffectSpec(
                group="g2", predictors=[], intercept=True, correlated=True
            ),
        ]
        # g1: 2 terms × 3 levels = 6; g2: 1 term × 4 levels = 4 → total 10
        theta = np.array([1.0, 0.5, 0.8, 1.2])  # 3 for g1 (corr) + 1 for g2
        L = make_lambda(theta, specs, n_levels=[3, 4])
        assert L.shape == (10, 10)

    def test_returns_sparse_csc(self) -> None:
        spec = RandomEffectSpec(
            group="g", predictors=["x"], intercept=True, correlated=True
        )
        theta = np.array([1.0, 0.0, 1.0])
        L = make_lambda(theta, [spec], n_levels=[3])
        assert sp.issparse(L)
        assert isinstance(L, sp.csc_matrix)
