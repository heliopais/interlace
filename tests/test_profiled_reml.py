"""Tests for profiled_reml.py — Lambda parameterisation, sparse Cholesky,
REML objective, and L-BFGS-B optimiser."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from interlace.formula import RandomEffectSpec
from interlace.profiled_reml import (
    REMLResult,
    _build_A11,
    _init_chol_factor,
    _precompute,
    _sparse_solve,
    _try_cholmod,
    fit_ml,
    fit_reml,
    make_lambda,
    make_lambda_diag,
    ml_objective,
    n_theta_for_spec,
    reml_gradient,
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


class TestSparseSolveShape:
    """_sparse_solve must preserve the 2D shape of the rhs even for single-col."""

    def test_preserves_2d_with_single_column(self) -> None:
        # spla.spsolve squeezes (q, 1) rhs to (q,); _sparse_solve must not.
        A = sp.eye(5, format="csc")
        B = np.ones((5, 1))
        out = _sparse_solve(A, B)
        assert out.shape == (5, 1)

    def test_preserves_2d_with_multiple_columns(self) -> None:
        A = sp.eye(5, format="csc")
        B = np.ones((5, 3))
        out = _sparse_solve(A, B)
        assert out.shape == (5, 3)

    def test_returns_1d_for_1d_rhs(self) -> None:
        A = sp.eye(5, format="csc")
        b = np.ones(5)
        out = _sparse_solve(A, b)
        assert out.shape == (5,)


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


# ---------------------------------------------------------------------------
# BOBYQA optimizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_groups,n_obs", [(10, 200), (20, 400)])
class TestFitREMLBobyqa:
    """BOBYQA path produces results consistent with L-BFGS-B."""

    def _make_dataset(
        self, rng: np.random.Generator, n_obs: int, n_groups: int
    ) -> dict:
        sigma2, sigma2_b = 2.0, 1.0
        group_codes = np.repeat(np.arange(n_groups), n_obs // n_groups)
        b_true = rng.normal(scale=np.sqrt(sigma2_b), size=n_groups)
        X = np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])
        beta_true = np.array([1.5, 0.8])
        y = (
            X @ beta_true
            + b_true[group_codes]
            + rng.normal(scale=np.sqrt(sigma2), size=n_obs)
        )
        from interlace.sparse_z import build_indicator_matrix

        Z = build_indicator_matrix(group_codes, n_groups)
        q_sizes = [n_groups]
        return {"y": y, "X": X, "Z": Z, "q_sizes": q_sizes}

    def test_bobyqa_converges(
        self, rng: np.random.Generator, n_groups: int, n_obs: int
    ) -> None:
        pytest.importorskip("pybobyqa")
        d = self._make_dataset(rng, n_obs, n_groups)
        result = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="bobyqa")
        assert isinstance(result, REMLResult)
        assert result.converged

    def test_bobyqa_theta_close_to_lbfgsb(
        self, rng: np.random.Generator, n_groups: int, n_obs: int
    ) -> None:
        pytest.importorskip("pybobyqa")
        d = self._make_dataset(rng, n_obs, n_groups)
        r_lbfgsb = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="lbfgsb")
        r_bobyqa = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="bobyqa")
        np.testing.assert_allclose(r_bobyqa.theta, r_lbfgsb.theta, rtol=0.05)

    def test_bobyqa_beta_close_to_lbfgsb(
        self, rng: np.random.Generator, n_groups: int, n_obs: int
    ) -> None:
        pytest.importorskip("pybobyqa")
        d = self._make_dataset(rng, n_obs, n_groups)
        r_lbfgsb = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="lbfgsb")
        r_bobyqa = fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="bobyqa")
        np.testing.assert_allclose(r_bobyqa.beta, r_lbfgsb.beta, atol=1e-3)

    def test_unknown_optimizer_raises(
        self, rng: np.random.Generator, n_groups: int, n_obs: int
    ) -> None:
        d = self._make_dataset(rng, n_obs, n_groups)
        with pytest.raises(ValueError, match="optimizer"):
            fit_reml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="invalid")


# ---------------------------------------------------------------------------
# Fake CHOLMOD factor — backs the CHOLMOD API with scipy.sparse.linalg so the
# coverage paths for the optional sksparse dependency can be exercised.
# ---------------------------------------------------------------------------


class _FakeFactor:
    """Mimics a sksparse.cholmod Factor object using scipy sparse solvers."""

    def __init__(self, A: sp.csc_matrix) -> None:
        self._A = A.tocsc()

    def cholesky(self, A: sp.csc_matrix) -> None:  # in-place numeric refactor
        self._A = A.tocsc()

    def logdet(self) -> float:
        return sparse_chol_logdet(self._A)

    def solve_A(self, b: np.ndarray) -> np.ndarray:
        b_arr = np.asarray(b)
        if b_arr.ndim == 2:
            return np.column_stack(
                [
                    np.asarray(spla.spsolve(self._A, b_arr[:, i]))
                    for i in range(b_arr.shape[1])
                ]
            )
        return np.asarray(spla.spsolve(self._A, b_arr))


class _FakeCholmod:
    """Mimics the sksparse.cholmod module (just the `cholesky` entry point)."""

    @staticmethod
    def cholesky(A: sp.csc_matrix) -> _FakeFactor:
        return _FakeFactor(A)


# ---------------------------------------------------------------------------
# _try_cholmod — line 173: `return cholmod` (only reachable when sksparse is
# installed; we mock sys.modules to exercise it without the real package)
# ---------------------------------------------------------------------------


class TestTryCholmod:
    def test_returns_none_without_sksparse(self) -> None:
        # Force ImportError even if sksparse is installed in the dev venv:
        # setting sys.modules[name] = None makes the import system raise
        # ImportError on `from sksparse import cholmod`, which is what
        # _try_cholmod is designed to catch.
        with patch.dict(sys.modules, {"sksparse": None, "sksparse.cholmod": None}):
            result = _try_cholmod()
        assert result is None

    def test_returns_module_when_sksparse_available(self) -> None:
        mock_cholmod = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "sksparse": MagicMock(cholmod=mock_cholmod),
                "sksparse.cholmod": mock_cholmod,
            },
        ):
            result = _try_cholmod()
        assert result is mock_cholmod


# ---------------------------------------------------------------------------
# reml_objective edge cases (lines 354-357, 370-371, 375)
# ---------------------------------------------------------------------------


class TestRemlObjectiveEdgeCases:
    def test_linalg_error_returns_inf(self, single_re_dataset: dict) -> None:
        """LinAlgError from la.solve → objective returns inf (lines 370-371)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        with patch(
            "interlace.profiled_reml.la.solve", side_effect=la.LinAlgError("test")
        ):
            val = reml_objective(d["theta_true"], d["y"], d["X"], Z, d["q_sizes"])
        assert val == np.inf

    def test_zero_y_ypy_nonpositive_returns_inf(self, single_re_dataset: dict) -> None:
        """y=0 makes y'Py=0 → objective returns inf (line 375)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        y_zero = np.zeros(len(d["y"]))
        val = reml_objective(d["theta_true"], y_zero, d["X"], Z, d["q_sizes"])
        assert val == np.inf

    def test_chol_factor_in_cache_used(self, single_re_dataset: dict) -> None:
        """When cache has chol_factor the CHOLMOD branch is taken (lines 354-357)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        cache = _precompute(d["y"], d["X"], Z)
        lambda_diag = make_lambda_diag(d["theta_true"], d["q_sizes"])
        A11_init = _build_A11(cache["ZtZ"], lambda_diag)
        cache["chol_factor"] = _FakeFactor(A11_init)
        val = reml_objective(
            d["theta_true"], d["y"], d["X"], Z, d["q_sizes"], _cache=cache
        )
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# fit_reml with mocked CHOLMOD (lines 449-450, 460)
# Side-effect: also covers the CHOLMOD branch inside reml_objective (354-357)
# ---------------------------------------------------------------------------


class TestFitRemlCholmodMock:
    def test_lbfgsb_with_cholmod_mock_qsizes_path(
        self, single_re_dataset: dict
    ) -> None:
        """Mocked CHOLMOD, q_sizes path: exercises the hasattr/cache lines."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        with patch("interlace.profiled_reml._try_cholmod", return_value=_FakeCholmod()):
            result = fit_reml(d["y"], d["X"], Z, d["q_sizes"])
        assert isinstance(result, REMLResult)
        assert result.converged

    def test_lbfgsb_with_cholmod_mock_specs_path(self, single_re_dataset: dict) -> None:
        """Mocked CHOLMOD, specs path: exercises lines 449-450 and 460."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        spec = RandomEffectSpec(
            group="g", predictors=[], intercept=True, correlated=True
        )
        with patch("interlace.profiled_reml._try_cholmod", return_value=_FakeCholmod()):
            result = fit_reml(
                d["y"],
                d["X"],
                Z,
                q_sizes=[],
                specs=[spec],
                n_levels=[d["q_sizes"][0]],
            )
        assert isinstance(result, REMLResult)


# ---------------------------------------------------------------------------
# ml_objective (lines 580, 604-607, 618-619, 623)
# ---------------------------------------------------------------------------


class TestMlObjective:
    def test_returns_finite_without_cache(self, single_re_dataset: dict) -> None:
        """Call without _cache → line 580 executed."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        val = ml_objective(d["theta_true"], d["y"], d["X"], Z, d["q_sizes"])
        assert np.isfinite(val)

    def test_linalg_error_returns_inf(self, single_re_dataset: dict) -> None:
        """LinAlgError from la.solve → ml_objective returns inf (lines 618-619)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        with patch(
            "interlace.profiled_reml.la.solve", side_effect=la.LinAlgError("test")
        ):
            val = ml_objective(d["theta_true"], d["y"], d["X"], Z, d["q_sizes"])
        assert val == np.inf

    def test_zero_y_ypy_nonpositive_returns_inf(self, single_re_dataset: dict) -> None:
        """y=0 makes y'Py=0 → ml_objective returns inf (line 623)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        y_zero = np.zeros(len(d["y"]))
        val = ml_objective(d["theta_true"], y_zero, d["X"], Z, d["q_sizes"])
        assert val == np.inf

    def test_chol_factor_in_cache_used(self, single_re_dataset: dict) -> None:
        """When cache has chol_factor the CHOLMOD branch is taken (lines 604-607)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        cache = _precompute(d["y"], d["X"], Z)
        lambda_diag = make_lambda_diag(d["theta_true"], d["q_sizes"])
        A11_init = _build_A11(cache["ZtZ"], lambda_diag)
        cache["chol_factor"] = _FakeFactor(A11_init)
        val = ml_objective(
            d["theta_true"], d["y"], d["X"], Z, d["q_sizes"], _cache=cache
        )
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# fit_ml (lines 661-664, 683-684, 694, 704-709, 712-717)
# ---------------------------------------------------------------------------


class TestFitMl:
    def test_invalid_optimizer_raises(self, single_re_dataset: dict) -> None:
        """fit_ml raises ValueError for unknown optimizer (lines 661-664)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        with pytest.raises(ValueError, match="optimizer"):
            fit_ml(d["y"], d["X"], Z, d["q_sizes"], optimizer="invalid")

    def test_nelder_mead_returns_result(self, single_re_dataset: dict) -> None:
        """nelder-mead optimizer path in fit_ml (lines 712-717)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        result = fit_ml(d["y"], d["X"], Z, d["q_sizes"], optimizer="nelder-mead")
        assert isinstance(result, REMLResult)
        assert np.isfinite(result.llf)

    def test_lbfgsb_with_cholmod_mock_qsizes_path(
        self, single_re_dataset: dict
    ) -> None:
        """Mocked CHOLMOD, q_sizes path: exercises cholmod block in fit_ml."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        with patch("interlace.profiled_reml._try_cholmod", return_value=_FakeCholmod()):
            result = fit_ml(d["y"], d["X"], Z, d["q_sizes"])
        assert isinstance(result, REMLResult)
        assert np.isfinite(result.llf)

    def test_lbfgsb_with_cholmod_mock_specs_path(self, single_re_dataset: dict) -> None:
        """Mocked CHOLMOD, specs path: exercises lines 683-684 and 694."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        spec = RandomEffectSpec(
            group="g", predictors=[], intercept=True, correlated=True
        )
        with patch("interlace.profiled_reml._try_cholmod", return_value=_FakeCholmod()):
            result = fit_ml(
                d["y"],
                d["X"],
                Z,
                q_sizes=[],
                specs=[spec],
                n_levels=[d["q_sizes"][0]],
            )
        assert isinstance(result, REMLResult)


@pytest.mark.parametrize("n_groups,n_obs", [(10, 200)])
class TestFitMlBobyqa:
    """BOBYQA path in fit_ml (lines 704-709) — skipped when pybobyqa absent."""

    def _make_dataset(
        self, rng: np.random.Generator, n_obs: int, n_groups: int
    ) -> dict:
        sigma2, sigma2_b = 2.0, 1.0
        group_codes = np.repeat(np.arange(n_groups), n_obs // n_groups)
        b_true = rng.normal(scale=np.sqrt(sigma2_b), size=n_groups)
        X = np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])
        y = (
            X @ np.array([1.5, 0.8])
            + b_true[group_codes]
            + rng.normal(scale=np.sqrt(sigma2), size=n_obs)
        )
        from interlace.sparse_z import build_indicator_matrix

        Z = build_indicator_matrix(group_codes, n_groups)
        return {"y": y, "X": X, "Z": Z, "q_sizes": [n_groups]}

    def test_bobyqa_converges(
        self, rng: np.random.Generator, n_groups: int, n_obs: int
    ) -> None:
        pytest.importorskip("pybobyqa")
        d = self._make_dataset(rng, n_obs, n_groups)
        result = fit_ml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="bobyqa")
        assert isinstance(result, REMLResult)
        assert result.converged

    def test_bobyqa_beta_close_to_lbfgsb(
        self, rng: np.random.Generator, n_groups: int, n_obs: int
    ) -> None:
        pytest.importorskip("pybobyqa")
        d = self._make_dataset(rng, n_obs, n_groups)
        r_lbfgsb = fit_ml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="lbfgsb")
        r_bobyqa = fit_ml(d["y"], d["X"], d["Z"], d["q_sizes"], optimizer="bobyqa")
        np.testing.assert_allclose(r_bobyqa.beta, r_lbfgsb.beta, atol=1e-3)


# ---------------------------------------------------------------------------
# reml_gradient — analytical gradient of the profiled REML deviance
# ---------------------------------------------------------------------------


class TestRemlGradient:
    """Verify reml_gradient matches finite differences and improves optimisation."""

    def test_gradient_matches_fd_single_factor(self, single_re_dataset: dict) -> None:
        """Gradient matches scipy finite-diff to rtol=1e-4 for a 1-factor model."""
        import scipy.optimize as opt

        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        cache = _precompute(d["y"], d["X"], Z)

        def obj(theta: np.ndarray) -> float:
            return reml_objective(theta, d["y"], d["X"], Z, d["q_sizes"], _cache=cache)

        def grad(theta: np.ndarray) -> np.ndarray:
            return reml_gradient(theta, d["y"], d["X"], Z, d["q_sizes"], _cache=cache)

        theta_test = d["theta_true"] * 1.2
        err = opt.check_grad(obj, grad, theta_test)
        fd_norm = np.linalg.norm(opt.approx_fprime(theta_test, obj, 1e-5))
        assert err / (fd_norm + 1e-10) < 1e-3

    def test_gradient_matches_fd_two_factors(self, rng: np.random.Generator) -> None:
        """Gradient matches scipy finite-diff to rtol=1e-3 for a 2-factor model."""
        import scipy.optimize as opt

        n, q1, q2 = 240, 8, 6
        sigma2, sb1, sb2 = 2.0, 1.0, 0.5
        gc1 = np.repeat(np.arange(q1), n // q1)
        gc2 = np.tile(np.arange(q2), n // q2)
        b1 = rng.normal(scale=np.sqrt(sb1), size=q1)
        b2 = rng.normal(scale=np.sqrt(sb2), size=q2)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = (
            X @ [1.5, 0.8]
            + b1[gc1]
            + b2[gc2]
            + rng.normal(scale=np.sqrt(sigma2), size=n)
        )

        Z1 = _make_Z(gc1, q1)
        Z2 = _make_Z(gc2, q2)
        Z = sp.hstack([Z1, Z2], format="csc")
        q_sizes = [q1, q2]
        cache = _precompute(y, X, Z)

        def obj(theta: np.ndarray) -> float:
            return reml_objective(theta, y, X, Z, q_sizes, _cache=cache)

        def grad(theta: np.ndarray) -> np.ndarray:
            return reml_gradient(theta, y, X, Z, q_sizes, _cache=cache)

        theta_test = np.array(
            [np.sqrt(sb1 / sigma2) * 1.1, np.sqrt(sb2 / sigma2) * 0.9]
        )
        err = opt.check_grad(obj, grad, theta_test)
        fd_norm = np.linalg.norm(opt.approx_fprime(theta_test, obj, 1e-5))
        assert err / (fd_norm + 1e-10) < 1e-3

    def test_fit_reml_with_gradient_matches_without(
        self, single_re_dataset: dict
    ) -> None:
        """fit_reml with jac=reml_gradient gives same beta/theta/llf as without."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])

        r_no_jac = fit_reml(d["y"], d["X"], Z, d["q_sizes"], use_gradient=False)
        r_with_jac = fit_reml(d["y"], d["X"], Z, d["q_sizes"], use_gradient=True)

        np.testing.assert_allclose(r_with_jac.beta, r_no_jac.beta, rtol=1e-4)
        np.testing.assert_allclose(r_with_jac.theta, r_no_jac.theta, rtol=1e-4)
        assert abs(r_with_jac.llf - r_no_jac.llf) < 1e-4

    def test_gradient_reduces_eval_count(self, single_re_dataset: dict) -> None:
        """fit_reml with gradient uses fewer objective evaluations than without."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])

        counts: dict[str, int] = {"no_jac": 0, "with_jac": 0}

        orig = reml_objective

        def counting_obj_no_jac(
            theta: np.ndarray, *args: object, **kw: object
        ) -> float:
            counts["no_jac"] += 1
            return orig(theta, *args, **kw)  # type: ignore[arg-type]

        def counting_obj_with_jac(
            theta: np.ndarray, *args: object, **kw: object
        ) -> float:
            counts["with_jac"] += 1
            return orig(theta, *args, **kw)  # type: ignore[arg-type]

        import unittest.mock as mock

        with mock.patch("interlace.profiled_reml.reml_objective", counting_obj_no_jac):
            fit_reml(d["y"], d["X"], Z, d["q_sizes"], use_gradient=False)
        with mock.patch(
            "interlace.profiled_reml.reml_objective", counting_obj_with_jac
        ):
            fit_reml(d["y"], d["X"], Z, d["q_sizes"], use_gradient=True)

        # Using gradient should require fewer objective evaluations
        assert counts["with_jac"] < counts["no_jac"]

    def test_gradient_reuses_cached_cholmod_factor(
        self, single_re_dataset: dict
    ) -> None:
        """When _cache holds a cholmod factor, reml_gradient must reuse it
        (no extra splu refactorisations) and produce a numerically identical
        result vs the splu-fallback path.

        Drives interlace-mxzk Phase A: the dense ``A11_inv = lu.solve(eye(q))``
        path triggers three separate splu factorisations per gradient call;
        with a cached factor we expect zero splu calls.
        """
        cholmod = _try_cholmod()
        if cholmod is None:
            pytest.skip("sksparse.cholmod not installed")

        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        theta = d["theta_true"] * 1.2

        # Path 1: no cached factor -> splu fallback (current behaviour).
        cache_no_chol = _precompute(d["y"], d["X"], Z)
        g_splu = reml_gradient(theta, d["y"], d["X"], Z, d["q_sizes"], cache_no_chol)

        # Path 2: prime _cache with a cholmod factor (mirrors fit_reml).
        cache_chol = _precompute(d["y"], d["X"], Z)
        ZtZ = sp.csc_matrix(cache_chol["ZtZ"])
        A11_init = _build_A11(ZtZ, make_lambda_diag(theta, d["q_sizes"]))
        factor, api = _init_chol_factor(cholmod, A11_init)
        assert factor is not None, "cholmod factor init failed"
        cache_chol["chol_factor"] = factor
        cache_chol["chol_api"] = api

        with patch.object(spla, "splu", wraps=spla.splu) as splu_spy:
            g_chol = reml_gradient(theta, d["y"], d["X"], Z, d["q_sizes"], cache_chol)

        # No splu calls when a cached cholmod factor is available.
        assert splu_spy.call_count == 0, (
            f"reml_gradient should not call splu when a cached cholmod factor "
            f"is in _cache (saw {splu_spy.call_count} call(s))"
        )
        np.testing.assert_allclose(g_chol, g_splu, rtol=1e-10, atol=1e-12)

    def test_gradient_splu_fallback_factorises_once(
        self, single_re_dataset: dict
    ) -> None:
        """Without cached cholmod, splu fallback should factorise A11 only
        once per gradient call (current code factorises three times)."""
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        theta = d["theta_true"] * 1.2
        cache_no_chol = _precompute(d["y"], d["X"], Z)
        # Strip any cholmod factor a fixture might have added.
        cache_no_chol.pop("chol_factor", None)
        cache_no_chol.pop("chol_api", None)

        with patch.object(spla, "splu", wraps=spla.splu) as splu_spy:
            reml_gradient(theta, d["y"], d["X"], Z, d["q_sizes"], cache_no_chol)

        assert splu_spy.call_count == 1, (
            f"splu fallback should factorise A11 exactly once per gradient "
            f"call (saw {splu_spy.call_count})"
        )

    def test_use_gradient_none_resolves_to_fd(self, single_re_dataset: dict) -> None:
        """``use_gradient=None`` (default) currently resolves to FD even for
        diagonal specs.

        End-to-end fit_reml with the analytic gradient regresses vs FD at
        q >= ~35 because L-BFGS-B's FD path is competitive with cheap obj
        calls.  The default flip waits on a faster selected-inverse
        (interlace-mxzk follow-up: numba Takahashi).
        """
        d = single_re_dataset
        Z = _make_Z(d["group_codes"], d["q_sizes"][0])
        with patch(
            "interlace.profiled_reml.reml_gradient", wraps=reml_gradient
        ) as grad_spy:
            fit_reml(d["y"], d["X"], Z, d["q_sizes"], use_gradient=None)
        assert grad_spy.call_count == 0, (
            "use_gradient=None should currently resolve to FD; saw "
            f"{grad_spy.call_count} gradient call(s)"
        )

    def test_use_gradient_auto_detect_slopes_uses_fd(
        self, rng: np.random.Generator
    ) -> None:
        """``use_gradient=None`` with random-slope specs -> FD path."""
        n, q1 = 200, 10
        gc1 = np.repeat(np.arange(q1), n // q1)
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        b = rng.normal(size=(q1, 2))
        y = X @ [1.0, 0.5] + (b[gc1, 0] + b[gc1, 1] * x) + rng.normal(size=n)

        # Build Z for (1 + x | g1) random slopes.
        rows = np.arange(n)
        cols0 = gc1 * 2
        cols1 = gc1 * 2 + 1
        Z = sp.csc_matrix(
            (
                np.concatenate([np.ones(n), x]),
                (np.concatenate([rows, rows]), np.concatenate([cols0, cols1])),
            ),
            shape=(n, q1 * 2),
        )
        specs = [RandomEffectSpec(group="g1", predictors=["x"], correlated=True)]
        with patch(
            "interlace.profiled_reml.reml_gradient", wraps=reml_gradient
        ) as grad_spy:
            fit_reml(
                y,
                X,
                Z,
                q_sizes=[q1 * 2],
                specs=specs,
                n_levels=[q1],
                use_gradient=None,
            )
        assert grad_spy.call_count == 0, (
            "Random-slope specs with use_gradient=None should fall back to "
            "forward-difference (gradient not implemented for slopes)"
        )

    def test_use_gradient_explicit_true_with_slopes_raises(
        self, rng: np.random.Generator
    ) -> None:
        """Explicit ``use_gradient=True`` with random slopes must raise."""
        n, q1 = 200, 10
        gc1 = np.repeat(np.arange(q1), n // q1)
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        b = rng.normal(size=(q1, 2))
        y = X @ [1.0, 0.5] + (b[gc1, 0] + b[gc1, 1] * x) + rng.normal(size=n)
        rows = np.arange(n)
        cols0 = gc1 * 2
        cols1 = gc1 * 2 + 1
        Z = sp.csc_matrix(
            (
                np.concatenate([np.ones(n), x]),
                (np.concatenate([rows, rows]), np.concatenate([cols0, cols1])),
            ),
            shape=(n, q1 * 2),
        )
        specs = [RandomEffectSpec(group="g1", predictors=["x"], correlated=True)]
        with pytest.raises(NotImplementedError, match="diagonal"):
            fit_reml(
                y,
                X,
                Z,
                q_sizes=[q1 * 2],
                specs=specs,
                n_levels=[q1],
                use_gradient=True,
            )
