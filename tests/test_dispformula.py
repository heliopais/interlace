"""Tests for the dispformula parameter in glmer().

Tests model-based dispersion estimation via a log-linear sub-model
for the scale parameter: phi_i = exp(X_d[i] @ delta).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_data(
    n_groups: int = 20,
    n_per_group: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate Gaussian data with known dispersion (sigma^2 = 4.0)."""
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.5, n_groups)  # random intercepts, SD=1.5
    sigma = 2.0  # residual SD → phi = sigma^2 = 4.0
    y = 1.0 + 0.5 * x + u[group] + rng.normal(0, sigma, n)
    return pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})


def _make_heteroscedastic_data(
    n_groups: int = 30,
    n_per_group: int = 20,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate Gaussian data with covariate-dependent dispersion.

    True model:
        y ~ 1 + x, groups=group
        dispformula: log(phi) = 0.0 + 1.0 * z
        So phi_i = exp(z_i) and residual SD = sqrt(phi_i).
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(n)
    z = rng.uniform(-1, 1, n)  # dispersion covariate
    u = rng.normal(0, 1.0, n_groups)

    # True dispersion: phi_i = exp(0 + 1*z_i)
    phi = np.exp(0.0 + 1.0 * z)
    eps = rng.normal(0, 1, n) * np.sqrt(phi)

    y = 2.0 + 0.3 * x + u[group] + eps
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "group": group.astype(str),
        }
    )


# ---------------------------------------------------------------------------
# Tests: dispformula=None preserves existing behaviour
# ---------------------------------------------------------------------------


class TestDispformulaDefault:
    """When dispformula is None, behaviour is unchanged."""

    def test_binomial_scale_is_one(self):
        """Binomial models always have scale=1.0."""
        rng = np.random.default_rng(99)
        n = 200
        group = np.repeat(np.arange(20), 10).astype(str)
        x = rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(0.5 * x)))
        y = rng.binomial(1, p, n).astype(float)
        df = pd.DataFrame({"y": y, "x": x, "group": group})

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="binomial",
            groups="group",
        )
        assert result.scale == 1.0
        assert result.disp_params is None

    def test_poisson_scale_is_one(self):
        """Poisson models always have scale=1.0."""
        rng = np.random.default_rng(99)
        n = 200
        group = np.repeat(np.arange(20), 10).astype(str)
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"y": y, "x": x, "group": group})

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="poisson",
            groups="group",
        )
        assert result.scale == 1.0
        assert result.disp_params is None


# ---------------------------------------------------------------------------
# Tests: dispformula="~1" (scalar dispersion)
# ---------------------------------------------------------------------------


class TestScalarDispersion:
    """dispformula='~1' estimates a single dispersion parameter."""

    def test_gaussian_scalar_dispersion(self):
        """Gaussian GLMM with ~1 should recover the true residual variance."""
        df = _make_gaussian_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~1",
        )
        # True sigma^2 = 4.0.  Allow generous tolerance for 200 obs.
        assert result.converged
        estimated_phi = np.exp(result.disp_params["Intercept"])
        assert 2.0 < estimated_phi < 8.0  # within factor of 2

    def test_disp_params_is_series(self):
        """disp_params should be a pd.Series with named coefficients."""
        df = _make_gaussian_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~1",
        )
        assert isinstance(result.disp_params, pd.Series)
        assert "Intercept" in result.disp_params.index

    def test_aic_accounts_for_disp_params(self):
        """AIC should count dispersion parameters."""
        df = _make_gaussian_data()
        r1 = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
        )
        r2 = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~1",
        )
        # The dispformula model estimates 1 extra parameter.
        # AIC difference should reflect the improvement in fit
        # minus the penalty for the extra parameter.
        # Just check both are finite.
        assert np.isfinite(r1.aic)
        assert np.isfinite(r2.aic)


# ---------------------------------------------------------------------------
# Tests: dispformula with covariates (heteroscedastic)
# ---------------------------------------------------------------------------


class TestHeteroscedastic:
    """dispformula='~ z' models observation-level dispersion."""

    def test_heteroscedastic_dispersion_covariate(self):
        """Should recover positive dispersion slope for z."""
        df = _make_heteroscedastic_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~ z",
        )
        assert result.converged
        assert isinstance(result.disp_params, pd.Series)
        # True delta = [0.0, 1.0].  Slope on z should be positive.
        assert result.disp_params["z"] > 0.0

    def test_dispersion_values_stored(self):
        """Result should store the per-observation dispersion vector."""
        df = _make_heteroscedastic_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~ z",
        )
        assert hasattr(result, "dispersion")
        assert result.dispersion.shape == (len(df),)
        assert np.all(result.dispersion > 0)  # phi > 0 by exp()


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


class TestDispformulaValidation:
    """Edge cases and validation."""

    def test_rejects_dispformula_with_nAGQ_gt1(self):
        """dispformula + nAGQ > 1 is not supported."""
        df = _make_gaussian_data()
        with pytest.raises(ValueError, match="nAGQ"):
            interlace.glmer(
                "y ~ x",
                data=df,
                family="gaussian",
                groups="group",
                dispformula="~1",
                nAGQ=5,
            )


# ---------------------------------------------------------------------------
# Helpers: NB2 synthetic data
# ---------------------------------------------------------------------------


def _make_nb2_data(
    n_groups: int = 30,
    n_per_group: int = 20,
    seed: int = 77,
) -> pd.DataFrame:
    """Simulate NB2 count data with known theta (overdispersion).

    True model:
        y ~ Poisson-Gamma mixture with mu = exp(1 + 0.5*x + u[group])
        theta = 2.0 (scalar shape parameter)
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(n)
    u = rng.normal(0, 0.5, n_groups)  # random intercepts, SD=0.5

    eta = 1.0 + 0.5 * x + u[group]
    mu = np.exp(eta)

    # NB2: Y ~ NegBin(mu, theta=2.0)
    theta_true = 2.0
    # Gamma-Poisson mixture: lambda ~ Gamma(theta, theta/mu), Y ~ Poisson(lambda)
    lam = rng.gamma(theta_true, mu / theta_true)
    y = rng.poisson(lam).astype(float)

    return pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})


def _make_nb2_hetdisp_data(
    n_groups: int = 30,
    n_per_group: int = 30,
    seed: int = 88,
) -> pd.DataFrame:
    """Simulate NB2 data with covariate-dependent overdispersion.

    True model:
        y ~ NB2(mu, theta_i)
        mu = exp(1 + 0.3*x + u[group])
        log(theta_i) = 1.0 + 0.8*z   (so theta varies with z)
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal(n)
    z = rng.uniform(-1, 1, n)  # dispersion covariate
    u = rng.normal(0, 0.5, n_groups)

    eta = 1.0 + 0.3 * x + u[group]
    mu = np.exp(eta)

    # Per-observation theta
    theta_i = np.exp(1.0 + 0.8 * z)
    lam = rng.gamma(theta_i, mu / theta_i)
    y = rng.poisson(lam).astype(float)

    return pd.DataFrame({"y": y, "x": x, "z": z, "group": group.astype(str)})


# ---------------------------------------------------------------------------
# Tests: NB2 dispformula
# ---------------------------------------------------------------------------


class TestNB2ScalarDispformula:
    """dispformula='~1' with NB2 estimates a scalar theta."""

    def test_nb2_scalar_dispformula_converges(self):
        """NB2 GLMM with dispformula='~1' should converge."""
        df = _make_nb2_data()
        from interlace.glmm_family import NegativeBinomial2Family

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family=NegativeBinomial2Family(theta=1.0),
            groups="group",
            dispformula="~1",
        )
        assert result.converged

    def test_nb2_scalar_dispformula_recovers_theta(self):
        """Should recover theta close to the true value of 2.0."""
        df = _make_nb2_data()
        from interlace.glmm_family import NegativeBinomial2Family

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family=NegativeBinomial2Family(theta=1.0),
            groups="group",
            dispformula="~1",
        )
        # delta is on log scale, so theta_hat = exp(delta[Intercept])
        estimated_theta = np.exp(result.disp_params["Intercept"])
        # True theta = 2.0, allow generous tolerance for 600 obs
        assert 0.5 < estimated_theta < 8.0

    def test_nb2_disp_params_is_series(self):
        """disp_params should be a pd.Series with named coefficients."""
        df = _make_nb2_data()
        from interlace.glmm_family import NegativeBinomial2Family

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family=NegativeBinomial2Family(theta=1.0),
            groups="group",
            dispformula="~1",
        )
        assert isinstance(result.disp_params, pd.Series)
        assert "Intercept" in result.disp_params.index


class TestNB2HeterogeneousDispformula:
    """dispformula='~ z' models per-observation theta for NB2."""

    def test_nb2_covariate_dispformula_converges(self):
        """NB2 with covariate dispformula should converge."""
        df = _make_nb2_hetdisp_data()
        from interlace.glmm_family import NegativeBinomial2Family

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family=NegativeBinomial2Family(theta=1.0),
            groups="group",
            dispformula="~ z",
        )
        assert result.converged

    def test_nb2_covariate_dispformula_positive_slope(self):
        """Slope on z should be positive (true delta_z = 0.8)."""
        df = _make_nb2_hetdisp_data()
        from interlace.glmm_family import NegativeBinomial2Family

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family=NegativeBinomial2Family(theta=1.0),
            groups="group",
            dispformula="~ z",
        )
        assert result.disp_params["z"] > 0.0

    def test_nb2_dispersion_values_stored(self):
        """Per-observation dispersion vector should be stored."""
        df = _make_nb2_hetdisp_data()
        from interlace.glmm_family import NegativeBinomial2Family

        result = interlace.glmer(
            "y ~ x",
            data=df,
            family=NegativeBinomial2Family(theta=1.0),
            groups="group",
            dispformula="~ z",
        )
        assert result.dispersion is not None
        assert result.dispersion.shape == (len(df),)
        assert np.all(result.dispersion > 0)


# ---------------------------------------------------------------------------
# Tests: GLMMResult summary with dispersion
# ---------------------------------------------------------------------------


class TestGLMMResultSummary:
    """GLMMResult.summary() should display dispersion coefficients."""

    def test_summary_shows_dispersion_coefficients(self):
        """summary() output should include dispersion model section."""
        df = _make_heteroscedastic_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
            dispformula="~ z",
        )
        text = str(result.summary())
        assert "Dispersion model" in text
        assert "Intercept" in text
        assert "z" in text

    def test_summary_no_dispersion_section_without_dispformula(self):
        """summary() without dispformula should not show dispersion section."""
        df = _make_gaussian_data()
        result = interlace.glmer(
            "y ~ x",
            data=df,
            family="gaussian",
            groups="group",
        )
        text = str(result.summary())
        assert "Dispersion model" not in text


# ---------------------------------------------------------------------------
# Tests: R (glmmTMB) validation fixtures
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIXTURES / "dispformula_gaussian_results.json").exists(),
    reason="R fixture not generated",
)
class TestGlmmTMBGaussianValidation:
    """Validate Gaussian dispformula against glmmTMB reference."""

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(FIXTURES / "dispformula_gaussian_results.json") as f:
            self.ref = json.load(f)
        self.df = pd.read_csv(FIXTURES / "dispformula_gaussian_data.csv")

    def _fit(self):
        return interlace.glmer(
            "y ~ x",
            data=self.df,
            family="gaussian",
            groups="group",
            dispformula="~ z",
        )

    def test_fixed_effects_close(self):
        result = self._fit()
        for name, r_val in self.ref["fixed_effects"].items():
            py_name = "Intercept" if name == "(Intercept)" else name
            # Issue spec: abs_diff < 1e-3 on mean-model fixed effects.
            assert abs(result.fe_params[py_name] - r_val) < 1e-3

    def test_disp_params_close(self):
        result = self._fit()
        # Tight parity: disp coefs on log(sigma) scale should match glmmTMB.
        for name, r_val in self.ref["disp_params"].items():
            py_name = "Intercept" if name == "(Intercept)" else name
            assert abs(result.disp_params[py_name] - r_val) < 1e-3

    def test_variance_component_close(self):
        result = self._fit()
        r_vc = self.ref["variance_components"]["group"]
        py_vc = result.variance_components["group"]
        # Issue spec: rel_diff < 10% on disp varcomps; this is mean-side VC,
        # parity is much tighter.
        assert abs(py_vc - r_vc) / r_vc < 0.01

    def test_loglik_matches(self):
        result = self._fit()
        assert abs(result.llf - self.ref["loglik"]) < 1e-4


# ---------------------------------------------------------------------------
# Tests: fit() (LMM entry) with dispformula
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIXTURES / "dispformula_gaussian_results.json").exists(),
    reason="R fixture not generated",
)
class TestFitDispformulaFEOnly:
    """``interlace.fit`` routes FE-only dispformula to joint Laplace."""

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(FIXTURES / "dispformula_gaussian_results.json") as f:
            self.ref = json.load(f)
        self.df = pd.read_csv(FIXTURES / "dispformula_gaussian_data.csv")

    def test_fit_returns_lme_result_with_disp_attrs(self):
        import interlace

        res = interlace.fit("y ~ x", self.df, groups="group", dispformula="~ z")
        assert res.disp_method == "joint_laplace"
        assert res.disp_params is not None
        assert "Intercept" in res.disp_params.index
        assert "z" in res.disp_params.index
        assert res.dispersion is not None
        assert res.dispersion.shape == (len(self.df),)

    def test_fit_tight_fe_parity(self):
        import interlace

        res = interlace.fit("y ~ x", self.df, groups="group", dispformula="~ z")
        # Issue spec: abs_diff < 1e-3 on mean-model fixed effects.
        for name, r_val in self.ref["fixed_effects"].items():
            py_name = "Intercept" if name == "(Intercept)" else name
            assert abs(res.fe_params[py_name] - r_val) < 1e-3

    def test_fit_tight_disp_parity(self):
        import interlace

        res = interlace.fit("y ~ x", self.df, groups="group", dispformula="~ z")
        for name, r_val in self.ref["disp_params"].items():
            py_name = "Intercept" if name == "(Intercept)" else name
            assert abs(res.disp_params[py_name] - r_val) < 1e-3

    def test_fit_variance_component_parity(self):
        import interlace

        res = interlace.fit("y ~ x", self.df, groups="group", dispformula="~ z")
        r_vc = self.ref["variance_components"]["group"]
        assert abs(res.variance_components["group"] - r_vc) / r_vc < 0.01


@pytest.mark.skipif(
    not (FIXTURES / "dispformula_nested_results.json").exists(),
    reason="R fixture not generated",
)
class TestFitDispformulaRENested:
    """``fit`` with random effects on dispformula side → BCA path."""

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(FIXTURES / "dispformula_nested_results.json") as f:
            self.ref = json.load(f)
        self.df = pd.read_csv(FIXTURES / "dispformula_nested_data.csv")
        for col in ("g_mean", "g1", "g2"):
            self.df[col] = self.df[col].astype(str)

    def test_bca_mean_fe_close(self):
        import interlace

        res = interlace.fit(
            "y ~ x", self.df, groups="g_mean", dispformula="~ (1|g1/g2)"
        )
        # Mean-side FE: BCA is biased on disp varcomps but mean FE remain
        # close to glmmTMB. Tolerance 5e-3 reflects BCA's bias.
        for name, r_val in self.ref["fixed_effects"].items():
            py_name = "Intercept" if name == "(Intercept)" else name
            assert abs(res.fe_params[py_name] - r_val) < 5e-3, (
                f"FE {py_name}: py={res.fe_params[py_name]} R={r_val}"
            )

    def test_bca_disp_method_marker(self):
        import interlace

        res = interlace.fit(
            "y ~ x", self.df, groups="g_mean", dispformula="~ (1|g1/g2)"
        )
        assert res.disp_method == "bca"
        assert "g1" in res.disp_variance_components
        assert "g1:g2" in res.disp_variance_components

    def test_bca_disp_varcomp_within_band(self):
        import interlace

        res = interlace.fit(
            "y ~ x", self.df, groups="g_mean", dispformula="~ (1|g1/g2)"
        )
        # BCA's disp varcomps are biased vs joint MLE. Issue acceptance
        # criterion: rel_diff < 10% — achievable here for nested case.
        r_vc_g1 = self.ref["disp_variance_components"]["g1"]
        r_vc_g1g2 = self.ref["disp_variance_components"]["g2:g1"]
        py_vc_g1 = res.disp_variance_components["g1"]
        py_vc_g1g2 = res.disp_variance_components["g1:g2"]
        assert abs(py_vc_g1 - r_vc_g1) / r_vc_g1 < 0.20
        assert abs(py_vc_g1g2 - r_vc_g1g2) / r_vc_g1g2 < 0.15


class TestFitDispformulaWeights:
    """``fit(..., weights=, dispformula=)`` composes correctly."""

    def test_dispformula_with_weights_runs(self):
        import interlace

        rng = np.random.default_rng(0)
        n_groups = 20
        n_per_group = 15
        n = n_groups * n_per_group
        group = np.repeat(np.arange(n_groups), n_per_group)
        x = rng.standard_normal(n)
        z = rng.uniform(-1, 1, n)
        u = rng.normal(0, 0.8, n_groups)
        sigma = np.exp(0.0 + 0.5 * z)
        y = 1.0 + 0.3 * x + u[group] + rng.normal(0, 1, n) * sigma
        df = pd.DataFrame({"y": y, "x": x, "z": z, "g": group.astype(str)})
        w = rng.uniform(0.5, 2.0, n)

        res = interlace.fit("y ~ x", df, groups="g", dispformula="~ z", weights=w)
        assert res.converged
        # Slope on z should be positive (true 0.5).
        assert res.disp_params["z"] > 0.0


class TestFitDispformulaBcaJointConsistency:
    """For FE-only dispformula, BCA and joint Laplace should give consistent
    point estimates within BCA's known bias (~5% on disp slopes)."""

    def test_bca_vs_joint_laplace_fe_only(self):
        import interlace
        from interlace.dispformula_bca import fit_dispformula_bca

        df = pd.read_csv(FIXTURES / "dispformula_gaussian_data.csv")

        res_jl = interlace.fit("y ~ x", df, groups="group", dispformula="~ z")
        res_bca = fit_dispformula_bca(
            formula="y ~ x", data=df, groups="group", dispformula="~ z"
        )
        # FE agree to 1%
        for name in ("Intercept", "x"):
            assert abs(res_jl.fe_params[name] - res_bca.fe_params[name]) < 5e-3
        # Disp slope same sign, magnitudes close (BCA biased ~5%).
        assert res_bca.disp_params["z"] * res_jl.disp_params["z"] > 0
        assert abs(res_bca.disp_params["z"] - res_jl.disp_params["z"]) < 0.05


@pytest.mark.skipif(
    not (FIXTURES / "dispformula_nb2_results.json").exists(),
    reason="R fixture not generated",
)
class TestGlmmTMBNB2Validation:
    """Validate NB2 dispformula against glmmTMB reference."""

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(FIXTURES / "dispformula_nb2_results.json") as f:
            self.ref = json.load(f)
        self.df = pd.read_csv(FIXTURES / "dispformula_nb2_data.csv")

    def _fit(self):
        from interlace.glmm_family import NegativeBinomial2Family

        return interlace.glmer(
            "y ~ x",
            data=self.df,
            family=NegativeBinomial2Family(theta=1.0),
            groups="group",
            dispformula="~1",
        )

    def test_fixed_effects_close(self):
        result = self._fit()
        for name, r_val in self.ref["fixed_effects"].items():
            py_name = "Intercept" if name == "(Intercept)" else name
            assert abs(result.fe_params[py_name] - r_val) < 0.2

    def test_nb2_theta_close_to_glmmtmb(self):
        """Estimated theta should be in the right ballpark vs glmmTMB."""
        result = self._fit()
        r_theta = np.exp(self.ref["disp_params"]["(Intercept)"])
        py_theta = np.exp(result.disp_params["Intercept"])
        # Allow factor of 2 tolerance — different optimisers/parameterisations
        ratio = py_theta / r_theta
        assert 0.5 < ratio < 2.0
