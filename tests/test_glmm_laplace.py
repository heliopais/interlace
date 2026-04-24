"""Tests for GLMM Laplace approximation (glmm_laplace.py).

Tests follow TDD: written before the implementation exists.

Coverage:
1. Gaussian family reduces to LMM (sanity check)
2. Binomial GLMM parity with lme4 on cbpp dataset
3. Poisson GLMM parity with lme4 on simulated data
4. PIRLS convergence and basic properties
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import interlace
from interlace.glmm_family import BinomialFamily, PoissonFamily
from interlace.glmm_laplace import GLMMResult, fit_glmm

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cbpp_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_cbpp_data.csv")


@pytest.fixture(scope="module")
def cbpp_ref() -> dict:
    return json.loads((FIXTURES / "glmm_cbpp_results.json").read_text())


@pytest.fixture(scope="module")
def cbpp_fit(cbpp_data) -> GLMMResult:
    """Fit binomial GLMM on cbpp: proportion ~ C(period) + (1|herd), weights=size."""
    return fit_glmm(
        formula="proportion ~ C(period)",
        data=cbpp_data,
        family="binomial",
        groups="herd",
        weights=cbpp_data["size"].values.astype(float),
    )


@pytest.fixture(scope="module")
def poisson_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "glmm_poisson_data.csv")


@pytest.fixture(scope="module")
def poisson_ref() -> dict:
    return json.loads((FIXTURES / "glmm_poisson_results.json").read_text())


@pytest.fixture(scope="module")
def poisson_fit(poisson_data) -> GLMMResult:
    """Fit Poisson GLMM: y ~ x + (1|group)."""
    return fit_glmm(
        formula="y ~ x",
        data=poisson_data,
        family="poisson",
        groups="group",
    )


# ---------------------------------------------------------------------------
# 1. Gaussian sanity: GLMM with Gaussian family ≈ LMM
# ---------------------------------------------------------------------------


class TestGaussianSanity:
    """GLMM with Gaussian family should give similar results to fit()."""

    @pytest.fixture(scope="class")
    def dyestuff_data(self) -> pd.DataFrame:
        return pd.read_csv(FIXTURES / "lme4_dyestuff_data.csv")

    @pytest.fixture(scope="class")
    def lmm_fit(self, dyestuff_data):
        return interlace.fit("Yield ~ 1", data=dyestuff_data, groups="Batch")

    @pytest.fixture(scope="class")
    def glmm_gaussian_fit(self, dyestuff_data):
        return fit_glmm(
            formula="Yield ~ 1",
            data=dyestuff_data,
            family="gaussian",
            groups="Batch",
        )

    def test_fixed_effects_close(self, lmm_fit, glmm_gaussian_fit):
        """Gaussian GLMM fixed effects should match LMM closely."""
        lmm_beta = lmm_fit.fe_params.values
        glmm_beta = glmm_gaussian_fit.fe_params.values
        assert_allclose(glmm_beta, lmm_beta, rtol=0.05)

    def test_variance_components_close(self, lmm_fit, glmm_gaussian_fit):
        """Gaussian GLMM variance components should match LMM closely."""
        lmm_vc = lmm_fit.variance_components["Batch"]
        glmm_vc = glmm_gaussian_fit.variance_components["Batch"]
        assert_allclose(glmm_vc, lmm_vc, rtol=0.15)


# ---------------------------------------------------------------------------
# 2. Binomial GLMM: parity with lme4 on cbpp
# ---------------------------------------------------------------------------


# Map R coefficient names → formulaic names
_CBPP_NAME_MAP = {
    "(Intercept)": "Intercept",
    "period2": "C(period)[T.2]",
    "period3": "C(period)[T.3]",
    "period4": "C(period)[T.4]",
}


class TestBinomialCBPP:
    """Binomial GLMM on cbpp dataset should match lme4::glmer()."""

    def test_converged(self, cbpp_fit):
        assert cbpp_fit.converged

    def test_fixed_effects(self, cbpp_fit, cbpp_ref):
        """Fixed effects within 0.05 of lme4."""
        ref_fe = cbpp_ref["fixed_effects"]
        for r_name, ref_val in ref_fe.items():
            py_name = _CBPP_NAME_MAP[r_name]
            assert abs(cbpp_fit.fe_params[py_name] - ref_val) < 0.05, (
                f"{py_name}: interlace={cbpp_fit.fe_params[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_fixed_effects_se(self, cbpp_fit, cbpp_ref):
        """Fixed effects SEs within 10% of lme4."""
        ref_se = cbpp_ref["fixed_effects_se"]
        for r_name, ref_val in ref_se.items():
            py_name = _CBPP_NAME_MAP[r_name]
            assert abs(cbpp_fit.fe_bse[py_name] - ref_val) / ref_val < 0.10, (
                f"{py_name}: interlace={cbpp_fit.fe_bse[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_variance_component(self, cbpp_fit, cbpp_ref):
        """Herd variance within 15% of lme4."""
        ref_vc = cbpp_ref["variance_components"]["herd"]
        fit_vc = cbpp_fit.variance_components["herd"]
        assert abs(fit_vc - ref_vc) / ref_vc < 0.15, (
            f"herd VC: interlace={fit_vc:.4f}, lme4={ref_vc:.4f}"
        )

    def test_loglik(self, cbpp_fit, cbpp_ref):
        """Log-likelihood within 1.0 of lme4."""
        assert abs(cbpp_fit.llf - cbpp_ref["loglik"]) < 1.0, (
            f"loglik: interlace={cbpp_fit.llf:.2f}, lme4={cbpp_ref['loglik']:.2f}"
        )

    def test_random_effects_correlation(self, cbpp_fit, cbpp_ref):
        """BLUPs should correlate > 0.95 with lme4."""
        ref_re = np.array(cbpp_ref["random_effects_herd"])
        fit_re = cbpp_fit.random_effects["herd"]
        # Sort by index to align (R uses string "1".."15", Python may use int)
        fit_vals = fit_re.sort_index().values
        corr = np.corrcoef(fit_vals, ref_re)[0, 1]
        assert corr > 0.95, f"BLUP correlation = {corr:.4f}"

    def test_result_has_family(self, cbpp_fit):
        assert isinstance(cbpp_fit.family, BinomialFamily)

    def test_nobs(self, cbpp_fit, cbpp_ref):
        assert cbpp_fit.nobs == cbpp_ref["nobs"]


# ---------------------------------------------------------------------------
# 3. Poisson GLMM parity with lme4
# ---------------------------------------------------------------------------


class TestPoissonGLMM:
    """Poisson GLMM should match lme4::glmer()."""

    def test_converged(self, poisson_fit):
        assert poisson_fit.converged

    def test_fixed_effects(self, poisson_fit, poisson_ref):
        """Fixed effects within 0.05 of lme4."""
        ref_fe = poisson_ref["fixed_effects"]
        name_map = {"(Intercept)": "Intercept", "x": "x"}
        for r_name, ref_val in ref_fe.items():
            py_name = name_map[r_name]
            assert abs(poisson_fit.fe_params[py_name] - ref_val) < 0.05, (
                f"{py_name}: interlace={poisson_fit.fe_params[py_name]:.4f}, "
                f"lme4={ref_val:.4f}"
            )

    def test_variance_component(self, poisson_fit, poisson_ref):
        """Group variance within 20% of lme4."""
        ref_vc = poisson_ref["variance_components"]["group"]
        fit_vc = poisson_fit.variance_components["group"]
        assert abs(fit_vc - ref_vc) / ref_vc < 0.20, (
            f"group VC: interlace={fit_vc:.4f}, lme4={ref_vc:.4f}"
        )

    def test_loglik(self, poisson_fit, poisson_ref):
        """Log-likelihood within 2.0 of lme4."""
        assert abs(poisson_fit.llf - poisson_ref["loglik"]) < 2.0, (
            f"loglik: interlace={poisson_fit.llf:.2f}, lme4={poisson_ref['loglik']:.2f}"
        )

    def test_result_has_family(self, poisson_fit):
        assert isinstance(poisson_fit.family, PoissonFamily)


# ---------------------------------------------------------------------------
# 4. GLMMResult basic properties
# ---------------------------------------------------------------------------


class TestGLMMResultProperties:
    """GLMMResult should expose the expected attributes."""

    def test_has_fe_params(self, cbpp_fit):
        assert hasattr(cbpp_fit, "fe_params")
        assert len(cbpp_fit.fe_params) == 4  # intercept + 3 period dummies

    def test_has_fe_bse(self, cbpp_fit):
        assert hasattr(cbpp_fit, "fe_bse")
        assert len(cbpp_fit.fe_bse) == 4

    def test_has_random_effects(self, cbpp_fit):
        assert "herd" in cbpp_fit.random_effects

    def test_has_theta(self, cbpp_fit):
        assert hasattr(cbpp_fit, "theta")
        assert len(cbpp_fit.theta) >= 1

    def test_has_aic_bic(self, cbpp_fit):
        assert hasattr(cbpp_fit, "aic")
        assert hasattr(cbpp_fit, "bic")
        assert np.isfinite(cbpp_fit.aic)
        assert np.isfinite(cbpp_fit.bic)

    def test_has_converged(self, cbpp_fit):
        assert isinstance(cbpp_fit.converged, bool)


# ---------------------------------------------------------------------------
# 5. ZINB2 GLMM parity with glmmTMB
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zinb2_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "zinb2_data.csv")


@pytest.fixture(scope="module")
def zinb2_ref() -> dict:
    return json.loads((FIXTURES / "zinb2_results.json").read_text())


@pytest.fixture(scope="module")
def zinb2_fit(zinb2_data) -> GLMMResult:
    """Fit ZINB2 GLMM: y ~ x + (1|group), family=ZeroInflatedNB2.

    We pass theta and pi as fixed constants (not jointly estimated).
    glmmTMB jointly optimises theta and pi, so our estimates will differ.
    This test validates that the ZINB2 family flows through fit_glmm.
    """
    from interlace.glmm_family import ZeroInflatedNB2Family

    return fit_glmm(
        formula="y ~ x",
        data=zinb2_data,
        family=ZeroInflatedNB2Family(theta=2.0, pi=0.3),
        groups="group",
    )


@pytest.fixture(scope="module")
def zinb2_fit_nb2_only(zinb2_data) -> GLMMResult:
    """Fit plain NB2 (pi=0) on the ZINB2 dataset as a baseline."""
    from interlace.glmm_family import ZeroInflatedNB2Family

    return fit_glmm(
        formula="y ~ x",
        data=zinb2_data,
        family=ZeroInflatedNB2Family(theta=2.0, pi=0.0),
        groups="group",
    )


class TestZINB2GLMM:
    """Zero-inflated NB2 GLMM should match glmmTMB.

    Theta and pi are passed as known constants (not jointly estimated),
    so some tolerance is expected vs glmmTMB's joint MLE.
    """

    def test_converged(self, zinb2_fit):
        assert zinb2_fit.converged

    def test_fixed_effects(self, zinb2_fit, zinb2_ref):
        """Fixed effects within 0.05 of glmmTMB."""
        ref_fe = zinb2_ref["fixed_effects"]
        name_map = {"(Intercept)": "Intercept", "x": "x"}
        for r_name, ref_val in ref_fe.items():
            py_name = name_map[r_name]
            assert abs(zinb2_fit.fe_params[py_name] - ref_val) < 0.05, (
                f"{py_name}: interlace={zinb2_fit.fe_params[py_name]:.4f}, "
                f"glmmTMB={ref_val:.4f}"
            )

    def test_variance_component(self, zinb2_fit, zinb2_ref):
        """Group variance within 20% of glmmTMB."""
        ref_vc = zinb2_ref["variance_components"]["group"]
        fit_vc = zinb2_fit.variance_components["group"]
        assert abs(fit_vc - ref_vc) / ref_vc < 0.20, (
            f"group VC: interlace={fit_vc:.4f}, glmmTMB={ref_vc:.4f}"
        )

    def test_loglik(self, zinb2_fit, zinb2_ref):
        """Log-likelihood within 5.0 of glmmTMB."""
        assert abs(zinb2_fit.llf - zinb2_ref["loglik"]) < 5.0, (
            f"loglik: interlace={zinb2_fit.llf:.2f}, glmmTMB={zinb2_ref['loglik']:.2f}"
        )

    def test_zi_improves_loglik(self, zinb2_fit, zinb2_fit_nb2_only):
        """Fitting with pi > 0 should give a better log-likelihood
        than pi=0 on zero-inflated data."""
        assert zinb2_fit.llf > zinb2_fit_nb2_only.llf, (
            f"ZI ll={zinb2_fit.llf:.2f} should be > NB2 ll={zinb2_fit_nb2_only.llf:.2f}"
        )

    def test_result_has_family(self, zinb2_fit):
        from interlace.glmm_family import ZeroInflatedNB2Family

        assert isinstance(zinb2_fit.family, ZeroInflatedNB2Family)


# ---------------------------------------------------------------------------
# 6. Hurdle (truncated) Poisson GLMM
# ---------------------------------------------------------------------------


def _simulate_hurdle_poisson(
    n_groups: int = 30,
    n_per_group: int = 20,
    beta0: float = 1.0,
    beta1: float = 0.5,
    sigma_u: float = 0.5,
    pi: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate data from a hurdle Poisson GLMM.

    Process:
      1. Draw group intercepts u_i ~ N(0, sigma_u^2).
      2. For each obs: eta = beta0 + beta1*x + u_{group}, mu = exp(eta).
      3. With probability pi, y=0 (structural zero).
      4. With probability 1-pi, y ~ ZeroTruncatedPoisson(mu).
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=n)
    u = rng.normal(0, sigma_u, size=n_groups)
    eta = beta0 + beta1 * x + u[group]
    mu = np.exp(eta)

    # Hurdle process
    y = np.zeros(n, dtype=int)
    is_count = rng.random(n) > pi  # 1-pi chance of count process
    for i in range(n):
        if is_count[i]:
            # Zero-truncated Poisson: sample Poisson, reject zeros
            while True:
                draw = rng.poisson(mu[i])
                if draw > 0:
                    y[i] = draw
                    break

    return pd.DataFrame({"y": y, "x": x, "group": group})


@pytest.fixture(scope="module")
def hurdle_poisson_data() -> pd.DataFrame:
    return _simulate_hurdle_poisson()


@pytest.fixture(scope="module")
def hurdle_poisson_fit(hurdle_poisson_data) -> GLMMResult:
    """Fit hurdle Poisson GLMM: y ~ x + (1|group)."""
    from interlace.glmm_family import HurdlePoissonFamily

    return fit_glmm(
        formula="y ~ x",
        data=hurdle_poisson_data,
        family=HurdlePoissonFamily(pi=0.3),
        groups="group",
    )


@pytest.fixture(scope="module")
def hurdle_poisson_fit_plain(hurdle_poisson_data) -> GLMMResult:
    """Fit plain Poisson (no hurdle) on the hurdle data as baseline."""
    return fit_glmm(
        formula="y ~ x",
        data=hurdle_poisson_data,
        family="poisson",
        groups="group",
    )


class TestHurdlePoissonFamily:
    """Unit tests for HurdlePoissonFamily class."""

    def test_class_exists(self):
        from interlace.glmm_family import HurdlePoissonFamily

        fam = HurdlePoissonFamily(pi=0.3)
        assert fam.name == "hurdle_poisson"
        assert fam.pi == 0.3

    def test_link_linkinv_roundtrip(self):
        from interlace.glmm_family import HurdlePoissonFamily

        fam = HurdlePoissonFamily(pi=0.3)
        mu = np.array([0.5, 1.0, 5.0, 10.0])
        assert_allclose(fam.linkinv(fam.link(mu)), mu)

    def test_mu_eta(self):
        """d(exp(eta))/d(eta) = exp(eta) = mu."""
        from interlace.glmm_family import HurdlePoissonFamily

        fam = HurdlePoissonFamily(pi=0.3)
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(fam.mu_eta(eta), np.exp(eta))

    def test_variance(self):
        """Variance function = mu (Poisson count component)."""
        from interlace.glmm_family import HurdlePoissonFamily

        fam = HurdlePoissonFamily(pi=0.3)
        mu = np.array([0.5, 1.0, 5.0])
        assert_allclose(fam.variance(mu), mu)

    def test_pi_validation(self):
        from interlace.glmm_family import HurdlePoissonFamily

        with pytest.raises(ValueError, match="pi must be in"):
            HurdlePoissonFamily(pi=-0.1)
        with pytest.raises(ValueError, match="pi must be in"):
            HurdlePoissonFamily(pi=1.0)

    def test_resolve_family(self):
        from interlace.glmm_family import HurdlePoissonFamily, resolve_family

        fam = resolve_family("hurdle_poisson")
        assert isinstance(fam, HurdlePoissonFamily)


class TestHurdlePoissonLoglik:
    """Test conditional log-likelihood for hurdle Poisson."""

    def test_loglik_zeros(self):
        """For y=0 observations, ll = log(pi)."""
        from interlace.glmm_family import HurdlePoissonFamily
        from interlace.glmm_laplace import _conditional_loglik

        fam = HurdlePoissonFamily(pi=0.3)
        y = np.array([0.0, 0.0, 0.0])
        mu = np.array([2.0, 3.0, 1.0])  # shouldn't matter
        w = np.ones(3)
        ll = _conditional_loglik(y, mu, w, fam)
        expected = 3.0 * np.log(0.3)
        assert_allclose(ll, expected, rtol=1e-10)

    def test_loglik_positive(self):
        """For y>0, ll = log(1-pi) + Poisson_ll - log(1-exp(-mu))."""
        from scipy.special import gammaln

        from interlace.glmm_family import HurdlePoissonFamily
        from interlace.glmm_laplace import _conditional_loglik

        fam = HurdlePoissonFamily(pi=0.3)
        y = np.array([2.0])
        mu = np.array([3.0])
        w = np.ones(1)
        ll = _conditional_loglik(y, mu, w, fam)
        # Manual: log(1-0.3) + [2*log(3) - 3 - lgamma(3)] - log(1 - exp(-3))
        pois_ll = 2.0 * np.log(3.0) - 3.0 - gammaln(3.0)
        trunc_corr = -np.log(1.0 - np.exp(-3.0))
        expected = np.log(0.7) + pois_ll + trunc_corr
        assert_allclose(ll, expected, rtol=1e-10)

    def test_loglik_pi_zero_is_truncated_poisson(self):
        """With pi=0, all obs are from truncated Poisson."""
        from scipy.special import gammaln

        from interlace.glmm_family import HurdlePoissonFamily
        from interlace.glmm_laplace import _conditional_loglik

        fam = HurdlePoissonFamily(pi=0.0)
        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([2.0, 2.0, 2.0])
        w = np.ones(3)
        ll = _conditional_loglik(y, mu, w, fam)
        # Manual: sum of [y*log(mu) - mu - lgamma(y+1) - log(1-exp(-mu))]
        pois_ll = y * np.log(mu) - mu - gammaln(y + 1)
        trunc_corr = -np.log(1.0 - np.exp(-mu))
        expected = float(np.sum(pois_ll + trunc_corr))
        assert_allclose(ll, expected, rtol=1e-10)


class TestHurdlePoissonGLMM:
    """Integration tests: fit hurdle Poisson GLMM on simulated data."""

    def test_converged(self, hurdle_poisson_fit):
        assert hurdle_poisson_fit.converged

    def test_intercept_reasonable(self, hurdle_poisson_fit):
        """Intercept should be within 0.5 of the true beta0=1.0."""
        intercept = hurdle_poisson_fit.fe_params["Intercept"]
        assert abs(intercept - 1.0) < 0.5, f"Intercept={intercept:.3f}, expected ~1.0"

    def test_slope_reasonable(self, hurdle_poisson_fit):
        """Slope should be within 0.3 of the true beta1=0.5."""
        slope = hurdle_poisson_fit.fe_params["x"]
        assert abs(slope - 0.5) < 0.3, f"Slope={slope:.3f}, expected ~0.5"

    def test_variance_component_positive(self, hurdle_poisson_fit):
        vc = hurdle_poisson_fit.variance_components["group"]
        assert vc > 0, f"Variance component should be positive, got {vc}"

    def test_result_has_family(self, hurdle_poisson_fit):
        from interlace.glmm_family import HurdlePoissonFamily

        assert isinstance(hurdle_poisson_fit.family, HurdlePoissonFamily)

    def test_hurdle_improves_over_poisson(
        self, hurdle_poisson_fit, hurdle_poisson_fit_plain
    ):
        """Hurdle model should fit better than plain Poisson on hurdle data."""
        assert hurdle_poisson_fit.llf > hurdle_poisson_fit_plain.llf, (
            f"Hurdle ll={hurdle_poisson_fit.llf:.2f} should be > "
            f"Poisson ll={hurdle_poisson_fit_plain.llf:.2f}"
        )

    def test_fittedvalues_shape(self, hurdle_poisson_fit, hurdle_poisson_data):
        assert hurdle_poisson_fit.fittedvalues.shape == (len(hurdle_poisson_data),)

    def test_aic_bic_finite(self, hurdle_poisson_fit):
        assert np.isfinite(hurdle_poisson_fit.aic)
        assert np.isfinite(hurdle_poisson_fit.bic)

    def test_glmer_api(self, hurdle_poisson_data):
        """Test that glmer() shortcut works with HurdlePoissonFamily."""
        from interlace.glmm_family import HurdlePoissonFamily

        result = interlace.glmer(
            "y ~ x",
            hurdle_poisson_data,
            family=HurdlePoissonFamily(pi=0.3),
            groups="group",
        )
        assert result.converged


# ---------------------------------------------------------------------------
# 7. Gamma GLMM
# ---------------------------------------------------------------------------


def _simulate_gamma_glmm(
    n_groups: int = 30,
    n_per_group: int = 20,
    beta0: float = 1.5,
    beta1: float = 0.4,
    sigma_u: float = 0.3,
    shape: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate data from a Gamma GLMM with log link.

    Process:
      1. Draw group intercepts u_i ~ N(0, sigma_u^2).
      2. For each obs: eta = beta0 + beta1*x + u_{group}, mu = exp(eta).
      3. y ~ Gamma(shape=shape, scale=mu/shape).
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=n)
    u = rng.normal(0, sigma_u, size=n_groups)
    eta = beta0 + beta1 * x + u[group]
    mu = np.exp(eta)
    y = rng.gamma(shape=shape, scale=mu / shape)
    return pd.DataFrame({"y": y, "x": x, "group": group})


@pytest.fixture(scope="module")
def gamma_data() -> pd.DataFrame:
    return _simulate_gamma_glmm()


@pytest.fixture(scope="module")
def gamma_fit(gamma_data) -> GLMMResult:
    """Fit Gamma GLMM: y ~ x + (1|group), log link, shape=5."""
    from interlace.glmm_family import GammaFamily

    return fit_glmm(
        formula="y ~ x",
        data=gamma_data,
        family=GammaFamily(link="log", shape=5.0),
        groups="group",
    )


class TestGammaFamily:
    """Unit tests for GammaFamily class."""

    def test_class_exists(self):
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily(link="log", shape=5.0)
        assert fam.name == "gamma"
        assert fam.shape == 5.0

    def test_default_link_is_log(self):
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily()
        assert fam._link == "log"

    def test_log_link_roundtrip(self):
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily(link="log")
        mu = np.array([0.5, 1.0, 5.0, 10.0])
        assert_allclose(fam.linkinv(fam.link(mu)), mu)

    def test_inverse_link_roundtrip(self):
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily(link="inverse")
        mu = np.array([0.5, 1.0, 5.0, 10.0])
        assert_allclose(fam.linkinv(fam.link(mu)), mu, rtol=1e-10)

    def test_mu_eta_log(self):
        """d(exp(eta))/d(eta) = exp(eta) = mu."""
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily(link="log")
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(fam.mu_eta(eta), np.exp(eta))

    def test_mu_eta_inverse(self):
        """d(1/eta)/d(eta) = -1/eta^2."""
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily(link="inverse")
        eta = np.array([0.1, 0.5, 1.0, 2.0])
        assert_allclose(fam.mu_eta(eta), -1.0 / eta**2)

    def test_variance(self):
        """Variance function = mu^2."""
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily()
        mu = np.array([0.5, 1.0, 5.0])
        assert_allclose(fam.variance(mu), mu**2)

    def test_shape_validation(self):
        from interlace.glmm_family import GammaFamily

        with pytest.raises(ValueError, match="shape must be positive"):
            GammaFamily(shape=0.0)
        with pytest.raises(ValueError, match="shape must be positive"):
            GammaFamily(shape=-1.0)

    def test_link_validation(self):
        from interlace.glmm_family import GammaFamily

        with pytest.raises(ValueError, match="link must be"):
            GammaFamily(link="probit")

    def test_resolve_family(self):
        from interlace.glmm_family import GammaFamily, resolve_family

        fam = resolve_family("gamma")
        assert isinstance(fam, GammaFamily)

    def test_dev_resids_positive(self):
        """Deviance residuals should be non-negative."""
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily()
        y = np.array([1.0, 2.0, 5.0])
        mu = np.array([1.5, 1.8, 4.5])
        wt = np.ones(3)
        dr = fam.dev_resids(y, mu, wt)
        assert np.all(dr >= 0)

    def test_dev_resids_zero_at_saturated(self):
        """Deviance should be zero when mu == y."""
        from interlace.glmm_family import GammaFamily

        fam = GammaFamily()
        y = np.array([1.0, 2.0, 5.0])
        mu = y.copy()
        wt = np.ones(3)
        dr = fam.dev_resids(y, mu, wt)
        assert_allclose(dr, 0.0, atol=1e-14)


class TestGammaLoglik:
    """Test conditional log-likelihood for Gamma family."""

    def test_loglik_manual(self):
        """Check against manual Gamma log-pdf calculation."""
        from scipy.special import gammaln

        from interlace.glmm_family import GammaFamily
        from interlace.glmm_laplace import _conditional_loglik

        fam = GammaFamily(shape=3.0)
        y = np.array([2.0, 5.0])
        mu = np.array([3.0, 4.0])
        w = np.ones(2)
        ll = _conditional_loglik(y, mu, w, fam)

        # Manual: sum_i [(shape-1)*log(y_i) - shape*y_i/mu_i
        #                - shape*log(mu_i) + shape*log(shape) - lgamma(shape)]
        shape = 3.0
        expected = float(
            np.sum(
                (shape - 1) * np.log(y)
                - shape * y / mu
                - shape * np.log(mu)
                + shape * np.log(shape)
                - gammaln(shape)
            )
        )
        assert_allclose(ll, expected, rtol=1e-10)

    def test_loglik_with_phi_override(self):
        """When phi is provided, it overrides family.shape per-observation."""
        from scipy.special import gammaln

        from interlace.glmm_family import GammaFamily
        from interlace.glmm_laplace import _conditional_loglik

        fam = GammaFamily(shape=1.0)  # default shape, overridden by phi
        y = np.array([2.0, 5.0])
        mu = np.array([3.0, 4.0])
        w = np.ones(2)
        phi = np.array([3.0, 7.0])  # per-observation shape
        ll = _conditional_loglik(y, mu, w, fam, phi=phi)

        expected = float(
            np.sum(
                (phi - 1) * np.log(y)
                - phi * y / mu
                - phi * np.log(mu)
                + phi * np.log(phi)
                - gammaln(phi)
            )
        )
        assert_allclose(ll, expected, rtol=1e-10)

    def test_higher_shape_tighter(self):
        """Higher shape → lower variance → higher log-likelihood at true mean."""
        from interlace.glmm_family import GammaFamily
        from interlace.glmm_laplace import _conditional_loglik

        y = np.array([2.0, 2.0, 2.0])
        mu = np.array([2.0, 2.0, 2.0])
        w = np.ones(3)

        fam_low = GammaFamily(shape=1.0)
        fam_high = GammaFamily(shape=10.0)
        ll_low = _conditional_loglik(y, mu, w, fam_low)
        ll_high = _conditional_loglik(y, mu, w, fam_high)
        assert ll_high > ll_low


class TestGammaGLMM:
    """Integration tests: fit Gamma GLMM on simulated data."""

    def test_converged(self, gamma_fit):
        assert gamma_fit.converged

    def test_intercept_reasonable(self, gamma_fit):
        """Intercept should be within 0.5 of the true beta0=1.5."""
        intercept = gamma_fit.fe_params["Intercept"]
        assert abs(intercept - 1.5) < 0.5, f"Intercept={intercept:.3f}, expected ~1.5"

    def test_slope_reasonable(self, gamma_fit):
        """Slope should be within 0.3 of the true beta1=0.4."""
        slope = gamma_fit.fe_params["x"]
        assert abs(slope - 0.4) < 0.3, f"Slope={slope:.3f}, expected ~0.4"

    def test_variance_component_positive(self, gamma_fit):
        vc = gamma_fit.variance_components["group"]
        assert vc > 0, f"Variance component should be positive, got {vc}"

    def test_result_has_family(self, gamma_fit):
        from interlace.glmm_family import GammaFamily

        assert isinstance(gamma_fit.family, GammaFamily)

    def test_fittedvalues_positive(self, gamma_fit):
        """Gamma fitted values must be positive."""
        assert np.all(gamma_fit.fittedvalues > 0)

    def test_fittedvalues_shape(self, gamma_fit, gamma_data):
        assert gamma_fit.fittedvalues.shape == (len(gamma_data),)

    def test_aic_bic_finite(self, gamma_fit):
        assert np.isfinite(gamma_fit.aic)
        assert np.isfinite(gamma_fit.bic)

    def test_glmer_api(self, gamma_data):
        """Test that glmer() shortcut works with GammaFamily."""
        from interlace.glmm_family import GammaFamily

        result = interlace.glmer(
            "y ~ x",
            gamma_data,
            family=GammaFamily(link="log", shape=5.0),
            groups="group",
        )
        assert result.converged

    def test_inverse_link_converges(self):
        """Gamma GLMM with inverse link should converge on suitable data."""
        from interlace.glmm_family import GammaFamily

        # Simulate from an inverse-link model: eta = 1/mu, mu = 1/eta
        rng = np.random.default_rng(123)
        n_groups, n_per = 20, 15
        n = n_groups * n_per
        group = np.repeat(np.arange(n_groups), n_per)
        x = rng.normal(0, 0.1, size=n)  # small x to keep eta > 0
        u = rng.normal(0, 0.02, size=n_groups)
        eta = 0.5 + 0.05 * x + u[group]  # eta > 0 for inverse link
        mu = 1.0 / eta
        shape = 10.0
        y = rng.gamma(shape=shape, scale=mu / shape)
        df = pd.DataFrame({"y": y, "x": x, "group": group})

        result = fit_glmm(
            formula="y ~ x",
            data=df,
            family=GammaFamily(link="inverse", shape=shape),
            groups="group",
        )
        assert result.converged
        assert np.all(result.fittedvalues > 0)


# ---------------------------------------------------------------------------
# 8. NegativeBinomial1 (NB1) GLMM
# ---------------------------------------------------------------------------


def _simulate_nb1_glmm(
    n_groups: int = 30,
    n_per_group: int = 20,
    beta0: float = 1.0,
    beta1: float = 0.5,
    sigma_u: float = 0.4,
    alpha: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate data from an NB1 GLMM with log link.

    NB1: V(mu) = mu * (1 + alpha).  Parameterised as NB(r, p) with
    r = mu/alpha (observation-dependent) and p = 1/(1+alpha).

    Process:
      1. Draw group intercepts u_i ~ N(0, sigma_u^2).
      2. For each obs: eta = beta0 + beta1*x + u_{group}, mu = exp(eta).
      3. y ~ NB(r=mu/alpha, p=1/(1+alpha)).
    """
    rng = np.random.default_rng(seed)
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=n)
    u = rng.normal(0, sigma_u, size=n_groups)
    eta = beta0 + beta1 * x + u[group]
    mu = np.exp(eta)

    p = 1.0 / (1.0 + alpha)
    r = mu / alpha
    y = rng.negative_binomial(r, p)
    return pd.DataFrame({"y": y.astype(float), "x": x, "group": group})


@pytest.fixture(scope="module")
def nb1_data() -> pd.DataFrame:
    return _simulate_nb1_glmm()


@pytest.fixture(scope="module")
def nb1_fit(nb1_data) -> GLMMResult:
    """Fit NB1 GLMM: y ~ x + (1|group), alpha=1.5."""
    from interlace.glmm_family import NegativeBinomial1Family

    return fit_glmm(
        formula="y ~ x",
        data=nb1_data,
        family=NegativeBinomial1Family(alpha=1.5),
        groups="group",
    )


class TestNB1Family:
    """Unit tests for NegativeBinomial1Family class."""

    def test_class_exists(self):
        from interlace.glmm_family import NegativeBinomial1Family

        fam = NegativeBinomial1Family(alpha=2.0)
        assert fam.name == "negativebinomial1"
        assert fam.alpha == 2.0

    def test_default_alpha(self):
        from interlace.glmm_family import NegativeBinomial1Family

        fam = NegativeBinomial1Family()
        assert fam.alpha == 1.0

    def test_link_linkinv_roundtrip(self):
        from interlace.glmm_family import NegativeBinomial1Family

        fam = NegativeBinomial1Family()
        mu = np.array([0.5, 1.0, 5.0, 10.0])
        assert_allclose(fam.linkinv(fam.link(mu)), mu)

    def test_mu_eta(self):
        """d(exp(eta))/d(eta) = exp(eta) = mu."""
        from interlace.glmm_family import NegativeBinomial1Family

        fam = NegativeBinomial1Family()
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(fam.mu_eta(eta), np.exp(eta))

    def test_variance(self):
        """Variance function = mu * (1 + alpha)."""
        from interlace.glmm_family import NegativeBinomial1Family

        fam = NegativeBinomial1Family(alpha=2.0)
        mu = np.array([0.5, 1.0, 5.0])
        assert_allclose(fam.variance(mu), mu * 3.0)

    def test_alpha_validation(self):
        from interlace.glmm_family import NegativeBinomial1Family

        with pytest.raises(ValueError, match="alpha must be positive"):
            NegativeBinomial1Family(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be positive"):
            NegativeBinomial1Family(alpha=-1.0)

    def test_resolve_family(self):
        from interlace.glmm_family import NegativeBinomial1Family, resolve_family

        fam = resolve_family("negativebinomial1")
        assert isinstance(fam, NegativeBinomial1Family)

    def test_dev_resids_finite(self):
        """Deviance residuals should be finite.

        NB1 deviance can be negative (unlike NB2) because r = mu/alpha
        is observation-dependent, so the saturated model at mu=y does not
        necessarily maximise the NB1 log-likelihood.
        """
        from interlace.glmm_family import NegativeBinomial1Family

        fam = NegativeBinomial1Family(alpha=1.5)
        y = np.array([1.0, 3.0, 7.0])
        mu = np.array([2.0, 2.5, 6.0])
        wt = np.ones(3)
        dr = fam.dev_resids(y, mu, wt)
        assert np.all(np.isfinite(dr))

    def test_dev_resids_zero_at_saturated(self):
        """Deviance should be zero when mu == y (for y > 0)."""
        from interlace.glmm_family import NegativeBinomial1Family

        fam = NegativeBinomial1Family(alpha=1.5)
        y = np.array([1.0, 3.0, 7.0])
        mu = y.copy()
        wt = np.ones(3)
        dr = fam.dev_resids(y, mu, wt)
        assert_allclose(dr, 0.0, atol=1e-12)


class TestNB1Loglik:
    """Test conditional log-likelihood for NB1 family."""

    def test_loglik_manual(self):
        """Check against manual NB1 log-pmf calculation."""
        from scipy.special import gammaln

        from interlace.glmm_family import NegativeBinomial1Family
        from interlace.glmm_laplace import _conditional_loglik

        alpha = 2.0
        fam = NegativeBinomial1Family(alpha=alpha)
        y = np.array([3.0, 0.0, 5.0])
        mu = np.array([2.0, 1.5, 4.0])
        w = np.ones(3)
        ll = _conditional_loglik(y, mu, w, fam)

        # Manual NB1: r = mu/alpha, p = 1/(1+alpha)
        # ll_i = lgamma(y+r) - lgamma(r) - lgamma(y+1)
        #        + r*log(p) + y*log(1-p)
        r = mu / alpha
        p = 1.0 / (1.0 + alpha)
        expected = float(
            np.sum(
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1)
                + r * np.log(p)
                + y * np.log(1.0 - p)
            )
        )
        assert_allclose(ll, expected, rtol=1e-10)

    def test_loglik_with_phi_override(self):
        """When phi is provided, it overrides alpha per-observation."""
        from scipy.special import gammaln

        from interlace.glmm_family import NegativeBinomial1Family
        from interlace.glmm_laplace import _conditional_loglik

        fam = NegativeBinomial1Family(alpha=1.0)  # overridden by phi
        y = np.array([2.0, 4.0])
        mu = np.array([3.0, 3.0])
        w = np.ones(2)
        phi = np.array([1.5, 3.0])  # per-observation alpha
        ll = _conditional_loglik(y, mu, w, fam, phi=phi)

        r = mu / phi
        p = 1.0 / (1.0 + phi)
        expected = float(
            np.sum(
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1)
                + r * np.log(p)
                + y * np.log(1.0 - p)
            )
        )
        assert_allclose(ll, expected, rtol=1e-10)

    def test_small_alpha_approaches_poisson(self):
        """As alpha → 0, NB1 log-likelihood should approach Poisson."""
        from interlace.glmm_family import NegativeBinomial1Family
        from interlace.glmm_laplace import _conditional_loglik

        y = np.array([2.0, 3.0, 1.0])
        mu = np.array([2.5, 2.5, 2.5])
        w = np.ones(3)

        # Poisson ll for reference
        from scipy.special import gammaln

        pois_ll = float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))

        # NB1 with small alpha should be close
        fam = NegativeBinomial1Family(alpha=0.001)
        nb1_ll = _conditional_loglik(y, mu, w, fam)
        assert_allclose(nb1_ll, pois_ll, rtol=0.01)


class TestNB1GLMM:
    """Integration tests: fit NB1 GLMM on simulated data."""

    def test_converged(self, nb1_fit):
        assert nb1_fit.converged

    def test_intercept_reasonable(self, nb1_fit):
        """Intercept should be within 0.5 of the true beta0=1.0."""
        intercept = nb1_fit.fe_params["Intercept"]
        assert abs(intercept - 1.0) < 0.5, f"Intercept={intercept:.3f}, expected ~1.0"

    def test_slope_reasonable(self, nb1_fit):
        """Slope should be within 0.3 of the true beta1=0.5."""
        slope = nb1_fit.fe_params["x"]
        assert abs(slope - 0.5) < 0.3, f"Slope={slope:.3f}, expected ~0.5"

    def test_variance_component_positive(self, nb1_fit):
        vc = nb1_fit.variance_components["group"]
        assert vc > 0, f"Variance component should be positive, got {vc}"

    def test_result_has_family(self, nb1_fit):
        from interlace.glmm_family import NegativeBinomial1Family

        assert isinstance(nb1_fit.family, NegativeBinomial1Family)

    def test_fittedvalues_shape(self, nb1_fit, nb1_data):
        assert nb1_fit.fittedvalues.shape == (len(nb1_data),)

    def test_aic_bic_finite(self, nb1_fit):
        assert np.isfinite(nb1_fit.aic)
        assert np.isfinite(nb1_fit.bic)

    def test_nb1_vs_poisson_better_fit(self, nb1_data):
        """NB1 should fit overdispersed NB1 data better than Poisson."""
        from interlace.glmm_family import NegativeBinomial1Family

        nb1_result = fit_glmm(
            formula="y ~ x",
            data=nb1_data,
            family=NegativeBinomial1Family(alpha=1.5),
            groups="group",
        )
        pois_result = fit_glmm(
            formula="y ~ x",
            data=nb1_data,
            family="poisson",
            groups="group",
        )
        assert nb1_result.llf > pois_result.llf, (
            f"NB1 ll={nb1_result.llf:.2f} should be > Poisson ll={pois_result.llf:.2f}"
        )

    def test_glmer_api(self, nb1_data):
        """Test that glmer() shortcut works with NB1Family."""
        from interlace.glmm_family import NegativeBinomial1Family

        result = interlace.glmer(
            "y ~ x",
            nb1_data,
            family=NegativeBinomial1Family(alpha=1.5),
            groups="group",
        )
        assert result.converged
