"""Tests for Cox PH with Gaussian frailty (coxme)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_coxme_data(
    n_groups: int = 50,
    n_per_group: int = 40,
    beta_true: np.ndarray | None = None,
    frailty_sd: float = 0.5,
    censoring_rate: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Simulate survival data with a Gaussian shared frailty.

    Model: h(t|x,b) = h0(t) * exp(x'beta + b_j)
    where h0(t) = 1 (unit exponential baseline), b_j ~ N(0, frailty_sd^2).
    """
    rng = np.random.default_rng(seed)
    if beta_true is None:
        beta_true = np.array([0.5, -0.3])

    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)

    # Covariates
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2])

    # Frailty
    b = rng.normal(0, frailty_sd, size=n_groups)
    eta = X @ beta_true + b[group]

    # Event times: exponential with rate exp(eta), so T = -log(U)/exp(eta)
    u = rng.uniform(size=n)
    event_time = -np.log(u) / np.exp(eta)

    # Censoring times
    censor_time = rng.exponential(scale=1.0 / censoring_rate, size=n)

    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    df = pd.DataFrame(
        {
            "time": time,
            "event": event,
            "x1": x1,
            "x2": x2,
            "group": group,
        }
    )
    truth = {
        "beta": beta_true,
        "frailty_sd": frailty_sd,
        "frailty_var": frailty_sd**2,
        "b": b,
    }
    return df, truth


# ---------------------------------------------------------------------------
# API surface tests
# ---------------------------------------------------------------------------


class TestCoxmeAPI:
    """Test that the public API exists and returns the right types."""

    @pytest.fixture()
    def sim_data(self):
        df, truth = _simulate_coxme_data(n_groups=20, n_per_group=20, seed=1)
        return df, truth

    def test_coxme_callable(self):
        assert callable(interlace.coxme)

    def test_coxme_returns_result(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert isinstance(result, interlace.CoxmeResult)

    def test_result_has_fe_params(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert isinstance(result.fe_params, pd.Series)
        assert len(result.fe_params) == 2
        assert "x1" in result.fe_params.index
        assert "x2" in result.fe_params.index

    def test_result_has_fe_bse(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert isinstance(result.fe_bse, pd.Series)
        assert len(result.fe_bse) == 2
        assert np.all(result.fe_bse > 0)

    def test_result_has_random_effects(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert "group" in result.random_effects
        assert len(result.random_effects["group"]) == 20

    def test_result_has_variance_components(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert "group" in result.variance_components
        assert result.variance_components["group"] > 0

    def test_result_has_concordance(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert 0.0 < result.concordance < 1.0

    def test_result_has_n_events(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert result.n_events == df["event"].sum()
        assert result.n_events > 0

    def test_result_converged(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert result.converged is True

    def test_result_has_llf(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert np.isfinite(result.llf)

    def test_result_has_nobs_ngroups(self, sim_data):
        df, _ = sim_data
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        assert result.nobs == len(df)
        assert result.ngroups == {"group": 20}


# ---------------------------------------------------------------------------
# Surv() formula parsing
# ---------------------------------------------------------------------------


class TestSurvFormula:
    """Test parsing of Surv(time, event) ~ ... formulas."""

    def test_surv_basic(self):
        from interlace.coxme import parse_surv_formula

        time_col, event_col, rhs = parse_surv_formula("Surv(time, status) ~ x1 + x2")
        assert time_col == "time"
        assert event_col == "status"
        assert rhs == "x1 + x2"

    def test_surv_whitespace(self):
        from interlace.coxme import parse_surv_formula

        time_col, event_col, rhs = parse_surv_formula("Surv( time , event ) ~ x1")
        assert time_col == "time"
        assert event_col == "event"
        assert rhs == "x1"

    def test_surv_no_surv_raises(self):
        from interlace.coxme import parse_surv_formula

        with pytest.raises(ValueError, match="Surv"):
            parse_surv_formula("time ~ x1 + x2")

    def test_surv_intercept_only_rhs(self):
        """Surv(t, e) ~ 1 should work (no covariates)."""
        from interlace.coxme import parse_surv_formula

        time_col, event_col, rhs = parse_surv_formula("Surv(t, e) ~ 1")
        assert time_col == "t"
        assert event_col == "e"
        assert rhs.strip() == "1"


# ---------------------------------------------------------------------------
# Parameter recovery on simulated data
# ---------------------------------------------------------------------------


class TestParameterRecovery:
    """With enough data, estimates should be close to true values."""

    @pytest.fixture(scope="class")
    def fitted(self):
        df, truth = _simulate_coxme_data(n_groups=50, n_per_group=40, seed=42)
        result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
        return result, truth

    def test_beta_x1_close(self, fitted):
        result, truth = fitted
        assert abs(result.fe_params["x1"] - truth["beta"][0]) < 0.05

    def test_beta_x2_close(self, fitted):
        result, truth = fitted
        assert abs(result.fe_params["x2"] - truth["beta"][1]) < 0.08

    def test_frailty_variance_recovers_order_of_magnitude(self, fitted):
        result, truth = fitted
        est_var = result.variance_components["group"]
        true_var = truth["frailty_var"]
        # Laplace approximation has known downward bias for Cox frailty;
        # 35% relative tolerance accommodates this.
        rel_diff = abs(est_var - true_var) / true_var
        assert rel_diff < 0.35

    def test_blup_correlation_with_true_frailties(self, fitted):
        result, truth = fitted
        blups = result.random_effects["group"].values
        true_b = truth["b"]
        corr = np.corrcoef(blups, true_b)[0, 1]
        assert corr > 0.90

    def test_concordance_better_than_chance(self, fitted):
        result, _ = fitted
        assert result.concordance > 0.68


# ---------------------------------------------------------------------------
# Breslow partial likelihood
# ---------------------------------------------------------------------------


class TestBreslowLoglik:
    """Unit tests for the Breslow partial log-likelihood."""

    def test_zero_coef_loglik(self):
        """With eta=0 everywhere, loglik = sum(delta_i * -log(n_at_risk))."""
        from interlace.coxme import breslow_loglik

        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.array([1, 1, 0, 1, 1])
        eta = np.zeros(5)
        ll = breslow_loglik(eta, time, event)
        # At each event: -log(risk set size)
        # t=1: risk=5, t=2: risk=4, t=4: risk=2, t=5: risk=1
        expected = -np.log(5) - np.log(4) - np.log(2) - np.log(1)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_loglik_increases_with_correct_ordering(self):
        """Positive eta for early events should increase the loglik."""
        from interlace.coxme import breslow_loglik

        time = np.array([1.0, 2.0, 3.0])
        event = np.array([1, 1, 0])
        # eta=0 baseline
        ll0 = breslow_loglik(np.zeros(3), time, event)
        # Positive eta for subjects who die early -> higher loglik
        ll1 = breslow_loglik(np.array([1.0, 0.5, -1.0]), time, event)
        assert ll1 > ll0

    def test_gradient_finite_difference(self):
        """Gradient matches finite differences."""
        from interlace.coxme import breslow_loglik, breslow_score

        rng = np.random.default_rng(123)
        time = rng.exponential(size=20)
        event = rng.binomial(1, 0.7, size=20)
        eta = rng.normal(size=20) * 0.3

        score = breslow_score(eta, time, event)
        eps = 1e-6
        grad_fd = np.zeros_like(eta)
        for i in range(len(eta)):
            eta_p = eta.copy()
            eta_p[i] += eps
            eta_m = eta.copy()
            eta_m[i] -= eps
            grad_fd[i] = (
                breslow_loglik(eta_p, time, event) - breslow_loglik(eta_m, time, event)
            ) / (2 * eps)
        np.testing.assert_allclose(score, grad_fd, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_covariate(self):
        """Model with a single covariate should work."""
        df, _ = _simulate_coxme_data(
            n_groups=20,
            n_per_group=20,
            beta_true=np.array([0.5, -0.3]),
            seed=7,
        )
        result = interlace.coxme("Surv(time, event) ~ x1", df, groups="group")
        assert len(result.fe_params) == 1
        assert result.converged

    def test_random_kwarg(self):
        """random=['(1|group)'] should work like groups='group'."""
        df, _ = _simulate_coxme_data(n_groups=20, n_per_group=20, seed=8)
        result = interlace.coxme(
            "Surv(time, event) ~ x1 + x2",
            df,
            random=["(1|group)"],
        )
        assert "group" in result.random_effects
        assert result.converged

    def test_no_groups_raises(self):
        df, _ = _simulate_coxme_data(n_groups=10, n_per_group=10, seed=9)
        with pytest.raises(ValueError, match="groups.*random"):
            interlace.coxme("Surv(time, event) ~ x1 + x2", df)
