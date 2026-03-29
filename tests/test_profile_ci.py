"""Tests for profile_confint in profile_ci.py.

Core invariants for each theta_i CI [lower, upper]:
  - lower < theta_hat_i < upper
  - 2*(llf_max - profile_loglik(lower_theta)) ≈ chi2(level, df=1)
  - 2*(llf_max - profile_loglik(upper_theta)) ≈ chi2(level, df=1)
  - level=0.99 gives strictly wider intervals than level=0.95
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace.profile_ci import profile_confint
from interlace.profiled_reml import fit_ml, profile_loglik

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def single_re_result(rng):
    """Simple intercept-only model fit via interlace.fit() — dyestuff-like."""
    import pandas as pd

    n, q = 150, 6
    codes = np.repeat(np.arange(q), n // q)
    b = rng.normal(scale=5.0, size=q)
    y = 10.0 + b[codes] + rng.normal(scale=8.0, size=n)
    df = pd.DataFrame({"y": y, "group": codes.astype(str)})
    return interlace.fit("y ~ 1", data=df, groups="group")


@pytest.fixture(scope="module")
def dyestuff_result():
    """Dyestuff-like model using the CSV fixture data."""
    data = pd.read_csv(FIXTURES / "lme4_dyestuff_data.csv")
    return interlace.fit("Yield ~ 1", data=data, groups="Batch")


@pytest.fixture(scope="module")
def two_re_result(rng):
    """Two crossed random effects."""
    import pandas as pd

    n, q1, q2 = 240, 15, 8
    codes1 = np.tile(np.arange(q1), n // q1)
    codes2 = np.repeat(np.arange(q2), n // q2)
    b1 = rng.normal(scale=2.0, size=q1)
    b2 = rng.normal(scale=1.5, size=q2)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = 3.0 + X[:, 1] * 0.5 + b1[codes1] + b2[codes2] + rng.normal(scale=3.0, size=n)
    df = pd.DataFrame(
        {"y": y, "x": X[:, 1], "g1": codes1.astype(str), "g2": codes2.astype(str)}
    )
    return interlace.fit("y ~ x", data=df, random=["(1|g1)", "(1|g2)"])


# ---------------------------------------------------------------------------
# Tests: structure and basic properties
# ---------------------------------------------------------------------------


class TestProfileConfintStructure:
    def test_returns_dataframe(self, single_re_result) -> None:
        ci = profile_confint(single_re_result)
        assert isinstance(ci, pd.DataFrame)

    def test_has_three_columns(self, single_re_result) -> None:
        ci = profile_confint(single_re_result)
        assert ci.shape[1] == 3

    def test_column_names_contain_percentages(self, single_re_result) -> None:
        ci = profile_confint(single_re_result)
        cols = list(ci.columns)
        assert "estimate" in cols
        # Two CI bound columns containing '%'
        pct_cols = [c for c in cols if "%" in str(c)]
        assert len(pct_cols) == 2

    def test_one_row_per_theta_single_re(self, single_re_result) -> None:
        ci = profile_confint(single_re_result)
        n_theta = len(single_re_result.theta)
        assert len(ci) == n_theta

    def test_two_rows_for_two_re(self, two_re_result) -> None:
        ci = profile_confint(two_re_result)
        n_theta = len(two_re_result.theta)
        assert len(ci) == n_theta

    def test_estimate_column_matches_ml_theta(self, single_re_result) -> None:
        """Estimate should be the ML theta, not REML theta."""
        ci = profile_confint(single_re_result)
        # Fit with ML directly to get reference
        y = single_re_result.model.endog
        X = single_re_result.model.exog
        Z = single_re_result._Z
        ml_fit = fit_ml(
            y,
            X,
            Z,
            q_sizes=[],
            specs=single_re_result._random_specs,
            n_levels=single_re_result._n_levels,
        )
        np.testing.assert_allclose(ci["estimate"].values, ml_fit.theta, rtol=1e-6)

    def test_level_99_wider_than_level_95(self, single_re_result) -> None:
        ci95 = profile_confint(single_re_result, level=0.95)
        ci99 = profile_confint(single_re_result, level=0.99)
        lo95_col = [c for c in ci95.columns if "%" in str(c)][0]
        hi95_col = [c for c in ci95.columns if "%" in str(c)][1]
        lo99_col = [c for c in ci99.columns if "%" in str(c)][0]
        hi99_col = [c for c in ci99.columns if "%" in str(c)][1]

        # 99% lower bound strictly below 95% lower bound (wider left tail)
        assert ci99[lo99_col].values[0] <= ci95[lo95_col].values[0]
        # 99% upper bound strictly above 95% upper bound (wider right tail)
        assert ci99[hi99_col].values[0] >= ci95[hi95_col].values[0]


# ---------------------------------------------------------------------------
# Tests: LRT invariant at CI boundaries
# ---------------------------------------------------------------------------


class TestProfileConfintLRTInvariant:
    """The LRT statistic must equal chi2_crit at each CI boundary."""

    def _check_lrt_at_boundary(self, result, ci, level, tol=0.05) -> None:
        """Verify 2*(llf_max - L(boundary)) ≈ chi2(level, df=1) for all params."""
        from scipy.stats import chi2

        y = result.model.endog
        X = result.model.exog
        Z = result._Z
        specs = result._random_specs
        n_levels = result._n_levels

        ml_fit = fit_ml(y, X, Z, q_sizes=[], specs=specs, n_levels=n_levels)
        llf_max = ml_fit.llf
        theta_hat = ml_fit.theta
        chi2_crit = chi2.ppf(level, df=1)

        lo_col = [c for c in ci.columns if "%" in str(c)][0]
        hi_col = [c for c in ci.columns if "%" in str(c)][1]

        lo_vals = ci[lo_col].values
        hi_vals = ci[hi_col].values
        for i, (lo_val, hi_val) in enumerate(zip(lo_vals, hi_vals, strict=True)):
            # Upper boundary
            theta_hi = theta_hat.copy()
            theta_hi[i] = hi_val
            ll_hi = profile_loglik(
                theta_hi, y, X, Z, [], specs=specs, n_levels=n_levels
            )
            lrt_hi = 2.0 * (llf_max - ll_hi)
            assert abs(lrt_hi - chi2_crit) < tol * chi2_crit, (
                f"theta[{i}] upper: LRT={lrt_hi:.4f}, expected {chi2_crit:.4f}"
            )

            # Lower boundary (might be 0 if boundary hit)
            if lo_val > 1e-7:
                theta_lo = theta_hat.copy()
                theta_lo[i] = lo_val
                ll_lo = profile_loglik(
                    theta_lo, y, X, Z, [], specs=specs, n_levels=n_levels
                )
                lrt_lo = 2.0 * (llf_max - ll_lo)
                assert abs(lrt_lo - chi2_crit) < tol * chi2_crit, (
                    f"theta[{i}] lower: LRT={lrt_lo:.4f}, expected {chi2_crit:.4f}"
                )

    def test_single_re_lrt_invariant(self, single_re_result) -> None:
        ci = profile_confint(single_re_result, level=0.95)
        self._check_lrt_at_boundary(single_re_result, ci, level=0.95)

    def test_dyestuff_lrt_invariant(self, dyestuff_result) -> None:
        ci = profile_confint(dyestuff_result, level=0.95)
        self._check_lrt_at_boundary(dyestuff_result, ci, level=0.95)

    def test_two_re_lrt_invariant(self, two_re_result) -> None:
        ci = profile_confint(two_re_result, level=0.95)
        self._check_lrt_at_boundary(two_re_result, ci, level=0.95)

    def test_bounds_bracket_mle(self, single_re_result) -> None:
        """Each CI must contain the MLE."""
        y = single_re_result.model.endog
        X = single_re_result.model.exog
        Z = single_re_result._Z
        ml_fit = fit_ml(
            y,
            X,
            Z,
            q_sizes=[],
            specs=single_re_result._random_specs,
            n_levels=single_re_result._n_levels,
        )
        theta_hat = ml_fit.theta

        ci = profile_confint(single_re_result, level=0.95)
        lo_col = [c for c in ci.columns if "%" in str(c)][0]
        hi_col = [c for c in ci.columns if "%" in str(c)][1]

        lo_vals = ci[lo_col].values
        hi_vals = ci[hi_col].values
        for i, (lo_val, hi_val) in enumerate(zip(lo_vals, hi_vals, strict=True)):
            assert lo_val <= theta_hat[i] <= hi_val, (
                f"theta[{i}] MLE {theta_hat[i]:.4f} not in [{lo_val:.4f}, {hi_val:.4f}]"
            )


# ---------------------------------------------------------------------------
# Tests: confint() method on CrossedLMEResult
# ---------------------------------------------------------------------------


class TestConfintMethod:
    def test_confint_profile_delegates(self, single_re_result) -> None:
        """result.confint(method='profile') matches profile_confint()."""
        ci_fn = profile_confint(single_re_result)
        ci_method = single_re_result.confint(method="profile")
        pd.testing.assert_frame_equal(ci_fn, ci_method)

    def test_confint_invalid_method_raises(self, single_re_result) -> None:
        with pytest.raises(ValueError, match="method"):
            single_re_result.confint(method="bootstrap")

    def test_confint_default_is_profile(self, single_re_result) -> None:
        ci_explicit = single_re_result.confint(method="profile")
        ci_default = single_re_result.confint()
        pd.testing.assert_frame_equal(ci_explicit, ci_default)
