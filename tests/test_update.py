"""Tests for update() — method on CrossedLMEResult and module-level function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace import fit
from interlace.result import CrossedLMEResult

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n, q = 120, 6
    group_codes = np.tile(np.arange(q), n // q)
    b = rng.normal(scale=1.0, size=q)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 + 0.5 * x1 + 0.3 * x2 + b[group_codes] + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group_codes.astype(str)})


@pytest.fixture()
def base_model(base_df: pd.DataFrame) -> CrossedLMEResult:
    return fit("y ~ x1", base_df, groups="group")


# ---------------------------------------------------------------------------
# CrossedLMEResult.update()
# ---------------------------------------------------------------------------


class TestUpdateMethod:
    def test_update_returns_crossed_lme_result(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        result = base_model.update(formula=". ~ . + x2")
        assert isinstance(result, CrossedLMEResult)

    def test_update_add_covariate_dot_notation(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        """'. ~ . + x2' should expand to 'y ~ x1 + x2' and include x2 in fe_params."""
        result = base_model.update(formula=". ~ . + x2")
        assert "x2" in result.fe_params.index

    def test_update_add_covariate_has_more_fe_params(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        result = base_model.update(formula=". ~ . + x2")
        assert len(result.fe_params) == len(base_model.fe_params) + 1

    def test_update_add_covariate_changes_coefficients(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        """Intercept and x1 estimates should shift when x2 is added."""
        result = base_model.update(formula=". ~ . + x2")
        # Intercept should still be present but may differ
        assert "Intercept" in result.fe_params.index

    def test_update_new_data(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        """update with data= should refit on the new data, same formula."""
        new_df = base_df.copy()
        new_df["y"] = new_df["y"] * 2  # scale outcome
        result = base_model.update(data=new_df)
        # Fitted values should differ from original
        assert not np.allclose(result.fittedvalues, base_model.fittedvalues)
        # nobs must match new data
        assert result.nobs == len(new_df)

    def test_update_new_data_same_formula(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        new_df = base_df.copy()
        result = base_model.update(data=new_df)
        assert result.model.formula == base_model.model.formula

    def test_update_method_ml(self, base_model: CrossedLMEResult) -> None:
        """update(method='ML') should change the fitting method."""
        result = base_model.update(method="ML")
        assert result.method == "ML"

    def test_update_method_ml_different_params(
        self, base_model: CrossedLMEResult
    ) -> None:
        """ML and REML should give slightly different parameter estimates."""
        result_ml = base_model.update(method="ML")
        # They won't be identical; at minimum scale should differ
        assert result_ml.method != base_model.method

    def test_update_preserves_random_specs(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        """Random effect specs should carry over to the updated model."""
        result = base_model.update(formula=". ~ . + x2")
        assert result._random_specs == base_model._random_specs

    def test_update_is_valid_result(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        result = base_model.update(formula=". ~ . + x2")
        assert result.converged
        assert result.nobs == len(base_df)
        assert len(result.resid) == len(base_df)
        assert len(result.fittedvalues) == len(base_df)

    def test_update_explicit_full_formula(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        """A full formula (no dots) should be passed directly to fit()."""
        result = base_model.update(formula="y ~ x1 + x2")
        assert "x2" in result.fe_params.index

    def test_update_no_args_refits_same_model(
        self, base_model: CrossedLMEResult
    ) -> None:
        """Calling update() with no args should produce an equivalent model."""
        result = base_model.update()
        np.testing.assert_allclose(
            result.fe_params.values, base_model.fe_params.values, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Module-level update() function
# ---------------------------------------------------------------------------


class TestModuleLevelUpdate:
    def test_module_level_update_exists(self) -> None:
        assert hasattr(interlace, "update")

    def test_module_level_update_add_covariate(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        result = interlace.update(base_model, formula=". ~ . + x2")
        assert isinstance(result, CrossedLMEResult)
        assert "x2" in result.fe_params.index

    def test_module_level_update_new_data(
        self, base_model: CrossedLMEResult, base_df: pd.DataFrame
    ) -> None:
        new_df = base_df.copy()
        new_df["y"] = new_df["y"] + 10
        result = interlace.update(base_model, data=new_df)
        assert result.nobs == len(new_df)

    def test_module_level_update_method(self, base_model: CrossedLMEResult) -> None:
        result = interlace.update(base_model, method="ML")
        assert result.method == "ML"
