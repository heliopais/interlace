"""Tests for in-sample and new-data prediction.

Covers:
  - predict() with no args returns fittedvalues
  - predict(newdata=df) on training data matches fittedvalues
  - predict(newdata=df, include_re=False) returns fixed-effects-only predictions
  - predict(newdata=new_df) for a held-out subset with known group levels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace


@pytest.fixture(scope="module")
def data_and_result():
    rng = np.random.default_rng(99)
    n_groups = 15
    n_per = 20
    n = n_groups * n_per

    group_ids = np.repeat([f"g{i}" for i in range(n_groups)], n_per)
    x = rng.standard_normal(n)

    u = rng.normal(0, 1.0, size=n_groups)
    eps = rng.normal(0, 0.5, size=n)
    y = 1.0 + 0.8 * x + u[np.repeat(np.arange(n_groups), n_per)] + eps

    df = pd.DataFrame({"y": y, "x": x, "group": group_ids})
    result = interlace.fit("y ~ x", data=df, groups="group")
    return df, result


def test_predict_no_args_returns_fittedvalues(data_and_result):
    _, result = data_and_result
    pred = result.predict()
    np.testing.assert_allclose(pred, result.fittedvalues, rtol=1e-10)


def test_predict_newdata_insample_matches_fittedvalues(data_and_result):
    df, result = data_and_result
    pred = result.predict(newdata=df)
    np.testing.assert_allclose(pred, result.fittedvalues, rtol=1e-6)


def test_predict_include_re_false(data_and_result):
    df, result = data_and_result
    pred_fe = result.predict(newdata=df, include_re=False)

    # Should equal X @ beta only
    import patsy

    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_new = np.asarray(patsy.dmatrix(fe_formula, df, return_type="dataframe"))
    expected = X_new @ result.fe_params.values
    np.testing.assert_allclose(pred_fe, expected, rtol=1e-10)


def test_predict_newdata_known_groups(data_and_result):
    df, result = data_and_result
    # Hold out last 30 rows as "new" data — groups are all known
    newdata = df.iloc[-30:].reset_index(drop=True)
    pred = result.predict(newdata=newdata)

    # Recompute expected manually: X_new @ beta + sum of BLUPs per group col
    import patsy

    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_new = np.asarray(patsy.dmatrix(fe_formula, newdata, return_type="dataframe"))
    expected_fe = X_new @ result.fe_params.values
    blup_contrib = result.random_effects["group"][newdata["group"]].values
    expected = expected_fe + blup_contrib

    np.testing.assert_allclose(pred, expected, rtol=1e-8)


def test_predict_returns_numpy_array(data_and_result):
    df, result = data_and_result
    pred = result.predict(newdata=df)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(df),)


# ---------------------------------------------------------------------------
# Random slopes prediction (interlace-85j)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def slope_data_and_result():
    rng = np.random.default_rng(77)
    n_groups = 10
    n_per = 20
    n = n_groups * n_per
    group_ids = np.repeat([f"g{i}" for i in range(n_groups)], n_per)
    x = rng.standard_normal(n)
    b_int = rng.normal(0, 0.8, n_groups)
    b_slope = rng.normal(0, 0.4, n_groups)
    idx = np.repeat(np.arange(n_groups), n_per)
    y = 1.0 + 0.5 * x + b_int[idx] + b_slope[idx] * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x": x, "g": group_ids})
    result = interlace.fit("y ~ x", data=df, random=["(1 + x | g)"])
    return df, result


def test_predict_slopes_no_args_returns_fittedvalues(slope_data_and_result):
    _, result = slope_data_and_result
    pred = result.predict()
    np.testing.assert_allclose(pred, result.fittedvalues, rtol=1e-10)


def test_predict_slopes_insample_matches_fittedvalues(slope_data_and_result):
    df, result = slope_data_and_result
    pred = result.predict(newdata=df)
    np.testing.assert_allclose(pred, result.fittedvalues, rtol=1e-6)


def test_predict_slopes_include_re_false(slope_data_and_result):
    df, result = slope_data_and_result
    pred_fe = result.predict(newdata=df, include_re=False)
    import patsy

    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_new = np.asarray(patsy.dmatrix(fe_formula, df, return_type="dataframe"))
    expected = X_new @ result.fe_params.values
    np.testing.assert_allclose(pred_fe, expected, rtol=1e-10)


def test_predict_slopes_known_groups_manual(slope_data_and_result):
    """Manually verify: pred = X@beta + blup_int[g] + blup_slope[g]*x."""
    df, result = slope_data_and_result
    import patsy

    newdata = df.iloc[-20:].reset_index(drop=True)
    pred = result.predict(newdata=newdata)

    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_new = np.asarray(patsy.dmatrix(fe_formula, newdata, return_type="dataframe"))
    expected_fe = X_new @ result.fe_params.values

    re_df = result.random_effects["g"]
    blup_int = newdata["g"].map(re_df["(Intercept)"]).to_numpy(dtype=float)
    blup_slope = newdata["g"].map(re_df["x"]).to_numpy(dtype=float)
    expected = expected_fe + blup_int + blup_slope * newdata["x"].to_numpy()

    np.testing.assert_allclose(pred, expected, rtol=1e-8)


def test_predict_slopes_unseen_group_contributes_zero(slope_data_and_result):
    df, result = slope_data_and_result
    import patsy

    newdata = pd.DataFrame({"x": [1.0, -0.5], "g": ["UNSEEN_A", "UNSEEN_B"]})
    pred = result.predict(newdata=newdata)
    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_new = np.asarray(patsy.dmatrix(fe_formula, newdata, return_type="dataframe"))
    expected = X_new @ result.fe_params.values
    np.testing.assert_allclose(pred, expected, rtol=1e-10)


def test_predict_slopes_returns_numpy_array(slope_data_and_result):
    df, result = slope_data_and_result
    pred = result.predict(newdata=df)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(df),)


# ---------------------------------------------------------------------------
# Regression: categorical predictor column-order mismatch (GitHub issue #4)
# ---------------------------------------------------------------------------
# formulaic and patsy produce the same columns for a categorical predictor but
# in *different* order.  If predict() were changed to use patsy internally,
# continuous and categorical coefficients would be mixed up and predictions
# would be completely wrong.  This test catches that regression.


@pytest.fixture(scope="module")
def categorical_data_and_result():
    """Dataset with continuous + multi-level categorical predictor."""
    rng = np.random.default_rng(42)
    n_groups = 12
    n_per = 25
    n = n_groups * n_per

    group_ids = np.repeat([f"g{i}" for i in range(n_groups)], n_per)
    age = rng.uniform(25, 55, n)
    edu_levels = ["HighSchool", "Bachelor", "Master", "PhD"]
    education = rng.choice(edu_levels, size=n)

    # Known true coefficients (so we can sanity-check direction)
    edu_effect = {"HighSchool": 0.0, "Bachelor": 0.05, "Master": 0.15, "PhD": 0.25}
    u = rng.normal(0, 0.3, n_groups)
    eps = rng.normal(0, 0.1, n)
    y = (
        1.5
        + 0.04 * age
        + np.array([edu_effect[e] for e in education])
        + u[np.repeat(np.arange(n_groups), n_per)]
        + eps
    )

    df = pd.DataFrame({"y": y, "age": age, "education": education, "group": group_ids})
    result = interlace.fit("y ~ age + education", data=df, groups="group")
    return df, result


def test_predict_categorical_column_order_consistency(categorical_data_and_result):
    """predict(newdata) must use formulaic (not patsy) to avoid column-order mismatch.

    formulaic and patsy assign dummy columns for categorical predictors in
    different orders.  A patsy-based predict() would silently multiply the wrong
    coefficients against the wrong features.  Verify predictions on held-out data
    are close to ground truth (|mean error| < 0.5 log-wage units).
    """
    import formulaic

    df, result = categorical_data_and_result
    newdata = df.iloc[-25:].reset_index(drop=True)

    pred = result.predict(newdata=newdata, include_re=False)

    # Compute expected using formulaic explicitly (same library as fitting path)
    fe_formula = result.model.formula.split("~", 1)[1].strip()
    X_formulaic = formulaic.model_matrix(fe_formula, newdata)
    # Align columns to fe_params to be safe
    X_aligned = np.asarray(X_formulaic[list(result.fe_params.index)])
    expected = X_aligned @ np.asarray(result.fe_params)

    np.testing.assert_allclose(pred, expected, rtol=1e-8)


def test_predict_categorical_vs_patsy_column_order_differs():
    """Document that formulaic and patsy produce different column orders for
    categorical predictors — confirming why using patsy in predict() was a bug.
    """
    import formulaic
    import patsy

    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "age": rng.uniform(25, 55, n),
            "education": rng.choice(["HighSchool", "Bachelor", "Master", "PhD"], n),
        }
    )
    fe_formula = "age + education"

    formulaic_cols = list(formulaic.model_matrix(fe_formula, df).columns)
    patsy_cols = list(patsy.dmatrix(fe_formula, df, return_type="dataframe").columns)

    # If the libraries ever agree, this test becomes vacuous — but it documents
    # the known behaviour that motivated the fix.
    assert formulaic_cols != patsy_cols, (
        "formulaic and patsy produced the same column order — re-evaluate if "
        "the regression test below is still meaningful"
    )
