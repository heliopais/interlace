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
