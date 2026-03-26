"""Tests for unseen group level handling in predict().

Contract: group levels in newdata not seen during training contribute 0
to the prediction (shrinkage to population mean). No error, no warning.

Covers:
  - Single-RE: fully unseen group → prediction equals X@beta
  - Single-RE: mixed rows (known + unseen groups) → correct per-row handling
  - Multi-RE: unseen in one factor, known in the other → only known contributes
  - No exception is raised for unseen levels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace


@pytest.fixture(scope="module")
def fitted_single_re():
    rng = np.random.default_rng(7)
    n_groups = 10
    n_per = 20
    group_ids = np.repeat([f"g{i}" for i in range(n_groups)], n_per)
    x = rng.standard_normal(n_groups * n_per)
    u = rng.normal(0, 1.0, size=n_groups)
    eps = rng.normal(0, 0.5, size=n_groups * n_per)
    y = 2.0 + 0.5 * x + u[np.repeat(np.arange(n_groups), n_per)] + eps
    df = pd.DataFrame({"y": y, "x": x, "group": group_ids})
    return df, interlace.fit("y ~ x", data=df, groups="group")


@pytest.fixture(scope="module")
def fitted_two_re():
    rng = np.random.default_rng(8)
    n_firms, n_depts, n = 10, 5, 500
    firms = rng.choice([f"f{i}" for i in range(n_firms)], n)
    depts = rng.choice([f"d{i}" for i in range(n_depts)], n)
    x = rng.standard_normal(n)
    y = (
        1.0
        + 0.6 * x
        + rng.normal(0, 0.8, n)
        + rng.normal(0, 0.4, n)
        + rng.normal(0, 0.3, n)
    )
    df = pd.DataFrame({"y": y, "x": x, "firm": firms, "dept": depts})
    return df, interlace.fit("y ~ x", data=df, groups=["firm", "dept"])


def test_unseen_group_no_error(fitted_single_re):
    _, result = fitted_single_re
    newdata = pd.DataFrame({"x": [1.0, 2.0], "group": ["UNSEEN_A", "UNSEEN_B"]})
    # Must not raise
    pred = result.predict(newdata=newdata)
    assert pred.shape == (2,)


def test_unseen_group_equals_fe_only(fitted_single_re):
    _, result = fitted_single_re
    newdata = pd.DataFrame({"x": [1.0, -0.5], "group": ["UNSEEN_A", "UNSEEN_B"]})
    pred_with_re = result.predict(newdata=newdata, include_re=True)
    pred_fe_only = result.predict(newdata=newdata, include_re=False)
    # Unseen groups contribute 0, so RE prediction == FE-only prediction
    np.testing.assert_allclose(pred_with_re, pred_fe_only, rtol=1e-10)


def test_mixed_known_and_unseen_groups(fitted_single_re):
    df, result = fitted_single_re
    known_group = df["group"].iloc[0]  # definitely seen during training
    newdata = pd.DataFrame(
        {
            "x": [1.0, 1.0],
            "group": [known_group, "UNSEEN_Z"],
        }
    )
    pred = result.predict(newdata=newdata)
    pred_fe = result.predict(newdata=newdata, include_re=False)

    # Known row: pred != pred_fe (unless BLUP happens to be 0)
    known_blup = result.random_effects["group"][known_group]
    np.testing.assert_allclose(pred[0], pred_fe[0] + known_blup, rtol=1e-8)

    # Unseen row: pred == pred_fe
    np.testing.assert_allclose(pred[1], pred_fe[1], rtol=1e-10)


def test_two_re_unseen_in_one_factor(fitted_two_re):
    df, result = fitted_two_re
    known_dept = df["dept"].iloc[0]
    newdata = pd.DataFrame(
        {
            "x": [0.0],
            "firm": ["UNSEEN_FIRM"],
            "dept": [known_dept],
        }
    )
    pred = result.predict(newdata=newdata, include_re=True)
    pred_fe = result.predict(newdata=newdata, include_re=False)

    # Only dept BLUP should contribute; firm is unseen → 0
    dept_blup = result.random_effects["dept"][known_dept]
    np.testing.assert_allclose(pred[0], pred_fe[0] + dept_blup, rtol=1e-8)


def test_two_re_both_unseen(fitted_two_re):
    _, result = fitted_two_re
    newdata = pd.DataFrame(
        {
            "x": [1.5],
            "firm": ["UNSEEN_FIRM"],
            "dept": ["UNSEEN_DEPT"],
        }
    )
    pred = result.predict(newdata=newdata, include_re=True)
    pred_fe = result.predict(newdata=newdata, include_re=False)
    np.testing.assert_allclose(pred, pred_fe, rtol=1e-10)
