"""Parity test: interlace coxme vs R coxme::coxme() reference values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURE = Path(__file__).parent / "fixtures" / "coxme_parity.json"


@pytest.fixture(scope="module")
def ref():
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fitted(ref):
    """Fit interlace.coxme on the same data used for the R fixture."""
    df = pd.DataFrame(
        {
            "time": ref["time"],
            "event": ref["event"],
            "x1": ref["x1"],
            "x2": ref["x2"],
            "group": ref["group"],
        }
    )
    result = interlace.coxme("Surv(time, event) ~ x1 + x2", df, groups="group")
    return result


class TestCoxmeParity:
    """Compare interlace.coxme output to R coxme::coxme() reference."""

    def test_beta_x1(self, fitted, ref):
        r_beta = ref["beta"]["x1"]
        assert abs(fitted.fe_params["x1"] - r_beta) < 0.02, (
            f"x1: interlace={fitted.fe_params['x1']:.4f}, R={r_beta:.4f}"
        )

    def test_beta_x2(self, fitted, ref):
        r_beta = ref["beta"]["x2"]
        assert abs(fitted.fe_params["x2"] - r_beta) < 0.02, (
            f"x2: interlace={fitted.fe_params['x2']:.4f}, R={r_beta:.4f}"
        )

    def test_se_x1(self, fitted, ref):
        r_se = ref["se"]["x1"]
        rel_diff = abs(fitted.fe_bse["x1"] - r_se) / r_se
        assert rel_diff < 0.01, (
            f"SE(x1): interlace={fitted.fe_bse['x1']:.5f}, "
            f"R={r_se:.5f}, rel={rel_diff:.3f}"
        )

    def test_se_x2(self, fitted, ref):
        r_se = ref["se"]["x2"]
        rel_diff = abs(fitted.fe_bse["x2"] - r_se) / r_se
        assert rel_diff < 0.01, (
            f"SE(x2): interlace={fitted.fe_bse['x2']:.5f}, "
            f"R={r_se:.5f}, rel={rel_diff:.3f}"
        )

    def test_frailty_variance(self, fitted, ref):
        r_var = ref["frailty_variance"]
        est_var = fitted.variance_components["group"]
        rel_diff = abs(est_var - r_var) / r_var
        assert rel_diff < 0.15, (
            f"frailty_var: interlace={est_var:.4f}, R={r_var:.4f}, rel={rel_diff:.3f}"
        )

    def test_blup_correlation(self, fitted, ref):
        """BLUPs should be highly correlated with R's ranef."""
        r_blups = np.array(list(ref["blups"].values()))
        # Align by group level (R uses 1-indexed factor levels)
        py_blups = fitted.random_effects["group"]
        # R groups are 1..30, Python uses the raw integer values
        py_arr = np.array([py_blups.get(i, 0.0) for i in sorted(py_blups.index)])
        corr = np.corrcoef(py_arr, r_blups)[0, 1]
        assert corr > 0.99, f"BLUP correlation: {corr:.4f}"
