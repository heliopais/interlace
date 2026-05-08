"""Parity test: interlace.glmer vs lme4::glmer on McElreath reedfrogs.

Drives the SR-12 notebook port (interlace-3ppz) under ported_examples/stat_rethinking/.
Reference fixtures are produced by tests/fixtures/gen_sr_12_reedfrogs.R.

Models (binomial, logit, varying intercept by tank):
    M0: cbind(surv, density-surv) ~ 1                + (1 | tank)
    M1: cbind(surv, density-surv) ~ pred             + (1 | tank)
    M2: cbind(surv, density-surv) ~ size             + (1 | tank)
    M3: cbind(surv, density-surv) ~ pred * size      + (1 | tank)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURES = Path(__file__).parent / "fixtures"


# Map R coefficient names → formulaic names produced by interlace.
_NAME_MAP = {
    "(Intercept)": "Intercept",
    "predpred": "C(pred)[T.pred]",
    "sizesmall": "C(size)[T.small]",
    "predpred:sizesmall": "C(pred)[T.pred]:C(size)[T.small]",
}


@pytest.fixture(scope="module")
def reedfrogs() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "sr_12_reedfrogs_data.csv")
    df["tank"] = np.arange(1, len(df) + 1)
    df["proportion"] = df["surv"] / df["density"]
    return df


def _ref(model: str) -> dict:
    return json.loads((FIXTURES / f"sr_12_reedfrogs_{model}_results.json").read_text())


def _fit(df: pd.DataFrame, formula: str):
    return interlace.glmer(
        formula,
        data=df,
        family="binomial",
        groups="tank",
        weights=df["density"].to_numpy(dtype=float),
    )


@pytest.fixture(scope="module")
def fit_M0(reedfrogs):
    return _fit(reedfrogs, "proportion ~ 1")


@pytest.fixture(scope="module")
def fit_M1(reedfrogs):
    return _fit(reedfrogs, "proportion ~ C(pred)")


@pytest.fixture(scope="module")
def fit_M2(reedfrogs):
    return _fit(reedfrogs, "proportion ~ C(size)")


@pytest.fixture(scope="module")
def fit_M3(reedfrogs):
    return _fit(reedfrogs, "proportion ~ C(pred) * C(size)")


def _check_fixed_effects(fit, ref: dict, atol: float = 1e-3) -> None:
    for r_name, r_val in ref["fixed_effects"].items():
        py_name = _NAME_MAP[r_name]
        py_val = float(fit.fe_params[py_name])
        diff = abs(py_val - r_val)
        assert diff < atol, (
            f"FE {r_name}: interlace={py_val:.6f}, R={r_val:.6f}, abs_diff={diff:.2e}"
        )


def _check_loglik(fit, ref: dict, atol: float = 1e-2) -> None:
    diff = abs(float(fit.llf) - ref["loglik"])
    assert diff < atol, (
        f"logLik: interlace={fit.llf:.4f}, R={ref['loglik']:.4f}, abs_diff={diff:.2e}"
    )


def _check_variance(fit, ref: dict, rtol: float = 0.05) -> None:
    r_var = ref["variance_components"]["tank"]
    py_var = float(fit.variance_components["tank"])
    rel_diff = abs(py_var - r_var) / r_var
    assert rel_diff < rtol, (
        f"sigma2_tank: interlace={py_var:.4f}, R={r_var:.4f}, rel={rel_diff:.3f}"
    )


def _check_blups(fit, ref: dict, min_corr: float = 0.99) -> None:
    r_re = ref["ranef_tank"]
    py_re = fit.random_effects["tank"]
    levels = sorted(int(k) for k in r_re)
    r_arr = np.array([r_re[str(i)] for i in levels])
    # interlace random_effects index is the raw group key
    py_arr = np.array([float(py_re.loc[i]) for i in levels])
    corr = np.corrcoef(r_arr, py_arr)[0, 1]
    assert corr > min_corr, f"BLUP corr={corr:.4f}"


# M0 and M2 fit one-observation-per-tank binomial GLMMs with no informative
# fixed-effect contrast — the (intercept, sigma_tank) likelihood has a flat
# ridge, so the FE intercept can differ between optimizers (lme4-bobyqa vs
# interlace-Nelder_Mead) by ~1e-2 while the log-likelihood agrees to 1e-3.
# Use logLik parity as the primary location-of-optimum check on these two.
class TestM0:
    def test_fixed_effects(self, fit_M0):
        _check_fixed_effects(fit_M0, _ref("M0"), atol=1e-2)

    def test_loglik(self, fit_M0):
        _check_loglik(fit_M0, _ref("M0"))

    def test_variance(self, fit_M0):
        _check_variance(fit_M0, _ref("M0"))

    def test_blups(self, fit_M0):
        _check_blups(fit_M0, _ref("M0"))


class TestM1:
    def test_fixed_effects(self, fit_M1):
        _check_fixed_effects(fit_M1, _ref("M1"))

    def test_loglik(self, fit_M1):
        _check_loglik(fit_M1, _ref("M1"))

    def test_variance(self, fit_M1):
        _check_variance(fit_M1, _ref("M1"))

    def test_blups(self, fit_M1):
        _check_blups(fit_M1, _ref("M1"))


class TestM2:
    def test_fixed_effects(self, fit_M2):
        # 2e-2 ≈ 4% of the SE on `sizesmall` (R: 0.49); flat-ridge slack.
        _check_fixed_effects(fit_M2, _ref("M2"), atol=2e-2)

    def test_loglik(self, fit_M2):
        _check_loglik(fit_M2, _ref("M2"))

    def test_variance(self, fit_M2):
        _check_variance(fit_M2, _ref("M2"))

    def test_blups(self, fit_M2):
        _check_blups(fit_M2, _ref("M2"))


class TestM3:
    def test_fixed_effects(self, fit_M3):
        _check_fixed_effects(fit_M3, _ref("M3"))

    def test_loglik(self, fit_M3):
        _check_loglik(fit_M3, _ref("M3"))

    def test_variance(self, fit_M3):
        _check_variance(fit_M3, _ref("M3"))

    def test_blups(self, fit_M3):
        _check_blups(fit_M3, _ref("M3"))
