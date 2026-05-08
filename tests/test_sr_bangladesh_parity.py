"""Parity test: interlace.glmer vs lme4::glmer on McElreath bangladesh.

Drives the SR-13 notebook port (interlace-deza) under
ported_examples/stat_rethinking/. Real-data binomial GLMM with single
varying-intercept on district; sets up T5 (varying slopes).

Reference fixtures from tests/fixtures/gen_sr_13_bangladesh.R.

Models (Bernoulli, logit, varying intercept by district):
    M0: use_contraception ~ 1                                       + (1|district)
    M1: use_contraception ~ urban                                   + (1|district)
    M2: use_contraception ~ urban + age_centered + living_children  + (1|district)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURES = Path(__file__).parent / "fixtures"


_NAME_MAP = {
    "(Intercept)": "Intercept",
    "urban": "urban",
    "age_centered": "age_centered",
    "living_children": "living_children",
}


@pytest.fixture(scope="module")
def bangladesh() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sr_13_bangladesh_data.csv")


def _ref(model: str) -> dict:
    return json.loads((FIXTURES / f"sr_13_bangladesh_{model}_results.json").read_text())


def _glmer(df: pd.DataFrame, formula: str):
    return interlace.glmer(formula, data=df, family="binomial", groups="district")


@pytest.fixture(scope="module")
def fit_M0(bangladesh):
    return _glmer(bangladesh, "use_contraception ~ 1")


@pytest.fixture(scope="module")
def fit_M1(bangladesh):
    return _glmer(bangladesh, "use_contraception ~ urban")


@pytest.fixture(scope="module")
def fit_M2(bangladesh):
    return _glmer(
        bangladesh,
        "use_contraception ~ urban + age_centered + living_children",
    )


def _check_fixed_effects(fit, ref: dict, atol: float = 1e-3) -> None:
    for r_name, r_val in ref["fixed_effects"].items():
        py_val = float(fit.fe_params[_NAME_MAP[r_name]])
        diff = abs(py_val - r_val)
        assert diff < atol, (
            f"FE {r_name}: interlace={py_val:.6f}, R={r_val:.6f}, abs_diff={diff:.2e}"
        )


def _check_loglik(fit, ref: dict, atol: float = 1e-3) -> None:
    diff = abs(float(fit.llf) - ref["loglik"])
    assert diff < atol, (
        f"logLik: interlace={fit.llf:.4f}, R={ref['loglik']:.4f}, abs_diff={diff:.2e}"
    )


def _check_variance(fit, ref: dict, rtol: float = 0.05) -> None:
    r_var = ref["variance_components"]["district"]
    py_var = float(fit.variance_components["district"])
    rel_diff = abs(py_var - r_var) / r_var
    assert rel_diff < rtol, (
        f"sigma2_district: interlace={py_var:.4f}, R={r_var:.4f}, rel={rel_diff:.3f}"
    )


def _check_blups(fit, ref: dict, min_corr: float = 0.99) -> None:
    r_re = ref["ranef_district"]
    py_re = fit.random_effects["district"]
    levels = sorted(int(k) for k in r_re)
    r_arr = np.array([r_re[str(i)] for i in levels])
    py_arr = np.array([float(py_re.loc[i]) for i in levels])
    corr = np.corrcoef(r_arr, py_arr)[0, 1]
    assert corr > min_corr, f"BLUP corr={corr:.4f}"


class TestM0:
    def test_fixed_effects(self, fit_M0):
        _check_fixed_effects(fit_M0, _ref("M0"))

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
        _check_fixed_effects(fit_M2, _ref("M2"))

    def test_loglik(self, fit_M2):
        _check_loglik(fit_M2, _ref("M2"))

    def test_variance(self, fit_M2):
        _check_variance(fit_M2, _ref("M2"))

    def test_blups(self, fit_M2):
        _check_blups(fit_M2, _ref("M2"))
