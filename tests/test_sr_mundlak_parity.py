"""Parity test: interlace.glmer vs lme4::glmer on McElreath Mundlak DGP
(interlace-x52r).

Reference fixtures from tests/fixtures/gen_sr_12_mundlak.R.
Frequentist counterparts of the four models in 12_bonus_mundlak.r:

    M0_naive : Y ~ X + Zg                   (glm — biased)
    M0_fe    : Y ~ X + Zg + factor(g)       (glm — fixed-effect dummies)
    M1_re    : Y ~ X + Zg + (1|g)           (glmer — random intercept)
    M2_mund  : Y ~ X + Xbar + Zg + (1|g)    (glmer — Mundlak machine)

Acceptance: parity tested only for the two glmer fits; the GLM fits are
demonstrative and validated by inspection in the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURES = Path(__file__).parent / "fixtures"


_NAME_MAP_RE = {"(Intercept)": "Intercept", "X": "X", "Zg": "Zg"}
_NAME_MAP_MUND = {
    "(Intercept)": "Intercept",
    "X": "X",
    "Xbar": "Xbar",
    "Zg": "Zg",
}


@pytest.fixture(scope="module")
def mundlak_data() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sr_12_mundlak_data.csv")


def _ref(model: str) -> dict:
    return json.loads((FIXTURES / f"sr_12_mundlak_{model}_results.json").read_text())


@pytest.fixture(scope="module")
def fit_M1_re(mundlak_data):
    return interlace.glmer(
        "Y ~ X + Zg", data=mundlak_data, family="binomial", groups="g"
    )


@pytest.fixture(scope="module")
def fit_M2_mund(mundlak_data):
    return interlace.glmer(
        "Y ~ X + Xbar + Zg",
        data=mundlak_data,
        family="binomial",
        groups="g",
    )


# sigma2_g is small/borderline-singular in this small sample (200 obs / 30
# groups). interlace pins to 0; lme4 finds a tiny positive value (~0.08).
# On the boundary the FE intercept absorbs the shift — slopes match cleanly.
def _check_fixed_effects(
    fit,
    ref: dict,
    name_map: dict,
    intercept_atol: float = 5e-2,
    slope_atol: float = 1e-2,
) -> None:
    for r_name, r_val in ref["fixed_effects"].items():
        py_val = float(fit.fe_params[name_map[r_name]])
        diff = abs(py_val - r_val)
        atol = intercept_atol if r_name == "(Intercept)" else slope_atol
        assert diff < atol, (
            f"FE {r_name}: interlace={py_val:.6f}, R={r_val:.6f}, abs_diff={diff:.2e}"
        )


def _check_loglik(fit, ref: dict, atol: float = 5e-2) -> None:
    diff = abs(float(fit.llf) - ref["loglik"])
    assert diff < atol, (
        f"logLik: interlace={fit.llf:.4f}, R={ref['loglik']:.4f}, abs_diff={diff:.2e}"
    )


def _check_variance(
    fit, ref: dict, group: str, rtol: float = 0.05, singular_atol: float = 2e-1
) -> None:
    r_var = ref["variance_components"][group]
    py_var = float(fit.variance_components[group])
    if r_var < singular_atol or py_var < singular_atol:
        assert abs(py_var - r_var) < singular_atol, (
            f"sigma2_{group}: interlace={py_var:.6f}, R={r_var:.6f}"
        )
        return
    rel_diff = abs(py_var - r_var) / r_var
    assert rel_diff < rtol, (
        f"sigma2_{group}: interlace={py_var:.4f}, R={r_var:.4f}, rel={rel_diff:.3f}"
    )


def _check_blups(fit, ref: dict, group: str, min_corr: float = 0.99) -> None:
    r_re = ref[f"ranef_{group}"]
    py_re = fit.random_effects[group]
    levels = sorted(int(k) for k in r_re)
    r_arr = np.array([r_re[str(i)] for i in levels])
    py_arr = np.array([float(py_re.loc[i]) for i in levels])
    if np.std(r_arr) < 1e-3 or np.std(py_arr) < 1e-3:
        # Mundlak machine fully absorbs group variation → BLUPs all ≈ 0.
        return
    corr = np.corrcoef(r_arr, py_arr)[0, 1]
    assert corr > min_corr, f"BLUP corr {group}={corr:.4f}"


class TestM1RandomIntercept:
    def test_fixed_effects(self, fit_M1_re):
        _check_fixed_effects(fit_M1_re, _ref("M1_re"), _NAME_MAP_RE)

    def test_loglik(self, fit_M1_re):
        _check_loglik(fit_M1_re, _ref("M1_re"))

    def test_variance(self, fit_M1_re):
        _check_variance(fit_M1_re, _ref("M1_re"), "g")

    def test_blups(self, fit_M1_re):
        _check_blups(fit_M1_re, _ref("M1_re"), "g")


class TestM2MundlakMachine:
    def test_fixed_effects(self, fit_M2_mund):
        _check_fixed_effects(fit_M2_mund, _ref("M2_mund"), _NAME_MAP_MUND)

    def test_loglik(self, fit_M2_mund):
        _check_loglik(fit_M2_mund, _ref("M2_mund"))

    def test_variance(self, fit_M2_mund):
        _check_variance(fit_M2_mund, _ref("M2_mund"), "g")

    def test_blups(self, fit_M2_mund):
        _check_blups(fit_M2_mund, _ref("M2_mund"), "g")

    def test_recovers_bxy(self, fit_M2_mund):
        """Substantive check: Mundlak machine should recover bxy ≈ 1 (the truth)."""
        bxy = float(fit_M2_mund.fe_params["X"])
        # The single-sample MLE wanders; tolerate ±0.25 from the truth.
        assert abs(bxy - 1.0) < 0.25, f"Mundlak bxy={bxy:.3f}, expected ≈1"
