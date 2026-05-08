"""Parity test: interlace.glmer vs lme4::glmer on McElreath chimpanzees.

Drives the SR-13 notebook port (interlace-61qf) under
ported_examples/stat_rethinking/. Crossed-RE binomial GLMM stresses
interlace's headline differentiator vs ``statsmodels.MixedLM``.

Reference fixtures from tests/fixtures/gen_sr_13_chimps.R.

Models (Bernoulli, logit, two crossed varying intercepts):
    M0: pulled_left ~ 1                  + (1|actor) + (1|block)
    M1: pulled_left ~ as.factor(tx)      + (1|actor) + (1|block)
    M2: pulled_left ~ prosoc_left * cond + (1|actor) + (1|block)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import interlace

FIXTURES = Path(__file__).parent / "fixtures"


_M1_NAME_MAP = {
    "(Intercept)": "Intercept",
    "tx2": "C(tx)[T.2]",
    "tx3": "C(tx)[T.3]",
    "tx4": "C(tx)[T.4]",
}

_M2_NAME_MAP = {
    "(Intercept)": "Intercept",
    "prosoc_left": "prosoc_left",
    "condition": "condition",
    "prosoc_left:condition": "prosoc_left:condition",
}


@pytest.fixture(scope="module")
def chimps() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "sr_13_chimpanzees_data.csv")
    df["tx"] = 1 + df["prosoc_left"] + 2 * df["condition"]
    return df


def _ref(model: str) -> dict:
    return json.loads((FIXTURES / f"sr_13_chimps_{model}_results.json").read_text())


def _glmer(df: pd.DataFrame, formula: str):
    return interlace.glmer(
        formula,
        data=df,
        family="binomial",
        groups=["actor", "block"],
    )


@pytest.fixture(scope="module")
def fit_M0(chimps):
    return _glmer(chimps, "pulled_left ~ 1")


@pytest.fixture(scope="module")
def fit_M1(chimps):
    return _glmer(chimps, "pulled_left ~ C(tx)")


@pytest.fixture(scope="module")
def fit_M2(chimps):
    return _glmer(chimps, "pulled_left ~ prosoc_left * condition")


# Tolerances reflect the chimps "boundary regime": block VC is singular in
# interlace (0.0) and near-singular in lme4 (~0.006). On the boundary the
# (FE intercept, RE block means) become exchangeable up to a shift, so the FE
# intercept absorbs the small block-VC discrepancy (~0.05, ~6% of the
# intercept SE 0.75); slopes are unaffected.
def _check_fixed_effects(
    fit,
    ref: dict,
    name_map: dict,
    intercept_atol: float = 5e-2,
    slope_atol: float = 1e-2,
) -> None:
    for r_name, r_val in ref["fixed_effects"].items():
        py_name = name_map[r_name]
        py_val = float(fit.fe_params[py_name])
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
    fit, ref: dict, group: str, rtol: float = 0.05, singular_atol: float = 2e-2
) -> None:
    r_var = ref["variance_components"][group]
    py_var = float(fit.variance_components[group])
    if r_var < singular_atol or py_var < singular_atol:
        # Near-singular component: both sit near the boundary; relative diff
        # is undefined. Assert both within singular_atol of zero/each other.
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
    # When the variance is near-zero (block in M0/M1/M2), all BLUPs sit near 0;
    # the correlation is dominated by floating-point noise. Skip the assertion.
    if np.std(r_arr) < 1e-3:
        return
    corr = np.corrcoef(r_arr, py_arr)[0, 1]
    assert corr > min_corr, f"BLUP corr {group}={corr:.4f}"


class TestM0:
    def test_fixed_effects(self, fit_M0):
        _check_fixed_effects(fit_M0, _ref("M0"), {"(Intercept)": "Intercept"})

    def test_loglik(self, fit_M0):
        _check_loglik(fit_M0, _ref("M0"))

    def test_variance_actor(self, fit_M0):
        _check_variance(fit_M0, _ref("M0"), "actor")

    def test_variance_block(self, fit_M0):
        _check_variance(fit_M0, _ref("M0"), "block")

    def test_blups_actor(self, fit_M0):
        _check_blups(fit_M0, _ref("M0"), "actor")

    def test_blups_block(self, fit_M0):
        _check_blups(fit_M0, _ref("M0"), "block")


class TestM1:
    def test_fixed_effects(self, fit_M1):
        _check_fixed_effects(fit_M1, _ref("M1"), _M1_NAME_MAP)

    def test_loglik(self, fit_M1):
        _check_loglik(fit_M1, _ref("M1"))

    def test_variance_actor(self, fit_M1):
        _check_variance(fit_M1, _ref("M1"), "actor")

    def test_variance_block(self, fit_M1):
        _check_variance(fit_M1, _ref("M1"), "block")

    def test_blups_actor(self, fit_M1):
        _check_blups(fit_M1, _ref("M1"), "actor")


class TestM2:
    def test_fixed_effects(self, fit_M2):
        _check_fixed_effects(fit_M2, _ref("M2"), _M2_NAME_MAP)

    def test_loglik(self, fit_M2):
        _check_loglik(fit_M2, _ref("M2"))

    def test_variance_actor(self, fit_M2):
        _check_variance(fit_M2, _ref("M2"), "actor")

    def test_variance_block(self, fit_M2):
        _check_variance(fit_M2, _ref("M2"), "block")

    def test_blups_actor(self, fit_M2):
        _check_blups(fit_M2, _ref("M2"), "actor")
