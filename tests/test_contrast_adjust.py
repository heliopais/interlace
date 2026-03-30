"""Tests for p-value adjustment in contrast() and the pairs() convenience function.

TDD: written before implementation.

Acceptance criteria for contrast(adjust=):
  - adjust='none' (default) → same p-values as before
  - adjust='bonferroni' → p-values >= unadjusted; min(1, p*n)
  - adjust='holm' → p-values >= unadjusted (step-down Bonferroni)
  - adjust='fdr' → p-values >= unadjusted (Benjamini-Hochberg)
  - adjust='tukey' → p-values >= unadjusted for pairwise contrasts
  - ValueError for unknown adjust= value
  - adjusted p-values are in [0, 1]
  - ordering of rows is unchanged by adjustment

Acceptance criteria for pairs():
  - pairs(emm) is equivalent to contrast(emm, method='pairwise', adjust='tukey')
  - pairs(emm, adjust='bonferroni') uses bonferroni instead
  - Returns DataFrame with same columns as contrast()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace.emmeans import contrast, emmeans, pairs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_way_result():
    """y ~ trt + (1|group); trt has 4 levels with known effects."""
    rng = np.random.default_rng(7)
    n_groups = 20
    n_per_cell = 8
    n_trt = 4
    n = n_groups * n_per_cell * n_trt

    group_ids = np.repeat(np.arange(n_groups), n_per_cell * n_trt)
    trt = np.tile(np.repeat(["A", "B", "C", "D"], n_per_cell), n_groups)
    u = rng.normal(0, 1.0, n_groups)
    trt_effects = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
    y = (
        5.0
        + np.array([trt_effects[t] for t in trt])
        + u[group_ids]
        + rng.normal(0, 0.5, n)
    )
    df = pd.DataFrame({"y": y, "trt": trt, "group": group_ids.astype(str)})
    return interlace.fit("y ~ trt", data=df, groups="group")


@pytest.fixture(scope="module")
def emm(one_way_result):
    return emmeans(one_way_result, "trt")


@pytest.fixture(scope="module")
def unadj(emm):
    return contrast(emm, method="pairwise", adjust="none")


# ---------------------------------------------------------------------------
# adjust='none': behaviour unchanged
# ---------------------------------------------------------------------------


def test_adjust_none_same_as_default(emm):
    tbl_none = contrast(emm, method="pairwise", adjust="none")
    tbl_default = contrast(emm, method="pairwise")
    pd.testing.assert_frame_equal(
        tbl_none.reset_index(drop=True), tbl_default.reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# adjust='bonferroni'
# ---------------------------------------------------------------------------


def test_bonferroni_pvalues_geq_unadjusted(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="bonferroni")
    assert (adj["p.value"].values >= unadj["p.value"].values - 1e-12).all()


def test_bonferroni_pvalues_capped_at_1(emm):
    adj = contrast(emm, method="pairwise", adjust="bonferroni")
    assert (adj["p.value"] <= 1.0 + 1e-12).all()


def test_bonferroni_pvalues_in_range(emm):
    adj = contrast(emm, method="pairwise", adjust="bonferroni")
    assert ((adj["p.value"] >= 0) & (adj["p.value"] <= 1)).all()


def test_bonferroni_row_order_unchanged(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="bonferroni")
    assert list(adj["contrast"]) == list(unadj["contrast"])


def test_bonferroni_estimates_unchanged(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="bonferroni")
    np.testing.assert_allclose(adj["estimate"].values, unadj["estimate"].values)


# ---------------------------------------------------------------------------
# adjust='holm'
# ---------------------------------------------------------------------------


def test_holm_pvalues_geq_unadjusted(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="holm")
    assert (adj["p.value"].values >= unadj["p.value"].values - 1e-12).all()


def test_holm_pvalues_in_range(emm):
    adj = contrast(emm, method="pairwise", adjust="holm")
    assert ((adj["p.value"] >= 0) & (adj["p.value"] <= 1)).all()


def test_holm_row_order_unchanged(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="holm")
    assert list(adj["contrast"]) == list(unadj["contrast"])


def test_holm_leq_bonferroni(emm):
    """Holm is uniformly more powerful than Bonferroni: holm p <= bonferroni p."""
    holm = contrast(emm, method="pairwise", adjust="holm")
    bonf = contrast(emm, method="pairwise", adjust="bonferroni")
    assert (holm["p.value"].values <= bonf["p.value"].values + 1e-12).all()


# ---------------------------------------------------------------------------
# adjust='fdr'
# ---------------------------------------------------------------------------


def test_fdr_pvalues_geq_unadjusted(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="fdr")
    assert (adj["p.value"].values >= unadj["p.value"].values - 1e-12).all()


def test_fdr_pvalues_in_range(emm):
    adj = contrast(emm, method="pairwise", adjust="fdr")
    assert ((adj["p.value"] >= 0) & (adj["p.value"] <= 1)).all()


def test_fdr_row_order_unchanged(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="fdr")
    assert list(adj["contrast"]) == list(unadj["contrast"])


# ---------------------------------------------------------------------------
# adjust='tukey'
# ---------------------------------------------------------------------------


def test_tukey_pvalues_geq_unadjusted(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="tukey")
    assert (adj["p.value"].values >= unadj["p.value"].values - 1e-12).all()


def test_tukey_pvalues_in_range(emm):
    adj = contrast(emm, method="pairwise", adjust="tukey")
    assert ((adj["p.value"] >= 0) & (adj["p.value"] <= 1)).all()


def test_tukey_row_order_unchanged(emm, unadj):
    adj = contrast(emm, method="pairwise", adjust="tukey")
    assert list(adj["contrast"]) == list(unadj["contrast"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_adjust_raises(emm):
    with pytest.raises(ValueError, match="adjust"):
        contrast(emm, method="pairwise", adjust="sidak")


# ---------------------------------------------------------------------------
# pairs() convenience function
# ---------------------------------------------------------------------------


def test_pairs_returns_dataframe(emm):
    tbl = pairs(emm)
    assert isinstance(tbl, pd.DataFrame)


def test_pairs_columns(emm):
    tbl = pairs(emm)
    for col in ["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]:
        assert col in tbl.columns


def test_pairs_row_count(emm):
    """4 levels -> C(4,2) = 6 pairwise contrasts."""
    tbl = pairs(emm)
    assert len(tbl) == 6


def test_pairs_default_is_tukey(emm):
    """pairs() default adjust='tukey' must match contrast(method='pairwise', adjust='tukey')."""  # noqa: E501
    expected = contrast(emm, method="pairwise", adjust="tukey")
    actual = pairs(emm)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_pairs_adjust_bonferroni(emm):
    expected = contrast(emm, method="pairwise", adjust="bonferroni")
    actual = pairs(emm, adjust="bonferroni")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API: pairs exported from interlace top-level
# ---------------------------------------------------------------------------


def test_pairs_exported_from_interlace():
    assert hasattr(interlace, "pairs")


def test_pairs_callable_via_interlace(emm):
    tbl = interlace.pairs(emm)
    assert isinstance(tbl, pd.DataFrame)
