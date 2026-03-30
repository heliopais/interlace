"""Tests for contrast() — linear contrasts on estimated marginal means.

TDD: written before implementation.

Acceptance criteria:
  - emmeans() still returns a DataFrame-like object (backwards compatible)
  - contrast(emm, method='pairwise') returns DataFrame with columns:
    ["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]
  - contrast(emm, method='trt.vs.ctrl') compares each level to the first
  - contrast(emm, method=custom_list) applies user-supplied contrast vectors
  - contrast(emm, method=custom_dict) uses dict keys as contrast names
  - Number of rows: pairwise -> n*(n-1)//2, trt.vs.ctrl -> n-1
  - SE > 0, df > 0, p.value in [0, 1]
  - Estimates for known differences are close to true values
  - t.ratio = estimate / SE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace.emmeans import contrast, emmeans

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_way_result():
    """y ~ trt + x + (1|group); trt has 3 levels A/B/C with effects 0/2/4."""
    rng = np.random.default_rng(42)
    n_groups = 20
    n_per_cell = 10
    n_trt = 3
    n = n_groups * n_per_cell * n_trt

    group_ids = np.repeat(np.arange(n_groups), n_per_cell * n_trt)
    trt = np.tile(np.repeat(["A", "B", "C"], n_per_cell), n_groups)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.0, n_groups)
    trt_effects = {"A": 0.0, "B": 2.0, "C": 4.0}
    y = (
        10.0
        + np.array([trt_effects[t] for t in trt])
        + 0.5 * x
        + u[group_ids]
        + rng.normal(0, 0.3, n)
    )
    df = pd.DataFrame({"y": y, "trt": trt, "x": x, "group": group_ids.astype(str)})
    return interlace.fit("y ~ trt + x", data=df, groups="group")


@pytest.fixture(scope="module")
def emm(one_way_result):
    return emmeans(one_way_result, "trt")


# ---------------------------------------------------------------------------
# Backwards compatibility: emmeans() still looks like a DataFrame
# ---------------------------------------------------------------------------


def test_emmeans_backwards_compat_dataframe(emm):
    """emmeans() result must still be usable as a DataFrame."""
    assert hasattr(emm, "columns")
    assert "estimate" in emm.columns
    assert "SE" in emm.columns
    assert len(emm) == 3  # A, B, C


# ---------------------------------------------------------------------------
# contrast(): structure
# ---------------------------------------------------------------------------


def test_contrast_pairwise_returns_dataframe(emm):
    tbl = contrast(emm, method="pairwise")
    assert isinstance(tbl, pd.DataFrame)


def test_contrast_pairwise_columns(emm):
    tbl = contrast(emm, method="pairwise")
    for col in ["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]:
        assert col in tbl.columns, f"Missing column: {col!r}"


def test_contrast_pairwise_row_count(emm):
    """3 levels -> 3 pairwise contrasts."""
    tbl = contrast(emm, method="pairwise")
    assert len(tbl) == 3  # C(3,2) = 3


def test_contrast_pairwise_se_positive(emm):
    tbl = contrast(emm, method="pairwise")
    assert (tbl["SE"] > 0).all()


def test_contrast_pairwise_df_positive(emm):
    tbl = contrast(emm, method="pairwise")
    assert (tbl["df"] > 0).all()


def test_contrast_pairwise_pvalue_in_range(emm):
    tbl = contrast(emm, method="pairwise")
    assert ((tbl["p.value"] >= 0) & (tbl["p.value"] <= 1)).all()


def test_contrast_pairwise_t_ratio_equals_estimate_over_se(emm):
    tbl = contrast(emm, method="pairwise")
    expected = tbl["estimate"] / tbl["SE"]
    np.testing.assert_allclose(tbl["t.ratio"].values, expected.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# contrast(): pairwise accuracy
# ---------------------------------------------------------------------------


def test_contrast_pairwise_b_minus_a_close_to_2(emm):
    """B - A true effect is 2.0; estimate should be within 0.2."""
    tbl = contrast(emm, method="pairwise")
    # Find the B - A row
    ba = tbl[tbl["contrast"].str.contains("B") & tbl["contrast"].str.contains("A")]
    assert len(ba) == 1
    est = float(ba["estimate"].iloc[0])
    # B - A or A - B; take abs
    assert abs(abs(est) - 2.0) < 0.2, f"B-A estimate={est:.4f}"


def test_contrast_pairwise_c_minus_a_close_to_4(emm):
    """C - A true effect is 4.0; estimate should be within 0.2."""
    tbl = contrast(emm, method="pairwise")
    ca = tbl[tbl["contrast"].str.contains("C") & tbl["contrast"].str.contains("A")]
    assert len(ca) == 1
    est = float(ca["estimate"].iloc[0])
    assert abs(abs(est) - 4.0) < 0.2, f"C-A estimate={est:.4f}"


# ---------------------------------------------------------------------------
# contrast(): trt.vs.ctrl
# ---------------------------------------------------------------------------


def test_contrast_trt_vs_ctrl_row_count(emm):
    """3 levels -> 2 treatment-vs-control contrasts."""
    tbl = contrast(emm, method="trt.vs.ctrl")
    assert len(tbl) == 2


def test_contrast_trt_vs_ctrl_columns(emm):
    tbl = contrast(emm, method="trt.vs.ctrl")
    for col in ["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]:
        assert col in tbl.columns


def test_contrast_trt_vs_ctrl_ctrl_is_first_level(emm):
    """Control is the first (alphabetically sorted) level: A."""
    tbl = contrast(emm, method="trt.vs.ctrl")
    # Both contrast names should reference A
    assert tbl["contrast"].str.contains("A").all()


def test_contrast_trt_vs_ctrl_b_minus_a_close_to_2(emm):
    tbl = contrast(emm, method="trt.vs.ctrl")
    ba = tbl[tbl["contrast"].str.contains("B")]
    assert len(ba) == 1
    est = float(ba["estimate"].iloc[0])
    assert abs(est - 2.0) < 0.2, f"B-A estimate={est:.4f}"


# ---------------------------------------------------------------------------
# contrast(): custom list
# ---------------------------------------------------------------------------


def test_contrast_custom_list(emm):
    """Custom list of contrast vectors (one per contrast)."""
    # 3 levels; contrast [1, -1, 0] = A - B
    c1 = np.array([1.0, -1.0, 0.0])
    tbl = contrast(emm, method=[c1])
    assert isinstance(tbl, pd.DataFrame)
    assert len(tbl) == 1
    for col in ["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]:
        assert col in tbl.columns


def test_contrast_custom_list_estimate(emm):
    """Custom contrast [1, -1, 0] should give A - B ≈ -2."""
    c1 = np.array([1.0, -1.0, 0.0])
    tbl = contrast(emm, method=[c1])
    est = float(tbl["estimate"].iloc[0])
    assert abs(est - (-2.0)) < 0.2, f"A-B estimate={est:.4f}"


def test_contrast_custom_dict(emm):
    """Custom dict maps names to contrast vectors."""
    c_dict = {"A_minus_B": np.array([1.0, -1.0, 0.0])}
    tbl = contrast(emm, method=c_dict)
    assert isinstance(tbl, pd.DataFrame)
    assert tbl["contrast"].iloc[0] == "A_minus_B"


# ---------------------------------------------------------------------------
# contrast(): error handling
# ---------------------------------------------------------------------------


def test_contrast_invalid_method_raises(emm):
    with pytest.raises(ValueError, match="method"):
        contrast(emm, method="invalid")


# ---------------------------------------------------------------------------
# Public API: contrast exported from interlace top-level
# ---------------------------------------------------------------------------


def test_contrast_exported_from_interlace():
    assert hasattr(interlace, "contrast")


def test_contrast_callable_via_interlace(emm):
    tbl = interlace.contrast(emm, method="pairwise")
    assert isinstance(tbl, pd.DataFrame)
