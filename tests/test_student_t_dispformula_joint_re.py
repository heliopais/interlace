"""Regression tests for interlace-cgb7 — random_effects label fidelity
and predict() correctness on the Student-t × dispformula joint Laplace
path (and incidentally on nested LMM predict, which shares the same
mechanism)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_nested_data(
    n_main: int = 4,
    n_sub_per_main: int = 3,
    n_typ_per_sub: int = 3,
    n_obs_per_typ: int = 15,
    nu_true: float = 4.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for mi in range(n_main):
        m = f"m{mi:02d}"
        main_re = rng.normal(0, 0.3)
        for si in range(n_sub_per_main):
            s = f"{m}_s{si}"
            sub_re = main_re + rng.normal(0, 0.2)
            for ti in range(n_typ_per_sub):
                t = f"{s}_t{ti}"
                typ_re = sub_re + rng.normal(0, 0.15)
                for _ in range(n_obs_per_typ):
                    rl = float(rng.integers(1, 5))
                    y = (
                        0.05
                        + 0.1 * (rl - 2.5)
                        + typ_re
                        + rng.standard_t(nu_true) * 0.15
                    )
                    rows.append(
                        {
                            "main": m,
                            "sub": s,
                            "typ": t,
                            "rl": rl,
                            "rl_cat": str(int(rl)),
                            "y": y,
                        }
                    )
    df = pd.DataFrame(rows)
    df["rl_std"] = (df["rl"] - df["rl"].mean()) / df["rl"].std()
    return df


@pytest.fixture(scope="module")
def fit_joint_t_disp():
    """Cached fit for the joint Student-t × dispformula path."""
    import interlace

    df = _make_nested_data()
    res = interlace.fit(
        formula="y ~ rl_std",
        random=["(1|main/sub/typ)"],
        dispformula="~ (1|rl_cat)",
        data=df,
        family="student_t",
        method="REML",
    )
    return df, res


def test_random_effects_indices_are_string_labels(fit_joint_t_disp) -> None:
    """random_effects['main'] should be indexed by the actual group labels,
    e.g. 'm00', 'm01', not by 0, 1, 2…"""
    df, res = fit_joint_t_disp
    idx_main = list(res.random_effects["main"].index)
    assert "m00" in idx_main, f"main RE index lost labels: {idx_main}"
    assert set(idx_main) == set(df["main"].unique())


def test_random_effects_indices_for_nested_specs(fit_joint_t_disp) -> None:
    """Nested levels ('main:sub', 'main:sub:typ') should also use labels."""
    df, res = fit_joint_t_disp
    # 'main:sub' nested level
    assert "main:sub" in res.random_effects
    idx_sub = list(res.random_effects["main:sub"].index)
    expected_sub = sorted(df["main"].astype(str) + ":" + df["sub"].astype(str))
    expected_sub_unique = sorted(set(expected_sub))
    assert set(idx_sub) == set(expected_sub_unique)
    # Deepest level
    assert "main:sub:typ" in res.random_effects


def test_predict_in_sample_matches_fittedvalues(fit_joint_t_disp) -> None:
    """predict(df_train) must equal result.fittedvalues."""
    df, res = fit_joint_t_disp
    pred = res.predict(df)
    np.testing.assert_allclose(pred, res.fittedvalues, atol=1e-6)


def test_predict_distinguishes_group_leaves(fit_joint_t_disp) -> None:
    """A grid of rows with identical fixed effects and distinct group
    leaves must produce distinct predictions."""
    df, res = fit_joint_t_disp
    df_new = df.drop_duplicates("typ").head(5).copy()
    df_new["rl_std"] = 0.0
    pred = res.predict(df_new)
    assert len(set(pred.round(8))) >= 4, (
        f"predict returned near-identical values: {pred}"
    )


def test_predict_unseen_levels_shrink_to_zero(fit_joint_t_disp) -> None:
    """Unseen leaf levels should contribute 0 (population shrinkage),
    so the prediction at unseen-leaf rows equals the fixed-effect part
    only when no ancestor level matches either."""
    df, res = fit_joint_t_disp
    df_new = pd.DataFrame(
        {
            "main": ["UNSEEN"],
            "sub": ["UNSEEN_SUB"],
            "typ": ["UNSEEN_TYP"],
            "rl_cat": ["3"],
            "rl_std": [0.0],
        }
    )
    pred = res.predict(df_new)[0]
    fe_only = float(res.fe_params["Intercept"])
    # rl_std=0 contributes nothing; no group level seen → FE only
    assert abs(pred - fe_only) < 1e-8


# ---------------------------------------------------------------------------
# Incidental: plain LMM nested predict should also be consistent.
# ---------------------------------------------------------------------------


def test_plain_lmm_nested_predict_matches_fittedvalues() -> None:
    """The same predict() fix should also make plain LMM with nested
    RE specs return predictions consistent with fittedvalues in-sample."""
    import interlace

    df = _make_nested_data(n_obs_per_typ=10, nu_true=200.0, seed=42)
    res = interlace.fit(
        formula="y ~ rl_std",
        random=["(1|main/sub)"],
        data=df,
    )
    np.testing.assert_allclose(res.predict(df), res.fittedvalues, atol=1e-6)
