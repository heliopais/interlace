"""Slow-path warning when sksparse/CHOLMOD is unavailable.

The hard paths (random-slopes LMM, GLMM Laplace) are materially slower without
CHOLMOD.  Users should see a one-shot ``UserWarning`` at fit time pointing
them at the ``[fast]`` extra, rather than silently sitting in the SuperLU
fallback.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import interlace
from interlace import _cholmod_warn


@pytest.fixture(autouse=True)
def _reset_warn_latch():
    _cholmod_warn._reset_for_tests()
    yield
    _cholmod_warn._reset_for_tests()


@pytest.fixture()
def _force_no_cholmod(monkeypatch):
    """Simulate an environment without sksparse installed."""
    import interlace.profiled_reml as pr

    monkeypatch.setattr(pr, "_try_cholmod", lambda: None)


@pytest.fixture()
def slope_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n, q = 200, 10
    group_codes = np.repeat(np.arange(q), n // q)
    x = rng.normal(size=n)
    b_int = rng.normal(scale=0.8, size=q)
    b_slope = rng.normal(scale=0.4, size=q)
    y = 1.0 + 0.5 * x + b_int[group_codes] + b_slope[group_codes] * x
    y += rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": group_codes.astype(str)})


@pytest.fixture()
def intercept_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n, q = 160, 8
    group_codes = np.repeat(np.arange(q), n // q)
    x = rng.normal(size=n)
    b = rng.normal(scale=0.7, size=q)
    y = 1.0 + 0.5 * x + b[group_codes] + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": group_codes.astype(str)})


@pytest.fixture()
def binom_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n, q = 200, 10
    group_codes = np.repeat(np.arange(q), n // q)
    x = rng.normal(size=n)
    b = rng.normal(scale=0.6, size=q)
    eta = -0.3 + 0.4 * x + b[group_codes]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"y": y, "x": x, "g": group_codes.astype(str)})


class TestSlowPathWarning:
    def test_random_slopes_lmm_warns_when_no_cholmod(
        self,
        _force_no_cholmod,
        slope_df: pd.DataFrame,
    ) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
        assert any("[fast]" in m and "scikit-sparse" in m for m in msgs), (
            f"expected slow-path warning, got: {msgs}"
        )

    def test_intercept_only_lmm_does_not_warn(
        self,
        _force_no_cholmod,
        intercept_df: pd.DataFrame,
    ) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            interlace.fit("y ~ x", data=intercept_df, random=["(1 | g)"])
        msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
        assert not any("scikit-sparse" in m for m in msgs), (
            f"intercept-only LMM should not warn, got: {msgs}"
        )

    def test_glmm_warns_when_no_cholmod(
        self,
        _force_no_cholmod,
        binom_df: pd.DataFrame,
    ) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            interlace.glmer(
                "y ~ x",
                data=binom_df,
                family="binomial",
                groups="g",
            )
        msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
        assert any("[fast]" in m and "scikit-sparse" in m for m in msgs), (
            f"expected slow-path warning, got: {msgs}"
        )

    def test_warns_only_once(
        self,
        _force_no_cholmod,
        slope_df: pd.DataFrame,
    ) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
            interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        msgs = [
            str(w.message)
            for w in captured
            if issubclass(w.category, UserWarning) and "scikit-sparse" in str(w.message)
        ]
        assert len(msgs) == 1, f"expected exactly one warning, got {len(msgs)}: {msgs}"

    def test_no_warning_when_cholmod_present(
        self,
        slope_df: pd.DataFrame,
    ) -> None:
        pytest.importorskip("sksparse")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            interlace.fit("y ~ x", data=slope_df, random=["(1 + x | g)"])
        msgs = [
            str(w.message)
            for w in captured
            if issubclass(w.category, UserWarning) and "scikit-sparse" in str(w.message)
        ]
        assert not msgs, f"should be silent when sksparse is installed, got: {msgs}"


class TestPublicHelper:
    def test_module_exposes_maybe_warn_slow_path(self) -> None:
        assert callable(_cholmod_warn.maybe_warn_slow_path)

    def test_message_mentions_fast_extra(self, _force_no_cholmod) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _cholmod_warn.maybe_warn_slow_path("Random-slopes LMM")
        msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
        assert len(msgs) == 1
        assert "interlace-lme[fast]" in msgs[0]
        assert "Random-slopes LMM" in msgs[0]
