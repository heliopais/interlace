"""Acceptance tests: n_influential parity.

We use statsmodels MixedLM as the R-validated reference (parity with R lme4
is already established in test_parity_single_re.py).  Criteria:

  - n_influential: CrossedLMEResult count within 5% of statsmodels count
    (or both zero, which passes trivially)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM

import interlace
from interlace.influence import n_influential


@pytest.fixture(scope="module")
def single_re_data() -> pd.DataFrame:
    """200 observations, 20 groups — matches test_parity_single_re fixture."""
    rng = np.random.default_rng(42)
    n_groups, n_per = 20, 10
    n = n_groups * n_per
    group_ids = np.repeat(np.arange(n_groups), n_per)
    x = rng.standard_normal(n)
    u = rng.normal(0, 1.2, n_groups)
    eps = rng.normal(0, 0.8, n)
    y = 2.0 + 1.5 * x + u[group_ids] + eps
    return pd.DataFrame({"y": y, "x": x, "group": group_ids.astype(str)})


@pytest.fixture(scope="module")
def models_single_re(single_re_data):
    sm = MixedLM.from_formula("y ~ x", groups="group", data=single_re_data).fit(
        reml=True
    )
    il = interlace.fit("y ~ x", data=single_re_data, groups="group")
    return sm, il


# ---------------------------------------------------------------------------
# n_influential acceptance
# ---------------------------------------------------------------------------


def test_n_influential_within_5pct_of_statsmodels(models_single_re):
    sm, il = models_single_re
    n_sm = n_influential(sm)
    n_il = n_influential(il)

    if n_sm == 0 and n_il == 0:
        return  # both report no influential obs — perfect agreement

    if n_sm == 0:
        # interlace may detect a few via Cook's D; allow small absolute difference
        assert n_il <= 2, f"n_influential: sm=0, il={n_il} (too many)"
        return

    rel_diff = abs(n_il - n_sm) / n_sm
    assert rel_diff <= 0.05, (
        f"n_influential rel_diff={rel_diff:.2%} (il={n_il}, sm={n_sm}), must be ≤5%"
    )


def test_n_influential_custom_threshold_parity(models_single_re, single_re_data):
    sm, il = models_single_re
    n = len(single_re_data)
    for threshold in (4.0 / n, 0.1, 0.5):
        n_sm = n_influential(sm, threshold=threshold)
        n_il = n_influential(il, threshold=threshold)
        if n_sm == 0:
            assert n_il <= 2, f"threshold={threshold}: sm=0, il={n_il}"
        else:
            rel_diff = abs(n_il - n_sm) / n_sm
            assert rel_diff <= 0.05, (
                f"threshold={threshold}: n_influential rel_diff={rel_diff:.2%} "
                f"(il={n_il}, sm={n_sm})"
            )
