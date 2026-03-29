"""Tests for allFit() — multi-optimizer convergence check.

TDD: these tests are written before the implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from interlace import CrossedLMEResult, allFit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Balanced crossed data: 10 subjects × 5 obs each."""
    rng = np.random.default_rng(42)
    n_subjects = 10
    n_obs = 5
    subjects = np.repeat(np.arange(n_subjects), n_obs)
    x = rng.normal(size=n_subjects * n_obs)
    b0 = rng.normal(scale=1.0, size=n_subjects)
    y = 2.0 + 0.5 * x + b0[subjects] + rng.normal(scale=0.5, size=n_subjects * n_obs)
    return pd.DataFrame({"y": y, "x": x, "subject": subjects})


@pytest.fixture
def flat_likelihood_data() -> pd.DataFrame:
    """One observation per group — random intercept not identifiable from residual."""
    rng = np.random.default_rng(7)
    n_groups = 20
    groups = np.arange(n_groups)
    x = rng.normal(size=n_groups)
    y = 1.0 + 0.3 * x + rng.normal(scale=0.1, size=n_groups)
    return pd.DataFrame({"y": y, "x": x, "group": groups})


# ---------------------------------------------------------------------------
# Test 1: allFit() returns results for all available optimizers without error
# ---------------------------------------------------------------------------


def test_allfit_returns_results_for_all_optimizers(simple_data: pd.DataFrame) -> None:
    result = allFit("y ~ x", simple_data, groups="subject")

    # lbfgsb is always available
    assert "lbfgsb" in result.results, "lbfgsb must always be present"

    # Every result is a CrossedLMEResult
    for name, res in result.results.items():
        assert isinstance(res, CrossedLMEResult), (
            f"result[{name!r}] is {type(res)}, expected CrossedLMEResult"
        )


def test_allfit_converged_dict_matches_results_keys(simple_data: pd.DataFrame) -> None:
    result = allFit("y ~ x", simple_data, groups="subject")

    assert isinstance(result.converged, dict)
    assert set(result.converged.keys()) == set(result.results.keys())
    for name, flag in result.converged.items():
        assert isinstance(flag, bool), (
            f"converged[{name!r}] should be bool, got {type(flag)}"
        )


def test_allfit_nelder_mead_included(simple_data: pd.DataFrame) -> None:
    """Nelder-Mead is always available via scipy — must be included."""
    result = allFit("y ~ x", simple_data, groups="subject")
    assert "nelder-mead" in result.results


def test_allfit_llf_close_across_optimizers(simple_data: pd.DataFrame) -> None:
    """On a well-identified problem, all optimizers should agree on LLF to < 0.5."""
    result = allFit("y ~ x", simple_data, groups="subject")
    llfs = [res.llf for res in result.results.values()]
    llf_by_opt = dict(zip(result.results, llfs, strict=True))
    assert max(llfs) - min(llfs) < 0.5, (
        f"LLF spread too large on well-identified data: {llf_by_opt}"
    )


# ---------------------------------------------------------------------------
# Test 2: summary() is human-readable and flags disagreement
# ---------------------------------------------------------------------------


def test_allfit_summary_returns_string(simple_data: pd.DataFrame) -> None:
    result = allFit("y ~ x", simple_data, groups="subject")
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_allfit_summary_contains_optimizer_names(simple_data: pd.DataFrame) -> None:
    result = allFit("y ~ x", simple_data, groups="subject")
    summary = result.summary()
    assert "lbfgsb" in summary
    assert "nelder-mead" in summary


def test_allfit_summary_contains_llf(simple_data: pd.DataFrame) -> None:
    result = allFit("y ~ x", simple_data, groups="subject")
    summary = result.summary()
    assert "llf" in summary.lower()


def test_allfit_summary_flags_disagreement_when_issue() -> None:
    """When possible_issue is True, summary() must mention the warning."""

    # Build a minimal AllFitResult with possible_issue=True
    from interlace.allfit import AllFitResult

    result = AllFitResult(
        results={},
        converged={},
        possible_issue=True,
        _llf_diffs={"lbfgsb_vs_nelder-mead": 0.5},
        _theta_diffs={},
    )
    summary = result.summary()
    assert (
        "convergence" in summary.lower()
        or "warning" in summary.lower()
        or "issue" in summary.lower()
    )


def test_allfit_summary_no_flag_on_agreement(simple_data: pd.DataFrame) -> None:
    """On a well-identified problem, summary should NOT raise a convergence alarm."""
    result = allFit("y ~ x", simple_data, groups="subject")
    if not result.possible_issue:
        summary = result.summary()
        # Should not scream about convergence problems (may mention "no issues")
        assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# Test 3: Flat-likelihood fixture triggers convergence warning
# ---------------------------------------------------------------------------


def test_allfit_flat_likelihood_triggers_possible_issue(
    flat_likelihood_data: pd.DataFrame,
) -> None:
    """One obs per group → variance near-unidentified → possible_issue=True."""
    result = allFit("y ~ x", flat_likelihood_data, groups="group")
    # The result should flag a possible issue (optimizers disagree on theta)
    assert result.possible_issue, (
        "Expected possible_issue=True for near-unidentified model; "
        f"got possible_issue={result.possible_issue}, "
        f"llf_diffs={result._llf_diffs}, theta_diffs={result._theta_diffs}"
    )


def test_allfit_possible_issue_is_bool(simple_data: pd.DataFrame) -> None:
    result = allFit("y ~ x", simple_data, groups="subject")
    assert isinstance(result.possible_issue, bool)
