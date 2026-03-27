"""allFit(): refit a model with all available optimizers and compare convergence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AllFitResult:
    """Result of an allFit() call.

    Attributes
    ----------
    results:
        Dict mapping optimizer name → CrossedLMEResult.
    converged:
        Dict mapping optimizer name → bool (optimizer-reported convergence).
    possible_issue:
        True when the max pairwise LLF difference exceeds 0.001 or any
        pairwise theta relative difference exceeds 1 %.
    _llf_diffs:
        Dict of pairwise LLF differences (for diagnostics / summary).
    _theta_diffs:
        Dict of pairwise max-relative theta differences (for diagnostics).
    """

    results: dict[str, Any]
    converged: dict[str, bool]
    possible_issue: bool
    _llf_diffs: dict[str, float] = field(default_factory=dict)
    _theta_diffs: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable table of per-optimizer fit statistics."""
        lines: list[str] = []
        lines.append("allFit() convergence summary")
        lines.append("=" * 60)

        # Header
        col_w = 14
        header = f"{'optimizer':<{col_w}}{'converged':>10}{'llf':>14}{'theta[0]':>12}"
        lines.append(header)
        lines.append("-" * 60)

        for name, res in self.results.items():
            try:
                theta0 = float(res.theta[0]) if hasattr(res, "theta") else float("nan")
                llf = float(res.llf) if hasattr(res, "llf") else float("nan")
            except Exception:
                theta0 = float("nan")
                llf = float("nan")
            conv = self.converged.get(name, False)
            row = (
                f"{name:<{col_w}}"
                f"{'yes' if conv else 'no':>10}"
                f"{llf:>14.4f}"
                f"{theta0:>12.4f}"
            )
            lines.append(row)

        lines.append("-" * 60)

        if self._llf_diffs:
            max_llf_diff = max(abs(v) for v in self._llf_diffs.values())
            lines.append(f"Max pairwise |Δllf|: {max_llf_diff:.6f}")
        if self._theta_diffs:
            max_theta_diff = max(abs(v) for v in self._theta_diffs.values())
            lines.append(f"Max pairwise rel Δtheta: {max_theta_diff:.4%}")

        if self.possible_issue:
            lines.append("")
            lines.append(
                "WARNING: Possible convergence issue — optimizers disagree on "
                "log-likelihood or variance parameters. Consider checking for "
                "near-singular fit or unidentified variance components."
            )
        else:
            lines.append("All optimizers agree (no convergence issues detected).")

        return "\n".join(lines)


def allFit(
    formula: str,
    data: Any,
    groups: str | list[str] | None = None,
    method: str = "REML",
    random: list[str] | None = None,
    theta0: Any = None,
) -> AllFitResult:
    """Refit a model with all available optimizers and compare convergence.

    Parameters
    ----------
    formula, data, groups, method, random, theta0:
        Passed through to :func:`interlace.fit` for each optimizer.

    Returns
    -------
    AllFitResult
        Contains per-optimizer results, convergence flags, pairwise diffs,
        and a ``possible_issue`` flag.
    """
    import numpy as np

    from interlace import fit

    # Determine which optimizers are available.
    optimizers = ["lbfgsb", "nelder-mead"]
    try:
        import pybobyqa as _  # noqa: F401

        optimizers.append("bobyqa")
    except ImportError:
        pass

    results: dict[str, Any] = {}
    converged: dict[str, bool] = {}

    for opt_name in optimizers:
        res = fit(
            formula,
            data,
            groups=groups,
            method=method,
            random=random,
            optimizer=opt_name,
            theta0=theta0,
        )
        results[opt_name] = res
        converged[opt_name] = bool(res.converged)

    # Compute pairwise diffs.
    llf_diffs: dict[str, float] = {}
    theta_diffs: dict[str, float] = {}
    opt_names = list(results.keys())
    for i in range(len(opt_names)):
        for j in range(i + 1, len(opt_names)):
            a, b = opt_names[i], opt_names[j]
            key = f"{a}_vs_{b}"

            llf_a = float(results[a].llf)
            llf_b = float(results[b].llf)
            llf_diffs[key] = llf_a - llf_b

            theta_a = np.asarray(results[a].theta, dtype=float)
            theta_b = np.asarray(results[b].theta, dtype=float)
            denom = np.maximum(np.abs(theta_a), 1e-8)
            rel_diff = float(np.max(np.abs(theta_a - theta_b) / denom))
            theta_diffs[key] = rel_diff

    # Flag if any pair disagrees beyond tolerance.
    llf_threshold = 0.001
    theta_threshold = 0.01  # 1 %
    possible_issue = any(abs(v) > llf_threshold for v in llf_diffs.values()) or any(
        v > theta_threshold for v in theta_diffs.values()
    )

    return AllFitResult(
        results=results,
        converged=converged,
        possible_issue=possible_issue,
        _llf_diffs=llf_diffs,
        _theta_diffs=theta_diffs,
    )
