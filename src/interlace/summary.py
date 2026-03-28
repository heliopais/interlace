"""summary() and VarCorr() for CrossedLMEResult.

Provides lme4-style formatted output:
  - VarCorr(result)  — variance-covariance table (grp, var1, var2, vcov, sdcor)
  - SummaryResult    — full text summary matching lme4::summary.merMod() layout
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult


class VarCorrResult:
    """Structured variance-covariance components from a fitted mixed model.

    Mirrors the output of R's ``VarCorr()`` function.
    """

    def __init__(self, result: CrossedLMEResult) -> None:
        self._result = result
        self._df = self._build_dataframe()

    def _build_dataframe(self) -> Any:
        import pandas as pd

        rows: list[dict[str, Any]] = []

        for spec in self._result._random_specs:
            grp = spec.group
            vc = self._result.variance_components[grp]

            if spec.n_terms == 1:
                # Intercept-only: scalar variance
                var = float(vc)
                rows.append(
                    {
                        "grp": grp,
                        "var1": "(Intercept)",
                        "var2": float("nan"),
                        "vcov": var,
                        "sdcor": float(np.sqrt(var)),
                    }
                )
            else:
                # Multi-term: vc is a covariance matrix (DataFrame or ndarray)
                term_names = (["(Intercept)"] if spec.intercept else []) + list(
                    spec.predictors
                )

                cov_mat = vc.values if hasattr(vc, "values") else np.asarray(vc)

                p = len(term_names)
                for i in range(p):
                    var_i = float(cov_mat[i, i])
                    rows.append(
                        {
                            "grp": grp,
                            "var1": term_names[i],
                            "var2": float("nan"),
                            "vcov": var_i,
                            "sdcor": float(np.sqrt(max(var_i, 0.0))),
                        }
                    )

                # Off-diagonal entries: correlations (upper triangle only)
                for i in range(p):
                    for j in range(i + 1, p):
                        cov_ij = float(cov_mat[i, j])
                        sd_i = float(np.sqrt(max(cov_mat[i, i], 0.0)))
                        sd_j = float(np.sqrt(max(cov_mat[j, j], 0.0)))
                        corr = (
                            cov_ij / (sd_i * sd_j) if (sd_i > 0 and sd_j > 0) else 0.0
                        )
                        rows.append(
                            {
                                "grp": grp,
                                "var1": term_names[i],
                                "var2": term_names[j],
                                "vcov": cov_ij,
                                "sdcor": corr,
                            }
                        )

        # Residual row
        rows.append(
            {
                "grp": "Residual",
                "var1": float("nan"),
                "var2": float("nan"),
                "vcov": float(self._result.scale),
                "sdcor": float(np.sqrt(self._result.scale)),
            }
        )

        return pd.DataFrame(rows)

    def as_dataframe(self) -> Any:
        """Return variance components as a DataFrame.

        Columns: ``grp``, ``var1``, ``var2``, ``vcov``, ``sdcor``

        Matches the output of ``as.data.frame(VarCorr(fit))`` in R.
        """
        return self._df


def VarCorr(result: CrossedLMEResult) -> VarCorrResult:  # noqa: N802  (match R name)
    """Return variance-covariance components from a fitted model.

    Parameters
    ----------
    result:
        A :class:`~interlace.result.CrossedLMEResult` from :func:`~interlace.fit`.

    Returns
    -------
    VarCorrResult
        Object with an :meth:`~VarCorrResult.as_dataframe` method returning
        columns ``grp``, ``var1``, ``var2``, ``vcov``, ``sdcor`` — matching
        ``as.data.frame(VarCorr(fit))`` in R.
    """
    return VarCorrResult(result)


class SummaryResult:
    """Human-readable summary of a fitted mixed model.

    Mirrors the output of ``summary.merMod()`` in R's lme4/lmerTest.
    """

    def __init__(self, result: CrossedLMEResult) -> None:
        self._result = result

    def __str__(self) -> str:
        return self._render()

    def __repr__(self) -> str:
        return self._render()

    def _render(self) -> str:
        r = self._result
        lines: list[str] = []

        # Header
        lines.append("Linear mixed model fit by REML")
        lines.append("")
        lines.append(f"Formula: {r.model.formula}")
        lines.append("")

        # REML criterion
        lines.append(f"REML criterion at convergence: {r.llf:.1f}")
        lines.append("")

        # Scaled residuals
        scaled = np.asarray(r.resid) / np.sqrt(r.scale)
        q = np.quantile(scaled, [0.0, 0.25, 0.5, 0.75, 1.0])
        lines.append("Scaled residuals:")
        lines.append(
            f"  Min: {q[0]:8.4f}  1Q: {q[1]:8.4f}  Median: {q[2]:8.4f}"
            f"  3Q: {q[3]:8.4f}  Max: {q[4]:8.4f}"
        )
        lines.append("")

        # Random effects (VarCorr table)
        vc_df = VarCorr(r).as_dataframe()
        lines.append("Random effects:")
        lines.append(f" {'Groups':<15} {'Name':<15} {'Variance':>12} {'Std.Dev.':>10}")
        for _, row in vc_df.iterrows():
            grp = str(row["grp"])
            var1 = "" if _is_na(row["var1"]) else str(row["var1"])
            var2 = row["var2"]
            vcov = float(row["vcov"])
            sdcor = float(row["sdcor"])
            if not _is_na(var2):
                # Correlation row
                lines.append(
                    f" {'':15} {'':15} {'':>12} {sdcor:>10.4f}  [corr {var1},{var2}]"
                )
            else:
                lines.append(f" {grp:<15} {var1:<15} {vcov:>12.6f} {sdcor:>10.6f}")
        lines.append("")

        # Number of obs and groups
        groups_str = "; ".join(f"{g}, {n}" for g, n in r.ngroups.items())
        lines.append(f"Number of obs: {r.nobs}, groups: {groups_str}")
        lines.append("")

        # Fixed effects
        lines.append("Fixed effects:")
        fe_arr = np.asarray(r.fe_params)
        bse_arr = np.asarray(r.fe_bse)
        pval_arr = np.asarray(r.fe_pvalues)
        tval_arr = fe_arr / bse_arr
        df_arr = np.asarray(r.fe_df)

        try:
            names = list(r.fe_params.index)
        except AttributeError:
            names = [f"x{i}" for i in range(len(fe_arr))]

        header = (
            f"  {'':20} {'Estimate':>12} {'Std. Error':>12}"
            f" {'df':>8} {'t value':>10} {'Pr(>|t|)':>12}"
        )
        lines.append(header)
        for name, est, se, df_val, tv, pv in zip(
            names, fe_arr, bse_arr, df_arr, tval_arr, pval_arr, strict=True
        ):
            stars = _pval_stars(pv)
            pv_str = f"{pv:.4e}" if pv < 0.001 else f"{pv:.4f}"
            line = (
                f"  {name:<20} {est:>12.4f} {se:>12.4f}"
                f" {df_val:>8.1f} {tv:>10.4f} {pv_str:>12} {stars}"
            )
            lines.append(line)
        lines.append("---")
        lines.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        # Convergence
        status = "converged" if r.converged else "DID NOT CONVERGE"
        lines.append(f"Optimizer convergence: {status}")
        lines.append(f"AIC: {r.aic:.2f}  BIC: {r.bic:.2f}  logLik: {r.llf:.2f}")

        return "\n".join(lines)


def _is_na(val: Any) -> bool:
    try:
        return bool(val != val)  # NaN check
    except (TypeError, ValueError):
        return False


def _pval_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return " "
