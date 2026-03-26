"""plotnine-based diagnostic plots for linear mixed models."""

from __future__ import annotations

from plotnine import (
    aes,
    geom_hline,
    geom_point,
    geom_smooth,
    geom_text,
    ggplot,
    labs,
    stat_qq,
    stat_qq_line,
    theme_bw,
)


def plot_resid(resid_df, type: str = "resid_vs_fitted") -> ggplot:  # noqa: A002
    """Residual plot from the output of :func:`~interlace.residuals.hlm_resid`.

    Parameters
    ----------
    resid_df:
        DataFrame with ``.resid`` and ``.fitted`` columns.
    type:
        ``"resid_vs_fitted"`` or ``"qq"``.
    """
    if type == "resid_vs_fitted":
        return (
            ggplot(resid_df, aes(x=".fitted", y=".resid"))
            + geom_point(alpha=0.5)
            + geom_hline(yintercept=0, linetype="dashed", color="red")
            + geom_smooth(method="loess", se=False, color="blue")
            + labs(title="Residuals vs Fitted", x="Fitted values", y="Residuals")
            + theme_bw()
        )
    if type == "qq":
        return (
            ggplot(resid_df, aes(sample=".resid"))
            + stat_qq()
            + stat_qq_line()
            + labs(title="Normal Q-Q Plot", x="Theoretical Quantiles", y="Sample Quantiles")
            + theme_bw()
        )
    msg = "type must be 'resid_vs_fitted' or 'qq'"
    raise ValueError(msg)


def plot_influence(influence_df, diag: str = "cooksd") -> ggplot:
    """Index plot of an influence diagnostic from :func:`~interlace.influence.hlm_influence`.

    Parameters
    ----------
    influence_df:
        DataFrame with at least one influence column.
    diag:
        Column to plot on the y-axis (default ``"cooksd"``).
    """
    df = influence_df.copy()
    if "index" not in df.columns:
        df["index"] = range(len(df))

    return (
        ggplot(df, aes(x="index", y=diag))
        + geom_point()
        + labs(title=f"Influence Diagnostic: {diag}", x="Observation Index", y=diag)
        + theme_bw()
    )


def dotplot_diag(
    influence_df,
    diag: str = "cooksd",
    cutoff: str | float = "internal",
    name: str | None = None,
) -> ggplot:
    """Ranked dotplot of an influence diagnostic with outlier labels.

    Parameters
    ----------
    influence_df:
        DataFrame from :func:`~interlace.influence.hlm_influence`.
    diag:
        Metric to plot.
    cutoff:
        ``"internal"`` uses 3×IQR above Q3; a float labels values above that
        threshold.
    name:
        Column to use for outlier labels.  Defaults to the first non-metric
        column in *influence_df*.
    """
    _metric_cols = {"cooksd", "mdffits", "covtrace", "covratio"}
    df = influence_df.copy()

    if name is None:
        candidates = [c for c in df.columns if c not in _metric_cols and "rvc." not in c]
        label_col = candidates[0] if candidates else "_label"
        if label_col == "_label":
            df["_label"] = range(len(df))
    else:
        label_col = name

    df = df.sort_values(diag).reset_index(drop=True)
    df["_plot_idx"] = range(len(df))

    if cutoff == "internal":
        q1, q3 = df[diag].quantile(0.25), df[diag].quantile(0.75)
        limit = q3 + 3.0 * (q3 - q1)
        df["_is_outlier"] = df[diag] > limit
    else:
        df["_is_outlier"] = df[diag] > float(cutoff)

    outliers = df[df["_is_outlier"]]
    nudge = df[diag].max() * 0.02

    return (
        ggplot(df, aes(x="_plot_idx", y=diag))
        + geom_point()
        + geom_text(
            aes(label=label_col),
            data=outliers,
            va="bottom",
            ha="right",
            size=8,
            nudge_y=nudge,
        )
        + labs(title=f"Dotplot of {diag}", x="Rank", y=diag)
        + theme_bw()
    )
