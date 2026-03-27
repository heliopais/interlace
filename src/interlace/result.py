"""CrossedLMEResult and ModelInfo dataclasses.

CrossedLMEResult is a drop-in replacement for statsmodels MixedLMResults,
exposing all attributes accessed by the gpgap diagnostics pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class _SimpleRE:
    """Lightweight random-effect store used when pandas is not installed.

    Exposes the same interface as a pandas Series for the attributes used
    by interlace diagnostics: ``values``, ``index``, ``name``.
    """

    values: np.ndarray
    index: list[Any]
    name: str

    def __len__(self) -> int:
        return len(self.values)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return self.values if dtype is None else self.values.astype(dtype)


@dataclass
class _DataWrapper:
    """Thin wrapper so result.model.data.frame mirrors statsmodels.

    ``frame`` holds the caller's original native frame (pandas, polars, etc.)
    so that downstream diagnostics can return results in the same type.
    ``_pandas_frame`` caches the pandas version for internal use; ``None``
    when pandas is not installed.
    """

    frame: Any  # native type as passed by the caller (pandas, polars, etc.)
    _pandas_frame: Any = field(repr=False, default=None)


@dataclass
class ModelInfo:
    """Container for model matrices and metadata (mirrors statsmodels model attr)."""

    exog: np.ndarray
    endog: np.ndarray
    groups: np.ndarray
    endog_names: str
    formula: str
    data: _DataWrapper


@dataclass
class CrossedLMEResult:
    """Result of a profiled REML fit for crossed random intercepts.

    Attribute names and structure mirror statsmodels MixedLMResults so that
    this object is a drop-in replacement for the gpgap diagnostics pipeline.
    """

    # Fixed effects — pd.Series when pandas is installed, np.ndarray otherwise
    fe_params: Any
    fe_bse: Any
    fe_pvalues: Any
    fe_conf_int: Any  # pd.DataFrame (pandas) or np.ndarray shape (p, 2)

    # Random effects — dict values are pd.Series/_SimpleRE (intercept-only)
    # or pd.DataFrame/np.ndarray (multi-term)
    random_effects: dict[str, Any]
    variance_components: dict[str, Any]
    theta: np.ndarray

    # Residuals and fitted values
    resid: np.ndarray
    fittedvalues: np.ndarray
    scale: float

    # Fixed-effects covariance matrix (p×p): scale * (X'Ω⁻¹X)⁻¹
    fe_cov: np.ndarray

    # Satterthwaite denominator DFs — pd.Series when pandas is installed
    fe_df: Any

    # Model matrices
    model: ModelInfo

    # Fit metadata
    converged: bool
    nobs: int
    ngroups: dict[str, int]
    method: str
    llf: float
    aic: float
    bic: float
    nparams: int  # p (FE) + n_theta (RE variances) + 1 (sigma²)

    # gpgap compatibility
    _gpgap_group_col: str
    _gpgap_vc_cols: list[str] = field(default_factory=list)

    # Random effect specs — stored so diagnostics can reconstruct random= for refits
    _random_specs: list[Any] = field(default_factory=list)

    # Fitting context needed for post-fit computations (Satterthwaite, etc.)
    _Z: Any = field(default=None, repr=False)  # scipy sparse (n, q)
    _n_levels: list[int] = field(default_factory=list)

    @property
    def fe_tvalues(self) -> Any:
        """Fixed-effect z-scores (estimate / SE).

        Named ``fe_tvalues`` for API symmetry with statsmodels; note that
        interlace uses the normal (z) distribution for inference, not t.
        """
        try:
            import pandas as _pd

            if isinstance(self.fe_params, _pd.Series):
                return self.fe_params / self.fe_bse
        except ImportError:
            pass
        return np.asarray(self.fe_params) / np.asarray(self.fe_bse)

    def summary(self) -> object:
        """Return a human-readable summary mirroring lme4's ``summary.merMod()``."""
        from interlace.summary import SummaryResult

        return SummaryResult(self)

    def predict(
        self,
        newdata: object | None = None,
        include_re: bool = True,
    ) -> np.ndarray:
        """Return predictions; see :func:`interlace.predict.predict`."""
        from interlace.predict import predict

        return predict(self, newdata=newdata, include_re=include_re)

    def bootstrap_se(
        self,
        statistic: str = "median",
        n_bootstrap: int = 1000,
        resample_level: str = "group",
        seed: int | None = None,
    ) -> float:
        """Bootstrap standard error for a scalar statistic of the response.

        Parameters
        ----------
        statistic:
            Statistic to compute on each bootstrap sample.  Only ``"median"``
            is currently supported.
        n_bootstrap:
            Number of bootstrap replicates.
        resample_level:
            ``"group"`` (default) resamples grouping-factor levels with
            replacement, then includes all observations from the sampled
            groups — matching R's ``boot`` package cluster bootstrap used in
            the ``gpgap`` reference implementation.  ``"observation"``
            resamples individual observations; this underestimates the SE
            when group variance is substantial.
        seed:
            Seed for the random number generator.  Pass an integer for
            reproducible results.

        Returns
        -------
        float
            Bootstrap standard error (``std(bootstrap_stats, ddof=1)``).
        """
        _SUPPORTED_STATISTICS = {"median"}
        _SUPPORTED_LEVELS = {"group", "observation"}

        if statistic not in _SUPPORTED_STATISTICS:
            raise ValueError(
                f"statistic={statistic!r} is not supported; "
                f"choose one of {sorted(_SUPPORTED_STATISTICS)}"
            )
        if resample_level not in _SUPPORTED_LEVELS:
            raise ValueError(
                f"resample_level={resample_level!r} is not supported; "
                f"choose one of {sorted(_SUPPORTED_LEVELS)}"
            )

        y = self.model.endog
        rng = np.random.default_rng(seed)

        if resample_level == "group":
            groups = self.model.groups
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            boot_stats = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                sampled = rng.choice(unique_groups, size=n_groups, replace=True)
                # Use repeat counts so duplicated groups contribute all their obs
                indices = np.concatenate([np.where(groups == g)[0] for g in sampled])
                boot_stats[i] = np.median(y[indices])
        else:  # observation
            n = len(y)
            boot_stats = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                indices = rng.integers(0, n, size=n)
                boot_stats[i] = np.median(y[indices])

        return float(np.std(boot_stats, ddof=1))
