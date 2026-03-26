"""CrossedLMEResult and ModelInfo dataclasses.

CrossedLMEResult is a drop-in replacement for statsmodels MixedLMResults,
exposing all attributes accessed by the gpgap diagnostics pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class _DataWrapper:
    """Thin wrapper so result.model.data.frame mirrors statsmodels.

    ``frame`` holds the caller's original native frame (pandas, polars, etc.)
    so that downstream diagnostics can return results in the same type.
    ``_pandas_frame`` caches the pandas version for internal use.
    """

    frame: object  # native type as passed by the caller
    _pandas_frame: pd.DataFrame = field(repr=False, default=None)  # type: ignore[assignment]


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

    # Fixed effects
    fe_params: pd.Series
    fe_bse: pd.Series
    fe_pvalues: pd.Series
    fe_conf_int: pd.DataFrame

    # Random effects
    random_effects: dict[str, pd.Series | pd.DataFrame]
    variance_components: dict[str, float | pd.DataFrame]
    theta: np.ndarray

    # Residuals and fitted values
    resid: np.ndarray
    fittedvalues: np.ndarray
    scale: float

    # Fixed-effects covariance matrix (p×p): scale * (X'Ω⁻¹X)⁻¹
    fe_cov: np.ndarray

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

    # gpgap compatibility
    _gpgap_group_col: str
    _gpgap_vc_cols: list[str] = field(default_factory=list)

    def predict(
        self,
        newdata: object | None = None,
        include_re: bool = True,
    ) -> np.ndarray:
        """Return predictions; see :func:`interlace.predict.predict`."""
        from interlace.predict import predict

        return predict(self, newdata=newdata, include_re=include_re)
