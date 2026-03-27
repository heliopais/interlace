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
