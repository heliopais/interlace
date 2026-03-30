"""OLS fitting with formulaic design matrices and HC3 robust standard errors.

Provides a statsmodels-compatible interface without depending on statsmodels
or pandas for the fitting step. Any narwhals-compatible frame (pandas, polars,
…) is accepted as input data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import formulaic
import narwhals as nw
import numpy as np
import pandas as pd


@dataclass
class _OLSDataWrapper:
    """Mirrors result.model.data.frame (statsmodels compat)."""

    frame: Any  # native type as passed by the caller (pandas, polars, etc.)


@dataclass
class _OLSModelInfo:
    """Container for OLS model matrices and metadata."""

    exog: np.ndarray
    endog: np.ndarray
    exog_names: list[str]
    formula: str
    data: _OLSDataWrapper
    _rhs_model_spec: Any  # formulaic ModelSpec for predict()


class OLSResult:
    """Result of an OLS fit via formulaic + numpy.

    Attributes match the statsmodels OLS interface used in gpgap so that this
    is a drop-in replacement.
    """

    def __init__(
        self,
        params: pd.Series,
        resid: np.ndarray,
        fittedvalues: np.ndarray,
        normalized_cov_params: np.ndarray,
        model: _OLSModelInfo,
    ) -> None:
        self.params = params
        self.resid = resid
        self.fittedvalues = fittedvalues
        self.normalized_cov_params = normalized_cov_params
        self.model = model

    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom: n - p, matching statsmodels OLS df_resid."""
        return float(len(self.resid) - len(self.params))

    @property
    def scale(self) -> float:
        """Residual variance: RSS / (n - p), matching statsmodels OLS scale."""
        return float(np.sum(self.resid**2) / self.df_resid)

    def hc3_bse(self) -> np.ndarray:
        """HC3 heteroskedasticity-consistent standard errors.

        Computes:
            h_ii = diag(X (X'X)^{-1} X')
            d_i  = e_i^2 / (1 - h_ii)^2
            cov_HC3 = (X'X)^{-1} (X' diag(d) X) (X'X)^{-1}
            se_HC3  = sqrt(diag(cov_HC3))

        Returns
        -------
        np.ndarray, shape (p,)
        """
        X = self.model.exog
        e = self.resid
        XtX_inv = self.normalized_cov_params
        # Hat matrix diagonal: h_ii = (X (X'X)^{-1} X')_ii
        # Computed without forming full hat matrix: h_ii = sum_j (X @ XtX_inv * X)_ij
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        d = e**2 / (1.0 - h) ** 2
        meat = X.T @ (d[:, None] * X)
        cov_hc3 = XtX_inv @ meat @ XtX_inv
        return np.sqrt(np.diag(cov_hc3))

    def predict(self, data: Any) -> np.ndarray:
        """Predict on new data by re-evaluating the RHS formula.

        Parameters
        ----------
        data:
            DataFrame containing the predictor columns. Any narwhals-compatible
            frame is accepted.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        nw_data = nw.from_native(data, eager_only=True)
        rhs_spec = self.model._rhs_model_spec
        X_new = np.asarray(formulaic.model_matrix(rhs_spec, nw_data), dtype=float)
        return np.asarray(X_new @ self.params.values, dtype=float)


def ols(formula: str, data: Any) -> OLSResult:
    """Fit an OLS model using a formulaic formula string.

    Parameters
    ----------
    formula:
        Formula string, e.g. ``"y ~ x1 + x2"``.
    data:
        DataFrame containing all variables. Any narwhals-compatible frame
        (pandas, polars, …) is accepted.

    Returns
    -------
    OLSResult
        Drop-in replacement for statsmodels OLS results used in gpgap.
    """
    nw_data = nw.from_native(data, eager_only=True)

    matrices = formulaic.model_matrix(formula, nw_data)
    X = np.asarray(matrices.rhs, dtype=float)
    y = np.asarray(matrices.lhs, dtype=float).squeeze()
    term_names: list[str] = list(matrices.rhs.columns)
    rhs_model_spec = matrices.rhs.model_spec

    # QR-based least squares
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fittedvalues = X @ beta
    resid = y - fittedvalues

    normalized_cov_params = np.linalg.inv(X.T @ X)

    params = pd.Series(beta, index=term_names)

    model = _OLSModelInfo(
        exog=X,
        endog=y,
        exog_names=term_names,
        formula=formula,
        data=_OLSDataWrapper(frame=data),
        _rhs_model_spec=rhs_model_spec,
    )

    return OLSResult(
        params=params,
        resid=resid,
        fittedvalues=fittedvalues,
        normalized_cov_params=normalized_cov_params,
        model=model,
    )
