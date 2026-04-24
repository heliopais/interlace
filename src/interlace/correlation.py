"""Residual correlation structures for linear mixed models.

Provides AR(1) and a base class for future correlation structures (compound
symmetry, Toeplitz, etc.).  The key operation is *whitening*: transforming
(y, X, Z) so the residual covariance becomes identity, allowing the standard
profiled-REML machinery to operate on the transformed data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Low-level whitening function
# ---------------------------------------------------------------------------


def _ar1_whiten_vector(v: np.ndarray, rho: float, dt: np.ndarray) -> np.ndarray:
    """Apply innovation-form whitening to a 1-D vector within a single group.

    For continuous-time AR(1) (Ornstein-Uhlenbeck) with correlation
    ``rho^|t_i - t_j|``:

        w_1 = v_1
        w_i = (v_i - rho^dt_i * v_{i-1}) / sqrt(1 - rho^{2*dt_i})

    Parameters
    ----------
    v : array, shape (n,)
        Input vector (one group, sorted by time).
    rho : float
        AR(1) correlation parameter (|rho| < 1).
    dt : array, shape (n-1,)
        Time gaps between consecutive observations.

    Returns
    -------
    array, shape (n,)
        Whitened vector.
    """
    n = len(v)
    w = np.empty(n)
    w[0] = v[0]
    if n == 1:
        return w

    rho_dt = rho**dt  # shape (n-1,)
    scale = np.sqrt(1.0 - rho_dt**2)
    # Guard against rho=0 → scale=1, rho=±1 → scale=0 (degenerate)
    scale = np.maximum(scale, 1e-15)

    w[1:] = (v[1:] - rho_dt * v[:-1]) / scale
    return w


def _ar1_whiten_matrix(M: np.ndarray, rho: float, dt: np.ndarray) -> np.ndarray:
    """Apply AR(1) whitening to each column of a 2-D matrix within a group.

    Parameters
    ----------
    M : array, shape (n, p)
        Input matrix (one group, sorted by time).
    rho, dt :
        As for :func:`_ar1_whiten_vector`.

    Returns
    -------
    array, shape (n, p)
    """
    if M.ndim == 1:
        return _ar1_whiten_vector(M, rho, dt)

    n, p = M.shape
    W = np.empty_like(M)
    W[0, :] = M[0, :]
    if n == 1:
        return W

    rho_dt = rho**dt  # (n-1,)
    scale = np.sqrt(1.0 - rho_dt**2)
    scale = np.maximum(scale, 1e-15)

    W[1:, :] = (M[1:, :] - rho_dt[:, None] * M[:-1, :]) / scale[:, None]
    return W


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CorStruct(ABC):
    """Base class for residual correlation structures."""

    time_col: str  # name of the time/order column in the data

    @property
    @abstractmethod
    def n_corr_params(self) -> int:
        """Number of correlation parameters to estimate."""

    @abstractmethod
    def setup(
        self,
        groups: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Pre-compute group indices and sort orders.

        Called once before optimisation starts.
        """

    @abstractmethod
    def whiten_data(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: sp.csc_matrix,
        rho_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, sp.csc_matrix]:
        """Return whitened (y*, X*, Z*) given correlation parameters."""

    @abstractmethod
    def log_det_R(
        self,
        rho_params: np.ndarray,
        **kwargs: object,
    ) -> float:
        """Compute log|R| for the current correlation parameters."""

    def unconstrained_bounds(
        self,
    ) -> list[tuple[float | None, float | None]]:
        """Bounds on the unconstrained correlation parameters for the optimizer.

        Returns a list of (lower, upper) tuples, one per correlation parameter.
        Default is unbounded (-inf, inf).
        """
        return [(None, None)] * self.n_corr_params


# ---------------------------------------------------------------------------
# AR(1) implementation
# ---------------------------------------------------------------------------


class AR1(CorStruct):
    """First-order autoregressive residual correlation.

    Within each group, residuals separated by time gap dt have correlation
    ``rho^dt`` (continuous-time AR(1) / Ornstein-Uhlenbeck process).

    Parameters
    ----------
    time : str
        Name of the column in the data that contains the time/order variable.
    """

    def __init__(self, time: str) -> None:
        self.time_col = time
        # Populated by setup()
        self._group_slices: list[slice] = []
        self._dt_per_group: list[np.ndarray] = []
        self._sort_idx: np.ndarray | None = None
        self._unsort_idx: np.ndarray | None = None
        self._n_obs: int = 0
        self._is_setup: bool = False

    @property
    def n_corr_params(self) -> int:
        return 1

    def setup(
        self,
        groups: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Pre-compute group slices, sort order, and time gaps.

        Data is sorted by (group, time) internally. The sort and unsort
        indices are stored so that whitening can be applied efficiently.
        """
        n = len(groups)
        self._n_obs = n

        # Sort by (group, time)
        sort_idx = np.lexsort((times, groups))
        self._sort_idx = sort_idx
        self._unsort_idx = np.argsort(sort_idx)

        sorted_groups = groups[sort_idx]
        sorted_times = times[sort_idx]

        # Find group boundaries
        self._group_slices = []
        self._dt_per_group = []

        unique_groups = np.unique(sorted_groups)
        for g in unique_groups:
            mask = sorted_groups == g
            indices = np.where(mask)[0]
            start, end = indices[0], indices[-1] + 1
            self._group_slices.append(slice(start, end))

            t_g = sorted_times[start:end]
            dt_g = np.diff(t_g).astype(np.float64)
            self._dt_per_group.append(dt_g)

        self._is_setup = True

    def whiten_data(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: sp.csc_matrix,
        rho_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, sp.csc_matrix]:
        """Apply AR(1) whitening to (y, X, Z).

        Parameters
        ----------
        y : shape (n,)
        X : shape (n, p)
        Z : sparse (n, q)
        rho_params : shape (1,)
            The single AR(1) parameter (already mapped from unconstrained space).

        Returns
        -------
        (y_w, X_w, Z_w) : whitened data in original observation order.
        """
        assert self._is_setup, "Call setup() before whiten_data()"
        rho = float(rho_params[0])

        sort_idx = self._sort_idx
        unsort_idx = self._unsort_idx

        # Sort data
        y_s = y[sort_idx]
        X_s = X[sort_idx]
        Z_dense_s = Z[sort_idx].toarray()  # dense for whitening; re-sparsified after

        # Whiten group by group
        y_w = np.empty_like(y_s)
        X_w = np.empty_like(X_s)
        Z_w = np.empty_like(Z_dense_s)

        for sl, dt_g in zip(self._group_slices, self._dt_per_group, strict=True):
            y_w[sl] = _ar1_whiten_vector(y_s[sl], rho, dt_g)
            X_w[sl] = _ar1_whiten_matrix(X_s[sl], rho, dt_g)
            Z_w[sl] = _ar1_whiten_matrix(Z_dense_s[sl], rho, dt_g)

        # Unsort back to original order
        y_w = y_w[unsort_idx]
        X_w = X_w[unsort_idx]
        Z_w_sparse = sp.csc_matrix(Z_w[unsort_idx])

        return y_w, X_w, Z_w_sparse

    def log_det_R(
        self,
        rho_params: np.ndarray,
        **kwargs: object,
    ) -> float:
        """Compute log|R(rho)| = sum over groups of sum_{j} log(1 - rho^{2*dt_j}).

        When rho=0, log|R| = 0 (R is the identity).
        """
        assert self._is_setup, "Call setup() before log_det_R()"
        rho = float(rho_params[0])

        if abs(rho) < 1e-15:
            return 0.0

        log_det = 0.0
        for dt_g in self._dt_per_group:
            if len(dt_g) == 0:
                continue
            rho_2dt = rho ** (2.0 * dt_g)
            # log|R_g| = sum log(1 - rho^{2*dt_j}) for each time gap
            # This comes from: det(R) = prod(1 - rho^{2*dt_j}) for AR(1)
            log_det += np.sum(np.log(np.maximum(1.0 - rho_2dt, 1e-300)))

        return float(log_det)

    # Convenience for kwargs-based interface
    def log_det_R_from_arrays(
        self,
        rho: float,
        groups: np.ndarray,
        times: np.ndarray,
        sort_idx: np.ndarray,
    ) -> float:
        """Compute log|R| from raw arrays (used in tests)."""
        if not self._is_setup:
            self.setup(groups, times)
        return self.log_det_R(
            np.array([rho]), groups=groups, times=times, sort_idx=sort_idx
        )


# ---------------------------------------------------------------------------
# Compound symmetry (exchangeable) implementation
# ---------------------------------------------------------------------------


class CompoundSymmetry(CorStruct):
    """Exchangeable residual correlation within groups.

    All pairs of residuals within a group share the same correlation ``rho``.
    The within-group correlation matrix is ``R_g = (1-rho)*I + rho*J``
    where ``J`` is the all-ones matrix.

    Parameters
    ----------
    time : str
        Name of the grouping column in the data (used by the CorStruct
        interface; the actual time values are irrelevant for CS).
    """

    def __init__(self, time: str) -> None:
        self.time_col = time
        self._group_slices: list[slice] = []
        self._group_sizes: list[int] = []
        self._n_obs: int = 0
        self._is_setup: bool = False

    @property
    def n_corr_params(self) -> int:
        return 1

    def setup(
        self,
        groups: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Pre-compute group slices and sizes.

        Data is sorted by group internally so whitening can operate on
        contiguous slices.
        """
        n = len(groups)
        self._n_obs = n

        # Sort by group (time is irrelevant for CS but keep the interface)
        sort_idx = np.lexsort((times, groups))
        self._sort_idx = sort_idx
        self._unsort_idx = np.argsort(sort_idx)

        sorted_groups = groups[sort_idx]

        self._group_slices = []
        self._group_sizes = []
        for g in np.unique(sorted_groups):
            indices = np.where(sorted_groups == g)[0]
            start, end = indices[0], indices[-1] + 1
            self._group_slices.append(slice(start, end))
            self._group_sizes.append(end - start)

        self._is_setup = True

    def unconstrained_bounds(
        self,
    ) -> list[tuple[float | None, float | None]]:
        """Bounds for the unconstrained rho parameter.

        CS requires ``-1/(n_max - 1) < rho < 1`` for positive definiteness.
        We map the lower limit (with a small margin) to unconstrained space
        via atanh.
        """
        assert self._is_setup, "Call setup() before unconstrained_bounds()"
        n_max = max(self._group_sizes)
        rho_lo = -1.0 / (n_max - 1) + 0.01 if n_max > 1 else -0.99
        from interlace.correlation import unconstrained_from_rho

        return [(unconstrained_from_rho(rho_lo), None)]

    def whiten_data(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: sp.csc_matrix,
        rho_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, sp.csc_matrix]:
        """Apply compound symmetry whitening to (y, X, Z).

        For a group of size n_g with correlation rho:
            R^{-1/2} v  =  v / a  -  c * sum(v) * 1
        where a = sqrt(1 - rho), c = (1/a - 1/sqrt(1 + (n_g-1)*rho)) / n_g.
        """
        assert self._is_setup, "Call setup() before whiten_data()"
        rho = float(rho_params[0])

        sort_idx = self._sort_idx
        unsort_idx = self._unsort_idx

        y_s = y[sort_idx]
        X_s = X[sort_idx]
        Z_dense_s = Z[sort_idx].toarray()

        y_w = np.empty_like(y_s)
        X_w = np.empty_like(X_s)
        Z_w = np.empty_like(Z_dense_s)

        if abs(rho) < 1e-15:
            # Identity — no whitening needed
            y_w[:] = y_s
            X_w[:] = X_s
            Z_w[:] = Z_dense_s
        else:
            a = np.sqrt(max(1.0 - rho, 1e-15))
            for sl, n_g in zip(self._group_slices, self._group_sizes, strict=True):
                denom = np.sqrt(max(1.0 + (n_g - 1) * rho, 1e-15))
                c = (1.0 / a - 1.0 / denom) / n_g

                y_w[sl] = y_s[sl] / a - c * np.sum(y_s[sl])
                X_w[sl] = X_s[sl] / a - c * np.sum(X_s[sl], axis=0)
                Z_w[sl] = Z_dense_s[sl] / a - c * np.sum(Z_dense_s[sl], axis=0)

        y_w = y_w[unsort_idx]
        X_w = X_w[unsort_idx]
        Z_w_sparse = sp.csc_matrix(Z_w[unsort_idx])

        return y_w, X_w, Z_w_sparse

    def log_det_R(
        self,
        rho_params: np.ndarray,
        **kwargs: object,
    ) -> float:
        """Compute log|R(rho)| for compound symmetry.

        For a group of size n_g:
            log|R_g| = (n_g - 1) * log(1 - rho) + log(1 + (n_g - 1) * rho)
        Total is the sum over all groups.
        """
        assert self._is_setup, "Call setup() before log_det_R()"
        rho = float(rho_params[0])

        if abs(rho) < 1e-15:
            return 0.0

        log_det = 0.0
        for n_g in self._group_sizes:
            if n_g <= 1:
                continue
            log_det += (n_g - 1) * np.log(max(1.0 - rho, 1e-300))
            log_det += np.log(max(1.0 + (n_g - 1) * rho, 1e-300))

        return float(log_det)


# ---------------------------------------------------------------------------
# Unconstrained parameterisation helpers
# ---------------------------------------------------------------------------


def rho_from_unconstrained(raw: float) -> float:
    """Map unconstrained real to (-1, 1) via tanh."""
    return float(np.tanh(raw))


def unconstrained_from_rho(rho: float) -> float:
    """Map (-1, 1) to unconstrained real via atanh."""
    return float(np.arctanh(np.clip(rho, -0.999, 0.999)))
