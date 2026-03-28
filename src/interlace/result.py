"""CrossedLMEResult and ModelInfo dataclasses.

CrossedLMEResult is a drop-in replacement for statsmodels MixedLMResults,
exposing all attributes accessed by the gpgap diagnostics pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _expand_dot_formula(new_formula: str, orig_formula: str) -> str:
    """Expand lme4-style dot notation in *new_formula* using *orig_formula*.

    A standalone ``.`` in the LHS or RHS is replaced by the corresponding
    part of *orig_formula*.  For example::

        _expand_dot_formula(". ~ . + x2", "y ~ x1")  # → "y ~ x1 + x2"
        _expand_dot_formula(". ~ . - x1", "y ~ x1 + x2")  # → "y ~ x1 + x2 - x1"

    If *new_formula* contains no ``.`` it is returned unchanged.
    """
    if "." not in new_formula:
        return new_formula

    orig_lhs, orig_rhs = (s.strip() for s in orig_formula.split("~", 1))

    if "~" not in new_formula:
        new_lhs = orig_lhs
        new_rhs_template = new_formula.strip()
    else:
        new_lhs, new_rhs_template = (s.strip() for s in new_formula.split("~", 1))
        if new_lhs == ".":
            new_lhs = orig_lhs

    # Replace a standalone '.' (not part of a variable name) with orig_rhs.
    # A standalone dot is preceded/followed by non-word, non-dot characters.
    new_rhs = re.sub(r"(?<![.\w])\.(?![.\w])", orig_rhs, new_rhs_template)

    return f"{new_lhs} ~ {new_rhs}"


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

    # Original fit() kwargs for update() replay
    _fit_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)

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

    def simulate(
        self,
        nsim: int = 1,
        seed: int | None = None,
    ) -> np.ndarray:
        """Simulate response vectors; see :func:`interlace.simulate.simulate`."""
        from interlace.simulate import simulate

        return simulate(self, nsim=nsim, seed=seed)

    def bootMer(
        self,
        statistic: Any = None,
        B: int = 500,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> Any:
        """Parametric bootstrap; see :func:`interlace.simulate.bootMer`."""
        from interlace.simulate import bootMer

        return bootMer(
            self, statistic=statistic, B=B, seed=seed, show_progress=show_progress
        )

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

    @property
    def random_effects_se(self) -> dict[str, Any]:
        """Standard errors for BLUP estimates.

        Returns the square-root of the diagonal of the posterior variance
        matrix ``Var(b | y) = σ² * Lambda * A11⁻¹ * Lambda'``.

        Returns
        -------
        dict[str, Any]
            Same structure as :attr:`random_effects`: for intercept-only specs
            a pd.Series/_SimpleRE per group, for multi-term specs a
            pd.DataFrame per group.
        """
        import scipy.sparse.linalg as _spla

        from interlace.profiled_reml import _build_A11, make_lambda

        Z = self._Z
        Lambda = make_lambda(self.theta, self._random_specs, self._n_levels)
        ZtZ = (Z.T @ Z).tocsc()
        A11 = _build_A11(ZtZ, Lambda)
        sigma2 = self.scale

        # Compute diag(Lambda * A11^{-1} * Lambda') via sparse factorization.
        # Solve A11 * W = Lambda' (dense RHS); W = A11^{-1} * Lambda'.
        # diag(Lambda * W) = row-wise dot product of Lambda and W'.
        factor = _spla.splu(A11)
        Lambda_arr = Lambda.toarray()  # (q, q)
        W = factor.solve(Lambda_arr.T)  # (q, q): A11^{-1} * Lambda'
        # diag_i = sum_k Lambda[i, k] * W[k, i]  =  row of Lambda · col of W
        var_blup = np.einsum("ij,ji->i", Lambda_arr, W)  # (q,)
        se_blup = np.sqrt(np.maximum(sigma2 * var_blup, 0.0))

        # Split into per-spec blocks (same layout as random_effects)
        try:
            import pandas as _pd

            _pandas_available = True
        except ImportError:
            _pandas_available = False

        result: dict[str, Any] = {}
        blup_offset = 0
        for spec, q_j in zip(self._random_specs, self._n_levels, strict=True):
            n_blups_j = spec.n_terms * q_j
            se_block = se_blup[blup_offset : blup_offset + n_blups_j]
            re_ref = self.random_effects[spec.group]

            if spec.n_terms == 1:
                if _pandas_available:
                    result[spec.group] = _pd.Series(
                        se_block, index=re_ref.index, name=spec.group
                    )
                else:
                    result[spec.group] = _SimpleRE(
                        values=se_block, index=re_ref.index, name=spec.group
                    )
            else:
                # se_block is term-first: reshape to (n_terms, q_j) then transpose
                se_mat = se_block.reshape(spec.n_terms, q_j).T
                if _pandas_available:
                    result[spec.group] = _pd.DataFrame(
                        se_mat, index=re_ref.index, columns=re_ref.columns
                    )
                else:
                    result[spec.group] = se_mat

            blup_offset += n_blups_j

        return result

    def random_effects_ci(self, level: float = 0.95) -> dict[str, Any]:
        """Confidence intervals for BLUP estimates (normal approximation).

        Returns symmetric CIs: ``blup ± z * se``, where ``z`` is the
        appropriate quantile of the standard normal.

        Parameters
        ----------
        level:
            Nominal coverage probability (default 0.95).

        Returns
        -------
        dict[str, Any]
            For intercept-only specs: pd.DataFrame with columns
            ``["lower", "upper"]`` and index matching group levels.
            For multi-term specs: pd.DataFrame with MultiIndex columns
            ``(term, bound)`` where bound is ``"lower"`` or ``"upper"``.
        """
        from scipy.stats import norm

        try:
            import pandas as _pd

            _pandas_available = True
        except ImportError:
            _pandas_available = False

        z = float(norm.ppf((1.0 + level) / 2.0))
        se_dict = self.random_effects_se

        result: dict[str, Any] = {}
        for group, re_val in self.random_effects.items():
            se_val = se_dict[group]
            blup_arr = np.asarray(re_val)
            se_arr = np.asarray(se_val)

            if _pandas_available:
                import pandas as _pd  # noqa: PLC0415

                re_index = re_val.index if hasattr(re_val, "index") else None

                if blup_arr.ndim == 1:
                    # intercept-only
                    result[group] = _pd.DataFrame(
                        {
                            "lower": blup_arr - z * se_arr,
                            "upper": blup_arr + z * se_arr,
                        },
                        index=re_index,
                    )
                else:
                    # multi-term: produce MultiIndex columns (term, lower/upper)
                    cols_terms = list(re_val.columns)
                    data: dict[tuple[str, str], Any] = {}
                    for i, term in enumerate(cols_terms):
                        data[(term, "lower")] = blup_arr[:, i] - z * se_arr[:, i]
                        data[(term, "upper")] = blup_arr[:, i] + z * se_arr[:, i]
                    result[group] = _pd.DataFrame(data, index=re_index)
            else:
                # numpy fallback: array of shape (q, 2) for intercept-only
                result[group] = np.column_stack(
                    [blup_arr - z * se_arr, blup_arr + z * se_arr]
                )

        return result

    def update(
        self,
        formula: str | None = None,
        data: Any = None,
        **kwargs: Any,
    ) -> CrossedLMEResult:
        """Refit the model with a modified formula, data, or fit arguments.

        Mirrors R's ``update.merMod()``.  Any argument not supplied is taken
        from the original fit.  The *formula* argument supports lme4-style
        dot notation: ``. ~ . + x2`` means "keep the original LHS and RHS,
        then add ``x2`` to the fixed effects".

        Parameters
        ----------
        formula:
            New fixed-effects formula.  May use ``.`` to refer to the
            corresponding part of the original formula.  If ``None``, the
            original formula is reused unchanged.
        data:
            New data frame.  If ``None``, the original data is reused.
        **kwargs:
            Additional keyword arguments forwarded to :func:`interlace.fit`
            (e.g. ``method``, ``groups``, ``random``, ``optimizer``).
            Override the stored values from the original fit.

        Returns
        -------
        CrossedLMEResult
        """
        from interlace import fit as _fit  # local import to avoid circular dep

        stored = self._fit_kwargs.copy()
        orig_formula: str = stored.pop("formula")
        orig_data: Any = stored.pop("data")

        resolved_formula = (
            _expand_dot_formula(formula, orig_formula)
            if formula is not None
            else orig_formula
        )
        resolved_data = data if data is not None else orig_data

        merged = {**stored, **kwargs}
        return _fit(resolved_formula, resolved_data, **merged)
