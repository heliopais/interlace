"""anova(): likelihood-ratio test for comparing nested linear mixed models.

Mimics lme4's anova.merMod() output format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.stats as stats

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult


def anova(m1: CrossedLMEResult, m2: CrossedLMEResult) -> Any:
    """Compare two nested linear mixed models via likelihood-ratio test.

    Both models must have been fitted with ``method='ML'``.  When models
    differ in their fixed effects, REML-fitted models cannot be compared by
    LRT because the REML criterion depends on the fixed-effect structure.

    Parameters
    ----------
    m1, m2:
        Two ``CrossedLMEResult`` objects. Order does not matter; the function
        sorts them by number of parameters (smaller model first).

    Returns
    -------
    pandas.DataFrame
        Two-row table with columns matching lme4's anova output::

            Df  AIC  BIC  logLik  deviance  Chisq  Chi Df  Pr(>Chisq)

        The first row corresponds to the simpler model; ``Chisq``,
        ``Chi Df``, and ``Pr(>Chisq)`` are ``NaN`` for it.

    Raises
    ------
    ValueError
        If either model was fitted with ``method='REML'``.
    """
    if m1.method == "REML" or m2.method == "REML":
        reml_model = "m1" if m1.method == "REML" else "m2"
        raise ValueError(
            f"{reml_model} was fitted with REML. "
            "Models compared by anova() must be fitted with method='ML'. "
            "Likelihood-ratio tests require ML log-likelihoods; REML "
            "log-likelihoods are not comparable between models with "
            "different fixed-effect structures."
        )

    import pandas as pd

    # Sort so smaller model (fewer params) is first
    models = sorted([m1, m2], key=lambda m: m.nparams)
    m_small, m_large = models

    df_small = m_small.nparams
    df_large = m_large.nparams
    chi_df = df_large - df_small

    chisq = 2.0 * (m_large.llf - m_small.llf)
    # Protect against floating-point noise giving a marginally negative stat
    chisq = max(chisq, 0.0)
    pvalue = float(stats.chi2.sf(chisq, df=chi_df))

    rows = [
        {
            "Df": df_small,
            "AIC": m_small.aic,
            "BIC": m_small.bic,
            "logLik": m_small.llf,
            "deviance": -2.0 * m_small.llf,
            "Chisq": np.nan,
            "Chi Df": np.nan,
            "Pr(>Chisq)": np.nan,
        },
        {
            "Df": df_large,
            "AIC": m_large.aic,
            "BIC": m_large.bic,
            "logLik": m_large.llf,
            "deviance": -2.0 * m_large.llf,
            "Chisq": chisq,
            "Chi Df": float(chi_df),
            "Pr(>Chisq)": pvalue,
        },
    ]

    return pd.DataFrame(rows)
