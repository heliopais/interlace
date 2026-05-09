"""One-shot UserWarning when sksparse/CHOLMOD is unavailable on a hard path.

CHOLMOD is materially faster than the SuperLU fallback on the random-slopes
LMM and Laplace GLMM paths.  When sksparse is missing we silently fall back,
which leaves users sitting in the slow path without knowing.  This module
emits a single ``UserWarning`` per process pointing at the ``[fast]`` extra.
"""

from __future__ import annotations

import warnings

_warned: bool = False

_MESSAGE = (
    "interlace: scikit-sparse (CHOLMOD) is not installed. "
    "{context} runs ~3-10x slower without it. "
    "Install with: pip install 'interlace-lme[fast]'"
)


def maybe_warn_slow_path(context: str) -> None:
    """Emit a one-shot ``UserWarning`` if sksparse is unavailable.

    Parameters
    ----------
    context:
        Short description of the fit being run, e.g. ``"Random-slopes LMM"``
        or ``"GLMM (Laplace)"``.  Inserted into the warning message.
    """
    global _warned
    if _warned:
        return
    from interlace.profiled_reml import _try_cholmod

    if _try_cholmod() is not None:
        return
    _warned = True
    warnings.warn(
        _MESSAGE.format(context=context),
        UserWarning,
        stacklevel=3,
    )


def _reset_for_tests() -> None:
    """Reset the once-per-process latch.  Test-only helper."""
    global _warned
    _warned = False
