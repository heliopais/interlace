"""Profile Sleepstudy and CBPP fits to confirm the hot spots in epic
``interlace-81ru``.

Outputs (under ``docs/perf/profiles/``):
  - ``<dataset>__<config>.prof``     -- cProfile dump (binary)
  - ``<dataset>__<config>_top.txt``  -- top-15 cumulative + self-time table
  - ``<dataset>__<config>.svg``      -- py-spy flamegraph (if py-spy works)
  - ``summary.json``                 -- timings + iter/call counts per run

Run with the repo root as CWD::

    uv run python docs/perf/profile_fits.py

Re-runnable: it overwrites the ``docs/perf/profiles/`` artefacts.
"""
# ruff: noqa: E501
from __future__ import annotations

import cProfile
import io
import json
import pstats
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
FIXTURES = REPO_ROOT / "tests" / "fixtures"
OUT_DIR = REPO_ROOT / "docs" / "perf" / "profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import interlace  # noqa: E402
from interlace import glmm_laplace, profiled_reml  # noqa: E402

# ---------------------------------------------------------------------------
# Fit specs (mirror manuscript benchmarks/bench_one.py)
# ---------------------------------------------------------------------------

def _fit_sleepstudy(df: pd.DataFrame) -> Any:
    return interlace.fit(
        "Reaction ~ Days", data=df, random=["(1 + Days | Subject)"],
    )


def _fit_cbpp(df: pd.DataFrame) -> Any:
    return interlace.glmer(
        "proportion ~ period",
        data=df,
        family="binomial",
        groups="herd",
        weights=np.array(df["size"], dtype=float),
    )


SPECS: dict[str, tuple[str, Callable[[pd.DataFrame], Any]]] = {
    "sleepstudy": ("lme4_sleepstudy_data.csv", _fit_sleepstudy),
    "cbpp":       ("glmm_cbpp_data.csv",       _fit_cbpp),
}


def _load(name: str) -> pd.DataFrame:
    csv_name, _ = SPECS[name]
    df = pd.read_csv(FIXTURES / csv_name)
    if name == "cbpp":
        df["period"] = df["period"].astype(str)
        df["herd"] = df["herd"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Instrumentation: wrap the functions we care about with call counters.
# ---------------------------------------------------------------------------

class CallCounter:
    """Context manager that counts calls to a list of (module, attr) targets."""

    def __init__(self, targets: list[tuple[Any, str]]) -> None:
        self.targets = targets
        self.counts: dict[str, int] = {}
        self._originals: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> CallCounter:
        for mod, attr in self.targets:
            orig = getattr(mod, attr)
            key = f"{mod.__name__}.{attr}"
            self.counts[key] = 0
            self._originals.append((mod, attr, orig))

            def make_wrapper(f: Any, k: str) -> Any:
                def wrapper(*a: Any, **kw: Any) -> Any:
                    self.counts[k] += 1
                    return f(*a, **kw)
                wrapper.__name__ = getattr(f, "__name__", k)
                return wrapper

            setattr(mod, attr, make_wrapper(orig, key))
        return self

    def __exit__(self, *exc: Any) -> None:
        for mod, attr, orig in self._originals:
            setattr(mod, attr, orig)


# ---------------------------------------------------------------------------
# CHOLMOD toggle: monkeypatch _try_cholmod to return None so the SuperLU
# fallback path is exercised. This is what the manuscript's 7x/11x ratios
# reflect (sksparse was not in the baseline dep set when those ran).
# ---------------------------------------------------------------------------

def _force_superlu(active: bool) -> Callable[[], None]:
    if not active:
        return lambda: None
    orig = profiled_reml._try_cholmod
    profiled_reml._try_cholmod = lambda: None  # type: ignore[assignment]

    def restore() -> None:
        profiled_reml._try_cholmod = orig  # type: ignore[assignment]

    return restore


def _detect_cholmod_used() -> bool:
    return profiled_reml._try_cholmod() is not None


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------

def _run_one(name: str, force_superlu: bool, n_warmup: int = 1, n_reps: int = 5) -> dict[str, Any]:
    df = _load(name)
    _, fit_fn = SPECS[name]
    config = "superlu" if force_superlu else "auto"
    label = f"{name}__{config}"

    restore = _force_superlu(force_superlu)
    try:
        # Warmup (no instrumentation, JIT caches etc.)
        for _ in range(n_warmup):
            fit_fn(df)

        # ---- Wall-clock timing ----
        times: list[float] = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            fit_fn(df)
            times.append(time.perf_counter() - t0)

        # ---- Call counters (one fit) ----
        targets = [
            (profiled_reml, "reml_objective"),
            (glmm_laplace, "_pirls"),
            (glmm_laplace, "_laplace_objective"),
            (glmm_laplace, "_laplace_objective_profiled"),
            (spla, "spsolve"),
            (spla, "splu"),
        ]
        with CallCounter(targets) as cc:
            fit_fn(df)
        counts = dict(cc.counts)

        # ---- cProfile (one fit) ----
        prof_path = OUT_DIR / f"{label}.prof"
        top_path = OUT_DIR / f"{label}_top.txt"
        prof = cProfile.Profile()
        prof.enable()
        fit_fn(df)
        prof.disable()
        prof.dump_stats(str(prof_path))

        buf_cum = io.StringIO()
        ps_cum = pstats.Stats(prof, stream=buf_cum).sort_stats("cumulative")
        ps_cum.print_stats(25)
        buf_self = io.StringIO()
        ps_self = pstats.Stats(prof, stream=buf_self).sort_stats("tottime")
        ps_self.print_stats(25)
        with open(top_path, "w") as f:
            f.write(f"# Profile: {label}\n\n")
            f.write("## Top 25 by cumulative time\n\n")
            f.write(buf_cum.getvalue())
            f.write("\n\n## Top 25 by self (tottime)\n\n")
            f.write(buf_self.getvalue())

        return {
            "dataset": name,
            "config": config,
            "n_reps": n_reps,
            "times_s": times,
            "median_s": float(np.median(times)),
            "min_s": float(np.min(times)),
            "call_counts": counts,
            "cholmod_available": _detect_cholmod_used() if not force_superlu else False,
            "prof": str(prof_path.relative_to(REPO_ROOT)),
            "top_txt": str(top_path.relative_to(REPO_ROOT)),
        }
    finally:
        restore()


# ---------------------------------------------------------------------------
# py-spy flamegraphs
# ---------------------------------------------------------------------------

def _pyspy_record(name: str, force_superlu: bool, duration_s: int = 8) -> str | None:
    """Run the fit in a tight loop under py-spy. Returns SVG path or None."""
    if shutil.which("py-spy") is None:
        return None
    config = "superlu" if force_superlu else "auto"
    label = f"{name}__{config}"
    svg_path = OUT_DIR / f"{label}.svg"

    inner = (
        "import sys, time;\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r});\n"
        f"import importlib, interlace, scipy.sparse.linalg as spla;\n"
        f"from interlace import profiled_reml;\n"
        f"FORCE = {force_superlu!r};\n"
        f"import pandas as pd, numpy as np;\n"
        f"df = pd.read_csv({str(FIXTURES / SPECS[name][0])!r});\n"
        + ("df['period']=df['period'].astype(str); df['herd']=df['herd'].astype(str);\n" if name == 'cbpp' else "")
        + "if FORCE:\n"
        "    profiled_reml._try_cholmod = lambda: None;\n"
        + (
            "fit = lambda: interlace.fit('Reaction ~ Days', data=df, random=['(1 + Days | Subject)'])\n"
            if name == "sleepstudy"
            else "fit = lambda: interlace.glmer('proportion ~ period', data=df, family='binomial', groups='herd', weights=np.array(df['size'], dtype=float))\n"
        )
        + "fit()\n"  # warmup
        + f"end = time.perf_counter() + {duration_s};\n"
        + "while time.perf_counter() < end: fit()\n"
    )

    cmd = [
        "py-spy", "record",
        "-o", str(svg_path),
        "-r", "200",   # 200 Hz sampling
        "--",
        sys.executable, "-c", inner,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"[py-spy] {label} failed:\n{res.stderr}\n")
        return None
    return str(svg_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    runs = [
        ("sleepstudy", False),
        ("sleepstudy", True),   # SuperLU fallback (manuscript baseline)
        ("cbpp",       False),  # CHOLMOD has no effect on GLMM path
    ]

    summary = {
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "interlace": getattr(interlace, "__version__", "unknown"),
            "cholmod_available": _detect_cholmod_used(),
        },
        "runs": [],
    }

    for name, force_superlu in runs:
        sys.stderr.write(f"\n=== {name} (force_superlu={force_superlu}) ===\n")
        out = _run_one(name, force_superlu)
        # Append py-spy svg
        svg = _pyspy_record(name, force_superlu)
        if svg:
            out["flamegraph"] = svg
        summary["runs"].append(out)
        sys.stderr.write(
            f"  median={out['median_s']:.3f}s  min={out['min_s']:.3f}s  "
            f"reml_obj={out['call_counts'].get('interlace.profiled_reml.reml_objective', 0)}  "
            f"pirls={out['call_counts'].get('interlace.glmm_laplace._pirls', 0)}  "
            f"spsolve={out['call_counts'].get('scipy.sparse.linalg.spsolve', 0)}\n"
        )

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    sys.stderr.write(f"\nWrote {OUT_DIR / 'summary.json'}\n")


if __name__ == "__main__":
    main()
