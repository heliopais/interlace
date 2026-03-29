# Installation

## From PyPI

```bash
pip install interlace-lme
```

Requires **Python ≥ 3.13**.

## Which extras do I need?

Use this decision guide to choose the right install command:

```
Your dataset:
├── n < 1 000 obs AND G < 100 group levels
│   └── pip install interlace-lme       ← base is fine
│
├── n > 10 000 OR G > 500 group levels
│   └── pip install "interlace-lme[cholmod]"   ← CHOLMOD speeds up fitting
│
├── Getting ConvergenceWarning / need to match lme4 output?
│   └── pip install "interlace-lme[bobyqa]"    ← BOBYQA for robustness/parity
│
└── Large data AND convergence issues?
    └── pip install "interlace-lme[cholmod,bobyqa]"
```

See [Performance](performance.md) for detailed benchmarks, and
[FAQ](faq.md#solver-choice-cholmod-vs-default) for solver guidance.

## Optional extras

### CHOLMOD sparse Cholesky

For faster REML optimisation on large models, install the `cholmod` extra:

```bash
pip install "interlace-lme[cholmod]"
```

This adds [scikit-sparse](https://scikit-sparse.readthedocs.io/), which
provides Python bindings to SuiteSparse/CHOLMOD.  When installed, interlace
automatically uses CHOLMOD's symbolic-then-numeric refactorisation path —
performing symbolic analysis once and reusing the sparsity pattern on every
REML iteration.  No code change required; the fast path is detected at
runtime.

### BOBYQA optimizer

For better parity with R/HLMdiag influence diagnostics, install the
`bobyqa` extra:

```bash
pip install "interlace-lme[bobyqa]"
```

This adds [Py-BOBYQA](https://numericalalgorithmsgroup.github.io/pybobyqa/),
a gradient-free trust-region optimizer that matches the algorithm used
by lme4 internally. Use it by passing `optimizer="bobyqa"` to
`interlace.fit()` or any influence function:

```python
import interlace

model = interlace.fit("y ~ x", data=df, groups="firm", optimizer="bobyqa")

from interlace.influence import hlm_influence
infl = hlm_influence(model, optimizer="bobyqa")
```

See the [Changelog](changelog.md) for benchmarked parity improvements.

## From source

```bash
git clone https://github.com/heliopais/interlace.git
cd interlace
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```
