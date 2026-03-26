# Installation

## From PyPI

```bash
pip install interlace-lme
```

Requires **Python ≥ 3.13**.

## Optional extras

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
