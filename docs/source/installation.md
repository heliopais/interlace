# Installation

## From PyPI

```bash
pip install interlace-lme
```

Requires **Python ≥ 3.13**.

## From source

```bash
git clone https://github.com/paishe01/interlace.git
cd interlace
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Dependencies

interlace depends on standard scientific Python packages that are likely already installed
in your environment:

| Package | Version |
|---|---|
| statsmodels | ≥ 0.14 |
| formulaic | ≥ 0.6 |
| numpy | ≥ 1.26 |
| scipy | ≥ 1.12 |
| pandas | ≥ 2.0 |
| plotnine | ≥ 0.15 |
| tqdm | ≥ 4.67 |
