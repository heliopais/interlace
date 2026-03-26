"""Sphinx configuration for interlace documentation."""

import sys
from pathlib import Path

# Make the package importable without installing it
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# -- Project information -------------------------------------------------------
project = "interlace"
copyright = "2025, interlace contributors"
author = "interlace contributors"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_external_toc",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# External TOC
external_toc_path = "_toc.yml"

# MyST (Markdown support)
myst_enable_extensions = ["colon_fence"]

# autodoc
autodoc_typehints = "description"
autodoc_member_order = "bysource"
add_module_names = False

# Napoleon (NumPy-style docstrings)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_title = "interlace"
