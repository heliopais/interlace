"""Sphinx configuration for interlace documentation."""

import sys
from pathlib import Path

# Make the package importable without installing it
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# -- Project information -------------------------------------------------------
project = "interlace"
copyright = "2026, Helio Pais and interlace contributors"
author = "Helio Pais and interlace contributors"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_external_toc",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

# External TOC
external_toc_path = "_toc.yml"

# MyST-NB (notebook execution + Markdown support)
myst_enable_extensions = ["colon_fence"]
nb_execution_timeout = 300  # seconds; hlm_influence runs n refits

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

html_theme_options = {
    "light_css_variables": {
        "font-stack": "'Lora', Georgia, 'Times New Roman', serif",
        "font-stack--monospace": "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace",
    },
    "dark_css_variables": {
        "font-stack": "'Lora', Georgia, 'Times New Roman', serif",
        "font-stack--monospace": "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/heliopais/interlace",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16" height="1em" width="1em" '
                'xmlns="http://www.w3.org/2000/svg">'
                '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59'
                ".4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49"
                "-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63"
                "-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51"
                "-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2"
                "-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68"
                " 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08"
                " 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25"
                ".54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013"
                ' 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
            ),
            "class": "",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "github-link.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
