# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../"))


project = "CodPy"
copyright = "2024, Jean-Marc MERCIER, Shohruh MIRYUSUPOV"
author = "Jean-Marc MERCIER, Shohruh MIRYUSUPOV"
release = "0.0.10"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Enables support for NumPy and Google style docstrings
    "sphinx.ext.mathjax",
    "sphinx_togglebutton",
    "sphinx_math_dollar",
]


mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}


mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,  # This line tells Sphinx to skip undocumented members
    "private-members": False,  # This line tells Sphinx to skip private members
    "special-members": "__init__",
    "exclude-members": "__weakref__, __init__",
}
