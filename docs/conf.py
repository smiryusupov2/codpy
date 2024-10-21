# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))

parent_path = os.path.dirname(__file__)
if parent_path not in sys.path: sys.path.append(parent_path)
parent_path = os.path.dirname(parent_path)
codpy_path = os.path.join(parent_path,"src","codpy")
if codpy_path not in sys.path: sys.path.append(codpy_path)
examples_path = os.path.join(parent_path,"examples")
if examples_path not in sys.path: sys.path.append(examples_path)

project = "CodPy"
copyright = "2024, Jean-Marc MERCIER, Shohruh MIRYUSUPOV"
author = "Jean-Marc MERCIER, Shohruh MIRYUSUPOV"
release = "0.1.11"

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
    "exclude-members": "__weakref__",
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples/",  # Directory where .py scripts are stored
    "gallery_dirs": "gallery_examples",  # Directory to save generated HTML and notebooks
    "filename_pattern": r".*\.py",  # Process all Python files
    "backreferences_dir": "gen_modules/backreferences",
    "within_subsection_order": "ExampleTitleSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "notebook_images": True,  # Include images in generated notebooks
}
