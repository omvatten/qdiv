# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the package root to sys.path so autodoc can find qdiv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# -- Project information ------------------------------------------------------
project = 'qdiv'
author = 'Oskar Modin'
release = '4.0.0'  # Match your package version

# -- General configuration ----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',          # Markdown support
    'nbsphinx',             # Jupyter notebooks
    'sphinx.ext.mathjax',   # <-- REQUIRED for rendering math in HTML
]


# Execute notebooks automatically when safe; disable on RTD if needed
nbsphinx_execute = 'auto'   # options: 'always', 'auto', 'never'
# If you expect occasional execution errors but want HTML built anyway:
# nbsphinx_allow_errors = True




myst_enable_extensions = [
    "dollarmath",  # enables $...$ and $$...$$ in Markdown
    "amsmath",     # optional, equation environments
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False, #This was True in original implementation
    "private-members": False,      # skip _private
    'imported-members': False,
    "inherited-members": False,    # skip inherited methods unless you need them
    'show-inheritance': False, #This was True in original implementation
}

jupyter_execute_notebooks = "auto" 
autodoc_member_order = 'bysource'

# Generate autosummary pages automatically
autosummary_generate = True

# Napoleon settings for docstring style
napoleon_google_docstring = False #Consider setting to False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True


# Mock heavy dependencies if needed (optional)
# autodoc_mock_imports = ["numba", "pandas", "matplotlib", "tqdm"]

# Templates path
templates_path = ['_templates']

# Patterns to ignore
#Keep build and checkpoints out of the source tree
exclude_patterns = ["_build", "**/.ipynb_checkpoints/**"]

# -- Options for HTML output --------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}

# -- Additional options -------------------------------------------------------
# Show type hints in the docs
autodoc_typehints = 'description'

# Include __init__ docstrings in class docs
autoclass_content = 'both'
