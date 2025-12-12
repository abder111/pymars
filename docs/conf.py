# Configuration file for the Sphinx documentation builder

import os
import sys

# Add the parent directory to the path so we can import pymars
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'PyMARS'
copyright = '2025, ES-SAFI ABDERRAHMAN, LAMGHARI YASSINE, CHAIBOU SAIDOU ABDOULAYE'
author = 'ES-SAFI ABDERRAHMAN, LAMGHARI YASSINE, CHAIBOU SAIDOU ABDOULAYE'
release = '0.1.0'
version = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

# Bibtex configuration
bibtex_bibfiles = ['references.bib']

# Templates path
templates_path = ['_templates']

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source suffix
source_suffix = '.rst'

# Master document
master_doc = 'index'

# Pygments style (syntax highlighting)
pygments_style = 'sphinx'

# HTML theme
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'analytics_id': '',
    'canonical_url': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'titles_only': False,
}

# HTML static path
html_static_path = ['_static']

# HTML favicon
html_favicon = None

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# LaTeX configuration
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{commath}
''',
}

latex_documents = [
    (master_doc, 'PyMARS.tex', 'PyMARS Documentation',
     author, 'manual'),
]

# Mathjax configuration
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# PDF output
pdf_documents = [
    (master_doc, 'PyMARS', 'PyMARS Documentation', author),
]
