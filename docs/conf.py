# -*- coding: utf-8 -*-

import os
import sys

cerrPath = os.path.join(os.getcwd(), "..")
sys.path.insert(0, cerrPath)
import cerr._version as ver
release = ver.version

# -- Project information -----------------------------------------------------

project = 'pyCERR'
copyright = 'GNU GPL v3'
author = 'Aditya Apte, Aditi Iyer, Eve LoCastro, Joseph Deasy'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    'myst_nb',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinxemoji.sphinxemoji',
]

sphinxemoji_style = 'twemoji'
sphinxemoji_source = 'https://unpkg.com/twemoji@latest/dist/twemoji.min.js'

autodoc_member_order = 'bysource'
suppress_warnings = [
    'autosectionlabel.*',
    'mystnb.nbcell',
    'myst.header',
    'myst.strikethrough',
]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
nb_execution_mode = 'off'
# nb_remove_code_outputs = True

master_doc = 'index'
copybutton_remove_prompts = True

exclude_patterns = [
    '.ipynb_checkpoints',
    'README.md',
    'conf.py',
    '.git',
]

pygments_style = 'sphinx'

# -- HTML configuration ---------------------------------------------------

html_codeblock_linenos_style = 'table'
html_base_url = 'https://github.com/cerr/pyCERR/'
html_logo = 'cerr_logo.png'
html_theme = 'furo'
html_show_sourcelink = True
html_last_updated_fmt = '%Y-%m-%d %H:%M:%S'
html_title = 'pyCERR docs'
