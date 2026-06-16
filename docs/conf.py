# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import shutil

sys.path.insert(0, os.path.abspath('.'))
pycerrPath = os.path.abspath('..')
sys.path.insert(0, pycerrPath)

# Copy version file
tmpVersionFie = os.path.join(pycerrPath, '.github/workflows/_version_tmp.py')
versionFile = os.path.join(pycerrPath, 'cerr/_version.py')
shutil.copyfile(tmpVersionFie, versionFile)

from cerr import dvh
from cerr import plan_container as pc

project = 'pyCERR'
copyright = 'GNU GPL v3'
author = 'Aditya Apte, Aditi Iyer, Eve LoCastro, Joseph Deasy'

version = '0.1'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',      # Supports Google / Numpy docstring
    'sphinx.ext.autodoc',       # Documentation from docstrings
    'sphinx.ext.doctest',       # Test snippets in documentation
    'sphinx.ext.todo',          # to-do syntax highlighting
    'sphinx.ext.ifconfig',      # Content based configuration
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Mock heavy / display-dependent dependencies so the docs build on a headless
# CI runner that has no GPU, Qt display, napari or VTK installed.
autodoc_mock_imports = [
    'numpy',
    'PyQt5', 'sip', 'qtpy', 'pyqtgraph',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qtagg',
    'pyvista', 'pyvistaqt', 'vtk',
    'napari', 'magicgui', 'superqt',
    'ipywidgets', 'IPython',
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

#source_suffix = ['.rst']

autodoc_default_options = {
    'members':         True,
    'member-order':    'bysource'
}
