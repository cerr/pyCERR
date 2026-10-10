"""Regenerate every documentation figure.

Usage::

    python docs/_figure_scripts/make_all.py              # all pages
    python docs/_figure_scripts/make_all.py dvh roe      # selected pages
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = ["quickstart", "tutorial_end_to_end", "planc", "dvh", "roe", "radiomics",
         "segmentation", "registration"]

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    for page in (sys.argv[1:] or PAGES):
        print("=== %s ===" % page)
        runpy.run_path(os.path.join(HERE, page + ".py"), run_name="__main__")
