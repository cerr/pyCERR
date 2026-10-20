"""Regenerate every documentation figure.

registration.py and registration_qa.py need the ``ants`` extra, and
registration_qa.py also the ``viewer`` extra.

Usage::

    python docs/_figure_scripts/make_all.py              # all pages
    python docs/_figure_scripts/make_all.py dvh roe      # selected pages
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = ["quickstart", "tutorial_end_to_end", "planc", "dvh", "roe", "radiomics",
         "texture", "segmentation", "body_outline", "structure_consensus", "seg_metrics",
         "ai_pipeline", "registration", "registration_qa"]

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    for page in (sys.argv[1:] or PAGES):
        print("=== %s ===" % page)
        runpy.run_path(os.path.join(HERE, page + ".py"), run_name="__main__")
