#!/usr/bin/env python

from setuptools import Extension, setup

setup(
  name = "pycerr",
  packages = ["cerr", "cerr.dataclasses", "cerr.contour", "cerr.radiomics",
              "cerr.utils", "cerr.dcm_export", "cerr.datasets"],
  package_data={
    # Include all the files in "datasets"
    "cerr.datasets": ["*"]
	},       
  zip_safe = False
)
