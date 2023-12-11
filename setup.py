#!/usr/bin/env python

from setuptools import Extension, setup

setup(
  name = "pycerr",
  packages = ["cerr", "cerr.dataclasses", "cerr.contour", "cerr.radiomics",
              "cerr.utils", "cerr.dcm_export", "cerr.datasets"],
  include_package_data = True,
  package_data={
    # Include all the files in "datasets"
    "cerr.datasets": ["*dcm", "*.txt"]
	},       
  zip_safe = False
)
