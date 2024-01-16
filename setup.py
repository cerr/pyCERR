#!/usr/bin/env python

from setuptools import Extension, setup

setup(
  name = "pycerr",
  packages = ["cerr", "cerr.dataclasses", "cerr.contour", "cerr.radiomics",
              "cerr.utils", "cerr.dcm_export", "cerr.datasets"],
  include_package_data = True,
  package_data = {'cerr.datasets': ['radiomics_settings/*.json',
            'radiomics_settings/IBSIsettings/IBSI2Phase1/*.json',
            'radiomics_phantom_dicom/PAT1/*.dcm',
  			'radiomics_phantom_dicom/PAT2/*.dcm',
  			'radiomics_phantom_dicom/PAT3/*.dcm',
  			'radiomics_phantom_dicom/PAT4/*.dcm']},
  zip_safe = False
)
