#!/usr/bin/env python

from setuptools import Extension, setup

setup(
  name = "pycerr",
  packages = ["cerr", "cerr.dataclasses", "cerr.contour", "cerr.radiomics",
              "cerr.utils", "cerr.dcm_export", "cerr.datasets",
              "cerr.registration", "cerr.viewer", "cerr.viewer.pycerr_gui",
              "cerr.roe", "cerr.imrtp", "cerr.uromt", "cerr.mri_metrics",
              "cerr.ai_models"],
  include_package_data = True,
  package_data = {'cerr.datasets': ['radiomics_settings/*.json',
                            'radiomics_settings/IBSIsettings/IBSI2Phase1/*.json',
                            'radiomics_phantom_dicom/PAT1/*.dcm',
                            'radiomics_phantom_dicom/PAT2/*.dcm',
                            'radiomics_phantom_dicom/PAT3/*.dcm',
                            'radiomics_phantom_dicom/PAT4/*.dcm'],
                  'cerr.registration': ['settings/*.txt'],
                  'cerr.viewer.pycerr_gui': ['CERR_logo.ico',
                                             'CERR_logo.icns',
                                             'CERR_logo.png'],
                  },
  zip_safe = False
)
