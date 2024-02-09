from dataclasses import dataclass, field
from typing import List
import numpy as np
import os
from pydicom import dcmread
from cerr.dataclasses import scan as scn
from cerr.utils import uid
import json

@dataclass
class Deform:
    baseScanUID: str = ""
    movScanUID: str = ""
    algorithm: str = ""
    algorithmParams: dict = field(default_factory=dict)
    deformParams: dict = field(default_factory=dict)
    deformUID: str = ""
    registrationTool: str = ""
    deformOutFileType: str = ""
    deformOutFilePath: str = ""
