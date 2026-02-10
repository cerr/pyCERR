from dataclasses import dataclass, field
import numpy as np
import os
from pydicom import dcmread
from cerr.dataclasses import scan as scn
from cerr.utils import uid
import json

def get_empty_list():
    return []
def get_empty_np_array():
    return np.empty((0,0,0))

@dataclass
class IM:
    IMSetup: dict = field(default_factory=list)
    IMDosimetry: dict = field(default_factory=list)
    IMUID: str = ""
