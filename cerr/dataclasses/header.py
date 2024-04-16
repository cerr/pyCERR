"""header module.

The header module defines metadata for header.
The header metadata are useful to keep track of pyCERR
version used in creation of plan container object as well as
date stamp when it is saved.

"""

from dataclasses import dataclass, field
from datetime import datetime, date
from cerr._version import version as cerrVersion

@dataclass
class Header:
    dt = datetime.now()
    dateCreated: str = dt.strftime("%Y%m%d")
    dateLastSaved: str = ""
    writer: str = ""
    version: str = cerrVersion

