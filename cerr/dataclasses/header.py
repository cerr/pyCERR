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

