from . import *

from jsonargparse.typing import register_type
from pathlib import Path
register_type(Path, type_check=lambda v, t: isinstance(v, t))
