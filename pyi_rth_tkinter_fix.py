"""Point a frozen executable at the bundled Tcl/Tk script libraries."""

from __future__ import annotations

import os
from pathlib import Path
import sys


bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
os.environ["TCL_LIBRARY"] = str(bundle_root / "_tcl_data")
os.environ["TK_LIBRARY"] = str(bundle_root / "_tk_data")
