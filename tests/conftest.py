"""Test configuration: ensure project root is on sys.path for imports.

Allows running `pytest` from the repository root without an editable install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
