"""Pytest configuration and path setup for local imports.

Ensures the repository root is on `sys.path` so tests can import top-level
packages like `models` and `utils` without installing the project.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

