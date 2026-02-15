from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def ensure_src_on_path() -> None:
    """Allow running repo scripts without installing the package."""

    src_str = str(SRC_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

