"""Load and minimally validate experiment JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: root must be object")
    if "content" not in data:
        raise ValueError(f"{p}: missing 'content' array")
    if not isinstance(data["content"], list):
        raise ValueError(f"{p}: 'content' must be list")
    return data
