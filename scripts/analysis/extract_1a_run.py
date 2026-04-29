"""Phase 1A: one row per experiment JSON file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .order_levels import order_level


def _parse_agents(data: Dict[str, Any]) -> Tuple[str, str, str, str]:
    p0m = p1m = p0t = p1t = ""
    for a in data.get("agents") or []:
        if not isinstance(a, dict):
            continue
        if a.get("player") == "P0" or a.get("index") == 0:
            p0m = str(a.get("model") or "")
            p0t = str(a.get("agent_type") or "")
        if a.get("player") == "P1" or a.get("index") == 1:
            p1m = str(a.get("model") or "")
            p1t = str(a.get("agent_type") or "")
    return p0m, p1m, p0t, p1t


def infer_order(data: Dict[str, Any]) -> str:
    for e in data.get("content") or []:
        if not isinstance(e, dict):
            continue
        ol = e.get("order_list") or []
        if ol:
            return str(ol[0])
    return ""


def extract_run_row(data: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    path = Path(source_path)
    rel = str(path.resolve())
    run_hash = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]
    order = infer_order(data)
    fin = data.get("total_order_finished") or []
    sc = int(data.get("total_score") or 0)
    success = bool(fin) or sc > 0
    content = data.get("content") or []
    horizon_len = len(content) if isinstance(content, list) else 0
    ts = data.get("total_timestamp") or []
    if isinstance(ts, list) and len(ts) > horizon_len:
        horizon_len = len(ts)

    p0m, p1m, p0t, p1t = _parse_agents(data)
    return {
        "run_id": run_hash,
        "file_stem": path.stem,
        "source_path": rel,
        "order": order,
        "order_level": order_level(order),
        "success": success,
        "total_score": sc,
        "horizon_len": horizon_len,
        "p0_model": p0m,
        "p1_model": p1m,
        "p0_agent_type": p0t,
        "p1_agent_type": p1t,
    }
