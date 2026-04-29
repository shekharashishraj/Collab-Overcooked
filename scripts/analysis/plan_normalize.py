"""Normalize Chef/Assistant plan strings to atoms and flags."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def strip_role_plan_prefix(plan_raw: str) -> str:
    if not plan_raw:
        return ""
    s = plan_raw.strip()
    s = re.sub(r"^(Chef|Assistant)\s*plan\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    return s


def unwrap_request(plan_line: str) -> str:
    """request('pickup(egg)') -> pickup(egg)"""
    m = re.search(r"request\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", plan_line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return plan_line


def split_atoms(normalized: str) -> List[str]:
    if not normalized:
        return []
    parts = []
    for chunk in normalized.split(";"):
        c = chunk.strip()
        if c:
            parts.append(c)
    return parts


def normalize_plan(plan_raw: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns (atoms, flags).
    """
    base = strip_role_plan_prefix(plan_raw or "")
    base = re.sub(r"\s+", " ", base.replace("\n", " ")).strip()
    flags: Dict[str, Any] = {
        "plan_is_empty": not bool(base),
        "plan_has_request": bool(re.search(r"\brequest\s*\(", base, re.IGNORECASE)),
    }
    # Unwrap outer request() for atomization
    inner = unwrap_request(base) if flags["plan_has_request"] else base
    atoms = split_atoms(inner)
    joined = ";".join(atoms)
    flags["plan_is_wait"] = bool(re.match(r"^wait\s*\(\s*\d+\s*\)$", joined.strip(), re.I)) or joined.strip().lower() == "wait(1)"
    flags["plan_atom_count"] = len(atoms)
    flags["plan_canonical"] = joined
    return atoms, flags


def say_flags(say_raw: str) -> Dict[str, Any]:
    s = (say_raw or "").strip()
    up = s.upper()
    return {
        "say_is_nothing": "[NOTHING]" in up,
        "say_len_chars": len(s),
    }
