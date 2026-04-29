"""Phase 1C: one row per structured communication turn (nested content.content)."""

from __future__ import annotations

from typing import Any, Dict, List

from .plan_normalize import normalize_plan, say_flags


def extract_comm_round_rows(
    run_id: str, data: Dict[str, Any], order_hint: str = ""
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in data.get("content") or []:
        if not isinstance(entry, dict):
            continue
        t = entry.get("timestamp")
        inner = entry.get("content") or {}
        if not isinstance(inner, dict):
            continue
        ol = entry.get("order_list") or []
        order = str(ol[0]) if ol else order_hint

        rounds = inner.get("content")
        if not isinstance(rounds, list):
            continue

        for r_idx, rnd in enumerate(rounds):
            if not isinstance(rnd, list) or not rnd:
                continue
            for turn_idx, blob in enumerate(rnd):
                if not isinstance(blob, dict):
                    continue
                if "agent" not in blob:
                    continue
                agent = int(blob["agent"])
                say_raw = str(blob.get("say") or "")
                plan_raw = str(blob.get("plan") or "")
                analysis_raw = str(blob.get("analysis") or "")
                atoms, pflags = normalize_plan(plan_raw)
                sflags = say_flags(say_raw)
                rows.append(
                    {
                        "run_id": run_id,
                        "timestep": t,
                        "round_idx": r_idx,
                        "turn_in_round": turn_idx,
                        "agent": agent,
                        "order": order,
                        "say_raw": say_raw,
                        "plan_raw": plan_raw,
                        "analysis_raw": analysis_raw,
                        "plan_atoms": ";".join(atoms),
                        "plan_atom_count": pflags.get("plan_atom_count", 0),
                        "plan_is_wait": bool(pflags.get("plan_is_wait")),
                        "plan_has_request": bool(pflags.get("plan_has_request")),
                        "plan_canonical": pflags.get("plan_canonical") or "",
                        "say_is_nothing": bool(sflags.get("say_is_nothing")),
                        "say_len_chars": int(sflags.get("say_len_chars") or 0),
                    }
                )
    return rows
