"""Phase 1B: one row per (run_id, timestep)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .scene_parse import parse_scene_from_observation


def _obs_index(inner: Dict[str, Any], idx: int) -> str:
    obs = inner.get("observation")
    if not isinstance(obs, list) or idx >= len(obs):
        return ""
    slot = obs[idx]
    if isinstance(slot, str):
        return slot
    return ""


def _err_slice(sd: Dict[str, Any], agent_idx: int) -> Dict[str, Any]:
    errs = sd.get("error") or []
    if agent_idx >= len(errs) or not isinstance(errs[agent_idx], dict):
        return {"validator_n": 0, "format_n": 0, "first_validator_msg": ""}
    er = errs[agent_idx]
    ve = er.get("validator_error") or {}
    fe = er.get("format_error") or {}
    msgs = ve.get("error_message") or []
    return {
        "validator_n": int(ve.get("error_num") or 0),
        "format_n": int(fe.get("error_num") or 0),
        "first_validator_msg": str(msgs[0]) if msgs else "",
    }


def extract_timestep_rows(
    run_id: str, data: Dict[str, Any], include_audit: bool = False
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    conv = data.get("timestep_conversations") or []

    for entry in data.get("content") or []:
        if not isinstance(entry, dict):
            continue
        t = entry.get("timestamp")
        inner = entry.get("content") or {}
        if not isinstance(inner, dict):
            inner = {}

        obs0 = _obs_index(inner, 0)
        obs1 = _obs_index(inner, 1)
        primary_obs = obs0 if obs0.strip() else obs1
        scene_parsed = parse_scene_from_observation(primary_obs)

        acts = entry.get("actions") or []
        a0 = acts[0] if len(acts) > 0 else ""
        a1 = acts[1] if len(acts) > 1 else ""

        sd = entry.get("statistical_data") or {}
        e0 = _err_slice(sd, 0)
        e1 = _err_slice(sd, 1)

        audit_json: Optional[str] = None
        if include_audit:
            for c in conv:
                if isinstance(c, dict) and c.get("timestep") == t:
                    audit_json = json.dumps(
                        {
                            "p0_dialog": c.get("p0_dialog"),
                            "p1_dialog": c.get("p1_dialog"),
                        },
                        ensure_ascii=False,
                    )
                    break

        row: Dict[str, Any] = {
            "run_id": run_id,
            "timestep": t,
            "actions_0": str(a0),
            "actions_1": str(a1),
            "map": str(entry.get("map") or ""),
            "score": sd.get("score", 0),
            "order_list": ";".join(str(x) for x in (entry.get("order_list") or [])),
            "validator_err_0": e0["validator_n"],
            "validator_err_1": e1["validator_n"],
            "format_err_0": e0["format_n"],
            "format_err_1": e1["format_n"],
            "first_validator_msg_0": e0["first_validator_msg"],
            "first_validator_msg_1": e1["first_validator_msg"],
            "scene_raw_p0": obs0,
            "scene_raw_p1": obs1,
            "chef_holds": scene_parsed.get("chef_holds"),
            "asst_holds": scene_parsed.get("asst_holds"),
            "chef_queue_str": scene_parsed.get("chef_queue_str"),
            "asst_queue_str": scene_parsed.get("asst_queue_str"),
            "pot_line": scene_parsed.get("pot_line"),
        }
        if include_audit:
            row["audit_json"] = audit_json
        rows.append(row)
    return rows
