"""Best-effort regex parse of Scene / Kitchen lines from observation text."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def _obs_str(obs_slot: Any) -> str:
    if isinstance(obs_slot, str):
        return obs_slot
    return ""


def extract_scene_block(obs_text: str) -> str:
    if not obs_text:
        return ""
    m = re.search(r"Scene\s+\d+:\s*(.*?)(?=Kitchen states:|$)", obs_text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) or "").strip() if m else ""


def extract_kitchen_tail(obs_text: str) -> str:
    if not obs_text:
        return ""
    m = re.search(r"Kitchen states:\s*(.*)$", obs_text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) or "").strip() if m else ""


def parse_scene_from_observation(obs_text: str) -> Dict[str, Optional[str]]:
    """
    Returns nullable fields for joins / deltas.
    """
    obs_text = _obs_str(obs_text)
    scene = extract_scene_block(obs_text)
    kitchen = extract_kitchen_tail(obs_text)

    chef_holds = asst_holds = None
    chef_queue = asst_queue = None

    if scene:
        m = re.search(r"<Chef>\s*holds\s*([^.]+)\.", scene, re.IGNORECASE)
        if m:
            chef_holds = m.group(1).strip()
        m = re.search(r"<Assistant>\s*holds\s*([^.]+)\.", scene, re.IGNORECASE)
        if m:
            asst_holds = m.group(1).strip()

        m = re.search(
            r"planned sequence of actions[^<]*for\s+Chef\s+is\s*\[(.*?)\]",
            scene,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            chef_queue = re.sub(r"\s+", " ", m.group(1).strip())
        m = re.search(
            r"planned sequence of actions[^<]*for\s+Assistant\s+is\s*\[(.*?)\]",
            scene,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            asst_queue = re.sub(r"\s+", " ", m.group(1).strip())

    pot_line = None
    if kitchen:
        pot_m = re.search(r"(<pot\d+>[^;\n]*)", kitchen, re.IGNORECASE)
        if pot_m:
            pot_line = pot_m.group(1).strip()

    return {
        "scene_body": scene or None,
        "kitchen_tail": kitchen or None,
        "chef_holds": chef_holds,
        "asst_holds": asst_holds,
        "chef_queue_str": chef_queue,
        "asst_queue_str": asst_queue,
        "pot_line": pot_line,
    }


def queue_atom_count(queue_str: Optional[str]) -> int:
    if not queue_str:
        return 0
    s = queue_str.strip()
    if not s:
        return 0
    return len([x for x in s.split(";") if x.strip()])
