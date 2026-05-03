"""Trace-grounded communication outcome (TGCO) audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


TGCO_EFFECTIVE = "Effective"
TGCO_ASSISTED = "Assisted"
TGCO_REDUNDANT = "Redundant"
TGCO_INEFFECTIVE = "Ineffective"

OBJECT_WORDS = (
    "bell_pepper",
    "sweet_potato",
    "green_bean",
    "boiled_egg",
    "potato_slices",
    "pumpkin_slices",
    "corn_slices",
    "egg",
    "potato",
    "pumpkin",
    "corn",
    "onion",
    "carrot",
    "mushroom",
    "dish",
)

DIRECTIVE_RE = re.compile(
    r"\b(please|can you|could you|would you|you should|make sure|go ahead and)\b",
    re.IGNORECASE,
)
SELF_PLAN_RE = re.compile(r"^\s*(i\b|i'll\b|i will\b|i'm\b|i am\b|we\b|we'll\b)", re.IGNORECASE)
WAIT_RE = re.compile(r"\bwait\b", re.IGNORECASE)


@dataclass(frozen=True)
class RequestUnit:
    timestep: int
    speaker: int
    target_agent: int
    action: str
    obj: str = ""
    location: str = ""
    raw_text: str = ""


@dataclass(frozen=True)
class ActionAtom:
    timestep: int
    action: str
    obj: str = ""
    location: str = ""
    raw_action: str = ""
    index: int = -1
    agent: int = -1


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _norm_obj(obj: str) -> str:
    return re.sub(r"\d+$", "", (obj or "").lower().replace(" ", "_").strip("_"))


def _find_object(text: str) -> str:
    low = text.lower().replace(" ", "_")
    for obj in OBJECT_WORDS:
        if obj in low:
            return obj
    return ""


def _split_clauses(text: str) -> List[str]:
    normalized = _norm_text(text)
    pieces = re.split(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b)\s*", normalized, flags=re.I)
    return [p.strip() for p in pieces if p.strip()]


def parse_request_units(text: str, speaker: int, timestep: int) -> List[RequestUnit]:
    """Extract explicit partner-directed task-action requests from a say message."""
    raw = _norm_text(text)
    if not raw or "[NOTHING]" in raw.upper():
        return []
    if re.match(r"^\s*please\s+wait\b", raw, re.IGNORECASE):
        return []
    if WAIT_RE.search(raw) and not re.search(r"\b(pick|place|put|cook|deliver|get|grab|take)\b", raw, re.I):
        return []
    if SELF_PLAN_RE.match(raw) and not DIRECTIVE_RE.search(raw):
        return []
    if not DIRECTIVE_RE.search(raw):
        return []

    target = 1 - int(speaker)
    units: List[RequestUnit] = []
    last_obj = _find_object(raw)

    for clause in _split_clauses(raw):
        clause_low = clause.lower()
        clause_obj = _find_object(clause) or last_obj
        if clause_obj:
            last_obj = clause_obj

        if re.search(r"\b(pick up|get|grab|take)\b", clause_low):
            location = ""
            if "ingredient dispenser" in clause_low or "ingredient_dispenser" in clause_low:
                location = "ingredient_dispenser"
            elif "counter" in clause_low:
                location = "counter"
            elif "pot" in clause_low:
                location = "pot"
            elif "oven" in clause_low:
                location = "oven"
            units.append(
                RequestUnit(timestep, speaker, target, "pickup", clause_obj, location, raw)
            )
            continue

        if re.search(r"\b(place|put|leave|set)\b", clause_low) and "counter" in clause_low:
            units.append(
                RequestUnit(
                    timestep,
                    speaker,
                    target,
                    "place_obj_on_counter",
                    clause_obj,
                    "counter",
                    raw,
                )
            )
            continue

        if re.search(r"\bput\b", clause_low) and re.search(r"\bpot|oven|blender|chopping board\b", clause_low):
            location = "pot" if "pot" in clause_low else ""
            if "oven" in clause_low:
                location = "oven"
            elif "blender" in clause_low:
                location = "blender"
            elif "chopping board" in clause_low:
                location = "chopping_board"
            units.append(
                RequestUnit(
                    timestep,
                    speaker,
                    target,
                    "put_obj_in_utensil",
                    clause_obj,
                    location,
                    raw,
                )
            )
            continue

        if re.search(r"\b(cook|boil|bake)\b", clause_low):
            location = "pot" if "pot" in clause_low or "boil" in clause_low else ""
            if "oven" in clause_low or "bake" in clause_low:
                location = "oven"
            units.append(RequestUnit(timestep, speaker, target, "cook", clause_obj, location, raw))
            continue

        if re.search(r"\bdeliver|serve\b", clause_low):
            units.append(RequestUnit(timestep, speaker, target, "deliver_soup", clause_obj, "", raw))

    return units


ACTION_RE = re.compile(r"^\s*([A-Za-z_]+)\s*\((.*?)\)\s*$")


def parse_action(action: str, timestep: int = -1, index: int = -1, agent: int = -1) -> ActionAtom:
    raw = _norm_text(action)
    m = ACTION_RE.match(raw)
    if not m:
        return ActionAtom(timestep, raw, raw_action=raw, index=index, agent=agent)
    name = m.group(1)
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    obj = _norm_obj(args[0]) if args else ""
    location = _norm_obj(args[1]) if len(args) > 1 else ""
    if name == "put_obj_in_utensil" and len(args) == 1:
        location = _norm_obj(args[0])
        obj = ""
    if name == "cook" and len(args) == 1:
        location = _norm_obj(args[0])
        obj = ""
    return ActionAtom(timestep, name, obj, location, raw, index, agent)


def iter_total_actions(total_action_list: Sequence[Sequence[Dict[str, Any]]], agent: int) -> Iterable[ActionAtom]:
    if agent >= len(total_action_list):
        return []
    atoms = []
    for idx, item in enumerate(total_action_list[agent] or []):
        if not isinstance(item, dict):
            continue
        atoms.append(
            parse_action(
                str(item.get("action") or ""),
                timestep=int(item.get("timestamp") or -1),
                index=idx,
                agent=agent,
            )
        )
    return atoms


def _action_matches_request(request: RequestUnit, action: ActionAtom) -> bool:
    if request.action != action.action:
        return False
    if request.obj and action.obj and request.obj != action.obj:
        return False
    if request.location and action.location and request.location != action.location:
        return False
    if request.action == "pickup" and request.obj and not action.obj:
        return False
    return True


def _find_prior_match(request: RequestUnit, actions: Iterable[ActionAtom]) -> Optional[ActionAtom]:
    prior = [
        action
        for action in actions
        if action.timestep <= request.timestep and _action_matches_request(request, action)
    ]
    return prior[-1] if prior else None


def _find_window_match(request: RequestUnit, actions: Iterable[ActionAtom], window: int) -> Optional[ActionAtom]:
    for action in actions:
        if request.timestep < action.timestep <= request.timestep + window:
            if _action_matches_request(request, action):
                return action
    return None


def validator_errors_between(
    content: Sequence[Dict[str, Any]], target_agent: int, start_timestep: int, end_timestep: int
) -> int:
    total = 0
    for entry in content or []:
        if not isinstance(entry, dict):
            continue
        timestep = entry.get("timestamp")
        if timestep is None or not (start_timestep < int(timestep) <= end_timestep):
            continue
        errors = ((entry.get("statistical_data") or {}).get("error") or [])
        if target_agent >= len(errors) or not isinstance(errors[target_agent], dict):
            continue
        validator = errors[target_agent].get("validator_error") or {}
        total += int(validator.get("error_num") or 0)
    return total


def _base_row(request: RequestUnit, source_file: str = "") -> Dict[str, Any]:
    return {
        "source_file": source_file,
        "message_timestep": request.timestep,
        "speaker": request.speaker,
        "target_agent": request.target_agent,
        "request_action": request.action,
        "request_object": request.obj,
        "request_location": request.location,
        "request_text": request.raw_text,
        "matched_total_action_agent": "",
        "matched_total_action_index": "",
        "matched_action": "",
        "matched_action_timestep": "",
        "validator_errors_between": 0,
        "tgco_label": TGCO_INEFFECTIVE,
        "evidence_source": "total_action_list",
    }


def classify_request(
    request: RequestUnit,
    total_action_list: Sequence[Sequence[Dict[str, Any]]],
    content: Sequence[Dict[str, Any]],
    window: int = 5,
    consumed_matches: Optional[Set[Tuple[int, int]]] = None,
    source_file: str = "",
) -> Dict[str, Any]:
    """Classify one request against the addressed partner's accepted action trace."""
    consumed_matches = consumed_matches or set()
    row = _base_row(request, source_file)
    target_actions = list(iter_total_actions(total_action_list, request.target_agent))

    prior_match = _find_prior_match(request, target_actions)
    if prior_match:
        row.update(
            {
                "matched_total_action_agent": prior_match.agent,
                "matched_total_action_index": prior_match.index,
                "matched_action": prior_match.raw_action,
                "matched_action_timestep": prior_match.timestep,
                "tgco_label": TGCO_REDUNDANT,
            }
        )
        return row

    match = _find_window_match(request, target_actions, window)
    if not match:
        return row

    row.update(
        {
            "matched_total_action_agent": match.agent,
            "matched_total_action_index": match.index,
            "matched_action": match.raw_action,
            "matched_action_timestep": match.timestep,
        }
    )
    if (match.agent, match.index) in consumed_matches:
        row["tgco_label"] = TGCO_REDUNDANT
        return row

    errors = validator_errors_between(content, request.target_agent, request.timestep, match.timestep)
    row["validator_errors_between"] = errors
    row["tgco_label"] = TGCO_ASSISTED if errors else TGCO_EFFECTIVE
    return row


def _iter_say_messages(data: Dict[str, Any]) -> Iterable[Tuple[int, int, str]]:
    for entry in data.get("content") or []:
        if not isinstance(entry, dict):
            continue
        timestep = int(entry.get("timestamp") or 0)
        inner = entry.get("content") or {}
        rounds = inner.get("content") if isinstance(inner, dict) else []
        if not isinstance(rounds, list):
            continue
        for rnd in rounds:
            if not isinstance(rnd, list):
                continue
            for blob in rnd:
                if not isinstance(blob, dict) or "agent" not in blob:
                    continue
                yield timestep, int(blob["agent"]), str(blob.get("say") or "")


def audit_run(data: Dict[str, Any], source_file: str = "", window: int = 5) -> List[Dict[str, Any]]:
    """Emit one evidence row per parsed request-action unit for a run."""
    rows: List[Dict[str, Any]] = []
    consumed: Set[Tuple[int, int]] = set()
    total_action_list = data.get("total_action_list") or [[], []]
    content = data.get("content") or []

    for timestep, speaker, say in _iter_say_messages(data):
        for request in parse_request_units(say, speaker=speaker, timestep=timestep):
            row = classify_request(
                request,
                total_action_list=total_action_list,
                content=content,
                window=window,
                consumed_matches=consumed,
                source_file=source_file,
            )
            if row["tgco_label"] in {TGCO_EFFECTIVE, TGCO_ASSISTED}:
                consumed.add((int(row["matched_total_action_agent"]), int(row["matched_total_action_index"])))
            rows.append(row)

    return rows
