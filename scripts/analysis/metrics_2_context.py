"""Phase 2: contextualization (instruction following, scene deltas, light grounding)."""

from __future__ import annotations

import re
from typing import List

import pandas as pd

from .scene_parse import queue_atom_count


def _verb(atom: str) -> str:
    atom = (atom or "").strip()
    m = re.match(r"^([a-zA-Z_]+)\s*\(", atom)
    return (m.group(1) if m else atom).lower()


def _actions_window(step_df: pd.DataFrame, run_id, t0, agent: int, delta: int) -> List[str]:
    out = []
    for dt in range(0, delta + 1):
        t = int(t0) + dt
        r = step_df[(step_df["run_id"] == run_id) & (step_df["timestep"] == t)]
        if r.empty:
            out.append("")
            continue
        col = "actions_0" if agent == 0 else "actions_1"
        out.append(str(r.iloc[0][col]))
    return out


def _atom_match(atoms: str, act: str) -> str:
    if not act or act == "None":
        return "none"
    a = act.strip()
    if not atoms:
        return "none"
    parts = [p.strip() for p in atoms.split(";") if p.strip()]
    for p in parts:
        if p == a:
            return "exact"
    va = _verb(a)
    for p in parts:
        if va and va == _verb(p):
            return "family"
    for p in parts:
        if va and va in p.replace(" ", "").lower():
            return "family"
    return "none"


def add_step_deltas(step_df: pd.DataFrame) -> pd.DataFrame:
    s = step_df.sort_values(["run_id", "timestep"]).copy()
    for col in ("chef_holds", "asst_holds", "pot_line"):
        s[col] = s[col].fillna("")
        s[f"delta_{col}"] = s.groupby("run_id")[col].transform(lambda x: x.astype(str) != x.astype(str).shift(1))
    s["chef_queue_len"] = s["chef_queue_str"].apply(queue_atom_count)
    s["asst_queue_len"] = s["asst_queue_str"].apply(queue_atom_count)
    s["delta_chef_queue_len"] = s.groupby("run_id")["chef_queue_len"].diff().fillna(0)
    s["delta_asst_queue_len"] = s.groupby("run_id")["asst_queue_len"].diff().fillna(0)
    return s


def compute_comm_context(
    comm_df: pd.DataFrame, step_df: pd.DataFrame, delta: int = 2
) -> pd.DataFrame:
    if comm_df.empty:
        return comm_df
    step_df = add_step_deltas(step_df)
    # index (run_id, timestep) -> row (dedupe if any)
    idx = step_df.drop_duplicates(subset=["run_id", "timestep"]).set_index(["run_id", "timestep"])

    match_best = []
    val_fail = []
    asst_holds_t = []
    asst_holds_tp = []
    ground_place = []

    for _, row in comm_df.iterrows():
        rid = row["run_id"]
        t = int(row["timestep"])
        ag = int(row["agent"])
        atoms = str(row.get("plan_atoms") or "")
        best = "none"
        for dt in range(0, delta + 1):
            act = _actions_window(step_df, rid, t, ag, dt)
            if not act:
                continue
            a0 = act[0] if dt == 0 else ""
            m = _atom_match(atoms, a0)
            if m == "exact":
                best = "exact"
                break
            if m == "family" and best == "none":
                best = "family"
        match_best.append(best)

        vf = 0
        for dt in range(0, delta + 1):
            key = (rid, t + dt)
            if key not in idx.index:
                continue
            er = idx.loc[key, f"validator_err_{ag}"]
            if isinstance(er, pd.Series):
                er = int(er.iloc[0])
            vf = max(vf, int(er))
        val_fail.append(1 if vf > 0 else 0)

        key0 = (rid, t)
        key1 = (rid, t + 1)
        ah0 = ah1 = ""
        if key0 in idx.index:
            v = idx.loc[key0, "asst_holds"]
            ah0 = str(v.iloc[0]) if isinstance(v, pd.Series) else str(v)
        if key1 in idx.index:
            v = idx.loc[key1, "asst_holds"]
            ah1 = str(v.iloc[0]) if isinstance(v, pd.Series) else str(v)
        asst_holds_t.append(ah0)
        asst_holds_tp.append(ah1)

        say = str(row.get("say_raw") or "").lower()
        conflict = 0
        if ag == 1 and ("placed" in say or "place" in say) and "counter" in say:
            if "nothing" not in (ah1 or "").lower() and "egg" in (ah1 or "").lower():
                conflict = 1
        ground_place.append(conflict)

    out = comm_df.copy()
    out["instr_match_best"] = match_best
    out["validator_fail_in_window"] = val_fail
    out["asst_holds_at_t"] = asst_holds_t
    out["asst_holds_at_t1"] = asst_holds_tp
    out["grounding_place_counter_heuristic"] = ground_place
    out["grounding_rule_id"] = "place_counter_vs_holds_t1"
    return out


def merge_step_metrics_to_timesteps(step_df: pd.DataFrame) -> pd.DataFrame:
    """Optional timestep-level context file."""
    return add_step_deltas(step_df)
