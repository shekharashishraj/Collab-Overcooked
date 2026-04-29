"""Phase 1D: attach statistical_data.communication + heuristic alignment."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def _entry_by_timestep(data: Dict[str, Any]) -> Dict[Any, Dict]:
    m: Dict[Any, Dict] = {}
    for entry in data.get("content") or []:
        if isinstance(entry, dict) and "timestamp" in entry:
            m[entry["timestamp"]] = entry
    return m


def _comm_slice(entry: Dict[str, Any], agent: int) -> Dict[str, Any]:
    sd = entry.get("statistical_data") or {}
    comm = sd.get("communication") or []
    if agent >= len(comm) or not isinstance(comm[agent], dict):
        return {"comm_call": 0, "comm_turns": [], "comm_tokens_sum": 0}
    c = comm[agent]
    turns = list(c.get("turn") or [])
    toks = list(c.get("token") or [])
    sm = 0
    for x in toks:
        try:
            sm += int(x)
        except (TypeError, ValueError):
            pass
    return {
        "comm_call": int(c.get("call") or 0),
        "comm_turns": turns,
        "comm_tokens_sum": sm,
    }


def enrich_comm_dataframe(df_comm: pd.DataFrame, run_id: str, data: Dict[str, Any]) -> pd.DataFrame:
    if df_comm.empty:
        return df_comm
    by_t = _entry_by_timestep(data)
    m = df_comm["run_id"] == run_id
    if not m.any():
        return df_comm

    out = df_comm.copy()
    arr_call, arr_tok, arr_json, arr_al, arr_cf = [], [], [], [], []

    for _, row in out[m].sort_values(["timestep", "round_idx", "turn_in_round"]).iterrows():
        t = row["timestep"]
        agent = int(row["agent"])
        entry = by_t.get(t) or {}
        st = _comm_slice(entry, agent)
        turns = [str(x) for x in st["comm_turns"]]
        turn_f = [s for s in turns if "[NOTHING]" not in s.upper()]

        sub_rows = out[m & (out["timestep"] == t) & (out["agent"] == agent)].sort_values(
            ["round_idx", "turn_in_round"]
        )
        nn_rows = [
            i
            for i, r2 in sub_rows.iterrows()
            if "[NOTHING]" not in str(r2["say_raw"]).upper()
        ]
        struct_n = len(nn_rows)
        if "[NOTHING]" in str(row["say_raw"]).upper():
            ord_idx = -1
        else:
            try:
                ord_idx = nn_rows.index(row.name)
            except ValueError:
                ord_idx = 0

        if turn_f and struct_n and ord_idx >= 0:
            conf = "exact_len_match" if struct_n == len(turn_f) else "fuzzy"
            al = turn_f[ord_idx] if ord_idx < len(turn_f) else ""
        elif "[NOTHING]" in str(row["say_raw"]).upper():
            conf = "nothing_say"
            al = ""
        else:
            conf = "no_comm_turns"
            al = ""

        arr_call.append(st["comm_call"])
        arr_tok.append(st["comm_tokens_sum"])
        arr_json.append("|".join(turns))
        arr_al.append(al)
        arr_cf.append(conf)

    out.loc[m, "comm_call"] = arr_call
    out.loc[m, "comm_tokens_sum"] = arr_tok
    out.loc[m, "comm_turns_json"] = arr_json
    out.loc[m, "turn_text_aligned"] = arr_al
    out.loc[m, "alignment_confidence"] = arr_cf
    return out


def enrich_all_comm(df_comm: pd.DataFrame, runs: List[Tuple[str, Dict[str, Any]]]) -> pd.DataFrame:
    out = df_comm.copy()
    for run_id, data in runs:
        out = enrich_comm_dataframe(out, run_id, data)
    return out
