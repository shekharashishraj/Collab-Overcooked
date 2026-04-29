"""Phase 3: waste score components (timestep grain + optional comm flags)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

from .scene_parse import queue_atom_count


def _similar(a: str, b: str) -> float:
    a, b = (a or "").lower(), (b or "").lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def compute_timestep_waste(
    step_df: pd.DataFrame,
    comm_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    nothing_token_threshold: int = 400,
    duplicate_sim_threshold: float = 0.85,
    tail_frac: float = 0.15,
    directive_min: int = 2,
) -> pd.DataFrame:
    """
    One row per (run_id, timestep) with waste flags and waste_score (unweighted sum).
    """
    if step_df.empty:
        return pd.DataFrame()

    s = step_df.sort_values(["run_id", "timestep"]).copy()
    run_meta = runs_df.set_index("run_id")

    # Comm tokens max per agent at timestep (from enriched comm)
    tok = (
        comm_df.groupby(["run_id", "timestep", "agent"], as_index=False)["comm_tokens_sum"]
        .max()
        .pivot_table(index=["run_id", "timestep"], columns="agent", values="comm_tokens_sum", fill_value=0)
        .reset_index()
    )
    tok = tok.rename(columns={0: "comm_tokens_p0", 1: "comm_tokens_p1"})
    s = s.merge(tok, on=["run_id", "timestep"], how="left")
    s["comm_tokens_p0"] = s["comm_tokens_p0"].fillna(0)
    s["comm_tokens_p1"] = s["comm_tokens_p1"].fillna(0)

    # Scene delta (both wait and no queue/hold change)
    def _no_scene_delta(r):
        return not bool(r.get("delta_chef_holds", False)) and not bool(r.get("delta_asst_holds", False))

    from .metrics_2_context import add_step_deltas

    s = add_step_deltas(s)

    both_wait = (s["actions_0"].astype(str) == "wait(1)") & (s["actions_1"].astype(str) == "wait(1)")
    w3a_idle = both_wait & s.apply(_no_scene_delta, axis=1)

    # [NOTHING] after high comm tokens at same t (any agent)
    nothing_rows = comm_df[comm_df["say_is_nothing"]]
    if not nothing_rows.empty and "comm_tokens_sum" in nothing_rows.columns:
        hi = nothing_rows.groupby(["run_id", "timestep"], as_index=False)["comm_tokens_sum"].max()
        hi = hi.rename(columns={"comm_tokens_sum": "tok_at_nothing"})
    else:
        hi = pd.DataFrame(columns=["run_id", "timestep", "tok_at_nothing"])
    s = s.merge(hi, on=["run_id", "timestep"], how="left")
    w3a_nothing_hi = (s["tok_at_nothing"].fillna(0) >= nothing_token_threshold) & s["tok_at_nothing"].notna()

    # Redundancy: duplicate say vs previous row same run agent (rolling)
    # simpler: merge last say from comm per agent
    last_say = comm_df.sort_values(["run_id", "timestep", "round_idx", "turn_in_round"]).copy()
    last_say["_say"] = last_say["say_raw"].astype(str)
    w3b_dup = []
    for _, r in s.iterrows():
        rid, t = r["run_id"], r["timestep"]
        prev = last_say[(last_say["run_id"] == rid) & (last_say["timestep"] < t)]
        cur = last_say[(last_say["run_id"] == rid) & (last_say["timestep"] == t)]
        mx = 0.0
        for _, c in cur.iterrows():
            ag = int(c["agent"])
            sa = str(c["say_raw"])
            if len(sa) < 20:
                continue
            prev_ag = prev[prev["agent"] == ag]
            for _, p in prev_ag.tail(8).iterrows():
                mx = max(mx, _similar(sa, str(p["say_raw"])))
        w3b_dup.append(1 if mx >= duplicate_sim_threshold else 0)

    s["waste_3b_near_duplicate_say"] = w3b_dup

    # 3c validator next step after directive comm
    dire = []
    for _, r in s.iterrows():
        rid, t = r["run_id"], r["timestep"]
        csub = comm_df[(comm_df["run_id"] == rid) & (comm_df["timestep"] == t)]
        score = 0
        for _, c in csub.iterrows():
            say = str(c["say_raw"]).lower()
            score += len(re.findall(r"\b(put|place|pick|deliver|must|need to)\b", say))
        dire.append(score)
    s["directive_lex_score"] = dire
    s = s.sort_values(["run_id", "timestep"])
    nxt_val = s.groupby("run_id")["validator_err_0"].shift(-1).fillna(0) + s.groupby("run_id")[
        "validator_err_1"
    ].shift(-1).fillna(0)
    s["waste_3c_validator_after_directive"] = (
        (nxt_val > 0) & (s["directive_lex_score"] >= directive_min)
    ).astype(int)

    # 3d late comm in tail when success or deliver near end
    s["_tmax"] = s.groupby("run_id")["timestep"].transform("max")
    s["_tail_start"] = s["_tmax"] - (s.groupby("run_id")["timestep"].transform("count") * tail_frac).clip(
        lower=1
    ).astype(int)
    s["_tail"] = s["timestep"] >= s["_tail_start"]
    succ = s["run_id"].map(lambda x: bool(run_meta.loc[x, "success"]) if x in run_meta.index else False)
    deliv = s["actions_0"].astype(str).str.contains("deliver", case=False) | s["actions_1"].astype(
        str
    ).str.contains("deliver", case=False)
    comm_any = (s["comm_tokens_p0"] + s["comm_tokens_p1"]) > 0
    s["waste_3d_late_talk"] = (s["_tail"] & comm_any & (succ | deliv)).astype(int)

    s["waste_3a_both_wait_no_scene"] = w3a_idle.astype(int)
    s["waste_3a_nothing_high_tokens"] = w3a_nothing_hi.fillna(False).astype(int)

    s["waste_score"] = (
        s["waste_3a_both_wait_no_scene"]
        + s["waste_3a_nothing_high_tokens"]
        + s["waste_3b_near_duplicate_say"]
        + s["waste_3c_validator_after_directive"]
        + s["waste_3d_late_talk"]
    )
    s.drop(
        columns=["_tmax", "_tail", "_tail_start", "tok_at_nothing"],
        inplace=True,
        errors="ignore",
    )
    return s
