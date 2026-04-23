#!/usr/bin/env python3
"""Compile experiment JSON -> per-timestep CSV + run summary CSV/JSON."""
from __future__ import annotations
import argparse, csv, glob, json, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ORDER_LEVELS = {
    "level_1": ["baked_bell_pepper","baked_sweet_potato","boiled_egg","boiled_mushroom","boiled_sweet_potato"],
    "level_2": ["baked_potato_slices","baked_pumpkin_slices","boiled_corn_slices","boiled_green_bean_slices","boiled_potato_slices"],
    "level_3": ["baked_bell_pepper_soup","baked_carrot_soup","baked_mushroom_soup","baked_potato_soup","baked_pumpkin_soup"],
    "level_4": ["sliced_bell_pepper_and_corn_stew","sliced_bell_pepper_and_lentil_stew","sliced_eggplant_and_chickpea_stew","sliced_pumpkin_and_chickpea_stew","sliced_zucchini_and_chickpea_stew"],
    "level_5": ["mashed_broccoli_and_bean_patty","mashed_carrot_and_chickpea_patty","mashed_cauliflower_and_lentil_patty","mashed_potato_and_pea_patty","mashed_sweet_potato_and_bean_patty"],
    "level_6": ["potato_carrot_and_onion_patty","romaine_lettuce_pea_and_tomato_patty","sweet_potato_spinach_and_mushroom_patty","taro_bean_and_bell_pepper_patty","zucchini_green_pea_and_onion_patty"],
}

def order_level(name: str) -> str:
    for lv, xs in ORDER_LEVELS.items():
        if name in xs: return lv
    return "unknown"

def trunc(s: str, n: int = 200) -> str:
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def comm_tokens(c: Dict[str, Any]) -> int:
    return int(sum(c.get("token") or []))

def comm_digest(turns: List[str]) -> str:
    out = []
    for t in turns or []:
        t = (t or "").strip()
        if t: out.append(trunc(t, 90))
        if len(out) >= 4: break
    return " | ".join(out)

def scene_snip(obs: Any) -> str:
    if not obs: return ""
    t = "\n".join(str(x) for x in obs) if isinstance(obs, list) else str(obs)
    m = re.search(r"Scene\s+\d+:[^\n]*", t)
    return trunc(m.group(0) if m else t, 220)

def collect_say_plan(inner: Dict[str, Any], agent: int) -> Tuple[List[str], List[str]]:
    says, plans = [], []
    for rnd in inner.get("content") or []:
        if not isinstance(rnd, list): continue
        for b in rnd:
            if isinstance(b, dict) and b.get("agent") == agent:
                if b.get("say"): says.append(str(b["say"]).strip())
                if b.get("plan"): plans.append(str(b["plan"]).strip())
    return says, plans

def last_say(says: List[str]) -> str:
    for s in reversed(says):
        if s.upper() != "[NOTHING]": return trunc(s, 140)
    return ""

def timestep_row(
    e: Dict[str, Any],
    src: str,
    p0_agent_type: str = "",
    p1_agent_type: str = "",
) -> Dict[str, Any]:
    acts = e.get("actions") or []
    inner = e.get("content") or {}
    sd = e.get("statistical_data") or {}
    comm = sd.get("communication") or [{}, {}]
    err = sd.get("error") or [{}, {}]
    ec = sd.get("error_correction") or [{}, {}]
    s0, p0 = collect_say_plan(inner, 0)
    s1, p1 = collect_say_plan(inner, 1)
    ms = e.get("agent_models") or []
    row: Dict[str, Any] = {
        "source_file": src,
        "timestep": e.get("timestamp"),
        "order": (e.get("order_list") or [""])[0],
        "score": sd.get("score", 0),
        "p0_action": acts[0] if acts else "",
        "p1_action": acts[1] if len(acts) > 1 else "",
        "scene_snippet": scene_snip(inner.get("observation") or []),
        "p0_last_llm_say": last_say(s0),
        "p1_last_llm_say": last_say(s1),
        "p0_last_plan": trunc(p0[-1], 150) if p0 else "",
        "p1_last_plan": trunc(p1[-1], 150) if p1 else "",
        "p0_model": ms[0].get("model", "") if ms else "",
        "p1_model": ms[1].get("model", "") if len(ms) > 1 else "",
        "p0_agent_type": p0_agent_type or (ms[0].get("agent_type", "") if ms else ""),
        "p1_agent_type": p1_agent_type
        or (ms[1].get("agent_type", "") if len(ms) > 1 else ""),
    }
    for i in (0, 1):
        c = comm[i] if i < len(comm) else {}
        turns = c.get("turn") or []
        row[f"p{i}_llm_calls"] = c.get("call", 0)
        row[f"p{i}_comm_tokens"] = comm_tokens(c)
        row[f"p{i}_comm_turns"] = len(turns)
        row[f"p{i}_comm_digest"] = comm_digest(turns)
        er = err[i] if i < len(err) else {}
        fe, ve = er.get("format_error") or {}, er.get("validator_error") or {}
        msgs = (ve.get("error_message") or [])
        cr = ec[i] if i < len(ec) else {}
        vc = (cr.get("validator_correction") or {}).get("correction_tokens") or []
        fc = (cr.get("format_correction") or {}).get("correction_tokens") or []
        row[f"p{i}_format_err"] = fe.get("error_num", 0)
        row[f"p{i}_validator_err"] = ve.get("error_num", 0)
        row[f"p{i}_validator_msg"] = trunc(msgs[0], 120) if msgs else ""
        row[f"p{i}_val_corr_tokens"] = int(sum(vc))
        row[f"p{i}_fmt_corr_tokens"] = int(sum(fc))
    return row

COLS = [
    "source_file",
    "timestep",
    "order",
    "score",
    "p0_action",
    "p1_action",
    "p0_llm_calls",
    "p1_llm_calls",
    "p0_comm_tokens",
    "p1_comm_tokens",
    "p0_comm_turns",
    "p1_comm_turns",
    "p0_comm_digest",
    "p1_comm_digest",
    "p0_format_err",
    "p1_format_err",
    "p0_validator_err",
    "p1_validator_err",
    "p0_validator_msg",
    "p1_validator_msg",
    "p0_val_corr_tokens",
    "p1_val_corr_tokens",
    "p0_fmt_corr_tokens",
    "p1_fmt_corr_tokens",
    "p0_last_llm_say",
    "p1_last_llm_say",
    "p0_last_plan",
    "p1_last_plan",
    "p0_model",
    "p1_model",
    "p0_agent_type",
    "p1_agent_type",
    "scene_snippet",
]

def _agents_models(agents: Any) -> Tuple[str, str]:
    if isinstance(agents, list):
        m0 = m1 = ""
        for a in agents:
            if not isinstance(a, dict):
                continue
            if a.get("player") == "P0" or a.get("index") == 0:
                m0 = str(a.get("model", ""))
            if a.get("player") == "P1" or a.get("index") == 1:
                m1 = str(a.get("model", ""))
        return m0, m1
    if isinstance(agents, dict):
        return str(agents.get("p0_model", "")), str(agents.get("p1_model", ""))
    return "", ""


def _agents_agent_types(agents: Any) -> Tuple[str, str]:
    if isinstance(agents, list):
        t0 = t1 = ""
        for a in agents:
            if not isinstance(a, dict):
                continue
            if a.get("player") == "P0" or a.get("index") == 0:
                t0 = str(a.get("agent_type", "") or "")
            if a.get("player") == "P1" or a.get("index") == 1:
                t1 = str(a.get("agent_type", "") or "")
        return t0, t1
    if isinstance(agents, dict):
        return str(agents.get("p0_agent_type", "")), str(agents.get("p1_agent_type", ""))
    return "", ""


def run_summary(data: Dict[str, Any], src: str) -> Dict[str, Any]:
    order = ""
    for e in data.get("content") or []:
        if e.get("order_list"): order = e["order_list"][0]; break
    fin = data.get("total_order_finished") or []
    sc = data.get("total_score", 0)
    stamps = data.get("total_timestamp") or []
    p0m, p1m = _agents_models(data.get("agents"))
    p0t, p1t = _agents_agent_types(data.get("agents"))
    p0v = p1v = p0f = p1f = ct = calls = 0
    for e in data.get("content") or []:
        sd = e.get("statistical_data") or {}
        for c in sd.get("communication") or []:
            ct += comm_tokens(c); calls += int(c.get("call") or 0)
        for i, er in enumerate(sd.get("error") or []):
            fe = (er.get("format_error") or {}).get("error_num", 0)
            ve = (er.get("validator_error") or {}).get("error_num", 0)
            if i==0: p0f += fe; p0v += ve
            elif i==1: p1f += fe; p1v += ve
    return {
        "source_file": src,
        "order": order,
        "order_level": order_level(order),
        "success": bool(fin) or int(sc) > 0,
        "total_score": sc,
        "orders_finished": ";".join(fin),
        "num_timesteps": len(stamps),
        "last_timestep": max(stamps) if stamps else -1,
        "total_comm_tokens": ct,
        "total_llm_comm_calls": calls,
        "sum_p0_validator_errors": p0v,
        "sum_p1_validator_errors": p1v,
        "sum_p0_format_errors": p0f,
        "sum_p1_format_errors": p1f,
        "p0_model": p0m,
        "p1_model": p1m,
        "p0_agent_type": p0t,
        "p1_agent_type": p1t,
    }

def expand(ps: Iterable[str]) -> List[str]:
    out = []
    for p in ps:
        out.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    return [x for x in out if x.endswith(".json")]

def wcsv(rows: List[Dict[str, Any]], cols: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)

def md(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    L = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        L.append("| " + " | ".join(str(r.get(c,"")).replace("|","\\|") for c in cols) + " |")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--combine-csv", default="")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json-summary", default="")
    a = ap.parse_args()
    paths = expand(a.inputs)
    if not paths: raise SystemExit("no json")
    allr, sums = [], []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        src = str(Path(path).resolve())
        p0t, p1t = _agents_agent_types(data.get("agents"))
        tr = [
            timestep_row(e, src, p0_agent_type=p0t, p1_agent_type=p1t)
            for e in data.get("content") or []
        ]
        allr.extend(tr); sums.append(run_summary(data, src))
        if a.csv_out:
            stem = Path(path).stem; d = Path(a.csv_out)
            wcsv(tr, COLS, d / f"{stem}_timesteps.csv")
            wcsv([sums[-1]], list(sums[-1].keys()), d / f"{stem}_run.csv")
    if a.combine_csv: wcsv(allr, COLS, Path(a.combine_csv))
    if a.json_summary: Path(a.json_summary).write_text(json.dumps(sums, indent=2), encoding="utf-8")
    if a.markdown:
        slim = ["timestep","score","p0_action","p1_action","p0_comm_tokens","p1_comm_tokens","p0_validator_err","p1_validator_err","p0_last_llm_say","p1_last_llm_say","scene_snippet"]
        first = str(Path(paths[0]).resolve())
        print(md([r for r in allr if r["source_file"]==first], slim))

if __name__ == "__main__":
    main()
