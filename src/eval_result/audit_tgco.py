#!/usr/bin/env python3
"""
Recompute TGCO-style outcome shares from experiment JSON logs.

Definition (aligned with eval_result/tgco.md prose):
- A *request unit* is an explicit request('...') in the speaker's `say` or `plan` string,
  interpreted as asking the *partner* to perform the inner action.
- Window: 15 environment timesteps after the dialogue turn's env timestamp.
- Effective: partner executes matching action in window, no partner validator errors on
  content rows with timestamp in (t, t_match].
- Assisted: partner matches in window but >=1 partner validator error in that interval.
- Redundant: partner had already executed the same normalized action before t, or exact
  duplicate request (same norm action + same speaker) seen earlier in episode.
- Ineffective: otherwise.

This is an automated approximation; manual audits may differ on edge cases (NL-only asks,
synonym actions, etc.).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

WIN = 15

REQUEST_RE = re.compile(r"request\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.I)
EXPERIMENT_ORDER_RE = re.compile(
    r"experiment_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)\.json$"
)


def normalize_action(s: str) -> str:
    s = s.replace(" ", "").lower()
    s = s.replace("start_cooking(", "cook(")
    return s


def extract_requests(say: str, plan: str) -> list[str]:
    text = f"{say or ''}\n{plan or ''}"
    out = []
    for m in REQUEST_RE.finditer(text):
        inner = m.group(1).strip()
        if inner:
            out.append(normalize_action(inner))
    return out


def partner_actions_timeline(total_action_list: list, partner: int) -> list[tuple[int, str]]:
    seq = total_action_list[partner] if partner < len(total_action_list) else []
    rows = []
    for e in seq:
        ts = int(e["timestamp"])
        act = normalize_action(e["action"])
        rows.append((ts, act))
    rows.sort(key=lambda x: x[0])
    return rows


def partner_validator_hits(
    content_rows: list[dict], partner: int, t_lo: int, t_hi: int
) -> int:
    hits = 0
    for row in content_rows:
        ts = int(row["timestamp"])
        if ts <= t_lo or ts > t_hi:
            continue
        errs = row.get("statistical_data", {}).get("error", [])
        if partner >= len(errs):
            continue
        ve = errs[partner].get("validator_error", {})
        hits += int(ve.get("error_num") or 0)
    return hits


def audit_episode(obj: dict) -> dict[str, int]:
    counts = defaultdict(int)
    total_action_list = obj.get("total_action_list") or [[], []]
    content_rows = obj.get("content") or []

    seen_requests: list[tuple[int, str, int]] = []  # (t, norm_action, speaker)

    for row in content_rows:
        t = int(row["timestamp"])
        inner = row.get("content") or {}
        blocks = inner.get("content") or []
        # blocks: list of rounds; first round is list of agent dicts
        if not blocks or not blocks[0]:
            continue
        for msg in blocks[0]:
            if not isinstance(msg, dict):
                continue
            agent = int(msg.get("agent", -1))
            if agent not in (0, 1):
                continue
            partner = 1 - agent
            say = msg.get("say") or ""
            plan = msg.get("plan") or ""
            if "[NOTHING]" in (say or "") and not REQUEST_RE.search(plan or ""):
                continue
            reqs = extract_requests(say, plan)
            if not reqs:
                continue
            timeline = partner_actions_timeline(total_action_list, partner)
            done_so_far = {act for ts, act in timeline if ts < t}

            for r in reqs:
                counts["_units"] += 1
                dup_req = any(
                    ts < t and sp == agent and ra == r for ts, ra, sp in seen_requests
                )
                if r in done_so_far or dup_req:
                    counts["Redundant"] += 1
                    seen_requests.append((t, r, agent))
                    continue
                t_match = None
                for ts, act in timeline:
                    if ts <= t:
                        continue
                    if ts > t + WIN:
                        break
                    if act == r:
                        t_match = ts
                        break
                seen_requests.append((t, r, agent))
                if t_match is None:
                    counts["Ineffective"] += 1
                    continue
                ve = partner_validator_hits(content_rows, partner, t, t_match)
                if ve > 0:
                    counts["Assisted"] += 1
                else:
                    counts["Effective"] += 1

    return dict(counts)


def pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def run_on_files(paths: list[Path]) -> tuple[dict[str, int], int]:
    agg = defaultdict(int)
    for p in paths:
        with open(p) as f:
            obj = json.load(f)
        c = audit_episode(obj)
        for k, v in c.items():
            agg[k] += v
    units = agg.get("_units", 0)
    return dict(agg), units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="experiment JSON files (default: built-in March gpt-4o 9-run sample)",
    )
    ap.add_argument(
        "--per-file",
        action="store_true",
        help="print counts per JSON then aggregate",
    )
    ap.add_argument(
        "--may-l1-gpt4o",
        action="store_true",
        help="audit src/data_runs/l1_gpt4o/baseline_gpt-4o/**/*.json (May 2026 L1 self-play)",
    )
    args = ap.parse_args()

    if args.may_l1_gpt4o:
        root = Path(__file__).resolve().parents[1] / "data_runs" / "l1_gpt4o" / "baseline_gpt-4o"
        paths = sorted(root.rglob("experiment_*.json"))
    elif args.paths:
        paths = [p for p in args.paths if p.suffix == ".json"]
    else:
        # Same 9 logs whose communication token sums match tgco.md (see audit note).
        root = Path(__file__).resolve().parents[1] / "data" / "gpt-4o"
        names = [
            "boiled_egg/experiment_2026-03-28_10-40-32_boiled_egg.json",
            "boiled_sweet_potato/experiment_2026-03-28_10-40-34_boiled_sweet_potato.json",
            "baked_sweet_potato/experiment_2026-03-28_10-40-35_baked_sweet_potato.json",
            "boiled_green_bean_slices/experiment_2026-03-28_12-10-05_boiled_green_bean_slices.json",
            "boiled_potato_slices/experiment_2026-03-28_12-10-05_boiled_potato_slices.json",
            "baked_potato_slices/experiment_2026-03-28_12-15-26_baked_potato_slices.json",
            "baked_bell_pepper_soup/experiment_2026-03-28_13-53-34_baked_bell_pepper_soup.json",
            "baked_potato_soup/experiment_2026-03-28_13-53-34_baked_potato_soup.json",
            "baked_pumpkin_soup/experiment_2026-03-28_14-19-21_baked_pumpkin_soup.json",
        ]
        paths = [root / n for n in names]

    if not paths:
        raise SystemExit("No JSON paths to audit.")

    if args.per_file:
        for p in paths:
            c, u = run_on_files([p])
            m = EXPERIMENT_ORDER_RE.match(p.name)
            dish = m.group(1) if m else p.stem
            print(f"\n== {p.name} (order={dish}) ==")
            print("  units:", u)
            for k in ("Effective", "Assisted", "Redundant", "Ineffective"):
                print(f"  {k}: {c.get(k, 0)} ({pct(c.get(k, 0), u)}%)")

    agg, units = run_on_files(paths)
    if args.per_file:
        print("\n--- Aggregate ---")
    print("Files:", len(paths))
    print("Total request units (_units):", units)
    for k in ("Effective", "Assisted", "Redundant", "Ineffective"):
        print(f"  {k}: {agg.get(k, 0)} ({pct(agg.get(k, 0), units)}%)")
    if not args.may_l1_gpt4o:
        print("tgco.md Overall reference: 12.7 / 27.0 / 13.9 / 46.3")


if __name__ == "__main__":
    main()
