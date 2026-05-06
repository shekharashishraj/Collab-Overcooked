#!/usr/bin/env python3
"""
Summarize partner-prediction / belief metrics from experiment JSON files.

Uses top-level `belief_metrics` if present (new runs). For older logs without it,
only agent_type and a note are available from `agents` / per-timestep content.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_belief(path: Path) -> dict[str, Any] | None:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def episode_accuracy_row(obj: dict, src: str) -> dict[str, Any]:
    bm = obj.get("belief_metrics")
    agents = obj.get("agents") or []
    order = (obj.get("total_order_finished") or [""])[0]
    row: dict[str, Any] = {
        "file": src,
        "order": order,
    }
    if not bm:
        row["status"] = "no_belief_metrics_key_re_run_main_py"
        row["p0_type"] = agents[0].get("agent_type") if agents else ""
        row["p1_type"] = agents[1].get("agent_type") if len(agents) > 1 else ""
        return row

    row["status"] = "ok"
    team = bm.get("team", {}).get("team", {})
    row["mean_accuracy_two_sided"] = team.get("mean_accuracy_two_sided")
    row["players_with_metrics"] = team.get("players_with_belief_metrics")

    for slot, key in ((0, "p0"), (1, "p1")):
        per = (bm.get("per_player") or [])[slot] if slot < len(bm.get("per_player") or []) else {}
        row[f"{key}_framework"] = per.get("framework")
        row[f"{key}_accuracy"] = per.get("accuracy")
        if per.get("accuracy_episode") is not None:
            row[f"{key}_accuracy"] = per.get("accuracy_episode")
        row[f"{key}_n_scored"] = per.get("n_predictions_scored") or per.get(
            "n_predictions_scored_episode"
        )
    return row


def main():
    ap = argparse.ArgumentParser(description="Summarize belief_metrics from experiment JSON logs")
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSON files or directories (recursive experiment_*.json)",
    )
    ap.add_argument(
        "--glob-may-l1",
        action="store_true",
        help="Use src/data_runs/l1_gpt4o/**/*.json",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    if args.glob_may_l1:
        files = sorted((root / "data_runs" / "l1_gpt4o").rglob("experiment_*.json"))
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("experiment_*.json")))
        elif p.suffix == ".json":
            files.append(p)

    if not files:
        raise SystemExit("No JSON files. Pass paths or --glob-may-l1.")

    rows = []
    for p in files:
        obj = load_belief(p)
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        rows.append(episode_accuracy_row(obj, rel))

    # Print markdown table
    print("| source | order | P0 fw | P0 acc | P1 fw | P1 acc | team mean | note |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        note = r["status"]
        print(
            f"| {Path(r['file']).name} | {r.get('order','')} | "
            f"{r.get('p0_framework','')} | {r.get('p0_accuracy','')} | "
            f"{r.get('p1_framework','')} | {r.get('p1_accuracy','')} | "
            f"{r.get('mean_accuracy_two_sided','')} | {note} |"
        )

    # Aggregate by framework (episode-level accuracies for sides that have metrics)
    by_fw: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["status"] != "ok":
            continue
        for side in ("p0", "p1"):
            fw = r.get(f"{side}_framework")
            acc = r.get(f"{side}_accuracy")
            if fw and acc is not None:
                by_fw[str(fw)].append(float(acc))

    print("\n### Mean episode accuracy by framework (player slots pooled)")
    for fw, xs in sorted(by_fw.items()):
        if not xs:
            continue
        print(f"  {fw}: n={len(xs)}, mean={sum(xs)/len(xs):.4f}")


if __name__ == "__main__":
    main()
