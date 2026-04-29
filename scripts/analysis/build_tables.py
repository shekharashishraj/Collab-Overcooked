#!/usr/bin/env python3
"""Build analysis Parquet/CSV tables from experiment JSON logs."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.analysis.extract_1a_run import extract_run_row
from scripts.analysis.extract_1b_step import extract_timestep_rows
from scripts.analysis.extract_1c_comm import extract_comm_round_rows
from scripts.analysis.extract_1d_align import enrich_all_comm
from scripts.analysis.load_json import load_json
from scripts.analysis.metrics_2_context import add_step_deltas, compute_comm_context
from scripts.analysis.metrics_3_waste import compute_timestep_waste
from scripts.analysis.metrics_4_lingua import add_linguistic_columns


def _expand_inputs(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p, recursive=True))
        if matches:
            out.extend(m for m in matches if m.endswith(".json"))
        elif Path(p).is_file() and str(p).endswith(".json"):
            out.append(str(p))
    return sorted(set(out))


def _write(df: pd.DataFrame, path: Path, fmt: str) -> None:
    if df.empty:
        print(f"[skip] empty: {path.name}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("csv", "both"):
        df.to_csv(path.with_suffix(".csv"), index=False)
    if fmt in ("parquet", "both"):
        try:
            df.to_parquet(path.with_suffix(".parquet"), index=False)
        except Exception as e:
            print(f"[warn] Parquet skipped for {path.name}: {e}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten experiment JSON to analysis tables")
    ap.add_argument("--inputs", nargs="+", required=True, help="Glob(s) or paths to experiment_*.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--format", choices=["parquet", "csv", "both"], default="both")
    ap.add_argument("--phases", default="1,2,3,4", help="Comma-separated: 1,2,3,4")
    ap.add_argument("--delta", type=int, default=2, help="Lookahead for instruction match (Phase 2)")
    ap.add_argument("--include-audit", action="store_true", help="Phase 1E: audit_json on timesteps")
    args = ap.parse_args()
    phases = {int(x.strip()) for x in args.phases.split(",") if x.strip()}

    paths = _expand_inputs(args.inputs)
    if not paths:
        print("No JSON files matched.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_list: List[Dict[str, Any]] = []
    steps_list: List[Dict[str, Any]] = []
    comm_list: List[Dict[str, Any]] = []
    run_pairs: List[Tuple[str, Dict[str, Any]]] = []

    for path in paths:
        data = load_json(path)
        rr = extract_run_row(data, path)
        rid = rr["run_id"]
        run_pairs.append((rid, data))
        runs_list.append(rr)
        steps_list.extend(
            extract_timestep_rows(rid, data, include_audit=args.include_audit and 1 in phases)
        )
        comm_list.extend(extract_comm_round_rows(rid, data, order_hint=rr.get("order") or ""))

    runs_df = pd.DataFrame(runs_list)
    step_df = pd.DataFrame(steps_list)
    comm_df = pd.DataFrame(comm_list)
    comm_enriched = enrich_all_comm(comm_df, run_pairs) if not comm_df.empty else comm_df
    step_metrics = add_step_deltas(step_df) if not step_df.empty else step_df

    if 1 in phases:
        _write(runs_df, out_dir / "runs", args.format)
        _write(step_df, out_dir / "timesteps", args.format)
        _write(comm_enriched, out_dir / "comm_rounds", args.format)

    comm_ctx = comm_enriched
    if 2 in phases and not comm_enriched.empty and not step_metrics.empty:
        _write(step_metrics, out_dir / "timesteps_metrics", args.format)
        comm_ctx = compute_comm_context(comm_enriched, step_metrics, delta=args.delta)
        _write(comm_ctx, out_dir / "comm_rounds_context", args.format)

    if 3 in phases and not step_metrics.empty:
        waste_df = compute_timestep_waste(step_metrics, comm_enriched, runs_df)
        _write(waste_df, out_dir / "timestep_waste", args.format)

    if 4 in phases and not comm_enriched.empty:
        base = comm_ctx if 2 in phases else comm_enriched
        ling = add_linguistic_columns(base)
        _write(ling, out_dir / "comm_rounds_lingua", args.format)

    print(f"Done: {len(paths)} JSON -> {out_dir}")


if __name__ == "__main__":
    main()
