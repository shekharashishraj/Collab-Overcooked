#!/usr/bin/env python3
"""
Generate all team-matrix figures from eval/data/team_metrics.csv.

Usage:
  python eval/generate_all_team_plots.py
  python eval/generate_all_team_plots.py --csv eval/data/team_metrics.csv --out eval/output

By default, rows with n_success < n_attempts are excluded (partial sweeps). Use
--include-partial-sweeps to plot every row. CSV columns n_success, n_attempts are optional;
if absent, rows with success_pct < 100 are excluded instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from plot_grouped_bars import plot_four_metrics_small_multiples, plot_tokens_and_verifier
from plot_heatmaps import load_team_csv, plot_default_heatmaps
from plot_scatter_tokens_success import plot_scatter


def filter_successful_runs_only(df):
    """Keep rows where every logged attempt succeeded (n_success == n_attempts).

    Rows without n_success/n_attempts are dropped unless columns exist: if columns
    are missing entirely, fall back to success_pct >= 100.
    """
    if "n_success" not in df.columns or "n_attempts" not in df.columns:
        return df.loc[df["success_pct"] >= 100].copy()
    ns = pd.to_numeric(df["n_success"], errors="coerce")
    na = pd.to_numeric(df["n_attempts"], errors="coerce")
    ok = ns.notna() & na.notna() & (ns == na)
    return df.loc[ok].copy()


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate team-matrix plots for eval/")
    parser.add_argument("--csv", type=Path, default=root / "data" / "team_metrics.csv")
    parser.add_argument("--out", type=Path, default=root / "output")
    parser.add_argument(
        "--include-partial-sweeps",
        action="store_true",
        help="Include team rows where n_success < n_attempts (default: exclude them).",
    )
    args = parser.parse_args()

    df = load_team_csv(args.csv)
    if not args.include_partial_sweeps:
        df = filter_successful_runs_only(df)
    if df.empty:
        raise SystemExit("No rows left after filtering to successful runs only. Check n_success/n_attempts in CSV.")
    args.out.mkdir(parents=True, exist_ok=True)

    heat_paths = plot_default_heatmaps(df, args.out)
    bar_path = plot_tokens_and_verifier(df, args.out / "bars_tokens_verif.png")
    extra_bars = plot_four_metrics_small_multiples(df, args.out / "bars_success_follow_adr_idensity.png")
    scatter_path = plot_scatter(df, args.out / "scatter_success_vs_tokens.png")

    print("Wrote:")
    for p in heat_paths:
        print(" ", p)
    print(" ", bar_path)
    print(" ", extra_bars)
    print(" ", scatter_path)


if __name__ == "__main__":
    main()
