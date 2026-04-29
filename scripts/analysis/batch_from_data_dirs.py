#!/usr/bin/env python3
"""Run build_tables once per experiment folder under src/data (mirrors folder name under analysis/)."""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Skip legacy single-model folder and aggregated metrics (not per-order experiment logs).
_DEFAULT_EXCLUDES = frozenset({"gpt-4o", "compiled_metrics"})


def _discover_run_dirs(data_root: Path, exclude: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("."):
            continue
        if name in exclude:
            continue
        pattern = str(p / "**" / "experiment_*.json")
        if not glob.glob(pattern, recursive=True):
            continue
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "For each immediate subfolder of --data-root that contains experiment_*.json, "
            "run build_tables and write CSV/Parquet under --out-root/<folder_name>/."
        )
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=_REPO_ROOT / "src" / "data",
        help="Usually repo/src/data",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=_REPO_ROOT / "analysis",
        help="Parent directory; each run folder becomes out-root/<folder_name>/",
    )
    ap.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Extra folder name(s) under data-root to skip (repeatable). "
        "Always skipped unless cleared via --no-default-excludes: gpt-4o, compiled_metrics",
    )
    ap.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not skip gpt-4o or compiled_metrics; only --exclude-dir names are skipped",
    )
    ap.add_argument("--format", choices=["parquet", "csv", "both"], default="both")
    ap.add_argument("--phases", default="1,2,3,4")
    ap.add_argument("--delta", type=int, default=2)
    ap.add_argument("--include-audit", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="List folders only; do not run build_tables")
    args = ap.parse_args()

    exclude: set[str] = set(args.exclude_dir)
    if not args.no_default_excludes:
        exclude |= set(_DEFAULT_EXCLUDES)
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    if not data_root.is_dir():
        print(f"Not a directory: {data_root}", file=sys.stderr)
        sys.exit(1)

    run_dirs = _discover_run_dirs(data_root, exclude)
    if not run_dirs:
        print(f"No run folders with experiment_*.json under {data_root} (after excludes).", file=sys.stderr)
        sys.exit(1)

    build_tables = _REPO_ROOT / "scripts" / "analysis" / "build_tables.py"
    if not build_tables.is_file():
        print(f"Missing {build_tables}", file=sys.stderr)
        sys.exit(1)

    for d in run_dirs:
        folder_name = d.name
        out_dir = out_root / folder_name
        pattern = str(d / "**" / "experiment_*.json")
        n = len(glob.glob(pattern, recursive=True))
        print(f"{folder_name}: {n} JSON -> {out_dir}", flush=True)
        if args.dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(build_tables),
            "--inputs",
            pattern,
            "--out-dir",
            str(out_dir),
            "--format",
            args.format,
            "--phases",
            args.phases,
            "--delta",
            str(args.delta),
        ]
        if args.include_audit:
            cmd.append("--include-audit")
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)

    if args.dry_run:
        print("(dry-run: no tables written)")


if __name__ == "__main__":
    main()
