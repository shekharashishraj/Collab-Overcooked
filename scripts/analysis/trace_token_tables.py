#!/usr/bin/env python3
"""Build trace-level, token-level, and TGCO summary tables for paper analysis."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.analysis.extract_1a_run import extract_run_row
from scripts.analysis.load_json import load_json
from scripts.analysis.tgco_audit import (
    TGCO_ASSISTED,
    TGCO_EFFECTIVE,
    TGCO_INEFFECTIVE,
    TGCO_REDUNDANT,
    audit_run,
)
from src.structured_interdependence import score as interdependence_score


MODEL_UNKNOWN = "?"
AGENT_TYPE_UNKNOWN = "?"


def _safe_sum_int(xs: Iterable[Any]) -> int:
    total = 0
    for x in xs:
        try:
            total += int(x or 0)
        except (TypeError, ValueError):
            continue
    return total


def _agent_meta(data: Dict[str, Any]) -> Tuple[str, str, str, str]:
    base = extract_run_row(data, "")
    return (
        base.get("p0_model") or MODEL_UNKNOWN,
        base.get("p1_model") or MODEL_UNKNOWN,
        base.get("p0_agent_type") or AGENT_TYPE_UNKNOWN,
        base.get("p1_agent_type") or AGENT_TYPE_UNKNOWN,
    )


def _comm_stats_for_agent(data: Dict[str, Any], agent: int) -> Dict[str, int]:
    tokens = calls = nonempty_turns = turns = 0
    for entry in data.get("content") or []:
        if not isinstance(entry, dict):
            continue
        comm = ((entry.get("statistical_data") or {}).get("communication") or [])
        if agent >= len(comm) or not isinstance(comm[agent], dict):
            continue
        c = comm[agent]
        toks = c.get("token") or []
        tokens += _safe_sum_int(toks)
        calls += int(c.get("call") or 0)
        for turn in c.get("turn") or []:
            s = str(turn or "").strip()
            if not s:
                continue
            turns += 1
            if "[NOTHING]" not in s.upper():
                nonempty_turns += 1
    return {
        "tokens": tokens,
        "calls": calls,
        "turns": turns,
        "nonempty_turns": nonempty_turns,
    }


def _validator_errors_for_agent(data: Dict[str, Any], agent: int) -> int:
    total = 0
    for entry in data.get("content") or []:
        if not isinstance(entry, dict):
            continue
        errors = ((entry.get("statistical_data") or {}).get("error") or [])
        if agent >= len(errors) or not isinstance(errors[agent], dict):
            continue
        validator = errors[agent].get("validator_error") or {}
        total += int(validator.get("error_num") or 0)
    return total


def _action_count(data: Dict[str, Any], agent: int) -> int:
    total_action_list = data.get("total_action_list") or []
    if agent >= len(total_action_list) or not isinstance(total_action_list[agent], list):
        return 0
    return len(total_action_list[agent])


def episode_metric_row(data: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    """Return one run-level row with trace, interdependence, and token metrics."""
    base = extract_run_row(data, source_path)
    p0_model, p1_model, p0_agent_type, p1_agent_type = _agent_meta(data)
    p0_comm = _comm_stats_for_agent(data, 0)
    p1_comm = _comm_stats_for_agent(data, 1)
    p0_actions = _action_count(data, 0)
    p1_actions = _action_count(data, 1)
    p0_val = _validator_errors_for_agent(data, 0)
    p1_val = _validator_errors_for_agent(data, 1)

    try:
        interdependence = interdependence_score(data)
    except Exception:
        interdependence = {}

    total_timestamps = data.get("total_timestamp") or []
    num_timesteps = len(total_timestamps) or int(base.get("horizon_len") or 0)

    row = {
        **base,
        "source_path": str(source_path),
        "num_timesteps": num_timesteps,
        "last_timestep": max(total_timestamps) if total_timestamps else num_timesteps - 1,
        "p0_model": p0_model,
        "p1_model": p1_model,
        "p0_agent_type": p0_agent_type,
        "p1_agent_type": p1_agent_type,
        "p0_actions": p0_actions,
        "p1_actions": p1_actions,
        "total_team_actions": p0_actions + p1_actions,
        "p0_validator_errors": p0_val,
        "p1_validator_errors": p1_val,
        "sum_validator_errors": p0_val + p1_val,
        "p0_comm_tokens": p0_comm["tokens"],
        "p1_comm_tokens": p1_comm["tokens"],
        "total_comm_tokens": p0_comm["tokens"] + p1_comm["tokens"],
        "p0_comm_calls": p0_comm["calls"],
        "p1_comm_calls": p1_comm["calls"],
        "total_comm_calls": p0_comm["calls"] + p1_comm["calls"],
        "p0_nonempty_comm_turns": p0_comm["nonempty_turns"],
        "p1_nonempty_comm_turns": p1_comm["nonempty_turns"],
        "total_nonempty_comm_turns": p0_comm["nonempty_turns"] + p1_comm["nonempty_turns"],
        "Int_cons": int(interdependence.get("Int_cons") or 0),
        "Int_non_cons": int(interdependence.get("Int_non_cons") or 0),
        "solo_completion": bool(interdependence.get("solo_completion") or False),
        "any_handoff": bool(interdependence.get("handoff_events") or []),
    }
    return row


def tgco_rows_for_episode(data: Dict[str, Any], source_path: str, window: int = 5) -> List[Dict[str, Any]]:
    """Return TGCO evidence rows enriched with order/model/role metadata."""
    run_row = episode_metric_row(data, source_path)
    rows = audit_run(data, source_file=source_path, window=window)
    out: List[Dict[str, Any]] = []
    for row in rows:
        speaker = int(row["speaker"])
        target = int(row["target_agent"])
        enriched = {
            **row,
            "run_id": run_row["run_id"],
            "order": run_row["order"],
            "order_level": run_row["order_level"],
            "p0_model": run_row["p0_model"],
            "p1_model": run_row["p1_model"],
            "p0_agent_type": run_row["p0_agent_type"],
            "p1_agent_type": run_row["p1_agent_type"],
            "team_model_pair": f"{run_row['p0_model']}__{run_row['p1_model']}",
            "team_agent_type_pair": f"{run_row['p0_agent_type']}__{run_row['p1_agent_type']}",
            "speaker_role": f"P{speaker}",
            "target_role": f"P{target}",
            "speaker_model": run_row[f"p{speaker}_model"],
            "target_model": run_row[f"p{target}_model"],
            "speaker_agent_type": run_row[f"p{speaker}_agent_type"],
            "target_agent_type": run_row[f"p{target}_agent_type"],
            "speaker_comm_tokens": run_row[f"p{speaker}_comm_tokens"],
            "target_comm_tokens": run_row[f"p{target}_comm_tokens"],
            "episode_total_comm_tokens": run_row["total_comm_tokens"],
        }
        out.append(enriched)
    return out


def _mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _group_run_summary(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for values, g in df.groupby(list(keys), dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(
            {
                "episodes": int(len(g)),
                "success_rate": _mean(g["success"].astype(float)),
                "mean_timesteps": _mean(g["num_timesteps"]),
                "mean_team_actions": _mean(g["total_team_actions"]),
                "mean_validator_errors_total": _mean(g["sum_validator_errors"]),
                "mean_validator_errors_p0": _mean(g["p0_validator_errors"]),
                "mean_validator_errors_p1": _mean(g["p1_validator_errors"]),
                "mean_Int_cons": _mean(g["Int_cons"]),
                "mean_Int_non_cons": _mean(g["Int_non_cons"]),
                "solo_completion_rate": _mean(g["solo_completion"].astype(float)),
                "any_handoff_rate": _mean(g["any_handoff"].astype(float)),
                "total_comm_tokens": int(g["total_comm_tokens"].sum()),
                "tokens_per_episode": _mean(g["total_comm_tokens"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(keys)).reset_index(drop=True)


def _model_episode_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        seen = set()
        for role in ("p0", "p1"):
            model = r[f"{role}_model"] or MODEL_UNKNOWN
            if model in seen:
                continue
            seen.add(model)
            out = r.to_dict()
            out["model"] = model
            rows.append(out)
    return pd.DataFrame(rows)


def _role_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        for idx, role in enumerate(("p0", "p1")):
            rows.append(
                {
                    "run_id": r["run_id"],
                    "order_level": r["order_level"],
                    "role": role.upper(),
                    "model": r[f"{role}_model"] or MODEL_UNKNOWN,
                    "agent_type": r[f"{role}_agent_type"] or AGENT_TYPE_UNKNOWN,
                    "success": r["success"],
                    "num_timesteps": r["num_timesteps"],
                    "actions": r[f"{role}_actions"],
                    "validator_errors": r[f"{role}_validator_errors"],
                    "comm_tokens": r[f"{role}_comm_tokens"],
                    "comm_calls": r[f"{role}_comm_calls"],
                    "nonempty_comm_turns": r[f"{role}_nonempty_comm_turns"],
                    "slot_index": idx,
                }
            )
    return pd.DataFrame(rows)


def _agent_type_episode_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        seen = set()
        for role in ("p0", "p1"):
            agent_type = r[f"{role}_agent_type"] or AGENT_TYPE_UNKNOWN
            if agent_type in seen:
                continue
            seen.add(agent_type)
            out = r.to_dict()
            out["agent_type"] = agent_type
            rows.append(out)
    return pd.DataFrame(rows)


def _role_summary(role_df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    rows = []
    for values, g in role_df.groupby(list(keys), dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(
            {
                "slots": int(len(g)),
                "success_rate": _mean(g["success"].astype(float)),
                "mean_timesteps": _mean(g["num_timesteps"]),
                "mean_actions": _mean(g["actions"]),
                "mean_validator_errors": _mean(g["validator_errors"]),
                "total_tokens": int(g["comm_tokens"].sum()),
                "tokens_per_slot": _mean(g["comm_tokens"]),
                "tokens_per_call": _safe_div(int(g["comm_tokens"].sum()), int(g["comm_calls"].sum())),
                "tokens_per_nonempty_turn": _safe_div(
                    int(g["comm_tokens"].sum()), int(g["nonempty_comm_turns"].sum())
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(keys)).reset_index(drop=True)


def _safe_div(num: float, den: float) -> Any:
    if not den:
        return None
    return float(num) / float(den)


def _tgco_summary(tgco_df: pd.DataFrame, run_df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if tgco_df.empty:
        return pd.DataFrame()
    run_tokens = run_df.groupby("order_level", dropna=False)["total_comm_tokens"].sum().to_dict()
    rows = []
    for values, g in tgco_df.groupby(list(keys), dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        counts = g["tgco_label"].value_counts().to_dict()
        total = int(len(g))
        effective = int(counts.get(TGCO_EFFECTIVE, 0))
        # For level-level TGCO, connect the exact level token total to Effective units.
        token_total = None
        key_list = list(keys)
        if key_list == ["order_level"]:
            token_total = int(run_tokens.get(row["order_level"], 0))
        elif "speaker_model" in key_list:
            token_total = int(g.drop_duplicates(["run_id", "speaker_role"])["speaker_comm_tokens"].sum())
        elif "target_model" in key_list:
            token_total = int(g.drop_duplicates(["run_id", "target_role"])["target_comm_tokens"].sum())
        elif "episode_total_comm_tokens" in g:
            token_total = int(g.drop_duplicates("run_id")["episode_total_comm_tokens"].sum())
        row.update(
            {
                "request_units": total,
                "effective": effective,
                "assisted": int(counts.get(TGCO_ASSISTED, 0)),
                "redundant": int(counts.get(TGCO_REDUNDANT, 0)),
                "ineffective": int(counts.get(TGCO_INEFFECTIVE, 0)),
                "effective_rate": effective / total if total else 0.0,
                "assisted_rate": int(counts.get(TGCO_ASSISTED, 0)) / total if total else 0.0,
                "redundant_rate": int(counts.get(TGCO_REDUNDANT, 0)) / total if total else 0.0,
                "ineffective_rate": int(counts.get(TGCO_INEFFECTIVE, 0)) / total if total else 0.0,
                "matched_units": int((g["matched_action"].astype(str) != "").sum()),
                "validator_assisted_units": int((g["validator_errors_between"].astype(int) > 0).sum()),
                "total_tokens": token_total,
                "tokens_per_effective": _safe_div(token_total or 0, effective),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(keys)).reset_index(drop=True)


def build_summary_tables(run_rows: List[Dict[str, Any]], tgco_rows: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Build all paper-facing trace, model, token, and TGCO tables."""
    run_df = pd.DataFrame(run_rows)
    tgco_df = pd.DataFrame(tgco_rows)
    tables: Dict[str, pd.DataFrame] = {
        "episode_trace_metrics": run_df,
        "tgco_request_units": tgco_df,
    }
    if run_df.empty:
        return tables

    tables["level_summary"] = _group_run_summary(run_df, ["order_level"])
    tables["model_by_level"] = _group_run_summary(_model_episode_rows(run_df), ["order_level", "model"])

    role_df = _role_rows(run_df)
    tables["model_role_by_level"] = _role_summary(role_df, ["order_level", "role", "model"])
    tables["agent_type_by_level"] = _group_run_summary(
        _agent_type_episode_rows(run_df), ["order_level", "agent_type"]
    )

    team_df = run_df.copy()
    team_df["p0_p1_model"] = team_df["p0_model"].astype(str) + "__" + team_df["p1_model"].astype(str)
    team_df["p0_p1_agent_type"] = (
        team_df["p0_agent_type"].astype(str) + "__" + team_df["p1_agent_type"].astype(str)
    )
    tables["team_model_by_level"] = _group_run_summary(team_df, ["order_level", "p0_model", "p1_model"])
    tables["team_agent_type_by_level"] = _group_run_summary(
        team_df, ["order_level", "p0_agent_type", "p1_agent_type"]
    )

    tables["token_model_by_level"] = _role_summary(role_df, ["order_level", "model"])
    tables["token_model_role_by_level"] = _role_summary(role_df, ["order_level", "role", "model"])
    tables["token_team_by_level"] = _group_run_summary(run_df, ["order_level", "p0_model", "p1_model"])[
        [
            "order_level",
            "p0_model",
            "p1_model",
            "episodes",
            "total_comm_tokens",
            "tokens_per_episode",
        ]
    ]

    if not tgco_df.empty:
        tables["tgco_by_level"] = _tgco_summary(tgco_df, run_df, ["order_level"])
        tables["tgco_by_speaker_model_level"] = _tgco_summary(
            tgco_df, run_df, ["order_level", "speaker_model"]
        )
        tables["tgco_by_target_model_level"] = _tgco_summary(
            tgco_df, run_df, ["order_level", "target_model"]
        )
        tables["tgco_by_team_model_level"] = _tgco_summary(
            tgco_df, run_df, ["order_level", "p0_model", "p1_model"]
        )

    return tables


def expand_inputs(patterns: Sequence[str], exclude_parts: Sequence[str] = ()) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            paths.extend(m for m in matches if m.endswith(".json"))
        elif Path(pattern).is_file() and str(pattern).endswith(".json"):
            paths.append(str(pattern))
    excluded = set(exclude_parts)
    out = []
    for path in sorted(set(paths)):
        parts = set(Path(path).parts)
        if parts & excluded:
            continue
        out.append(path)
    return out


def build_tables_from_paths(paths: Sequence[str], window: int = 5) -> Dict[str, pd.DataFrame]:
    run_rows: List[Dict[str, Any]] = []
    tgco_rows: List[Dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        run_rows.append(episode_metric_row(data, path))
        tgco_rows.extend(tgco_rows_for_episode(data, path, window=window))
    return build_summary_tables(run_rows, tgco_rows)


def write_tables(tables: Dict[str, pd.DataFrame], out_dir: Path, fmt: str = "csv") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df.empty:
            continue
        if fmt in ("csv", "both"):
            df.to_csv(out_dir / f"{name}.csv", index=False)
        if fmt in ("parquet", "both"):
            try:
                df.to_parquet(out_dir / f"{name}.parquet", index=False)
            except Exception as exc:
                print(f"[warn] Parquet skipped for {name}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trace-level, token-level, and TGCO summary tables from experiment JSON logs."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Glob(s) or paths to experiment_*.json")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--window", type=int, default=5, help="TGCO request-action window in timesteps")
    parser.add_argument(
        "--exclude-part",
        action="append",
        default=["gpt-4o", "compiled_metrics"],
        help="Path component to exclude; default excludes legacy gpt-4o and compiled_metrics.",
    )
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="csv")
    args = parser.parse_args()

    paths = expand_inputs(args.inputs, exclude_parts=args.exclude_part)
    if not paths:
        print("No JSON files matched after excludes.", file=sys.stderr)
        sys.exit(1)

    tables = build_tables_from_paths(paths, window=args.window)
    write_tables(tables, args.out_dir, fmt=args.format)
    print(f"Done: {len(paths)} JSON -> {args.out_dir}")
    for name, df in tables.items():
        if not df.empty:
            print(f"  {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
