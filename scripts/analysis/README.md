# Experiment JSON analysis pipeline

Builds normalized tables from `experiment_*.json` logs produced by `src/main.py`.

## Run

### Batch: one output tree per `src/data/<folder>/`

Skips `gpt-4o` and `compiled_metrics` by default. Writes `analysis/<same-folder-name>/` (runs, timesteps, …).

```bash
python3 scripts/analysis/batch_from_data_dirs.py --dry-run
python3 scripts/analysis/batch_from_data_dirs.py
```

Flags: `--data-root`, `--out-root` (default `analysis/`), `--exclude-dir` (extra skips), `--no-default-excludes`, same `--format` / `--phases` / `--delta` / `--include-audit` as `build_tables.py`.

### Single glob

From the **repository root**:

```bash
python3 scripts/analysis/build_tables.py \
  --inputs 'src/data/**/*.json' \
  --out-dir analysis_out/run1 \
  --format both \
  --phases 1,2,3,4 \
  --delta 2
```

Flags:

| Flag | Meaning |
|------|---------|
| `--inputs` | One or more globs or file paths (recursive `**` supported) |
| `--out-dir` | Output directory |
| `--format` | `parquet`, `csv`, or `both` (Parquet needs `pyarrow`) |
| `--phases` | Subset e.g. `1` or `1,2,3,4` |
| `--delta` | Phase 2 lookahead for instruction vs `actions_*` |
| `--include-audit` | Adds `audit_json` on timestep rows from `timestep_conversations` |

## Output tables

| File | Phase | Keys / description |
|------|-------|----------------------|
| `runs` | 1A | `run_id` (sha16 of path), `file_stem`, `source_path`, `order`, `order_level`, `success`, models, `agent_type`s |
| `timesteps` | 1B | `(run_id, timestep)` — `actions_*`, `map`, errors, `scene_raw_*`, parsed `chef_holds`, queues, `pot_line` |
| `comm_rounds` | 1C+1D | `(run_id, timestep, round_idx, turn_in_round, agent)` — `say_raw`, `plan_raw`, `comm_*`, `turn_text_aligned`, `alignment_confidence` |
| `timesteps_metrics` | 2 | Same as timesteps + `delta_*` columns |
| `comm_rounds_context` | 2 | Comm rows + `instr_match_best`, `validator_fail_in_window`, grounding heuristic |
| `timestep_waste` | 3 | Per timestep `waste_*` flags + `waste_score` |
| `comm_rounds_lingua` | 4 | Linguistic feature columns on comm rows (uses Phase 2 output if Phase 2 ran) |

## Joins

- `timesteps.run_id` = `runs.run_id`
- `comm_rounds.run_id` + `timestep` → `timesteps`
- `timestep_waste` aligns on `(run_id, timestep)`

## Caveats

1. CSV row count vs `wc -l`: `map` and text fields may contain newlines; use pandas row counts for logic.
2. **`observation[1]`** may be `[]` in JSON; loaders coerce to empty string; Scene parse prefers P0 observation when it contains the joint `Scene` block.
3. **Alignment** (`turn_text_aligned`): zips structured non-`[NOTHING]` says with `communication.turn` strings; `alignment_confidence` is `exact_len_match`, `fuzzy`, `nothing_say`, or `no_comm_turns`.
4. **Grounding heuristic** (`grounding_place_counter_heuristic`): keyword trigger on assistant “place/counter” vs `asst_holds` at `t+1`; high false positive rate possible — use as exploratory only.
5. **Parquet**: install `pyarrow` or use `--format csv`.

## Verification

Run on a small glob and inspect row counts:

```bash
python3 scripts/analysis/build_tables.py \
  --inputs 'src/data/gpt-4o__claude-sonnet-4-20250514/boiled_egg/experiment_*.json' \
  --out-dir /tmp/collab_analysis_smoke \
  --format csv --phases 1,2,3,4
```
