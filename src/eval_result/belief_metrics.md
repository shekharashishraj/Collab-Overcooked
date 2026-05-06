# Partner belief / prediction metrics

New experiment logs (`main.py` **exp** mode) include a top-level **`belief_metrics`** block written at **episode end**.

## Per framework

| Framework | What is measured | Match rule |
|-----------|------------------|------------|
| **baseline** | — | No `partner_prediction` instrumentation (metrics block explains). |
| **reflexion** | — | Same as baseline for belief tracking. |
| **proagent** | Each time the partner completes an ML action, compare to your latest **`Chef`/`Assistant` `partner_prediction:`** line (parsed after each planner reply). | Lenient substring / token overlap (`agents.py`). |
| **a-tom** | Same prediction hook + **`tom_event_log`** (full episode) and a rolling **last 12** completions (`tom_accuracy` deque). | Stricter string match than ProAgent (substring only). |

## JSON shape

```json
"belief_metrics": {
  "team": {
    "team": {
      "mean_accuracy_two_sided": 0.42,
      "players_with_belief_metrics": 2
    }
  },
  "per_player": [
    { "player": "P0", "framework": "proagent", "accuracy": 0.5, "events": [ ... ] },
    { "player": "P1", "framework": "proagent", "accuracy": 0.35, "events": [ ... ] }
  ]
}
```

Each **event**: `{ "timestep", "prediction", "actual", "match" }`.

## Aggregate across runs

```bash
python3 src/eval_result/belief_metrics_report.py --glob-may-l1
python3 src/eval_result/belief_metrics_report.py path/to/dir/
```

Logs recorded **before** this export was added only have agent types in `agents[]`; the report will say `no_belief_metrics_key_re_run_main_py`. Re-run experiments to populate metrics.

## Prompts

- Pro-Agent overlay: `prompts/agents/proagent/belief_correction.txt`
- A-ToM overlay: `prompts/agents/a-tom/partner_model.txt`, `partner_output_format.txt`
