# TGCO re-audit — May 2026 Level 1 (GPT-4o self-play)

**Corpus:** `src/data_runs/l1_gpt4o/baseline_gpt-4o/**/experiment_*.json` (5 episodes, one per Level‑1 dish).

**Method:** Automated rules in `audit_tgco.py` (explicit `request('...')` in `say`/`plan`, 15-step window, validator buckets). This is an approximation of the manual TGCO protocol in `tgco.md`; NL-only asks are not counted.

## Per episode

| Order | Request units | Effective | Assisted | Redundant | Ineffective |
| --- | ---:| ---:| ---:| ---:| ---:|
| baked_bell_pepper | 2 | 100.0% | 0.0% | 0.0% | 0.0% |
| baked_sweet_potato | 10 | 0.0% | 40.0% | 40.0% | 20.0% |
| boiled_egg | 11 | 45.5% | 0.0% | 9.1% | 45.5% |
| boiled_mushroom | 10 | 30.0% | 0.0% | 30.0% | 40.0% |
| boiled_sweet_potato | 11 | 27.3% | 9.1% | 45.5% | 18.2% |

## Aggregate (Level 1, May batch)

| Metric | Count | % of units |
| --- | ---:| ---:|
| Request units | 44 | 100% |
| Effective | 13 | **29.5%** |
| Assisted | 5 | **11.4%** |
| Redundant | 13 | **29.5%** |
| Ineffective | 13 | **29.5%** |

## Command

```bash
python3 src/eval_result/audit_tgco.py --may-l1-gpt4o --per-file
```
