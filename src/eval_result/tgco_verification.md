# TGCO table vs experiment logs — verification

## What `tgco.md` is tied to

The prose says a **9-run sample** with **three logs per level** and gives token totals:

| Level | Total tokens (md) |
|-------|-------------------|
| 1 | 5,896 |
| 2 | 34,620 |
| 3 | 43,443 |

Those totals match (within rounding) **nine JSON files under `src/data/gpt-4o/` dated 2026-03-28**, not the May 2026 `src/data_runs/l1_gpt4o/` tree.

Closest triples by sum of `statistical_data.communication` token counts:

- **Level 1:** `experiment_2026-03-28_10-40-32_boiled_egg.json`, `..._10-40-34_boiled_sweet_potato.json`, `..._10-40-35_baked_sweet_potato.json` → sum **5,874** (target 5,896).
- **Level 2:** `..._12-10-05_boiled_green_bean_slices.json`, `..._12-10-05_boiled_potato_slices.json`, `..._12-15-26_baked_potato_slices.json` → sum **34,619** (target 34,620).
- **Level 3:** `..._13-53-34_baked_bell_pepper_soup.json`, `..._13-53-34_baked_potato_soup.json`, `..._14-19-21_baked_pumpkin_soup.json` → sum **43,446** (target 43,443).

So the **narrative and token table** in `tgco.md` are internally consistent with that **March self-play GPT‑4o** batch.

## What is *not* the same corpus

- **`src/data_runs/l1_gpt4o/`** (May 2026): five Level‑1 GPT‑4o runs — **different dates, different traces**, not the three Level‑1 files above.
- **`reuslts /nonrc_table1/`**: copies of some May runs; same story.
- **`tgco_spider.png`**: plots the **percentages written in `tgco.md`**, which come from the **manual** TGCO audit, not from a script checked into this repo.

## Automated re-audit (approximate)

`audit_tgco.py` implements a **rough** rule-based pass: only explicit `request('...')` strings in `say`/`plan`, 15-step window, validator buckets as in the md. It does **not** implement natural-language request units or the exact redundancy / “repeat pattern” rules described in prose.

On the **same nine March files**, it does **not** reproduce the md’s per-level or overall percentages (e.g. overall Effective ≈ **12.4%** vs **12.7%** in md — close — but Assisted / Redundant / Ineffective splits differ a lot). So the **spider chart should not be regenerated from `audit_tgco.py`** unless you first align the script with the paper’s manual coding protocol.

## Tests

There are **no pytest/unittest files** in this repository that assert TGCO counts.

## Recommendation

1. **For the paper:** keep a small manifest (paths + hashes) of the nine JSON files used for the TGCO subsection.
2. **To refresh the figure:** either re-hand-code the table from the same protocol, or extend `audit_tgco.py` until its counts match a labeled gold subset, then regenerate the spider chart from script output.
