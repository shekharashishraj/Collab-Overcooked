---
name: paper-experiment-commands
overview: Run the paper's required RC experiments with two episodes per dish per level, organized by Results section/table. NRC is explicitly deferred because the current repo exposes only `new_env`; Gemma-31B commands are separated for later H200 execution.
todos:
  - id: run-table1-closed
    content: Run Table 1 RC self-play for GPT-4o and Sonnet-4 across all 30 dishes with 2 episodes each.
    status: pending
  - id: run-table2-closed
    content: Run Table 2 closed-model directed cross-play for GPT-4o/Sonnet-4 and Sonnet-4/GPT-4o.
    status: pending
  - id: run-table4-closed
    content: Run Table 4 Pro-Agent and AToM self-play for GPT-4o and Sonnet-4.
    status: pending
  - id: aggregate-closed
    content: Aggregate closed-model logs into Tables 1-5 and Figure 1 metrics.
    status: pending
  - id: run-gemma-h200
    content: Run Gemma-31B H200 self-play, cross-play, Pro-Agent, and AToM command blocks later.
    status: pending
  - id: defer-nrc
    content: Defer NRC rows until an NRC layout or command-line flag exists.
    status: pending
isProject: false
---

# Experiment Command Plan

## Preconditions
- Run commands from the repo source directory: `cd /Users/ashishrajshekhar/Desktop/Collab-Overcooked/src`.
- Current executable layout is only `--layout new_env`, so these commands replace **RC** placeholder values only.
- Use `--episode 2` to satisfy “2 runs of each dish in every level.” There are 30 dishes total, so each model/framework/pairing cell produces 60 episodes.
- Use separate `--statistics_save_dir` folders per result table so aggregation is clean.
- Gemma-31B is deferred to H200 and uses OpenAI-compatible vLLM-style serving via `--openai_base_url` and `--local_server_api` placeholders.

## Shared Dish Lists
Use these shell arrays in each section:

```bash
LEVEL1=(boiled_egg boiled_mushroom boiled_sweet_potato baked_sweet_potato baked_bell_pepper)
LEVEL2=(boiled_potato_slices boiled_green_bean_slices boiled_corn_slices baked_potato_slices baked_pumpkin_slices)
LEVEL3=(baked_bell_pepper_soup baked_carrot_soup baked_mushroom_soup baked_potato_soup baked_pumpkin_soup)
LEVEL4=(sliced_bell_pepper_and_corn_stew sliced_bell_pepper_and_lentil_stew sliced_eggplant_and_chickpea_stew sliced_pumpkin_and_chickpea_stew sliced_zucchini_and_chickpea_stew)
LEVEL5=(mashed_broccoli_and_bean_patty mashed_carrot_and_chickpea_patty mashed_cauliflower_and_lentil_patty mashed_potato_and_pea_patty mashed_sweet_potato_and_bean_patty)
LEVEL6=(potato_carrot_and_onion_patty romaine_lettuce_pea_and_tomato_patty sweet_potato_spinach_and_mushroom_patty taro_bean_and_bell_pepper_patty zucchini_green_pea_and_onion_patty)
ALL_DISHES=("${LEVEL1[@]}" "${LEVEL2[@]}" "${LEVEL3[@]}" "${LEVEL4[@]}" "${LEVEL5[@]}" "${LEVEL6[@]}")
```

## Results 5.1 / Table 1: RC Task Completion by Model
Purpose: replace RC rows for success, completion time/timesteps, and token cost. NRC rows are blocked until an NRC layout/script exists.

### GPT-4o RC Self-Play
```bash
mkdir -p data_runs/table1_rc
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 gpt-4o --model_p1 gpt-4o \
    --agent_p0 baseline --agent_p1 baseline \
    --statistics_save_dir data_runs/table1_rc
 done
```

### Sonnet-4 RC Self-Play
```bash
mkdir -p data_runs/table1_rc
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 claude-sonnet-4-20250514 --model_p1 claude-sonnet-4-20250514 \
    --agent_p0 baseline --agent_p1 baseline \
    --statistics_save_dir data_runs/table1_rc
 done
```

### Gemma-31B RC Self-Play, H200 Later
```bash
mkdir -p data_runs/table1_rc_gemma
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 google/gemma-3-31b-it --model_p1 google/gemma-3-31b-it \
    --agent_p0 baseline --agent_p1 baseline \
    --openai_base_url http://localhost:8000/v1 \
    --local_server_api http://localhost:8000/v1 \
    --statistics_save_dir data_runs/table1_rc_gemma
 done
```

## Results 5.2 / Table 2: Role-Sensitive Cross-Model Play
Purpose: replace the 9 role-sensitive pairings. The self-play rows can reuse Table 1 logs; run the cross rows below.

### Closed-Model Cross Pairings Now
```bash
mkdir -p data_runs/table2_cross
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 gpt-4o --model_p1 claude-sonnet-4-20250514 \
    --agent_p0 baseline --agent_p1 baseline \
    --statistics_save_dir data_runs/table2_cross

  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 claude-sonnet-4-20250514 --model_p1 gpt-4o \
    --agent_p0 baseline --agent_p1 baseline \
    --statistics_save_dir data_runs/table2_cross
 done
```

### Gemma-31B Cross Pairings, H200 Later
```bash
mkdir -p data_runs/table2_cross_gemma
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 gpt-4o --model_p1 google/gemma-3-31b-it \
    --agent_p0 baseline --agent_p1 baseline \
    --openai_base_url http://localhost:8000/v1 --local_server_api http://localhost:8000/v1 \
    --statistics_save_dir data_runs/table2_cross_gemma

  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 google/gemma-3-31b-it --model_p1 gpt-4o \
    --agent_p0 baseline --agent_p1 baseline \
    --openai_base_url http://localhost:8000/v1 --local_server_api http://localhost:8000/v1 \
    --statistics_save_dir data_runs/table2_cross_gemma

  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 claude-sonnet-4-20250514 --model_p1 google/gemma-3-31b-it \
    --agent_p0 baseline --agent_p1 baseline \
    --openai_base_url http://localhost:8000/v1 --local_server_api http://localhost:8000/v1 \
    --statistics_save_dir data_runs/table2_cross_gemma

  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 google/gemma-3-31b-it --model_p1 claude-sonnet-4-20250514 \
    --agent_p0 baseline --agent_p1 baseline \
    --openai_base_url http://localhost:8000/v1 --local_server_api http://localhost:8000/v1 \
    --statistics_save_dir data_runs/table2_cross_gemma
 done
```

## Results 5.3 / Table 3: Per-Level Structured Interdependence
No new rollouts needed. Compute Table 3 by aggregating the Table 2 baseline logs by level:
- Level 1: `LEVEL1`
- Level 2: `LEVEL2`
- Level 3: `LEVEL3`
- If keeping all six levels in the paper, extend the table to Levels 1-6 using all arrays above.

Use logs from:
- `data_runs/table1_rc` for GPT/GPT and Sonnet/Sonnet self-play
- `data_runs/table2_cross` for GPT/Sonnet directed cross-play
- `data_runs/table1_rc_gemma` and `data_runs/table2_cross_gemma` after H200 Gemma runs

## Results 5.4 / Table 4: Baseline vs Pro-Agent vs AToM
Purpose: replace model/framework rows. Baseline rows can reuse Table 1 self-play logs; run Pro-Agent and AToM self-play for each model.

### GPT-4o Pro-Agent and AToM
```bash
mkdir -p data_runs/table4_frameworks
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 gpt-4o --model_p1 gpt-4o \
    --agent_p0 proagent --agent_p1 proagent \
    --statistics_save_dir data_runs/table4_frameworks

  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 gpt-4o --model_p1 gpt-4o \
    --agent_p0 a-tom --agent_p1 a-tom \
    --statistics_save_dir data_runs/table4_frameworks
 done
```

### Sonnet-4 Pro-Agent and AToM
```bash
mkdir -p data_runs/table4_frameworks
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 claude-sonnet-4-20250514 --model_p1 claude-sonnet-4-20250514 \
    --agent_p0 proagent --agent_p1 proagent \
    --statistics_save_dir data_runs/table4_frameworks

  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 claude-sonnet-4-20250514 --model_p1 claude-sonnet-4-20250514 \
    --agent_p0 a-tom --agent_p1 a-tom \
    --statistics_save_dir data_runs/table4_frameworks
 done
```

### Gemma-31B Pro-Agent and AToM, H200 Later
```bash
mkdir -p data_runs/table4_frameworks_gemma
for dish in "${ALL_DISHES[@]}"; do
  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 google/gemma-3-31b-it --model_p1 google/gemma-3-31b-it \
    --agent_p0 proagent --agent_p1 proagent \
    --openai_base_url http://localhost:8000/v1 --local_server_api http://localhost:8000/v1 \
    --statistics_save_dir data_runs/table4_frameworks_gemma

  python main.py --layout new_env --order "$dish" --episode 2 --horizon 120 \
    --model_p0 google/gemma-3-31b-it --model_p1 google/gemma-3-31b-it \
    --agent_p0 a-tom --agent_p1 a-tom \
    --openai_base_url http://localhost:8000/v1 --local_server_api http://localhost:8000/v1 \
    --statistics_save_dir data_runs/table4_frameworks_gemma
 done
```

## Results 5.5 / Figure 1: TGCO Audit
No extra rollouts needed. Compute Figure 1 from the same baseline cross-model logs used for Table 2:
- `data_runs/table1_rc`
- `data_runs/table2_cross`
- later `data_runs/table1_rc_gemma`
- later `data_runs/table2_cross_gemma`

Metrics to aggregate per pairing:
- Redundant action/message rate
- Delayed or unfulfilled request rate
- Verifier calls / validator errors
- IDensity

## Results 5.6: Qualitative Case Studies
No grid run needed. Select representative successful-but-inefficient traces after aggregation:
- Case Study 1: Sonnet/Sonnet `boiled_egg` from Table 1 logs.
- Case Study 2: Gemma/GPT-4o or GPT-4o/Gemma from H200 cross logs after Gemma runs.

## Appendix B.1 / Table 5: Per-Level Scaffold Breakdown
No additional commands beyond Table 4. Aggregate Table 4 framework logs by level:
- Baseline: reuse `data_runs/table1_rc`
- Pro-Agent/AToM: use `data_runs/table4_frameworks`
- Gemma later: use `data_runs/table4_frameworks_gemma`

## Appendix B.2 / Table 6: Fixed-Window ADR Robustness
No additional rollouts. Recompute ADR twice on the same Table 2 logs:
- Level-normalized window: current paper method
- Fixed window: `w = 20`

## Execution Order
1. Run Table 1 closed-model RC self-play: GPT-4o and Sonnet-4.
2. Run Table 2 closed-model cross-play: GPT-4o/Sonnet-4 and Sonnet-4/GPT-4o.
3. Run Table 4 closed-model framework runs: GPT-4o and Sonnet-4 with Pro-Agent and AToM.
4. Aggregate Tables 1-5 and Figure 1 for closed models.
5. Later on H200, run Gemma commands for Tables 1, 2, and 4.
6. After NRC layout/script exists, run the same Table 1 grid with `--layout <nrc_layout>` or the future NRC flag and replace NRC rows.

## Important Notes
- `main.py` currently supports `--agent_p0/--agent_p1` choices: `baseline`, `proagent`, `a-tom`, `reflexion`.
- `main.py` currently supports only `--layout new_env`, so NRC cannot be honestly produced from the current CLI.
- The paper’s Table 2 says 9 role-sensitive pairings; with Gemma deferred, closed-model execution covers 4 of 9 rows now: GPT/GPT, GPT/Sonnet, Sonnet/GPT, Sonnet/Sonnet.
- The paper says 3 levels, but your repo has 6 level prefixes in `src/prompts/recipe`. Since you asked for every dish in every level, the commands run all 6 levels. You can report only Levels 1-3 or update the paper to Levels 1-6.