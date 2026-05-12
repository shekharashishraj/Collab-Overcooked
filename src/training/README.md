# RL Training Pipeline for Collab-Overcooked SLMs

Two-stage pipeline to lift Qwen2.5-7B-Instruct from ~8% / 0% Success Rate at L1
/ L2 toward closed-source-class performance, using your existing 119 successful
GPT-4o trajectories.

## What's in this folder

```
src/training/
├── extract_sft_data.py    # builds SFT JSONL from data/gpt-4o/*/experiment_*.json
├── sft_train.py           # LoRA SFT via TRL SFTTrainer
├── reward.py              # episode reward: success + time + verr + ferr + red + follow
├── rollout_env.py         # subprocess wrapper around src/main.py
├── grpo_train.py          # custom GRPO loop (multi-turn, group-relative advantage)
├── eval_trained.py        # delegates to src/evaluation.py + adds reward.py columns
├── serve_vllm.sh          # vLLM launcher with --enable-lora
├── run_pipeline.sh        # end-to-end orchestrator
├── configs/
│   ├── sft_qwen7b.yaml
│   └── grpo_qwen7b.yaml
└── data/                  # populated by extract_sft_data.py
```

## Setup (once, on the H200 box)

```bash
conda activate collab-overcooked
pip install -r src/training/requirements_training.txt

# Smoke-test the existing rollout loop still works.
cd src && python main.py --horizon 3 --order boiled_egg --gpt_model gpt-3.5-turbo-0125
cd ..
```

## Stage 1 — SFT data + LoRA SFT

```bash
python src/training/extract_sft_data.py
# expect: ~5800 train pairs, ~840 holdout pairs (boiled_egg + baked_potato_slices)

python src/training/sft_train.py --config src/training/configs/sft_qwen7b.yaml
# saves LoRA adapter -> ckpt/qwen7b-sft
```

Smoke variant — 100 steps, ~10 minutes:
```bash
python src/training/sft_train.py \
    --config src/training/configs/sft_qwen7b.yaml \
    --max_steps 100 \
    --output_dir ckpt/qwen7b-sft-smoke
```

## Stage 2 — vLLM server + GRPO

In one shell:
```bash
bash src/training/serve_vllm.sh ckpt/qwen7b-sft 8000
```

In another shell:
```bash
python src/training/grpo_train.py --config src/training/configs/grpo_qwen7b.yaml
# saves -> ckpt/qwen7b-grpo/step-{10,20,...}  and ckpt/qwen7b-grpo/final
```

GRPO is the dominant cost (~50h for 200 outer steps with 14 tasks × 4 seeds).
Bring it down by reducing `tasks` and `outer_steps` in the config for a first run.

Hot-reloading the LoRA adapter on a live vLLM (requires
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` — already set by `serve_vllm.sh`) is
attempted automatically by `grpo_train.py`. If your vLLM build doesn't support
the dynamic endpoint, the trainer warns and continues; restart vLLM manually
between checkpoints.

## Stage 3 — Evaluation

Point vLLM at the final adapter (restart it):
```bash
bash src/training/serve_vllm.sh ckpt/qwen7b-grpo/final 8000
```

Then:
```bash
python src/training/eval_trained.py \
    --policy qwen-policy \
    --server http://localhost:8000/v1 \
    --levels 1 2 \
    --episodes 10

# Aggregate using the project's existing scripts:
cd src && python organize_result.py && python convert_result.py
```

You get:
- `src/eval_result/converted_data.csv` — level-aggregated SR / F1 / similarity / redundancy / initiate_collaboration / respond_collaboration (the official numbers).
- `src/eval_result/<tag>_summary.json` — our additional reward.py columns (`reward_chef`, `reward_assistant`, `follow_*`, `verr_*`).

## Targets (from the project plan)

| Stage         | L1 SR target | L2 SR target | Notes |
|---------------|--------------|--------------|-------|
| Baseline      | 8%           | 0%           | Qwen2.5-7B-Instruct as reported in *Overlooked* |
| After SFT     | ≥ 35%        | ≥ 10%        | Pure behavior cloning on 5.8k pairs |
| After GRPO    | ≥ 55%        | ≥ 25%        | Plus ADR ≥ 0.5, FollowRate ≥ 0.65 in self-play |

If GRPO underperforms SFT, increase `reward.w_verr` and `reward.w_red` in the
GRPO config so dense per-step signals dominate the sparse success reward.

## Reusing for Llama-3.1-8B

Change `base_model` in both YAMLs to `meta-llama/Llama-3.1-8B-Instruct` and
re-run. The data is model-agnostic; the LoRA target modules are the same names
on Llama.

## Known sharp edges

- `rollout_env.py` spawns `src/main.py` subprocesses; the existing rollout loop
  imports `tensorflow` at module load (CPU-only via `CUDA_VISIBLE_DEVICES=-1`).
  Each subprocess costs ~3s of import overhead; with `max_parallel=8`, that's
  hidden behind the LLM call latency.
- The GRPO trainer co-locates the trainable LoRA *and* the reference model on
  the same GPU as vLLM (H200 has the headroom — 141 GB). If you ever switch to
  H100 80 GB, push vLLM onto a second GPU or drop the reference and accept a
  bias in the KL estimator.
- `serve_vllm.sh` sets `--gpu-memory-utilization 0.55` to leave room for the
  trainer. Bump it back up to 0.85+ during pure evaluation.
- `prompts/recipe` lookup in `extract_sft_data.py` is by case-insensitive
  substring; if you add a new task whose name is a prefix of another, fix the
  match in `_load_recipe`.
