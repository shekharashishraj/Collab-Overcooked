# RL Training Pipeline for Collab-Overcooked SLMs

Two-stage pipeline to lift Qwen2.5-7B-Instruct from ~8% / 0% Success Rate at L1
/ L2 toward closed-source-class performance, using the existing 119 successful
GPT-4o trajectories in [`src/data/gpt-4o/`](../data/gpt-4o/).

For the H200 operator workflow (background processes, tmux, GPU sharing), see
[`../../CLAUDE.md`](../../CLAUDE.md).

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
├── run_pipeline.sh        # end-to-end orchestrator (resumable via SKIP_* env vars)
├── configs/
│   ├── sft_qwen7b.yaml
│   └── grpo_qwen7b.yaml
└── data/                  # populated by extract_sft_data.py (gitignored)
```

## Setup — isolated conda env on H200

The original project pins **Python 3.8**, but Qwen2.5 / TRL / vLLM ≥ 0.6 need
**Python ≥ 3.10**. Use a fresh env so the training stack does not collide with
any existing `collab-overcooked` installation.

```bash
# 1. Fresh env
conda create -n collab-rl python=3.11 -y
conda activate collab-rl

# 2. Base project deps + overcooked_ai
pip install -r requirements.txt
pip install -e lib/overcooked_ai

# 3. Training-stack deps (Torch / TRL / PEFT / vLLM)
pip install -r src/training/requirements_training.txt

# 4. CUDA visibility check — should print the H200
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 5. Smoke-test the existing rollout still runs end-to-end
cd src && python main.py --horizon 3 --order boiled_egg --gpt_model gpt-3.5-turbo-0125
cd ..
```

## Stage 1 — SFT data + LoRA SFT

```bash
python src/training/extract_sft_data.py
# expect: ~5800 train pairs, ~840 holdout pairs (boiled_egg + baked_potato_slices)

python src/training/sft_train.py --config src/training/configs/sft_qwen7b.yaml
# saves LoRA adapter -> ckpt/qwen7b-sft  (≈6h for 3 epochs on H200)
```

Smoke variant — 100 steps, ~10 minutes:
```bash
python src/training/sft_train.py \
    --config src/training/configs/sft_qwen7b.yaml \
    --max_steps 100 \
    --output_dir ckpt/qwen7b-sft-smoke
```

## Stage 2 — vLLM server + GRPO

Two long-running processes share the H200. Run them in **separate tmux panes**
(or one of them via `nohup &`). The vLLM server caps its memory at 0.55 of the
H200 so the trainer's reference + policy fit alongside.

Pane A:
```bash
bash src/training/serve_vllm.sh ckpt/qwen7b-sft 8000
```

Pane B:
```bash
python src/training/grpo_train.py --config src/training/configs/grpo_qwen7b.yaml
# saves -> ckpt/qwen7b-grpo/step-{10,20,...}  and ckpt/qwen7b-grpo/final
```

GRPO is the dominant cost (~50h for 200 outer steps × 14 tasks × G=4 rollouts).
First-pass tuning: drop `outer_steps` to 30 and `tasks` to 4 in the config to
get end-to-end timing data before committing to the full run.

Hot-reloading the LoRA adapter on a live vLLM (requires
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` — already set by `serve_vllm.sh`) is
attempted automatically by `grpo_train.py`. If your vLLM build doesn't support
the dynamic endpoint, the trainer warns and continues; restart vLLM manually
between checkpoints.

## Stage 3 — Evaluation

Point vLLM at the final adapter (restart pane A):
```bash
bash src/training/serve_vllm.sh ckpt/qwen7b-grpo/final 8000
```

Then in pane B:
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
- `src/eval_result/<tag>_summary.json` — additional reward.py columns (`reward_chef`, `reward_assistant`, `follow_*`, `verr_*`).

## Targets

| Stage         | L1 SR target | L2 SR target | Notes |
|---------------|--------------|--------------|-------|
| Baseline      | 8%           | 0%           | Qwen2.5-7B-Instruct as reported in *Overlooked* |
| After SFT     | ≥ 35%        | ≥ 10%        | Pure behavior cloning on 5.8k pairs |
| After GRPO    | ≥ 55%        | ≥ 25%        | Plus ADR ≥ 0.5, FollowRate ≥ 0.65 in self-play |

If GRPO underperforms SFT, increase `reward.w_verr` and `reward.w_red` in
`configs/grpo_qwen7b.yaml` so dense per-step signals dominate the sparse
success reward.

## Resuming after a crash

Each stage of `run_pipeline.sh` is gated by a `SKIP_*` env var:

```bash
SKIP_EXTRACT=1 SKIP_SFT=1 bash src/training/run_pipeline.sh   # only smoke + GRPO + eval
SKIP_EXTRACT=1 SKIP_SFT=1 SKIP_GRPO=1 bash src/training/run_pipeline.sh   # eval only
```

`grpo_train.py` itself resumes from the latest checkpoint under `ckpt/qwen7b-grpo/`
if you pass `--config` pointing at the same `output_dir`. Just edit the config's
`sft_adapter` to the last saved step and re-run.

## Reusing for Llama-3.1-8B

Change `base_model` in both YAMLs to `meta-llama/Llama-3.1-8B-Instruct` and
re-run. The data is model-agnostic; the LoRA target modules are the same names
on Llama. Expect comparable SFT timing and slightly worse GRPO sample
efficiency (Llama-3.1 has a weaker tool-use prior than Qwen2.5).

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
- `openai_key.txt` is read at import time by `src/collab/modules.py`.
  `rollout_env.py` writes a stub key if missing — safe to ignore the warning.
