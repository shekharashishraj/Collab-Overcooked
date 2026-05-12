# CLAUDE.md — Operator Guide for the H200 Training Box

This file is loaded automatically by Claude Code at the start of every session
in this repo. It describes **how to run the SFT+GRPO pipeline on the H200**
in an isolated conda env, and the conventions Claude should follow when
operating the box.

If you are reading this and you are *not* Claude Code, treat it as the
human-facing H200 runbook.

---

## Target model: `Qwen/Qwen3.5-9B` (base)

The default training target is the **base** Qwen3.5-9B checkpoint (no
Instruct/Chat suffix). A few consequences:

- **No chat template ships with the base tokenizer.** SFT installs Qwen's
  standard ChatML template via `src/training/chat_template.py` and saves the
  patched tokenizer alongside the LoRA adapter. vLLM (`serve_vllm.sh`) is
  pointed at the adapter dir for the tokenizer so the same template is used
  at rollout time.
- **Thinking mode is disabled.** Qwen3 supports `<think>...</think>` blocks;
  the SFT data (from GPT-4o) has no thinking traces, so we omit them from the
  template to avoid teaching the model an unused token.
- **Zero-shot baseline ≈ 0%.** Because there's no chat template out of the box,
  the model can't follow our `{Role} analysis / plan / say` format until SFT
  has run. Don't report a "Qwen3.5-9B baseline" without first running SFT.
- **9B fits on H200 comfortably.** Full FT is possible; LoRA (the default) is
  preferred for faster iteration. The reference model in `grpo_train.py`
  co-locates on the same GPU.

If the user asks to switch to an Instruct variant (e.g. `Qwen3.5-9B-Instruct`),
the helper is idempotent — it leaves an existing chat template alone — so the
only edit needed is `model_name_or_path` / `base_model` in the two YAML configs.

## Box assumptions

- Single GPU: NVIDIA H200 (141 GB HBM3e), CUDA 12.x driver.
- Disk: ≥ 200 GB free under `$HOME` (for model weights + checkpoints + rollout JSONs).
- Network: outbound HTTPS to `huggingface.co` and `pypi.org` is available.
- Long-running jobs are expected — use `tmux` (or `screen`) so SSH disconnects don't kill them.

If any of these are false, **stop and ask the user** before changing the plan.

---

## Environment isolation: `collab-rl`

The original project pins **Python 3.8**; the training stack needs **≥ 3.10**.
The two stacks live in separate conda envs so they don't collide:

| Env | Python | Used for |
|---|---|---|
| `collab-overcooked` (optional, existing) | 3.8 | Original GPT-4o/GPT-3.5 rollouts as published |
| `collab-rl` (this guide) | **3.11** | SFT, GRPO, vLLM serving, eval against trained Qwen |

For this work, **always activate `collab-rl`** before running anything under
`src/training/`. The pipeline calls `python main.py` as a subprocess and that
subprocess inherits the active env — so a single `collab-rl` env is the
simplest and the recommended setup.

### One-shot setup

```bash
cd ~/Collab-Overcooked   # or wherever the repo is cloned

# 1. Fresh env
conda create -n collab-rl python=3.11 -y
conda activate collab-rl

# 2. Project deps + overcooked_ai (editable install)
pip install -r requirements.txt
pip install -e lib/overcooked_ai

# 3. Training stack
pip install -r src/training/requirements_training.txt

# 4. Sanity checks
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
python -c "import trl, peft, vllm; print('trl', trl.__version__, 'peft', peft.__version__, 'vllm', vllm.__version__)"
```

If `mpi4py` install fails on Py 3.11, **skip it** — it is not needed for the
RL pipeline. The original README only requires it for some research utilities.

---

## Disk layout the pipeline assumes

```
~/Collab-Overcooked/                  # repo root, $REPO
├── src/data/gpt-4o/...               # 119 expert trajectories (committed)
├── src/training/data/*.jsonl         # SFT corpus (regenerated, gitignored)
├── ckpt/qwen3p5-9b-sft/                  # SFT LoRA adapter
├── ckpt/qwen3p5-9b-grpo/step-*/          # GRPO checkpoints (every save_every outer steps)
├── ckpt/qwen3p5-9b-grpo/final/           # final adapter
├── src/data/<run_tag>/...            # per-rollout output JSONs (cleanable)
└── src/eval_result/...               # final per-task metric CSVs
```

`src/data/<run_tag>/` accumulates rollout JSONs across GRPO and eval — easily
hundreds of MB. Wipe with:

```bash
find src/data -maxdepth 1 -type d -name "grpo-*" -mtime +1 -exec rm -rf {} +
find src/data -maxdepth 1 -type d -name "eval-*" -mtime +1 -exec rm -rf {} +
```

---

## Running the pipeline — recommended flow

The pipeline has **three long-running stages** that share the GPU. Use two
tmux panes (server + trainer) plus a scratch pane for inspection.

### Pane 0 (scratch) — extract SFT data + run SFT

```bash
conda activate collab-rl
cd ~/Collab-Overcooked

python src/training/extract_sft_data.py
# expect: total pairs : ~6673, chef pairs : ~3994, assistant pairs : ~1835

# Smoke first: 100 steps in ~10 min
python src/training/sft_train.py \
    --config src/training/configs/sft_qwen3p5_9b.yaml \
    --max_steps 100 \
    --output_dir ckpt/qwen3p5-9b-sft-smoke

# Full SFT: ~6h
python src/training/sft_train.py --config src/training/configs/sft_qwen3p5_9b.yaml
```

Watch SFT progress: `tail -f ckpt/qwen3p5-9b-sft/trainer_state.json` and
`nvidia-smi -l 5`.

### Pane A (server) — vLLM with LoRA hot-reload

```bash
conda activate collab-rl
cd ~/Collab-Overcooked
bash src/training/serve_vllm.sh ckpt/qwen3p5-9b-sft 8000
```

Verify it's up from pane 0:

```bash
curl -s -H "Authorization: Bearer token-abc123" \
    http://localhost:8000/v1/models | python -m json.tool
```

You should see `qwen-policy` in the list.

### Pane B (trainer) — GRPO

```bash
conda activate collab-rl
cd ~/Collab-Overcooked

# Smoke: one outer step, single task, group of 2 rollouts
python src/training/grpo_train.py \
    --config src/training/configs/grpo_qwen3p5_9b.yaml \
    --max_outer_steps 1 --tasks boiled_egg --group_size 2

# Full GRPO: ~50h on H200
python src/training/grpo_train.py --config src/training/configs/grpo_qwen3p5_9b.yaml
```

Trainer logs land in `ckpt/qwen3p5-9b-grpo/train_log.jsonl` (one JSON per outer
step with reward/success/loss/KL for each role).

### Eval pass

When GRPO finishes, restart pane A pointing at the final adapter:

```bash
bash src/training/serve_vllm.sh ckpt/qwen3p5-9b-grpo/final 8000
```

Then in pane B:

```bash
python src/training/eval_trained.py \
    --policy qwen-policy \
    --levels 1 2 \
    --episodes 10
cd src && python organize_result.py && python convert_result.py
```

---

## Guidance for Claude Code when operating this repo

These are rules Claude should follow when working on this codebase, *especially*
when invoked on the H200 box.

1. **Always activate `collab-rl` first.** If a command fails with
   `ModuleNotFoundError: No module named 'trl'` or similar, the env is wrong.
   Don't pip-install into the base env to "fix" it — re-activate.

2. **Long-running jobs go in the background.** SFT is ~6h, GRPO is ~50h. When
   running them from Claude Code, use `Bash` with `run_in_background: true` and
   stream logs from the resulting file. Never block the foreground tool call
   for hours.

3. **Don't kill the vLLM server casually.** Reloading the model takes ~1 min.
   Before you `pkill vllm`, confirm with the user that GRPO/eval is paused.

4. **GPU memory is shared.** `serve_vllm.sh` reserves 0.55 of the H200, leaving
   ~63 GB for the trainer. If you change `--gpu-memory-utilization`, also
   change the trainer's batch sizes — otherwise OOM. For *pure evaluation* (no
   trainer running), bump it to 0.85.

5. **Trajectory JSONs are recoverable, checkpoints are not.** Don't delete
   anything under `ckpt/`. You can delete anything under `src/data/<run_tag>/`
   that's > 1 day old.

6. **The pipeline must not modify `src/main.py`, `src/collab/`, `src/evaluation.py`,
   `src/eval_utils.py`, `src/organize_result.py`, or `src/convert_result.py`.**
   These are the original benchmark; we wrap them, never edit them. If you find
   yourself wanting to edit one, stop and propose the change to the user first.

7. **When SFT loss plateaus or GRPO advantages collapse to zero**, the first
   knobs to turn are documented in [`src/training/README.md`](src/training/README.md):
   reward weights, `rollout_temperature`, `group_size`. Do not change the
   policy architecture or LoRA rank without discussing with the user.

8. **Verify before claiming "training is working".** "Loss went down" is not
   evidence. The pipeline-level checks that count:
   - SFT: sample from `ckpt/qwen3p5-9b-sft` on a held-out user prompt and check the
     output parses cleanly as `{Role} analysis: ... plan: ... say: ...`.
   - GRPO: check `ckpt/qwen3p5-9b-grpo/train_log.jsonl` — `roleN_mean_success`
     must trend up across at least 10 outer steps before claiming RL is
     working.
   - Eval: actual L1/L2 SR numbers from `converted_data.csv`, not internal
     training metrics.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: overcooked_ai_py` | `pip install -e lib/overcooked_ai` not run in `collab-rl` | Re-run the install in the right env. |
| `openai.AuthenticationError: ...` from a Qwen rollout | `openai_key.txt` is checked even for open-source models | `rollout_env.py` auto-stubs it; if you see this from `main.py` directly, `echo sk-stub > src/openai_key.txt` |
| `RuntimeError: CUDA out of memory` in trainer | vLLM is using too much GPU | Lower `--gpu-memory-utilization` in `serve_vllm.sh` (e.g. 0.45) |
| GRPO rollouts all return `score=0` | SFT was insufficient or temperature too low | Verify SFT smoke first; raise `rollout_temperature` to 1.0–1.1 |
| `vllm: command not found` in `serve_vllm.sh` | Wrong env or vllm not installed | `conda activate collab-rl && pip install vllm>=0.6.3` |
| LoRA hot-reload prints a WARN | vLLM build doesn't support the dynamic endpoint | Restart vLLM manually with the new adapter path; the trainer continues |
| `ConnectionRefusedError` on `localhost:8000` | vLLM server not up yet | Wait for "Uvicorn running on" line in pane A; ~60s on first launch |

---

## Quick checks Claude should run on session start

If a user asks "is the box set up correctly" or starts a session implying
they're on the H200, run these in parallel and report:

```bash
conda env list | grep collab-rl                              # env exists
which python && python --version                              # 3.11 on collab-rl
python -c "import torch; print(torch.cuda.get_device_name(0))"  # H200 visible
python -c "import trl, peft, vllm" && echo OK                 # stack installed
test -f ckpt/qwen3p5-9b-sft/adapter_model.safetensors && echo "SFT done"
test -f ckpt/qwen3p5-9b-grpo/final/adapter_model.safetensors && echo "GRPO done"
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/models  # 200 = vLLM up, 000 = down
```
