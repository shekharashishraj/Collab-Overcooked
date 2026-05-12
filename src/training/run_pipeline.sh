#!/usr/bin/env bash
# End-to-end pipeline orchestrator for Collab-Overcooked SLM RL training.
#
# Run this from the repo root on the H200 box. It assumes:
#   - Conda env `collab-rl` is active (Python 3.11 + training stack).
#     See CLAUDE.md / src/training/README.md for the one-shot setup.
#   - `vllm serve ...` will be started in a SEPARATE shell when prompted.
#
# Stages (each guarded by --skip-* flags so you can resume after a failure):
#   1. SFT data extraction
#   2. LoRA SFT
#   3. Smoke rollout against the SFT adapter (sanity check)
#   4. GRPO online RL
#   5. Final evaluation

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- env sanity ---------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "collab-rl" ]]; then
  echo "[run_pipeline] WARNING: active conda env is '${CONDA_DEFAULT_ENV:-none}', expected 'collab-rl'."
  echo "[run_pipeline] See CLAUDE.md for one-shot setup. Continuing in 5s; Ctrl-C to abort..."
  sleep 5
fi
python - <<'PY' || { echo "[run_pipeline] training stack missing; aborting"; exit 1; }
import importlib, sys
for m in ("torch", "transformers", "peft", "trl", "datasets", "yaml"):
    try:
        importlib.import_module(m)
    except Exception as e:
        print(f"[run_pipeline] missing dep '{m}': {e}", file=sys.stderr)
        sys.exit(1)
PY
# ------------------------------------------------------------------------------

SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_GRPO="${SKIP_GRPO:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

SFT_CFG="src/training/configs/sft_qwen3p5_9b.yaml"
GRPO_CFG="src/training/configs/grpo_qwen3p5_9b.yaml"

echo "=== [1/5] SFT data extraction ==="
if [[ "$SKIP_EXTRACT" != "1" ]]; then
  python src/training/extract_sft_data.py
else
  echo "skipped"
fi

echo
echo "=== [2/5] LoRA SFT on Qwen/Qwen3.5-9B (base) (H200, ~8h for 3 epochs) ==="
if [[ "$SKIP_SFT" != "1" ]]; then
  python src/training/sft_train.py --config "$SFT_CFG"
else
  echo "skipped"
fi

echo
echo "=== START VLLM IN ANOTHER SHELL NOW: ==="
echo "    bash src/training/serve_vllm.sh ckpt/qwen3p5-9b-sft 8000"
echo "Press ENTER to continue once vLLM is up and serving on :8000 ..."
read -r _ignore

echo
echo "=== [3/5] Smoke rollout against SFT adapter ==="
if [[ "$SKIP_SMOKE" != "1" ]]; then
  python src/training/rollout_env.py \
      --task boiled_egg --seed 0 \
      --model qwen-policy --server http://localhost:8000/v1 \
      --horizon 30 --timeout 600
else
  echo "skipped"
fi

echo
echo "=== [4/5] GRPO online RL (~50h for 200 outer steps; checkpoint every 10) ==="
if [[ "$SKIP_GRPO" != "1" ]]; then
  python src/training/grpo_train.py --config "$GRPO_CFG"
else
  echo "skipped"
fi

echo
echo "=== [5/5] Final evaluation (L1 + L2, 10 episodes per task) ==="
if [[ "$SKIP_EVAL" != "1" ]]; then
  # Point vLLM at the final GRPO adapter (manual restart of vLLM may be needed).
  echo "[!] Restart vLLM with: bash src/training/serve_vllm.sh ckpt/qwen3p5-9b-grpo/final 8000"
  echo "    Press ENTER when ready ..."
  read -r _ignore
  python src/training/eval_trained.py \
      --policy qwen-policy \
      --server http://localhost:8000/v1 \
      --levels 1 2 \
      --episodes 10
  # Aggregate via the project's existing scripts:
  (cd src && python organize_result.py && python convert_result.py) || true
else
  echo "skipped"
fi

echo
echo "=== DONE ==="
echo "SFT adapter:    ckpt/qwen3p5-9b-sft"
echo "GRPO adapter:   ckpt/qwen3p5-9b-grpo/final"
echo "Eval CSVs:      src/eval_result/converted_data.csv  +  src/eval_result/statistics_data.csv"
