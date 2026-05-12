#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server hosting Qwen3.5-9B (base) with a
# loaded LoRA adapter aliased as `qwen-policy`. Used by both rollout collection
# (during GRPO) and final evaluation.
#
# Usage:
#   bash src/training/serve_vllm.sh ckpt/qwen3p5-9b-sft 8000
#
# The server stays in the foreground; run from a separate shell or tmux.

set -euo pipefail

ADAPTER="${1:-ckpt/qwen3p5-9b-sft}"
PORT="${2:-8000}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
ALIAS="${ALIAS:-qwen-policy}"

# Note: the adapter dir saved by sft_train.py contains the ChatML
# tokenizer_config.json. vLLM uses the BASE_MODEL's tokenizer by default —
# point --tokenizer at the adapter dir to inherit the installed chat template.
TOKENIZER_DIR="${TOKENIZER_DIR:-$ADAPTER}"

if [[ ! -d "$ADAPTER" ]]; then
  echo "[serve_vllm] Adapter dir not found: $ADAPTER" >&2
  exit 1
fi

ADAPTER_ABS="$(cd "$ADAPTER" && pwd)"

# H200: leave a couple GB for the training process if running co-located.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

exec vllm serve "$BASE_MODEL" \
  --tokenizer "$TOKENIZER_DIR" \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "${ALIAS}=${ADAPTER_ABS}" \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --dtype bfloat16 \
  --api-key token-abc123 \
  --served-model-name "$BASE_MODEL"
