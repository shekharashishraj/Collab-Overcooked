"""Chat-template installation for base Qwen models.

Qwen3.5-9B (base) ships without a `chat_template`. Both the SFT and GRPO
trainers need one so `tokenizer.apply_chat_template(messages, ...)` works and
so the saved tokenizer can be reused by vLLM at rollout time.

We install the standard Qwen ChatML template (the same one Qwen2.5-Instruct
and Qwen3-Instruct use), without thinking-mode tags — Qwen3's `<think>...
</think>` is intentionally disabled here because the GPT-4o supervision has
no thinking traces, so teaching the model to emit them would be confusing.

If the loaded tokenizer already has a chat_template (e.g. Instruct variant),
we leave it untouched unless `force=True`.
"""

# Standard Qwen ChatML template, thinking-mode disabled. Keeps `{% generation %}`
# markers so TRL's response-token masking still works.
QWEN_CHATML_NO_THINK = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{%- elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{%- endif %}"
)

# Qwen3 ChatML tokens — required for ChatML to parse correctly on a base
# tokenizer that doesn't already know them.
CHATML_SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]


def ensure_chat_template(tokenizer, force: bool = False) -> bool:
    """Install the Qwen ChatML template on `tokenizer` if missing.

    Returns True if we modified the tokenizer, False if it was already set up.
    Side effects: may add special tokens; caller should resize embeddings on
    the model if `len(tokenizer)` changed (sft_train.py / grpo_train.py do this).
    """
    changed = False
    if tokenizer.chat_template is None or force:
        tokenizer.chat_template = QWEN_CHATML_NO_THINK
        changed = True

    existing = set(tokenizer.get_vocab().keys())
    missing = [t for t in CHATML_SPECIAL_TOKENS if t not in existing]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        changed = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        changed = True

    return changed
