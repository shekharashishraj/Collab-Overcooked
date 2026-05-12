"""Stage 2: custom GRPO loop over Collab-Overcooked rollouts.

Outer loop per step:
    1. (Re)load current LoRA adapter onto the vLLM server (separate process).
    2. Sample G rollouts per (task, role) using rollout_env.rollout_batch.
    3. Compute episode reward per rollout via reward.compute_episode_reward.
    4. Compute group-relative advantage = (R - mean_group) / (std_group + eps).
    5. Build per-step (prompt, response) batches; advantage broadcast within rollout.
    6. Apply REINFORCE-with-advantage loss + KL-to-reference penalty (β·KL).
    7. Save LoRA adapter; next outer step reloads it onto vLLM.

Inner_epochs=1 keeps the update on-policy (ratio≈1) so we can skip the clipped
surrogate in v1. Bump it to 2-4 and turn on `clip_ratio` once the loop is stable.

Reference model is the frozen SFT adapter — kept as a separate PEFT inference
adapter on the same base model.
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.reward import RewardWeights, compute_episode_reward  # noqa: E402
from training.rollout_env import (  # noqa: E402
    RolloutResult,
    extract_step_pairs,
    rollout_batch,
)


def load_yaml(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def build_chat_text(tokenizer, system: str, user: str, completion: Optional[str] = None) -> str:
    """Render messages using the model's chat template.
    If completion is None we add the generation prompt; otherwise we include the
    assistant turn and return the full sequence."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if completion is not None:
        messages.append({"role": "assistant", "content": completion})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(completion is None),
    )


def build_system_for(role_idx: int, task: str) -> str:
    """Defer to the SFT extractor's prompt assembly so train/test prompts match."""
    from training.extract_sft_data import build_system_prompt  # local import to avoid heavy init
    actor = "chef" if role_idx == 0 else "assistant"
    return build_system_prompt(actor, task)


def reload_vllm_adapter(server_url: str, adapter_dir: str, model_alias: str):
    """Tell the running vLLM server to swap LoRA. Uses the OpenAI-compatible /v1
    LoRA management endpoint (vllm>=0.6) if available; otherwise the user must
    restart vLLM externally between outer steps and we just no-op.
    """
    import urllib.error
    import urllib.request

    payload = {"lora_name": model_alias, "lora_path": str(Path(adapter_dir).resolve())}
    data = json.dumps(payload).encode("utf-8")
    for endpoint in ("/v1/load_lora_adapter", "/load_lora_adapter"):
        url = server_url.rstrip("/") + endpoint
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 200 <= resp.status < 300:
                    print(f"[grpo] vLLM reloaded adapter via {endpoint}")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
            continue
    print("[grpo] WARN: could not hot-reload LoRA on vLLM; ensure server is configured.")
    return False


def collect_rollouts(
    cfg: dict, current_adapter: str
) -> list[RolloutResult]:
    """One rollout sweep over all configured tasks × G seeds."""
    reload_vllm_adapter(cfg["vllm_server"], current_adapter, cfg["vllm_model_alias"])
    return rollout_batch(
        tasks=cfg["tasks"],
        seeds_per_task=cfg["group_size"],
        model_name=cfg["vllm_model_alias"],
        server_api=cfg["vllm_server"],
        horizon=cfg["horizon"],
        max_parallel=cfg["rollout_max_parallel"],
        timeout_s=cfg["rollout_timeout_s"],
    )


def compute_advantages(rollouts: list[RolloutResult], role_idx: int, weights: RewardWeights, horizon: int):
    """Group by task. Within each group of G rollouts, compute normalized advantage."""
    reward_info: dict[str, list[dict]] = defaultdict(list)
    info_by_rollout: list[dict] = []
    for r in rollouts:
        info = compute_episode_reward(r.traj, agent_idx=role_idx, weights=weights, horizon=horizon)
        info["task"] = r.task
        info["seed"] = r.seed
        info_by_rollout.append(info)
        reward_info[r.task].append(info)

    advantages: list[float] = []
    for r, info in zip(rollouts, info_by_rollout):
        group = reward_info[r.task]
        rs = [g["reward"] for g in group]
        mu = sum(rs) / len(rs)
        var = sum((x - mu) ** 2 for x in rs) / max(1, len(rs))
        std = math.sqrt(var) + 1e-6
        adv = (info["reward"] - mu) / std
        advantages.append(adv)
    return advantages, info_by_rollout


def tokenize_pair(tokenizer, system: str, user: str, completion: str, max_len: int, device):
    """Return prompt_ids, response_ids (1D tensors on `device`)."""
    full = build_chat_text(tokenizer, system, user, completion)
    prompt = build_chat_text(tokenizer, system, user, completion=None)
    full_ids = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]
    if full_ids.shape[0] <= prompt_ids.shape[0]:
        return None  # response was truncated to nothing
    response_ids = full_ids[prompt_ids.shape[0]:]
    return full_ids.to(device), prompt_ids.to(device), response_ids.to(device)


def response_logprobs(model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Per-token log-probs of the response under `model`, with grad to its params."""
    out = model(full_ids.unsqueeze(0))
    logits = out.logits[0]  # [T, V]
    # Predict token t from logits at t-1
    shift_logits = logits[:-1, :]
    shift_labels = full_ids[1:]
    logp = F.log_softmax(shift_logits, dim=-1)
    tok_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [T-1]
    # Response tokens are positions [prompt_len-1 : T-1] in shift_labels
    return tok_logp[prompt_len - 1 :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_outer_steps", type=int, default=None, help="Override for smoke tests.")
    ap.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks for a smoke test.")
    ap.add_argument("--group_size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.max_outer_steps is not None:
        cfg["outer_steps"] = args.max_outer_steps
    if args.tasks:
        cfg["tasks"] = args.tasks
    if args.group_size is not None:
        cfg["group_size"] = args.group_size

    random.seed(cfg.get("seed", 17))
    torch.manual_seed(cfg.get("seed", 17))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[grpo] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[grpo] loading base model {cfg['base_model']} ...")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    base.gradient_checkpointing_enable()
    base.to(device)

    # Trainable policy: load SFT adapter as trainable LoRA on top of base
    print(f"[grpo] loading trainable adapter from {cfg['sft_adapter']} ...")
    policy = PeftModel.from_pretrained(base, cfg["sft_adapter"], is_trainable=True)
    policy.print_trainable_parameters()

    # Reference: separate frozen PEFT model on a separate base copy. To save memory we
    # share weights via a second adapter name pointing to the same path.
    print(f"[grpo] loading reference adapter from {cfg['ref_adapter']} ...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    ref_base.to(device)
    ref = PeftModel.from_pretrained(ref_base, cfg["ref_adapter"])
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg["learning_rate"]
    )

    reward_weights = RewardWeights(**cfg["reward"])
    current_adapter_dir = str(Path(cfg["sft_adapter"]).resolve())

    for outer in range(cfg["outer_steps"]):
        t_outer = time.time()
        print(f"\n========== outer step {outer} ==========")

        rollouts = collect_rollouts(cfg, current_adapter_dir)
        rollouts = [r for r in rollouts if r.traj]
        if not rollouts:
            print("[grpo] WARN: zero usable rollouts; skipping outer step")
            continue

        # Train both roles per outer step
        outer_metrics = {"outer": outer, "n_rollouts": len(rollouts)}
        for role_idx in cfg["agent_roles"]:
            advantages, infos = compute_advantages(rollouts, role_idx, reward_weights, cfg["horizon"])
            mean_r = sum(i["reward"] for i in infos) / len(infos)
            mean_succ = sum(i["success"] for i in infos) / len(infos)
            print(
                f"[grpo] role={role_idx}  reward_mean={mean_r:.3f}  succ={mean_succ:.2%}"
            )

            # Build per-step training items
            policy.train()
            total_loss = 0.0
            total_kl = 0.0
            n_items = 0
            optim.zero_grad(set_to_none=True)

            for r, adv in zip(rollouts, advantages):
                steps = extract_step_pairs(r.traj, role_idx)
                if not steps:
                    continue
                system_prompt = build_system_for(role_idx, r.task)
                for step in steps:
                    pack = tokenize_pair(
                        tokenizer,
                        system_prompt,
                        step["observation"],
                        step["completion"],
                        cfg["max_seq_length"],
                        device,
                    )
                    if pack is None:
                        continue
                    full_ids, prompt_ids, response_ids = pack
                    prompt_len = prompt_ids.shape[0]

                    logp_pi = response_logprobs(policy, full_ids, prompt_len)
                    with torch.no_grad():
                        logp_ref = response_logprobs(ref, full_ids, prompt_len)

                    # k1 KL estimator: KL ≈ mean(logp_pi - logp_ref)
                    kl_per_tok = (logp_pi - logp_ref).detach()
                    kl_term = (logp_pi - logp_ref).mean()

                    # REINFORCE-with-advantage (on-policy, K=1)
                    pg = -adv * logp_pi.mean()
                    loss = pg + cfg["kl_beta"] * kl_term

                    (loss / max(1, cfg["grad_accumulation"])).backward()
                    total_loss += float(loss.detach())
                    total_kl += float(kl_per_tok.mean())
                    n_items += 1

                    if n_items % cfg["grad_accumulation"] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in policy.parameters() if p.requires_grad],
                            cfg["max_grad_norm"],
                        )
                        optim.step()
                        optim.zero_grad(set_to_none=True)

            # Flush any remaining grad
            if n_items % cfg["grad_accumulation"] != 0 and n_items > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    cfg["max_grad_norm"],
                )
                optim.step()
                optim.zero_grad(set_to_none=True)

            outer_metrics[f"role{role_idx}_mean_reward"] = mean_r
            outer_metrics[f"role{role_idx}_mean_success"] = mean_succ
            outer_metrics[f"role{role_idx}_loss"] = total_loss / max(1, n_items)
            outer_metrics[f"role{role_idx}_kl"] = total_kl / max(1, n_items)
            outer_metrics[f"role{role_idx}_n_items"] = n_items

        outer_metrics["wall_s"] = time.time() - t_outer
        with open(log_path, "a") as fh:
            fh.write(json.dumps(outer_metrics) + "\n")
        print(f"[grpo] outer {outer} done in {outer_metrics['wall_s']:.1f}s")

        if (outer + 1) % cfg["save_every"] == 0 or (outer + 1) == cfg["outer_steps"]:
            save_dir = output_dir / f"step-{outer + 1}"
            policy.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            current_adapter_dir = str(save_dir.resolve())
            print(f"[grpo] saved checkpoint -> {save_dir}")

    # Final save
    final_dir = output_dir / "final"
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[grpo] final adapter -> {final_dir}")


if __name__ == "__main__":
    main()
