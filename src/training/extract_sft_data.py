"""Extract (system, user, assistant) SFT pairs from GPT-4o trajectories.

Each trajectory JSON encodes per-timestep state for two agents (Chef=0, Assistant=1).
Within one timestep, only one agent is "active" (non-empty observation + content);
the active agent's `content[ai]` is a list of one or more parsed LLM replies. We
reconstruct the original completion text from the parsed dict, pair it with the
observation as the user message, and build the matching role-specific system
prompt by replaying the same template assembly the env uses at inference time.
"""

import argparse
import json
import os
import random
import sys
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PROMPT_DIR = SRC / "prompts" / "gpt"
RECIPE_DIR = SRC / "prompts" / "recipe"

CHEF_WORKFLOW = """- The usual workflow for the chef is:
  1. Read the cooking process from your recipe. All of your decisions must be strictly guided by the recipe and should not lead to unfounded behavior.
  2. Ask the assistant to pick up ingredients from the ingredient dispenser and use the correct utensil to handle them according to the recipe. Since you do not have access to all the objects, you need to assign some tasks to the assistant while you perform other tasks in parallel.
  3. Work in parallel with the assistant to finish the order in the shortest time possible, unless there is nothing you can do in the current situation. If you have nothing to do, you can wait.
  4. Serve the dish (optional). If the recipe specifies that the dish needs to be served on a plate, you must use `fill_dish_with_food(utensil_name)` to serve the dish from the utensil first; otherwise, just pick up the food from the utensil.
  5. Use `deliver_soup()."""

ASSISTANT_WORKFLOW = """The usual workflow for the Assistant is:
- 1. Ask the Chef for guidance, since you do not have the recipe and need the Chef to help you plan.
- 2. Follow the Chef’s instructions unless they are incorrect. For example, if the Chef requests a utensil that is not available on your side, you should refuse and inform him. """

CHARACTER_PLAYER = (
    "Suppose you are a player who is proficient in the overcooked_ai game. "
    "Your goal is to cooperate with your teammate who is also a LLM agent in "
    "order to get a high score."
)


def _read(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


def _load_recipe(order: str) -> str:
    """Match the recipe loading the env does (case-insensitive substring)."""
    for name in os.listdir(RECIPE_DIR):
        if order in name.lower():
            return _read(RECIPE_DIR / name) + "\n\n\n"
    return "You do not have the recipe\n"


def build_system_prompt(actor: str, order: str) -> str:
    """Replicate src/collab/collab.py::load_prompt_file for the 'origin' mode."""
    role = "Chef" if actor == "chef" else "Assistant"
    teammate = "Assistant" if actor == "chef" else "Chef"

    prompt = _read(PROMPT_DIR / "prompt.txt")
    comm = _read(PROMPT_DIR / "communication_rule.txt")
    env_rule = _read(PROMPT_DIR / "environment_rule.txt").replace("{role}", role)
    skill = _read(PROMPT_DIR / f"{actor}_skill.txt")

    prompt = prompt.replace("{communication_rule}", comm)
    prompt = prompt.replace("{role}", role)
    prompt = prompt.replace("{teammate}", teammate)
    prompt = prompt.replace("{environment_rule}", env_rule)
    prompt = prompt.replace("{character}", CHARACTER_PLAYER)
    prompt = prompt.replace("{skill}", skill)

    if actor == "chef":
        prompt = prompt.replace("{workflow}", CHEF_WORKFLOW)
        prompt = prompt.replace(
            "{has_recipe}",
            "You have recipe, so you need to direct yourself and your teammates to complete the order.",
        )
    else:
        prompt = prompt.replace("{workflow}", ASSISTANT_WORKFLOW)
        prompt = prompt.replace(
            "{job}",
            "You only need to ask and follow Chef's instruction in communication without making plan by yourself. Because you do not have recipe, which means your plan is likely to be wrong.",
        )
        prompt = prompt.replace(
            "{has_recipe}",
            "You do not have recipe, and you should always ask the chef for guidance, rather than making bad decisions on your own.",
        )

    prompt = prompt.replace("{recipe}", _load_recipe(order))
    return prompt


def serialize_completion(role: str, parsed: dict) -> str:
    """Inverse of the parser: dict -> raw LLM text the env expects."""
    analysis = parsed.get("analysis", "")
    plan = parsed.get("plan", "")
    say = parsed.get("say", "[NOTHING]")
    # The env's parser strips a leading "Chef plan: " from `plan`; trajectories sometimes
    # already include or omit that prefix. Normalize so completions match the template.
    plan = plan.strip()
    plan_prefix = f"{role} plan:"
    if plan.lower().startswith(plan_prefix.lower()):
        plan_body = plan[len(plan_prefix):].lstrip()
    else:
        plan_body = plan
    return (
        f"{role} analysis: {analysis}\n"
        f"{role} plan: {plan_body}\n"
        f"{role} say: {say}"
    )


def step_had_validator_error(step_dict: dict, agent_idx: int) -> bool:
    stat = step_dict.get("statistical_data", {})
    err = stat.get("error", [{}, {}])
    if agent_idx >= len(err):
        return False
    n_verr = err[agent_idx].get("validator_error", {}).get("error_num", 0)
    n_ferr = err[agent_idx].get("format_error", {}).get("error_num", 0)
    return (n_verr > 0) or (n_ferr > 0)


def episode_succeeded(traj: dict) -> bool:
    return bool(traj.get("total_score", 0)) and traj.get("total_score", 0) > 0


def extract_pairs_from_traj(traj: dict, order: str) -> list[dict]:
    """Return [{system, user, assistant, role, task, step}] pairs."""
    pairs = []
    for t_idx, step in enumerate(traj.get("content", [])):
        body = step.get("content", {})
        observations = body.get("observation", [[], []])
        replies = body.get("content", [[], []])
        for ai in (0, 1):
            obs = observations[ai] if ai < len(observations) else []
            reps = replies[ai] if ai < len(replies) else []
            if not obs or not reps:
                continue
            if step_had_validator_error(step, ai):
                continue
            actor = "chef" if ai == 0 else "assistant"
            role = "Chef" if ai == 0 else "Assistant"
            system_prompt = build_system_prompt(actor, order)
            user_message = obs if isinstance(obs, str) else str(obs)
            # Only the first reply per timestep — it is the primary answer to `obs`.
            # Multi-turn correction replies have a different (longer) prompt context that
            # is not faithfully recoverable from the trace alone; skipping them keeps
            # the SFT pair clean.
            parsed = reps[0]
            if not isinstance(parsed, dict):
                continue
            if parsed.get("agent") not in (ai, str(ai), None):
                # Guard against off-by-one tagging
                continue
            completion = serialize_completion(role, parsed)
            if not completion.strip() or "analysis:" not in completion.lower():
                continue
            pairs.append(
                {
                    "system": system_prompt,
                    "user": user_message,
                    "assistant": completion,
                    "role": actor,
                    "task": order,
                    "step": t_idx,
                }
            )
    return pairs


def to_chat_record(pair: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": pair["system"]},
            {"role": "user", "content": pair["user"]},
            {"role": "assistant", "content": pair["assistant"]},
        ],
        "meta": {"role": pair["role"], "task": pair["task"], "step": pair["step"]},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default=str(SRC / "data" / "gpt-4o"))
    p.add_argument("--out", dest="out_dir", default=str(SRC / "training" / "data"))
    p.add_argument(
        "--holdout",
        nargs="*",
        default=["boiled_egg", "baked_potato_slices"],
        help="Tasks reserved for eval-only; their trajectories are not used for SFT.",
    )
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    files = sorted(glob(str(in_dir / "*" / "experiment_*.json")))

    n_files = 0
    n_success = 0
    n_pairs = 0
    chef_records: list[dict] = []
    assistant_records: list[dict] = []
    holdout_records: list[dict] = []

    for fpath in files:
        order = os.path.basename(os.path.dirname(fpath))
        try:
            with open(fpath, "r") as fh:
                traj = json.load(fh)
        except Exception as e:
            print(f"[skip] {fpath}: {e}", file=sys.stderr)
            continue
        n_files += 1
        if not episode_succeeded(traj):
            continue
        n_success += 1
        pairs = extract_pairs_from_traj(traj, order)
        for pr in pairs:
            rec = to_chat_record(pr)
            if order in args.holdout:
                holdout_records.append(rec)
            else:
                (chef_records if pr["role"] == "chef" else assistant_records).append(rec)
            n_pairs += 1

    rng.shuffle(chef_records)
    rng.shuffle(assistant_records)
    rng.shuffle(holdout_records)

    chef_path = out_dir / "sft_qwen_chef.jsonl"
    asst_path = out_dir / "sft_qwen_assistant.jsonl"
    all_path = out_dir / "sft_qwen_all.jsonl"
    eval_path = out_dir / "sft_qwen_heldout.jsonl"

    def _dump(records, path):
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    _dump(chef_records, chef_path)
    _dump(assistant_records, asst_path)
    _dump(chef_records + assistant_records, all_path)
    _dump(holdout_records, eval_path)

    print(f"files seen      : {n_files}")
    print(f"successful eps  : {n_success}")
    print(f"total pairs     : {n_pairs}")
    print(f"chef pairs      : {len(chef_records)} -> {chef_path}")
    print(f"assistant pairs : {len(assistant_records)} -> {asst_path}")
    print(f"combined        : {len(chef_records) + len(assistant_records)} -> {all_path}")
    print(f"holdout pairs   : {len(holdout_records)} -> {eval_path}")


if __name__ == "__main__":
    main()
