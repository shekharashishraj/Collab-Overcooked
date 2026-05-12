"""End-to-end evaluation of a trained Qwen policy on Collab-Overcooked.

Pipeline (matches the project's existing scripts):
    1. For each (task, partner, episode), run src/main.py via rollout_env.
    2. Aggregate rollouts under data/<run_tag>/<policy_name>/<task>/*.json.
    3. Invoke src/evaluation.py to compute per-task metrics.
    4. Invoke src/organize_result.py + src/convert_result.py to produce CSVs.
    5. Add an extra column for our `reward.compute_episode_reward` totals so we
       can see the TGCO-proxy terms side-by-side with the official metrics.

This wrapper does NOT replace evaluation.py; it just orchestrates rollouts and
delegates metric computation to the existing code.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from training.reward import RewardWeights, compute_episode_reward  # noqa: E402
from training.rollout_env import rollout_batch  # noqa: E402


LEVEL_1 = ["baked_bell_pepper", "baked_sweet_potato", "boiled_egg", "boiled_mushroom", "boiled_sweet_potato"]
LEVEL_2 = ["baked_potato_slices", "baked_pumpkin_slices", "boiled_corn_slices", "boiled_green_bean_slices", "boiled_potato_slices"]


def run_official_eval(model_name: str, log_dir: Path, save_dir: Path):
    """Call src/evaluation.py in fix_task / AUTO mode against the run's JSONs."""
    cmd = [
        sys.executable,
        "evaluation.py",
        "--test_mode",
        "fix_task",
        "--model",
        model_name,
        "--order",
        "AUTO",
        "--log_dir",
        str(log_dir),
        "--save_dir",
        str(save_dir),
    ]
    print("[eval_trained] $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(SRC), check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="vLLM model alias for the trained policy.")
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--levels", nargs="*", default=["1", "2"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--max_parallel", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--run_tag", default=None)
    args = ap.parse_args()

    tasks: list[str] = []
    if "1" in args.levels:
        tasks += LEVEL_1
    if "2" in args.levels:
        tasks += LEVEL_2

    tag = args.run_tag or f"eval-{args.policy}"
    log_dir_rel = f"data/{tag}"
    log_dir_abs = SRC / log_dir_rel
    save_dir_rel = f"eval_result/{tag}"

    print(f"[eval_trained] running {len(tasks)} tasks × {args.episodes} eps -> {log_dir_abs}")
    results = rollout_batch(
        tasks=tasks,
        seeds_per_task=args.episodes,
        model_name=args.policy,
        server_api=args.server,
        horizon=args.horizon,
        max_parallel=args.max_parallel,
        run_tag_prefix=tag,
        timeout_s=args.timeout,
    )

    # Per-rollout reward report
    weights = RewardWeights()
    rows = []
    for r in results:
        if not r.traj:
            rows.append({"task": r.task, "seed": r.seed, "success": 0, "score": 0, "n_steps": 0, "reward": None, "error": r.error})
            continue
        info_chef = compute_episode_reward(r.traj, agent_idx=0, weights=weights, horizon=args.horizon)
        info_asst = compute_episode_reward(r.traj, agent_idx=1, weights=weights, horizon=args.horizon)
        rows.append({
            "task": r.task,
            "seed": r.seed,
            "success": int(r.success),
            "score": r.score,
            "n_steps": r.n_steps,
            "reward_chef": info_chef["reward"],
            "reward_assistant": info_asst["reward"],
            "redundancy_chef": info_chef["redundancy"],
            "redundancy_assistant": info_asst["redundancy"],
            "follow_chef": info_chef["follow"],
            "follow_assistant": info_asst["follow"],
            "verr_chef": info_chef["verr_pen"],
            "verr_assistant": info_asst["verr_pen"],
        })

    out_csv = REPO / f"src/eval_result/{tag}_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"[eval_trained] wrote per-rollout summary -> {out_csv}")

    # Now delegate to existing evaluation.py for the official F1/similarity/redundancy
    # metrics, then to organize_result.py / convert_result.py for CSV aggregation.
    run_official_eval(args.policy, Path(log_dir_rel), Path(save_dir_rel))
    print("[eval_trained] official evaluation done.")


if __name__ == "__main__":
    main()
