"""Wrapper that runs Collab-Overcooked rollouts via src/main.py as a subprocess.

The existing rollout loop in src/collab/collab.py mutates module-level globals
(statistics_dict, turn_statistics_dict), so we use process isolation to avoid
state leakage across rollouts. Each rollout writes its own JSON under
src/data/<run_tag>/<order>/experiment_*.json — we parse that file to recover the
trajectory.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"


@dataclass
class RolloutResult:
    task: str
    seed: int
    run_tag: str
    json_path: str
    success: bool
    score: int
    n_steps: int
    traj: dict = field(default_factory=dict)
    stdout_path: str = ""
    error: str = ""


def _ensure_openai_key_stub():
    """modules.py reads openai_key.txt at import time. Stub it for Qwen runs."""
    keyfile = SRC / "openai_key.txt"
    if not keyfile.exists() or keyfile.stat().st_size == 0:
        keyfile.write_text("sk-stub-not-used-for-open-source\n")


def rollout_once(
    task: str,
    seed: int,
    model_name: str,
    server_api: str,
    horizon: int = 30,
    run_tag: str = "",
    log_dir: Path = REPO / "src" / "training" / "rollout_logs",
    timeout_s: int = 600,
) -> RolloutResult:
    """Execute one episode and return RolloutResult."""
    _ensure_openai_key_stub()
    tag = run_tag or f"grpo-{uuid.uuid4().hex[:8]}"
    save_dir = SRC / "data" / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{tag}_{task}_s{seed}.log"

    cmd = [
        sys.executable,
        "main.py",
        "--order",
        task,
        "--gpt_model",
        model_name,
        "--local_server_api",
        server_api,
        "--horizon",
        str(horizon),
        "--episode",
        "1",
        "--statistics_save_dir",
        f"data/{tag}",
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # main.py runs on CPU; the policy lives on vLLM
    res = RolloutResult(task=task, seed=seed, run_tag=tag, json_path="", success=False, score=0, n_steps=0)
    res.stdout_path = str(stdout_path)
    try:
        with open(stdout_path, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(SRC),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            )
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                res.error = "timeout"
                return res
    except Exception as e:
        res.error = f"spawn_error: {e}"
        return res

    candidates = sorted(glob(str(SRC / "data" / tag / model_name / task / "experiment_*.json")))
    if not candidates:
        res.error = "no_output_json"
        return res
    res.json_path = candidates[-1]
    try:
        with open(res.json_path) as fh:
            traj = json.load(fh)
    except Exception as e:
        res.error = f"json_load_failed: {e}"
        return res
    res.traj = traj
    res.score = int(traj.get("total_score", 0) or 0)
    res.n_steps = len(traj.get("total_timestamp", []))
    res.success = res.score > 0
    return res


def rollout_batch(
    tasks: list[str],
    seeds_per_task: int,
    model_name: str,
    server_api: str,
    horizon: int = 30,
    max_parallel: int = 8,
    run_tag_prefix: str = "grpo",
    timeout_s: int = 600,
) -> list[RolloutResult]:
    """Run tasks × seeds rollouts in parallel and return all results."""
    jobs = []
    for task in tasks:
        for seed in range(seeds_per_task):
            jobs.append((task, seed))

    results: list[RolloutResult] = []
    with cf.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []
        for task, seed in jobs:
            tag = f"{run_tag_prefix}-{task}-s{seed}-{uuid.uuid4().hex[:6]}"
            futures.append(
                pool.submit(
                    rollout_once,
                    task=task,
                    seed=seed,
                    model_name=model_name,
                    server_api=server_api,
                    horizon=horizon,
                    run_tag=tag,
                    timeout_s=timeout_s,
                )
            )
        for fut in cf.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(
                    RolloutResult(task="?", seed=-1, run_tag="", json_path="", success=False, score=0, n_steps=0, error=str(e))
                )
    return results


def extract_step_pairs(traj: dict, agent_idx: int) -> list[dict]:
    """Pull per-step (observation, completion) pairs from a rollout trace for the
    given agent. Mirrors the SFT extractor's logic but keeps invalid/error steps
    too — GRPO learns from failures via the episode advantage.
    """
    from training.extract_sft_data import serialize_completion  # type: ignore

    out: list[dict] = []
    role = "Chef" if agent_idx == 0 else "Assistant"
    for t_idx, step in enumerate(traj.get("content", [])):
        body = step.get("content", {})
        obs = body.get("observation", [[], []])[agent_idx]
        reps = body.get("content", [[], []])[agent_idx]
        if not obs or not reps:
            continue
        parsed = reps[0]
        if not isinstance(parsed, dict):
            continue
        completion = serialize_completion(role, parsed)
        out.append(
            {
                "observation": obs if isinstance(obs, str) else str(obs),
                "completion": completion,
                "step": t_idx,
            }
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="boiled_egg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen-policy")
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    r = rollout_once(
        task=args.task,
        seed=args.seed,
        model_name=args.model,
        server_api=args.server,
        horizon=args.horizon,
        timeout_s=args.timeout,
    )
    print(
        json.dumps(
            {
                "task": r.task,
                "seed": r.seed,
                "success": r.success,
                "score": r.score,
                "n_steps": r.n_steps,
                "json_path": r.json_path,
                "stdout_path": r.stdout_path,
                "error": r.error,
            },
            indent=2,
        )
    )
