"""Reward computation for GRPO over Collab-Overcooked trajectories.

R_ep = w_succ  · 1[total_score > 0]
     − w_time  · clip(T / horizon, 0, 1)
     − w_verr  · clip(n_validator_errors / 10, 0, 1)
     − w_ferr  · clip(n_format_errors / 5, 0, 1)
     − w_red   · redundant_action_rate
     + w_follow · follow_proxy

The terms are all derivable from the per-step `statistical_data` already serialized
into experiment_*.json by src/collab/collab.py — we just read the trace.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardWeights:
    w_succ: float = 1.0
    w_time: float = 0.3
    w_verr: float = 0.1
    w_ferr: float = 0.1
    w_red: float = 0.2
    w_follow: float = 0.3


def _redundancy_rate(traj: dict, agent_idx: int) -> float:
    """Fraction of (agent_idx)'s emitted skills that repeat a previously-completed one.

    Matches the redundancy notion used by src/eval_utils.py — same skill name
    repeated in this agent's action list with no intervening state change.
    """
    actions_lists = traj.get("total_action_list", [[], []])
    if agent_idx >= len(actions_lists):
        return 0.0
    acts = [a.get("action", "") for a in actions_lists[agent_idx] if isinstance(a, dict)]
    if not acts:
        return 0.0
    seen = set()
    redundant = 0
    for a in acts:
        if a in seen:
            redundant += 1
        seen.add(a)
    return redundant / max(1, len(acts))


def _count_errors(traj: dict, agent_idx: int) -> tuple[int, int]:
    """(format_error_count, validator_error_count) across all timesteps for agent_idx."""
    n_format = 0
    n_validator = 0
    for step in traj.get("content", []):
        err = step.get("statistical_data", {}).get("error", [{}, {}])
        if agent_idx >= len(err):
            continue
        n_format += err[agent_idx].get("format_error", {}).get("error_num", 0)
        n_validator += err[agent_idx].get("validator_error", {}).get("error_num", 0)
    return n_format, n_validator


def _follow_proxy(traj: dict, agent_idx: int) -> float:
    """Proxy for TGCO FollowRate: fraction of `say` turns by *partner* that were
    followed by an environment-valid action from this agent within 3 steps.

    Cheap to compute without the full TGCO classifier — we look at whether the
    partner's `say` (non-[NOTHING]) message at step t coincides with this agent
    successfully completing any action within [t, t+3].
    """
    other = 1 - agent_idx
    content = traj.get("content", [])
    if not content:
        return 0.0
    asks = 0
    served = 0
    for t_idx, step in enumerate(content):
        comm = step.get("statistical_data", {}).get("communication", [{}, {}])
        if other >= len(comm):
            continue
        turns = comm[other].get("turn", [])
        had_request = any(isinstance(s, str) and s.strip() and s.strip() != "[NOTHING]" for s in turns)
        if not had_request:
            continue
        asks += 1
        window = content[t_idx : t_idx + 4]
        own_actions = []
        for w_step in window:
            inner = w_step.get("content", {})
            act_list = inner.get("action_list", [[], []])
            if agent_idx < len(act_list):
                own_actions.extend(act_list[agent_idx])
        if any(a and a != "wait(1)" for a in own_actions):
            served += 1
    if asks == 0:
        return 1.0  # no request → no penalty
    return served / asks


def compute_episode_reward(
    traj: dict,
    agent_idx: int = 0,
    weights: Optional[RewardWeights] = None,
    horizon: int = 120,
) -> dict:
    """Return scalar `reward` plus the individual term values for logging."""
    w = weights or RewardWeights()
    score = traj.get("total_score", 0) or 0
    success = 1.0 if score > 0 else 0.0

    n_steps = len(traj.get("total_timestamp", []))
    time_pen = min(max(n_steps / max(1, horizon), 0.0), 1.0)

    n_ferr, n_verr = _count_errors(traj, agent_idx)
    ferr_pen = min(n_ferr / 5.0, 1.0)
    verr_pen = min(n_verr / 10.0, 1.0)

    red_rate = _redundancy_rate(traj, agent_idx)
    follow = _follow_proxy(traj, agent_idx)

    r = (
        w.w_succ * success
        - w.w_time * time_pen
        - w.w_verr * verr_pen
        - w.w_ferr * ferr_pen
        - w.w_red * red_rate
        + w.w_follow * follow
    )

    return {
        "reward": float(r),
        "success": success,
        "time_pen": time_pen,
        "ferr_pen": ferr_pen,
        "verr_pen": verr_pen,
        "redundancy": red_rate,
        "follow": follow,
        "n_steps": n_steps,
        "score": score,
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="experiment_*.json")
    ap.add_argument("--agent", type=int, default=0)
    args = ap.parse_args()
    with open(args.path) as f:
        traj = json.load(f)
    out = compute_episode_reward(traj, agent_idx=args.agent)
    print(json.dumps(out, indent=2))
