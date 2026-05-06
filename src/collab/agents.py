"""Agent architecture subclasses (ProAgent, A-ToM, Reflexion) built on LLMAgents."""
from __future__ import annotations

import copy
import os
import re
from collections import deque

from collab.collab import LLMAgents, PROMPT_DIR


def _read_overlay(*parts: str) -> str:
    path = os.path.join(PROMPT_DIR, "agents", *parts)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class ProAgentLLM(LLMAgents):
    """Belief correction via partner_prediction + mismatch feedback in observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = "proagent"
        self.belief_corrections = []
        self._pending_partner_prediction = ""
        self._partner_belief_feedback = ""

    def load_prompt_file(self, mode="origin"):
        super().load_prompt_file(mode)
        if mode != "origin":
            return self.planner.instruction_head_list[0]["content"]
        overlay = _read_overlay("proagent", "belief_correction.txt")
        overlay = overlay.replace("{role}", self.name)
        base = self.planner.instruction_head_list[0]["content"]
        merged = base + "\n\n" + overlay
        self.planner.instruction_head_list[0]["content"] = merged
        return merged

    def on_planner_response(self, response, state):
        role = "Chef" if self.agent_index == 0 else "Assistant"
        m = re.search(
            rf"(?:{role})\s+partner_prediction\s*:(.*?)(?=(?:{role})\s+analysis\s*:)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            self._pending_partner_prediction = m.group(1).strip()
            self.teammate_intentions_dict[state.timestep] = self._pending_partner_prediction

    def _hook_after_teammate_ml_record(self, state):
        actual = state.ml_actions[1 - self.agent_index]
        if actual is None:
            return
        pred = getattr(self, "_pending_partner_prediction", "") or ""
        actual_str = str(actual)
        pred_l, act_l = pred.lower(), actual_str.lower()
        ok = bool(pred) and (act_l in pred_l or pred_l in act_l or any(tok in pred_l for tok in act_l.split() if len(tok) > 4))
        self.belief_corrections.append(
            {
                "timestep": state.timestep,
                "prediction": pred,
                "actual": actual_str,
                "match": ok,
            }
        )
        if ok:
            self._partner_belief_feedback = (
                f"Partner finished `{actual_str}`. Your last partner_prediction appears consistent."
            )
        else:
            self._partner_belief_feedback = (
                f"Belief correction: you predicted `{pred or '[empty]'}` but partner finished `{actual_str}`. "
                f"Update your model of your partner."
            )

    def generate_state_prompt(self, state):
        base = super().generate_state_prompt(state)
        fb = getattr(self, "_partner_belief_feedback", "")
        if fb:
            return f"[Belief feedback]\n{fb}\n\n" + base
        return base


class AToMAgent(LLMAgents):
    """Theory-of-mind style partner modeling with rolling prediction accuracy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = "a-tom"
        self._tom_window = 12
        self.tom_accuracy = deque(maxlen=self._tom_window)
        self.tom_event_log: list[dict] = []  # full-episode log (export / analysis)
        self._pending_partner_prediction = ""
        self._tom_feedback_line = ""

    def load_prompt_file(self, mode="origin"):
        super().load_prompt_file(mode)
        if mode != "origin":
            return self.planner.instruction_head_list[0]["content"]
        m1 = _read_overlay("a-tom", "partner_model.txt")
        m2 = _read_overlay("a-tom", "partner_output_format.txt")
        overlay = (m1 + "\n\n" + m2).replace("{role}", self.name)
        base = self.planner.instruction_head_list[0]["content"]
        merged = base + "\n\n" + overlay
        self.planner.instruction_head_list[0]["content"] = merged
        return merged

    def on_planner_response(self, response, state):
        role = "Chef" if self.agent_index == 0 else "Assistant"
        m = re.search(
            rf"(?:{role})\s+partner_prediction\s*:(.*?)(?=(?:{role})\s+analysis\s*:)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            self._pending_partner_prediction = m.group(1).strip()

    def _hook_after_teammate_ml_record(self, state):
        actual = state.ml_actions[1 - self.agent_index]
        if actual is None:
            return
        pred = getattr(self, "_pending_partner_prediction", "") or ""
        actual_str = str(actual)
        pred_l, act_l = pred.lower(), actual_str.lower()
        ok = bool(pred) and (act_l in pred_l or pred_l in act_l)
        self.tom_accuracy.append(ok)
        self.tom_event_log.append(
            {
                "timestep": state.timestep,
                "prediction": pred,
                "actual": actual_str,
                "match": ok,
            }
        )
        correct = sum(1 for x in self.tom_accuracy if x)
        n = len(self.tom_accuracy)
        self._tom_feedback_line = (
            f"Your partner-prediction accuracy over the last {n} partner completions: "
            f"{correct}/{n} correct."
        )

    def generate_state_prompt(self, state):
        base = super().generate_state_prompt(state)
        fb = getattr(self, "_tom_feedback_line", "")
        if fb and len(self.tom_accuracy) > 0:
            return f"[ToM feedback]\n{fb}\n\n" + base
        return base


class ReflexionAgent(LLMAgents):
    """Cross-episode reflections injected into the system prompt."""

    def __init__(self, *args, reflexion_memory_buffer=None, **kwargs):
        self.episode_memory = reflexion_memory_buffer if reflexion_memory_buffer is not None else []
        super().__init__(*args, **kwargs)
        self.agent_type = "reflexion"

    def load_prompt_file(self, mode="origin"):
        super().load_prompt_file(mode)
        if mode != "origin":
            return self.planner.instruction_head_list[0]["content"]
        tmpl = _read_overlay("reflexion", "episode_memory.txt")
        if self.episode_memory:
            block = "\n".join(f"- {m}" for m in self.episode_memory[-6:])
            mem = tmpl.replace("{memory_block}", block).replace("{role}", self.name)
            base = self.planner.instruction_head_list[0]["content"]
            merged = base + "\n\n" + mem
            self.planner.instruction_head_list[0]["content"] = merged
            return merged
        return self.planner.instruction_head_list[0]["content"]

    def generate_episode_reflection(self, statistics_snapshot: dict):
        """One post-episode LLM call; appends a short memory string."""
        score = statistics_snapshot.get("total_score", 0)
        finished = statistics_snapshot.get("total_order_finished") or []
        n_steps = len(statistics_snapshot.get("total_timestamp") or [])
        summary = (
            f"Episode summary: score={score}, orders_finished={finished}, "
            f"timesteps_logged={n_steps}. "
            f"Write 2-3 sentences: what worked, what failed, and one concrete tactic for the next episode. "
            f"Plain text only, no role prefixes."
        )
        ih_backup = copy.deepcopy(self.planner.instruction_head_list)
        dh_backup = copy.deepcopy(self.planner.dialog_history_list)
        try:
            self.planner.instruction_head_list = [
                {
                    "role": "system",
                    "content": (
                        "You compress cooperative Overcooked-style game episodes into "
                        "short, actionable reflections for the same agent in future episodes."
                    ),
                }
            ]
            self.planner.dialog_history_list = []
            self.planner.current_user_message = {"role": "user", "content": summary}
            resp, _ = self.planner.query(
                key=self.openai_api_key(),
                proxy=self.proxy,
                stop=None,
                trace=True,
                map="",
            )
            text = (resp or "").strip()
            if text:
                self.episode_memory.append(text[:1200])
        finally:
            self.planner.instruction_head_list = ih_backup
            self.planner.dialog_history_list = dh_backup


def belief_metrics_for_agent(agent: LLMAgents, slot: int) -> dict:
    """
    Episode summary for partner-action prediction (belief / ToM scaffolding).
    Written into experiment JSON under belief_metrics.
    """
    role = "chef" if slot == 0 else "assistant"
    base = {
        "player": f"P{slot}",
        "role": role,
        "agent_type": getattr(agent, "agent_type", "baseline"),
        "model": getattr(agent, "model", None),
    }

    if isinstance(agent, ProAgentLLM):
        bc = agent.belief_corrections
        n = len(bc)
        matches = sum(1 for x in bc if x.get("match"))
        intents = getattr(agent, "teammate_intentions_dict", {}) or {}
        base.update(
            {
                "framework": "proagent",
                "instrumentation": "partner_prediction vs next partner ML completion",
                "n_predictions_scored": n,
                "n_correct": matches,
                "accuracy": round(matches / n, 4) if n else None,
                "n_timesteps_with_intention_logged": len(intents),
                "events": bc,
            }
        )
        return base

    if isinstance(agent, AToMAgent):
        events = getattr(agent, "tom_event_log", []) or []
        n = len(events)
        matches = sum(1 for x in events if x.get("match"))
        window = list(agent.tom_accuracy)
        wn = len(window)
        wm = sum(1 for x in window if x)
        base.update(
            {
                "framework": "a-tom",
                "instrumentation": "partner_prediction vs next partner ML completion (stricter match than ProAgent)",
                "rolling_window_maxlen": agent._tom_window,
                "n_predictions_scored_episode": n,
                "n_correct_episode": matches,
                "accuracy_episode": round(matches / n, 4) if n else None,
                "rolling_window_correct": wm,
                "rolling_window_n": wn,
                "rolling_accuracy": round(wm / wn, 4) if wn else None,
                "events": events,
            }
        )
        return base

    base.update(
        {
            "framework": getattr(agent, "agent_type", "baseline"),
            "instrumentation": None,
            "note": "No partner_prediction belief hooks for baseline/reflexion in this codebase.",
        }
    )
    return base


def team_belief_summary(agent0: LLMAgents, agent1: LLMAgents) -> dict:
    """Aggregate episode belief-tracking stats across both players."""
    m0 = belief_metrics_for_agent(agent0, 0)
    m1 = belief_metrics_for_agent(agent1, 1)

    def acc(m: dict):
        a = m.get("accuracy")
        if a is None:
            a = m.get("accuracy_episode")
        return a

    scores = [x for x in (acc(m0), acc(m1)) if x is not None]
    out = {
        "mean_accuracy_two_sided": round(sum(scores) / len(scores), 4)
        if scores
        else None,
        "players_with_belief_metrics": sum(
            1
            for m in (m0, m1)
            if m.get("n_predictions_scored") or m.get("n_predictions_scored_episode")
        ),
    }
    return {"team": out, "per_player": [m0, m1]}
