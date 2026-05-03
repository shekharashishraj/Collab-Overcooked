# Phase 1 inspection: ProAgent + Adaptive-ToM vs Collab-Overcooked

Local paths (after clone): `third_party/ProAgent/`, `third_party/Adaptive-ToM/overcooked/`.  
Collab references: [`src/collab/agents.py`](../src/collab/agents.py), [`src/prompts/agents/proagent/`](../src/prompts/agents/proagent/), [`src/prompts/agents/a-tom/`](../src/prompts/agents/a-tom/).

---

## 1. ProAgent (upstream)

### Prompting structure

| Piece | Role | Location |
|--------|------|----------|
| Planner system prompt | Rules, skills, layout-specific examples; levels **l1-p** (plan only), **l2-ap** (analysis + plan), **l3-aip** (analysis + **intention for partner** + plan) | `third_party/ProAgent/src/prompts/gpt/planner/{l1-p,l2-ap,l3-aip}/<layout>_<agent_index>.txt` (e.g. [`l2-ap/cramped_room_0.txt`](ProAgent/src/prompts/gpt/planner/l2-ap/cramped_room_0.txt)) |
| Explainer system prompt | Separate GPT module for failure post-mortem | `third_party/ProAgent/src/prompts/gpt/explainer/player0.txt` (and layout-specific variants) |
| User observation | Built in code: layout string + per-step scene (players, pots, counters) | [`proagent.py`](ProAgent/src/proagent/proagent.py) `generate_layout_prompt()`, `generate_state_prompt()` |

Output convention (l2-ap examples): free text with `Analysis:` and `Plan for Player 0: "skill"`. **l3-aip** adds `Intention for Player 1: "skill"` before ego plan ([`l3-aip/cramped_room_0.txt`](ProAgent/src/prompts/gpt/planner/l3-aip/cramped_room_0.txt)).

### Belief / partner tracking

- **`teammate_ml_actions_dict`**: when `state.ml_actions[1-self.agent_index]` is non-null, records partner’s completed ML action at timestep `t-1` ([`proagent.py` ~302–304](ProAgent/src/proagent/proagent.py)).
- **`teammate_intentions_dict`**: for **`prompt_level == "l3-aip"`**, after each planner response, parses partner’s intended ML action from the same response via `parse_ml_action(response, 1-self.agent_index)` (regex `layer 0:` / `layer 1:` then first token) and stores by timestep ([~526–528](ProAgent/src/proagent/proagent.py)).
- **`belief_revision`**: when `l3-aip` **and** `belief_revision=True`, **`generate_belief_prompt()`** prepends to the user message: full dicts of inferred intentions vs real behaviors ([~281–286, 502–506](ProAgent/src/proagent/proagent.py)).

### Feedback to the model

| Trigger | Mechanism | Injected where |
|---------|-----------|----------------|
| ML action invalid / motion fails | `validate_current_ml_action` false → `generate_failure_feedback`: state prompt + “failed at {ml_action}”; **explainer** GPT produces explanation; explanation appended as **user** message to **planner** dialog ([~644–656](ProAgent/src/proagent/proagent.py)) | Planner `dialog_history_list` |
| Retry hint | `Module.query(..., trace=False)` appends to last user content: “Based on the failure explanation…” ([`modules.py` ~75–76](ProAgent/src/proagent/modules.py)) | Same planner query |
| Step success | `generate_success_feedback`: short validation line; may add user line to planner history if not `wait` ([~638–642](ProAgent/src/proagent/proagent.py)) | Planner history |

There is **no** automatic string compare of “predicted partner ML” vs “actual partner ML” in the same way Collab’s `ProAgentLLM` does; partner modeling is **in-prompt** (l3) plus optional **dict replay** (belief_revision), not post-hoc `partner_prediction` regex.

---

## 2. Adaptive-ToM — `overcooked/` (upstream)

### Prompting structure

| Piece | Role | Location |
|--------|------|----------|
| Game rules + tips | System/user bootstrap; response format `<current task>`, `<reasoning>`, `<action>` | [`LLM_agent/base_agent.py`](Adaptive-ToM/overcooked/LLM_agent/base_agent.py) `_generate_message()`, `game_tips`; rules from [`game_prompts/symmetric_room.py`](Adaptive-ToM/overcooked/game_prompts/symmetric_room.py) |
| Per-step user content | `## Game State` + `_add_history()` + partner prediction section + legal actions | `ToM0Agent`, `ToM1Agent._state_to_description`, `AdaptiveAgent._state_to_description` |

### Belief / partner tracking

- **ToM0**: partner state in description depends on `add_partner_state` ([`base_agent.py`](Adaptive-ToM/overcooked/LLM_agent/base_agent.py) ~218–221).
- **ToM1**: inner **`ToM0Agent`** as partner model; histories swapped; one **LLM** call returns predicted partner action; appended as `## Predicted Partner Action` ([`tom_1st_agent.py`](Adaptive-ToM/overcooked/LLM_agent/tom_1st_agent.py)).
- **Adaptive (`AdaptiveAgent`)**: three inner agents (0th/1st/2nd); **`_generate_prediction_candidates`** runs three partner predictions (thread pool); **`tom_loss`** incremented when prediction ≠ true partner action ([`adaptive_tom_agent.py`](Adaptive-ToM/overcooked/LLM_agent/adaptive_tom_agent.py) `update_tom_loss`, called from [`overcooked/main.py`](Adaptive-ToM/overcooked/main.py) ~184). Selection: **FTL** (argmin loss) or **Hedge** (softmax sample over losses).

### Feedback to the model

- **Explicit natural-language “you were wrong”** is **not** the main loop; the model sees **one** fused line: `### Predicted Partner Action … **{selected_prediction}**` where `selected_prediction` comes from FTL/Hedge over the three models ([`adaptive_tom_agent.py` ~169–189](Adaptive-ToM/overcooked/LLM_agent/adaptive_tom_agent.py)).
- **Implicit feedback**: loss accumulates over episodes so later steps shift probability mass (Hedge) or leader (FTL). Loss history in `tom_loss_list`; optional logs to `player*_loss.txt` per upstream README.

---

## 3. Collab-Overcooked (current lightweight)

### Prompting

- Single planner pipeline in **`LLMAgents`** + **text overlays**: ProAgent → [`belief_correction.txt`](../src/prompts/agents/proagent/belief_correction.txt); A-ToM → [`partner_model.txt`](../src/prompts/agents/a-tom/partner_model.txt) + [`partner_output_format.txt`](../src/prompts/agents/a-tom/partner_output_format.txt).
- No separate explainer module; no l1/l2/l3 prompt files per layout.

### Belief / partner tracking

- **`ProAgentLLM`**: parses `{role} partner_prediction:` from planner text; stores in `teammate_intentions_dict`; compares to **`state.ml_actions`** of partner with substring heuristics ([`agents.py`](../src/collab/agents.py) ~52–66).
- **`AToMAgent`**: same `partner_prediction` parse; rolling deque **`tom_accuracy`** (substring match only).

### Feedback

- **`ProAgentLLM`**: `[Belief feedback]` prepended to state prompt when partner completes an ML action (~78–82).
- **`AToMAgent`**: `[ToM feedback]` rolling accuracy line (~135–139).

---

## 4. Gap summary table

| Dimension | ProAgent upstream | Adaptive-ToM overcooked | Collab current |
|-----------|-------------------|-------------------------|----------------|
| **Prompt structure** | Layout-specific `.txt` system prompts; l1/l2/l3; separate **explainer** | Single format (`<current task>` / `<reasoning>` / `<action>`); symmetric-room rules module | Shared chef planner + small **overlay** files |
| **Partner / belief** | l3: partner **intention** parsed from same planner reply; dicts of intention vs behavior; optional **belief_revision** block | **0/1/2**: increasing nested LLM partner simulation; **adaptive**: 3 predictions + **online loss** | Single **regex** `partner_prediction` from ego planner only |
| **Feedback** | **Verifier** + explainer GPT → planner history; optional belief dict in observation | **Hedge/FTL** reweights which prediction text is shown; loss vs ground truth | **Deterministic** string match → one feedback sentence or accuracy line |
| **Ground truth for “prediction”** | Parser maps response to ML string; teammate behavior from env `ml_actions` | Discrete **low-level** `joint_action` vs predicted string (via `update_tom_loss`) | Partner **`ml_actions`** when recorded |

---

## 5. Phase 2 hints (not implemented here)

- To approximate **ProAgent** more faithfully: consider **explainer-on-parse-failure** pattern and/or **l3-aip-style** intention line in prompts, plus optional **belief dict** block (not only last-step feedback).
- To approximate **Adaptive-ToM**: would require **multiple LLM calls** per step (or distilled single prompt) and **loss-based** or fixed-order selection, not only rolling accuracy text.
