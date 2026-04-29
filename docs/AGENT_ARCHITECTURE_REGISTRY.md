# Agent Architecture Registry — Implementation Reference

This document describes the **Agent Architecture Registry** feature: decoupling **agent architecture** (prompting / bookkeeping behavior) from **model selection** (which LLM backs each player). It was added to support multiple agent “modes” (`baseline`, `proagent`, `a-tom`, `reflexion`) while reusing the same environment loop, validator, and `Module.query` LLM stack.

---

## 1. Goals

| Concern | Mechanism |
|--------|-----------|
| **Transport** | Existing `--p0` / `--p1` (`LLMPair`, `Human`) unchanged |
| **Model IDs** | Existing `--model_p0` / `--model_p1` (and `--gpt_model` fallback) unchanged |
| **Architecture** | New `--agent_p0` / `--agent_p1` → subclass of `LLMAgents` via registry |
| **Logging** | `agent_type` in JSON metadata, save paths, and compiler outputs |
| **Reflexion persistence** | Shared per-slot lists across episodes within one process |

---

## 2. New and Modified Files

```
src/
  main.py                          # CLI, metadata, save path, Reflexion episode hook
  utils.py                         # AGENT_REGISTRY, get_agent_class, make_agent routing
  compile_experiment_run.py        # p0_agent_type / p1_agent_type in CSV + JSON summary
  collab/
    collab.py                      # baseline agent_type, hooks, parse_response
    agents.py                      # ProAgentLLM, AToMAgent, ReflexionAgent
  prompts/
    agents/
      proagent/belief_correction.txt
      a-tom/partner_model.txt
      a-tom/partner_output_format.txt
      reflexion/episode_memory.txt
docs/
  AGENT_ARCHITECTURE_REGISTRY.md   # this file
```

---

## 3. Command-Line Interface (`src/main.py`)

### New arguments

| Flag | Choices | Default | Purpose |
|------|---------|---------|---------|
| `--agent_p0` | `baseline`, `proagent`, `a-tom`, `reflexion` | `baseline` | Architecture for P0 when `--p0 LLMPair` |
| `--agent_p1` | same | `baseline` | Architecture for P1 when `--p1 LLMPair` |

**Note:** `a-tom` contains a hyphen; it must be passed exactly as registered in argparse.

### Unchanged arguments (still authoritative for models)

- `--model_p0`, `--model_p1`, `--gpt_model`
- `--p0`, `--p1` (algorithm / transport)

### Example runs

From repository **`src/`** (so `PROMPT_DIR` resolves to `src/prompts/`):

```bash
cd src
python3 main.py --order boiled_egg \
  --model_p0 gpt-4o --model_p1 claude-sonnet-4-20250514 \
  --agent_p0 proagent --agent_p1 baseline
```

---

## 4. Agent Registry and Factory (`src/utils.py`)

### `AGENT_REGISTRY`

Maps CLI string → Python class:

| Key | Class |
|-----|--------|
| `baseline` | `LLMAgents` (`collab.collab`) |
| `proagent` | `ProAgentLLM` |
| `a-tom` | `AToMAgent` |
| `reflexion` | `ReflexionAgent` |

### `get_agent_class(agent_type: str)`

- Normalizes with `(agent_type or "baseline").lower()`.
- Unknown values fall back to **`LLMAgents`**.

### `make_agent(alg, mdp, layout, **gptargs)`

**Popped kwargs** (so they are not passed to `LLMAgents.__init__` unless applicable):

1. **`agent_type`** — default `"baseline"`.
2. **`reflexion_memory_buffer`** — optional `list` shared across episodes for `ReflexionAgent` only.

For `alg == "LLMPair"`:

- `cls = get_agent_class(agent_type)`
- If `cls is ReflexionAgent`: `ReflexionAgent(mlam, layout, reflexion_memory_buffer=..., **gptargs)`
- Else: `cls(mlam, layout, **gptargs)`

`Stay`, `Random`, `Greedy` branches are unchanged.

---

## 5. Base Class Extensions (`src/collab/collab.py`)

### `LLMAgents.__init__`

- Sets **`self.agent_type = "baseline"`** so every LLM agent has an attribute for logging. Subclasses overwrite this after `super().__init__`.

### Hooks (intended for subclasses)

1. **`on_planner_response(self, response, state)`**  
   - Default: no-op.  
   - Called in **`generate_ml_action`** immediately after `self.planner.query(...)` returns and the raw `response` is printed — **before** `parse_response` for talk/analysis/plan.  
   - Used to scrape **`partner_prediction`** from the model output (ProAgent, A-ToM).

2. **`_hook_after_teammate_ml_record(self, state)`**  
   - Default: no-op.  
   - Called in **`action`** right after the block that appends to **`teammate_ml_actions`** when `state.ml_actions[1 - self.agent_index]` is non-`None`.  
   - Used to compare predicted vs actual partner ML actions (ProAgent, A-ToM).

### `parse_response(..., mode="analysis")`

When the model includes an optional block:

`Chef partner_prediction: ...` **before** `Chef analysis:` (and similarly for `Assistant`),

the parser uses a **wider regex** so analysis extraction still works. If there is no `partner_prediction` line, the **original** analysis pattern is used. **Plan** and **talk** modes were not structurally changed for this feature; models are instructed to keep the usual ordering: `partner_prediction` → `analysis` → `plan` → `say`.

---

## 6. Architecture Implementations (`src/collab/agents.py`)

All subclasses inherit the full `LLMAgents` action loop, validation, and communication. They specialize **`load_prompt_file`**, optional **`on_planner_response`** / **`_hook_after_teammate_ml_record`**, and sometimes **`generate_state_prompt`**.

### Overlay loading

**`_read_overlay(*parts)`** reads UTF-8 text from:

`os.path.join(PROMPT_DIR, "agents", *parts)`

where **`PROMPT_DIR`** is `os.path.join(os.getcwd(), "prompts")` (see `collab.py`). Running **`main.py` from `src/`** is required for default paths.

---

### 6.1 `ProAgentLLM`

| Aspect | Behavior |
|--------|----------|
| **`agent_type`** | `"proagent"` |
| **`load_prompt_file("origin")`** | After `super()`, appends `prompts/agents/proagent/belief_correction.txt` (with `{role}` → `Chef` / `Assistant`). Other modes (`correct`, `reflection`) only get the base prompt. |
| **`on_planner_response`** | Regex-extracts `{Role} partner_prediction:` before `{Role} analysis:`; stores text in **`_pending_partner_prediction`** and **`teammate_intentions_dict[timestep]`**. |
| **`_hook_after_teammate_ml_record`** | Compares prediction string to actual `state.ml_actions[other]` with simple substring / token heuristics; appends dicts to **`belief_corrections`**; sets **`_partner_belief_feedback`**. |
| **`generate_state_prompt`** | Prepends **`[Belief feedback]\n...`** when feedback is set. |

---

### 6.2 `AToMAgent`

| Aspect | Behavior |
|--------|----------|
| **`agent_type`** | `"a-tom"` |
| **`load_prompt_file("origin")`** | Appends `a-tom/partner_model.txt` + `a-tom/partner_output_format.txt`. |
| **`on_planner_response`** | Same partner_prediction extraction as ProAgent (no `teammate_intentions_dict` write in this class). |
| **`_hook_after_teammate_ml_record`** | Stricter match than ProAgent for “correct”; pushes bool into **`tom_accuracy`** (`deque`, maxlen **12**). |
| **`generate_state_prompt`** | Prepends **`[ToM feedback]`** with rolling **X / N** correct partner predictions. |

---

### 6.3 `ReflexionAgent`

| Aspect | Behavior |
|--------|----------|
| **`agent_type`** | `"reflexion"` |
| **`__init__`** | Accepts **`reflexion_memory_buffer`**; **`self.episode_memory`** is that list or a new `[]`. |
| **`load_prompt_file("origin")`** | If **`episode_memory`** non-empty, appends block from `reflexion/episode_memory.txt` (`{memory_block}`, `{role}` substituted), up to last **6** entries. |
| **`generate_episode_reflection(statistics_snapshot)`** | Builds a short textual summary from `total_score`, `total_order_finished`, `total_timestamp`; temporarily replaces planner system message and clears dialog cache; calls **`planner.query`** once; appends trimmed response (**≤1200** chars) to **`episode_memory`**. Restores planner state in **`finally`**. |

**Cross-episode memory:** `main.py` initializes **`variant["_reflexion_buffers"] = {"0": [], "1": []}`** once per run. For each slot where `agent_p*` is `reflexion`, the **same list object** is passed as `reflexion_memory_buffer`, so reflections accumulate across **`--episode`** iterations even though agents are recreated each episode.

---

## 7. Driver Integration (`src/main.py`)

### `_save_models_dir_segment(variant)`

Previously: `{model_p0}__{model_p1}` (sanitized).

Now: **`{agent_p0}_{model_p0}__{agent_p1}_{model_p1}`** (each part sanitized), e.g.  
`proagent_gpt-4o__baseline_claude-sonnet-4-20250514`.

### Agent construction loop

For each `LLMPair` slot:

- **`arch`** = `variant["agent_p0"]` or `variant["agent_p1"]` by `actor_num`.
- **`rbuf`** = `variant["_reflexion_buffers"][str(actor_num)]` if `arch == "reflexion"`, else `None`.
- **`make_agent(..., agent_type=arch, reflexion_memory_buffer=rbuf)`**.

### Statistics metadata

- **`statistics_dict["agents"]`**: each dict includes **`agent_type`**. For non-`LLMPair` slots, a small helper maps to **`human`** (or the lowercased algorithm name).
- **`turn_statistics_dict_both["agent_models"]`**: each entry includes **`agent_type`** (from the agent object or slot helper).
- **`statistics_dict["timestep_conversations"]`**: each timestep dict includes **`p0_agent_type`** and **`p1_agent_type`**.

### Post-episode Reflexion

After the **`exp`** inner timestep loop for an episode, for each agent **`isinstance(ag, ReflexionAgent)`**, calls **`generate_episode_reflection(copy.deepcopy(statistics_dict))`**.

---

## 8. Compiler (`src/compile_experiment_run.py`)

### `_agents_agent_types(agents)`

Mirrors `_agents_models`: walks `statistics_dict["agents"]` list of dicts keyed by `player` / `index`, returns **`(p0_agent_type, p1_agent_type)`**. Empty string if missing (old JSON).

### `run_summary(...)`

Output dict now includes **`p0_agent_type`** and **`p1_agent_type`**.

### `timestep_row(...)`

Accepts optional **`p0_agent_type`**, **`p1_agent_type`**; falls back to `agent_models[i].agent_type` when present. **`COLS`** extended with these two columns.

---

## 9. Backward Compatibility

- **Old experiment JSON** without `agent_type` on `agents`: compiler leaves agent-type columns empty; model columns unchanged.
- **Baseline runs**: omitting `--agent_p0` / `--agent_p1` yields **`baseline`** everywhere.
- **`parse_response`**: responses without `partner_prediction` use the legacy analysis regex.

---

## 10. Operational Notes

1. **Working directory:** Run `main.py` from **`src/`** so `prompts/` and keys align with existing layout.
2. **Reflexion cost:** Each Reflexion agent issues **one extra LLM call per episode** at episode end.
3. **Belief / ToM matching:** String heuristics only; they do not replace environment validation.
4. **Greedy / Human:** `agent_p*` flags apply only when `make_agent` is invoked for **`LLMPair`**; Human slots still get `agent_type` metadata from the transport slot in JSON.

---

## 11. Related Code Pointers

| Topic | Location |
|-------|-----------|
| Registry | `src/utils.py` — `AGENT_REGISTRY`, `get_agent_class`, `make_agent` |
| Hooks | `src/collab/collab.py` — `on_planner_response`, `_hook_after_teammate_ml_record`, `generate_ml_action`, `action` |
| Analysis regex | `src/collab/collab.py` — `parse_response`, `mode == "analysis"` |
| Subclasses | `src/collab/agents.py` |
| CLI + logging + Reflexion driver | `src/main.py` |
| Metrics export | `src/compile_experiment_run.py` |

---

*Generated as project documentation for the Agent Architecture Registry implementation.*
