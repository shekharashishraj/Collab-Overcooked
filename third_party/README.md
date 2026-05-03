# third_party reference clones

Local clones of upstream research code for **prompting, belief tracking, and feedback** inspection only. They are **gitignored** and must be created on each machine.

## Clone

From the **repository root** (parent of `third_party/`):

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/shekharashishraj/ProAgent.git ProAgent
git clone https://github.com/shekharashishraj/Adaptive-ToM.git Adaptive-ToM
```

## Do not merge runtimes

- Do **not** add `third_party/ProAgent` or `third_party/Adaptive-ToM` to `PYTHONPATH` in this project’s main environment.
- Upstream stacks target older Python / TensorFlow 1.x and bundled `overcooked` layouts; this repo uses [`lib/overcooked_ai`](../lib/overcooked_ai) and [`src/`](../src/). Treat these trees as **read-only reference**.

## Licenses

- [ProAgent](https://github.com/shekharashishraj/ProAgent) — fork of PKU-Alignment/ProAgent; see `ProAgent/LICENSE` after clone.
- [Adaptive-ToM](https://github.com/shekharashishraj/Adaptive-ToM) — fork of ChunjiangMonkey/Adaptive-ToM; see `Adaptive-ToM/readme.md` and repo license after clone.

## Inspection summary

See [INSPECTION.md](INSPECTION.md) for Phase 1 notes: prompting structure, belief/partner tracking, feedback loops, and a gap table vs Collab-Overcooked [`src/collab/agents.py`](../src/collab/agents.py) and [`src/prompts/agents/`](../src/prompts/agents/).
