#!/usr/bin/env python3
"""Radar (spider) chart for TGCO outcome shares by level.

Level 1: May 2026 L1 GPT-4o batch (`audit_tgco.py --may-l1-gpt4o`).
Levels 2–3: fixed paper/table values supplied separately.
Overall: macro-average of Level 1–3 percentages (equal weight per level).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CATEGORIES = ["Effective", "Assisted", "Redundant", "Ineffective"]

LEVEL1 = [29.5, 11.4, 29.5, 29.5]
LEVEL2 = [15.4, 39.4, 5.8, 39.4]
LEVEL3 = [6.8, 21.8, 22.6, 48.9]

def _macro_overall(rows: list[list[float]]) -> list[float]:
    n = len(rows)
    return [round(sum(rows[j][i] for j in range(n)) / n, 1) for i in range(4)]

OVERALL = _macro_overall([LEVEL1, LEVEL2, LEVEL3])

SERIES = {
    "Level 1": LEVEL1,
    "Level 2": LEVEL2,
    "Level 3": LEVEL3,
    "Overall": OVERALL,
}

COLORS = {
    "Level 1": "#2e7d32",
    "Level 2": "#1565c0",
    "Level 3": "#c62828",
    "Overall": "#6a1b9a",
}


def _radial_top(series: dict[str, list[float]], headroom: float = 1.08) -> float:
    """Upper radial limit (percent) so the plot is not mostly empty 40–100."""
    m = max(max(v) for v in series.values())
    top = int(np.ceil(m * headroom / 5.0) * 5)
    return float(min(100, max(50, top)))


def main():
    n = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for label, values in SERIES.items():
        vals = list(values) + [values[0]]
        color = COLORS[label]
        ax.plot(angles, vals, "o-", linewidth=2.2, label=label, color=color, markersize=6)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, size=11)
    r_top = _radial_top(SERIES)
    ax.set_ylim(0, r_top)
    ring_step = 10
    rings = list(range(ring_step, int(r_top) + 1, ring_step))
    if rings and rings[-1] < r_top:
        rings.append(int(r_top))
    elif not rings:
        rings = [int(r_top)]
    ax.set_yticks(rings)
    ax.set_yticklabels([f"{t}%" for t in rings], size=9)
    ax.grid(True, linestyle=":", alpha=0.7)

    ax.set_title(
        "TGCO request–action outcomes (% of units)\nby task level "
        f"(radial scale 0–{int(r_top)}%)",
        fontsize=14,
        fontweight="600",
        pad=20,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.02),
        frameon=True,
        fancybox=True,
        shadow=False,
    )

    out = Path(__file__).resolve().parent / "tgco_spider.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
