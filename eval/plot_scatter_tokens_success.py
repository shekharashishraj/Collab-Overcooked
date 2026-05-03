#!/usr/bin/env python3
"""Scatter: Success (%) vs Tokens/Ep. (log scale) with optional Verif. Err. point size."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _short_team(row: pd.Series) -> str:
    def s(x: str) -> str:
        x = str(x)
        if x.startswith("GPT"):
            return "GPT-4o"
        if x.startswith("Claude"):
            return "Claude"
        if x.startswith("Gemma"):
            return "Gemma"
        return x[:8]

    return f"{s(row['chef'])}→{s(row['assistant'])}"


def plot_scatter(df: pd.DataFrame, out_path: Path, scale_verif: bool = True) -> Path:
    df = df.copy()
    df["_lab"] = df.apply(_short_team, axis=1)
    tok = df["tokens_ep"].clip(lower=1)
    sizes = df["verif_err"].clip(lower=0.5) * 35 if scale_verif else 80

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(df["success_pct"], tok, s=sizes, c=df["follow_rate"], cmap="viridis", alpha=0.85, edgecolors="black", linewidths=0.6)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Follow rate")

    for _, r in df.iterrows():
        ax.annotate(
            r["_lab"],
            (r["success_pct"], r["tokens_ep"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            ha="left",
        )

    ax.set_xlabel("Success (%)")
    ax.set_ylabel("Tokens / episode (log scale)")
    ax.set_yscale("log")
    ax.set_title("Outcome vs communication cost (point size ∝ Verif. err.)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
