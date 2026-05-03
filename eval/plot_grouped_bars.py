#!/usr/bin/env python3
"""Grouped bar charts for Tokens/Ep. and Verif. Err. across Chef×Assistant teams."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _team_label(row: pd.Series) -> str:
    c = str(row["chef"]).replace(" ", "\n")
    a = str(row["assistant"]).replace(" ", "\n")
    return f"{c}\n→\n{a}"


def plot_tokens_and_verifier(df: pd.DataFrame, out_path: Path) -> Path:
    df = df.copy()
    df["_label"] = df.apply(_team_label, axis=1)
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].bar(x, df["tokens_ep"], color="#4e79a7", edgecolor="black", linewidth=0.3)
    axes[0].set_ylabel("Tokens / episode")
    axes[0].set_title("Communication tokens per episode by team")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x, df["verif_err"], color="#e15759", edgecolor="black", linewidth=0.3)
    axes[1].set_ylabel("Verif. err.")
    axes[1].set_title("Verifier window count by team")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["_label"], fontsize=7)
    fig.suptitle("Team comparison (Chef × Assistant)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_four_metrics_small_multiples(df: pd.DataFrame, out_path: Path) -> Path:
    """ADR, IDensity, FollowRate, Success in one figure (four rows of bars)."""
    df = df.copy()
    df["_label"] = df.apply(_team_label, axis=1)
    metrics = [
        ("success_pct", "Success (%)", "#59a14f"),
        ("follow_rate", "Follow rate", "#76b7b2"),
        ("adr_dup_say", "ADR (dup-say)", "#edc948"),
        ("idensity", "IDensity", "#b07aa1"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 9), sharex=True)
    x = np.arange(len(df))
    for ax, (col, title, color) in zip(axes, metrics):
        ax.bar(x, df[col], color=color, edgecolor="black", linewidth=0.3)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(df["_label"], fontsize=7)
    fig.suptitle("Secondary metrics by team", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
