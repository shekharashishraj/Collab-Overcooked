#!/usr/bin/env python3
"""Chef × Assistant heatmaps for scalar metrics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _model_order() -> List[str]:
    return ["GPT-4o", "Claude Sonnet 4", "Gemma-31B"]


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    order = _model_order()
    p = df.pivot(index="chef", columns="assistant", values=metric)
    return p.reindex(index=order, columns=order)


def load_team_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "chef",
        "assistant",
        "success_pct",
        "adr_dup_say",
        "idensity",
        "follow_rate",
        "tokens_ep",
        "verif_err",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cell_fmt: str = ".2f",
    annotate_values: Optional[pd.DataFrame] = None,
) -> Path:
    """If annotate_values is set, show those numbers in cells (e.g. raw tokens on log-colored map)."""
    p = pivot_metric(df, metric)
    arr = p.to_numpy(dtype=float)
    ann = annotate_values.reindex(index=p.index, columns=p.columns).to_numpy() if annotate_values is not None else arr

    fig, ax = plt.subplots(figsize=(7, 5.5))
    arr_plot = np.ma.masked_invalid(arr)
    im = ax.imshow(arr_plot, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(p.columns)))
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_xticklabels(p.columns, rotation=25, ha="right")
    ax.set_yticklabels(p.index)
    ax.set_xlabel("Assistant")
    ax.set_ylabel("Chef")
    ax.set_title(title)

    lo, hi = np.nanmin(arr), np.nanmax(arr)
    mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else 0.5

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            a = ann[i, j]
            if np.isnan(v):
                txt = "—"
            elif isinstance(a, (int, float)) and np.isfinite(a):
                txt = format(a, cell_fmt)
            else:
                txt = str(a)
            use_white = np.isfinite(v) and v > mid
            ax.text(j, i, txt, ha="center", va="center", color="white" if use_white else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_tokens_log_heatmap(df: pd.DataFrame, out_path: Path) -> Path:
    df2 = df.copy()
    df2["_logtok"] = np.log10(df2["tokens_ep"].clip(lower=1))
    raw = pivot_metric(df, "tokens_ep")
    logp = pivot_metric(df2, "_logtok")
    return plot_metric_heatmap(
        df2,
        "_logtok",
        "Tokens / episode (log₁₀ color scale)",
        out_path,
        cmap="cividis",
        vmin=float(np.nanmin(logp.to_numpy())),
        vmax=float(np.nanmax(logp.to_numpy())),
        cell_fmt=".0f",
        annotate_values=raw,
    )


def plot_default_heatmaps(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    paths: List[Path] = []
    specs: List[Tuple[str, str, str, Optional[float], Optional[float], str]] = [
        ("success_pct", "Success (%)", "Blues", 0.0, 100.0, ".0f"),
        ("follow_rate", "Follow rate", "Greens", 0.0, 1.0, ".2f"),
        ("adr_dup_say", "ADR (dup-say proxy)", "Oranges", 0.0, 1.0, ".2f"),
        ("idensity", "IDensity", "Purples", None, None, ".2f"),
        ("verif_err", "Verifier window count", "Reds", None, None, ".1f"),
    ]
    for col, title, cmap, vmin, vmax, fmt in specs:
        paths.append(plot_metric_heatmap(df, col, title, out_dir / f"heatmap_{col}.png", cmap=cmap, vmin=vmin, vmax=vmax, cell_fmt=fmt))
    paths.append(plot_tokens_log_heatmap(df, out_dir / "heatmap_tokens_ep_log.png"))
    return paths
