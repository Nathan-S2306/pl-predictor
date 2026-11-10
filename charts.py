"""Matplotlib figures used by the app. Each function returns a Figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#ffffff"
INK = "#14201a"
INK_2 = "#3d4a42"
MUTED = "#5b6a60"
GRID = "#e6ece5"
BLUE = "#0f5132"
# One hue, light -> dark: probability is a magnitude.
SEQ = LinearSegmentedColormap.from_list(
    "seq_green", [SURFACE, "#dcefe3", "#a3d4b7", "#55a87c", "#1f7a4d", "#0b3d26"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _cell_text(ax, x, y, text, value, vmax, size=8):
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color="#ffffff" if value > 0.55 * vmax else INK)


def scoreline_heatmap(P: np.ndarray, home: str, away: str, max_show: int = 5):
    """Probability of each exact score, home goals down the side, away goals across."""
    M = P[:max_show + 1, :max_show + 1]
    vmax = float(M.max())
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=vmax, aspect="equal")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            _cell_text(ax, j, i, f"{v:.0%}" if v >= 0.005 else "", v, vmax)
    ax.set_xticks(range(max_show + 1)); ax.set_yticks(range(max_show + 1))
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel(f"{away} goals", labelpad=8); ax.set_ylabel(f"{home} goals", labelpad=8)
    ax.set_xticks(np.arange(-.5, max_show + 1), minor=True); ax.set_yticks(np.arange(-.5, max_show + 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def position_heatmap(position_probs, label_min: float = 0.05, highlight: str | None = None):
    """Chance of each finishing position. Rows are teams (best first), columns are positions."""
    teams = list(position_probs.index)
    M = position_probs.to_numpy()
    n_t, n_p = M.shape
    vmax = float(M.max())
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * n_t + 1.0))
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    for i in range(n_t):
        for j in range(n_p):
            v = M[i, j]
            if v >= label_min:
                _cell_text(ax, j, i, f"{v * 100:.0f}", v, vmax, size=7.5)
    ax.set_xticks(range(n_p)); ax.set_xticklabels(range(1, n_p + 1))
    ax.set_yticks(range(n_t)); ax.set_yticklabels(teams)
    for lab in ax.get_yticklabels():
        if lab.get_text() == highlight:
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Final position (cells show % chance; blank = under 5%)", labelpad=8)
    ax.set_xticks(np.arange(-.5, n_p), minor=True); ax.set_yticks(np.arange(-.5, n_t), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5); ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def points_histogram(points: np.ndarray, team: str, current_points: int):
    """Distribution of a team's final points across simulations."""
    lo, hi = int(points.min()), int(points.max())
    counts = np.bincount(points - lo, minlength=hi - lo + 1) / len(points)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.bar(np.arange(lo, hi + 1), counts, width=0.8, color=BLUE)
    mean = float(points.mean())
    ax.plot([mean, mean], [0, counts.max() * 1.06], color=INK, linewidth=1)
    ax.text(mean, counts.max() * 1.08, f"average {mean:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(f"{team} final points (has {current_points} now)"); ax.set_ylabel("Share of simulations")
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(length=0)
    fig.tight_layout()
    return fig
