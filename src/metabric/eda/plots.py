"""Plot helpers: one consistent look for every EDA figure (static PNG; the CSV is the table view).

Palette: reference instance from the dataviz method. Categorical hues in fixed order, one-hue
sequential ramp for magnitude, blue <-> gray <-> red for diverging; recessive grid; thin marks.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALIVE, DIED = CAT[0], CAT[1]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#a32323"])


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8,
        "font.family": "sans-serif", "lines.linewidth": 2, "patch.linewidth": 0,
    })


def fig(w: float = 6.4, h: float = 4.0, nrows: int = 1, ncols: int = 1, **kw):
    setup()
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def dot_ci(ax, labels, rate, lo, hi, color=CAT[0], ref: float | None = None, ref_label: str | None = None) -> None:
    """Horizontal dot + 95% interval per level; optional dashed reference line (e.g. overall rate)."""
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, color=color, linewidth=2)
    ax.plot(rate, y, "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if ref is not None:
        ax.axvline(ref, color=INK2, linestyle="--", linewidth=1)
        if ref_label:
            ax.text(ref, len(labels) - 0.4, f" {ref_label}", color=INK2, fontsize=7, va="bottom")
    grid(ax, "x")
