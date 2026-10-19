"""Figure style: one validated palette, recessive chrome, thin marks.

Categorical slots are assigned in fixed order and follow the *method*, never
its rank, so a method keeps its colour across every figure in the repository.
Identification states use the reserved status palette and are always paired
with a text label (never colour alone).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# fixed method -> slot assignment (colour follows the entity)
METHOD_SLOT = {
    "ismb": 0,
    "jointdiag": 1,
    "boot_percentile": 2,
    "sensitivity": 3,
    "ar_grid": 4,
    "fastica": 5,
    "topk_sae": 6,
    "gmm_bic": 7,
}

STATUS = {
    "identified": "#0ca30c",  # good
    "partial": "#fab219",  # warning
    "misspecified": "#ec835a",  # serious
    "confidently wrong": "#d03b3b",  # critical
}

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def method_color(name: str, fallback_index: int = 0) -> str:
    base = name.split("@")[0].split("+")[0]
    return CATEGORICAL[METHOD_SLOT.get(base, fallback_index) % len(CATEGORICAL)]


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def finish(fig, path, caption: str | None = None) -> None:
    if caption:
        fig.text(0.0, -0.02, caption, ha="left", va="top", fontsize=8, color=INK_2, wrap=True)
    fig.savefig(path)
    plt.close(fig)
