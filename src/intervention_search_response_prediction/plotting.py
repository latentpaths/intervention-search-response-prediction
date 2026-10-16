"""Shared styles for scientific result plots."""

from __future__ import annotations

from pathlib import Path

NAVY = "#27465F"
TEAL = "#147D78"
RUST = "#B45C3C"
BRIER = NAVY
DISCRIMINATION = TEAL
TARGET = RUST
BRIER_LIGHT = "#7793A9"
DISCRIMINATION_LIGHT = "#6B9F99"
TARGET_LIGHT = "#C78E72"
DISCRIMINATION_BG = "#EDF6F4"
SLATE = "#798994"
INK = "#26323A"
MUTED = "#5D6870"
GRID = "#DCE2E5"
PALE = "#F3F6F6"


def configure() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "font.size": 8,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.facecolor": "white",
        }
    )


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.035)
