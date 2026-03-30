"""
Shared plotting utilities used by analysis and data scripts.

Provides consistent styling, subplot creation, legend placement,
dual-outline significance markers, and bar-label helpers.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.container import BarContainer


# ── house style ──────────────────────────────────────────────────────
_STYLE_APPLIED = False

def _apply_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.30,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.framealpha": 0.95,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    _STYLE_APPLIED = True


# ── subplot factory ──────────────────────────────────────────────────
def make_subplots(
    figsize: tuple[float, float] = (10, 6),
    nrows: int = 1,
    ncols: int = 1,
    constrained: bool = False,
    **kwargs,
):
    """Create a figure and axes with the house style applied.

    Returns ``(fig, ax)`` when *nrows* and *ncols* are both 1, otherwise
    ``(fig, axes)`` (the raw array from ``plt.subplots``).
    """
    _apply_style()
    layout = "constrained" if constrained else "tight"
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        layout=layout,
        **kwargs,
    )
    return fig, axes


# ── legend helper ────────────────────────────────────────────────────
def place_legend(
    ax,
    *,
    location: str = "best",
    anchor: tuple[float, float] | None = None,
    ncol: int = 1,
    **kwargs,
) -> None:
    """Place a legend with sensible defaults.

    When *anchor* is given the legend is placed outside the axes using
    ``bbox_to_anchor``.
    """
    kw: dict[str, Any] = dict(
        loc=location,
        ncol=ncol,
        framealpha=0.95,
        edgecolor="#cccccc",
        **kwargs,
    )
    if anchor is not None:
        kw["bbox_to_anchor"] = anchor
    ax.legend(**kw)


# ── annotation helper ───────────────────────────────────────────────
def add_annotation(ax, text: str, xy, **kwargs) -> None:
    """Thin wrapper around ``ax.annotate`` with default font styling."""
    defaults = dict(fontsize=10, fontweight="bold")
    defaults.update(kwargs)
    ax.annotate(text, xy=xy, **defaults)


# ── significance markers ────────────────────────────────────────────
def add_dual_outline(ax, bar, *, color: str = "#d62728", linewidth: float = 2.5) -> None:
    """Draw a coloured outline around a bar to mark statistical significance."""
    rect = bar.get_bbox()
    x0, y0 = rect.x0, rect.y0
    w = rect.width
    h = rect.height
    border = plt.Rectangle(
        (x0, y0), w, h,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        zorder=bar.get_zorder() + 1,
    )
    ax.add_patch(border)


def _sig_stars(p: float | None) -> str:
    if p is None or np.isnan(p):
        return ""
    if p <= 0.0001:
        return "****"
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def label_bars_with_significance(
    ax,
    bars,
    *,
    pvalues: Sequence[float | None],
    fmt: str = "{:.4f}",
    fontsize: int = 10,
) -> None:
    """Add coefficient labels with significance stars above / below each bar."""
    for bar, pval in zip(bars, pvalues):
        height = bar.get_height()
        stars = _sig_stars(pval)
        label = fmt.format(height)
        if stars:
            label = f"{label}{stars}"
        va = "bottom" if height >= 0 else "top"
        offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=fontsize,
            fontweight="bold",
        )


# ── text adjustment ─────────────────────────────────────────────────
def adjust_text_labels(texts, **kwargs) -> None:
    """Wrapper around adjustText.adjust_text with sane defaults.

    Falls back to a no-op if adjustText is not installed so scripts
    never crash just because of label adjustment.
    """
    try:
        from adjustText import adjust_text
        defaults = dict(
            force_text=(0.3, 0.5),
            expand_text=(1.05, 1.2),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )
        defaults.update(kwargs)
        adjust_text(texts, **defaults)
    except ImportError:
        pass


__all__ = [
    "make_subplots",
    "place_legend",
    "add_annotation",
    "add_dual_outline",
    "label_bars_with_significance",
    "adjust_text_labels",
]
