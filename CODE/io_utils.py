"""
Shared utilities for managing analysis outputs.

Provides a single place to determine where artefacts (figures, tables,
textual raw outputs) are written under the OUTPUTS directory.

Directory layout expected:
    Codebase/
        CODE/       <- this file lives here
        DATA/       <- input_data.csv, descriptive exports
        OUTPUTS/    <- raw_outputs/, figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

ANALYSIS_ROOT = Path(__file__).resolve().parent          # CODE/
PROJECT_ROOT = ANALYSIS_ROOT.parent                       # Codebase/
OUTPUT_ROOT = PROJECT_ROOT / "OUTPUTS"
FIGURES_DIR = OUTPUT_ROOT / "figures"
RAW_OUTPUTS_DIR = OUTPUT_ROOT / "raw_outputs"

for directory in (FIGURES_DIR, RAW_OUTPUTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

_FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}


def _target_dir(filename: str) -> Path:
    """Return the output path for a given filename."""
    ext = Path(filename).suffix.lower()
    if ext in _FIGURE_EXTENSIONS:
        return FIGURES_DIR / filename
    return RAW_OUTPUTS_DIR / filename


def save_figure(fig, filename: str, **kwargs) -> Path:
    """Save a matplotlib figure under the structured output tree."""
    target = _target_dir(filename)
    fig.savefig(target, **kwargs)
    plt.close(fig)
    print(f"[saved] figure -> {target.relative_to(PROJECT_ROOT)}")
    return target


def write_text(filename: str, content: str, *, encoding: str = "utf-8") -> Path:
    """Write textual content to the structured raw-output directory."""
    target = _target_dir(filename)
    target.write_text(content, encoding=encoding)
    print(f"[saved] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


def write_with_writer(
    filename: str, writer: Callable[[object], None], *, encoding: str = "utf-8"
) -> Path:
    """Write content through a callable that receives an open file handle."""
    target = _target_dir(filename)
    with target.open("w", encoding=encoding) as handle:
        writer(handle)
    print(f"[saved] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


def copy_to_output(source: Path, filename: str | None = None) -> Path:
    """Copy an existing file into the structured output directory."""
    filename = filename or source.name
    target = _target_dir(filename)
    target.write_bytes(Path(source).read_bytes())
    print(f"[copied] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


__all__ = [
    "ANALYSIS_ROOT",
    "PROJECT_ROOT",
    "OUTPUT_ROOT",
    "FIGURES_DIR",
    "RAW_OUTPUTS_DIR",
    "save_figure",
    "write_text",
    "write_with_writer",
    "copy_to_output",
]
