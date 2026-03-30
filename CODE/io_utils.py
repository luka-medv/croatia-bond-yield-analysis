

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

ANALYSIS_ROOT = Path(__file__).resolve().parent          
PROJECT_ROOT = ANALYSIS_ROOT.parent                       
OUTPUT_ROOT = PROJECT_ROOT / "OUTPUTS"
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
REPORTS_DIR = OUTPUT_ROOT / "reports"

for directory in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

_FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
_TABLE_EXTENSIONS = {".tex"}


def _target_dir(filename: str) -> Path:
    
    ext = Path(filename).suffix.lower()
    if ext in _FIGURE_EXTENSIONS:
        return FIGURES_DIR / filename
    if ext in _TABLE_EXTENSIONS:
        return TABLES_DIR / filename
    return REPORTS_DIR / filename


def save_figure(fig, filename: str, **kwargs) -> Path:
    
    target = _target_dir(filename)
    fig.savefig(target, **kwargs)
    plt.close(fig)
    print(f"[saved] figure -> {target.relative_to(PROJECT_ROOT)}")
    return target


def write_text(filename: str, content: str, *, encoding: str = "utf-8") -> Path:
    
    target = _target_dir(filename)
    target.write_text(content, encoding=encoding)
    print(f"[saved] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


def write_with_writer(
    filename: str, writer: Callable[[object], None], *, encoding: str = "utf-8"
) -> Path:
    
    target = _target_dir(filename)
    with target.open("w", encoding=encoding) as handle:
        writer(handle)
    print(f"[saved] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


def copy_to_output(source: Path, filename: str | None = None) -> Path:
    
    filename = filename or source.name
    target = _target_dir(filename)
    target.write_bytes(Path(source).read_bytes())
    print(f"[copied] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


__all__ = [
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
    ,
]
