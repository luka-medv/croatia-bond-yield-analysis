"""
Compatibility wrapper for the combined Column C placebo report.

The source-of-truth placebo logic lives inside:
  - h1_ecb_rate_hike_impact.py
  - h2_euro_adoption_impact.py

This script only aggregates the two standalone placebo reports into the
legacy combined output file `placebo_column_c_results.txt`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from io_utils import write_text
from placebo_utils import extract_report_body

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT.parent / "OUTPUTS" / "reports"
H1_REPORT = REPORTS_DIR / "h1_placebo_results.txt"
H2_REPORT = REPORTS_DIR / "h2_placebo_results.txt"


def _load_body(report_path: Path) -> list[str]:
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing dependency: {report_path.name}. "
            "Run the corresponding H1/H2 analysis first."
    )
    return extract_report_body(report_path.read_text(encoding="utf-8"))


def _trim_to_placebo_section(lines: list[str]) -> list[str]:
    trimmed = []
    for line in lines:
        if line.startswith("Main Effect"):
            break
        trimmed.append(line)
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    return trimmed


def run() -> None:
    lines = [
        "=" * 70,
        "PLACEBO TESTS - COLUMN C SPECIFICATION",
        "Each regression includes the actual treatment indicator,",
        "so that placebo coefficients are net of the real treatment effect.",
        "HC3 robust standard errors throughout.",
        "=" * 70,
        "",
    ]

    h1_body = _trim_to_placebo_section(_load_body(H1_REPORT))
    h2_body = _trim_to_placebo_section(_load_body(H2_REPORT))

    lines.extend(h1_body)
    lines.append("")
    lines.extend(h2_body)
    lines.append("")

    write_text("placebo_column_c_results.txt", "\n".join(lines) + "\n")
    print("\nSaved: placebo_column_c_results.txt")


if __name__ == "__main__":
    run()
