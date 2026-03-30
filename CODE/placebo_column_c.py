"""
Compatibility wrapper for the combined Column C placebo report.

The source-of-truth placebo logic now lives inside:
  - h1_ecb_rate_hike_impact.py
  - h2_euro_adoption_impact.py

This script keeps the legacy combined output file
`placebo_column_c_results.txt` for backwards compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from io_utils import write_text
from h1_ecb_rate_hike_impact import prepare_h1_panel, run_h1_column_c_placebos
from h2_euro_adoption_impact import prepare_h2_panel, run_h2_column_c_placebos

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "DATA" / "input_data.csv"


def _sig_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def _format_section(title: str, payload: dict) -> list[str]:
    lines = [
        title,
        f"  Panel: {', '.join(payload['countries'])}",
        f"  Sample: 2021-01-01 to 2024-12-31  N={payload['sample_n']}",
        "",
    ]

    for result in payload['results']:
        lines.append(
            f"  {result['name']} ({result['date']}, -{result['months_before']}mo): "
            f"coef={result['coefficient']:+.4f}  SE={result['se']:.4f}  "
            f"p={result['pvalue']:.4f} {_sig_stars(result['pvalue'])}"
        )
    return lines


def run() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    h1_payload = run_h1_column_c_placebos(
        prepare_h1_panel(df),
        save_report=False,
        save_plot=False,
        verbose=True,
    )

    print()

    h2_payload = run_h2_column_c_placebos(
        prepare_h2_panel(df),
        save_report=False,
        save_plot=False,
        verbose=True,
    )

    lines = [
        "=" * 70,
        "PLACEBO TESTS - COLUMN C SPECIFICATION",
        "Each regression includes the actual treatment indicator,",
        "so that placebo coefficients are net of the real treatment effect.",
        "HC3 robust standard errors throughout.",
        "=" * 70,
        "",
    ]
    lines.extend(_format_section("H1: ECB rate hike (27 July 2022)", h1_payload))
    lines.extend([
        "",
    ])
    lines.extend(_format_section("H2: Euro adoption (1 January 2023)", h2_payload))
    lines.append("")

    write_text("placebo_column_c_results.txt", "\n".join(lines) + "\n")
    print("\nSaved: placebo_column_c_results.txt")


if __name__ == "__main__":
    run()
