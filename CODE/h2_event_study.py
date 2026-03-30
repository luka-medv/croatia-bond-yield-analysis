

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from io_utils import save_figure, write_text
from plot_utils import make_subplots, place_legend

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  
        pass

COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
EVENT_DATE = pd.Timestamp("2023-01-01")
EVENT_LABEL = "Euro Adoption 2023-01-01"


def abnormal_series(
    df: pd.DataFrame, country: str, event_date: pd.Timestamp
) -> Tuple[pd.Series, pd.Series] | None:
    

    c = df[df["country"] == country].sort_values("date").set_index("date")
    if c.empty:
        return None

    pre_window = c.loc[
        (c.index < event_date) & (c.index >= event_date - pd.Timedelta(days=30))
    ]
    if pre_window.empty:
        return None

    baseline = pre_window["bond_yield_10y"].mean()
    s5 = (
        c.loc[
            (c.index >= event_date - pd.Timedelta(days=5))
            & (c.index <= event_date + pd.Timedelta(days=5))
        ]["bond_yield_10y"]
        - baseline
    )
    s3 = (
        c.loc[
            (c.index >= event_date - pd.Timedelta(days=3))
            & (c.index <= event_date + pd.Timedelta(days=3))
        ]["bond_yield_10y"]
        - baseline
    )
    return s5.dropna(), s3.dropna()


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "DATA" / "input_data.csv"


def run() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["country"].isin(COUNTRIES)].copy()
    

    lines = [
         * 80,
        ,
         * 80,
        ,
        ,
        ,
        ,
    ]

    values_window5: list[float] = []
    values_window3: list[float] = []

    lines.append(f"Event: {EVENT_LABEL}")
    for country in COUNTRIES:
        series = abnormal_series(df, country, EVENT_DATE)
        if series is None:
            lines.append(f"  {country}: no data")
            values_window5.append(np.nan)
            values_window3.append(np.nan)
            continue

        s5, s3 = series
        t5 = stats.ttest_1samp(s5.values, 0.0, nan_policy="omit")
        t3 = stats.ttest_1samp(s3.values, 0.0, nan_policy="omit")
        lines.append(
            
            
        )
        values_window5.append(float(s5.mean()))
        values_window3.append(float(s3.mean()))
    lines.append("")

    write_text("h2_event_study_results.txt", "\n".join(lines) + "\n")

    x = np.arange(len(COUNTRIES))
    palette = ["#0F6CE0", "#0F6CE0"]

    def _plot(values: list[float], window_label: str, color: str, filename: str, *, hatch: str = "") -> None:
        fig, ax = make_subplots(figsize=(10, 6))

        bars = ax.bar(
            x,
            values,
            0.6,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.0,
            hatch=hatch,
        )
        ax.axhline(0, color="#666666", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in COUNTRIES], fontweight="bold")
        ax.set_ylabel(f"Mean abnormal yield (pp), {window_label}", fontweight="bold")
        ax.margins(x=0.04, y=0.18)

        ax.bar_label(bars, fmt="%.3f", padding=6, fontsize=11, color="white")

        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            ymin, ymax = -0.1, 0.1
        else:
            ymin, ymax = float(finite.min()), float(finite.max())
        span = max(0.1, ymax - ymin)
        ax.set_ylim(ymin - span * 0.25, ymax + span * 0.35)

        save_figure(fig, filename, dpi=300)

    _plot(values_window5, "+/-5 days", palette[0], "h2_event_study_window5.png")
    _plot(values_window3, "+/-3 days", palette[1], "h2_event_study_window3.png")


if __name__ == "__main__":
    run()
