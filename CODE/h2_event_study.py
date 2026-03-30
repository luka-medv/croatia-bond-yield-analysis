"""
H2 Event Study: immediate yield reactions around Croatia's euro adoption.

- Event: 2023-01-01 (official euro introduction).
- Windows: +/-5 days and +/-3 days.
- Method: abnormal yield = yield - 30d pre-event average; one-sample t-test vs 0.
- Countries: Croatia, Slovenia, Slovakia, Lithuania, France, Germany.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Tuple

import pandas as pd
from scipy import stats

from io_utils import write_text

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - defensive
        pass

COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
EVENT_DATE = pd.Timestamp("2023-01-01")
EVENT_LABEL = "Euro Adoption 2023-01-01"


def abnormal_series(
    df: pd.DataFrame, country: str, event_date: pd.Timestamp
) -> Tuple[pd.Series, pd.Series] | None:
    """Return abnormal yield windows (+/-5d and +/-3d) or None when data missing."""

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
        "=" * 80,
        "H2 EVENT STUDY: +/-5d and +/-3d around euro adoption (abnormal yields)",
        "=" * 80,
        "",
        "Abnormal yield = yield - mean(yield over t-30..t-1). One-sample t-test vs 0.",
        "Countries: HR, SI, SK, LT, FR, DE.",
        "",
    ]

    lines.append(f"Event: {EVENT_LABEL}")
    for country in COUNTRIES:
        series = abnormal_series(df, country, EVENT_DATE)
        if series is None:
            lines.append(f"  {country}: no data")
            continue

        s5, s3 = series
        t5 = stats.ttest_1samp(s5.values, 0.0, nan_policy="omit")
        t3 = stats.ttest_1samp(s3.values, 0.0, nan_policy="omit")
        lines.append(
            f"  {country}: +/-5d mean={s5.mean():.4f}, t={t5.statistic:.2f}, "
            f"p={t5.pvalue:.4f}; +/-3d mean={s3.mean():.4f}, "
            f"t={t3.statistic:.2f}, p={t3.pvalue:.4f}"
        )
    lines.append("")

    write_text("h2_event_study_results.txt", "\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
