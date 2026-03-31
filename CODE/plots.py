"""
Generate the final figure set used in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = PROJECT_ROOT / "DATA" / "input_data.csv"
FIGURES_DIR = PROJECT_ROOT / "OUTPUTS" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MA_WINDOW = 30
PAPER_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
CONTROL_COUNTRIES = ["Slovenia", "Slovakia", "Lithuania"]
COLORS = {
    "Croatia": "#D2691E",
    "Slovenia": "#0F6CE0",
    "Slovakia": "#0E7490",
    "Lithuania": "#16A34A",
    "France": "#7C3AED",
    "Germany": "#374151",
}

plt.rcParams.update(
    {
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
        "legend.fontsize": 10,
        "legend.framealpha": 0.95,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def _save_figure(fig, filename: str, **kwargs) -> Path:
    target = FIGURES_DIR / filename
    fig.savefig(target, **kwargs)
    plt.close(fig)
    print(f"[saved] figure -> {target.relative_to(PROJECT_ROOT)}")
    return target


def _moving_average(series: pd.Series) -> pd.Series:
    return series.rolling(MA_WINDOW, min_periods=10).mean()


def _decorate_date_axis(fig, ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_country_series(ax, subdf: pd.DataFrame, column: str, *, use_ma: bool = False) -> None:
    for country in PAPER_COUNTRIES:
        country_df = subdf[subdf["country"] == country].sort_values("date")
        if country_df.empty:
            continue
        series = country_df[column]
        if use_ma:
            series = _moving_average(series)
        linewidth = 2.8 if country == "Croatia" else 1.8
        linestyle = "--" if country == "Germany" and column == "bond_yield_10y" else "-"
        ax.plot(
            country_df["date"],
            series,
            label=country,
            color=COLORS[country],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.95 if country == "Croatia" else 0.85,
        )


def _create_figure_3_1(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    full = df[df["country"].isin(PAPER_COUNTRIES)].copy()
    _plot_country_series(ax, full, "bond_yield_10y", use_ma=True)
    ax.axvline(pd.Timestamp("2023-01-01"), color="#0E7490", linestyle="--", linewidth=2, alpha=0.75)
    ax.set_title("10-Year Government Bond Yields: All Countries (2015-2024)", fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("10-Year Bond Yield (%)", fontweight="bold")
    ax.legend(loc="upper left", ncol=2)
    _decorate_date_axis(fig, ax)
    _save_figure(fig, "figure_3_1_all_countries_yields.png")


def _create_figure_3_3(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    spread_countries = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France"]
    subdf = df[df["country"].isin(spread_countries)].copy()
    for country in spread_countries:
        country_df = subdf[subdf["country"] == country].sort_values("date")
        if country_df.empty:
            continue
        linewidth = 2.8 if country == "Croatia" else 1.8
        ax.plot(
            country_df["date"],
            _moving_average(country_df["spread_vs_germany"]),
            label=country,
            color=COLORS[country],
            linewidth=linewidth,
            alpha=0.95 if country == "Croatia" else 0.85,
        )
    ax.axhline(0, color="#374151", linewidth=1.0, alpha=0.5)
    ax.axvline(pd.Timestamp("2023-01-01"), color="#0E7490", linestyle="--", linewidth=2, alpha=0.75)
    ax.set_title("Government Bond Yield Spreads Relative to Germany", fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Spread vs Germany (pp)", fontweight="bold")
    ax.legend(loc="upper left", ncol=2)
    _decorate_date_axis(fig, ax)
    _save_figure(fig, "figure_3_3_spreads_vs_germany.png")


def _create_figure_4_1(df: pd.DataFrame) -> None:
    panel = df[
        (df["country"].isin(["Croatia", *CONTROL_COUNTRIES]))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    croatia = panel[panel["country"] == "Croatia"].set_index("date")["bond_yield_10y"].sort_index()
    control = panel[panel["country"] != "Croatia"].groupby("date")["bond_yield_10y"].mean().sort_index()
    croatia_ma = _moving_average(croatia)
    control_ma = _moving_average(control)
    event_date = pd.Timestamp("2022-07-27")

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(croatia.index, croatia.values, color="#e74c3c", alpha=0.15, linewidth=0.5)
    ax.plot(control.index, control.values, color="#3498db", alpha=0.15, linewidth=0.5)
    ax.plot(croatia_ma.index, croatia_ma.values, color="#e74c3c", linewidth=3, label="Croatia (Treatment)")
    ax.plot(control_ma.index, control_ma.values, color="#3498db", linewidth=3, label="Control Group (SI, SK, LT)")
    ax.axvline(event_date, color="#9A3412", linestyle="--", linewidth=2.5, alpha=0.8, label="ECB Rate Hike (27 Jul 2022)")
    ax.axvspan(panel["date"].min(), event_date, alpha=0.06, color="#3498db")
    ax.axvspan(event_date, panel["date"].max(), alpha=0.06, color="#e74c3c")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("10-Year Bond Yield (%)", fontweight="bold")
    ax.legend(loc="upper left")
    _decorate_date_axis(fig, ax)
    _save_figure(fig, "figure_4_1_h1_did_visual.png")


def _create_figure_4_5(df: pd.DataFrame) -> None:
    countries = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
    panel = df[
        (df["country"].isin(countries))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    croatia = panel[panel["country"] == "Croatia"].set_index("date")["bond_yield_10y"].sort_index()
    control = panel[panel["country"] != "Croatia"].groupby("date")["bond_yield_10y"].mean().sort_index()
    croatia_ma = _moving_average(croatia)
    control_ma = _moving_average(control)
    event_date = pd.Timestamp("2023-02-02")

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(croatia.index, croatia.values, color="#e74c3c", alpha=0.15, linewidth=0.5)
    ax.plot(control.index, control.values, color="#3498db", alpha=0.15, linewidth=0.5)
    ax.plot(croatia_ma.index, croatia_ma.values, color="#e74c3c", linewidth=3, label="Croatia (Treatment)")
    ax.plot(control_ma.index, control_ma.values, color="#3498db", linewidth=3, label="Control Group (SI, SK, LT, FR, DE)")
    ax.axvline(event_date, color="#9A3412", linestyle="--", linewidth=2.5, alpha=0.8, label="ECB Rate Hike (2 Feb 2023)")
    ax.axvline(pd.Timestamp("2022-07-27"), color="#6B7280", linestyle=":", linewidth=1.5, alpha=0.5, label="First ECB Hike (27 Jul 2022)")
    ax.axvspan(panel["date"].min(), event_date, alpha=0.06, color="#3498db")
    ax.axvspan(event_date, panel["date"].max(), alpha=0.06, color="#e74c3c")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("10-Year Bond Yield (%)", fontweight="bold")
    ax.legend(loc="upper left")
    _decorate_date_axis(fig, ax)
    _save_figure(fig, "figure_4_5_h1b_did_visual.png")


def _create_figure_4_7(df: pd.DataFrame) -> None:
    panel = df[
        (df["country"].isin(["Croatia", *CONTROL_COUNTRIES]))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    croatia = panel[panel["country"] == "Croatia"].set_index("date")["spread_vs_germany"].sort_index()
    control = panel[panel["country"] != "Croatia"].groupby("date")["spread_vs_germany"].mean().sort_index()
    croatia_ma = _moving_average(croatia)
    control_ma = _moving_average(control)
    event_date = pd.Timestamp("2023-01-01")

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(croatia.index, croatia.values, color="#e74c3c", alpha=0.15, linewidth=0.5)
    ax.plot(control.index, control.values, color="#3498db", alpha=0.15, linewidth=0.5)
    ax.plot(croatia_ma.index, croatia_ma.values, color="#e74c3c", linewidth=3, label="Croatia (Treatment)")
    ax.plot(control_ma.index, control_ma.values, color="#3498db", linewidth=3, label="Control Group (SI, SK, LT)")
    ax.axvline(event_date, color="#0E7490", linestyle="--", linewidth=2.5, alpha=0.8, label="Euro Adoption (1 Jan 2023)")
    ax.axvspan(panel["date"].min(), event_date, alpha=0.06, color="#3498db")
    ax.axvspan(event_date, panel["date"].max(), alpha=0.06, color="#0E7490")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Yield Spread vs Germany (pp)", fontweight="bold")
    ax.legend(loc="upper left")
    _decorate_date_axis(fig, ax)
    _save_figure(fig, "figure_4_7_h2_spread_convergence.png")


def _create_figure_3_4(df: pd.DataFrame) -> None:
    croatia = df[df["country"] == "Croatia"].sort_values("date")
    germany = df[df["country"] == "Germany"].sort_values("date")
    euro_date = pd.Timestamp("2023-01-01")
    events = [
        ("2017-10-01", "Euro-Intent\nStrategy", "#4B5563", 0, 0),
        ("2019-07-05", "ERM II\nApplication", "#4B5563", 1, 0),
        ("2020-07-10", "ERM II\nEntry", "#4B5563", 0, 0),
        ("2022-06-01", "Convergence\nAssessment", "#4B5563", 2, -120),
        ("2022-07-12", "Council\nApproves Entry", "#92400E", 0, -90),
        ("2022-07-27", "ECB Hike\n(+50 bps)", "#B91C1C", 1, 0),
        ("2023-01-01", "EURO\nADOPTION", "#0E7490", 2, 60),
    ]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.2, 1], hspace=0.05)
    ax_ann = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax_ann)
    ax2 = fig.add_subplot(gs[2], sharex=ax_ann)

    ax1.plot(croatia["date"], croatia["bond_yield_10y"], linewidth=2.5, label="Croatia 10Y Yield", color="#D2691E")
    ax1.plot(germany["date"], germany["bond_yield_10y"], linewidth=1.8, alpha=0.7, linestyle="--", label="Germany 10Y Yield (Reference)", color="#374151")
    ax2.plot(croatia["date"], croatia["spread_vs_germany"], linewidth=2.5, label="Croatia Spread vs Germany", color="#D2691E")
    ax2.axhline(0, linestyle="-", linewidth=0.8, alpha=0.3, color="gray")

    for axis in (ax_ann, ax1, ax2):
        axis.axvspan(euro_date, croatia["date"].max(), alpha=0.08, color="#0E7490", zorder=0)

    for date_str, label, color, level, x_nudge in events:
        date = pd.to_datetime(date_str)
        is_euro = "EURO" in label and "ADOPTION" in label
        linewidth = 2.5 if is_euro else 1.8
        alpha = 0.85 if is_euro else 0.65
        linestyle = "-" if is_euro else "--"
        for axis in (ax_ann, ax1, ax2):
            axis.axvline(date, linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color, zorder=2)

    ax_ann.set_xlim(ax1.get_xlim())
    ax_ann.set_ylim(0, 3)
    ax_ann.axis("off")
    y_positions = {0: 0.4, 1: 1.4, 2: 2.4}

    for date_str, label, color, level, x_nudge in events:
        date = pd.to_datetime(date_str)
        label_date = date + pd.Timedelta(days=x_nudge)
        y = y_positions[level]
        is_euro = "EURO" in label and "ADOPTION" in label
        ax_ann.annotate(
            label,
            xy=(date, 0),
            xytext=(label_date, y),
            fontsize=9.5 if is_euro else 8.5,
            fontweight="bold" if is_euro else "semibold",
            ha="center",
            va="center",
            linespacing=1.1,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#DBEAFE" if is_euro else "#F9FAFB",
                edgecolor=color,
                linewidth=1.5 if is_euro else 1.0,
                alpha=0.95,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=1.0,
                shrinkA=0,
                shrinkB=2,
                connectionstyle="arc3,rad=0.15" if x_nudge != 0 else "arc3,rad=0",
            ),
        )

    ax1.set_ylabel("10-Year Bond Yield (%)", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_xlabel("Date", fontweight="bold")
    ax2.set_ylabel("Spread vs Germany (pp)", fontweight="bold")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Croatia's Euro Adoption Timeline: Yields and Spreads", fontsize=15, fontweight="bold", y=0.98)
    _save_figure(fig, "figure_3_4_croatia_euro_timeline.png")


def run(*, verbose: bool = True) -> list[Path]:
    if verbose:
        print("=" * 80)
        print("GENERATING PAPER FIGURES")
        print("=" * 80)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if verbose:
        print(f"[ok] Loaded {len(df):,} observations")

    _create_figure_3_1(df)
    _create_figure_3_3(df)
    _create_figure_3_4(df)
    _create_figure_4_1(df)
    _create_figure_4_5(df)
    _create_figure_4_7(df)

    if verbose:
        print("[ok] Paper figures generated")

    return [
        FIGURES_DIR / "figure_3_1_all_countries_yields.png",
        FIGURES_DIR / "figure_3_3_spreads_vs_germany.png",
        FIGURES_DIR / "figure_3_4_croatia_euro_timeline.png",
        FIGURES_DIR / "figure_4_1_h1_did_visual.png",
        FIGURES_DIR / "figure_4_5_h1b_did_visual.png",
        FIGURES_DIR / "figure_4_7_h2_spread_convergence.png",
    ]


if __name__ == "__main__":
    run()
