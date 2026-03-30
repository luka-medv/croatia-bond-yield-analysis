"""
Generate the paper's descriptive LaTeX tables from the verified input panel.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

from io_utils import write_text

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "DATA" / "input_data.csv"
CORE_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"


def export_table(df: pd.DataFrame, filename: str, *, index: bool = False, booktabs: bool = False, **kwargs) -> None:
    latex = df.to_latex(index=index, escape=False, **kwargs)
    if not booktabs:
        latex = latex.replace("\\toprule", "\\hline")
        latex = latex.replace("\\midrule", "\\hline")
        latex = latex.replace("\\bottomrule", "\\hline")
    write_text(filename, latex)


def _format_two_decimals(value: float) -> str:
    return f"{Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"


def run() -> None:
    print("=" * 80)
    print("GENERATING DESCRIPTIVE TABLES")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"[ok] Loaded {len(df):,} observations")

    df_h1 = df[
        (df["country"].isin(CORE_COUNTRIES)) &
        (df["date"] >= START_DATE) &
        (df["date"] <= END_DATE)
    ].copy()
    df_h1["period"] = df_h1["post_july_2022_hike"].map({0: "Pre-Hike", 1: "Post-Hike"})

    df_h2 = df[
        (df["country"].isin(CORE_COUNTRIES)) &
        (df["date"] >= START_DATE) &
        (df["date"] <= END_DATE)
    ].copy()
    df_h2["period"] = df_h2["post_euro_adoption"].map({0: "Pre-Euro", 1: "Post-Euro"})

    coverage_rows = []
    for country in sorted(df["country"].unique()):
        sub = df[df["country"] == country]
        total = len(sub)
        missing = int(sub["bond_yield_10y"].isna().sum())
        if total > 0:
            missing_pct = (missing / total) * 100
            coverage_pct = 100 - missing_pct
            date_min = sub["date"].min()
            date_max = sub["date"].max()
            sample_range = f"{date_min.date()}--{date_max.date()}"
        else:
            missing_pct = 0.0
            coverage_pct = 0.0
            sample_range = ""
        coverage_rows.append({
            "Country": country,
            "Observations": total,
            "Missing (\\%)": missing_pct,
            "Coverage (\\%)": coverage_pct,
            "Sample Range": sample_range,
        })

    coverage_df = pd.DataFrame(coverage_rows)
    export_table(
        coverage_df,
        "data_coverage_table.tex",
        index=False,
        column_format="lrrrl",
        formatters={
            "Observations": lambda x: f"{int(x):,}",
            "Missing (\\%)": lambda x: f"{x:.1f}~\\%",
            "Coverage (\\%)": lambda x: f"{x:.1f}~\\%",
        },
        booktabs=True,
    )

    descriptive_stats_df = (
        df.groupby("country")["bond_yield_10y"]
        .agg(
            Mean="mean",
            Median="median",
            **{"Std. Dev.": "std"},
            Min="min",
            Max="max",
        )
        .reset_index()
        .rename(columns={"country": "Country"})
        .sort_values("Country")
    )
    export_table(
        descriptive_stats_df,
        "descriptive_statistics_table.tex",
        index=False,
        column_format="lrrrrr",
        formatters={
            "Mean": _format_two_decimals,
            "Median": _format_two_decimals,
            "Std. Dev.": _format_two_decimals,
            "Min": _format_two_decimals,
            "Max": _format_two_decimals,
        },
        booktabs=False,
    )

    h1_stats_df = (
        df_h1.groupby(["country", "period"])["bond_yield_10y"]
        .agg(Mean="mean", StdDev="std", Min="min", Max="max", N="count")
        .reset_index()
        .rename(columns={"country": "Country", "period": "Period", "StdDev": "Std Dev"})
    )
    export_table(
        h1_stats_df,
        "h1_descriptive_statistics.tex",
        index=False,
        column_format="llrrrrr",
        formatters={
            "Mean": lambda x: f"{x:.4f}",
            "Std Dev": lambda x: f"{x:.4f}",
            "Min": lambda x: f"{x:.4f}",
            "Max": lambda x: f"{x:.4f}",
            "N": lambda x: f"{int(x):,}",
        },
        booktabs=False,
    )

    h2_stats_df = (
        df_h2.groupby(["country", "period"])["spread_vs_germany"]
        .agg(Mean="mean", StdDev="std", Min="min", Max="max", N="count")
        .reset_index()
        .rename(columns={"country": "Country", "period": "Period", "StdDev": "Std Dev"})
    )
    export_table(
        h2_stats_df,
        "h2_spread_statistics.tex",
        index=False,
        column_format="llrrrrr",
        formatters={
            "Mean": lambda x: f"{x:.4f}",
            "Std Dev": lambda x: f"{x:.4f}",
            "Min": lambda x: f"{x:.4f}",
            "Max": lambda x: f"{x:.4f}",
            "N": lambda x: f"{int(x):,}",
        },
        booktabs=False,
    )

    macro_summary_df = (
        df.groupby("country")
        .agg(
            **{
                "Bond Yield Mean": ("bond_yield_10y", "mean"),
                "Bond Yield SD": ("bond_yield_10y", "std"),
                "GDP Growth Mean": ("gdp_growth_quarterly", "mean"),
                "Inflation Mean": ("inflation_hicp", "mean"),
                "Public Debt Mean": ("public_debt_gdp", "mean"),
            }
        )
        .reset_index()
        .rename(columns={"country": "Country"})
        .fillna(0.0)
    )
    export_table(
        macro_summary_df,
        "macro_summary_table.tex",
        index=False,
        column_format="lrrrrr",
        formatters={
            "Bond Yield Mean": lambda x: f"{x:.2f}",
            "Bond Yield SD": lambda x: f"{x:.2f}",
            "GDP Growth Mean": lambda x: f"{x:.2f}",
            "Inflation Mean": lambda x: f"{x:.2f}",
            "Public Debt Mean": lambda x: f"{x:.1f}",
        },
        booktabs=False,
    )

    print("[ok] Descriptive tables generated")


if __name__ == "__main__":
    run()
