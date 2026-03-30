"""
Refresh the descriptive CSV exports used to build paper tables in Word.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "DATA"
DATA_PATH = DATA_DIR / "input_data.csv"
CORE_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
ALL_COUNTRIES = ["Croatia", "France", "Germany", "Lithuania", "Slovakia", "Slovenia"]
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"


def run() -> None:
    print("=" * 80)
    print("REFRESHING DESCRIPTIVE CSV EXPORTS")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"[ok] Loaded {len(df):,} observations")

    macro = (
        df[df["country"].isin(ALL_COUNTRIES)]
        .groupby("country")
        .agg(
            Bond_Yield_Mean=("bond_yield_10y", "mean"),
            Bond_Yield_SD=("bond_yield_10y", "std"),
            GDP_Growth_Mean=("gdp_growth_quarterly", "mean"),
            Inflation_Mean=("inflation_hicp", "mean"),
            Public_Debt_Mean=("public_debt_gdp", "mean"),
        )
        .reindex(ALL_COUNTRIES)
        .reset_index()
        .rename(columns={"country": "Country"})
        .round({
            "Bond_Yield_Mean": 2,
            "Bond_Yield_SD": 2,
            "GDP_Growth_Mean": 2,
            "Inflation_Mean": 2,
            "Public_Debt_Mean": 1,
        })
    )
    macro.to_csv(DATA_DIR / "descriptive_stats_macro.csv", index=False)

    h1_panel = df[
        (df["country"].isin(CORE_COUNTRIES))
        & (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
    ].copy()
    h1_panel["Period"] = h1_panel["post_july_2022_hike"].map({0: "PreHike", 1: "PostHike"})
    h1_desc = (
        h1_panel.groupby(["country", "Period"])["bond_yield_10y"]
        .agg(Mean="mean", SD="std", Min="min", Max="max", N="count")
        .reset_index()
        .rename(columns={"country": "Country"})
        .round({"Mean": 4, "SD": 4, "Min": 4, "Max": 4})
    )
    h1_desc.to_csv(DATA_DIR / "descriptive_stats_h1_yields.csv", index=False)

    h2_panel = df[
        (df["country"].isin(CORE_COUNTRIES))
        & (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
    ].copy()
    h2_panel["Period"] = h2_panel["post_euro_adoption"].map({0: "PreEuro", 1: "PostEuro"})
    h2_desc = (
        h2_panel.groupby(["country", "Period"])["spread_vs_germany"]
        .agg(Mean="mean", SD="std", Min="min", Max="max", N="count")
        .reset_index()
        .rename(columns={"country": "Country"})
        .round({"Mean": 4, "SD": 4, "Min": 4, "Max": 4})
    )
    h2_desc.to_csv(DATA_DIR / "descriptive_stats_h2_spreads.csv", index=False)

    print("[ok] Refreshed DATA/descriptive_stats_macro.csv")
    print("[ok] Refreshed DATA/descriptive_stats_h1_yields.csv")
    print("[ok] Refreshed DATA/descriptive_stats_h2_spreads.csv")


if __name__ == "__main__":
    run()
