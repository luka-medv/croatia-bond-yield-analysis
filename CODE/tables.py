"""
Generate descriptive CSV exports and all textual raw outputs used for paper tables.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

import H1
import H2

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "DATA"
DATA_PATH = DATA_DIR / "input_data.csv"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "OUTPUTS" / "raw_outputs"

CORE_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
ALL_COUNTRIES = ["Croatia", "France", "Germany", "Lithuania", "Slovakia", "Slovenia"]
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"


def refresh_descriptive_exports(*, verbose: bool = True) -> list[Path]:
    if verbose:
        print("=" * 80)
        print("REFRESHING DESCRIPTIVE CSV EXPORTS")
        print("=" * 80)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if verbose:
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
        .round(
            {
                "Bond_Yield_Mean": 2,
                "Bond_Yield_SD": 2,
                "GDP_Growth_Mean": 2,
                "Inflation_Mean": 2,
                "Public_Debt_Mean": 1,
            }
        )
    )
    macro_path = DATA_DIR / "descriptive_stats_macro.csv"
    macro.to_csv(macro_path, index=False)

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
    h1_path = DATA_DIR / "descriptive_stats_h1_yields.csv"
    h1_desc.to_csv(h1_path, index=False)

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
    h2_path = DATA_DIR / "descriptive_stats_h2_spreads.csv"
    h2_desc.to_csv(h2_path, index=False)

    if verbose:
        print("[ok] Refreshed DATA/descriptive_stats_macro.csv")
        print("[ok] Refreshed DATA/descriptive_stats_h1_yields.csv")
        print("[ok] Refreshed DATA/descriptive_stats_h2_spreads.csv")

    return [macro_path, h1_path, h2_path]


def run(*, verbose: bool = True) -> None:
    sys.path.insert(0, str(ROOT))
    t0 = time.time()

    expected_files = [
        DATA_DIR / "descriptive_stats_macro.csv",
        DATA_DIR / "descriptive_stats_h1_yields.csv",
        DATA_DIR / "descriptive_stats_h2_spreads.csv",
        RAW_OUTPUTS_DIR / "h1_regression_results.txt",
        RAW_OUTPUTS_DIR / "h1_placebo_results.txt",
        RAW_OUTPUTS_DIR / "h1_hac_results.txt",
        RAW_OUTPUTS_DIR / "h1_event_study_results.txt",
        RAW_OUTPUTS_DIR / "h1b_regression_results.txt",
        RAW_OUTPUTS_DIR / "h2_regression_results.txt",
        RAW_OUTPUTS_DIR / "h2_placebo_results.txt",
        RAW_OUTPUTS_DIR / "h2_hac_results.txt",
        RAW_OUTPUTS_DIR / "h2_event_study_results.txt",
    ]

    refresh_descriptive_exports(verbose=verbose)
    H1.run(verbose=verbose)
    H2.run(verbose=verbose)

    missing = [path for path in expected_files if not path.exists()]
    print("\n" + "=" * 60)
    if missing:
        for path in missing:
            print(f"  [MISSING] {path.name}")
        print(f"  DONE: {len(expected_files) - len(missing)} verified, {len(missing)} missing  ({time.time() - t0:.1f}s)")
        raise SystemExit(1)

    print(f"  DONE: {len(expected_files)} outputs verified  ({time.time() - t0:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    run()
