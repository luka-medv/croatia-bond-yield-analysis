"""
Generate the final paper tables referenced in the thesis document.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

from io_utils import write_text

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "DATA" / "input_data.csv"
CORE_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
ALL_COUNTRIES = ["Croatia", "France", "Germany", "Lithuania", "Slovakia", "Slovenia"]
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"


def _quantize(value: float, pattern: str) -> str:
    return str(Decimal(str(value)).quantize(Decimal(pattern), rounding=ROUND_HALF_UP))


def _to_latex(
    df: pd.DataFrame,
    filename: str,
    *,
    index: bool = False,
    column_format: str | None = None,
    formatters: dict[str, object] | None = None,
) -> None:
    latex = df.to_latex(
        index=index,
        escape=False,
        column_format=column_format,
        formatters=formatters,
    )
    latex = latex.replace("\\toprule", "\\hline")
    latex = latex.replace("\\midrule", "\\hline")
    latex = latex.replace("\\bottomrule", "\\hline")
    write_text(filename, latex)


def _build_variable_definitions_table() -> pd.DataFrame:
    rows = [
        ("Bond Yield", "10-year government bond yield (%)", "Investing.com"),
        ("Spread vs Germany", "Yield difference relative to the German Bund (percentage points)", "Authors' calculation"),
        ("Croatia Indicator", "Equals 1 for Croatia and 0 for control countries", "Authors' calculation"),
        ("Post July 2022", "Equals 1 on/after 27 July 2022, 0 beforehand", "Authors' calculation"),
        ("Post Euro Adoption", "Equals 1 on/after 1 January 2023, 0 beforehand", "Authors' calculation"),
        ("GDP Growth", "Quarterly GDP growth (%, seasonally adjusted); forward-filled from native release dates", "Eurostat"),
        ("Inflation (HICP)", "Harmonised Index of Consumer Prices, year-on-year rate (%); forward-filled from monthly releases", "Eurostat"),
        ("Public Debt", "General government gross debt (% of GDP); forward-filled from annual releases", "Eurostat"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Definition", "Source"])


def run() -> None:
    print("=" * 80)
    print("GENERATING PAPER TABLES")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"[ok] Loaded {len(df):,} observations")

    # Table 3.1
    table_3_1 = (
        df[df["country"].isin(ALL_COUNTRIES)]
        .groupby("country")
        .agg(
            **{
                "Mean Bond Yield": ("bond_yield_10y", "mean"),
                "SD Bond Yield": ("bond_yield_10y", "std"),
                "Mean GDP Growth": ("gdp_growth_quarterly", "mean"),
                "Mean Inflation": ("inflation_hicp", "mean"),
                "Mean Public Debt": ("public_debt_gdp", "mean"),
            }
        )
        .reindex(ALL_COUNTRIES)
        .reset_index()
        .rename(columns={"country": "Country"})
    )
    _to_latex(
        table_3_1,
        "table_3_1_macro_summary.tex",
        column_format="lrrrrr",
        formatters={
            "Mean Bond Yield": lambda x: _quantize(x, "0.01"),
            "SD Bond Yield": lambda x: _quantize(x, "0.01"),
            "Mean GDP Growth": lambda x: _quantize(x, "0.01"),
            "Mean Inflation": lambda x: _quantize(x, "0.01"),
            "Mean Public Debt": lambda x: _quantize(x, "0.1"),
        },
    )

    # Table 3.2
    table_3_2 = _build_variable_definitions_table()
    latex_3_2 = table_3_2.to_latex(
        index=False,
        escape=False,
        column_format="lp{9cm}l",
    )
    latex_3_2 = latex_3_2.replace("\\toprule", "\\hline")
    latex_3_2 = latex_3_2.replace("\\midrule", "\\hline")
    latex_3_2 = latex_3_2.replace("\\bottomrule", "\\hline")
    latex_3_2 += (
        "\n% Note: All constructed variables are binary indicators. "
        "Macroeconomic variables are forward-filled from their native "
        "frequencies (quarterly, monthly, annual) to daily observations.\n"
    )
    write_text("table_3_2_variable_definitions.tex", latex_3_2)

    # Table 4.2
    h1_panel = df[
        (df["country"].isin(CORE_COUNTRIES))
        & (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
    ].copy()
    h1_panel["Period"] = h1_panel["post_july_2022_hike"].map({0: "Pre-Hike", 1: "Post-Hike"})
    table_4_2 = (
        h1_panel.groupby(["country", "Period"])["bond_yield_10y"]
        .agg(Mean="mean", StdDev="std", Min="min", Max="max", N="count")
        .reset_index()
        .rename(columns={"country": "Country", "StdDev": "Std Dev"})
    )
    _to_latex(
        table_4_2,
        "table_4_2_h1_descriptive.tex",
        column_format="llrrrrr",
        formatters={
            "Mean": lambda x: f"{x:.4f}",
            "Std Dev": lambda x: f"{x:.4f}",
            "Min": lambda x: f"{x:.4f}",
            "Max": lambda x: f"{x:.4f}",
            "N": lambda x: f"{int(x):,}",
        },
    )

    # Table 4.8
    h2_panel = df[
        (df["country"].isin(CORE_COUNTRIES))
        & (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
    ].copy()
    h2_panel["Period"] = h2_panel["post_euro_adoption"].map({0: "Pre-Euro", 1: "Post-Euro"})
    table_4_8 = (
        h2_panel.groupby(["country", "Period"])["spread_vs_germany"]
        .agg(Mean="mean", StdDev="std", Min="min", Max="max", N="count")
        .reset_index()
        .rename(columns={"country": "Country", "StdDev": "Std Dev"})
    )
    _to_latex(
        table_4_8,
        "table_4_8_h2_spread_descriptive.tex",
        column_format="llrrrrr",
        formatters={
            "Mean": lambda x: f"{x:.4f}",
            "Std Dev": lambda x: f"{x:.4f}",
            "Min": lambda x: f"{x:.4f}",
            "Max": lambda x: f"{x:.4f}",
            "N": lambda x: f"{int(x):,}",
        },
    )

    print("[ok] Paper tables generated")


if __name__ == "__main__":
    run()
