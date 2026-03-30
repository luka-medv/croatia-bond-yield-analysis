"""
Hypothesis 1b: Impact of ECB Monetary Policy Tightening (Feb 2, 2023) on Croatian Bond Yields

Scope (per application):
- Countries: Croatia (treatment) vs small Eurozone (Slovenia, Slovakia, Lithuania) and large EU (France, Germany)
- Method: Difference-in-Differences with macro controls and country FE
- Inference: Newey-West (HAC) standard errors (primary); HC3 also reported
- Robustness: Excluding France and Germany
- Data: Investing.com daily 10Y yields (2015-2024), Eurostat macro; Latvia excluded due to missing data
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from io_utils import save_figure, write_text
from stats_utils import build_vif_table
from plot_utils import (
    make_subplots,
    add_dual_outline,
    label_bars_with_significance,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - defensive
        pass

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'DATA' / 'input_data.csv'


def run() -> None:
    print("=" * 80)
    print("H1b: ECB Tightening on 2023-02-02 (Croatia vs SI/SK/LT + FR/DE)")
    print("Difference-in-Differences with HAC SE, VIF and robustness (exclude FR/DE)")
    print("=" * 80)

    
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    countries = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
    dfh = df[
        (df["country"].isin(countries))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()

    required_cols = {"post_feb_2023_hike", "croatia_x_post_feb2023"}
    missing = required_cols - set(dfh.columns)
    if missing:
        raise RuntimeError(f"Required indicators missing in data: {missing}")

    # Rename interaction term for table clarity
    dfh.rename(columns={'croatia_x_post_feb2023': 'croatia_ex_post'}, inplace=True)

    formula = (
        "bond_yield_10y ~ C(country) + post_feb_2023_hike + "
        "croatia_ex_post + gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
    )
    model_hac = smf.ols(formula, data=dfh).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    model_hc3 = smf.ols(formula, data=dfh).fit(cov_type="HC3")

    coef = "croatia_ex_post"
    did_hac = (
        model_hac.params.get(coef),
        model_hac.bse.get(coef),
        model_hac.pvalues.get(coef),
    )
    did_hc3 = (
        model_hc3.params.get(coef),
        model_hc3.bse.get(coef),
        model_hc3.pvalues.get(coef),
    )

    df_small = dfh[~dfh["country"].isin(["France", "Germany"])].copy()
    model_small = smf.ols(formula, data=df_small).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    r_did_hac = (
        model_small.params.get(coef),
        model_small.bse.get(coef),
        model_small.pvalues.get(coef),
    )

    vif = build_vif_table(dfh, ["gdp_growth_quarterly", "inflation_hicp", "public_debt_gdp"])

    try:
        ftest = model_hac.f_test("gdp_growth_quarterly = inflation_hicp = public_debt_gdp = 0")
        ftest_str = f"F={float(ftest.fvalue):.3f}, p={float(ftest.pvalue):.4f}"
    except Exception:
        ftest_str = "N/A"

    lines = [
        "=" * 80,
        "H1b: ECB FEB 2, 2023 RATE HIKE - DIFFERENCE-IN-DIFFERENCES RESULTS",
        "=" * 80,
        "",
        "Panel: Croatia (treatment) vs Slovenia, Slovakia, Lithuania, France, Germany",
        "Data source: Investing.com daily 10Y yields (2015-2024); Eurostat macro",
        "Latvia excluded due to incomplete data on Investing.com.",
        "",
        "MAIN SPECIFICATION (country FE + controls)",
        f"  HAC (Newey-West, maxlags=5) DiD coef: {did_hac[0]:.4f}  SE: {did_hac[1]:.4f}  p: {did_hac[2]:.4f}",
        f"  HC3 robust DiD coef: {did_hc3[0]:.4f}  SE: {did_hc3[1]:.4f}  p: {did_hc3[2]:.4f}",
        "",
        "ROBUSTNESS (exclude France and Germany)",
        f"  HAC DiD coef: {r_did_hac[0]:.4f}  SE: {r_did_hac[1]:.4f}  p: {r_did_hac[2]:.4f}",
        "",
        f"CONTROLS JOINT SIGNIFICANCE (HAC model): {ftest_str}",
        "",
        "VIF (controls)",
    ]
    for _, row in vif.iterrows():
        try:
            val = float(row["VIF"])
            lines.append(f"  {row['Variable']}: VIF={val:.2f}")
        except Exception:
            lines.append(f"  {row['Variable']}: VIF=N/A")
    lines.append("")
    write_text("h1b_regression_results.txt", "\n".join(lines) + "\n")

    fig, ax = make_subplots(figsize=(9, 6))
    specs = ["Main (HAC)", "Exclude FR/DE (HAC)"]
    coefs = [did_hac[0], r_did_hac[0]]
    pvals = [did_hac[2], r_did_hac[2]]
    bars = ax.bar(range(len(specs)), coefs, color="#0F6CE0", edgecolor="#0F6CE0", alpha=0.9, width=0.6)

    for idx, bar in enumerate(bars):
        if pvals[idx] is not None and pvals[idx] < 0.05:
            add_dual_outline(ax, bar)

    ax.axhline(0, color="black", linewidth=1.2)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":", linewidth=0.8)
    ax.set_ylabel("DiD Coefficient (percentage points)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Specification", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(specs, fontsize=11)

    label_bars_with_significance(ax, bars, pvalues=pvals)

    legend_elements = [
        Patch(facecolor="#0F6CE0", edgecolor="#0F6CE0", alpha=0.9, label="Spec (p >= 0.05)"),
        Patch(facecolor="#0F6CE0", edgecolor="#d62728", linewidth=2, alpha=0.9, label="Spec (p < 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10, framealpha=0.95)

    save_figure(fig, "h1b_robustness_checks.png", dpi=300)

    print("Saved artefacts for H1b analysis.")


if __name__ == "__main__":
    run()
