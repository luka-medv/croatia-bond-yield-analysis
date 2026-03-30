

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import statsmodels.formula.api as smf

from io_utils import save_figure, write_text
from plot_utils import add_dual_outline, label_bars_with_significance

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "DATA" / "input_data.csv"


def run() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    
    countries_h1 = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
    dfh1 = df[
        (df["country"].isin(countries_h1))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    dfh1.rename(columns={"croatia_x_post_july2022": "croatia_ex_post"}, inplace=True)

    h1_placebo_dates = [
        ("2021-04-27", 15, "Placebo 1"),
        ("2021-07-27", 12, "Placebo 2"),
        ("2021-10-27", 9, "Placebo 3"),
        ("2022-01-27", 6, "Placebo 4"),
        ("2022-04-27", 3, "Placebo 5"),
    ]

    print("=" * 70)
    print("H1 PLACEBO TESTS — Column C (controlling for actual treatment)")
    print("=" * 70)

    h1_results = []
    for i, (date_str, months_before, name) in enumerate(h1_placebo_dates):
        pdate = pd.Timestamp(date_str)
        col_post = f"post_placebo_{i + 1}"
        col_inter = f"croatia_x_placebo_{i + 1}"
        dfh1[col_post] = (dfh1["date"] >= pdate).astype(int)
        dfh1[col_inter] = dfh1["is_croatia"] * dfh1[col_post]

        formula = (
            
            
        )

        model = smf.ols(formula, data=dfh1).fit(cov_type="HC3")
        coef = model.params[col_inter]
        se = model.bse[col_inter]
        pval = model.pvalues[col_inter]

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name} ({date_str}, -{months_before}mo): {coef:+.4f}  SE={se:.4f}  p={pval:.4f} {sig}")

        h1_results.append({
            : name,
            : date_str,
            : months_before,
            : coef,
            : se,
            : pval,
        })

    
    countries_h2 = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
    dfh2 = df[
        (df["country"].isin(countries_h2))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    dfh2.rename(columns={"croatia_x_post_euro": "croatia_ex_post"}, inplace=True)

    h2_placebo_dates = [
        ("2021-01-01", 24, "Placebo 1"),
        ("2021-07-01", 18, "Placebo 2"),
        ("2022-01-01", 12, "Placebo 3"),
        ("2022-07-01", 6, "Placebo 4"),
        ("2022-10-01", 3, "Placebo 5"),
    ]

    print()
    print("=" * 70)
    print("H2 PLACEBO TESTS — Column C (controlling for actual treatment)")
    print("=" * 70)

    h2_results = []
    for i, (date_str, months_before, name) in enumerate(h2_placebo_dates):
        pdate = pd.Timestamp(date_str)
        col_post = f"post_placebo_{i + 1}"
        col_inter = f"croatia_x_placebo_{i + 1}"
        dfh2[col_post] = (dfh2["date"] >= pdate).astype(int)
        dfh2[col_inter] = dfh2["is_croatia"] * dfh2[col_post]

        formula = (
            
            
        )

        model = smf.ols(formula, data=dfh2).fit(cov_type="HC3")
        coef = model.params[col_inter]
        se = model.bse[col_inter]
        pval = model.pvalues[col_inter]

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name} ({date_str}, -{months_before}mo): {coef:+.4f}  SE={se:.4f}  p={pval:.4f} {sig}")

        h2_results.append({
            : name,
            : date_str,
            : months_before,
            : coef,
            : se,
            : pval,
        })

    
    lines = [
         * 70,
        ,
        ,
        ,
        ,
         * 70,
        ,
        ,
        ,
        ,
        ,
    ]
    for r in h1_results:
        sig = "***" if r["pvalue"] < 0.001 else "**" if r["pvalue"] < 0.01 else "*" if r["pvalue"] < 0.05 else ""
        lines.append(f"  {r['name']} ({r['date']}, -{r['months_before']}mo): "
                      )

    lines += [
        ,
        ,
        ,
        ,
        ,
    ]
    for r in h2_results:
        sig = "***" if r["pvalue"] < 0.001 else "**" if r["pvalue"] < 0.01 else "*" if r["pvalue"] < 0.05 else ""
        lines.append(f"  {r['name']} ({r['date']}, -{r['months_before']}mo): "
                      )

    lines.append("")
    write_text("placebo_column_c_results.txt", "\n".join(lines) + "\n")
    print("\nSaved: placebo_column_c_results.txt")

    
    h1_main = smf.ols(
        
        ,
        data=dfh1,
    ).fit(cov_type="HC3")
    h1_did_coef = h1_main.params["croatia_ex_post"]
    h1_did_pval = h1_main.pvalues["croatia_ex_post"]

    h2_main = smf.ols(
        
        ,
        data=dfh2,
    ).fit(cov_type="HC3")
    h2_did_coef = h2_main.params["croatia_ex_post"]
    h2_did_pval = h2_main.pvalues["croatia_ex_post"]

    
    def _plot_placebo(results, did_coef, did_pval, actual_label, legend_actual, filename):
        fig, ax = plt.subplots(figsize=(10, 6))
        viz = []
        for r in results:
            viz.append({
                : f"{r['name']}\n({r['date']})\n-{r['months_before']}mo",
                : r["coefficient"],
                : r["pvalue"],
                : "Placebo",
            })
        viz.append({
            : actual_label,
            : did_coef,
            : did_pval,
            : "Actual",
        })
        pdf = pd.DataFrame(viz)

        bars = ax.bar(
            range(len(pdf)), pdf["coef"],
            color="#0F6CE0", edgecolor="#0F6CE0", alpha=0.9, width=0.7,
        )
        for idx, row in pdf.iterrows():
            bar = bars[idx]
            if row["Type"] == "Actual":
                bar.set_edgecolor("white")
                bar.set_linewidth(2)
            if row["pval"] is not None and row["pval"] < 0.05:
                add_dual_outline(ax, bar)

        ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
        ax.grid(True, alpha=0.3, axis="y", linestyle=":", linewidth=0.8)
        ax.set_ylabel("DiD Coefficient (percentage points)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Timeline (Earlier <- -> Later)", fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(pdf)))
        ax.set_xticklabels(pdf["Test"], fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        label_bars_with_significance(ax, bars, pvalues=pdf["pval"].tolist())

        legend_elements = [
            Patch(facecolor="#0F6CE0", edgecolor="#0F6CE0", alpha=0.9, label="Placebo (p >= 0.05)"),
            Patch(facecolor="#0F6CE0", edgecolor="white", linewidth=2, alpha=0.9, label=legend_actual),
            Patch(facecolor="#0F6CE0", edgecolor="#d62728", linewidth=2, alpha=0.9, label="Significant Placebo (p < 0.05)"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=11, framealpha=0.95)
        plt.tight_layout()
        save_figure(fig, filename, dpi=300)

    _plot_placebo(
        h1_results, h1_did_coef, h1_did_pval,
        , "Actual ECB Rate Hike",
        ,
    )
    _plot_placebo(
        h2_results, h2_did_coef, h2_did_pval,
        , "Actual Euro Adoption",
        ,
    )


if __name__ == "__main__":
    run()
