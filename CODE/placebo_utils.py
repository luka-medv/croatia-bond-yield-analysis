"""
Shared helpers for Column C placebo estimation, reporting, and plotting.
"""

from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.patches import Patch

from io_utils import save_figure, write_text
from plot_utils import add_dual_outline, label_bars_with_significance


def sig_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def compute_placebo_payload(
    panel: pd.DataFrame,
    placebo_tests: Sequence[tuple[str, int, str]],
    *,
    placebo_formula_builder: Callable[[int, str, str], str],
    main_formula: str,
    effect_name: str = "croatia_ex_post",
    cov_type: str = "HC3",
) -> dict:
    work = panel.copy()
    results = []

    for idx, (date_str, months_before, name) in enumerate(placebo_tests, start=1):
        placebo_date = pd.Timestamp(date_str)
        col_post = f"post_placebo_{idx}"
        col_inter = f"croatia_x_placebo_{idx}"
        work[col_post] = (work["date"] >= placebo_date).astype(int)
        work[col_inter] = work["is_croatia"] * work[col_post]

        model = smf.ols(
            placebo_formula_builder(idx, col_post, col_inter),
            data=work,
        ).fit(cov_type=cov_type)

        results.append({
            "name": name,
            "date": date_str,
            "months_before": months_before,
            "coefficient": model.params[col_inter],
            "se": model.bse[col_inter],
            "pvalue": model.pvalues[col_inter],
        })

    main_model = smf.ols(main_formula, data=work).fit(cov_type=cov_type)

    return {
        "results": results,
        "main_coef": main_model.params[effect_name],
        "main_se": main_model.bse[effect_name],
        "main_pval": main_model.pvalues[effect_name],
        "sample_n": len(work),
        "countries": sorted(work["country"].unique()),
    }


def build_standalone_placebo_report(
    *,
    header_title: str,
    section_title: str,
    sample_start: str,
    sample_end: str,
    payload: dict,
    main_label: str,
) -> str:
    lines = [
        "=" * 70,
        header_title,
        "Each regression includes the actual treatment indicator,",
        "so that placebo coefficients are net of the real treatment effect.",
        "HC3 robust standard errors throughout.",
        "=" * 70,
        "",
        section_title,
        f"  Panel: {', '.join(payload['countries'])}",
        f"  Sample: {sample_start} to {sample_end}  N={payload['sample_n']}",
        "",
    ]

    for result in payload["results"]:
        lines.append(
            f"  {result['name']} ({result['date']}, -{result['months_before']}mo): "
            f"coef={result['coefficient']:+.4f}  SE={result['se']:.4f}  "
            f"p={result['pvalue']:.4f} {sig_stars(result['pvalue'])}"
        )

    lines.extend([
        "",
        main_label,
        f"  coef={payload['main_coef']:+.4f}  SE={payload['main_se']:.4f}  p={payload['main_pval']:.4f}",
        "",
    ])
    return "\n".join(lines) + "\n"


def write_standalone_placebo_report(
    filename: str,
    *,
    header_title: str,
    section_title: str,
    sample_start: str,
    sample_end: str,
    payload: dict,
    main_label: str,
) -> None:
    write_text(
        filename,
        build_standalone_placebo_report(
            header_title=header_title,
            section_title=section_title,
            sample_start=sample_start,
            sample_end=sample_end,
            payload=payload,
            main_label=main_label,
        ),
    )


def save_placebo_plot(
    payload: dict,
    filename: str,
    *,
    actual_label: str,
    legend_actual: str,
) -> None:
    rows = []
    for result in payload["results"]:
        rows.append({
            "Test": f"{result['name']}\n({result['date']})\n-{result['months_before']}mo",
            "coef": result["coefficient"],
            "pval": result["pvalue"],
            "Type": "Placebo",
        })
    rows.append({
        "Test": actual_label,
        "coef": payload["main_coef"],
        "pval": payload["main_pval"],
        "Type": "Actual",
    })

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(plot_df)),
        plot_df["coef"],
        color="#0F6CE0",
        edgecolor="#0F6CE0",
        alpha=0.9,
        width=0.7,
    )

    for idx, row in plot_df.iterrows():
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
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["Test"], fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    label_bars_with_significance(ax, bars, pvalues=plot_df["pval"].tolist())

    legend_elements = [
        Patch(facecolor="#0F6CE0", edgecolor="#0F6CE0", alpha=0.9, label="Placebo (p >= 0.05)"),
        Patch(facecolor="#0F6CE0", edgecolor="white", linewidth=2, alpha=0.9, label=legend_actual),
        Patch(facecolor="#0F6CE0", edgecolor="#d62728", linewidth=2, alpha=0.9, label="Significant Placebo (p < 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=11, framealpha=0.95)
    plt.tight_layout()
    save_figure(fig, filename, dpi=300)


def extract_report_body(text: str) -> list[str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("H1:") or stripped.startswith("H2:"):
            return lines[idx:]
    return lines
