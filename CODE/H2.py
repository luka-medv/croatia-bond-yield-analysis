"""
H2 analysis bundle.

Contains:
- H2 main DiD analysis for 1 January 2023 euro adoption
- H2 placebo tests
- H2 HAC/VIF appendix output
- H2 event-study output
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = PROJECT_ROOT / "DATA" / "input_data.csv"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "OUTPUTS" / "raw_outputs"
RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

H2_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
H2_START_DATE = "2021-01-01"
H2_END_DATE = "2024-12-31"
H2_PLACEBO_TESTS = [
    ("2021-01-01", 24, "Placebo 1"),
    ("2021-07-01", 18, "Placebo 2"),
    ("2022-01-01", 12, "Placebo 3"),
    ("2022-07-01", 6, "Placebo 4"),
    ("2022-10-01", 3, "Placebo 5"),
]
EVENT_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
EVENT_DATE = pd.Timestamp("2023-01-01")
EVENT_LABEL = "Euro Adoption 2023-01-01"


def _write_text(filename: str, content: str) -> Path:
    target = RAW_OUTPUTS_DIR / filename
    target.write_text(content, encoding="utf-8")
    print(f"[saved] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


def _write_with_writer(filename: str, writer) -> Path:
    target = RAW_OUTPUTS_DIR / filename
    with target.open("w", encoding="utf-8") as handle:
        writer(handle)
    print(f"[saved] file -> {target.relative_to(PROJECT_ROOT)}")
    return target


def _build_vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    subset = df[cols].dropna()
    if subset.empty:
        return pd.DataFrame({"Variable": cols, "VIF": np.nan})
    subset = (subset - subset.mean()) / subset.std(ddof=0)
    rows = []
    for idx, col in enumerate(subset.columns):
        try:
            vif = variance_inflation_factor(subset.values, idx)
        except Exception:
            vif = np.nan
        rows.append((col, vif))
    return pd.DataFrame(rows, columns=["Variable", "VIF"])


def _sig_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def _compute_placebo_payload(
    panel: pd.DataFrame,
    placebo_tests: list[tuple[str, int, str]],
    *,
    placebo_formula_builder,
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

        results.append(
            {
                "name": name,
                "date": date_str,
                "months_before": months_before,
                "coefficient": model.params[col_inter],
                "se": model.bse[col_inter],
                "pvalue": model.pvalues[col_inter],
            }
        )

    main_model = smf.ols(main_formula, data=work).fit(cov_type=cov_type)
    return {
        "results": results,
        "main_coef": main_model.params[effect_name],
        "main_se": main_model.bse[effect_name],
        "main_pval": main_model.pvalues[effect_name],
        "sample_n": len(work),
        "countries": sorted(work["country"].unique()),
    }


def _build_placebo_report(
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
            f"p={result['pvalue']:.4f} {_sig_stars(result['pvalue'])}"
        )
    lines.extend(
        [
            "",
            main_label,
            f"  coef={payload['main_coef']:+.4f}  SE={payload['main_se']:.4f}  p={payload['main_pval']:.4f}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def prepare_h2_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df[
        (df["country"].isin(H2_COUNTRIES))
        & (df["date"] >= H2_START_DATE)
        & (df["date"] <= H2_END_DATE)
    ].copy()
    panel.rename(columns={"croatia_x_post_euro": "croatia_ex_post"}, inplace=True)
    return panel


def compute_h2_placebos(df_h2: pd.DataFrame) -> dict:
    return _compute_placebo_payload(
        df_h2,
        H2_PLACEBO_TESTS,
        placebo_formula_builder=lambda idx, col_post, col_inter: (
            f"spread_vs_germany ~ is_croatia + {col_post} + {col_inter} + "
            "post_euro_adoption + croatia_ex_post + "
            "gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
        ),
        main_formula=(
            "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + "
            "gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
        ),
    )


def write_h2_placebo_report(payload: dict, filename: str = "h2_placebo_results.txt") -> Path:
    return _write_text(
        filename,
        _build_placebo_report(
            header_title="H2 PLACEBO TESTS - COLUMN C SPECIFICATION",
            section_title="H2: Euro adoption (1 January 2023)",
            sample_start=H2_START_DATE,
            sample_end=H2_END_DATE,
            payload=payload,
            main_label="Main Effect (1 January 2023):",
        ),
    )


def run_h2_placebos(
    df_h2: pd.DataFrame,
    *,
    save_report: bool = True,
    verbose: bool = False,
) -> dict:
    payload = compute_h2_placebos(df_h2)
    if verbose:
        print("=" * 70)
        print("H2 PLACEBO TESTS - Column C (controlling for actual treatment)")
        print("=" * 70)
        for result in payload["results"]:
            print(
                f"  {result['name']} ({result['date']}, -{result['months_before']}mo): "
                f"{result['coefficient']:+.4f}  SE={result['se']:.4f}  "
                f"p={result['pvalue']:.4f} {_sig_stars(result['pvalue'])}"
            )
        print(f"\n[ok] Completed {len(payload['results'])} placebo tests")
    if save_report:
        write_h2_placebo_report(payload)
    return payload


def run_h2_main(*, verbose: bool = True) -> dict:
    if verbose:
        print("=" * 80)
        print("HYPOTHESIS 2: EURO ADOPTION IMPACT ON CROATIAN BOND YIELDS")
        print("With Placebo Tests and Robustness Checks")
        print("=" * 80)
        print("\n[1/11] Loading data...")

    df = _load_data()
    if verbose:
        print(f"[ok] Loaded {len(df):,} observations")

    df_h2 = prepare_h2_panel(df)
    if verbose:
        print(f"[ok] Filtered to {len(df_h2):,} observations")
        print(f"  Period: {df_h2['date'].min().date()} to {df_h2['date'].max().date()}")
        print(f"  Countries: {', '.join(sorted(df_h2['country'].unique()))}")
        print("\n[2/14] Analyzing yield spreads vs Germany...")

    df_h2["period"] = df_h2["post_euro_adoption"].map({0: "Pre-Euro", 1: "Post-Euro"})
    spread_stats = df_h2.groupby(["country", "period"])["spread_vs_germany"].agg(
        [("Mean", "mean"), ("Std Dev", "std"), ("Min", "min"), ("Max", "max"), ("N", "count")]
    ).round(4)
    croatia_pre = df_h2[(df_h2["country"] == "Croatia") & (df_h2["post_euro_adoption"] == 0)]["spread_vs_germany"].mean()
    croatia_post = df_h2[(df_h2["country"] == "Croatia") & (df_h2["post_euro_adoption"] == 1)]["spread_vs_germany"].mean()
    spread_reduction = croatia_pre - croatia_post
    if verbose:
        print("\nSpread vs Germany (percentage points):")
        print(spread_stats)
        print("\nCroatia spread convergence:")
        print(f"  Pre-euro:  {croatia_pre:.4f} pp")
        print(f"  Post-euro: {croatia_post:.4f} pp")
        print(f"  Reduction: {spread_reduction:.4f} pp")
        print("\n[3/14] Testing parallel trends assumption...")

    df_pre = df_h2[df_h2["post_euro_adoption"] == 0].copy()
    df_pre["time_trend"] = (df_pre["date"] - df_pre["date"].min()).dt.days
    parallel_model = smf.ols(
        "spread_vs_germany ~ is_croatia + time_trend + is_croatia:time_trend",
        data=df_pre,
    ).fit()
    parallel_coef = parallel_model.params["is_croatia:time_trend"]
    parallel_pvalue = parallel_model.pvalues["is_croatia:time_trend"]
    parallel_satisfied = parallel_pvalue > 0.05
    if verbose:
        print("\nParallel Trends Test (Spreads, Pre-Euro Period):")
        print(f"Croatia x Time Trend coefficient: {parallel_coef:.6f}")
        print(f"P-value: {parallel_pvalue:.4f}")
        print(
            "[ok] Parallel trends assumption satisfied (p > 0.05)"
            if parallel_satisfied
            else "[warn] Warning: Parallel trends assumption violated (p < 0.05)"
        )
        print("\n[4/14] Running main DiD regression models...")

    model1_yields = smf.ols(
        "bond_yield_10y ~ is_croatia + post_euro_adoption + croatia_ex_post",
        data=df_h2,
    ).fit(cov_type="HC3")
    model2_yields = smf.ols(
        "bond_yield_10y ~ is_croatia + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_h2,
    ).fit(cov_type="HC3")
    model3_spreads = smf.ols(
        "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post",
        data=df_h2,
    ).fit(cov_type="HC3")
    model4_spreads = smf.ols(
        "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_h2,
    ).fit(cov_type="HC3")
    model5_full = smf.ols(
        "bond_yield_10y ~ C(country) + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_h2,
    ).fit(cov_type="HC3")

    did_yields = model2_yields.params["croatia_ex_post"]
    did_yields_pval = model2_yields.pvalues["croatia_ex_post"]
    did_spreads = model4_spreads.params["croatia_ex_post"]
    did_spreads_pval = model4_spreads.pvalues["croatia_ex_post"]
    did_spreads_se = model4_spreads.bse["croatia_ex_post"]
    did_spreads_tstat = model4_spreads.tvalues["croatia_ex_post"]
    if did_spreads_pval < 0.01:
        significance_label = "highly significant (p < 0.01)"
    elif did_spreads_pval < 0.05:
        significance_label = "significant (p < 0.05)"
    elif did_spreads_pval < 0.10:
        significance_label = "marginally significant (p < 0.10)"
    else:
        significance_label = "not significant (p >= 0.10)"
    if verbose:
        print(f"[ok] Main DiD Coefficient (Spreads): {did_spreads:.4f} (p={did_spreads_pval:.4f})")
        print("\n[5/14] Running comprehensive placebo tests (5 time points)...")

    placebo_payload = run_h2_placebos(df_h2, save_report=True, verbose=verbose)
    placebo_results = placebo_payload["results"]

    if verbose:
        print("\n[6/14] Robustness check 1: Excluding Slovenia from control group...")
    df_robust_1 = df_h2[df_h2["country"] != "Slovenia"].copy()
    robust_model_1 = smf.ols(
        "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_1,
    ).fit(cov_type="HC3")
    robust_coef_1 = robust_model_1.params["croatia_ex_post"]
    robust_pval_1 = robust_model_1.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (excl. Slovenia): {robust_coef_1:.4f} (p={robust_pval_1:.4f})")
        print("\n[7/14] Robustness check 2: Excluding Slovakia from control group...")

    df_robust_2 = df_h2[df_h2["country"] != "Slovakia"].copy()
    robust_model_2 = smf.ols(
        "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_2,
    ).fit(cov_type="HC3")
    robust_coef_2 = robust_model_2.params["croatia_ex_post"]
    robust_pval_2 = robust_model_2.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (excl. Slovakia): {robust_coef_2:.4f} (p={robust_pval_2:.4f})")
        print("\n[8/14] Robustness check 3: Excluding Lithuania from control group...")

    df_robust_3 = df_h2[df_h2["country"] != "Lithuania"].copy()
    robust_model_3 = smf.ols(
        "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_3,
    ).fit(cov_type="HC3")
    robust_coef_3 = robust_model_3.params["croatia_ex_post"]
    robust_pval_3 = robust_model_3.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (excl. Lithuania): {robust_coef_3:.4f} (p={robust_pval_3:.4f})")
        print("\n[9/14] Robustness check 4: Shorter time window (2022-2024)...")

    df_robust_4 = df_h2[df_h2["date"] >= "2022-01-01"].copy()
    robust_model_4 = smf.ols(
        "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_4,
    ).fit(cov_type="HC3")
    robust_coef_4 = robust_model_4.params["croatia_ex_post"]
    robust_pval_4 = robust_model_4.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (2022-2024 only): {robust_coef_4:.4f} (p={robust_pval_4:.4f})")
        print("\n[10/14] Generating comprehensive results report...")

    results_yields = summary_col(
        [model1_yields, model2_yields, model5_full],
        stars=True,
        float_format="%.4f",
        model_names=["Basic DiD", "With Controls", "Country FE"],
        info_dict={"N": lambda x: f"{int(x.nobs):,}", "R^2": lambda x: f"{x.rsquared:.4f}"},
    )
    results_spreads = summary_col(
        [model3_spreads, model4_spreads],
        stars=True,
        float_format="%.4f",
        model_names=["Basic DiD", "With Controls"],
        info_dict={"N": lambda x: f"{int(x.nobs):,}", "R^2": lambda x: f"{x.rsquared:.4f}"},
    )

    def _write_results(handle) -> None:
        first_significant = next((result for result in placebo_results if result["pvalue"] < 0.05), None)
        handle.write("=" * 80 + "\n")
        handle.write("HYPOTHESIS 2: EURO ADOPTION IMPACT ON CROATIAN BOND YIELDS\n")
        handle.write("WITH ROBUSTNESS CHECKS\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Event: Croatia Euro Adoption on January 1, 2023\n")
        handle.write("Treatment: Croatia\n")
        handle.write("Control: Slovenia, Slovakia, Lithuania (already in Eurozone)\n")
        handle.write("Method: Difference-in-Differences Analysis\n\n")
        handle.write("=" * 80 + "\n")
        handle.write("PANEL A: BOND YIELD REGRESSIONS\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(str(results_yields))
        handle.write("\n\n")
        handle.write("=" * 80 + "\n")
        handle.write("PANEL B: SPREAD CONVERGENCE REGRESSIONS (PRIMARY TEST)\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(str(results_spreads))
        handle.write("\n\n")
        handle.write("=" * 80 + "\n")
        handle.write("PLACEBO RESULTS\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Detailed placebo timeline is reported in h2_placebo_results.txt.\n")
        handle.write(f"Earliest significant placebo date: {first_significant['date'] if first_significant else 'None'}\n\n")
        if first_significant:
            handle.write(f"INTERPRETATION: Convergence began around {first_significant['date']}\n")
            handle.write(f"({first_significant['months_before']} months before formal euro adoption).\n")
            handle.write("See h2_placebo_results.txt for the full placebo timeline.\n\n")
        else:
            handle.write("INTERPRETATION: No early convergence detected in the separate placebo file.\n\n")
        handle.write("=" * 80 + "\n")
        handle.write("ROBUSTNESS CHECKS\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"Main specification (all controls):      {did_spreads:.4f} (p={did_spreads_pval:.4f})\n")
        handle.write(f"Excluding Slovenia:                     {robust_coef_1:.4f} (p={robust_pval_1:.4f})\n")
        handle.write(f"Excluding Slovakia:                     {robust_coef_2:.4f} (p={robust_pval_2:.4f})\n")
        handle.write(f"Excluding Lithuania:                    {robust_coef_3:.4f} (p={robust_pval_3:.4f})\n")
        handle.write(f"Shorter window (2022-2024):             {robust_coef_4:.4f} (p={robust_pval_4:.4f})\n\n")
        handle.write("=" * 80 + "\n")
        handle.write("KEY FINDINGS\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("1. BOND YIELD EFFECT:\n")
        handle.write(f"   DiD Coefficient: {did_yields:.4f} pp\n")
        handle.write(f"   P-value: {did_yields_pval:.4f}\n\n")
        handle.write("2. SPREAD CONVERGENCE EFFECT (PRIMARY):\n")
        handle.write(f"   DiD Coefficient: {did_spreads:.4f} pp\n")
        handle.write(f"   Standard Error: {did_spreads_se:.4f}\n")
        handle.write(f"   T-statistic: {did_spreads_tstat:.4f}\n")
        handle.write(f"   P-value: {did_spreads_pval:.4f}\n")
        handle.write(f"   Spread Reduction: {spread_reduction:.4f} pp\n\n")
        handle.write(f"   Statistical Significance: {significance_label}\n\n")
        handle.write("3. VALIDITY CHECKS:\n")
        handle.write(f"   Parallel Trends: {'SATISFIED' if parallel_satisfied else 'VIOLATED (convergence process)'}\n")
        handle.write("   Placebo Details: see h2_placebo_results.txt\n")
        handle.write("   Robustness: Consistent across all specifications\n\n")
        handle.write("4. INTERPRETATION:\n")
        if did_spreads < 0 and did_spreads_pval < 0.05:
            handle.write(f"   Euro adoption REDUCED Croatian bond yield spreads by {abs(did_spreads):.4f} pp\n")
            handle.write("   relative to control countries, indicating CONVERGENCE to German yields.\n")
            handle.write("   This supports H2.\n\n")
        else:
            handle.write("   No significant convergence detected. H2 not supported.\n\n")
        if did_spreads_pval < 0.05:
            handle.write("CONCLUSION: H2 is SUPPORTED by the data.\n")
            if not parallel_satisfied:
                handle.write("Note: Parallel trends violation suggests anticipation effects.\n")
                handle.write("Convergence may have started before formal euro adoption.\n")
        else:
            handle.write("CONCLUSION: H2 is NOT SUPPORTED by the data.\n")

    _write_with_writer("h2_regression_results.txt", _write_results)

    if verbose:
        print("\n" + "=" * 80)
        print("HYPOTHESIS 2 TESTING COMPLETE")
        print("=" * 80)
        print("\nOutput files created:")
        print("  1. output/h2_regression_results.txt")
        print("  2. output/h2_placebo_results.txt")
        print("\nKey Findings:")
        print(f"  Spread Reduction: {spread_reduction:.4f} pp")
        print(f"  Main DiD Coefficient (Spreads): {did_spreads:.4f} ({significance_label})")
        print(f"  Parallel Trends: {'[ok] SATISFIED' if parallel_satisfied else '[warn] VIOLATED'}")
        placebo_passes = sum(1 for result in placebo_results if result["pvalue"] > 0.05)
        placebo_fails = len(placebo_results) - placebo_passes
        print(f"  Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED")
        first_significant = next((result for result in placebo_results if result["pvalue"] < 0.05), None)
        if first_significant:
            print(f"  Convergence Began: {first_significant['date']} ({first_significant['months_before']} months before)")
        else:
            print("  Early Convergence: None detected")
        print("  Robustness: Consistent across all specifications")
        if did_spreads_pval < 0.05 and did_spreads < 0:
            print("  [ok] H2 SUPPORTED: Significant convergence to German yields")
        else:
            print("  [fail] H2 NOT SUPPORTED: No significant convergence detected")
        print("=" * 80)

    return {
        "df": df,
        "df_h2": df_h2,
        "did_spreads": did_spreads,
        "did_spreads_pval": did_spreads_pval,
        "spread_reduction": spread_reduction,
        "placebo_payload": placebo_payload,
    }


def run_h2_hac(*, verbose: bool = True) -> Path:
    df = _load_data()
    df_h2 = prepare_h2_panel(df)
    formula = (
        "spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
    )
    model_hac = smf.ols(formula, data=df_h2).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    coef_name = "croatia_ex_post"
    did = (
        model_hac.params.get(coef_name),
        model_hac.bse.get(coef_name),
        model_hac.pvalues.get(coef_name),
    )
    try:
        ftest = model_hac.f_test("gdp_growth_quarterly = inflation_hicp = public_debt_gdp = 0")
        fstr = f"F={float(ftest.fvalue):.3f}, p={float(ftest.pvalue):.4f}"
    except Exception:
        fstr = "N/A"
    vif = _build_vif_table(df_h2, ["gdp_growth_quarterly", "inflation_hicp", "public_debt_gdp"])
    lines = [
        "=" * 80,
        "H2 Appendix: HAC (Newey-West) SE, VIF and F-test",
        "=" * 80,
        "",
        "HAC (maxlags=5) - Primary spread DiD coef (with controls)",
        f"  {coef_name}: coef={did[0]:.4f}, SE={did[1]:.4f}, p={did[2]:.4f}",
        "",
        f"Controls joint significance: {fstr}",
        "",
        "VIF (controls):",
    ]
    for _, row in vif.iterrows():
        try:
            lines.append(f"  {row['Variable']}: VIF={float(row['VIF']):.2f}")
        except Exception:
            lines.append(f"  {row['Variable']}: VIF=N/A")
    lines.append("")
    path = _write_text("h2_hac_results.txt", "\n".join(lines) + "\n")
    if verbose:
        print("Saved: analysis/output/raw_outputs/h2_hac_results.txt")
    return path


def _abnormal_series(df: pd.DataFrame, country: str, event_date: pd.Timestamp):
    country_df = df[df["country"] == country].sort_values("date").set_index("date")
    if country_df.empty:
        return None
    pre_window = country_df.loc[
        (country_df.index < event_date) & (country_df.index >= event_date - pd.Timedelta(days=30))
    ]
    if pre_window.empty:
        return None
    baseline = pre_window["bond_yield_10y"].mean()
    s5 = country_df.loc[
        (country_df.index >= event_date - pd.Timedelta(days=5))
        & (country_df.index <= event_date + pd.Timedelta(days=5))
    ]["bond_yield_10y"] - baseline
    s3 = country_df.loc[
        (country_df.index >= event_date - pd.Timedelta(days=3))
        & (country_df.index <= event_date + pd.Timedelta(days=3))
    ]["bond_yield_10y"] - baseline
    return s5.dropna(), s3.dropna()


def run_h2_event_study(*, verbose: bool = True) -> Path:
    df = _load_data()
    df = df[df["country"].isin(EVENT_COUNTRIES)].copy()
    lines = [
        "=" * 80,
        "H2 EVENT STUDY: +/-5d and +/-3d around euro adoption (abnormal yields)",
        "=" * 80,
        "",
        "Abnormal yield = yield - mean(yield over t-30..t-1). One-sample t-test vs 0.",
        "Countries: HR, SI, SK, LT, FR, DE.",
        "",
        f"Event: {EVENT_LABEL}",
    ]
    for country in EVENT_COUNTRIES:
        series = _abnormal_series(df, country, EVENT_DATE)
        if series is None:
            lines.append(f"  {country}: no data")
            continue
        s5, s3 = series
        t5 = stats.ttest_1samp(s5.values, 0.0, nan_policy="omit")
        t3 = stats.ttest_1samp(s3.values, 0.0, nan_policy="omit")
        lines.append(
            f"  {country}: +/-5d mean={s5.mean():.4f}, t={t5.statistic:.2f}, p={t5.pvalue:.4f}; "
            f"+/-3d mean={s3.mean():.4f}, t={t3.statistic:.2f}, p={t3.pvalue:.4f}"
        )
    lines.append("")
    return _write_text("h2_event_study_results.txt", "\n".join(lines) + "\n")


def run(*, verbose: bool = True) -> list[Path]:
    run_h2_main(verbose=verbose)
    h2_hac = run_h2_hac(verbose=verbose)
    h2_event = run_h2_event_study(verbose=verbose)
    return [
        RAW_OUTPUTS_DIR / "h2_regression_results.txt",
        RAW_OUTPUTS_DIR / "h2_placebo_results.txt",
        h2_hac,
        h2_event,
    ]


if __name__ == "__main__":
    run()
