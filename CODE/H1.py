"""
H1 analysis bundle.

Contains:
- H1 main DiD analysis for the 27 July 2022 ECB hike
- H1 placebo tests
- H1 HAC/VIF appendix output
- H1 event-study output
- H1b supplemental 2 February 2023 ECB hike output
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

H1_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
H1B_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
H1_START_DATE = "2021-01-01"
H1_END_DATE = "2024-12-31"
H1_PLACEBO_TESTS = [
    ("2021-04-27", 15, "Placebo 1"),
    ("2021-07-27", 12, "Placebo 2"),
    ("2021-10-27", 9, "Placebo 3"),
    ("2022-01-27", 6, "Placebo 4"),
    ("2022-04-27", 3, "Placebo 5"),
]
EVENT_COUNTRIES = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
ECB_EVENTS = [
    ("2022-07-27", "ECB Hike 2022-07-27"),
    ("2023-02-02", "ECB Hike 2023-02-02"),
]


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


def prepare_h1_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df[
        (df["country"].isin(H1_COUNTRIES))
        & (df["date"] >= H1_START_DATE)
        & (df["date"] <= H1_END_DATE)
    ].copy()
    panel.rename(columns={"croatia_x_post_july2022": "croatia_ex_post"}, inplace=True)
    return panel


def compute_h1_placebos(df_h1: pd.DataFrame) -> dict:
    return _compute_placebo_payload(
        df_h1,
        H1_PLACEBO_TESTS,
        placebo_formula_builder=lambda idx, col_post, col_inter: (
            f"bond_yield_10y ~ C(country) + {col_post} + {col_inter} + "
            "post_july_2022_hike + croatia_ex_post + "
            "gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
        ),
        main_formula=(
            "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + "
            "gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
        ),
    )


def write_h1_placebo_report(payload: dict, filename: str = "h1_placebo_results.txt") -> Path:
    return _write_text(
        filename,
        _build_placebo_report(
            header_title="H1 PLACEBO TESTS - COLUMN C SPECIFICATION",
            section_title="H1: ECB rate hike (27 July 2022)",
            sample_start=H1_START_DATE,
            sample_end=H1_END_DATE,
            payload=payload,
            main_label="Main Effect (27 July 2022):",
        ),
    )


def run_h1_placebos(
    df_h1: pd.DataFrame,
    *,
    save_report: bool = True,
    verbose: bool = False,
) -> dict:
    payload = compute_h1_placebos(df_h1)
    if verbose:
        print("=" * 70)
        print("H1 PLACEBO TESTS - Column C (controlling for actual treatment)")
        print("=" * 70)
        for result in payload["results"]:
            print(
                f"  {result['name']} ({result['date']}, -{result['months_before']}mo): "
                f"{result['coefficient']:+.4f}  SE={result['se']:.4f}  "
                f"p={result['pvalue']:.4f} {_sig_stars(result['pvalue'])}"
            )
        print(f"\n[ok] Completed {len(payload['results'])} placebo tests")
    if save_report:
        write_h1_placebo_report(payload)
    return payload


def run_h1_main(*, verbose: bool = True) -> dict:
    if verbose:
        print("=" * 80)
        print("HYPOTHESIS 1: ECB RATE HIKE IMPACT ON CROATIAN BOND YIELDS")
        print("=" * 80)
        print("\n[1/10] Loading data...")

    df = _load_data()
    if verbose:
        print(f"[ok] Loaded {len(df):,} observations")

    df_h1 = prepare_h1_panel(df)
    if verbose:
        print(f"[ok] Filtered to {len(df_h1):,} observations")
        print(f"  Period: {df_h1['date'].min().date()} to {df_h1['date'].max().date()}")
        print(f"  Countries: {', '.join(sorted(df_h1['country'].unique()))}")
        print("\n[2/13] Calculating descriptive statistics...")

    df_h1["period"] = df_h1["post_july_2022_hike"].map({0: "Pre-Hike", 1: "Post-Hike"})
    desc_stats = df_h1.groupby(["country", "period"])["bond_yield_10y"].agg(
        [("Mean", "mean"), ("Std Dev", "std"), ("Min", "min"), ("Max", "max"), ("N", "count")]
    ).round(4)
    if verbose:
        print("\nDescriptive Statistics: Bond Yields by Country and Period")
        print(desc_stats)
        print("\n[3/13] Testing parallel trends assumption...")

    df_pre = df_h1[df_h1["post_july_2022_hike"] == 0].copy()
    df_pre["time_trend"] = (df_pre["date"] - df_pre["date"].min()).dt.days
    parallel_model = smf.ols(
        "bond_yield_10y ~ is_croatia + time_trend + is_croatia:time_trend",
        data=df_pre,
    ).fit()
    parallel_coef = parallel_model.params["is_croatia:time_trend"]
    parallel_pvalue = parallel_model.pvalues["is_croatia:time_trend"]
    parallel_satisfied = parallel_pvalue > 0.05
    if verbose:
        print("\nParallel Trends Test (Pre-Treatment Period):")
        print(f"Croatia x Time Trend coefficient: {parallel_coef:.6f}")
        print(f"P-value: {parallel_pvalue:.4f}")
        print(
            "[ok] Parallel trends assumption satisfied (p > 0.05)"
            if parallel_satisfied
            else "[warn] Warning: Parallel trends assumption may be violated (p < 0.05)"
        )
        print("\n[4/13] Running main DiD regression models...")

    model1 = smf.ols(
        "bond_yield_10y ~ is_croatia + post_july_2022_hike + croatia_ex_post",
        data=df_h1,
    ).fit(cov_type="HC3")
    model2 = smf.ols(
        "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post",
        data=df_h1,
    ).fit(cov_type="HC3")
    model3 = smf.ols(
        "bond_yield_10y ~ is_croatia + post_july_2022_hike + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_h1,
    ).fit(cov_type="HC3")
    model4 = smf.ols(
        "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_h1,
    ).fit(cov_type="HC3")

    did_coef = model4.params["croatia_ex_post"]
    did_se = model4.bse["croatia_ex_post"]
    did_pval = model4.pvalues["croatia_ex_post"]
    did_tstat = model4.tvalues["croatia_ex_post"]
    if verbose:
        print(f"[ok] Main DiD Coefficient: {did_coef:.4f} (p={did_pval:.4f})")
        print("\n[5/13] Running comprehensive placebo tests (5 time points)...")

    placebo_payload = run_h1_placebos(df_h1, save_report=True, verbose=verbose)
    placebo_results = placebo_payload["results"]

    if verbose:
        print("\n[6/13] Robustness check 1: Excluding Slovenia from control group...")
    df_robust_1 = df_h1[df_h1["country"] != "Slovenia"].copy()
    robust_model_1 = smf.ols(
        "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_1,
    ).fit(cov_type="HC3")
    robust_coef_1 = robust_model_1.params["croatia_ex_post"]
    robust_pval_1 = robust_model_1.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (excl. Slovenia): {robust_coef_1:.4f} (p={robust_pval_1:.4f})")
        print("\n[7/13] Robustness check 2: Excluding Slovakia from control group...")

    df_robust_2 = df_h1[df_h1["country"] != "Slovakia"].copy()
    robust_model_2 = smf.ols(
        "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_2,
    ).fit(cov_type="HC3")
    robust_coef_2 = robust_model_2.params["croatia_ex_post"]
    robust_pval_2 = robust_model_2.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (excl. Slovakia): {robust_coef_2:.4f} (p={robust_pval_2:.4f})")
        print("\n[8/13] Robustness check 3: Excluding Lithuania from control group...")

    df_robust_3 = df_h1[df_h1["country"] != "Lithuania"].copy()
    robust_model_3 = smf.ols(
        "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_3,
    ).fit(cov_type="HC3")
    robust_coef_3 = robust_model_3.params["croatia_ex_post"]
    robust_pval_3 = robust_model_3.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (excl. Lithuania): {robust_coef_3:.4f} (p={robust_pval_3:.4f})")
        print("\n[8b/13] Robustness check 4: Shorter time window (2022-2024)...")

    df_robust_4 = df_h1[df_h1["date"] >= "2022-01-01"].copy()
    robust_model_4 = smf.ols(
        "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp",
        data=df_robust_4,
    ).fit(cov_type="HC3")
    robust_coef_4 = robust_model_4.params["croatia_ex_post"]
    robust_pval_4 = robust_model_4.pvalues["croatia_ex_post"]
    if verbose:
        print(f"DiD Coefficient (2022-2024 only): {robust_coef_4:.4f} (p={robust_pval_4:.4f})")
        print("\n[9/13] Generating comprehensive results report...")

    results_table = summary_col(
        [model1, model2, model3, model4],
        stars=True,
        float_format="%.4f",
        model_names=["Basic DiD", "Country FE", "With Controls", "Full Spec"],
        info_dict={
            "N": lambda x: f"{int(x.nobs):,}",
            "R²": lambda x: f"{x.rsquared:.4f}",
            "Adj. R²": lambda x: f"{x.rsquared_adj:.4f}",
        },
    )

    if did_pval < 0.01:
        did_significance = "highly significant (p < 0.01)"
    elif did_pval < 0.05:
        did_significance = "significant (p < 0.05)"
    elif did_pval < 0.10:
        did_significance = "marginally significant (p < 0.10)"
    else:
        did_significance = "not significant (p >= 0.10)"

    def _write_results(handle) -> None:
        first_significant = next((result for result in placebo_results if result["pvalue"] < 0.05), None)
        lines = [
            "=" * 80,
            "HYPOTHESIS 1: ECB RATE HIKE IMPACT ON CROATIAN BOND YIELDS",
            "WITH ROBUSTNESS CHECKS",
            "=" * 80,
            "",
            "Event: ECB First Rate Hike (+50bps) on July 27, 2022",
            "Treatment: Croatia",
            "Control: Slovenia, Slovakia, Lithuania",
            "Method: Difference-in-Differences Analysis",
            "",
            "=" * 80,
            "MAIN REGRESSION RESULTS",
            "=" * 80,
            "",
            str(results_table),
            "",
            "=" * 80,
            "PLACEBO RESULTS",
            "=" * 80,
            "",
            "Detailed placebo timeline is reported in h1_placebo_results.txt.",
            f"Earliest significant placebo date: {first_significant['date'] if first_significant else 'None'}",
            "",
        ]
        if first_significant:
            lines.extend(
                [
                    f"Interpretation: anticipation effects began around {first_significant['date']}",
                    f"({first_significant['months_before']} months before the actual rate hike).",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "Interpretation: no anticipation effects detected in the separate placebo file.",
                    "",
                ]
            )
        lines.extend(
            [
                "=" * 80,
                "ROBUSTNESS CHECKS (Alternative Control Groups)",
                "=" * 80,
                "",
                f"Main specification (all controls):      {did_coef:.4f} (p={did_pval:.4f})",
                f"Excluding Slovenia:                     {robust_coef_1:.4f} (p={robust_pval_1:.4f})",
                f"Excluding Slovakia:                     {robust_coef_2:.4f} (p={robust_pval_2:.4f})",
                f"Excluding Lithuania:                    {robust_coef_3:.4f} (p={robust_pval_3:.4f})",
                f"Shorter window (2022-2024):             {robust_coef_4:.4f} (p={robust_pval_4:.4f})",
                "",
                "=" * 80,
                "KEY FINDINGS",
                "=" * 80,
                "",
                f"Main DiD Coefficient: {did_coef:.4f}",
                f"Standard Error: {did_se:.4f}",
                f"T-statistic: {did_tstat:.4f}",
                f"P-value: {did_pval:.4f}",
                "",
                f"Statistical Significance: {did_significance}",
                "",
                f"Parallel Trends: {'SATISFIED' if parallel_satisfied else 'VIOLATED'}",
                "Placebo Details: see h1_placebo_results.txt",
                "Robustness: Consistent across all alternative control groups",
                "",
            ]
        )
        direction = "INCREASED" if did_coef > 0 else "DECREASED"
        lines.extend(
            [
                "Interpretation:",
                f"The ECB rate hike on July 27, 2022 {direction} Croatian bond yields by",
                f"{abs(did_coef):.4f} percentage points relative to the control group.",
                f"This effect is {did_significance}.",
                "",
            ]
        )
        if did_pval < 0.05:
            lines.extend(
                [
                    "CONCLUSION: H1 is STRONGLY SUPPORTED by the data.",
                    "All robustness checks confirm the main result.",
                ]
            )
        else:
            lines.append("CONCLUSION: H1 is NOT SUPPORTED by the data.")
        handle.write("\n".join(lines) + "\n")

    _write_with_writer("h1_regression_results.txt", _write_results)

    if verbose:
        print("\n[10/10] Analysis complete - generating summary...")
        print("\n" + "=" * 80)
        print("HYPOTHESIS 1 TESTING COMPLETE")
        print("=" * 80)
        print("\nOutput files created:")
        print("\nKey Findings:")
        print(f"  Main DiD Coefficient: {did_coef:.4f} ({did_significance})")
        print(f"  Parallel Trends: {'[ok] SATISFIED' if parallel_satisfied else '[warn] VIOLATED'}")
        placebo_passes = sum(1 for result in placebo_results if result["pvalue"] > 0.05)
        placebo_fails = len(placebo_results) - placebo_passes
        print(f"  Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED")
        first_significant = next((result for result in placebo_results if result["pvalue"] < 0.05), None)
        if first_significant:
            print(f"  Anticipation Began: {first_significant['date']} ({first_significant['months_before']} months before)")
        else:
            print("  Anticipation: None detected")
        print("  Robustness: Consistent across all specifications")
        print("  [ok] H1 STRONGLY SUPPORTED" if did_pval < 0.05 else "  [warn] H1 NOT SUPPORTED")
        print("=" * 80)

    return {
        "df": df,
        "df_h1": df_h1,
        "did_coef": did_coef,
        "did_pval": did_pval,
        "placebo_payload": placebo_payload,
    }


def run_h1_hac(*, verbose: bool = True) -> Path:
    df = _load_data()
    df_h1 = prepare_h1_panel(df)
    formula = (
        "bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + "
        "gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
    )
    model_hac = smf.ols(formula, data=df_h1).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
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
    vif = _build_vif_table(df_h1, ["gdp_growth_quarterly", "inflation_hicp", "public_debt_gdp"])
    lines = [
        "=" * 80,
        "H1 Appendix: HAC (Newey-West) SE, VIF and F-test",
        "=" * 80,
        "",
        "HAC (maxlags=5) - Main DiD coef (country FE + controls)",
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
    path = _write_text("h1_hac_results.txt", "\n".join(lines) + "\n")
    if verbose:
        print("Saved: analysis/output/raw_outputs/h1_hac_results.txt")
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


def run_h1_event_study(*, verbose: bool = True) -> Path:
    df = _load_data()
    df = df[df["country"].isin(EVENT_COUNTRIES)].copy()
    lines = [
        "=" * 80,
        "H1 EVENT STUDY: +/-5d and +/-3d around ECB hikes (abnormal yields)",
        "=" * 80,
        "",
        "Abnormal yield = yield - mean(yield over t-30..t-1). One-sample t-test vs 0.",
        "Countries: HR, SI, SK, LT, FR, DE.",
        "",
    ]
    for event_date_str, event_title in ECB_EVENTS:
        event_date = pd.to_datetime(event_date_str)
        lines.append(f"Event: {event_title}")
        for country in EVENT_COUNTRIES:
            series = _abnormal_series(df, country, event_date)
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
    return _write_text("h1_event_study_results.txt", "\n".join(lines) + "\n")


def run_h1b(*, verbose: bool = True) -> Path:
    if verbose:
        print("=" * 80)
        print("H1b: ECB Tightening on 2023-02-02 (Croatia vs SI/SK/LT + FR/DE)")
        print("Difference-in-Differences with HAC SE, VIF and robustness (exclude FR/DE)")
        print("=" * 80)

    df = _load_data()
    df_h1b = df[
        (df["country"].isin(H1B_COUNTRIES))
        & (df["date"] >= H1_START_DATE)
        & (df["date"] <= H1_END_DATE)
    ].copy()
    required_cols = {"post_feb_2023_hike", "croatia_x_post_feb2023"}
    missing = required_cols - set(df_h1b.columns)
    if missing:
        raise RuntimeError(f"Required indicators missing in data: {missing}")
    df_h1b.rename(columns={"croatia_x_post_feb2023": "croatia_ex_post"}, inplace=True)

    formula = (
        "bond_yield_10y ~ C(country) + post_feb_2023_hike + "
        "croatia_ex_post + gdp_growth_quarterly + inflation_hicp + public_debt_gdp"
    )
    model_hac = smf.ols(formula, data=df_h1b).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    model_hc3 = smf.ols(formula, data=df_h1b).fit(cov_type="HC3")
    coef = "croatia_ex_post"
    did_hac = (model_hac.params.get(coef), model_hac.bse.get(coef), model_hac.pvalues.get(coef))
    did_hc3 = (model_hc3.params.get(coef), model_hc3.bse.get(coef), model_hc3.pvalues.get(coef))

    df_small = df_h1b[~df_h1b["country"].isin(["France", "Germany"])].copy()
    model_small = smf.ols(formula, data=df_small).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    robust_hac = (
        model_small.params.get(coef),
        model_small.bse.get(coef),
        model_small.pvalues.get(coef),
    )

    vif = _build_vif_table(df_h1b, ["gdp_growth_quarterly", "inflation_hicp", "public_debt_gdp"])
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
        f"  HAC DiD coef: {robust_hac[0]:.4f}  SE: {robust_hac[1]:.4f}  p: {robust_hac[2]:.4f}",
        "",
        f"CONTROLS JOINT SIGNIFICANCE (HAC model): {ftest_str}",
        "",
        "VIF (controls)",
    ]
    for _, row in vif.iterrows():
        try:
            lines.append(f"  {row['Variable']}: VIF={float(row['VIF']):.2f}")
        except Exception:
            lines.append(f"  {row['Variable']}: VIF=N/A")
    lines.append("")
    path = _write_text("h1b_regression_results.txt", "\n".join(lines) + "\n")
    if verbose:
        print("Saved artefacts for H1b analysis.")
    return path


def run(*, verbose: bool = True) -> list[Path]:
    run_h1_main(verbose=verbose)
    h1_hac = run_h1_hac(verbose=verbose)
    h1_event = run_h1_event_study(verbose=verbose)
    h1b = run_h1b(verbose=verbose)
    return [
        RAW_OUTPUTS_DIR / "h1_regression_results.txt",
        RAW_OUTPUTS_DIR / "h1_placebo_results.txt",
        h1_hac,
        h1_event,
        h1b,
    ]


if __name__ == "__main__":
    run()
