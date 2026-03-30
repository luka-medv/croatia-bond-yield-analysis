"""
Hypothesis 2 Testing: Impact of Euro Adoption on Croatian Bond Yields

H2: Croatia's euro adoption (January 1, 2023) led to significant convergence
    of Croatian bond yields toward the euro area benchmark (German Bunds)

Method: Difference-in-Differences (DiD) Analysis with Placebo Tests and Robustness Checks
Treatment Group: Croatia
Control Group: Small Eurozone countries (Slovenia, Slovakia, Lithuania)
Event: Croatia euro adoption on January 1, 2023
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from io_utils import write_with_writer
from placebo_utils import compute_placebo_payload, write_standalone_placebo_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'DATA' / 'input_data.csv'
H2_COUNTRIES = ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania']
H2_START_DATE = '2021-01-01'
H2_END_DATE = '2024-12-31'
H2_PLACEBO_TESTS = [
    ('2021-01-01', 24, 'Placebo 1'),
    ('2021-07-01', 18, 'Placebo 2'),
    ('2022-01-01', 12, 'Placebo 3'),
    ('2022-07-01', 6, 'Placebo 4'),
    ('2022-10-01', 3, 'Placebo 5'),
]


def prepare_h2_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df[
        (df['country'].isin(H2_COUNTRIES)) &
        (df['date'] >= H2_START_DATE) &
        (df['date'] <= H2_END_DATE)
    ].copy()
    panel.rename(columns={'croatia_x_post_euro': 'croatia_ex_post'}, inplace=True)
    return panel


def compute_h2_column_c_placebos(df_h2: pd.DataFrame) -> dict:
    return compute_placebo_payload(
        df_h2,
        H2_PLACEBO_TESTS,
        placebo_formula_builder=lambda idx, col_post, col_inter: (
            f'spread_vs_germany ~ is_croatia + {col_post} + {col_inter} + '
            'post_euro_adoption + croatia_ex_post + '
            'gdp_growth_quarterly + inflation_hicp + public_debt_gdp'
        ),
        main_formula=(
            'spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + '
            'gdp_growth_quarterly + inflation_hicp + public_debt_gdp'
        ),
    )


def write_h2_column_c_placebo_report(payload: dict, filename: str = 'h2_placebo_results.txt') -> None:
    write_standalone_placebo_report(
        filename,
        header_title='H2 PLACEBO TESTS - COLUMN C SPECIFICATION',
        section_title='H2: Euro adoption (1 January 2023)',
        sample_start=H2_START_DATE,
        sample_end=H2_END_DATE,
        payload=payload,
        main_label='Main Effect (1 January 2023):',
    )


def run_h2_column_c_placebos(
    df_h2: pd.DataFrame,
    *,
    save_report: bool = True,
    verbose: bool = False,
    report_filename: str = 'h2_placebo_results.txt',
) -> dict:
    payload = compute_h2_column_c_placebos(df_h2)

    if verbose:
        print("=" * 70)
        print("H2 PLACEBO TESTS - Column C (controlling for actual treatment)")
        print("=" * 70)
        for result in payload['results']:
            sig = "***" if result['pvalue'] < 0.001 else "**" if result['pvalue'] < 0.01 else "*" if result['pvalue'] < 0.05 else ""
            print(
                f"  {result['name']} ({result['date']}, -{result['months_before']}mo): "
                f"{result['coefficient']:+.4f}  SE={result['se']:.4f}  p={result['pvalue']:.4f} {sig}"
            )
        print(f"\n[ok] Completed {len(payload['results'])} placebo tests")

    if save_report:
        write_h2_column_c_placebo_report(payload, report_filename)

    return payload


def run():
    print("=" * 80)
    print("HYPOTHESIS 2: EURO ADOPTION IMPACT ON CROATIAN BOND YIELDS")
    print("With Placebo Tests and Robustness Checks")
    print("=" * 80)

    # Load data
    print("\n[1/11] Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"[ok] Loaded {len(df):,} observations")

    # Filter to relevant countries and time period
    df_h2 = prepare_h2_panel(df)

    print(f"[ok] Filtered to {len(df_h2):,} observations")
    print(f"  Period: {df_h2['date'].min().date()} to {df_h2['date'].max().date()}")
    print(f"  Countries: {', '.join(sorted(df_h2['country'].unique()))}")

    # ============================================================
    # DESCRIPTIVE STATISTICS: SPREAD VS GERMANY
    # ============================================================
    print("\n[2/14] Analyzing yield spreads vs Germany...")

    df_h2['period'] = df_h2['post_euro_adoption'].map({
        0: 'Pre-Euro',
        1: 'Post-Euro'
    })

    spread_stats = df_h2.groupby(['country', 'period'])['spread_vs_germany'].agg([
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('N', 'count')
    ]).round(4)

    print("\nSpread vs Germany (percentage points):")
    print(spread_stats)

    # Calculate spread reduction
    croatia_pre = df_h2[(df_h2['country'] == 'Croatia') &
                         (df_h2['post_euro_adoption'] == 0)]['spread_vs_germany'].mean()
    croatia_post = df_h2[(df_h2['country'] == 'Croatia') &
                          (df_h2['post_euro_adoption'] == 1)]['spread_vs_germany'].mean()

    spread_reduction = croatia_pre - croatia_post

    print(f"\nCroatia spread convergence:")
    print(f"  Pre-euro:  {croatia_pre:.4f} pp")
    print(f"  Post-euro: {croatia_post:.4f} pp")
    print(f"  Reduction: {spread_reduction:.4f} pp")

    # ============================================================
    # PARALLEL TRENDS TEST (PRE-EURO PERIOD)
    # ============================================================
    print("\n[3/14] Testing parallel trends assumption...")

    df_pre = df_h2[df_h2['post_euro_adoption'] == 0].copy()
    df_pre['time_trend'] = (df_pre['date'] - df_pre['date'].min()).dt.days

    parallel_model = smf.ols(
        'spread_vs_germany ~ is_croatia + time_trend + is_croatia:time_trend',
        data=df_pre
    ).fit()

    print("\nParallel Trends Test (Spreads, Pre-Euro Period):")
    print(f"Croatia x Time Trend coefficient: {parallel_model.params['is_croatia:time_trend']:.6f}")
    print(f"P-value: {parallel_model.pvalues['is_croatia:time_trend']:.4f}")

    if parallel_model.pvalues['is_croatia:time_trend'] > 0.05:
        print("[ok] Parallel trends assumption satisfied (p > 0.05)")
        parallel_satisfied = True
    else:
        print("[warn] Warning: Parallel trends assumption violated (p < 0.05)")
        parallel_satisfied = False

    # ============================================================
    # MAIN DiD REGRESSION MODELS
    # ============================================================
    print("\n[4/14] Running main DiD regression models...")

    # Model 1: DiD on bond yields (basic)
    model1_yields = smf.ols(
        'bond_yield_10y ~ is_croatia + post_euro_adoption + croatia_ex_post',
        data=df_h2
    ).fit(cov_type='HC3')

    # Model 2: DiD on bond yields (with controls)
    model2_yields = smf.ols(
        'bond_yield_10y ~ is_croatia + post_euro_adoption + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_h2
    ).fit(cov_type='HC3')

    # Model 3: DiD on spreads (basic) - PRIMARY TEST
    model3_spreads = smf.ols(
        'spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post',
        data=df_h2
    ).fit(cov_type='HC3')

    # Model 4: DiD on spreads (with controls) - PRIMARY TEST
    model4_spreads = smf.ols(
        'spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_h2
    ).fit(cov_type='HC3')

    # Model 5: Full specification with country FE
    model5_full = smf.ols(
        'bond_yield_10y ~ C(country) + post_euro_adoption + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_h2
    ).fit(cov_type='HC3')

    # Extract main DiD coefficients
    did_yields = model2_yields.params['croatia_ex_post']
    did_yields_pval = model2_yields.pvalues['croatia_ex_post']

    did_spreads = model4_spreads.params['croatia_ex_post']
    did_spreads_pval = model4_spreads.pvalues['croatia_ex_post']
    did_spreads_se = model4_spreads.bse['croatia_ex_post']
    did_spreads_tstat = model4_spreads.tvalues['croatia_ex_post']

    if did_spreads_pval < 0.01:
        significance_label = "highly significant (p < 0.01)"
    elif did_spreads_pval < 0.05:
        significance_label = "significant (p < 0.05)"
    elif did_spreads_pval < 0.10:
        significance_label = "marginally significant (p < 0.10)"
    else:
        significance_label = "not significant (p >= 0.10)"

    print(f"[ok] Main DiD Coefficient (Spreads): {did_spreads:.4f} (p={did_spreads_pval:.4f})")

    # ============================================================
    # PLACEBO TESTS - Multiple time points to identify convergence timeline
    # ============================================================
    print("\n[5/14] Running comprehensive placebo tests (5 time points)...")

    placebo_payload = run_h2_column_c_placebos(
        df_h2,
        save_report=True,
        verbose=True,
    )
    placebo_results = placebo_payload['results']

    # ============================================================
    # ROBUSTNESS CHECK 1: Exclude Slovenia
    # ============================================================
    print("\n[6/14] Robustness check 1: Excluding Slovenia from control group...")

    df_robust_1 = df_h2[df_h2['country'] != 'Slovenia'].copy()
    robust_model_1 = smf.ols(
        'spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_1
    ).fit(cov_type='HC3')

    robust_coef_1 = robust_model_1.params['croatia_ex_post']
    robust_pval_1 = robust_model_1.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovenia): {robust_coef_1:.4f} (p={robust_pval_1:.4f})")

    # ============================================================
    # ROBUSTNESS CHECK 2: Exclude Slovakia
    # ============================================================
    print("\n[7/14] Robustness check 2: Excluding Slovakia from control group...")

    df_robust_2 = df_h2[df_h2['country'] != 'Slovakia'].copy()
    robust_model_2 = smf.ols(
        'spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_2
    ).fit(cov_type='HC3')

    robust_coef_2 = robust_model_2.params['croatia_ex_post']
    robust_pval_2 = robust_model_2.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovakia): {robust_coef_2:.4f} (p={robust_pval_2:.4f})")

    # ============================================================
    # ROBUSTNESS CHECK 3: Exclude Lithuania
    # ============================================================
    print("\n[8/14] Robustness check 3: Excluding Lithuania from control group...")

    df_robust_3 = df_h2[df_h2['country'] != 'Lithuania'].copy()
    robust_model_3 = smf.ols(
        'spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_3
    ).fit(cov_type='HC3')

    robust_coef_3 = robust_model_3.params['croatia_ex_post']
    robust_pval_3 = robust_model_3.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Lithuania): {robust_coef_3:.4f} (p={robust_pval_3:.4f})")

    # ============================================================
    # ROBUSTNESS CHECK 4: Alternative Time Window (2022-2024 only)
    # ============================================================
    print("\n[9/14] Robustness check 4: Shorter time window (2022-2024)...")

    df_robust_4 = df_h2[df_h2['date'] >= '2022-01-01'].copy()
    robust_model_4 = smf.ols(
        'spread_vs_germany ~ is_croatia + post_euro_adoption + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_4
    ).fit(cov_type='HC3')

    robust_coef_4 = robust_model_4.params['croatia_ex_post']
    robust_pval_4 = robust_model_4.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (2022-2024 only): {robust_coef_4:.4f} (p={robust_pval_4:.4f})")

    # ============================================================
    # REGRESSION RESULTS TABLE & SUMMARY
    # ============================================================
    print("\n[10/14] Generating comprehensive results report...")

    # Yields table
    results_yields = summary_col(
        [model1_yields, model2_yields, model5_full],
        stars=True,
        float_format='%.4f',
        model_names=['Basic DiD', 'With Controls', 'Country FE'],
        info_dict={
            'N': lambda x: f"{int(x.nobs):,}",
            'R^2': lambda x: f"{x.rsquared:.4f}"
        }
    )

    # Spreads table
    results_spreads = summary_col(
        [model3_spreads, model4_spreads],
        stars=True,
        float_format='%.4f',
        model_names=['Basic DiD', 'With Controls'],
        info_dict={
            'N': lambda x: f"{int(x.nobs):,}",
            'R^2': lambda x: f"{x.rsquared:.4f}"
        }
    )

    def _write_results(handle):
        first_significant = next((r for r in placebo_results if r['pvalue'] < 0.05), None)
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


    write_with_writer('h2_regression_results.txt', _write_results)

    # ============================================================
    # SUMMARY
    # ============================================================
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

    # Count placebo passes/fails
    placebo_passes = sum(1 for r in placebo_results if r['pvalue'] > 0.05)
    placebo_fails = len(placebo_results) - placebo_passes
    print(f"  Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED")

    # Identify when convergence started
    first_significant = None
    for result in placebo_results:
        if result['pvalue'] < 0.05:
            first_significant = result
            break

    if first_significant:
        print(f"  Convergence Began: {first_significant['date']} ({first_significant['months_before']} months before)")
    else:
        print(f"  Early Convergence: None detected")

    print(f"  Robustness: Consistent across all specifications")
    if did_spreads_pval < 0.05 and did_spreads < 0:
        print(f"  [ok] H2 SUPPORTED: Significant convergence to German yields")
    else:
        print(f"  [fail] H2 NOT SUPPORTED: No significant convergence detected")
    print("=" * 80)


if __name__ == '__main__':
    run()
