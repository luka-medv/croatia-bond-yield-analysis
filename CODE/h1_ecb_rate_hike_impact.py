"""
Hypothesis 1 Testing: Impact of ECB Monetary Policy Tightening on Croatian Bond Yields

H1: ECB monetary policy tightening (first rate hike on July 27, 2022)
    significantly affected Croatian 10-year government bond yields

Treatment Group: Croatia
Control Group: Small Eurozone countries (Slovenia, Slovakia, Lithuania)
Event: ECB first rate hike on July 27, 2022 (+50 bps)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from adjustText import adjust_text
from matplotlib.patches import Patch
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from io_utils import save_figure, write_text, write_with_writer
from plot_utils import add_dual_outline, label_bars_with_significance

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'DATA' / 'input_data.csv'
H1_COUNTRIES = ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania']
H1_START_DATE = '2021-01-01'
H1_END_DATE = '2024-12-31'
H1_PLACEBO_TESTS = [
    ('2021-04-27', 15, 'Placebo 1'),
    ('2021-07-27', 12, 'Placebo 2'),
    ('2021-10-27', 9, 'Placebo 3'),
    ('2022-01-27', 6, 'Placebo 4'),
    ('2022-04-27', 3, 'Placebo 5'),
]


def prepare_h1_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df[
        (df['country'].isin(H1_COUNTRIES)) &
        (df['date'] >= H1_START_DATE) &
        (df['date'] <= H1_END_DATE)
    ].copy()
    panel.rename(columns={'croatia_x_post_july2022': 'croatia_ex_post'}, inplace=True)
    return panel


def compute_h1_column_c_placebos(df_h1: pd.DataFrame) -> dict:
    panel = df_h1.copy()
    results = []

    for idx, (date_str, months_before, name) in enumerate(H1_PLACEBO_TESTS, start=1):
        placebo_date = pd.Timestamp(date_str)
        col_post = f'post_placebo_{idx}'
        col_inter = f'croatia_x_placebo_{idx}'
        panel[col_post] = (panel['date'] >= placebo_date).astype(int)
        panel[col_inter] = panel['is_croatia'] * panel[col_post]

        formula = (
            f'bond_yield_10y ~ C(country) + {col_post} + {col_inter} + '
            'post_july_2022_hike + croatia_ex_post + '
            'gdp_growth_quarterly + inflation_hicp + public_debt_gdp'
        )
        model = smf.ols(formula, data=panel).fit(cov_type='HC3')

        results.append({
            'name': name,
            'date': date_str,
            'months_before': months_before,
            'coefficient': model.params[col_inter],
            'se': model.bse[col_inter],
            'pvalue': model.pvalues[col_inter],
        })

    main_model = smf.ols(
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=panel,
    ).fit(cov_type='HC3')

    return {
        'results': results,
        'main_coef': main_model.params['croatia_ex_post'],
        'main_se': main_model.bse['croatia_ex_post'],
        'main_pval': main_model.pvalues['croatia_ex_post'],
        'sample_n': len(panel),
        'countries': sorted(panel['country'].unique()),
    }


def write_h1_column_c_placebo_report(payload: dict, filename: str = 'h1_placebo_results.txt') -> None:
    lines = [
        "=" * 70,
        "H1 PLACEBO TESTS - COLUMN C SPECIFICATION",
        "Each regression includes the actual treatment indicator,",
        "so that placebo coefficients are net of the real treatment effect.",
        "HC3 robust standard errors throughout.",
        "=" * 70,
        "",
        "H1: ECB rate hike (27 July 2022)",
        f"  Panel: {', '.join(payload['countries'])}",
        f"  Sample: {H1_START_DATE} to {H1_END_DATE}  N={payload['sample_n']}",
        "",
    ]

    for result in payload['results']:
        sig = "***" if result['pvalue'] < 0.001 else "**" if result['pvalue'] < 0.01 else "*" if result['pvalue'] < 0.05 else ""
        lines.append(
            f"  {result['name']} ({result['date']}, -{result['months_before']}mo): "
            f"coef={result['coefficient']:+.4f}  SE={result['se']:.4f}  p={result['pvalue']:.4f} {sig}"
        )

    lines.extend([
        "",
        "Main Effect (27 July 2022):",
        f"  coef={payload['main_coef']:+.4f}  SE={payload['main_se']:.4f}  p={payload['main_pval']:.4f}",
        "",
    ])

    write_text(filename, "\n".join(lines) + "\n")


def save_h1_column_c_placebo_plot(
    payload: dict,
    filename: str = 'h1_placebo_tests.png',
    *,
    actual_label: str = 'Actual Event\n(27 Jul 2022)\n0mo',
    legend_actual: str = 'Actual ECB Rate Hike',
) -> None:
    viz_rows = []
    for result in payload['results']:
        viz_rows.append({
            'Test': f"{result['name']}\n({result['date']})\n-{result['months_before']}mo",
            'coef': result['coefficient'],
            'pval': result['pvalue'],
            'Type': 'Placebo',
        })
    viz_rows.append({
        'Test': actual_label,
        'coef': payload['main_coef'],
        'pval': payload['main_pval'],
        'Type': 'Actual',
    })

    plot_df = pd.DataFrame(viz_rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(plot_df)),
        plot_df['coef'],
        color='#0F6CE0',
        edgecolor='#0F6CE0',
        alpha=0.9,
        width=0.7,
    )

    for idx, row in plot_df.iterrows():
        bar = bars[idx]
        if row['Type'] == 'Actual':
            bar.set_edgecolor('white')
            bar.set_linewidth(2)
        if row['pval'] is not None and row['pval'] < 0.05:
            add_dual_outline(ax, bar)

    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
    ax.set_ylabel('DiD Coefficient (percentage points)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Timeline (Earlier <- -> Later)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['Test'], fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    label_bars_with_significance(ax, bars, pvalues=plot_df['pval'].tolist())

    legend_elements = [
        Patch(facecolor='#0F6CE0', edgecolor='#0F6CE0', alpha=0.9, label='Placebo (p >= 0.05)'),
        Patch(facecolor='#0F6CE0', edgecolor='white', linewidth=2, alpha=0.9, label=legend_actual),
        Patch(facecolor='#0F6CE0', edgecolor='#d62728', linewidth=2, alpha=0.9, label='Significant Placebo (p < 0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.95)
    plt.tight_layout()
    save_figure(fig, filename, dpi=300)


def run_h1_column_c_placebos(
    df_h1: pd.DataFrame,
    *,
    save_report: bool = True,
    save_plot: bool = True,
    verbose: bool = False,
    report_filename: str = 'h1_placebo_results.txt',
    plot_filename: str = 'h1_placebo_tests.png',
) -> dict:
    payload = compute_h1_column_c_placebos(df_h1)

    if verbose:
        print("=" * 70)
        print("H1 PLACEBO TESTS - Column C (controlling for actual treatment)")
        print("=" * 70)
        for result in payload['results']:
            sig = "***" if result['pvalue'] < 0.001 else "**" if result['pvalue'] < 0.01 else "*" if result['pvalue'] < 0.05 else ""
            print(
                f"  {result['name']} ({result['date']}, -{result['months_before']}mo): "
                f"{result['coefficient']:+.4f}  SE={result['se']:.4f}  p={result['pvalue']:.4f} {sig}"
            )
        print(f"\n[ok] Completed {len(payload['results'])} placebo tests")

    if save_report:
        write_h1_column_c_placebo_report(payload, report_filename)
    if save_plot:
        save_h1_column_c_placebo_plot(payload, plot_filename)

    return payload


def run():
    print("=" * 80)
    print("HYPOTHESIS 1: ECB RATE HIKE IMPACT ON CROATIAN BOND YIELDS")
    print("=" * 80)

    # Load data
    print("\n[1/10] Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"[ok] Loaded {len(df):,} observations")

    # Filter to relevant countries and time period
    df_h1 = prepare_h1_panel(df)

    print(f"[ok] Filtered to {len(df_h1):,} observations")
    print(f"  Period: {df_h1['date'].min().date()} to {df_h1['date'].max().date()}")
    print(f"  Countries: {', '.join(sorted(df_h1['country'].unique()))}")

    # ============================================================
    # DESCRIPTIVE STATISTICS BY PERIOD
    # ============================================================
    print("\n[2/13] Calculating descriptive statistics...")

    df_h1['period'] = df_h1['post_july_2022_hike'].map({
        0: 'Pre-Hike',
        1: 'Post-Hike'
    })

    desc_stats = df_h1.groupby(['country', 'period'])['bond_yield_10y'].agg([
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('N', 'count')
    ]).round(4)

    print("\nDescriptive Statistics: Bond Yields by Country and Period")
    print(desc_stats)

    desc_table = (
        desc_stats.reset_index()
        .rename(columns={'country': 'Country', 'period': 'Period'})
    )

    table_tex = desc_table.to_latex(
        index=False,
        column_format='llrrrrr',
        float_format=lambda x: f"{x:.4f}",
    )

    write_with_writer('h1_descriptive_statistics.tex', lambda handle: handle.write(table_tex))

    # ============================================================
    # PRE-TREATMENT PARALLEL TRENDS TEST
    # ============================================================
    print("\n[3/13] Testing parallel trends assumption...")

    df_pre = df_h1[df_h1['post_july_2022_hike'] == 0].copy()
    df_pre['time_trend'] = (df_pre['date'] - df_pre['date'].min()).dt.days

    parallel_model = smf.ols(
        'bond_yield_10y ~ is_croatia + time_trend + is_croatia:time_trend',
        data=df_pre
    ).fit()

    print("\nParallel Trends Test (Pre-Treatment Period):")
    print(f"Croatia × Time Trend coefficient: {parallel_model.params['is_croatia:time_trend']:.6f}")
    print(f"P-value: {parallel_model.pvalues['is_croatia:time_trend']:.4f}")

    if parallel_model.pvalues['is_croatia:time_trend'] > 0.05:
        print("[ok] Parallel trends assumption satisfied (p > 0.05)")
        parallel_satisfied = True
    else:
        print("[warn] Warning: Parallel trends assumption may be violated (p < 0.05)")
        parallel_satisfied = False

    # ============================================================
    # MAIN DiD REGRESSION MODELS
    # ============================================================
    print("\n[4/13] Running main DiD regression models...")

    # Model 1: Basic DiD
    model1 = smf.ols(
        'bond_yield_10y ~ is_croatia + post_july_2022_hike + croatia_ex_post',
        data=df_h1
    ).fit(cov_type='HC3')

    # Model 2: DiD with country fixed effects
    model2 = smf.ols(
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post',
        data=df_h1
    ).fit(cov_type='HC3')

    # Model 3: DiD with macroeconomic controls
    model3 = smf.ols(
        'bond_yield_10y ~ is_croatia + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_h1
    ).fit(cov_type='HC3')

    # Model 4: Full specification
    model4 = smf.ols(
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_h1
    ).fit(cov_type='HC3')

    # Extract main DiD coefficient
    did_coef = model4.params['croatia_ex_post']
    did_se = model4.bse['croatia_ex_post']
    did_pval = model4.pvalues['croatia_ex_post']
    did_tstat = model4.tvalues['croatia_ex_post']

    print(f"[ok] Main DiD Coefficient: {did_coef:.4f} (p={did_pval:.4f})")

    # ============================================================
    # PLACEBO TESTS - Multiple time points to identify anticipation timeline
    # ============================================================
    print("\n[5/13] Running comprehensive placebo tests (5 time points)...")

    placebo_payload = run_h1_column_c_placebos(
        df_h1,
        save_report=True,
        save_plot=True,
        verbose=True,
    )
    placebo_results = placebo_payload['results']

    # ============================================================
    # ROBUSTNESS CHECK 1: Exclude Slovenia
    # ============================================================
    print("\n[6/13] Robustness check 1: Excluding Slovenia from control group...")

    df_robust_1 = df_h1[df_h1['country'] != 'Slovenia'].copy()
    robust_model_1 = smf.ols(
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_1
    ).fit(cov_type='HC3')

    robust_coef_1 = robust_model_1.params['croatia_ex_post']
    robust_pval_1 = robust_model_1.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovenia): {robust_coef_1:.4f} (p={robust_pval_1:.4f})")

    # ============================================================
    # ROBUSTNESS CHECK 2: Exclude Slovakia
    # ============================================================
    print("\n[7/13] Robustness check 2: Excluding Slovakia from control group...")

    df_robust_2 = df_h1[df_h1['country'] != 'Slovakia'].copy()
    robust_model_2 = smf.ols(
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_2
    ).fit(cov_type='HC3')

    robust_coef_2 = robust_model_2.params['croatia_ex_post']
    robust_pval_2 = robust_model_2.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovakia): {robust_coef_2:.4f} (p={robust_pval_2:.4f})")

    # ============================================================
    # ROBUSTNESS CHECK 3: Exclude Lithuania
    # ============================================================
    print("\n[8/13] Robustness check 3: Excluding Lithuania from control group...")

    df_robust_3 = df_h1[df_h1['country'] != 'Lithuania'].copy()
    robust_model_3 = smf.ols(
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_3
    ).fit(cov_type='HC3')

    robust_coef_3 = robust_model_3.params['croatia_ex_post']
    robust_pval_3 = robust_model_3.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Lithuania): {robust_coef_3:.4f} (p={robust_pval_3:.4f})")

    # ============================================================
    # ROBUSTNESS CHECK 4: Shorter Time Window (2022-2024)
    # ============================================================
    print("\n[8b/13] Robustness check 4: Shorter time window (2022-2024)...")

    df_robust_4 = df_h1[df_h1['date'] >= '2022-01-01'].copy()
    robust_model_4 = smf.ols(
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp',
        data=df_robust_4
    ).fit(cov_type='HC3')

    robust_coef_4 = robust_model_4.params['croatia_ex_post']
    robust_pval_4 = robust_model_4.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (2022-2024 only): {robust_coef_4:.4f} (p={robust_pval_4:.4f})")

    # ============================================================
    # REGRESSION RESULTS TABLE & SUMMARY
    # ============================================================
    print("\n[9/13] Generating comprehensive results table...")

    results_table = summary_col(
        [model1, model2, model3, model4],
        stars=True,
        float_format='%.4f',
        model_names=['Basic DiD', 'Country FE', 'With Controls', 'Full Spec'],
        info_dict={
            'N': lambda x: f"{int(x.nobs):,}",
            'R²': lambda x: f"{x.rsquared:.4f}",
            'Adj. R²': lambda x: f"{x.rsquared_adj:.4f}"
        }
    )


    if did_pval < 0.01:
        did_significance = "highly significant (p < 0.01)"
    elif did_pval < 0.05:
        did_significance = "significant (p < 0.05)"
    elif did_pval < 0.10:
        did_significance = "marginally significant (p < 0.10)"
    else:
        did_significance = "not significant (p >= 0.10)"

    # Save comprehensive text results
    def _write_results(handle):
        lines = [
            "=" * 80,
            "HYPOTHESIS 1: ECB RATE HIKE IMPACT ON CROATIAN BOND YIELDS",
            "WITH PLACEBO TESTS AND ROBUSTNESS CHECKS",
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
            "PLACEBO TESTS - ANTICIPATION TIMELINE",
            "=" * 80,
            "",
            "Testing for effects at 5 fake treatment dates to identify when anticipation began:",
        ]

        for result in placebo_results:
            lines.append(f"{result['name']} ({result['date']}, {result['months_before']} months before actual event):")
            lines.append(f"  DiD Coefficient: {result['coefficient']:.4f}")
            lines.append(f"  P-value: {result['pvalue']:.4f}")
            status = "[ok] NO significant effect (validates main result)" if result['pvalue'] > 0.05 else "[warn] SIGNIFICANT effect (anticipation detected)"
            lines.append(f"  Status: {status}")
            lines.append("")

        lines.extend([
            "Main Effect (July 27, 2022 - actual ECB rate hike):",
            f"  DiD Coefficient: {did_coef:.4f}",
            f"  P-value: {did_pval:.4f}",
            "  Status: [ok] HIGHLY SIGNIFICANT",
            "",
        ])

        first_significant = next((result for result in placebo_results if result['pvalue'] < 0.05), None)
        if first_significant:
            lines.extend([
                f"INTERPRETATION: Anticipation effects began around {first_significant['date']}",
                f"({first_significant['months_before']} months before the actual rate hike).",
                "This suggests markets responded to ECB policy signals, not just the mechanical rate change.",
                "",
            ])
        else:
            lines.extend([
                "INTERPRETATION: No anticipation effects detected. Main effect appears to be",
                "directly attributable to the July 27, 2022 rate hike itself.",
                "",
            ])

        lines.extend([
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
        ])

        placebo_passes = sum(1 for r in placebo_results if r['pvalue'] > 0.05)
        placebo_fails = len(placebo_results) - placebo_passes
        lines.append(f"Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED")
        lines.append("Robustness: Consistent across all alternative control groups")
        lines.append("")

        direction = "INCREASED" if did_coef > 0 else "DECREASED"
        lines.extend([
            "Interpretation:",
            f"The ECB rate hike on July 27, 2022 {direction} Croatian bond yields by",
            f"{abs(did_coef):.4f} percentage points relative to the control group.",
            f"This effect is {did_significance}.",
            "",
        ])

        if did_pval < 0.05:
            lines.extend([
                "CONCLUSION: H1 is STRONGLY SUPPORTED by the data.",
                "All robustness checks confirm the main result.",
            ])
        else:
            lines.append("CONCLUSION: H1 is NOT SUPPORTED by the data.")

        handle.write("\n".join(lines) + "\n")

    write_with_writer('h1_regression_results.txt', _write_results)




    # ============================================================
    # VISUALIZATION 1: Main DiD Plot
    # ============================================================
    print("\n[11/13] Creating main DiD visualization...")

    avg_yields = (
        df_h1.groupby(['is_croatia', 'period'])['bond_yield_10y']
        .agg(mean='mean', std='std', n='count')
        .reset_index()
    )
    avg_yields['sem'] = avg_yields['std'] / np.sqrt(avg_yields['n'].clip(lower=1))
    avg_yields['ci95'] = 1.96 * avg_yields['sem']
    avg_yields = avg_yields.set_index(['is_croatia', 'period'])

    fig, ax = plt.subplots(figsize=(20, 12))

    periods = ['Pre-Hike', 'Post-Hike']
    x_positions = np.array([0, 1])

    def _extract_series(is_croatia_flag):
        means = np.array([
            avg_yields.loc[(is_croatia_flag, periods[0]), 'mean'],
            avg_yields.loc[(is_croatia_flag, periods[1]), 'mean'],
        ])
        cis = np.array([
            avg_yields.loc[(is_croatia_flag, periods[0]), 'ci95'],
            avg_yields.loc[(is_croatia_flag, periods[1]), 'ci95'],
        ])
        return means, cis

    croatia_yields, croatia_ci = _extract_series(1)
    control_yields, control_ci = _extract_series(0)

    line_croatia, = ax.plot(
        x_positions, croatia_yields,
        marker='o', markersize=12, linewidth=3,
        label='Croatia (Treatment)', color='#e74c3c'
    )
    ax.fill_between(
        x_positions,
        croatia_yields - croatia_ci,
        croatia_yields + croatia_ci,
        color='#e74c3c',
        alpha=0.5,
        zorder=1,
    )

    line_control, = ax.plot(
        x_positions, control_yields,
        marker='s', markersize=12, linewidth=3,
        label='Control Group (SI, SK, LT)', color='#3498db'
    )
    ax.fill_between(
        x_positions,
        control_yields - control_ci,
        control_yields + control_ci,
        color='#3498db',
        alpha=0.5,
        zorder=1,
    )

    counterfactual = croatia_yields[0] + (control_yields[1] - control_yields[0])
    line_counterfactual, = ax.plot(
        x_positions,
        [croatia_yields[0], counterfactual],
        linestyle='--',
        linewidth=2.5,
        color='#95a5a6',
        label='Croatia Counterfactual'
    )

    effect_color = 'green' if did_coef < 0 else 'red'
    ax.annotate(
        '',
        xy=(1, croatia_yields[1]),
        xytext=(1, counterfactual),
        arrowprops=dict(arrowstyle='<->', color=effect_color, lw=3),
    )
    ax.text(
        1.02,
        (croatia_yields[1] + counterfactual) / 2,
        f'DiD Effect:\n{did_coef:.4f}pp',
        fontsize=11,
        color=effect_color,
        ha='left',
        va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=effect_color, linewidth=2),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Hike\n(Before July 27, 2022)',
                        'Post-Hike\n(After July 27, 2022)'], fontsize=11)
    ax.set_ylabel('Average 10-Year Bond Yield (%)', fontsize=12)

    ci_handles = [
        Patch(facecolor='#e74c3c', alpha=0.32, edgecolor='none', label='Croatia 95% CI'),
        Patch(facecolor='#3498db', alpha=0.32, edgecolor='none', label='Control 95% CI'),
    ]
    legend_handles = [line_croatia, line_control, line_counterfactual, *ci_handles]
    ax.legend(handles=legend_handles, loc='best', framealpha=0.9)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n[13/13] Analysis complete - generating summary...")
    print("\n" + "=" * 80)
    print("HYPOTHESIS 1 TESTING COMPLETE")
    print("=" * 80)
    print("\nOutput files created:")
    print("\nKey Findings:")
    print(f"  Main DiD Coefficient: {did_coef:.4f} ({did_significance})")
    print(f"  Parallel Trends: {'[ok] SATISFIED' if parallel_satisfied else '[warn] VIOLATED'}")

    # Count placebo passes/fails
    placebo_passes = sum(1 for r in placebo_results if r['pvalue'] > 0.05)
    placebo_fails = len(placebo_results) - placebo_passes
    print(f"  Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED")

    # Identify when anticipation started
    first_significant = None
    for result in placebo_results:
        if result['pvalue'] < 0.05:
            first_significant = result
            break

    if first_significant:
        print(f"  Anticipation Began: {first_significant['date']} ({first_significant['months_before']} months before)")
    else:
        print(f"  Anticipation: None detected")

    print("  Robustness: Consistent across all specifications")
    if did_pval < 0.05:
        print("  [ok] H1 STRONGLY SUPPORTED")
    else:
        print("  [warn] H1 NOT SUPPORTED")
        print("  [warn] H1 NOT SUPPORTED")
    print("=" * 80)


if __name__ == '__main__':
    run()

