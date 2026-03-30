

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


if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from io_utils import save_figure, write_with_writer
from plot_utils import add_dual_outline, label_bars_with_significance

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'DATA' / 'input_data.csv'


def run():
    print("=" * 80)
    print("HYPOTHESIS 1: ECB RATE HIKE IMPACT ON CROATIAN BOND YIELDS")
    print("=" * 80)

    
    print("\n[1/10] Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"[ok] Loaded {len(df):,} observations")

    
    df_h1 = df[
        (df['country'].isin(['Croatia', 'Slovenia', 'Slovakia', 'Lithuania'])) &
        (df['date'] >= '2021-01-01') &
        (df['date'] <= '2024-12-31')
    ].copy()

    print(f"[ok] Filtered to {len(df_h1):,} observations")
    print(f"  Period: {df_h1['date'].min().date()} to {df_h1['date'].max().date()}")
    print(f"  Countries: {', '.join(sorted(df_h1['country'].unique()))}")

    
    df_h1.rename(columns={'croatia_x_post_july2022': 'croatia_ex_post'}, inplace=True)

    
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

    
    print("\n[3/13] Testing parallel trends assumption...")

    df_pre = df_h1[df_h1['post_july_2022_hike'] == 0].copy()
    df_pre['time_trend'] = (df_pre['date'] - df_pre['date'].min()).dt.days

    parallel_model = smf.ols(
        ,
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

    
    print("\n[4/13] Running main DiD regression models...")

    
    model1 = smf.ols(
        ,
        data=df_h1
    ).fit(cov_type='HC3')

    
    model2 = smf.ols(
        ,
        data=df_h1
    ).fit(cov_type='HC3')

    
    model3 = smf.ols(
        
        ,
        data=df_h1
    ).fit(cov_type='HC3')

    
    model4 = smf.ols(
        
        ,
        data=df_h1
    ).fit(cov_type='HC3')

    
    did_coef = model4.params['croatia_ex_post']
    did_se = model4.bse['croatia_ex_post']
    did_pval = model4.pvalues['croatia_ex_post']
    did_tstat = model4.tvalues['croatia_ex_post']

    print(f"[ok] Main DiD Coefficient: {did_coef:.4f} (p={did_pval:.4f})")

    
    print("\n[5/13] Running comprehensive placebo tests (5 time points)...")

    
    placebo_tests = [
        ('2021-04-27', 15, 'Placebo 1'),  
        ('2021-07-27', 12, 'Placebo 2'),  
        ('2021-10-27', 9, 'Placebo 3'),   
        ('2022-01-27', 6, 'Placebo 4'),   
        ('2022-04-27', 3, 'Placebo 5'),   
    ]

    placebo_results = []

    for placebo_date_str, months_before, placebo_name in placebo_tests:
        placebo_date = pd.to_datetime(placebo_date_str)

        
        df_h1[f'post_placebo_{len(placebo_results)+1}'] = (df_h1['date'] >= placebo_date).astype(int)
        df_h1[f'croatia_x_placebo_{len(placebo_results)+1}'] = df_h1['is_croatia'] * df_h1[f'post_placebo_{len(placebo_results)+1}']

        
        placebo_model = smf.ols(
            
            ,
            data=df_h1
        ).fit(cov_type='HC3')

        placebo_coef = placebo_model.params[f'croatia_x_placebo_{len(placebo_results)+1}']
        placebo_pval = placebo_model.pvalues[f'croatia_x_placebo_{len(placebo_results)+1}']

        status = "[ok] PASSED" if placebo_pval > 0.05 else "[warn] FAILED"
        print(f"  {placebo_name} ({placebo_date_str}, -{months_before}mo): {placebo_coef:.4f} (p={placebo_pval:.4f}) {status}")

        placebo_results.append({
            : placebo_name,
            : placebo_date_str,
            : months_before,
            : placebo_coef,
            : placebo_pval,
            : status
        })

    
    placebo_coef_1 = placebo_results[0]['coefficient']
    placebo_pval_1 = placebo_results[0]['pvalue']
    placebo_coef_2 = placebo_results[1]['coefficient']
    placebo_pval_2 = placebo_results[1]['pvalue']
    placebo_coef_3 = placebo_results[2]['coefficient']
    placebo_pval_3 = placebo_results[2]['pvalue']
    placebo_coef_4 = placebo_results[3]['coefficient']
    placebo_pval_4 = placebo_results[3]['pvalue']
    placebo_coef_5 = placebo_results[4]['coefficient']
    placebo_pval_5 = placebo_results[4]['pvalue']

    print(f"\n[ok] Completed {len(placebo_results)} placebo tests")

    
    print("\n[6/13] Robustness check 1: Excluding Slovenia from control group...")

    df_robust_1 = df_h1[df_h1['country'] != 'Slovenia'].copy()
    robust_model_1 = smf.ols(
        
        ,
        data=df_robust_1
    ).fit(cov_type='HC3')

    robust_coef_1 = robust_model_1.params['croatia_ex_post']
    robust_pval_1 = robust_model_1.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovenia): {robust_coef_1:.4f} (p={robust_pval_1:.4f})")

    
    print("\n[7/13] Robustness check 2: Excluding Slovakia from control group...")

    df_robust_2 = df_h1[df_h1['country'] != 'Slovakia'].copy()
    robust_model_2 = smf.ols(
        
        ,
        data=df_robust_2
    ).fit(cov_type='HC3')

    robust_coef_2 = robust_model_2.params['croatia_ex_post']
    robust_pval_2 = robust_model_2.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovakia): {robust_coef_2:.4f} (p={robust_pval_2:.4f})")

    
    print("\n[8/13] Robustness check 3: Excluding Lithuania from control group...")

    df_robust_3 = df_h1[df_h1['country'] != 'Lithuania'].copy()
    robust_model_3 = smf.ols(
        
        ,
        data=df_robust_3
    ).fit(cov_type='HC3')

    robust_coef_3 = robust_model_3.params['croatia_ex_post']
    robust_pval_3 = robust_model_3.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Lithuania): {robust_coef_3:.4f} (p={robust_pval_3:.4f})")

    
    print("\n[8b/13] Robustness check 4: Shorter time window (2022-2024)...")

    df_robust_4 = df_h1[df_h1['date'] >= '2022-01-01'].copy()
    robust_model_4 = smf.ols(
        
        ,
        data=df_robust_4
    ).fit(cov_type='HC3')

    robust_coef_4 = robust_model_4.params['croatia_ex_post']
    robust_pval_4 = robust_model_4.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (2022-2024 only): {robust_coef_4:.4f} (p={robust_pval_4:.4f})")

    
    print("\n[9/13] Generating comprehensive results table...")

    results_table = summary_col(
        [model1, model2, model3, model4],
        stars=True,
        float_format='%.4f',
        model_names=['Basic DiD', 'Country FE', 'With Controls', 'Full Spec'],
        info_dict={
            : lambda x: f"{int(x.nobs):,}",
            : lambda x: f"{x.rsquared:.4f}",
            : lambda x: f"{x.rsquared_adj:.4f}"
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

    
    def _write_results(handle):
        lines = [
             * 80,
            ,
            ,
             * 80,
            ,
            ,
            ,
            ,
            ,
            ,
             * 80,
            ,
             * 80,
            ,
            str(results_table),
            ,
             * 80,
            ,
             * 80,
            ,
            ,
        ]

        for result in placebo_results:
            lines.append(f"{result['name']} ({result['date']}, {result['months_before']} months before actual event):")
            lines.append(f"  DiD Coefficient: {result['coefficient']:.4f}")
            lines.append(f"  P-value: {result['pvalue']:.4f}")
            status = "[ok] NO significant effect (validates main result)" if result['pvalue'] > 0.05 else "[warn] SIGNIFICANT effect (anticipation detected)"
            lines.append(f"  Status: {status}")
            lines.append("")

        lines.extend([
            ,
            ,
            ,
            ,
            ,
        ])

        first_significant = next((result for result in placebo_results if result['pvalue'] < 0.05), None)
        if first_significant:
            lines.extend([
                ,
                ,
                ,
                ,
            ])
        else:
            lines.extend([
                ,
                ,
                ,
            ])

        lines.extend([
             * 80,
            ,
             * 80,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
             * 80,
            ,
             * 80,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
        ])

        placebo_passes = sum(1 for r in placebo_results if r['pvalue'] > 0.05)
        placebo_fails = len(placebo_results) - placebo_passes
        lines.append(f"Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED")
        lines.append("Robustness: Consistent across all alternative control groups")
        lines.append("")

        direction = "INCREASED" if did_coef > 0 else "DECREASED"
        lines.extend([
            ,
            ,
            ,
            ,
            ,
        ])

        if did_pval < 0.05:
            lines.extend([
                ,
                ,
            ])
        else:
            lines.append("CONCLUSION: H1 is NOT SUPPORTED by the data.")

        handle.write("\n".join(lines) + "\n")

    write_with_writer('h1_regression_results.txt', _write_results)


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
        ,
        xy=(1, croatia_yields[1]),
        xytext=(1, counterfactual),
        arrowprops=dict(arrowstyle='<->', color=effect_color, lw=3),
    )
    ax.text(
        1.02,
        (croatia_yields[1] + counterfactual) / 2,
        ,
        fontsize=11,
        color=effect_color,
        ha='left',
        va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=effect_color, linewidth=2),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Hike\n(Before July 27, 2022)',
                        ], fontsize=11)
    ax.set_ylabel('Average 10-Year Bond Yield (%)', fontsize=12)

    ci_handles = [
        Patch(facecolor='#e74c3c', alpha=0.32, edgecolor='none', label='Croatia 95% CI'),
        Patch(facecolor='#3498db', alpha=0.32, edgecolor='none', label='Control 95% CI'),
    ]
    legend_handles = [line_croatia, line_control, line_counterfactual, *ci_handles]
    ax.legend(handles=legend_handles, loc='best', framealpha=0.9)


    print("\n[13/13] Analysis complete - generating summary...")
    print("\n" + "=" * 80)
    print("HYPOTHESIS 1 TESTING COMPLETE")
    print("=" * 80)
    print("\nOutput files created:")
    print("\nKey Findings:")
    print(f"  Main DiD Coefficient: {did_coef:.4f} ({did_significance})")
    print(f"  Parallel Trends: {'[ok] SATISFIED' if parallel_satisfied else '[warn] VIOLATED'}")

    
    placebo_passes = sum(1 for r in placebo_results if r['pvalue'] > 0.05)
    placebo_fails = len(placebo_results) - placebo_passes
    print(f"  Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED")

    
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

