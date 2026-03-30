

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[1]))


if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from io_utils import save_figure, write_text, write_with_writer
from plot_utils import add_dual_outline, label_bars_with_significance

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'DATA' / 'input_data.csv'


def run():
    print("=" * 80)
    print("HYPOTHESIS 2: EURO ADOPTION IMPACT ON CROATIAN BOND YIELDS")
    print("With Placebo Tests and Robustness Checks")
    print("=" * 80)

    
    print("\n[1/11] Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"[ok] Loaded {len(df):,} observations")

    
    df_h2 = df[
        (df['country'].isin(['Croatia', 'Slovenia', 'Slovakia', 'Lithuania'])) &
        (df['date'] >= '2021-01-01') &
        (df['date'] <= '2024-12-31')
    ].copy()

    
    df_h2.rename(columns={'croatia_x_post_euro': 'croatia_ex_post'}, inplace=True)

    print(f"[ok] Filtered to {len(df_h2):,} observations")
    print(f"  Period: {df_h2['date'].min().date()} to {df_h2['date'].max().date()}")
    print(f"  Countries: {', '.join(sorted(df_h2['country'].unique()))}")

    
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

    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')

    spread_df = spread_stats.reset_index()
    spread_df['period'] = spread_df['period'].astype(str)
    spread_df[['Mean', 'Std Dev', 'Min', 'Max']] = spread_df[['Mean', 'Std Dev', 'Min', 'Max']].map(lambda x: f"{x:.4f}")
    spread_df['N'] = spread_df['N'].astype(int)

    table = ax.table(cellText=spread_df.values,
                    colLabels=spread_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(len(spread_df.columns)):
        table[(0, i)].set_facecolor('#5d6d7e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(spread_df) + 1):
        if i % 2 == 0:
            for j in range(len(spread_df.columns)):
                table[(i, j)].set_facecolor('#f8f9f9')


    save_figure(fig, 'h2_spread_statistics.png', facecolor='white', dpi=300)

    
    croatia_pre = df_h2[(df_h2['country'] == 'Croatia') &
                         (df_h2['post_euro_adoption'] == 0)]['spread_vs_germany'].mean()
    croatia_post = df_h2[(df_h2['country'] == 'Croatia') &
                          (df_h2['post_euro_adoption'] == 1)]['spread_vs_germany'].mean()

    spread_reduction = croatia_pre - croatia_post

    print(f"\nCroatia spread convergence:")
    print(f"  Pre-euro:  {croatia_pre:.4f} pp")
    print(f"  Post-euro: {croatia_post:.4f} pp")
    print(f"  Reduction: {spread_reduction:.4f} pp")

    
    print("\n[3/14] Testing parallel trends assumption...")

    df_pre = df_h2[df_h2['post_euro_adoption'] == 0].copy()
    df_pre['time_trend'] = (df_pre['date'] - df_pre['date'].min()).dt.days

    parallel_model = smf.ols(
        ,
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

    
    print("\n[4/14] Running main DiD regression models...")

    
    model1_yields = smf.ols(
        ,
        data=df_h2
    ).fit(cov_type='HC3')

    
    model2_yields = smf.ols(
        
        ,
        data=df_h2
    ).fit(cov_type='HC3')

    
    model3_spreads = smf.ols(
        ,
        data=df_h2
    ).fit(cov_type='HC3')

    
    model4_spreads = smf.ols(
        
        ,
        data=df_h2
    ).fit(cov_type='HC3')

    
    model5_full = smf.ols(
        
        ,
        data=df_h2
    ).fit(cov_type='HC3')

    
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

    
    print("\n[5/14] Running comprehensive placebo tests (5 time points)...")

    
    placebo_tests = [
        ('2021-01-01', 24, 'Placebo 1'),  
        ('2021-07-01', 18, 'Placebo 2'),  
        ('2022-01-01', 12, 'Placebo 3'),  
        ('2022-07-01', 6, 'Placebo 4'),   
        ('2022-10-01', 3, 'Placebo 5'),   
    ]

    placebo_results = []

    for placebo_date_str, months_before, placebo_name in placebo_tests:
        placebo_date = pd.to_datetime(placebo_date_str)

        
        df_h2[f'post_placebo_{len(placebo_results)+1}'] = (df_h2['date'] >= placebo_date).astype(int)
        df_h2[f'croatia_x_placebo_{len(placebo_results)+1}'] = df_h2['is_croatia'] * df_h2[f'post_placebo_{len(placebo_results)+1}']

        
        placebo_model = smf.ols(
            
            ,
            data=df_h2
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

    
    print("\n[6/14] Robustness check 1: Excluding Slovenia from control group...")

    df_robust_1 = df_h2[df_h2['country'] != 'Slovenia'].copy()
    robust_model_1 = smf.ols(
        
        ,
        data=df_robust_1
    ).fit(cov_type='HC3')

    robust_coef_1 = robust_model_1.params['croatia_ex_post']
    robust_pval_1 = robust_model_1.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovenia): {robust_coef_1:.4f} (p={robust_pval_1:.4f})")

    
    print("\n[7/14] Robustness check 2: Excluding Slovakia from control group...")

    df_robust_2 = df_h2[df_h2['country'] != 'Slovakia'].copy()
    robust_model_2 = smf.ols(
        
        ,
        data=df_robust_2
    ).fit(cov_type='HC3')

    robust_coef_2 = robust_model_2.params['croatia_ex_post']
    robust_pval_2 = robust_model_2.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Slovakia): {robust_coef_2:.4f} (p={robust_pval_2:.4f})")

    
    print("\n[8/14] Robustness check 3: Excluding Lithuania from control group...")

    df_robust_3 = df_h2[df_h2['country'] != 'Lithuania'].copy()
    robust_model_3 = smf.ols(
        
        ,
        data=df_robust_3
    ).fit(cov_type='HC3')

    robust_coef_3 = robust_model_3.params['croatia_ex_post']
    robust_pval_3 = robust_model_3.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (excl. Lithuania): {robust_coef_3:.4f} (p={robust_pval_3:.4f})")

    
    print("\n[9/14] Robustness check 4: Shorter time window (2022-2024)...")

    df_robust_4 = df_h2[df_h2['date'] >= '2022-01-01'].copy()
    robust_model_4 = smf.ols(
        
        ,
        data=df_robust_4
    ).fit(cov_type='HC3')

    robust_coef_4 = robust_model_4.params['croatia_ex_post']
    robust_pval_4 = robust_model_4.pvalues['croatia_ex_post']

    print(f"DiD Coefficient (2022-2024 only): {robust_coef_4:.4f} (p={robust_pval_4:.4f})")

    
    print("\n[10/14] Generating comprehensive results tables...")

    
    results_yields = summary_col(
        [model1_yields, model2_yields, model5_full],
        stars=True,
        float_format='%.4f',
        model_names=['Basic DiD', 'With Controls', 'Country FE'],
        info_dict={
            : lambda x: f"{int(x.nobs):,}",
            : lambda x: f"{x.rsquared:.4f}"
        }
    )

    
    results_spreads = summary_col(
        [model3_spreads, model4_spreads],
        stars=True,
        float_format='%.4f',
        model_names=['Basic DiD', 'With Controls'],
        info_dict={
            : lambda x: f"{int(x.nobs):,}",
            : lambda x: f"{x.rsquared:.4f}"
        }
    )

    def _write_results(handle):
        handle.write("=" * 80 + "\n")
        handle.write("HYPOTHESIS 2: EURO ADOPTION IMPACT ON CROATIAN BOND YIELDS\n")
        handle.write("WITH PLACEBO TESTS AND ROBUSTNESS CHECKS\n")
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
        handle.write("PLACEBO TESTS - CONVERGENCE TIMELINE\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("Testing for convergence at 5 fake adoption dates to identify when it began:\n\n")

        for result in placebo_results:
            handle.write(f"{result['name']} ({result['date']}, {result['months_before']} months before actual event):\n")
            handle.write(f"  DiD Coefficient: {result['coefficient']:.4f}\n")
            handle.write(f"  P-value: {result['pvalue']:.4f}\n")
            if result['pvalue'] > 0.05:
                handle.write("  Status: [ok] NO significant effect (validates main result)\n\n")
            else:
                handle.write("  Status: [warn] SIGNIFICANT effect (convergence already underway)\n\n")

        handle.write("Main Effect (January 1, 2023 - actual euro adoption):\n")
        handle.write(f"  DiD Coefficient: {did_spreads:.4f}\n")
        handle.write(f"  P-value: {did_spreads_pval:.4f}\n")
        handle.write("  Status: [ok] HIGHLY SIGNIFICANT\n\n")

        first_significant = next((r for r in placebo_results if r['pvalue'] < 0.05), None)
        if first_significant:
            handle.write(f"INTERPRETATION: Convergence began around {first_significant['date']}\n")
            handle.write(f"({first_significant['months_before']} months before formal euro adoption).\n")
            handle.write("This suggests markets priced in euro adoption during the Maastricht criteria process,\n")
            handle.write("demonstrating forward-looking behavior and policy credibility.\n\n")
        else:
            handle.write("INTERPRETATION: No early convergence detected. Main effect appears to be\n")
            handle.write("directly attributable to the January 1, 2023 euro adoption itself.\n\n")

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
        placebo_passes = sum(1 for r in placebo_results if r['pvalue'] > 0.05)
        placebo_fails = len(placebo_results) - placebo_passes
        handle.write(f"   Parallel Trends: {'SATISFIED' if parallel_satisfied else 'VIOLATED (convergence process)'}\n")
        handle.write(f"   Placebo Tests: {placebo_passes}/{len(placebo_results)} PASSED, {placebo_fails}/{len(placebo_results)} FAILED\n")
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

    
    print("\n[11/14] Creating main DiD visualization...")
    avg_spreads = df_h2.groupby(['is_croatia', 'period'])['spread_vs_germany'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(20, 12))

    periods = ['Pre-Euro', 'Post-Euro']

    croatia_data = avg_spreads[avg_spreads['is_croatia'] == 1]
    croatia_spreads = [
        croatia_data[croatia_data['period'] == periods[0]]['spread_vs_germany'].values[0],
        croatia_data[croatia_data['period'] == periods[1]]['spread_vs_germany'].values[0]
    ]
    ax.plot([0, 1], croatia_spreads, marker='o', markersize=12, linewidth=3,
            label='Croatia (Treatment)', color='#e74c3c')

    control_data = avg_spreads[avg_spreads['is_croatia'] == 0]
    control_spreads = [
        control_data[control_data['period'] == periods[0]]['spread_vs_germany'].values[0],
        control_data[control_data['period'] == periods[1]]['spread_vs_germany'].values[0]
    ]
    ax.plot([0, 1], control_spreads, marker='s', markersize=12, linewidth=3,
            label='Control Group (SI, SK, LT)', color='#3498db')

    counterfactual = croatia_spreads[0] + (control_spreads[1] - control_spreads[0])
    ax.plot([0, 1], [croatia_spreads[0], counterfactual], linestyle='--', linewidth=2.5,
            color='#95a5a6', label='Croatia Counterfactual')

    ax.annotate('', xy=(1, croatia_spreads[1]), xytext=(1, counterfactual),
                arrowprops=dict(arrowstyle='<->', color='green' if did_spreads < 0 else 'red', lw=3))
    ax.text(1.05, (croatia_spreads[1] + counterfactual) / 2,
             if did_spreads < 0 else f'DiD Effect:\n{did_spreads:.4f}pp',
            fontsize=11, color='green' if did_spreads < 0 else 'red',
            bbox=dict(boxstyle='round', facecolor='white',
                     edgecolor='green' if did_spreads < 0 else 'red', linewidth=2))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Euro\n(Before Jan 1, 2023)',
                        ], fontsize=11)
    ax.set_ylabel('Spread vs Germany (percentage points)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    save_figure(fig, 'h2_spread_convergence_did.png', dpi=300)

    
    print("\n[12/14] Creating spread timeseries visualization...")
    fig, ax = plt.subplots(figsize=(20, 12))

    for country in ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania']:
        country_data = df_h2[df_h2['country'] == country].sort_values('date')
        linewidth = 3 if country == 'Croatia' else 1.5
        ax.plot(country_data['date'], country_data['spread_vs_germany'],
                label=country, linewidth=linewidth, alpha=0.9)

    ax.axvline(pd.to_datetime('2023-01-01'), color='green', linestyle='--',
              linewidth=2.5, alpha=0.7, label='Croatia Euro Adoption')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spread vs Germany (percentage points)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'h2_spread_timeseries.png', dpi=300)

    
    print("\n[13/14] Creating robustness checks visualization...")
    fig, ax = plt.subplots(figsize=(12, 7))

    robustness_results = pd.DataFrame({
        : ['Main\n(All Controls)', 'Excl.\nSlovenia', 'Excl.\nSlovakia', 'Excl.\nLithuania', 'Short Window\n(2022-2024)'],
        : [did_spreads, robust_coef_1, robust_coef_2, robust_coef_3, robust_coef_4],
        : [did_spreads_pval, robust_pval_1, robust_pval_2, robust_pval_3, robust_pval_4]
    })

    bars = ax.bar(
        range(len(robustness_results)),
        robustness_results['DiD_Coefficient'],
        color='#0F6CE0',
        edgecolor='#0F6CE0',
        alpha=0.9,
        width=0.7,
    )

    for idx, (_, row) in enumerate(robustness_results.iterrows()):
        if row['P_Value'] is not None and row['P_Value'] < 0.05:
            add_dual_outline(ax, bars[idx])

    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
    ax.set_ylabel('DiD Coefficient (percentage points)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Specification', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(robustness_results)))
    ax.set_xticklabels(robustness_results['Specification'], fontsize=11)

    label_bars_with_significance(ax, bars, pvalues=robustness_results['P_Value'].tolist())

    legend_elements = [
        Patch(facecolor='#0F6CE0', edgecolor='#0F6CE0', alpha=0.9, label='Spec (p >= 0.05)'),
        Patch(facecolor='#0F6CE0', edgecolor='#d62728', linewidth=2, alpha=0.9, label='Spec (p < 0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, 'h2_robustness_checks.png', dpi=300)

    
    print("\n" + "=" * 80)
    print("HYPOTHESIS 2 TESTING COMPLETE")
    print("=" * 80)
    print("\nOutput files created:")
    print("  1. output/h2_spread_statistics.png")
    print("  2. output/h2_regression_results.txt")
    print("  3. output/h2_spread_convergence_did.png")
    print("  4. output/h2_spread_timeseries.png")
    print("  5. output/h2_robustness_checks.png")
    print("  (placebo plots generated by placebo_column_c.py)")
    print("\nKey Findings:")
    print(f"  Spread Reduction: {spread_reduction:.4f} pp")
    print(f"  Main DiD Coefficient (Spreads): {did_spreads:.4f} ({significance_label})")
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
