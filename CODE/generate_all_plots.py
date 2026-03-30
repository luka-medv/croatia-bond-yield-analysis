

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
import matplotlib.dates as mdates
from adjustText import adjust_text
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.patches import Patch


from io_utils import save_figure
from plot_utils import make_subplots, add_annotation, place_legend, add_dual_outline, label_bars_with_significance

print("=" * 80)
print("REGENERATING ALL PLOTS WITH PROFESSIONAL COLORS")
print("=" * 80)


print("\n[1/N] Loading data...")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'DATA'
df = pd.read_csv(DATA_DIR / 'input_data.csv', parse_dates=['date'])
print(f"[info] Loaded {len(df):,} observations")

KEY_EVENT_LINES = [
    (
        pd.to_datetime('2022-07-27'),
        ,
        ,
        {'text_offset_days': -45, 'ha': 'right'}
    ),
    (
        pd.to_datetime('2023-01-01'),
        ,
        ,
        {'text_offset_days': 45, 'ha': 'left'}
    ),
]


colors = {
    : '#D2691E',
    : '#1E3A8A',
    : '#15803D',
    : '#7E22CE',
    : '#92400E',
    : '#374151'
}
all_countries = ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'France', 'Germany']


def save_with_layout(fig, filename, **kwargs):
    try:
        fig.tight_layout(pad=1.05)
    except Exception:
        pass
    save_figure(fig, filename, **kwargs)


def _render_event_labels(
    ax,
    events,
    *,
    threshold_days: int = 210,
    base_frac: float = 1.01,
    step_frac: float = 0.12,
    anchor_frac: float = 0.88,
    max_frac: float = 1.08,
):
    
    if not events:
        return

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    anchor_y = ymin + span * anchor_frac
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    threshold = pd.Timedelta(days=threshold_days)

    last_at_level: dict[int, pd.Timestamp] = {}
    sorted_events = []
    for event in events:
        if len(event) == 3:
            ts, label, color = event
            options = {}
        elif len(event) == 4:
            ts, label, color, options = event
        else:
            raise ValueError("Each event must be a 3- or 4-tuple")
        sorted_events.append((pd.to_datetime(ts), label, color, options or {}))
    sorted_events.sort(key=lambda item: item[0])

    for ts, label, color, options in sorted_events:
        ax.axvline(ts, linestyle='--', linewidth=2.3, alpha=0.75, color=color, zorder=1.5)

        level = 0
        while level in last_at_level and (ts - last_at_level[level]) <= threshold:
            level += 1
        last_at_level[level] = ts

        y_frac = min(base_frac + level * step_frac, max_frac)
        x_offset_days = options.get('text_offset_days', 0)
        ha = options.get('ha', 'center')
        bbox_kwargs = {
            : 'round,pad=0.35',
            : 'white',
            : color,
            : 1.15,
            : 0.97,
        }
        bbox_kwargs.update(options.get('bbox', {}))
        arrowprops = {
            : '-|>',
            : color,
            : 1.0,
            : 0,
            : 6,
        }
        arrowprops.update(options.get('arrowprops', {}))

        ax.annotate(
            label,
            xy=(ts, anchor_y),
            xycoords='data',
            xytext=(ts + pd.Timedelta(days=x_offset_days), y_frac),
            textcoords=text_transform,
            ha=ha,
            va='bottom',
            bbox=bbox_kwargs,
            arrowprops=arrowprops,
            fontsize=options.get('fontsize', 10),
            fontweight=options.get('fontweight', 'bold'),
            annotation_clip=False,
        )


def annotate_vertical_events(ax, events=KEY_EVENT_LINES):
    _render_event_labels(ax, events)


print("\n[2/N] Creating all countries yields plot...")

fig, ax = make_subplots()

for country in all_countries:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else (2.5 if country == 'Germany' else 2)
    alpha = 1.0 if country in ['Croatia', 'Germany'] else 0.75
    ax.plot(data['date'], data['bond_yield_10y'],
            label=country, linewidth=linewidth, alpha=alpha)

annotate_vertical_events(ax)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('10-Year Government Bond Yield (%)', fontsize=13, fontweight='bold')
place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_with_layout(fig, '01_all_countries_yields.png')


print("\n[3/N] Creating Croatia vs small eurozone comparison...")

fig, ax = make_subplots()

for country in ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania']:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 4 if country == 'Croatia' else 2.5
    alpha = 1.0 if country == 'Croatia' else 0.8
    ax.plot(data['date'], data['bond_yield_10y'],
            label=country, linewidth=linewidth, alpha=alpha)

euro_date = pd.to_datetime('2023-01-01')
ax.axvspan(euro_date, df['date'].max(), alpha=0.1, zorder=0)
annotate_vertical_events(ax)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('10-Year Government Bond Yield (%)', fontsize=13, fontweight='bold')
place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_with_layout(fig, '02_croatia_vs_small_eurozone.png')


print("\n[4/N] Creating spreads vs Germany plot...")

fig, ax = make_subplots()

for country in ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'France']:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 4 if country == 'Croatia' else (3 if country == 'France' else 2.5)
    alpha = 1.0 if country in ['Croatia', 'France'] else 0.75
    ax.plot(data['date'], data['spread_vs_germany'],
            label=country, linewidth=linewidth, alpha=alpha)

euro_date = pd.to_datetime('2023-01-01')
ax.axvspan(euro_date, df['date'].max(), alpha=0.1, zorder=0)
ax.axhline(0, linestyle='-', linewidth=1.5, alpha=0.5)
annotate_vertical_events(ax)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Spread vs German Bund (percentage points)', fontsize=13, fontweight='bold')
place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_with_layout(fig, '03_spreads_vs_germany.png')


print("\n[5/N] Creating correlation heatmap...")

pivot_df = df.pivot_table(
    values='bond_yield_10y',
    index='date',
    columns='country'
)
correlation_matrix = pivot_df.corr()
fig, ax = make_subplots()
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap=cmap,
            center=0.9, vmin=0.8, vmax=1.0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=ax)
plt.setp(ax.get_xticklabels(), fontsize=11, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontsize=11, fontweight='bold', rotation=0)
save_with_layout(fig, '05_correlation_heatmap.png')


print("\n[6/N] Creating volatility comparison plot...")
fig, ax = make_subplots()
for country in all_countries:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else (2.5 if country == 'Germany' else 2)
    alpha = 1.0 if country in ['Croatia', 'Germany'] else 0.75
    ax.plot(data['date'], data['yield_std_30d'],
            label=country, linewidth=linewidth, alpha=alpha)

annotate_vertical_events(ax)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('30-Day Rolling Standard Deviation', fontsize=13, fontweight='bold')
place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_with_layout(fig, '06_volatility_comparison.png')


print("\n[7/N] Creating Croatia euro adoption timeline...")

croatia_data = df[df['country'] == 'Croatia'].sort_values('date')
fig, axes = make_subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax1, ax2 = axes

ax1.plot(croatia_data['date'], croatia_data['bond_yield_10y'],
         linewidth=3.5, label='Croatia 10Y Yield')
germany_data = df[df['country'] == 'Germany'].sort_values('date')
ax1.plot(germany_data['date'], germany_data['bond_yield_10y'],
         linewidth=2.5, alpha=0.7,
         linestyle='--', label='Germany 10Y Yield (Reference)')

events = [
    ('2018-07-10', 'Maastricht\nProcess\nBegins'),
    ('2022-07-12', 'EU Council\nApproves\nEuro Entry', {'x_offset': -70, 'ha': 'right'}),
    ('2023-01-01', 'EURO\nADOPTION', {'x_offset': 70, 'ha': 'left'}),
]

for entry in events:
    date_str, label = entry[:2]
    date = pd.to_datetime(date_str)
    ax1.axvline(date, linestyle='--', linewidth=2.5, alpha=0.7)

ylim = ax1.get_ylim()
ymin, ymax = ylim
vertical_margin = (ymax - ymin) * 0.03

for i, entry in enumerate(events):
    date_str, label = entry[:2]
    options = entry[2] if len(entry) >= 3 else {}
    date = pd.to_datetime(date_str)
    series = croatia_data.set_index('date')['bond_yield_10y'].sort_index()
    y_pos = series.asof(date)
    if pd.isna(y_pos):
        y_pos = series.iloc[-1]
    offset = 20
    default_valign = 'bottom'
    if (y_pos + vertical_margin) >= ymax:
        offset = -30
        default_valign = 'top'
    x_offset = options.get('x_offset', 0)
    y_offset = options.get('y_offset', offset)
    ha = options.get('ha', 'center')
    va = options.get('va', default_valign)
    bbox_kwargs = dict(boxstyle='round,pad=0.4', facecolor='aliceblue', edgecolor='black', linewidth=1, alpha=0.95)
    bbox_kwargs.update(options.get('bbox', {}))
    arrow_kwargs = dict(arrowstyle='-|>', color='black', lw=1.2, shrinkA=0, shrinkB=6)
    arrow_kwargs.update(options.get('arrowprops', {}))
    add_annotation(
        ax1,
        label,
        (date, y_pos),
        xytext=(x_offset, y_offset),
        textcoords='offset points',
        ha=ha,
        va=va,
        bbox=bbox_kwargs,
        arrowprops=arrow_kwargs,
        fontsize=10,
        fontweight='bold'
    )

ax1.set_ylabel('10-Year Bond Yield (%)', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticklabels([])

ax2.plot(croatia_data['date'], croatia_data['spread_vs_germany'],
         linewidth=3.5, label='Croatia Spread vs Germany')
ax2.axhline(0, linestyle='-', linewidth=1.5, alpha=0.5)

for entry in events:
    date_str, label = entry[:2]
    date = pd.to_datetime(date_str)
    ax2.axvline(date, linestyle='--', linewidth=2.5, alpha=0.7)

euro_date = pd.to_datetime('2023-01-01')
ax2.axvspan(euro_date, croatia_data['date'].max(), alpha=0.15,
           zorder=0)

ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Spread vs Germany (pp)', fontsize=12, fontweight='bold')
year_locator = mdates.YearLocator()
year_formatter = mdates.DateFormatter('%Y')
ax2.xaxis.set_major_locator(year_locator)
ax2.xaxis.set_major_formatter(year_formatter)
ax2.legend(loc='best')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig.autofmt_xdate()
save_with_layout(fig, '09_croatia_euro_timeline.png')


print("\n[8/N] Creating additional plots...")

fig, ax = plt.subplots(figsize=(12, 7))
for country in all_countries:
    country_data = df[df['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else 2.5
    ax.plot(country_data['date'], country_data['gdp_growth_quarterly'],
            label=country, linewidth=linewidth, color=colors[country], alpha=0.9)
gdp_events = [
    (pd.to_datetime('2020-03-01'), 'COVID-19 Shock\nMar 2020', '#b91c1c'),
    (pd.to_datetime('2023-01-01'), 'Euro Adoption\nJan 2023', '#0E7490'),
]
annotate_vertical_events(ax, gdp_events)
ax.axhline(0, color='#374151', linestyle='-', linewidth=1.5, alpha=0.6)
ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Quarterly GDP Growth (%)', fontweight='bold')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.grid(True, alpha=0.3, linewidth=0.6)
fig.autofmt_xdate()
save_with_layout(fig, '11_gdp_growth_comparison.png')


fig, ax = plt.subplots(figsize=(12, 7))
for country in all_countries:
    country_data = df[df['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else 2.5
    ax.plot(country_data['date'], country_data['public_debt_gdp'],
            label=country, linewidth=linewidth, color=colors[country], alpha=0.9)
ax.axhline(60, color='#DC143C', linestyle=':', linewidth=2.0,
          alpha=0.6, label='Maastricht Criterion (60%)')
debt_events = [
    (pd.to_datetime('2020-03-01'), 'COVID-19 Shock\nMar 2020', '#6b7280'),
    (pd.to_datetime('2023-01-01'), 'Euro Adoption\nJan 2023', '#0E7490'),
]
annotate_vertical_events(ax, debt_events)
ax.set_xlabel('Date', fontweight='bold')
ax.set_ylabel('Public Debt (% of GDP)', fontweight='bold')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.grid(True, alpha=0.3, linewidth=0.6)
fig.autofmt_xdate()
save_with_layout(fig, '12_public_debt_comparison.png')


print("\n[9/N] Creating comprehensive inflation timeline visualization...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))


for country in all_countries:
    country_data = df[df['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else 2.5
    ax1.plot(country_data['date'], country_data['inflation_hicp'],
            label=country, linewidth=linewidth, color=colors[country], alpha=0.95)
key_events_inflation = {
    : 'COVID-19\nPandemic',
    : 'Russia-Ukraine\nWar',
    : 'ECB First\nRate Hike',
    : 'Croatia\nEuro Adoption'
}
event_positions = [0.95, 0.82, 0.70, 0.88]
for i, (date_str, label) in enumerate(key_events_inflation.items()):
    date = pd.to_datetime(date_str)
    ax1.axvline(date, color='#374151', linestyle='--', alpha=0.5, linewidth=2.0)
    y_pos = ax1.get_ylim()[1] * event_positions[i % len(event_positions)]
    ax1.text(date, y_pos, label, rotation=0, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                     edgecolor='#374151', alpha=0.96, linewidth=1.8), fontweight='bold')
ax1.set_xlabel('Date', fontweight='bold')
ax1.set_ylabel('HICP Inflation (Annual Rate of Change, %)', fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.3, linewidth=0.6)
ax1.axhline(2, color='#374151', linestyle=':', linewidth=1.5, alpha=0.6, label='ECB Target (2%)')


df_recent = df[df['date'] >= '2021-01-01']
for country in all_countries:
    country_data = df_recent[df_recent['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else 2.5
    ax2.plot(country_data['date'], country_data['inflation_hicp'],
            label=country, linewidth=linewidth, color=colors[country], alpha=0.95)
ax2.axvspan(pd.to_datetime('2023-07-01'), pd.to_datetime('2024-12-31'),
           alpha=0.12, color='#15803D')
phase_annotations = []
txt1 = ax2.text(pd.to_datetime('2021-06-01'), 7.8, 'Phase 1:\nInflation\nShock', 
        fontsize=11, ha='center', fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                  edgecolor='#DC143C', alpha=0.96, linewidth=1.8))
phase_annotations.append(txt1)
txt2 = ax2.text(pd.to_datetime('2022-09-01'), 8.5, 'Phase 2:\nPeak\nInflation', 
        fontsize=11, ha='center', fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                  edgecolor='#8B0000', alpha=0.96, linewidth=1.8))
phase_annotations.append(txt2)
txt3 = ax2.text(pd.to_datetime('2024-03-01'), 3.5, 'Phase 3:\nNormalization', 
        fontsize=11, ha='center', fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                  edgecolor='#15803D', alpha=0.96, linewidth=1.8))
phase_annotations.append(txt3)
try:
    adjust_text(phase_annotations,
               ax=ax2,
               only_move={'texts': 'y'},
               expand_text=(1.2, 1.3),
               force_text=(0.5, 0.7),
               lim=1000)
except Exception:
    pass
ax2.set_xlabel('Date', fontweight='bold')
ax2.set_ylabel('HICP Inflation (%)', fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.3, linewidth=0.6)
ax2.axhline(2, color='#374151', linestyle=':', linewidth=1.5, alpha=0.6)
plt.tight_layout()
save_with_layout(fig, '15_inflation_complete_timeline.png')


print("\n[10/N] Creating DiD plots...")


df_h1 = df[
    (df['country'].isin(['Croatia', 'Slovenia', 'Slovakia', 'Lithuania'])) &
    (df['date'] >= '2021-01-01') &
    (df['date'] <= '2024-12-31')
].copy()
df_h1['post_july_2022_hike'] = (df_h1['date'] >= '2022-07-27').astype(int)
df_h1['is_croatia'] = (df_h1['country'] == 'Croatia').astype(int)
df_h1['croatia_ex_post'] = df_h1['is_croatia'] * df_h1['post_july_2022_hike']

model4 = smf.ols(
    
    ,
    data=df_h1
).fit(cov_type='HC3')
did_coef = model4.params['croatia_ex_post']

avg_yields = (
    df_h1.groupby(['is_croatia', 'post_july_2022_hike'])['bond_yield_10y']
    .agg(mean='mean', std='std', n='count')
    .reset_index()
)
avg_yields['sem'] = avg_yields['std'] / np.sqrt(avg_yields['n'].clip(lower=1))
avg_yields['ci95'] = 1.96 * avg_yields['sem']
avg_yields = avg_yields.set_index(['is_croatia', 'post_july_2022_hike'])

fig, ax = plt.subplots(figsize=(12, 8))
periods = [0, 1]
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
    alpha=0.32,
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
    alpha=0.32,
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
save_with_layout(fig, 'h1_did_visualization.png')


print("\n" + "=" * 80)
print("ALL PLOTS REGENERATION COMPLETE")
print("=" * 80)
