"""
Regenerate key data plots with professional colors and collision-free text
Focuses on the main time series and comparison plots for thesis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
import matplotlib.dates as mdates

from plot_utils import make_subplots, add_annotation, place_legend

print("=" * 80)
print("REGENERATING DATA PLOTS WITH PROFESSIONAL COLORS")
print("=" * 80)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'OUTPUTS' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/7] Loading data...")
DATA_DIR = PROJECT_ROOT / 'DATA'
df = pd.read_csv(DATA_DIR / 'input_data.csv', parse_dates=['date'])
print(f"[info] Loaded {len(df):,} observations")

KEY_EVENT_LINES = [
    (pd.to_datetime('2022-07-27'), 'ECB Rate Hike\n27 Jul 2022', '#9A3412'),
    (pd.to_datetime('2023-01-01'), 'Euro Adoption\n1 Jan 2023', '#0E7490'),
]


def _render_event_labels(
    ax,
    events,
    *,
    threshold_days: int = 210,
    base_frac: float = 1.01,
    step_frac: float = 0.18,
    anchor_frac: float = 0.88,
    max_frac: float = 1.25,
):
    """Render vertical event markers with staggered, non-overlapping labels."""
    if not events:
        return

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    anchor_y = ymin + span * anchor_frac
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    threshold = pd.Timedelta(days=threshold_days)

    last_at_level: dict[int, pd.Timestamp] = {}
    sorted_events = [
        (pd.to_datetime(ts), label, color) for ts, label, color in events
    ]
    sorted_events.sort(key=lambda item: item[0])

    # Compute horizontal nudges to separate close labels
    nudge_days = {}
    for i, (ts, label, color) in enumerate(sorted_events):
        nudge_days[i] = 0
        for j in range(i):
            ts_j = sorted_events[j][0]
            if abs((ts - ts_j).days) < threshold_days:
                nudge_days[j] = -60  # push earlier label left
                nudge_days[i] = 60   # push later label right

    for i, (ts, label, color) in enumerate(sorted_events):
        ax.axvline(ts, linestyle='--', linewidth=2.3, alpha=0.75, color=color, zorder=1.5)

        level = 0
        while level in last_at_level and (ts - last_at_level[level]) <= threshold:
            level += 1
        last_at_level[level] = ts

        y_frac = min(base_frac + level * step_frac, max_frac)
        text_x = ts + pd.Timedelta(days=nudge_days.get(i, 0))
        ax.annotate(
            label,
            xy=(ts, anchor_y),
            xycoords='data',
            xytext=(text_x, y_frac),
            textcoords=text_transform,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=color, linewidth=1.15, alpha=0.97),
            arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0, shrinkA=0, shrinkB=6),
            fontsize=9,
            fontweight='bold',
            annotation_clip=False,
        )


def annotate_vertical_events(ax, events=KEY_EVENT_LINES):
    _render_event_labels(ax, events)



# ============================================================
# PLOT 1: All Countries Yields
# ============================================================
print("\n[2/7] Creating all countries yields plot...")

fig, ax = make_subplots()

for country in ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'France', 'Germany']:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else (2.5 if country == 'Germany' else 2)
    alpha = 1.0 if country in ['Croatia', 'Germany'] else 0.75
    ax.plot(data['date'], data['bond_yield_10y'],
            label=country, linewidth=linewidth, alpha=alpha)

# Key events
annotate_vertical_events(ax)

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('10-Year Government Bond Yield (%)', fontsize=13, fontweight='bold')

place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(FIGURES_DIR / '01_all_countries_yields.png', dpi=300, facecolor='white')
print("[saved] ../plots/01_all_countries_yields.png")
plt.close(fig)

# ============================================================
# PLOT 2: Croatia vs Small Eurozone
# ============================================================
print("\n[3/7] Creating Croatia vs small eurozone comparison...")

fig, ax = make_subplots()

for country in ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania']:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 4 if country == 'Croatia' else 2.5
    alpha = 1.0 if country == 'Croatia' else 0.8
    ax.plot(data['date'], data['bond_yield_10y'],
            label=country, linewidth=linewidth, alpha=alpha)

# Shade post-euro adoption period
euro_date = pd.to_datetime('2023-01-01')
ax.axvspan(euro_date, df['date'].max(), alpha=0.1, zorder=0)

# Key events
annotate_vertical_events(ax)

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('10-Year Government Bond Yield (%)', fontsize=13, fontweight='bold')

place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(FIGURES_DIR / '02_croatia_vs_small_eurozone.png', dpi=300, facecolor='white')
print("[saved] ../plots/02_croatia_vs_small_eurozone.png")
plt.close(fig)

# ============================================================
# PLOT 3: Spreads vs Germany
# ============================================================
print("\n[4/7] Creating spreads vs Germany plot...")

fig, ax = make_subplots()

for country in ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'France']:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 4 if country == 'Croatia' else (3 if country == 'France' else 2.5)
    alpha = 1.0 if country in ['Croatia', 'France'] else 0.75
    ax.plot(data['date'], data['spread_vs_germany'],
            label=country, linewidth=linewidth, alpha=alpha)

# Shade post-euro adoption period
euro_date = pd.to_datetime('2023-01-01')
ax.axvspan(euro_date, df['date'].max(), alpha=0.1, zorder=0)

# Reference line
ax.axhline(0, linestyle='-', linewidth=1.5, alpha=0.5)

# Key events
annotate_vertical_events(ax)

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Spread vs German Bund (percentage points)', fontsize=13, fontweight='bold')

place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(FIGURES_DIR / '03_spreads_vs_germany.png', dpi=300, facecolor='white')
print("[saved] ../plots/03_spreads_vs_germany.png")
plt.close(fig)

# ============================================================
# PLOT 4: Correlation Heatmap
# ============================================================
print("\n[5/7] Creating correlation heatmap...")

# Pivot data for correlation
pivot_df = df.pivot_table(
    values='bond_yield_10y',
    index='date',
    columns='country'
)

correlation_matrix = pivot_df.corr()

fig, ax = make_subplots()

# Professional diverging colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap=cmap,
            center=0.9, vmin=0.8, vmax=1.0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=ax)

plt.setp(ax.get_xticklabels(), fontsize=11, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontsize=11, fontweight='bold', rotation=0)

fig.savefig(FIGURES_DIR / '05_correlation_heatmap.png', dpi=300, facecolor='white')
print("[saved] ../plots/05_correlation_heatmap.png")
plt.close(fig)

# ============================================================
# PLOT 5: Volatility Comparison
# ============================================================
print("\n[6/7] Creating volatility comparison plot...")
fig, ax = make_subplots()

for country in ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'France', 'Germany']:
    data = df[df['country'] == country].sort_values('date')
    linewidth = 3.5 if country == 'Croatia' else (2.5 if country == 'Germany' else 2)
    alpha = 1.0 if country in ['Croatia', 'Germany'] else 0.75
    ax.plot(data['date'], data['yield_std_30d'],
            label=country, linewidth=linewidth, alpha=alpha)

# Key events
annotate_vertical_events(ax)

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('30-Day Rolling Standard Deviation', fontsize=13, fontweight='bold')

place_legend(ax, location='upper center', anchor=(0.5, -0.18), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(FIGURES_DIR / '06_volatility_comparison.png', dpi=300, facecolor='white')
print("[saved] ../plots/06_volatility_comparison.png")
plt.close(fig)

# ============================================================
# PLOT 6: Croatia Euro Timeline
# ============================================================
print("\n[7/7] Creating Croatia euro adoption timeline...")

croatia_data = df[df['country'] == 'Croatia'].sort_values('date')

fig, axes = make_subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
ax1, ax2 = axes

# Top panel: Bond yields
ax1.plot(croatia_data['date'], croatia_data['bond_yield_10y'],
         linewidth=3.5, label='Croatia 10Y Yield')

# Germany for reference
germany_data = df[df['country'] == 'Germany'].sort_values('date')
ax1.plot(germany_data['date'], germany_data['bond_yield_10y'],
         linewidth=2.5, alpha=0.7,
         linestyle='--', label='Germany 10Y Yield (Reference)')

# Key events with better positioning
events = [
    ('2018-07-10', 'Maastricht\nProcess\nBegins'),
    ('2022-07-12', 'EU Council\nApproves\nEuro Entry'),
    ('2023-01-01', 'EURO\nADOPTION'),
]

for date_str, label in events:
    date = pd.to_datetime(date_str)
    ax1.axvline(date, linestyle='--', linewidth=2.5, alpha=0.7)

# Add event labels anchored to their vertical lines
ylim = ax1.get_ylim()
ymin, ymax = ylim
vertical_margin = (ymax - ymin) * 0.03

for i, (date_str, label) in enumerate(events):
    date = pd.to_datetime(date_str)
    series = croatia_data.set_index('date')['bond_yield_10y'].sort_index()
    y_pos = series.asof(date)
    if pd.isna(y_pos):
        y_pos = series.iloc[-1]
    offset = 20
    valign = 'bottom'
    if (y_pos + vertical_margin) >= ymax:
        offset = -30
        valign = 'top'
    add_annotation(
        ax1,
        label,
        (date, y_pos),
        xytext=(0, offset),
        textcoords='offset points',
        ha='center',
        va=valign,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='aliceblue', edgecolor='black', linewidth=1, alpha=0.95),
        arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2, shrinkA=0, shrinkB=6),
        fontsize=10,
        fontweight='bold'
    )

ax1.set_ylabel('10-Year Bond Yield (%)', fontsize=12, fontweight='bold')

ax1.legend(loc='best')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticklabels([])

# Bottom panel: Spreads
ax2.plot(croatia_data['date'], croatia_data['spread_vs_germany'],
         linewidth=3.5, label='Croatia Spread vs Germany')
ax2.axhline(0, linestyle='-', linewidth=1.5, alpha=0.5)

# Same events
for date_str, label in events:
    date = pd.to_datetime(date_str)
    ax2.axvline(date, linestyle='--', linewidth=2.5, alpha=0.7)

# Shade post-adoption
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

fig.savefig(FIGURES_DIR / '09_croatia_euro_timeline.png', dpi=300, facecolor='white')
print('[saved] ../plots/09_croatia_euro_timeline.png')
plt.close(fig)

print("\n" + "=" * 80)
print("DATA PLOT REGENERATION COMPLETE")
print("=" * 80)
print("\nRegenerated plots with professional colors:")
print("  1. All countries yields")
print("  2. Croatia vs small eurozone")
print("  3. Spreads vs Germany")
print("  4. Correlation heatmap")
print("  5. Volatility comparison")
print("  6. Croatia euro timeline")
print("\nAll plots now feature:")
print("  - Professional color palette")
print("  - Collision-aware text placement")
print("  - Enhanced visual hierarchy")
print("  - Publication-ready resolution (300 DPI)")
print("=" * 80)
