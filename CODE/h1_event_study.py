"""
H1 Event Study: Immediate yield reactions around ECB events
- Events: 2022-07-27 (first hike), 2023-02-02 (second hike)
- Windows: +/-5 days and +/-3 days
- Method: Abnormal yield = yield - 30d pre-event average; one-sample t-test vs 0
- Countries: Croatia, Slovenia, Slovakia, Lithuania, France, Germany
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from io_utils import save_figure, write_text
from plot_utils import make_subplots, place_legend

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

COUNTRIES = ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'France', 'Germany']
EVENTS = [
    ('2022-07-27', 'ECB Hike 2022-07-27'),
    ('2023-02-02', 'ECB Hike 2023-02-02'),
]


def abnormal_series(df: pd.DataFrame, country: str, event_date: pd.Timestamp):
    c = df[df['country'] == country].sort_values('date').set_index('date')
    if c.empty:
        return None
    pre_window = c.loc[(c.index < event_date) & (c.index >= event_date - pd.Timedelta(days=30))]
    if pre_window.empty:
        return None
    baseline = pre_window['bond_yield_10y'].mean()
    # Build window +/-5 and +/-3
    s5 = c.loc[(c.index >= event_date - pd.Timedelta(days=5)) & (c.index <= event_date + pd.Timedelta(days=5))]['bond_yield_10y'] - baseline
    s3 = c.loc[(c.index >= event_date - pd.Timedelta(days=3)) & (c.index <= event_date + pd.Timedelta(days=3))]['bond_yield_10y'] - baseline
    return s5.dropna(), s3.dropna()


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'DATA' / 'input_data.csv'


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df[df['country'].isin(COUNTRIES)].copy()
    

    lines = [
        '=' * 80,
        'H1 EVENT STUDY: +/-5d and +/-3d around ECB hikes (abnormal yields)',
        '=' * 80,
        '',
        'Abnormal yield = yield - mean(yield over t-30..t-1). One-sample t-test vs 0.',
        'Countries: HR, SI, SK, LT, FR, DE.',
        ''
    ]

    plot_data5 = {title: [] for _, title in EVENTS}
    plot_data3 = {title: [] for _, title in EVENTS}

    for event_date_str, event_title in EVENTS:
        event_date = pd.to_datetime(event_date_str)
        lines.append(f'Event: {event_title}')
        for country in COUNTRIES:
            series = abnormal_series(df, country, event_date)
            if series is None:
                lines.append(f'  {country}: no data')
                plot_data5[event_title].append(np.nan)
                plot_data3[event_title].append(np.nan)
                continue

            s5, s3 = series
            t5 = stats.ttest_1samp(s5.values, 0.0, nan_policy='omit')
            t3 = stats.ttest_1samp(s3.values, 0.0, nan_policy='omit')
            lines.append(
                f"  {country}: +/-5d mean={s5.mean():.4f}, t={t5.statistic:.2f}, p={t5.pvalue:.4f}; "
                f"+/-3d mean={s3.mean():.4f}, t={t3.statistic:.2f}, p={t3.pvalue:.4f}"
            )
            plot_data5[event_title].append(float(s5.mean()))
            plot_data3[event_title].append(float(s3.mean()))
        lines.append('')

    write_text('h1_event_study_results.txt', '\n'.join(lines) + '\n')

    x = np.arange(len(COUNTRIES))
    width = 0.35
    palette = ['#0F6CE0', '#103E75']

    def _plot_event_window(values_dict, window_label: str, filename: str) -> None:
        fig, ax = make_subplots(figsize=(10, 6))
        left_values = np.array(values_dict[EVENTS[0][1]], dtype=float)
        right_values = np.array(values_dict[EVENTS[1][1]], dtype=float)

        bars_left = ax.bar(
            x - width / 2,
            left_values,
            width,
            label=EVENTS[0][1],
            color=palette[0],
            alpha=0.85,
            edgecolor='white',
            linewidth=1.0,
        )
        bars_right = ax.bar(
            x + width / 2,
            right_values,
            width,
            label=EVENTS[1][1],
            color=palette[1],
            alpha=0.85,
            edgecolor='white',
            linewidth=1.0,
        )

        ax.axhline(0, color='#666666', linewidth=1.2, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in COUNTRIES], rotation=0, fontweight='bold')
        ax.set_ylabel(f'Mean abnormal yield (pp), {window_label}', fontweight='bold')
        ax.margins(x=0.02, y=0.15)

        ax.bar_label(bars_left, fmt='%.3f', padding=6, fontsize=11, color='white')
        ax.bar_label(bars_right, fmt='%.3f', padding=6, fontsize=11, color='white')

        combined = np.concatenate([left_values, right_values])
        finite = combined[np.isfinite(combined)]
        if finite.size == 0:
            ymin, ymax = -0.1, 0.1
        else:
            ymin, ymax = float(finite.min()), float(finite.max())
        span = max(0.1, ymax - ymin)
        ax.set_ylim(ymin - span * 0.25, ymax + span * 0.35)

        place_legend(ax, ncol=2)
        save_figure(fig, filename, dpi=300)

    _plot_event_window(plot_data5, '+/-5 days', 'h1_event_study_window5.png')
    _plot_event_window(plot_data3, '+/-3 days', 'h1_event_study_window3.png')

if __name__ == '__main__':
    run()
