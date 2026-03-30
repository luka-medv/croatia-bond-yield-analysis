"""
H1 Event Study: Immediate yield reactions around ECB events
- Events: 2022-07-27 (first hike), 2023-02-02 (second hike)
- Windows: +/-5 days and +/-3 days
- Method: Abnormal yield = yield - 30d pre-event average; one-sample t-test vs 0
- Countries: Croatia, Slovenia, Slovakia, Lithuania, France, Germany
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from io_utils import write_text

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

    table_rows = []

    for event_date_str, event_title in EVENTS:
        event_date = pd.to_datetime(event_date_str)
        lines.append(f'Event: {event_title}')
        for country in COUNTRIES:
            series = abnormal_series(df, country, event_date)
            if series is None:
                lines.append(f'  {country}: no data')
                table_rows.append({
                    "Event": event_title,
                    "Country": country,
                    "Mean +/-5d": np.nan,
                    "t-stat +/-5d": np.nan,
                    "p-value +/-5d": np.nan,
                    "Mean +/-3d": np.nan,
                    "t-stat +/-3d": np.nan,
                    "p-value +/-3d": np.nan,
                })
                continue

            s5, s3 = series
            t5 = stats.ttest_1samp(s5.values, 0.0, nan_policy='omit')
            t3 = stats.ttest_1samp(s3.values, 0.0, nan_policy='omit')
            lines.append(
                f"  {country}: +/-5d mean={s5.mean():.4f}, t={t5.statistic:.2f}, p={t5.pvalue:.4f}; "
                f"+/-3d mean={s3.mean():.4f}, t={t3.statistic:.2f}, p={t3.pvalue:.4f}"
            )
            table_rows.append({
                "Event": event_title,
                "Country": country,
                "Mean +/-5d": float(s5.mean()),
                "t-stat +/-5d": float(t5.statistic),
                "p-value +/-5d": float(t5.pvalue),
                "Mean +/-3d": float(s3.mean()),
                "t-stat +/-3d": float(t3.statistic),
                "p-value +/-3d": float(t3.pvalue),
            })
        lines.append('')

    write_text('h1_event_study_results.txt', '\n'.join(lines) + '\n')
    table_df = pd.DataFrame(table_rows)
    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="llrrrrrr",
        float_format=lambda x: f"{x:.4f}",
    )
    latex = latex.replace("\\toprule", "\\hline")
    latex = latex.replace("\\midrule", "\\hline")
    latex = latex.replace("\\bottomrule", "\\hline")
    write_text("table_4_4_h1_event_study.tex", latex)

if __name__ == '__main__':
    run()
