"""
H1 Appendix: HAC (Newey-West) standard errors, VIF, and F-test
- Re-run H1 main full specification with HAC and report diagnostics.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import statsmodels.formula.api as smf

from io_utils import write_text
from stats_utils import build_vif_table

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / 'DATA' / 'input_data.csv'


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    dfh = df[(df['country'].isin(['Croatia', 'Slovenia', 'Slovakia', 'Lithuania'])) &
             (df['date'] >= '2021-01-01') & (df['date'] <= '2024-12-31')].copy()

    dfh.rename(columns={'croatia_x_post_july2022': 'croatia_ex_post'}, inplace=True)

    formula = (
        'bond_yield_10y ~ C(country) + post_july_2022_hike + croatia_ex_post + '
        'gdp_growth_quarterly + inflation_hicp + public_debt_gdp'
    )
    model_hac = smf.ols(formula, data=dfh).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    coef = 'croatia_ex_post'
    did = (model_hac.params.get(coef), model_hac.bse.get(coef), model_hac.pvalues.get(coef))

    try:
        ftest = model_hac.f_test('gdp_growth_quarterly = inflation_hicp = public_debt_gdp = 0')
        fstr = f'F={float(ftest.fvalue):.3f}, p={float(ftest.pvalue):.4f}'
    except Exception:
        fstr = 'N/A'

    vif = build_vif_table(dfh, ['gdp_growth_quarterly', 'inflation_hicp', 'public_debt_gdp'])

    lines = [
        '=' * 80,
        'H1 Appendix: HAC (Newey-West) SE, VIF and F-test',
        '=' * 80,
        '',
        'HAC (maxlags=5) - Main DiD coef (country FE + controls)',
        f"  {coef}: coef={did[0]:.4f}, SE={did[1]:.4f}, p={did[2]:.4f}",
        '',
        f'Controls joint significance: {fstr}',
        '',
        'VIF (controls):',
    ]
    for _, row in vif.iterrows():
        try:
            lines.append(f"  {row['Variable']}: VIF={float(row['VIF']):.2f}")
        except Exception:
            lines.append(f"  {row['Variable']}: VIF=N/A")

    lines.append('')
    write_text('h1_hac_results.txt', '\n'.join(lines) + '\n')

print('Saved: analysis/output/raw_outputs/h1_hac_results.txt')


if __name__ == '__main__':
    run()
