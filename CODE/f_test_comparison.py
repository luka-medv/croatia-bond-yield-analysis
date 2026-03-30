

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import statsmodels.formula.api as smf

from io_utils import write_text

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "DATA" / "input_data.csv"


def f_test_controls(model, controls):
    constraint = " = ".join(controls) + " = 0"
    result = model.f_test(constraint)
    return float(result.fvalue), float(result.pvalue)


def run() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    controls = ["gdp_growth_quarterly", "inflation_hicp", "public_debt_gdp"]
    lines = []

    
    countries_h1 = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
    dfh1 = df[
        (df["country"].isin(countries_h1))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    dfh1.rename(columns={"croatia_x_post_july2022": "croatia_ex_post"}, inplace=True)

    formula_h1 = (
        
        
    )

    m_hc3 = smf.ols(formula_h1, data=dfh1).fit(cov_type="HC3")
    m_hac = smf.ols(formula_h1, data=dfh1).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    m_ols = smf.ols(formula_h1, data=dfh1).fit()

    lines.append("=" * 70)
    lines.append("F-TEST COMPARISON: MACRO CONTROLS JOINT SIGNIFICANCE")
    lines.append("Constraint: gdp_growth_quarterly = inflation_hicp = public_debt_gdp = 0")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"H1: ECB rate hike (27 July 2022)  N={len(dfh1)}")
    lines.append(f"  HC3:  F={f_test_controls(m_hc3, controls)[0]:>10.3f}  p={f_test_controls(m_hc3, controls)[1]:.6f}")
    lines.append(f"  HAC:  F={f_test_controls(m_hac, controls)[0]:>10.3f}  p={f_test_controls(m_hac, controls)[1]:.6f}")
    lines.append(f"  OLS:  F={f_test_controls(m_ols, controls)[0]:>10.3f}  p={f_test_controls(m_ols, controls)[1]:.6f}")

    print(f"H1  HC3: F={f_test_controls(m_hc3, controls)[0]:.3f}")
    print(f"H1  HAC: F={f_test_controls(m_hac, controls)[0]:.3f}")
    print(f"H1  OLS: F={f_test_controls(m_ols, controls)[0]:.3f}")

    
    countries_h1b = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
    dfh1b = df[
        (df["country"].isin(countries_h1b))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    dfh1b.rename(columns={"croatia_x_post_feb2023": "croatia_ex_post"}, inplace=True)

    formula_h1b = (
        
        
    )

    m_hc3b = smf.ols(formula_h1b, data=dfh1b).fit(cov_type="HC3")
    m_hacb = smf.ols(formula_h1b, data=dfh1b).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    m_olsb = smf.ols(formula_h1b, data=dfh1b).fit()

    lines.append("")
    lines.append(f"H1b: ECB rate hike (2 Feb 2023)  N={len(dfh1b)}")
    lines.append(f"  HC3:  F={f_test_controls(m_hc3b, controls)[0]:>10.3f}  p={f_test_controls(m_hc3b, controls)[1]:.6f}")
    lines.append(f"  HAC:  F={f_test_controls(m_hacb, controls)[0]:>10.3f}  p={f_test_controls(m_hacb, controls)[1]:.6f}")
    lines.append(f"  OLS:  F={f_test_controls(m_olsb, controls)[0]:>10.3f}  p={f_test_controls(m_olsb, controls)[1]:.6f}")

    print(f"H1b HC3: F={f_test_controls(m_hc3b, controls)[0]:.3f}")
    print(f"H1b HAC: F={f_test_controls(m_hacb, controls)[0]:.3f}")

    
    countries_h2 = ["Croatia", "Slovenia", "Slovakia", "Lithuania"]
    dfh2 = df[
        (df["country"].isin(countries_h2))
        & (df["date"] >= "2021-01-01")
        & (df["date"] <= "2024-12-31")
    ].copy()
    dfh2.rename(columns={"croatia_x_post_euro": "croatia_ex_post"}, inplace=True)

    formula_h2 = (
        
        
    )

    m_hc3h2 = smf.ols(formula_h2, data=dfh2).fit(cov_type="HC3")
    m_hach2 = smf.ols(formula_h2, data=dfh2).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    m_olsh2 = smf.ols(formula_h2, data=dfh2).fit()

    lines.append("")
    lines.append(f"H2: Euro adoption spread (1 Jan 2023)  N={len(dfh2)}")
    lines.append(f"  HC3:  F={f_test_controls(m_hc3h2, controls)[0]:>10.3f}  p={f_test_controls(m_hc3h2, controls)[1]:.6f}")
    lines.append(f"  HAC:  F={f_test_controls(m_hach2, controls)[0]:>10.3f}  p={f_test_controls(m_hach2, controls)[1]:.6f}")
    lines.append(f"  OLS:  F={f_test_controls(m_olsh2, controls)[0]:>10.3f}  p={f_test_controls(m_olsh2, controls)[1]:.6f}")

    print(f"H2  HC3: F={f_test_controls(m_hc3h2, controls)[0]:.3f}")
    print(f"H2  HAC: F={f_test_controls(m_hach2, controls)[0]:.3f}")

    
    lines.append("")
    lines.append("-" * 70)
    lines.append("NOTE: Document para [107] says 'under HC3 inference (F = 110.570)'")
    lines.append("but F=110.570 matches the HAC estimator, not HC3.")
    lines.append("HC3 F-test for H1 gives F=630.812.")
    lines.append("")

    write_text("f_test_comparison_results.txt", "\n".join(lines) + "\n")
    print("\nSaved: f_test_comparison_results.txt")


if __name__ == "__main__":
    run()
