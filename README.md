# Impact of ECB Monetary Policy and Euro Adoption on Croatian Bond Yields

Empirical analysis of the effects of ECB rate hikes and Croatia's euro adoption on sovereign bond yields, using Difference-in-Differences (DiD) models, event studies, placebo tests, and robust inference.

## Hypotheses

- **H1**: The ECB's first rate hike on 27 July 2022 had a statistically significant differential effect on Croatian 10-year bond yields relative to comparable euro area peers
- **H1b**: The ECB's rate hike on 2 February 2023 had a differential effect on Croatian yields after euro adoption
- **H2**: Croatia's euro adoption on 1 January 2023 led to convergence of Croatian-German yield spreads

## Methodology

- Difference-in-Differences estimation with baseline and country fixed-effects specifications
- HC3 and HAC (Newey-West) robust standard errors
- Parallel trends diagnostics and placebo tests
- Event-study analysis around key policy dates
- Robustness checks with alternative control-group compositions and time windows

## Structure

```text
CODE/           Minimal analysis scripts
DATA/           Bond yield, macroeconomic, and ECB event datasets
OUTPUTS/
  figures/      Plots and visualizations
  tables/       LaTeX tables
  reports/      Text summaries of statistical analyses
```

## Usage

```bash
pip install pandas numpy statsmodels matplotlib seaborn scipy requests adjustText
python CODE/run_all.py
```

`run_all.py` regenerates the paper tables, paper figures, regression reports, placebo outputs, event studies, HAC/VIF appendix outputs, and comparison tests.

The paper-specific artefacts are organized through `paper_tables.py` and `paper_figures.py`, and both are invoked from `run_all.py`.

## Data Sources

- **Bond yields**: Investing.com (10-year government bond yields)
- **Macroeconomic indicators**: Eurostat (GDP growth, HICP inflation, public debt)
- **ECB events**: European Central Bank policy rate decisions

## Analysis Panels

- **Core H1/H2 panel**: Croatia, Slovenia, Slovakia, Lithuania
- **H1b extension**: Croatia, Slovenia, Slovakia, Lithuania, France, Germany
- **Latvia**: excluded from the final panel due to incomplete Investing.com bond-yield data
