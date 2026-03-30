# Impact of ECB Monetary Policy and Euro Adoption on Croatian Bond Yields

Empirical analysis examining the effects of ECB interest rate hikes and Croatia's euro adoption on sovereign bond yields, using Difference-in-Differences (DiD) methodology with robust inference.

## Hypotheses

- **H1**: The ECB's July 2022 rate hike significantly increased Croatian 10-year bond yields relative to comparable eurozone peers
- **H1b**: The February 2023 rate hike had a differential impact on Croatian yields post-euro adoption
- **H2**: Croatia's euro adoption (January 2023) led to convergence of yield spreads toward Germany

## Methodology

- Difference-in-Differences estimation with country fixed effects
- HAC (Newey-West) and HC3 robust standard errors
- Parallel trends testing and placebo tests
- Event study analysis around key policy dates
- Robustness checks with alternative control groups

## Structure

```
CODE/           Python analysis scripts (run_all.py is the entry point)
DATA/           Bond yield, macroeconomic, and ECB event datasets
OUTPUTS/
  figures/      Publication-quality plots and visualizations
  tables/       LaTeX tables for regression results
  reports/      Text summaries of statistical analyses
```

## Usage

```bash
pip install pandas numpy statsmodels matplotlib scipy
python CODE/run_all.py
```

## Data Sources

- **Bond yields**: Investing.com (10-year government bond yields)
- **Macroeconomic indicators**: Eurostat (GDP growth, HICP inflation, public debt)
- **ECB events**: European Central Bank policy rate decisions

## Countries

Treatment: Croatia | Control: Slovenia, Slovakia, Lithuania, Latvia, France, Germany
