# Impact of ECB Monetary Policy and Euro Adoption on Croatian Bond Yields

Code, data, and outputs for the empirical analysis in the paper.

## Structure

```text
CODE/           H1.py, H2.py, tables.py, plots.py
DATA/           Analytical dataset and descriptive CSV exports
OUTPUTS/
  figures/      Figures
  raw_outputs/  Text outputs used for tables and result checks
```

## Usage

```bash
pip install pandas numpy statsmodels matplotlib scipy
python CODE/tables.py
python CODE/plots.py
```

Running `CODE/tables.py` writes:
- descriptive CSV exports to `DATA/`
- text outputs to `OUTPUTS/raw_outputs/`

Running `CODE/plots.py` writes:
- figures to `OUTPUTS/figures/`

## Main Files

- `CODE/H1.py`: H1, H1b, H1 placebo, H1 event study, H1 HAC
- `CODE/H2.py`: H2, H2 placebo, H2 event study, H2 HAC
- `CODE/tables.py`: descriptive CSV exports and textual raw outputs
- `CODE/plots.py`: final figure files
- `DATA/input_data.csv`: analytical panel used in the regressions
- `DATA/descriptive_stats_*.csv`: descriptive exports
- `OUTPUTS/raw_outputs/*.txt`: regression, placebo, robustness, event-study, and HAC outputs
- `OUTPUTS/figures/*.png`: figure files

## Data Sources

- Bond yields: Investing.com
- Macroeconomic indicators: Eurostat
- ECB events: European Central Bank policy rate decisions
