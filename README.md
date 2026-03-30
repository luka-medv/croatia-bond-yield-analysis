# Impact of ECB Monetary Policy and Euro Adoption on Croatian Bond Yields

Code, data, and outputs for the empirical analysis in the paper.

## Structure

```text
CODE/           Analysis scripts
DATA/           Analytical dataset and descriptive CSV exports
OUTPUTS/
  figures/      Figures
  raw_outputs/  Text outputs used for tables and result checks
```

## Usage

```bash
pip install pandas numpy statsmodels matplotlib seaborn scipy
python CODE/run_all.py
```

Running `CODE/run_all.py` writes:
- descriptive CSV exports to `DATA/`
- text outputs to `OUTPUTS/raw_outputs/`
- figures to `OUTPUTS/figures/`

`adjustText` is optional. If installed, it is used only for figure label placement where needed.

## Main Files

- `CODE/run_all.py`: main entry point
- `DATA/input_data.csv`: analytical panel used in the regressions
- `DATA/descriptive_stats_*.csv`: descriptive exports
- `OUTPUTS/raw_outputs/*.txt`: regression, placebo, robustness, event-study, and HAC outputs
- `OUTPUTS/figures/*.png`: figure files

## Data Sources

- Bond yields: Investing.com
- Macroeconomic indicators: Eurostat
- ECB events: European Central Bank policy rate decisions
