

import sys
from pathlib import Path

import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw_data"

def load_raw_data():
    
    print("=" * 60)
    print("MERGING DATA INTO INPUT_DATA.CSV")
    print("=" * 60)

    print("\nLoading raw data files...")

    
    try:
        df_bonds = pd.read_csv(RAW_DIR / 'bond_yields.csv', parse_dates=['date'])
        print(f"  Bond yields: {len(df_bonds)} records")
    except FileNotFoundError:
        print("  bond_yields.csv not found!")
        df_bonds = pd.DataFrame()

    try:
        df_gdp = pd.read_csv(RAW_DIR / 'gdp_growth.csv', parse_dates=['date'])
        print(f"  GDP growth: {len(df_gdp)} records")
    except FileNotFoundError:
        print("  gdp_growth.csv not found!")
        df_gdp = pd.DataFrame()

    try:
        df_inflation = pd.read_csv(RAW_DIR / 'inflation_hicp.csv', parse_dates=['date'])
        print(f"  Inflation: {len(df_inflation)} records")
    except FileNotFoundError:
        print("  inflation_hicp.csv not found!")
        df_inflation = pd.DataFrame()

    try:
        df_debt = pd.read_csv(RAW_DIR / 'public_debt.csv', parse_dates=['date'])
        print(f"  Public debt: {len(df_debt)} records")
    except FileNotFoundError:
        print("  public_debt.csv not found!")
        df_debt = pd.DataFrame()

    return df_bonds, df_gdp, df_inflation, df_debt


def merge_datasets(df_bonds, df_gdp, df_inflation, df_debt):
    
    print("\nMerging datasets...")

    if df_bonds.empty:
        print("  Cannot merge: bond yields data is missing!")
        return pd.DataFrame()

    
    df_merged = df_bonds.copy()

    
    weekend_mask = df_merged['date'].dt.dayofweek >= 5
    n_weekends = weekend_mask.sum()
    df_merged = df_merged[~weekend_mask].copy()
    print(f"  Base dataset (bonds): {len(df_merged)} records ({n_weekends} weekend rows removed)")

    
    df_merged['year'] = df_merged['date'].dt.year
    df_merged['month'] = df_merged['date'].dt.month
    df_merged['quarter'] = df_merged['date'].dt.quarter

    
    if not df_gdp.empty:
        df_gdp['year'] = df_gdp['date'].dt.year
        df_gdp['quarter'] = df_gdp['date'].dt.quarter

        df_merged = df_merged.merge(
            df_gdp[['country', 'year', 'quarter', 'gdp_growth_quarterly']],
            on=['country', 'year', 'quarter'],
            how='left'
        )
        print(f"  ✓ Merged GDP growth (quarterly)")

    
    if not df_inflation.empty:
        df_inflation['year'] = df_inflation['date'].dt.year
        df_inflation['month'] = df_inflation['date'].dt.month

        df_merged = df_merged.merge(
            df_inflation[['country', 'year', 'month', 'inflation_hicp']],
            on=['country', 'year', 'month'],
            how='left'
        )
        print(f"  ✓ Merged inflation (monthly)")

    
    if not df_debt.empty:
        df_debt['year'] = df_debt['date'].dt.year

        df_merged = df_merged.merge(
            df_debt[['country', 'year', 'public_debt_gdp']],
            on=['country', 'year'],
            how='left'
        )
        print(f"  ✓ Merged public debt (annual)")

    
    df_merged = df_merged.sort_values(['country', 'date'])

    for var in ['gdp_growth_quarterly', 'inflation_hicp', 'public_debt_gdp']:
        if var in df_merged.columns:
            df_merged[var] = df_merged.groupby('country')[var].ffill()

    
    df_merged['is_croatia'] = (df_merged['country'] == 'Croatia').astype(int)

    
    small_eurozone = ['Croatia', 'Slovenia', 'Slovakia', 'Lithuania']
    df_merged['is_small_eurozone'] = df_merged['country'].isin(small_eurozone).astype(int)

    
    df_merged['post_euro_adoption'] = (df_merged['date'] >= '2023-01-01').astype(int)

    
    df_merged['post_july_2022_hike'] = (df_merged['date'] >= '2022-07-27').astype(int)

    
    df_merged['post_feb_2023_hike'] = (df_merged['date'] >= '2023-02-02').astype(int)

    
    df_merged['year_numeric'] = df_merged['year']
    df_merged['month_numeric'] = df_merged['month']
    df_merged['day_of_year'] = df_merged['date'].dt.dayofyear

    
    df_merged = df_merged.sort_values(['country', 'date'])

    return df_merged


def add_additional_features(df):
    
    print("\nAdding additional features...")

    if df.empty:
        return df

    
    germany_yields = df[df['country'] == 'Germany'][['date', 'bond_yield_10y']].rename(
        columns={'bond_yield_10y': 'germany_yield'}
    )

    df = df.merge(germany_yields, on='date', how='left')

    if 'germany_yield' in df.columns:
        df['spread_vs_germany'] = df['bond_yield_10y'] - df['germany_yield']
        print("  ✓ Added spread vs Germany")

    
    df['yield_change_1d'] = df.groupby('country')['bond_yield_10y'].diff(1)
    df['yield_change_5d'] = df.groupby('country')['bond_yield_10y'].diff(5)
    df['yield_change_30d'] = df.groupby('country')['bond_yield_10y'].diff(30)
    print("  ✓ Added yield changes (1d, 5d, 30d)")

    
    df['yield_ma_30d'] = df.groupby('country')['bond_yield_10y'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    df['yield_std_30d'] = df.groupby('country')['bond_yield_10y'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std()
    )
    print("  ✓ Added rolling statistics (30d MA, STD)")

    
    df['croatia_x_post_july2022'] = df['is_croatia'] * df['post_july_2022_hike']
    df['croatia_x_post_feb2023'] = df['is_croatia'] * df['post_feb_2023_hike']
    df['croatia_x_post_euro'] = df['is_croatia'] * df['post_euro_adoption']
    print("  ✓ Added DiD interaction terms")

    return df


def generate_summary_statistics(df):
    
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    if df.empty:
        print("No data to summarize!")
        return

    print(f"\nTotal records: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"  {', '.join(sorted(df['country'].unique()))}")

    print("\nData availability by country:")
    for country in sorted(df['country'].unique()):
        country_data = df[df['country'] == country]
        non_null_yields = country_data['bond_yield_10y'].notna().sum()
        print(f"  {country:15s}: {non_null_yields:5,} bond yield observations")

    print("\nKey variables:")
    key_vars = ['bond_yield_10y', 'gdp_growth_quarterly', 'inflation_hicp', 'public_debt_gdp']
    for var in key_vars:
        if var in df.columns:
            non_null = df[var].notna().sum()
            pct = (non_null / len(df)) * 100
            print(f"  {var:25s}: {non_null:6,} non-null ({pct:5.1f}%)")

    print("\nDescriptive statistics (bond yields):")
    yield_stats = df.groupby('country')['bond_yield_10y'].describe()[['mean', 'std', 'min', 'max']]
    print(yield_stats.round(3))


def main():
    
    
    df_bonds, df_gdp, df_inflation, df_debt = load_raw_data()

    
    df_merged = merge_datasets(df_bonds, df_gdp, df_inflation, df_debt)

    if df_merged.empty:
        print("\nERROR: Merge failed - no data to save!")
        return None

    
    df_merged = add_additional_features(df_merged)

    
    output_path = SCRIPT_DIR / 'input_data.csv'
    df_merged.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"MERGED DATA SAVED TO: {output_path.name}")
    print(f"{'='*60}")

    
    generate_summary_statistics(df_merged)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)

    return df_merged


if __name__ == "__main__":
    main()
