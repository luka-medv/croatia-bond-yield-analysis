"""
Download macroeconomic data from Eurostat JSON API.

Datasets:
  1. GDP growth (quarterly, q-o-q): namq_10_gdp
     unit=CLV_PCH_PRE, s_adj=SCA, na_item=B1GQ
  2. HICP inflation (monthly, y-o-y): prc_hicp_manr
     coicop=CP00, unit=RCH_A
  3. Public debt (annual, % GDP): gov_10dd_edpt1
     na_item=GD, sector=S13, unit=PC_GDP
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw_data"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["HR", "SI", "SK", "LV", "LT", "FR", "DE"]
COUNTRY_NAMES = {
    "HR": "Croatia", "SI": "Slovenia", "SK": "Slovakia",
    "LV": "Latvia", "LT": "Lithuania", "FR": "France", "DE": "Germany",
}

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def _fetch_eurostat(dataset: str, params: dict, timeout: int = 30) -> dict | None:
    url = f"{BASE}/{dataset}"
    params["format"] = "JSON"
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}")
        return None
    return resp.json()


def _extract_time_values(data: dict) -> list[tuple[str, float]]:
    """Extract (time_label, value) pairs from a single-series Eurostat JSON response."""
    values = data.get("value", {})
    if not values:
        return []

    # Find the time dimension and its labels
    dim_ids = data["id"]
    time_idx = dim_ids.index("time") if "time" in dim_ids else None
    if time_idx is None:
        return []

    time_labels = list(data["dimension"]["time"]["category"]["index"].keys())
    sizes = data["size"]

    results = []
    for flat_key, val in values.items():
        flat = int(flat_key)
        # Decode flat index to per-dimension indices
        indices = []
        remainder = flat
        for s in reversed(sizes):
            indices.append(remainder % s)
            remainder //= s
        indices.reverse()

        t_idx = indices[time_idx]
        if t_idx < len(time_labels):
            results.append((time_labels[t_idx], float(val)))

    results.sort(key=lambda x: x[0])
    return results


# ============================================================
# 1. GDP GROWTH (Quarterly, q-o-q, seasonally + calendar adjusted)
# ============================================================

def download_gdp() -> pd.DataFrame:
    print("\n[1/3] Downloading GDP Growth (quarterly, q-o-q) ...")
    all_rows: list[dict] = []

    for code in COUNTRIES:
        name = COUNTRY_NAMES[code]
        print(f"  {name} ...", end=" ")
        data = _fetch_eurostat("namq_10_gdp", {
            "geo": code,
            "unit": "CLV_PCH_PRE",
            "s_adj": "SCA",
            "na_item": "B1GQ",
            "sinceTimePeriod": "2015-Q1",
        })
        if data is None:
            print("FAILED")
            continue

        pairs = _extract_time_values(data)
        for time_key, value in pairs:
            # time_key like "2015-Q1" -> date 2015-01-01
            year, q = time_key.split("-Q")
            month = int(q) * 3 - 2
            all_rows.append({
                "date": pd.Timestamp(f"{year}-{month:02d}-01"),
                "country": name,
                "country_code": code,
                "gdp_growth_quarterly": value,
            })
        print(f"{len(pairs)} quarters")
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).sort_values(["country", "date"]).reset_index(drop=True)
    return df


# ============================================================
# 2. HICP INFLATION (Monthly, annual rate of change)
# ============================================================

def download_inflation() -> pd.DataFrame:
    print("\n[2/3] Downloading HICP Inflation (monthly, y-o-y) ...")
    all_rows: list[dict] = []

    for code in COUNTRIES:
        name = COUNTRY_NAMES[code]
        print(f"  {name} ...", end=" ")
        data = _fetch_eurostat("prc_hicp_manr", {
            "geo": code,
            "coicop": "CP00",
            "unit": "RCH_A",
            "sinceTimePeriod": "2015-01",
        })
        if data is None:
            print("FAILED")
            continue

        pairs = _extract_time_values(data)
        for time_key, value in pairs:
            # time_key like "2015-01" -> date 2015-01-01
            all_rows.append({
                "date": pd.Timestamp(f"{time_key}-01"),
                "country": name,
                "country_code": code,
                "inflation_hicp": value,
            })
        print(f"{len(pairs)} months")
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).sort_values(["country", "date"]).reset_index(drop=True)
    return df


# ============================================================
# 3. PUBLIC DEBT (Annual, % of GDP)
# ============================================================

def download_debt() -> pd.DataFrame:
    print("\n[3/3] Downloading Public Debt (annual, % GDP) ...")
    all_rows: list[dict] = []

    for code in COUNTRIES:
        name = COUNTRY_NAMES[code]
        print(f"  {name} ...", end=" ")
        count = 0
        for year in range(2015, 2025):
            data = _fetch_eurostat("gov_10dd_edpt1", {
                "geo": code,
                "na_item": "GD",
                "sector": "S13",
                "unit": "PC_GDP",
                "time": str(year),
            })
            if data is None:
                continue
            values = data.get("value", {})
            if not values:
                continue
            val = float(next(iter(values.values())))
            all_rows.append({
                "date": pd.Timestamp(f"{year}-01-01"),
                "country": name,
                "country_code": code,
                "public_debt_gdp": val,
            })
            count += 1
        print(f"{count} years")
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).sort_values(["country", "date"]).reset_index(drop=True)
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DOWNLOADING REAL EUROSTAT DATA")
    print("=" * 70)

    df_gdp = download_gdp()
    df_inflation = download_inflation()
    df_debt = download_debt()

    print("\n" + "=" * 70)
    saved = 0

    if not df_gdp.empty:
        path = RAW_DIR / "gdp_growth.csv"
        df_gdp.to_csv(path, index=False)
        print(f"  GDP Growth:   {len(df_gdp):>4} records -> {path.name}")
        saved += 1

    if not df_inflation.empty:
        path = RAW_DIR / "inflation_hicp.csv"
        df_inflation.to_csv(path, index=False)
        print(f"  Inflation:    {len(df_inflation):>4} records -> {path.name}")
        saved += 1

    if not df_debt.empty:
        path = RAW_DIR / "public_debt.csv"
        df_debt.to_csv(path, index=False)
        print(f"  Public Debt:  {len(df_debt):>4} records -> {path.name}")
        saved += 1

    print("=" * 70)
    if saved == 3:
        print("ALL 3 DATASETS SAVED SUCCESSFULLY")
    else:
        print(f"WARNING: only {saved}/3 datasets saved")
    print("=" * 70)

    return df_gdp, df_inflation, df_debt


if __name__ == "__main__":
    main()
