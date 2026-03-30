

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def build_vif_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    
    subset = df[list(cols)].dropna()
    if subset.empty:
        return pd.DataFrame({"Variable": cols, "VIF": np.nan})
    subset = (subset - subset.mean()) / subset.std(ddof=0)
    rows = []
    for idx, col in enumerate(subset.columns):
        try:
            val = variance_inflation_factor(subset.values, idx)
        except Exception:
            val = np.nan
        rows.append((col, val))
    return pd.DataFrame(rows, columns=["Variable", "VIF"])
