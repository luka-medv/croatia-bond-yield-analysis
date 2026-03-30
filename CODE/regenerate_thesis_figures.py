"""
Regenerate the four publication-quality thesis figures from clean data.
Based on revision_fixes/code scripts but with correct paths and coefficients.

Figures:
  4.1  H1 time-series DiD  (30-day MA, daily shading)
  4.5  H1b time-series DiD (30-day MA, daily shading)
  4.7  H2 spread convergence DiD (30-day MA, daily shading)
  3.4  Croatia euro adoption timeline (two-panel + annotation strip)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

# House style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.30,
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "DATA"
OUTPUT_DIR = SCRIPT_DIR.parent / "OUTPUTS" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "input_data.csv", parse_dates=["date"])
print(f"Loaded {len(df):,} observations (weekend-free)")

MA_WINDOW = 30

# ======================================================================
# FIGURE 4.1 - H1: ECB Rate Hike Time-Series DiD
# ======================================================================
print("\nCreating Figure 4.1 (H1 time-series DiD)...")

df_h1 = df[
    (df["country"].isin(["Croatia", "Slovenia", "Slovakia", "Lithuania"])) &
    (df["date"] >= "2021-01-01") &
    (df["date"] <= "2024-12-31")
].copy()

croatia_h1 = (
    df_h1[df_h1["country"] == "Croatia"]
    .set_index("date")["bond_yield_10y"]
    .sort_index()
)
control_h1 = (
    df_h1[df_h1["country"] != "Croatia"]
    .groupby("date")["bond_yield_10y"]
    .mean()
    .sort_index()
)

croatia_ma = croatia_h1.rolling(MA_WINDOW, min_periods=10).mean()
control_ma = control_h1.rolling(MA_WINDOW, min_periods=10).mean()

event_date = pd.to_datetime("2022-07-27")

fig, ax = plt.subplots(figsize=(13, 7))

# Raw daily data (light, thin)
ax.plot(croatia_h1.index, croatia_h1.values,
        color="#e74c3c", alpha=0.15, linewidth=0.5)
ax.plot(control_h1.index, control_h1.values,
        color="#3498db", alpha=0.15, linewidth=0.5)

# Moving averages (bold)
ax.plot(croatia_ma.index, croatia_ma.values,
        color="#e74c3c", linewidth=3, label="Croatia (Treatment)")
ax.plot(control_ma.index, control_ma.values,
        color="#3498db", linewidth=3, label="Control Group (SI, SK, LT)")

# Vertical event line
ax.axvline(event_date, color="#9A3412", linestyle="--", linewidth=2.5,
           alpha=0.8, zorder=5, label="ECB Rate Hike (27 Jul 2022)")

# Shaded regions
ax.axvspan(df_h1["date"].min(), event_date,
           alpha=0.06, color="#3498db", zorder=0)
ax.axvspan(event_date, df_h1["date"].max(),
           alpha=0.06, color="#e74c3c", zorder=0)

# Period labels
ax.text(pd.to_datetime("2021-10-01"), ax.get_ylim()[1] * 0.95,
        "Pre-Hike", fontsize=12, fontweight="bold", color="#3498db",
        alpha=0.6, ha="center")
ax.text(pd.to_datetime("2023-10-01"), ax.get_ylim()[1] * 0.95,
        "Post-Hike", fontsize=12, fontweight="bold", color="#e74c3c",
        alpha=0.6, ha="center")

# Annotation: DiD effect (UPDATED coefficient)
post_start = event_date + pd.Timedelta(days=60)
croatia_post_val = croatia_ma.loc[croatia_ma.index >= post_start].iloc[0]
control_post_val = control_ma.loc[control_ma.index >= post_start].iloc[0]

ax.annotate(
    "Croatian yields rise\nmore slowly than\ncontrol group\n(DiD = \u22121.57 pp)",
    xy=(post_start + pd.Timedelta(days=90), (croatia_post_val + control_post_val) / 2),
    xytext=(60, 50),
    textcoords="offset points",
    fontsize=10, fontweight="bold", color="#2d6a2d", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#2d6a2d", linewidth=1.5, alpha=0.9),
    arrowprops=dict(arrowstyle="-|>", color="#2d6a2d", lw=1.5),
)

ax.set_xlabel("Date", fontsize=12, fontweight="bold")
ax.set_ylabel("10-Year Bond Yield (%)", fontsize=12, fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=45)
ax.legend(loc="upper left", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out = OUTPUT_DIR / "h1_did_timeseries.png"
fig.savefig(out)
plt.close(fig)
print(f"  Saved: {out}")


# ======================================================================
# FIGURE 4.5 - H1b: ECB Feb 2023 Hike Time-Series DiD
# ======================================================================
print("\nCreating Figure 4.5 (H1b time-series DiD)...")

countries_h1b = ["Croatia", "Slovenia", "Slovakia", "Lithuania", "France", "Germany"]
df_h1b = df[
    (df["country"].isin(countries_h1b)) &
    (df["date"] >= "2021-01-01") &
    (df["date"] <= "2024-12-31")
].copy()

croatia_1b = (
    df_h1b[df_h1b["country"] == "Croatia"]
    .set_index("date")["bond_yield_10y"]
    .sort_index()
)
control_1b = (
    df_h1b[df_h1b["country"] != "Croatia"]
    .groupby("date")["bond_yield_10y"]
    .mean()
    .sort_index()
)

croatia_1b_ma = croatia_1b.rolling(MA_WINDOW, min_periods=10).mean()
control_1b_ma = control_1b.rolling(MA_WINDOW, min_periods=10).mean()

event_date_1b = pd.to_datetime("2023-02-02")

fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(croatia_1b.index, croatia_1b.values, color="#e74c3c", alpha=0.15, linewidth=0.5)
ax.plot(control_1b.index, control_1b.values, color="#3498db", alpha=0.15, linewidth=0.5)

ax.plot(croatia_1b_ma.index, croatia_1b_ma.values,
        color="#e74c3c", linewidth=3, label="Croatia (Treatment)")
ax.plot(control_1b_ma.index, control_1b_ma.values,
        color="#3498db", linewidth=3, label="Control Group (SI, SK, LT, FR, DE)")

ax.axvline(event_date_1b, color="#9A3412", linestyle="--", linewidth=2.5,
           alpha=0.8, zorder=5, label="ECB Rate Hike (2 Feb 2023)")
ax.axvline(pd.to_datetime("2022-07-27"), color="#6B7280", linestyle=":",
           linewidth=1.5, alpha=0.5, zorder=4, label="First ECB Hike (27 Jul 2022)")

ax.axvspan(df_h1b["date"].min(), event_date_1b,
           alpha=0.06, color="#3498db", zorder=0)
ax.axvspan(event_date_1b, df_h1b["date"].max(),
           alpha=0.06, color="#e74c3c", zorder=0)

ax.text(pd.to_datetime("2021-10-01"), ax.get_ylim()[1] * 0.95,
        "Pre-Feb 2023", fontsize=12, fontweight="bold", color="#3498db",
        alpha=0.6, ha="center")
ax.text(pd.to_datetime("2023-10-01"), ax.get_ylim()[1] * 0.95,
        "Post-Feb 2023", fontsize=12, fontweight="bold", color="#e74c3c",
        alpha=0.6, ha="center")

# DiD annotation (UPDATED coefficient)
post_start_1b = event_date_1b + pd.Timedelta(days=90)
c_val = croatia_1b_ma.loc[croatia_1b_ma.index >= post_start_1b].iloc[0]
ctrl_val = control_1b_ma.loc[control_1b_ma.index >= post_start_1b].iloc[0]

ax.annotate(
    "Croatian yields diverge\nfrom expanded control\n(DiD = \u22121.52 pp)",
    xy=(post_start_1b + pd.Timedelta(days=60), (c_val + ctrl_val) / 2),
    xytext=(60, 55),
    textcoords="offset points",
    fontsize=10, fontweight="bold", color="#2d6a2d", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#2d6a2d", linewidth=1.5, alpha=0.9),
    arrowprops=dict(arrowstyle="-|>", color="#2d6a2d", lw=1.5),
)

ax.set_xlabel("Date", fontsize=12, fontweight="bold")
ax.set_ylabel("10-Year Bond Yield (%)", fontsize=12, fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=45)
ax.legend(loc="upper left", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out = OUTPUT_DIR / "h1b_did_timeseries.png"
fig.savefig(out)
plt.close(fig)
print(f"  Saved: {out}")


# ======================================================================
# FIGURE 4.7 - H2: Euro Adoption Spread Convergence Time-Series DiD
# ======================================================================
print("\nCreating Figure 4.7 (H2 spread convergence time-series DiD)...")

df_h2 = df[
    (df["country"].isin(["Croatia", "Slovenia", "Slovakia", "Lithuania"])) &
    (df["date"] >= "2021-01-01") &
    (df["date"] <= "2024-12-31")
].copy()

croatia_spread = (
    df_h2[df_h2["country"] == "Croatia"]
    .set_index("date")["spread_vs_germany"]
    .sort_index()
)
control_spread = (
    df_h2[df_h2["country"] != "Croatia"]
    .groupby("date")["spread_vs_germany"]
    .mean()
    .sort_index()
)

croatia_spread_ma = croatia_spread.rolling(MA_WINDOW, min_periods=10).mean()
control_spread_ma = control_spread.rolling(MA_WINDOW, min_periods=10).mean()

euro_date = pd.to_datetime("2023-01-01")

fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(croatia_spread.index, croatia_spread.values,
        color="#e74c3c", alpha=0.15, linewidth=0.5)
ax.plot(control_spread.index, control_spread.values,
        color="#3498db", alpha=0.15, linewidth=0.5)

ax.plot(croatia_spread_ma.index, croatia_spread_ma.values,
        color="#e74c3c", linewidth=3, label="Croatia (Treatment)")
ax.plot(control_spread_ma.index, control_spread_ma.values,
        color="#3498db", linewidth=3, label="Control Group (SI, SK, LT)")

ax.axvline(euro_date, color="#0E7490", linestyle="--", linewidth=2.5,
           alpha=0.8, zorder=5, label="Euro Adoption (1 Jan 2023)")

ax.axvspan(df_h2["date"].min(), euro_date,
           alpha=0.06, color="#3498db", zorder=0)
ax.axvspan(euro_date, df_h2["date"].max(),
           alpha=0.06, color="#0E7490", zorder=0)

ax.text(pd.to_datetime("2021-10-01"), ax.get_ylim()[1] * 0.95,
        "Pre-Euro", fontsize=12, fontweight="bold", color="#3498db",
        alpha=0.6, ha="center")
ax.text(pd.to_datetime("2024-01-01"), ax.get_ylim()[1] * 0.95,
        "Post-Euro", fontsize=12, fontweight="bold", color="#0E7490",
        alpha=0.6, ha="center")

# Annotation: convergence (UPDATED coefficient)
ax.annotate(
    "Croatian spreads\nconverge toward\ncontrol group\n(DiD = \u22120.72 pp)",
    xy=(pd.to_datetime("2023-06-01"),
        (croatia_spread_ma.loc["2023-06-01":].iloc[0] +
         control_spread_ma.loc["2023-06-01":].iloc[0]) / 2),
    xytext=(80, 50),
    textcoords="offset points",
    fontsize=10, fontweight="bold", color="#2d6a2d", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#2d6a2d", linewidth=1.5, alpha=0.9),
    arrowprops=dict(arrowstyle="-|>", color="#2d6a2d", lw=1.5),
)

ax.set_xlabel("Date", fontsize=12, fontweight="bold")
ax.set_ylabel("Yield Spread vs Germany (pp)", fontsize=12, fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=45)
ax.legend(loc="upper left", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out = OUTPUT_DIR / "h2_spread_convergence_timeseries.png"
fig.savefig(out)
plt.close(fig)
print(f"  Saved: {out}")


# ======================================================================
# FIGURE 3.4 - Croatia Euro Adoption Timeline (two-panel + annotation)
# ======================================================================
print("\nCreating Figure 3.4 (Croatia euro adoption timeline)...")

plt.rcParams.update({"font.family": "sans-serif"})

croatia_data = df[df["country"] == "Croatia"].sort_values("date")
germany_data = df[df["country"] == "Germany"].sort_values("date")

# Milestones: (date, short_label, color, stagger_level, x_nudge_days)
events = [
    ("2017-10-01", "Euro-Intent\nStrategy",       "#4B5563", 0,  0),
    ("2019-07-05", "ERM II\nApplication",          "#4B5563", 1,  0),
    ("2020-07-10", "ERM II\nEntry",                "#4B5563", 0,  0),
    ("2022-06-01", "Convergence\nAssessment",      "#4B5563", 2, -120),
    ("2022-07-12", "Council\nApproves Entry",      "#92400E", 0, -90),
    ("2022-07-27", "ECB Hike\n(+50 bps)",          "#B91C1C", 1,  0),
    ("2023-01-01", "EURO\nADOPTION",               "#0E7490", 2,  60),
]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.2, 1], hspace=0.05)
ax_ann = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax_ann)
ax2 = fig.add_subplot(gs[2], sharex=ax_ann)

# Panel 1: Yields
ax1.plot(croatia_data["date"], croatia_data["bond_yield_10y"],
         linewidth=2.5, label="Croatia 10Y Yield", color="#D2691E", zorder=3)
ax1.plot(germany_data["date"], germany_data["bond_yield_10y"],
         linewidth=1.8, alpha=0.7, linestyle="--",
         label="Germany 10Y Yield (Reference)", color="#374151", zorder=3)

euro_date = pd.to_datetime("2023-01-01")
for ax in [ax1, ax2, ax_ann]:
    ax.axvspan(euro_date, croatia_data["date"].max(),
               alpha=0.08, color="#0E7490", zorder=0)

ax1.axvspan(euro_date, croatia_data["date"].max(),
            alpha=0.08, color="#0E7490", zorder=0, label="Post-Euro Adoption")

# Panel 2: Spreads
ax2.plot(croatia_data["date"], croatia_data["spread_vs_germany"],
         linewidth=2.5, label="Croatia Spread vs Germany", color="#D2691E", zorder=3)
ax2.axhline(0, linestyle="-", linewidth=0.8, alpha=0.3, color="gray")

# Draw milestone lines across all panels
for date_str, label, color, level, x_nudge in events:
    date = pd.to_datetime(date_str)
    is_euro = "EURO" in label and "ADOPTION" in label
    lw = 2.5 if is_euro else 1.8
    alpha = 0.85 if is_euro else 0.65
    ls = "-" if is_euro else "--"
    for ax in [ax_ann, ax1, ax2]:
        ax.axvline(date, linestyle=ls, linewidth=lw, alpha=alpha,
                   color=color, zorder=2)

# Annotation strip: labels
ax_ann.set_xlim(ax1.get_xlim())
ax_ann.set_ylim(0, 3)
ax_ann.axis("off")

y_positions = {0: 0.4, 1: 1.4, 2: 2.4}

for date_str, label, color, level, x_nudge in events:
    date = pd.to_datetime(date_str)
    label_date = date + pd.Timedelta(days=x_nudge)
    y = y_positions[level]
    is_euro = "EURO" in label and "ADOPTION" in label

    fontsize = 9.5 if is_euro else 8.5
    fontweight = "bold" if is_euro else "semibold"
    facecolor = "#DBEAFE" if is_euro else "#F9FAFB"
    edgecolor = color
    edge_lw = 1.5 if is_euro else 1.0
    conn = "arc3,rad=0.15" if x_nudge != 0 else "arc3,rad=0"

    ax_ann.annotate(
        label,
        xy=(date, 0),
        xytext=(label_date, y),
        fontsize=fontsize,
        fontweight=fontweight,
        ha="center", va="center",
        linespacing=1.1,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=facecolor,
                  edgecolor=edgecolor, linewidth=edge_lw, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0,
                        shrinkA=0, shrinkB=2, connectionstyle=conn),
    )

# Formatting
ax1.set_ylabel("10-Year Bond Yield (%)", fontsize=12, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
ax2.set_ylabel("Spread vs Germany (pp)", fontsize=12, fontweight="bold")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Croatia's Euro Adoption Timeline: Yields and Spreads",
             fontsize=15, fontweight="bold", y=0.98)

out = OUTPUT_DIR / "09_croatia_euro_timeline_FIXED.png"
fig.savefig(out)
plt.close(fig)
print(f"  Saved: {out}")

print("\nAll 4 thesis figures regenerated.")
