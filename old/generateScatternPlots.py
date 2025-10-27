#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make one scatter plot per section/ROI from a CSV time series (InSAR displacement in mm).

Each plot shows:
  - Scatter of displacement vs date
  - Linear trend (velocity) for the whole period and for the last N days
  - Quadratic fit for acceleration (whole period and last N days)
  - Stats box with velocity [mm/yr] and acceleration [mm/yr^2] for both windows
  - A shaded region marking the last N days window

Assumptions:
  - CSV has one date column (any parseable format) and multiple ROI columns (numerical, displacement in mm)
  - Missing values are allowed

Usage:
  python plot_insar_sections.py --csv path/to/file.csv --outdir ./plots --last-days 90
"""

import argparse
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import timedelta

# ----------------------- Helpers -----------------------

DATE_CANDIDATES = [
    "date", "Date", "DATE",
    "data", "Data", "DATA",
    "timestamp", "Timestamp", "TIMESTAMP",
    "time", "Time", "TIME"
]

def find_date_col(df: pd.DataFrame, user_date_col: str = None) -> str:
    if user_date_col:
        if user_date_col in df.columns:
            return user_date_col
        raise ValueError(f"Requested date column '{user_date_col}' not found. Available: {list(df.columns)}")
    # try common names
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: try to guess by attempting to parse each column
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    raise ValueError("Could not identify a date column. Pass --date-col explicitly.")

def as_years(dt_series: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    """Convert datetime to fractional years relative to t0."""
    delta_days = (dt_series - t0).dt.total_seconds() / (24*3600)
    return delta_days.to_numpy() / 365.25

def polyfit_stats(t_years: np.ndarray, y_mm: np.ndarray, deg: int):
    """Fit polynomial and return coefficients and function for plotting."""
    if len(t_years) < (deg + 1) or np.isfinite(y_mm).sum() < (deg + 1):
        return None, None
    # clean NaNs
    m = np.isfinite(t_years) & np.isfinite(y_mm)
    t = t_years[m]
    y = y_mm[m]
    if len(t) < (deg + 1):
        return None, None
    coeffs = np.polyfit(t, y, deg)  # highest power first
    p = np.poly1d(coeffs)
    return coeffs, p

def velocity_mm_per_year_from_linear(coeffs):
    # For y = b1 * t + b0, with t in years => velocity = b1 [mm/yr]
    if coeffs is None or len(coeffs) != 2:
        return np.nan
    return float(coeffs[0])

def acceleration_mm_per_year2_from_quadratic(coeffs):
    # For y = a * t^2 + b * t + c; acceleration = d^2 y / dt^2 = 2a [mm/yr^2]
    if coeffs is None or len(coeffs) != 3:
        return np.nan
    return float(2.0 * coeffs[0])

def format_stats_box(v_all, a_all, v_last, a_last):
    def fmt(x, unit):
        return "n/a" if (x is None or not np.isfinite(x)) else f"{x:,.3f} {unit}"
    return (
        f"Whole period:\n"
        f"  Velocity: {fmt(v_all, 'mm/yr')}\n"
        f"  Accel.:   {fmt(a_all, 'mm/yr²')}\n"
        f"Last window:\n"
        f"  Velocity: {fmt(v_last, 'mm/yr')}\n"
        f"  Accel.:   {fmt(a_last, 'mm/yr²')}"
    )

# ----------------------- Plotting -----------------------

def plot_section(
    dates: pd.Series,
    disp_mm: pd.Series,
    section_name: str,
    out_path: Path,
    last_days: int = 90,
    dpi: int = 220
):
    # Drop rows where date missing
    s = pd.DataFrame({"date": dates, "disp": disp_mm}).dropna(subset=["date"])
    # If all NaN in disp, skip
    if s["disp"].notna().sum() < 2:
        print(f"[WARN] Not enough data to plot '{section_name}'. Skipping.")
        return

    # Sort by date and drop duplicate dates (keep last)
    s = s.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    t0 = s["date"].iloc[0]
    t_yrs = as_years(s["date"], t0)
    y_mm = s["disp"].to_numpy()

    # Whole-period fits
    lin_all, p1_all = polyfit_stats(t_yrs, y_mm, deg=1)
    quad_all, p2_all = polyfit_stats(t_yrs, y_mm, deg=2)
    v_all = velocity_mm_per_year_from_linear(lin_all)
    a_all = acceleration_mm_per_year2_from_quadratic(quad_all)

    # Last-N-days window
    last_start = s["date"].max() - timedelta(days=last_days)
    s_last = s[s["date"] >= last_start].copy()
    if len(s_last) >= 3:
        t0_last = s_last["date"].iloc[0]
        t_last_yrs = as_years(s_last["date"], t0_last)
        y_last_mm = s_last["disp"].to_numpy()
        lin_last, p1_last = polyfit_stats(t_last_yrs, y_last_mm, deg=1)
        quad_last, p2_last = polyfit_stats(t_last_yrs, y_last_mm, deg=2)
        v_last = velocity_mm_per_year_from_linear(lin_last)
        a_last = acceleration_mm_per_year2_from_quadratic(quad_last)
    else:
        p1_last = p2_last = None
        v_last = a_last = np.nan

    # Build smooth t-grid for plotting fits
    t_grid = np.linspace(t_yrs.min(), t_yrs.max(), 300)
    dates_grid = [t0 + pd.Timedelta(days=float(t * 365.25)) for t in t_grid]

    # For last window, grid within that window
    dates_grid_last = None
    t_grid_last = None
    if len(s_last) >= 3:
        t_grid_last = np.linspace(t_last_yrs.min(), t_last_yrs.max(), 200)
        dates_grid_last = [t0_last + pd.Timedelta(days=float(t * 365.25)) for t in t_grid_last]

    # ---------------- Figure ----------------
    plt.figure(figsize=(10, 5.2))
    ax = plt.gca()

    # Scatter black color
    ax.scatter(s["date"], s["disp"], s=14, alpha=0.9, edgecolor="none", label="Measurements", color="black")

    # Whole-period linear fit
    if p1_all is not None:
        ax.plot(dates_grid, p1_all(t_grid), linewidth=2.0,linestyle="--", label="Linear trend (whole)")

    # Whole-period quadratic curve (for accel.)
    # if p2_all is not None:
    #     ax.plot(dates_grid, p2_all(t_grid), linewidth=1.3, linestyle="--", label="Quadratic fit (whole)")

    # Shade last window
    ax.axvspan(last_start, s["date"].max(), alpha=0.10, label=f"Last {last_days} days")

    # Last-window linear fit
    if p1_last is not None and dates_grid_last is not None:
        ax.plot(dates_grid_last, p1_last(t_grid_last), linewidth=2.0, linestyle=":", label="Linear trend (last)")

    # Last-window quadratic curve
    # if p2_last is not None and dates_grid_last is not None:
    #     ax.plot(dates_grid_last, p2_last(t_grid_last), linewidth=1.3, linestyle="--", dashes=(3,2), label="Quadratic fit (last)")

    # Labels & grid
    ax.set_title(f"{section_name} — InSAR Displacement Time Series", loc="left", fontsize=14, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Displacement [mm]")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_ylim(-50, 50)
    # Date formatting
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # Stats box
    stats_txt = format_stats_box(v_all, a_all, v_last, a_last)
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, linewidth=0.8)
    ax.text(0.99, 0.02, stats_txt, transform=ax.transAxes, fontsize=10,
            ha="right", va="bottom", bbox=bbox_props, family="monospace")

    # Legend
    ax.legend(loc="upper left", frameon=True)

    # Tight layout and save
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[OK] Saved {out_path}")

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Plot one scatter per ROI with velocity & acceleration (whole + last N days).")
    ap.add_argument("--csv", required=True, help="Path to CSV file.")
    ap.add_argument("--date-col", default=None, help="Name of the date column (optional).")
    ap.add_argument("--outdir", default="./plots", help="Directory to save the plots.")
    ap.add_argument("--last-days", type=int, default=90, help="Window in days for 'last' stats (default: 90).")
    ap.add_argument("--dpi", type=int, default=220, help="Output DPI (default: 220).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Read CSV; let pandas try best to parse
    df = pd.read_csv(csv_path)
    date_col = find_date_col(df, args.date_col)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Identify ROI columns = numeric columns except the date column
    roi_cols = [c for c in df.columns if c != date_col]
    # Try to keep only numeric ROI columns
    numeric_cols = []
    for c in roi_cols:
        # If convertible to numeric (with NaNs ok), accept
        try:
            pd.to_numeric(df[c], errors="coerce")
            numeric_cols.append(c)
        except Exception:
            pass

    if not numeric_cols:
        raise ValueError("No numeric ROI columns found in the CSV.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    for col in numeric_cols:
        out_file = outdir / f"{col}_timeseries_plot.png"
        plot_section(
            dates=df[date_col],
            disp_mm=pd.to_numeric(df[col], errors="coerce"),
            section_name=col,
            out_path=out_file,
            last_days=args.last_days,
            dpi=args.dpi
        )

if __name__ == "__main__":
    main()
