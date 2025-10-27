#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Velocity & Acceleration plots from raw displacement (_noFilter.csv).

Input:
  - The CSV written by your existing script:
      /insar-data/<PROJECT>/report/<PROJECT>_ts_noFilter.csv
    with columns: date, S1, S2, ...

Output:
  - /insar-data/<PROJECT>/report/<PROJECT>_Velocity.png
  - /insar-data/<PROJECT>/report/<PROJECT>_Acceleration.png
  - (optional) CSVs with derived series:
      *_velocity_mm_per_yr.csv, *_acceleration_mm_per_yr2.csv

Notes:
  - Derivatives are computed on the raw (no-filter) series.
  - Time deltas are handled in continuous days using exact timestamps.
  - NaNs are linearly interpolated only to evaluate derivatives; the
    resulting velocity/acceleration are set back to NaN wherever the
    original displacement was NaN to avoid overconfidence.

Usage:
  python make_vel_acc_plots.py --project-name Tocantins24meses
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LINEWIDTH = 2.0  # keep same style as your displacement plot
DPI = 200

def _to_seconds(dts: pd.Series) -> np.ndarray:
    """DatetimeIndex/Series -> float seconds since first sample."""
    t0 = dts.iloc[0]
    return (dts - t0).dt.total_seconds().to_numpy(dtype=float)

def _interp_time(y: pd.Series) -> pd.Series:
    """Time-based linear interpolation without extrapolating ends."""
    return y.interpolate(method="time", limit_direction="both")

def _derivative(y: pd.Series, t_sec: np.ndarray) -> np.ndarray:
    """
    Safe first derivative dy/dt using np.gradient on an interpolated copy.
    - Interpolate NaNs to compute gradient.
    - Mask derivative to NaN where original y was NaN.
    """
    y_orig = y.copy()
    y_i = _interp_time(y)  # interpolated copy for stable gradient
    dy_dt = np.gradient(y_i.to_numpy(dtype=float), t_sec, edge_order=2)
    # put back NaN where original displacement is NaN
    dy_dt[np.isnan(y_orig.to_numpy())] = np.nan
    return dy_dt

def _second_derivative(y: pd.Series, t_sec: np.ndarray) -> np.ndarray:
    """
    Second derivative d2y/dt2 by differentiating the first derivative.
    Keeps NaN mask consistent with original y.
    """
    dy_dt = _derivative(y, t_sec)
    # Interpolate dy/dt to compute second derivative, but preserve NaNs at the end
    dy_dt_series = pd.Series(dy_dt, index=y.index)
    dy_dt_i = _interp_time(dy_dt_series)
    d2y_dt2 = np.gradient(dy_dt_i.to_numpy(dtype=float), t_sec, edge_order=2)
    mask = np.isnan(y.to_numpy())
    d2y_dt2[mask] = np.nan
    return d2y_dt2

def compute_vel_acc(df: pd.DataFrame):
    """
    Given df with 'date' + sections (mm), return two DataFrames:
      vel_df: mm/yr
      acc_df: mm/yr^2
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    t_sec = _to_seconds(df.index.to_series())
    sec_per_year = 365.25 * 86400.0

    vel = {"date": df.index}
    acc = {"date": df.index}

    for col in df.columns:
        y = df[col]
        dy_dt = _derivative(y, t_sec)                # mm per second
        d2y_dt2 = _second_derivative(y, t_sec)       # mm per second^2

        vel[col] = dy_dt * sec_per_year             # mm/yr
        acc[col] = d2y_dt2 * (sec_per_year**2)      # mm/yr^2

    vel_df = pd.DataFrame(vel)
    acc_df = pd.DataFrame(acc)
    return vel_df, acc_df

def _plot_lines(dates, series_dict, title, y_label, save_path, sort_by_last=True):
    """
    Generic multi-line plot matching the displacement style:
      - lines with LINEWIDTH
      - grid=True
      - legend outside on the right
      - sorted by last value (desc) if requested
    """
    # Prepare ordering
    names = list(series_dict.keys())
    if sort_by_last:
        last_vals = [(name, np.nan_to_num(series_dict[name][-1], nan=-np.inf))
                     for name in names]
        names = [n for n, _ in sorted(last_vals, key=lambda x: x[1], reverse=True)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in names:
        ax.plot(dates, series_dict[name], label=name, linewidth=LINEWIDTH)

    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI)
        print(f"Saved figure: {save_path}")
    plt.show()

def main(project_name: str, csv_path: str = None, save_csv: bool = True):
    report_dir = Path(f"/insar-data/{project_name}/report")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Default input CSV path (from your existing script)
    if csv_path is None:
        csv_path = report_dir / f"{project_name}_ts_noFilter.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {csv_path}")

    print(f"Reading raw displacement: {csv_path}")
    df = pd.read_csv(csv_path)

    # Compute velocity/acceleration
    print("Computing velocity (mm/yr) and acceleration (mm/yr^2)…")
    vel_df, acc_df = compute_vel_acc(df)

    # Optionally save derived CSVs
    if save_csv:
        vel_out = report_dir / f"{project_name}_velocity_mm_per_yr.csv"
        acc_out = report_dir / f"{project_name}_acceleration_mm_per_yr2.csv"
        vel_df.to_csv(vel_out, index=False)
        acc_df.to_csv(acc_out, index=False)
        print(f"Saved CSV: {vel_out}")
        print(f"Saved CSV: {acc_out}")

    # Prepare data for plotting (match displacement style: multi-line, legend sorted by last)
    dates = pd.to_datetime(vel_df["date"])
    section_cols = [c for c in vel_df.columns if c != "date"]

    vel_series = {name: vel_df[name].to_numpy(dtype=float) for name in section_cols}
    acc_series = {name: acc_df[name].to_numpy(dtype=float) for name in section_cols}

    # Plots
    vel_png = report_dir / f"{project_name}_Velocity.png"
    acc_png = report_dir / f"{project_name}_Acceleration.png"

    _plot_lines(
        dates,
        vel_series,
        title="Velocity of displacement time series",
        y_label="Velocity (mm/yr)",
        save_path=vel_png,
        sort_by_last=True,
    )

    _plot_lines(
        dates,
        acc_series,
        title="Acceleration of displacement time series",
        y_label="Acceleration (mm/yr²)",
        save_path=acc_png,
        sort_by_last=True,
    )

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make velocity/acceleration plots from _noFilter.csv (MintPy/Miaplpy output)."
    )
    parser.add_argument("--project-name", required=True, type=str,
                        help="Project name (used to infer /insar-data/<PROJECT>/report paths)")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Optional direct path to *_ts_noFilter.csv (overrides project-name).")
    parser.add_argument("--no-save-csv", action="store_true",
                        help="Do not write derived CSVs (velocity/acceleration).")
    args = parser.parse_args()
    main(args.project_name, args.csv_path, save_csv=(not args.no_save_csv))
