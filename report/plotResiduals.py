#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot residuals from fit and actual time series for each section.

For each section, creates a plot showing:
  1. Original time series
  2. Fitted environmental + trend model
  3. Residuals after removing environmental effects
  4. Residuals after removing environmental + trend

Usage:
  python plotResiduals.py --csv input.csv --output-dir output/
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple


# -------------------- From calculateProb.py --------------------

def _parse_date_col(df: pd.DataFrame) -> str:
    """Find a plausible date column name and coerce to datetime."""
    candidates = [
        "date", "Date", "DATE", "Data", "timestamp", "Timestamp", "TIMESTAMP",
    ]
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            return c
    # If not found: try first column
    c0 = df.columns[0]
    df[c0] = pd.to_datetime(df[c0])
    return c0


def design_matrix_temp(t_days: np.ndarray, temp: Optional[np.ndarray]) -> np.ndarray:
    """Build a design matrix for environmental effects.
    If temp is provided, use [1, T, T^2, sin/cos annual, sin/cos semiannual].
    Otherwise use seasonal proxy only.
    """
    t_days = np.asarray(t_days)
    tt = t_days.astype(float)
    omega1 = 2 * np.pi / 365.2425
    omega2 = 2 * omega1
    X = [np.ones_like(tt)]
    if temp is not None:
        T = np.asarray(temp, dtype=float)
        X += [T, T * T]
    X += [np.sin(omega1 * tt), np.cos(omega1 * tt), np.sin(omega2 * tt), np.cos(omega2 * tt)]
    return np.column_stack(X)


def fit_and_remove_environment(t: np.ndarray, y: np.ndarray, temp: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit environmental/seasonal model and return fitted values, residuals and R^2_T.
    R^2_T is the fraction explained by the environmental model.
    
    Returns:
        fitted_env: fitted environmental component
        resid: residuals after removing environmental component
        r2: R^2 of environmental fit
    """
    mask = np.isfinite(t) & np.isfinite(y)
    if temp is not None:
        mask &= np.isfinite(temp)
    t_clean, y_clean = t[mask], y[mask]
    T = temp[mask] if temp is not None else None
    
    if len(t_clean) < 6:
        return np.zeros_like(y), y.copy(), 0.0
    
    X = design_matrix_temp(t_clean, T)
    beta, *_ = np.linalg.lstsq(X, y_clean, rcond=None)
    y_hat = X @ beta
    
    ss_res = float(np.sum((y_clean - y_hat) ** 2))
    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    
    # Reconstruct full fitted values
    X_full = design_matrix_temp(t, temp)
    fitted_env = X_full @ beta
    resid = y - fitted_env
    
    return fitted_env, resid, float(max(0.0, min(1.0, r2)))


def huber_weights(r: np.ndarray, c: float = 1.345) -> np.ndarray:
    """Huber IRLS weights for residuals r / s, where s is MAD scale."""
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    s = 1.4826 * mad if mad > 0 else (np.std(r) + 1e-9)
    u = r / (s + 1e-12)
    w = np.ones_like(u)
    mask = np.abs(u) > c
    w[mask] = c / (np.abs(u[mask]) + 1e-12)
    return w


def robust_polyfit(t: np.ndarray, y: np.ndarray, deg: int = 1, max_iter: int = 50,
                   tol: float = 1e-6) -> Tuple[np.ndarray, float, np.ndarray]:
    """IRLS robust polynomial fit minimizing Huber loss.
    Returns (coeffs_low_to_high, sigma_resid, weights).
    """
    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < deg + 1:
        return np.zeros(deg + 1), np.nan, np.ones_like(t)
    
    X = np.vander(t, N=deg + 1, increasing=True)  # [1, t, t^2, ...]
    # Init with OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(max_iter):
        y_hat = X @ beta
        r = y - y_hat
        w = huber_weights(r)
        W = np.diag(w)
        beta_new, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    
    return beta, 0.0, w


def fit_trend(t_years: np.ndarray, resid: np.ndarray, deg: int = 2) -> Tuple[np.ndarray, float, float, float, float]:
    """Fit a polynomial trend to the residuals (after env removal).
    
    Returns:
        fitted_trend: fitted polynomial values
        velocity: velocity in mm/yr (b1 coefficient)
        acceleration: acceleration in mm/yr² (2*b2 coefficient)
        sigma_v: standard error of velocity
        sigma_a: standard error of acceleration
    """
    beta, sigma_resid, w = robust_polyfit(t_years, resid, deg=deg)
    X = np.vander(t_years, N=deg + 1, increasing=True)
    fitted_trend = X @ beta
    
    # Extract velocity and acceleration (same as calculateProb.py)
    b0, b1, b2 = beta.tolist()
    velocity = float(b1)
    acceleration = float(2.0 * b2)
    
    # Calculate standard errors
    W = np.diag(w)
    XtWX = X.T @ W @ X
    try:
        import math
        cov = np.linalg.inv(XtWX) * sigma_resid**2
        sigma_v = math.sqrt(max(0.0, cov[1, 1]))
        sigma_a = 2.0 * math.sqrt(max(0.0, cov[2, 2]))
    except np.linalg.LinAlgError:
        sigma_v = float("nan")
        sigma_a = float("nan")
    
    return fitted_trend, velocity, acceleration, sigma_v, sigma_a


# -------------------- Plotting --------------------

def plot_section_residuals(section_name: str, dates: pd.DatetimeIndex, 
                          y_original: np.ndarray, fitted_env: np.ndarray, 
                          resid_env: np.ndarray, fitted_trend: np.ndarray,
                          resid_full: np.ndarray, r2_env: float, 
                          velocity: float, acceleration: float,
                          sigma_v: float, sigma_a: float,
                          output_path: Path, units: str = "mm"):
    """Create a comprehensive plot for one section showing all components."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Create summary text for velocity and acceleration
    vel_text = f"v = {velocity:.3f} ± {sigma_v:.3f} {units}/yr"
    acc_text = f"a = {acceleration:.3f} ± {sigma_a:.3f} {units}/yr²"
    metrics_text = f"{vel_text}  |  {acc_text}"
    
    # Subplot 1: Original time series + environmental fit
    ax1 = axes[0]
    ax1.plot(dates, y_original, 'o-', alpha=0.6, label='Original data', markersize=4)
    ax1.plot(dates, fitted_env, 'r-', linewidth=2, label=f'Environmental fit (R²={r2_env:.3f})')
    ax1.set_ylabel(f'Displacement ({units})')
    ax1.set_title(f'{section_name}: Original Time Series & Environmental Component')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Subplot 2: Residuals after removing environment + trend fit
    ax2 = axes[1]
    ax2.plot(dates, resid_env, 'o-', alpha=0.6, label='Residual (env removed)', 
             markersize=4, color='blue')
    ax2.plot(dates, fitted_trend, 'g-', linewidth=2, label='Polynomial trend fit (deg=2)')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel(f'Displacement ({units})')
    ax2.set_title(f'{section_name}: Residual after Environmental Removal & Trend Fit\n{metrics_text}')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Add text box with velocity and acceleration in the corner
    textstr = f'Velocity: {velocity:.3f} ± {sigma_v:.3f} {units}/yr\nAcceleration: {acceleration:.3f} ± {sigma_a:.3f} {units}/yr²'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Subplot 3: Final residuals (after removing environment + trend)
    ax3 = axes[2]
    ax3.plot(dates, resid_full, 'o-', alpha=0.7, label='Final residual', 
             markersize=4, color='purple')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel(f'Displacement ({units})')
    ax3.set_title(f'{section_name}: Final Residual (Environment + Trend Removed)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Rotate x-axis labels for better readability
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot residuals from fit and actual time series for each section"
    )
    parser.add_argument("--csv", required=True, help="Input CSV with Date + S1..SN displacement")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--units", choices=["mm", "m"], default="mm", 
                       help="Units of displacement in CSV (default mm)")
    parser.add_argument("--temp-col", default=None, 
                       help="Name of temperature column in CSV (optional)")
    parser.add_argument("--sections", default=None, 
                       help="Comma-separated list of sections to plot (e.g., 'S1,S5,S10'). If not specified, plots all sections.")
    
    args = parser.parse_args()
    
    # Read CSV
    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    date_col = _parse_date_col(df)
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Gather section columns
    sec_cols = [c for c in df.columns if re.fullmatch(r"S\d+", c)]
    if not sec_cols:
        raise SystemExit("No section columns S1..SN found in CSV.")
    sec_cols = sorted(sec_cols, key=lambda s: int(s[1:]))
    
    # Filter sections if specified
    if args.sections:
        requested_sections = [s.strip() for s in args.sections.split(',')]
        sec_cols = [s for s in sec_cols if s in requested_sections]
        if not sec_cols:
            raise SystemExit(f"None of the requested sections found in CSV: {requested_sections}")
        print(f"Processing {len(sec_cols)} requested sections: {', '.join(sec_cols)}")
    else:
        print(f"Processing all {len(sec_cols)} sections")
    
    # Units to mm if needed
    Y = df[sec_cols].astype(float).to_numpy()
    if args.units == "m":
        Y = Y * 1000.0
        units_label = "mm"
    else:
        units_label = args.units
    
    # Time arrays
    dates = pd.to_datetime(df[date_col].values)
    t_days = (dates - dates.min()).days.astype(float)
    t_years = t_days / 365.2425
    
    # Temperature (optional)
    temp_all = None
    if args.temp_col and args.temp_col in df.columns:
        temp_all = df[args.temp_col].astype(float).to_numpy()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each section
    print(f"\nGenerating plots...")
    for idx, sec in enumerate(sec_cols):
        y = Y[:, idx]
        
        # Step 1: Fit and remove environmental component
        fitted_env, resid_env, r2_env = fit_and_remove_environment(t_days, y, temp_all)
        
        # Step 2: Fit polynomial trend to residuals
        fitted_trend, velocity, acceleration, sigma_v, sigma_a = fit_trend(t_years, resid_env, deg=2)
        
        # Step 3: Final residuals (env + trend removed)
        resid_full = resid_env - fitted_trend
        
        # Plot
        output_path = output_dir / f"{sec}_residuals.png"
        plot_section_residuals(
            section_name=sec,
            dates=dates,
            y_original=y,
            fitted_env=fitted_env,
            resid_env=resid_env,
            fitted_trend=fitted_trend,
            resid_full=resid_full,
            r2_env=r2_env,
            velocity=velocity,
            acceleration=acceleration,
            sigma_v=sigma_v,
            sigma_a=sigma_a,
            output_path=output_path,
            units=units_label
        )
    
    print(f"\n✓ Done! Generated {len(sec_cols)} plots in {output_dir}")


if __name__ == "__main__":
    main()

