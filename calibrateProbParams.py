#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate risk-scoring parameters from an InSAR bridge time-series CSV.

Input CSV format (like your file):
  - One date/time column (e.g., Date / Data / Timestamp)
  - Section columns named S1..SN with LOS displacement (mm by default; use --units m if meters)
  - Optional temperature column (use --temp-col to name it). If absent, uses seasonal harmonics.

Outputs (printed and optionally saved as JSON/YAML):
  - tau_v       : velocity threshold in mm/yr
  - tau_a       : acceleration threshold in mm/yr^2
  - tau_sigma   : uncertainty scale for confidence weight
  - m0          : baseline step magnitude for change-point scoring (mm)
  - K_days      : size of the recent window in days for change-point detection
  - neighbor_radius: suggested neighbor radius for spatial consistency
  - N_cap       : suggested acquisition-count cap for confidence

Method (data-driven):
  1) Build a "stable" reference set of sections with high environmental explainability (R2_T >= --stable-r2)
     and low absolute velocity (|v| < --stable-vel) and no recent change point. If too small, relax automatically.
  2) Compute distributions over the stable set and set parameters:
       tau_sigma = 75th percentile of sigma_v
       tau_v     = max( 99th percentile of |v| , 3 * median(sigma_v) )
       tau_a     = max( 99th percentile of |a| , 3 * median(sigma_a) )
       m0        = max( 3 * median(residual_RMSE) , 99th percentile of half-window mean diffs )
       K_days    = clamp(median_dt * 12, K_min, K_max) so that ~12 acquisitions fall in the recent window
       neighbor_radius: 1 by default; if spans are long (avg sections per span >= 6) suggest 2
       N_cap     = median number of valid acquisitions across sections, rounded up to nearest 5

Robustness:
  - Robust polynomial fits via IRLS (Huber). Velocity and acceleration come from the detrended residuals.
  - Change-point magnitude is estimated from a two-half mean difference inside the last K_days.

Example
  python calibrate_insar_risk_params.py \
    --csv /insar-data/Tocantins24meses_ts_noFilter.csv \
    --pillars "'S2,S3','S4,S5','S6,S7','S8,S9','S10,S11','S12,S13','S14,S15','S16,S17','S18,S19','S20,S21','S22,S23','S24,S25','S26,S27','S28,S29'" \
    --out-json /insar-data/risk_calibration.json
"""
from __future__ import annotations
import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------- Utilities -------------------------

def _parse_date_col(df: pd.DataFrame) -> str:
    candidates = ["date", "Date", "DATE", "Data", "timestamp", "Timestamp", "TIMESTAMP"]
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            return c
    c0 = df.columns[0]
    df[c0] = pd.to_datetime(df[c0])
    return c0


def parse_pillars(pillars_str: Optional[str]) -> List[Tuple[str, str]]:
    if not pillars_str:
        return []
    pairs = re.findall(r"S\d+\s*,\s*S\d+", pillars_str)
    out = []
    for p in pairs:
        a, b = [s.strip().replace("'", "").replace('"', '') for s in p.split(",")]
        out.append((a, b))
    return out


def spans_from_pillars(section_names: Sequence[str], pillar_pairs: List[Tuple[str, str]]
                       ) -> List[Tuple[int, int]]:
    name_to_idx = {name: i for i, name in enumerate(section_names)}
    breaks = []
    for a, b in pillar_pairs:
        if a in name_to_idx and b in name_to_idx:
            br = min(name_to_idx[a], name_to_idx[b])
            breaks.append(br)
    breaks = sorted(set(breaks))
    spans = []
    start = 0
    for br in breaks:
        spans.append((start, br))
        start = br + 1
    spans.append((start, len(section_names) - 1))
    return spans

# ---------------------- Robust regression ----------------------

def huber_weights(r: np.ndarray, c: float = 1.345) -> np.ndarray:
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
    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < deg + 1:
        return np.zeros(deg + 1), float("nan"), np.ones_like(t)
    X = np.vander(t, N=deg + 1, increasing=True)
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
    y_hat = X @ beta
    r = y - y_hat
    dof = max(1, len(y) - (deg + 1))
    sigma = math.sqrt((w * r * r).sum() / dof)
    return beta, sigma, w

# -------------------- Environment detrending --------------------

def design_matrix_temp(t_days: np.ndarray, temp: Optional[np.ndarray]) -> np.ndarray:
    tt = t_days.astype(float)
    omega1 = 2 * np.pi / 365.2425
    omega2 = 2 * omega1
    X = [np.ones_like(tt)]
    if temp is not None:
        T = np.asarray(temp, dtype=float)
        X += [T, T * T]
    X += [np.sin(omega1 * tt), np.cos(omega1 * tt), np.sin(omega2 * tt), np.cos(omega2 * tt)]
    return np.column_stack(X)


def fit_and_remove_environment(t: np.ndarray, y: np.ndarray, temp: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    mask = np.isfinite(t) & np.isfinite(y)
    if temp is not None:
        mask &= np.isfinite(temp)
    t, y = t[mask], y[mask]
    T = temp[mask] if temp is not None else None
    if len(t) < 6:
        return y.copy(), 0.0
    X = design_matrix_temp(t, T)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    resid = y - y_hat
    return resid, float(max(0.0, min(1.0, r2)))

# -------------------- Change-point proxy --------------------

def half_window_step(t_days: np.ndarray, y: np.ndarray, K_days: int) -> float:
    mask = np.isfinite(t_days) & np.isfinite(y)
    t_days = t_days[mask]
    y = y[mask]
    if len(t_days) < 6:
        return 0.0
    tmax = float(t_days.max())
    recent = t_days >= (tmax - K_days)
    idx = np.where(recent)[0]
    if len(idx) < 4:
        return 0.0
    mid = idx[0] + len(idx) // 2
    y1 = y[idx[0]:mid]
    y2 = y[mid:idx[-1] + 1]
    if len(y1) < 2 or len(y2) < 2:
        return 0.0
    return float(abs(np.mean(y2) - np.mean(y1)))

# -------------------- Main calibration --------------------

def main():
    ap = argparse.ArgumentParser(description="Calibrate risk-scoring parameters from bridge InSAR CSV")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--units", choices=["mm", "m"], default="mm", help="Units of displacement (default mm)")
    ap.add_argument("--temp-col", default=None, help="Optional temperature column name")
    ap.add_argument("--pillars", default=None, help="Optional pillar pairs string ")
    ap.add_argument("--stable-r2", type=float, default=0.6, help="R2 threshold for stable set")
    ap.add_argument("--stable-vel", type=float, default=1.0, help="|v| mm/yr threshold for stable set")
    ap.add_argument("--K-min", type=int, default=90, help="Min recent window days")
    ap.add_argument("--K-max", type=int, default=180, help="Max recent window days")
    ap.add_argument("--out-json", default=None, help="Optional path to save suggested params as JSON")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    date_col = _parse_date_col(df)
    df = df.sort_values(date_col).reset_index(drop=True)

    # Section columns
    sec_cols = [c for c in df.columns if re.fullmatch(r"S\d+", c)]
    if not sec_cols:
        raise SystemExit("No S1..SN columns found.")
    sec_cols = sorted(sec_cols, key=lambda s: int(s[1:]))

    # Data arrays
    Y = df[sec_cols].astype(float).to_numpy()
    if args.units == "m":
        Y = Y * 1000.0
    t = pd.to_datetime(df[date_col].values)
    t_days = (t - t.min()).days.astype(float)

    temp_vec = None
    if args.temp_col and args.temp_col in df.columns:
        temp_vec = df[args.temp_col].astype(float).to_numpy()

    # Choose K_days based on cadence: about 12 acquisitions inside window, bounded [K-min, K-max]
    if len(t) >= 3:
        dt = np.diff(t.values.astype('datetime64[D]')).astype(float)
        med_dt = float(np.median(dt)) if len(dt) else 12.0
    else:
        med_dt = 12.0
    K_target = int(round(med_dt * 12))
    K_days = max(args.K_min, min(args.K_max, K_target))

    # Fit per section: detrend environment, robust quad fit to residuals
    t_years = (t_days - t_days.min()) / 365.2425
    v, a, sigma_v, sigma_a, r2_T, rmse = [], [], [], [], [], []
    step_recent = []

    for i in range(Y.shape[1]):
        y = Y[:, i]
        resid, r2 = fit_and_remove_environment(t_days, y, temp_vec)
        r2_T.append(r2)
        beta2, sigma_r, w = robust_polyfit(t_years, resid, deg=2)
        b0, b1, b2 = beta2.tolist()
        vel = float(b1)                # mm/yr
        acc = float(2.0 * b2)          # mm/yr^2
        # approximate SEs
        X = np.column_stack([np.ones_like(t_years), t_years, t_years**2])
        W = np.diag(w)
        XtWX = X.T @ W @ X
        try:
            cov = np.linalg.inv(XtWX) * sigma_r**2
            se_v = float(math.sqrt(max(0.0, cov[1, 1])))
            se_a = float(math.sqrt(max(0.0, 4.0 * cov[2, 2])))
        except np.linalg.LinAlgError:
            se_v, se_a = float("nan"), float("nan")
        v.append(vel)
        a.append(acc)
        sigma_v.append(se_v)
        sigma_a.append(se_a)
        # residual RMSE per section
        rmse.append(float(np.sqrt(np.nanmean((resid - np.nanmean(resid))**2))))
        # change-point proxy magnitude in last K_days
        step_recent.append(half_window_step(t_days, resid, K_days))

    v = np.array(v)
    a = np.array(a)
    sigma_v = np.array(sigma_v)
    sigma_a = np.array(sigma_a)
    r2_T = np.array(r2_T)
    rmse = np.array(rmse)
    step_recent = np.array(step_recent)

    # Build stable set mask and auto-relax if needed
    mask_stable = (r2_T >= args.stable_r2) & (np.abs(v) < args.stable_vel) & (step_recent < np.nanpercentile(step_recent, 80))
    if mask_stable.sum() < max(5, int(0.2 * len(sec_cols))):
        # Relax velocity and R2 thresholds
        v_thresh = max(args.stable_vel, float(np.nanpercentile(np.abs(v), 40)))
        r2_thresh = min(args.stable_r2, float(np.nanpercentile(r2_T, 60)))
        mask_stable = (r2_T >= r2_thresh) & (np.abs(v) < v_thresh)
    if mask_stable.sum() == 0:
        mask_stable = np.isfinite(v)

    # Helper percentiles on stable set
    def P(x, p):
        x = x[mask_stable]
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan")
        return float(np.nanpercentile(x, p))

    med_sigma_v = P(sigma_v, 50)
    med_sigma_a = P(sigma_a, 50)

    # Parameter suggestions
    tau_sigma = P(sigma_v, 75)
    tau_v = max(P(np.abs(v), 99), 3 * med_sigma_v if np.isfinite(med_sigma_v) else 0.0)
    tau_a = max(P(np.abs(a), 99), 3 * med_sigma_a if np.isfinite(med_sigma_a) else 0.0)

    m0_from_rmse = 3.0 * P(rmse, 50)
    m0_from_steps = P(step_recent, 99)
    if not np.isfinite(m0_from_steps):
        m0_from_steps = float(np.nanpercentile(step_recent[np.isfinite(step_recent)], 95)) if np.isfinite(step_recent).any() else 3.0
    m0 = max(m0_from_rmse if np.isfinite(m0_from_rmse) else 3.0,
             m0_from_steps if np.isfinite(m0_from_steps) else 3.0)
    # fallback
    if not np.isfinite(tau_sigma) or tau_sigma <= 0:
        tau_sigma = float(np.nanpercentile(sigma_v[np.isfinite(sigma_v)], 75)) if np.isfinite(sigma_v).any() else 1.0
    if not np.isfinite(tau_v) or tau_v <= 0:
        tau_v = 5.0
    if not np.isfinite(tau_a) or tau_a <= 0:
        tau_a = 1.0
    if not np.isfinite(m0) or m0 <= 0:
        m0 = 3.0

    # Neighbor radius suggestion based on span lengths
    pillars = parse_pillars(args.pillars)
    spans = spans_from_pillars(sec_cols, pillars) if pillars else [(0, len(sec_cols) - 1)]
    span_lengths = [b - a + 1 for a, b in spans]
    avg_span_len = float(np.mean(span_lengths)) if span_lengths else len(sec_cols)
    neighbor_radius = 2 if avg_span_len >= 6 else 1

    # N cap suggestion
    N_valid_per_sec = np.sum(np.isfinite(Y), axis=0)
    N_med = int(np.median(N_valid_per_sec)) if N_valid_per_sec.size else 40
    # round up to nearest 5
    N_cap = int(int(np.ceil(N_med / 5.0)) * 5)
    if N_cap < 20:
        N_cap = 20

    # Summary
    result = {
        "tau_v_mm_per_yr": round(float(tau_v), 4),
        "tau_a_mm_per_yr2": round(float(tau_a), 4),
        "tau_sigma_mm_per_yr": round(float(tau_sigma), 4),
        "m0_step_mm": round(float(m0), 4),
        "K_days": int(K_days),
        "neighbor_radius": int(neighbor_radius),
        "N_cap": int(N_cap),
        "diagnostics": {
            "stable_count": int(mask_stable.sum()),
            "total_sections": int(len(sec_cols)),
            "median_sigma_v": round(float(med_sigma_v), 4) if np.isfinite(med_sigma_v) else None,
            "median_sigma_a": round(float(med_sigma_a), 4) if np.isfinite(med_sigma_a) else None,
            "avg_span_len": round(float(avg_span_len), 2),
            "median_dt_days": round(float(med_dt), 2),
        }
    }

    print("Suggested calibration parameters:\n")
    for k, vout in result.items():
        if k != "diagnostics":
            print(f"  {k}: {vout}")
    print("\nDiagnostics:")
    for k, vout in result["diagnostics"].items():
        print(f"  {k}: {vout}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
