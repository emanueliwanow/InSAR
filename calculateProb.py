#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge section risk scoring from InSAR time series (1=low … 5=high)
----------------------------------------------------------------------
Inputs
  - CSV with one date/time column and columns S1..SN for section LOS displacement
    (units: mm recommended; if meters, use --units m to convert).
  - Pillar positions as pairs like "'S2,S3','S4,S5',..." meaning a pillar
    is located between S2 and S3, etc. These pairs define span boundaries.

Methodology (condensed from the prior discussion):
  1) Optional environmental/thermal detrending per section using either a
     temperature column named "Temp" (if present) or a seasonal proxy
     (annual & semiannual sin/cos). Compute R^2_T (explainability) and use
     residuals for risk indicators.
  2) Indicators (normalized to 0–1):
       - S_v: |velocity| vs threshold (with t-stat weighting)
       - S_a: |acceleration| vs threshold (with t-stat weighting)
       - S_cp: recent change/step magnitude within a K-day window
       - S_sp: spatial consistency with neighbors in the same span (last 6 mo)
       - S_temp: protective factor = 1 - R^2_T
  3) Confidence weight W in [0,1] (data quantity & noise):
       W = clip(min(1, N/40) * exp(-sigma_v / tau_sigma), 0, 1)
  4) Aggregate: S = 0.30*S_v + 0.20*S_a + 0.25*S_cp + 0.15*S_sp + 0.10*S_temp
     Then S* = W * S. Map to class: Class = 1 + round(4 * clip(S*,0,1)).

Notes
  - This is a practical, dependency-light implementation using numpy/pandas.
  - Robust polynomial fits use IRLS (Huber). Change-point is a simple two-window
    step detector in the last K days (no external packages).
  - Spatial neighbors are adjacent sections within the same span (no crossing
    pillars). You can widen the window with --neighbor-radius.
  - If your CSV contains a per-section temporal coherence or quality column,
    you can pass them via --qc-csv to refine W (optional; see stub in code).

Example
  python insar_bridge_risk.py \
    --csv /insar-data/Tocantins24meses_ts_noFilter.csv \
    --pillars "'S2,S3','S4,S5','S6,S7','S8,S9','S10,S11','S12,S13','S14,S15','S16,S17','S18,S19','S20,S21','S22,S23','S24,S25','S26,S27','S28,S29'" \
    --out-csv /insar-data/bridge_risk.csv \
    --save-heatmap /insar-data/bridge_risk_heatmap.png

"""
from __future__ import annotations
import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- Utilities -------------------------

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


def parse_pillars(pillars_str: str) -> List[Tuple[str, str]]:
    """Parse pillars string like "'S2,S3','S4,S5'" into list [("S2","S3"), ...].
    Accepts semicolon- or comma-separated pairs; quotes optional.
    """
    if not pillars_str:
        return []
    # Extract pairs "S<number>,S<number>"
    pairs = re.findall(r"S\d+\s*,\s*S\d+", pillars_str)
    if not pairs:
        # Try splitting by separators and cleaning
        items = re.split(r"[;]", pillars_str)
        pairs = []
        for it in items:
            m = re.findall(r"S\d+", it)
            if len(m) >= 2:
                pairs.append(f"{m[0]},{m[1]}")
    out = []
    for p in pairs:
        a, b = [s.strip().replace("'", "").replace('"', '') for s in p.split(",")]
        out.append((a, b))
    return out


def spans_from_pillars(section_names: Sequence[str], pillar_pairs: List[Tuple[str, str]]
                       ) -> List[Tuple[int, int]]:
    """Return list of spans as index intervals [start_idx, end_idx] inclusive.
    A pillar between Sx,Sx+1 creates a break *after* Sx.
    """
    name_to_idx = {name: i for i, name in enumerate(section_names)}
    breaks = []  # break after index i
    for a, b in pillar_pairs:
        if a in name_to_idx and b in name_to_idx:
            i, j = name_to_idx[a], name_to_idx[b]
            # Expect j == i+1; if not, break after min(i,j)
            br = min(i, j)
            if br not in breaks:
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
    """Huber IRLS weights for residuals r / s, where s is MAD scale.
    c=1.345 gives ~95% efficiency under Gaussian noise.
    """
    # Robust scale via MAD
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
        # Fallback to zeros
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
    # Residual std (weighted)
    y_hat = X @ beta
    r = y - y_hat
    dof = max(1, len(y) - (deg + 1))
    sigma = math.sqrt((w * r * r).sum() / dof)
    return beta, sigma, w

# -------------------- Temperature / seasonality --------------------

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


def fit_and_remove_environment(t: np.ndarray, y: np.ndarray, temp: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Fit environmental/seasonal model and return residuals and R^2_T.
    R^2_T is the fraction explained by the environmental model.
    """
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

# -------------------- Change-point (recent step) --------------------

def recent_step_score(t_days: np.ndarray, resid: np.ndarray, K_days: int = 180,
                      m0_mm: float = 3.0) -> Tuple[float, float, Optional[pd.Timestamp]]:
    """Detect a recent step-like change in the last K days using a two-window mean diff.
    Returns (S_cp in 0..1, step_magnitude_mm, step_date).
    """
    mask = np.isfinite(t_days) & np.isfinite(resid)
    t_days = t_days[mask]
    resid = resid[mask]
    if len(t_days) < 6:
        return 0.0, 0.0, None
    # Focus on last K days
    tmax = t_days.max()
    recent_mask = t_days >= (tmax - K_days)
    if recent_mask.sum() < 4:
        # If too few, extend window to last third of data
        q = np.quantile(t_days, 2/3)
        recent_mask = t_days >= q
    idx = np.where(recent_mask)[0]
    if len(idx) < 4:
        return 0.0, 0.0, None
    # Split recent window at its midpoint
    mid = idx[0] + len(idx) // 2
    y1 = resid[idx[0]:mid]
    y2 = resid[mid:idx[-1] + 1]
    if len(y1) < 2 or len(y2) < 2:
        return 0.0, 0.0, None
    m1, m2 = float(np.mean(y1)), float(np.mean(y2))
    step = abs(m2 - m1)  # mm if y is mm
    # Age ~ distance from split midpoint to end
    t_mid = (t_days[mid] + t_days[mid - 1]) / 2.0 if mid - idx[0] > 0 else t_days[mid]
    age = max(0.0, (tmax - t_mid))
    # Score: magnitude vs m0 and exponential decay with age
    S_mag = min(1.0, step / max(1e-6, m0_mm))
    S_age = math.exp(-age / float(K_days))
    S_cp = S_mag * S_age
    # Approximate step date
    # Find first index after midpoint as change date
    step_date = None
    return S_cp, step, step_date

# -------------------- Spatial consistency --------------------

def spatial_consistency(section_idx: int, span_bounds: Tuple[int, int],
                        v_recent: Dict[int, float], sigma_recent: Dict[int, float],
                        neighbor_radius: int = 1) -> float:
    """Fraction of neighbors in the same span with consistent recent slope.
    Consistency = same sign and |Δv| <= 1 * pooled sigma.
    """
    i0, i1 = span_bounds
    if i0 == i1:
        return 0.0
    i = section_idx
    left = max(i0, i - neighbor_radius)
    right = min(i1, i + neighbor_radius)
    # Exclude itself
    neighbors = [j for j in range(left, right + 1) if j != i]
    if not neighbors:
        return 0.0
    vi = v_recent.get(i, np.nan)
    si = sigma_recent.get(i, np.nan)
    if not np.isfinite(vi) or not np.isfinite(si) or si <= 0:
        return 0.0
    ok = 0
    for j in neighbors:
        vj = v_recent.get(j, np.nan)
        sj = sigma_recent.get(j, np.nan)
        if not np.isfinite(vj) or not np.isfinite(sj) or sj <= 0:
            continue
        same_sign = (vi >= 0 and vj >= 0) or (vi < 0 and vj < 0)
        spooled = math.sqrt(si * si + sj * sj)
        similar_mag = abs(vi - vj) <= spooled
        ok += 1 if (same_sign and similar_mag) else 0
    return ok / max(1, len(neighbors))

# -------------------- Scoring --------------------
@dataclass
class Scores:
    S_v: float
    S_a: float
    S_cp: float
    S_sp: float
    S_temp: float
    W: float
    S: float
    S_star: float
    risk_class: int


@dataclass
class FitMetrics:
    v_mm_per_yr: float
    a_mm_per_yr2: float
    sigma_v: float
    sigma_a: float
    t_v: float
    t_a: float
    r2_temp: float
    cp_mm: float


def score_section(t_days: np.ndarray, y_mm: np.ndarray, temp: Optional[np.ndarray],
                  tau_v=5.0, tau_a=1.0, t_min=2.0, m0=3.0, K=180,
                  tau_sigma=1.0,
                  v_recent_mm_per_yr: Optional[float] = None,
                  sigma_recent: Optional[float] = None) -> Tuple[Scores, FitMetrics]:
    """Compute all indicators and the final risk score for one section."""
    # 1) Remove environmental component
    resid, r2_T = fit_and_remove_environment(t_days, y_mm, temp)

    # Time in years for polynomial fits
    t_years = (t_days - t_days.min()) / 365.2425

    # 2) Robust poly fit: velocity & acceleration on residuals
    beta2, sigma_resid, w = robust_polyfit(t_years, resid, deg=2)
    # y ~ b0 + b1*t + b2*t^2 ; velocity = b1 (mm/yr), acceleration = 2*b2 (mm/yr^2)
    b0, b1, b2 = beta2.tolist()
    v = float(b1)
    a = float(2.0 * b2)

    # Crude standard errors from weighted LS (approx.)
    X = np.column_stack([np.ones_like(t_years), t_years, t_years**2])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    try:
        cov = np.linalg.inv(XtWX) * sigma_resid**2
        se_b1 = math.sqrt(max(0.0, cov[1, 1]))
        se_b2 = math.sqrt(max(0.0, cov[2, 2]))
    except np.linalg.LinAlgError:
        se_b1 = float("nan")
        se_b2 = float("nan")

    sigma_v = float(se_b1)
    sigma_a = float(2.0 * se_b2)
    t_v = abs(v) / (sigma_v + 1e-9)
    t_a = abs(a) / (sigma_a + 1e-9)

    # 3) Change-point in last K days
    S_cp, cp_mm, _ = recent_step_score(t_days, resid, K_days=K, m0_mm=m0)

    # 4) Spatial consistency (deferred: we inject v_recent & sigma_recent)
    # Here, just placeholders; real value set by caller after computing neighbors
    S_sp = 0.0

    # 5) Temperature-explainability
    S_temp = 1.0 - r2_T

    # 6) Indicator scores
    S_v = min(1.0, abs(v) / max(1e-6, tau_v)) * min(1.0, t_v / max(1e-6, t_min))
    S_a = min(1.0, abs(a) / max(1e-6, tau_a)) * min(1.0, t_a / max(1e-6, t_min))

    # Confidence weight (data count & noise)
    N = np.isfinite(y_mm).sum()
    W_conf = max(0.0, min(1.0, min(1.0, N / 40.0) * math.exp(- (sigma_v if np.isfinite(sigma_v) else 1.0) / max(1e-6, tau_sigma))))

    # Aggregate S (without S_sp yet)
    S = 0.30 * S_v + 0.20 * S_a + 0.25 * S_cp + 0.15 * S_sp + 0.10 * S_temp
    S_star = W_conf * S
    risk_class = int(1 + round(4 * max(0.0, min(1.0, S_star))))

    scores = Scores(S_v, S_a, S_cp, S_sp, S_temp, W_conf, S, S_star, risk_class)
    metrics = FitMetrics(v, a, sigma_v, sigma_a, t_v, t_a, r2_T, cp_mm)
    return scores, metrics

# -------------------- Main driver --------------------

def main():
    ap = argparse.ArgumentParser(description="Bridge section risk scoring from InSAR time series")
    ap.add_argument("--csv", required=True, help="Input CSV with Date + S1..SN displacement (mm by default)")
    ap.add_argument("--pillars", required=True,
                    help="Pillar pairs like: '\'S2,S3\',\'S4,S5\'' (quotes optional)")
    ap.add_argument("--out-csv", required=False, default=None, help="Output CSV with scores per section")
    ap.add_argument("--save-heatmap", required=False, default=None, help="Optional PNG path for 1xN risk heatmap")
    ap.add_argument("--units", choices=["mm", "m"], default="mm", help="Units of displacement in CSV (default mm)")
    ap.add_argument("--neighbor-radius", type=int, default=1, help="Neighbor radius within a span (default 1)")
    ap.add_argument("--K-days", type=int, default=180, help="Recent window for change-point (default 180)")
    ap.add_argument("--tau-v", type=float, default=5.0, help="Velocity threshold mm/yr (default 5)")
    ap.add_argument("--tau-a", type=float, default=1.0, help="Acceleration threshold mm/yr^2 (default 1)")
    ap.add_argument("--m0", type=float, default=3.0, help="Step magnitude baseline mm (default 3)")
    ap.add_argument("--t-min", type=float, default=2.0, help="Min t-stat for v/a weighting (default 2)")
    ap.add_argument("--tau-sigma", type=float, default=1.0, help="Noise scale for W (default 1 mm/yr)")
    ap.add_argument("--temp-col", default=None, help="Name of temperature column in CSV (optional)")
    ap.add_argument("--no-seasonal-proxy", action="store_true",
                    help="Disable seasonal sin/cos proxy if no temperature is provided")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    date_col = _parse_date_col(df)
    df = df.sort_values(date_col).reset_index(drop=True)

    # Gather section columns S1..SN in order
    sec_cols = [c for c in df.columns if re.fullmatch(r"S\d+", c)]
    if not sec_cols:
        raise SystemExit("No section columns S1..SN found in CSV.")
    sec_cols = sorted(sec_cols, key=lambda s: int(s[1:]))

    # Units to mm
    Y = df[sec_cols].astype(float).to_numpy()
    if args.units == "m":
        Y = Y * 1000.0

    # Time arrays
    t = pd.to_datetime(df[date_col].values)
    t_days = (t - t.min()).days.astype(float)

    # Temperature (optional)
    temp_all = None
    if args.temp_col and args.temp_col in df.columns:
        temp_all = df[args.temp_col].astype(float).to_numpy()
    elif not args.no_seasonal_proxy:
        # We'll build harmonics inside the model; no external temp array needed
        temp_all = None
    else:
        temp_all = None

    # Parse pillars and build spans
    pillars = parse_pillars(args.pillars)
    spans = spans_from_pillars(sec_cols, pillars)

    # Prepare outputs
    results = []
    # For spatial consistency: compute recent slopes for each section (last 180 days default)
    v_recent_map: Dict[int, float] = {}
    s_recent_map: Dict[int, float] = {}

    # Precompute recent window mask
    Kdays = int(args.K_days)
    tmax = float(t_days.max())
    recent_mask = t_days >= (tmax - Kdays)
    t_recent_years = (t_days[recent_mask] - t_days.min()) / 365.2425

    for idx, sec in enumerate(sec_cols):
        y = Y[:, idx]
        tr = t_recent_years
        yr = y[recent_mask]
        # Robust linear fit on recent window for spatial consistency
        b1 = np.nan
        se1 = np.nan
        if len(yr) >= 4 and len(tr) >= 4:
            beta1, sigma_r, w1 = robust_polyfit(tr, yr, deg=1)
            # y ~ c0 + c1 * t_years
            b1 = float(beta1[1])
            # Approximate se of slope
            Xr = np.column_stack([np.ones_like(tr), tr])
            Wr = np.diag(w1)
            XtWX = Xr.T @ Wr @ Xr
            try:
                cov = np.linalg.inv(XtWX) * sigma_r**2
                se1 = math.sqrt(max(0.0, cov[1, 1]))
            except np.linalg.LinAlgError:
                se1 = float("nan")
        v_recent_map[idx] = b1
        s_recent_map[idx] = se1 if np.isfinite(se1) else 1.0

    # Score each section
    for idx, sec in enumerate(sec_cols):
        y = Y[:, idx]
        temp_vec = temp_all if temp_all is None else temp_all
        scores, metrics = score_section(
            t_days=t_days,
            y_mm=y,
            temp=temp_vec,
            tau_v=args.tau_v,
            tau_a=args.tau_a,
            t_min=args.t_min,
            m0=args.m0,
            K=args.K_days,
            tau_sigma=args.tau_sigma,
        )
        # Inject spatial consistency based on recent slopes within span
        # Find span for this index
        span_idx = None
        for si, (a, b) in enumerate(spans):
            if a <= idx <= b:
                span_idx = si
                sbounds = (a, b)
                break
        if span_idx is None:
            sbounds = (0, len(sec_cols) - 1)
        S_sp = spatial_consistency(idx, sbounds, v_recent_map, s_recent_map, neighbor_radius=args.neighbor_radius)

        # Recompute aggregate S with S_sp now
        S = 0.40 * scores.S_v + 0.20 * scores.S_a + 0.20 * scores.S_cp + 0.10 * S_sp + 0.10 * scores.S_temp
        alpha = 0.2
        S_star = (alpha + (1-alpha)*scores.W) * S
        risk_class = int(1 + round(4 * max(0.0, min(1.0, S_star))))

        results.append({
            "Section": sec,
            "SpanID": spans.index(sbounds),
            "S_v": round(scores.S_v, 4),
            "S_a": round(scores.S_a, 4),
            "S_cp": round(scores.S_cp, 4),
            "S_sp": round(S_sp, 4),
            "S_temp": round(scores.S_temp, 4),
            "W": round(scores.W, 4),
            "S": round(S, 4),
            "S_star": round(S_star, 4),
            "RiskClass": risk_class,
            # Fit metrics for auditability
            "v_mm_per_yr": round(metrics.v_mm_per_yr, 3),
            "sigma_v": round(metrics.sigma_v, 3) if np.isfinite(metrics.sigma_v) else np.nan,
            "a_mm_per_yr2": round(metrics.a_mm_per_yr2, 3),
            "sigma_a": round(metrics.sigma_a, 3) if np.isfinite(metrics.sigma_a) else np.nan,
            "t_v": round(metrics.t_v, 3),
            "t_a": round(metrics.t_a, 3),
            "R2_temp": round(metrics.r2_temp, 3),
            "cp_mm": round(metrics.cp_mm, 3),
        })

    out = pd.DataFrame(results).sort_values("Section")

    # Print compact summary
    print("\nPer-section risk class (1–5):")
    print(out[["Section", "SpanID", "RiskClass", "S_star", "v_mm_per_yr", "a_mm_per_yr2"]].to_string(index=False))

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"\nSaved scores to: {args.out_csv}")

    if args.save_heatmap:
        # 1×N heatmap ordered by section index
        classes = (
            out.sort_values(
                by=["Section"],
                key=lambda s: s.astype(str).str.extract(r"(\d+)", expand=False).astype(int)
            )["RiskClass"]
            .to_numpy()
        )
        N = len(classes)
        fig, ax = plt.subplots(figsize=(max(8, N * 0.35), 1.6))
        im = ax.imshow(classes[np.newaxis, :], aspect="auto", vmin=1, vmax=5)
        ax.set_yticks([])
        ax.set_xticks(range(N))
        ax.set_xticklabels([f"S{i+1}" for i in range(N)], rotation=90)
        ax.set_title("Bridge Sections Risk (1=low, 5=high)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Risk Class")
        # Mark pillars as thin vertical lines
        # build break positions from spans
        breaks = [b for (_, b) in spans[:-1]]
        for br in breaks:
            ax.axvline(br + 0.5, color="k", linewidth=1, alpha=0.7)
        plt.tight_layout()
        fig.savefig(args.save_heatmap, dpi=200)
        plt.close(fig)
        print(f"Saved heatmap to: {args.save_heatmap}")


if __name__ == "__main__":
    main()
