#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute coherence statistics per ROI (GeoJSON) from Miaplpy/MintPy ifgramStack.h5.

Outputs two CSV files with per-ROI stats for:
  (A) the whole period (all interferograms)
  (B) only interferograms whose *second* date is within the last N days
      (default: 90) relative to the max date in the stack.

Stats per ROI (columns):
  - roi
  - n_pixels_used   (count of finite coherence samples used = pixels * IFGs)
  - n_ifgs_used     (time slices used after the filter)
  - mean_coh
  - median_coh
  - var_coh
  - std_coh
  - min_coh
  - max_coh

Dependencies:
    pip install mintpy h5py numpy geopandas shapely pyproj scipy pandas

Example:
    python coherence_stats_by_roi.py \
      --ifgram /insar-data/PROJECT/miaplpy/network_delaunay_4/inputs/ifgramStack.h5 \
      --polygons /insar-data/ROIs.geojson \
      --geometry /insar-data/PROJECT/miaplpy/network_delaunay_4/inputs/geometryRadar.h5 \
      --out-prefix /insar-data/PROJECT/metrics/coh_stats \
      --last-days 90
"""

import argparse
import itertools
from pathlib import Path
import warnings
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid
from pyproj import Transformer
from mintpy.utils import readfile


# ==================== Lon/Lat helpers ====================

def try_read_lonlat_from_geometry(geom_path: Path):
    """Try to read 2D longitude/latitude from a MintPy geometry*.h5 file."""
    if geom_path is None or not Path(geom_path).exists():
        return None, None
    lon_candidates = ("longitude", "/longitude", "lon", "/lon")
    lat_candidates = ("latitude", "/latitude", "lat", "/lat")
    lon = lat = None
    for name in lon_candidates:
        try:
            lon, _ = readfile.read(str(geom_path), datasetName=name)
            break
        except Exception:
            continue
    for name in lat_candidates:
        try:
            lat, _ = readfile.read(str(geom_path), datasetName=name)
            break
        except Exception:
            continue
    if lon is not None and lat is not None and np.asarray(lon).ndim == 2 and np.asarray(lat).ndim == 2:
        return np.array(lon, dtype=np.float64), np.array(lat, dtype=np.float64)
    return None, None


def lonlat_from_corner_refs(h5_path: Path, shape_rc):
    """Reconstruct lon/lat (R,C) from lat_ref1..4 / lon_ref1..4 via bilinear interpolation."""
    R, C = shape_rc
    _, atr = readfile.read(str(h5_path))  # only need attributes
    attrs = {k.lower(): v for k, v in dict(atr).items()}
    try:
        lats = [float(attrs[f'lat_ref{i}']) for i in range(1, 5)]
        lons = [float(attrs[f'lon_ref{i}']) for i in range(1, 5)]
    except Exception as e:
        raise RuntimeError("lat_ref*/lon_ref* not found; provide geometry*.h5 or corner refs in H5.") from e

    v = np.linspace(0.0, 1.0, R)[:, None]
    u = np.linspace(0.0, 1.0, C)[None, :]

    def bilinear(c00, c10, c01, c11):
        return ((1 - u) * (1 - v) * c00 +
                u * (1 - v) * c10 +
                (1 - u) * v * c01 +
                u * v * c11)

    def score_grid(LON, LAT):
        dlon_x = np.nanmedian(np.diff(LON, axis=1))
        dlat_y = np.nanmedian(np.diff(LAT, axis=0))
        flips_x = int(np.sum(np.sign(np.diff(LON, axis=1)) != np.sign(dlon_x)))
        flips_y = int(np.sum(np.sign(np.diff(LAT, axis=0)) != np.sign(dlat_y)))
        dlon_dy = np.nanmedian(np.diff(LON, axis=0))
        dlat_dx = np.nanmedian(np.diff(LAT, axis=1))
        det_sign = np.sign(dlon_x * dlat_y - dlon_dy * dlat_dx)
        return (flips_x + flips_y) + (0 if det_sign != 0 else 1000), abs(dlon_x) + abs(dlat_y)

    best = None
    corners = list(zip(lons, lats))  # (lon, lat)
    for perm in itertools.permutations(corners, 4):
        (lon00, lat00), (lon10, lat10), (lon01, lat01), (lon11, lat11) = perm
        LON = bilinear(lon00, lon10, lon01, lon11)
        LAT = bilinear(lat00, lat10, lat01, lat11)
        lon_ok = (min(lons) - 1.0) <= float(np.nanmin(LON)) <= float(np.nanmax(LON)) <= (max(lons) + 1.0)
        lat_ok = (min(lats) - 1.0) <= float(np.nanmin(LAT)) <= float(np.nanmax(LAT)) <= (max(lats) + 1.0)
        if not (lon_ok and lat_ok):
            continue
        score = score_grid(LON, LAT)
        if (best is None) or (score < best[0]):
            best = (score, (LON, LAT))
    if best is None:
        raise RuntimeError("Failed to determine a consistent corner assignment for lon/lat.")
    return best[1]


def project_points(lon2d, lat2d, metric_crs: str):
    """Project lon/lat (2D) to X/Y in metric CRS."""
    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    X, Y = transformer.transform(lon2d, lat2d)
    return np.asarray(X), np.asarray(Y)


# ==================== ROI handling ====================

def load_roi_gdf(path: Path, declared_crs: str, to_crs: str):
    """
    Load polygons, normalize CRS (CRS84 -> EPSG:4326), fix invalid geometries,
    and reproject to 'to_crs'. Keeps individual features (no dissolve).
    """
    gdf = gpd.read_file(path)
    # Normalize CRS: CRS84 ~= EPSG:4326 (lon/lat axis order)
    if gdf.crs is None:
        gdf = gdf.set_crs(declared_crs)
    else:
        if "CRS84" in str(gdf.crs).upper():
            gdf = gdf.set_crs("EPSG:4326")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf.to_crs(to_crs)

    # Pick a name column
    name_fields = ["Name", "name", "roi", "ROI", "section", "Section", "id", "ID"]
    def pick_name(row):
        for nf in name_fields:
            if nf in row and str(row[nf]).strip() not in ["", "None", "nan"]:
                return str(row[nf])
        return f"ROI_{row.name}"

    gdf["__roi_name__"] = gdf.apply(pick_name, axis=1)
    return gdf[["__roi_name__", "geometry"]].copy()


def raster_points_mask_for_roi(X, Y, roi_geom):
    """
    Given projected X/Y (2D) in the same CRS as roi_geom, return a boolean mask (R,C)
    of points strictly inside the ROI (boundary excluded).
    """
    # Use vectorized contains if available
    try:
        from shapely import vectorized
        inside = vectorized.contains(roi_geom, X, Y)
        return inside
    except Exception:
        pass

    # Fallback: point-in-polygon via GeoPandas (slower)
    R, C = X.shape
    pts = gpd.GeoSeries(gpd.points_from_xy(X.ravel(), Y.ravel()), crs=None)
    gdf_pts = gpd.GeoDataFrame(geometry=pts)
    # Ensure CRS is same; set from polygon
    gdf_pts.set_crs(roi_geom.crs, allow_override=True, inplace=True)
    inside = gdf_pts.within(roi_geom).values.reshape(R, C)
    return inside


# ==================== ifgramStack readers ====================

def read_coherence_stack(h5_path: Path):
    """
    Return (coh, attrs) where coh has shape (M, R, C): M interferograms.
    Tries common dataset names and falls back to scanning the file.
    """
    ds_names = ["coherence", "/coherence", "coh", "/coh", "coh_stack", "/coh_stack"]
    for dsn in ds_names:
        try:
            arr, atr = readfile.read(str(h5_path), datasetName=dsn)
            arr = np.asarray(arr)
            if arr.ndim == 3:
                return arr, dict(atr)
        except Exception:
            continue
    # Fall back: raw scan for a 3D float dataset with 'coh' or 'coherence' in the name
    with h5py.File(h5_path, "r") as f:
        candidates = []
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj
                if data.ndim == 3 and np.issubdtype(data.dtype, np.floating):
                    if "coh" in name.lower() or "coherence" in name.lower():
                        candidates.append(name)
        f.visititems(lambda name, obj: visit(name, obj))
        for name in candidates:
            arr = f[name][:]
            atr = {k: v for k, v in f.attrs.items()}
            if arr.ndim == 3:
                return np.asarray(arr), atr
    raise RuntimeError("Could not find a 3D coherence dataset in the ifgramStack file.")


# -------- robust date utilities (fixes your error) --------

def _to_str(x):
    import numpy as np
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)

def _parse_date_any(s):
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    raise ValueError(f"Unrecognized date format: {s!r}")

def read_ifg_second_dates(h5_path: Path):
    """
    Return np.array of datetimes (length M), one per interferogram = the *second* date.
    Handles multiple layouts:
      A) date (N) + date12 (M,2) integers (indices into date)
      B) ifgramList/date12 as strings 'YYYYMMDD_YYYYMMDD' or 'YYYYMMDD-YYYYMMDD'
      C) datasets that are Nx2 arrays of strings/bytes with start/end dates
    """
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())

        # ---- Case A: acquisitions list + pair indices ----
        for date_key in ("date", "dateList"):
            if date_key in keys:
                dates_ds = f[date_key][:]
                dates_arr = np.squeeze(dates_ds)
                if dates_arr.ndim == 1:
                    acquisitions = [_to_str(v) for v in dates_arr.tolist()]
                    acquisitions_dt = [_parse_date_any(s) for s in acquisitions]
                    for d12_key in ("date12", "ifgramList"):
                        if d12_key in keys:
                            d12 = f[d12_key][:]
                            # If indices
                            if d12.ndim == 2 and d12.dtype.kind in ("i", "u") and d12.shape[1] >= 2:
                                idx2 = np.maximum(d12[:, 0], d12[:, 1]).astype(int)
                                return np.array([acquisitions_dt[i] for i in idx2])
                            # If strings like 'YYYYMMDD_YYYYMMDD'
                            if d12.dtype.kind in ("S", "U"):
                                vals = [_to_str(x) for x in d12.tolist()]
                                seconds = []
                                for v in vals:
                                    sep = "_" if "_" in v else "-" if "-" in v else None
                                    if sep is None:
                                        raise ValueError(f"Unrecognized pair string: {v!r}")
                                    s2 = v.split(sep)[1]
                                    seconds.append(_parse_date_any(s2))
                                return np.array(seconds)

        # ---- Case C: a direct Nx2 array of strings/bytes with start/end dates ----
        for key in ("ifgramList", "date12", "date"):
            if key in keys:
                ds = f[key][:]
                arr = np.asarray(ds)
                if arr.ndim == 2 and arr.shape[1] >= 2 and arr.dtype.kind in ("S", "U", "O"):
                    seconds = [_parse_date_any(_to_str(v[1])) for v in arr]
                    return np.array(seconds)

        # ---- Fallback: search any string dataset with pair strings ----
        string_like = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in ("S","U"):
                string_like.append(name)
        f.visititems(lambda n, o: visitor(n, o))

        for name in string_like:
            vals = [_to_str(v) for v in f[name][:].tolist()]
            ok, seconds = True, []
            for v in vals:
                sep = "_" if "_" in v else "-" if "-" in v else None
                if sep is None:
                    ok = False
                    break
                parts = v.split(sep)
                if len(parts) != 2:
                    ok = False
                    break
                try:
                    seconds.append(_parse_date_any(parts[1]))
                except Exception:
                    ok = False
                    break
            if ok and seconds:
                return np.array(seconds)

    raise RuntimeError("Could not determine interferogram 'second' dates from the HDF5 structure.")


# ==================== Statistics ====================

def stats_for_roi(coh_stack_sel):
    """
    coh_stack_sel: array (M, R, C) masked for ROI and (optionally) time-filtered.
    We flatten both spatial and time slices we are using.
    """
    x = np.asarray(coh_stack_sel).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n_pixels_used=0, n_ifgs_used=0,
                    mean_coh=np.nan, median_coh=np.nan, var_coh=np.nan,
                    std_coh=np.nan, min_coh=np.nan, max_coh=np.nan)
    return dict(
        n_pixels_used=x.size,        # pixels * interferograms actually used
        mean_coh=float(np.nanmean(x)),
        median_coh=float(np.nanmedian(x)),
        var_coh=float(np.nanvar(x, ddof=0)),
        std_coh=float(np.nanstd(x, ddof=0)),
        min_coh=float(np.nanmin(x)),
        max_coh=float(np.nanmax(x)),
    )


# ==================== Main ====================

def main():
    ap = argparse.ArgumentParser(description="Per-ROI coherence stats from ifgramStack.h5 over full period and last N days.")
    ap.add_argument("--ifgram", required=True, type=Path, help="Path to ifgramStack.h5 (Miaplpy/MintPy).")
    ap.add_argument("--polygons", required=True, type=Path, help="GeoJSON with ROI polygons.")
    ap.add_argument("--geometry", type=Path, default=None, help="Optional geometry*.h5 with longitude/latitude.")
    ap.add_argument("--polygons-crs", type=str, default="EPSG:4326", help="CRS to assume if polygons lack one (CRS84 ≈ EPSG:4326).")
    ap.add_argument("--metric-crs", type=str, default="EPSG:3857", help="Projected CRS for point-in-polygon tests (e.g., EPSG:31983 for São Paulo).")
    ap.add_argument("--last-days", type=int, default=90, help="Window for the 'recent' period (default: 90 days).")
    ap.add_argument("--out-prefix", required=True, type=Path, help="Output prefix for CSV files.")
    args = ap.parse_args()

    # 1) Read coherence stack
    coh, atr = read_coherence_stack(args.ifgram)   # (M,R,C)
    M, R, C = coh.shape

    # 2) Get per-interferogram 'second' dates (robust)
    second_dates = read_ifg_second_dates(args.ifgram)  # (M,)
    if len(second_dates) != M:
        raise RuntimeError(f"Inconsistent sizes: coherence stack M={M} but found {len(second_dates)} date entries.")

    # 3) Build lon/lat and projected coordinates for point-in-polygon
    lon2d, lat2d = (None, None)
    if args.geometry:
        lon2d, lat2d = try_read_lonlat_from_geometry(args.geometry)
    if lon2d is None or lat2d is None:
        lon2d, lat2d = lonlat_from_corner_refs(args.ifgram, (R, C))
    X, Y = project_points(lon2d, lat2d, args.metric_crs)

    # 4) Load ROI polygons (keep features separate)
    roi_gdf = load_roi_gdf(args.polygons, args.polygons_crs, args.metric_crs)

    # 5) Build time filters
    max_date = np.max(second_dates)
    cutoff = max_date - timedelta(days=int(args.last_days))
    mask_last = (second_dates >= cutoff)
    mask_full = np.ones_like(mask_last, dtype=bool)

    # 6) Compute stats per ROI
    rows_full = []
    rows_last = []

    for _, row in roi_gdf.iterrows():
        name = row["__roi_name__"]
        geom = row["geometry"]
        inside_mask = raster_points_mask_for_roi(X, Y, geom)  # (R,C)

        # Mask coherence outside ROI: set to NaN so it won't affect stats
        coh_roi = np.where(inside_mask[None, :, :], coh, np.nan)  # (M,R,C)

        # --- FULL PERIOD ---
        sel_full = coh_roi[mask_full, :, :]
        st_full = stats_for_roi(sel_full)
        st_full["n_ifgs_used"] = int(np.count_nonzero(mask_full))
        rows_full.append({"roi": name, **st_full})

        # --- LAST N DAYS ---
        if np.count_nonzero(mask_last) == 0:
            st_last = dict(n_pixels_used=0, n_ifgs_used=0,
                           mean_coh=np.nan, median_coh=np.nan, var_coh=np.nan,
                           std_coh=np.nan, min_coh=np.nan, max_coh=np.nan)
        else:
            sel_last = coh_roi[mask_last, :, :]
            st_last = stats_for_roi(sel_last)
            st_last["n_ifgs_used"] = int(np.count_nonzero(mask_last))
        rows_last.append({"roi": name, **st_last})

    # 7) Write CSVs
    out_full = Path(f"{args.out_prefix}_full_period.csv")
    out_last = Path(f"{args.out_prefix}_last{args.last_days}d.csv")

    pd.DataFrame(rows_full).to_csv(out_full, index=False)
    pd.DataFrame(rows_last).to_csv(out_last, index=False)

    print(f"Saved: {out_full}")
    print(f"Saved: {out_last}")


if __name__ == "__main__":
    main()
