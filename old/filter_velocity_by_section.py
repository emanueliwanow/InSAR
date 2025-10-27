#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter MintPy velocity.h5 by GeoJSON sections (robust CRS) + temporal coherence threshold.

Keeps pixels whose centers are strictly inside the (optionally shrunken) sections geometry
AND whose temporal coherence >= threshold. Writes:
  /velocity  -> NaN outside selection
  /mask      -> uint8 (1 kept, 0 removed)

Dependencies:
    pip install mintpy h5py numpy geopandas shapely

Example:
    python filter_velocity_sections_tc.py \
      --velocity /insar-data/PROJECT/miaplpy2/network_delaunay_4/velocity.h5 \
      --polygons /insar-data/PinheirosSP.geojson \
      --tc /insar-data/PROJECT/miaplpy2/network_delaunay_4/temporalCoherence.h5 \
      --tc-thr 0.7 \
      --geometry /insar-data/PROJECT/miaplpy2/network_delaunay_4/inputs/geometryRadar.h5 \
      --metric-crs EPSG:31983 \
      --shrink-eps-m 0.2 \
      --out /insar-data/PROJECT/velocity.sections_tc0p70.h5
"""
import argparse
import itertools
import warnings
from pathlib import Path

import h5py
import numpy as np
import geopandas as gpd
from shapely import make_valid
from mintpy.utils import readfile


# ----------------- I/O helpers -----------------
def read_2d_dataset(path: Path, dataset_candidates):
    """Read a 2D dataset via MintPy readfile with name guessing."""
    # try default (no datasetName)
    try:
        arr, atr = readfile.read(str(path))
        if np.asarray(arr).ndim == 2:
            return np.array(arr), dict(atr)
    except Exception:
        pass
    # try candidates
    for name in dataset_candidates:
        try:
            arr, atr = readfile.read(str(path), datasetName=name)
            if np.asarray(arr).ndim == 2:
                return np.array(arr), dict(atr)
        except Exception:
            continue
    raise RuntimeError(f"Could not read a 2D dataset from {path} using candidates: {dataset_candidates}.")


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
    """
    Reconstruct lon/lat (R,C) by bilinear interpolation from lat_ref1..4 / lon_ref1..4.
    Automatically chooses a consistent corner assignment.
    """
    R, C = shape_rc
    _, atr = readfile.read(str(h5_path))  # need attrs only
    attrs = {k.lower(): v for k, v in dict(atr).items()}
    try:
        lats = [float(attrs[f'lat_ref{i}']) for i in range(1, 5)]
        lons = [float(attrs[f'lon_ref{i}']) for i in range(1, 5)]
    except Exception as e:
        raise RuntimeError("lat_ref*/lon_ref* not found; need geometry*.h5 or corner refs.") from e

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


# ----------------- polygon loading & selection -----------------
def load_sections_as_metric(path: Path, declared_crs: str, metric_crs: str):
    """
    Load polygons, normalize CRS (CRS84 -> EPSG:4326), fix invalid geometries,
    dissolve, and return a single (Multi)Polygon in a metric CRS.
    """
    gdf = gpd.read_file(path)

    # Normalize CRS handling: CRS84 ~= EPSG:4326 (lon/lat axis order)
    if gdf.crs is None:
        gdf = gdf.set_crs(declared_crs)
    else:
        crs_str = str(gdf.crs).upper()
        if "CRS84" in crs_str:
            gdf = gdf.set_crs("EPSG:4326")

    # Fix invalid geometries
    def _valid(g):
        try:
            return make_valid(g)
        except Exception:
            return g.buffer(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["geometry"] = gdf.geometry.apply(_valid)

    # Keep polygonal and dissolve
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise RuntimeError("No polygon geometry found in sections file.")
    gdf = gdf.dissolve().to_crs(metric_crs).reset_index(drop=True)
    return gdf.geometry.iloc[0]  # shapely (Multi)Polygon in metric CRS


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Filter velocity.h5 by sections + temporal coherence >= threshold (center-in-polygon).")
    ap.add_argument("--velocity", required=True, type=Path, help="Path to existing velocity.h5")
    ap.add_argument("--polygons", required=True, type=Path, help="Sections GeoJSON (CRS84/EPSG:4326 or other)")
    ap.add_argument("--out", required=True, type=Path, help="Output velocity.h5 (filtered)")
    ap.add_argument("--geometry", type=Path, default=None, help="Optional MintPy geometry*.h5 with lon/lat")
    ap.add_argument("--dataset-name", type=str, default="velocity", help="Dataset name to read/write (default: velocity)")

    # Temporal coherence inputs
    ap.add_argument("--tc", type=Path, required=True, help="temporalCoherence.h5 path")
    ap.add_argument("--tc-dataset", type=str, default="temporalCoherence",
                    help="Dataset name for temporal coherence (tries common aliases).")
    ap.add_argument("--tc-thr", type=float, required=True,
                    help="Temporal coherence threshold (keep TC >= value).")

    # Robust polygon selection options
    ap.add_argument("--polygons-crs", type=str, default="EPSG:4326",
                    help="CRS if the file lacks one (CRS84 ≈ EPSG:4326).")
    ap.add_argument("--metric-crs", type=str, default="EPSG:3857",
                    help="Projected CRS for geometry ops (e.g., EPSG:31983 for São Paulo).")
    ap.add_argument("--shrink-eps-m", type=float, default=0.2,
                    help="Negative buffer (meters) to shrink polygons before selection (0 to disable).")
    args = ap.parse_args()

    # Read velocity and TC
    vel, atr = read_2d_dataset(args.velocity, dataset_candidates=(args.dataset_name, f"/{args.dataset_name}", "velocity", "/velocity"))
    R, C = vel.shape
    N = R * C

    tc2d, _ = read_2d_dataset(args.tc, dataset_candidates=(args.tc_dataset, f"/{args.tc_dataset}",
                                                           "temporalCoherence", "/temporalCoherence"))
    if tc2d.shape != (R, C):
        raise ValueError(f"Temporal coherence shape {tc2d.shape} != velocity shape {(R, C)}.")

    # Get lon/lat of pixel centers
    lon2d, lat2d = (None, None)
    if args.geometry:
        lon2d, lat2d = try_read_lonlat_from_geometry(args.geometry)
        if lon2d is not None:
            print("Using lon/lat from geometry file.")
    if lon2d is None or lat2d is None:
        print("Reconstructing lon/lat from corner references...")
        lon2d, lat2d = lonlat_from_corner_refs(args.velocity, (R, C))

    # Load sections as metric geometry and (optionally) shrink slightly
    poly_metric = load_sections_as_metric(args.polygons, args.polygons_crs, args.metric_crs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poly_for_test = poly_metric.buffer(-float(args.shrink_eps_m)) if args.shrink_eps_m > 0 else poly_metric

    # Build points in WGS84 then project to the metric CRS
    points_wgs84 = gpd.GeoDataFrame(
        {"idx": np.arange(N, dtype=np.int64)},
        geometry=gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()),
        crs="EPSG:4326",
    )
    points_metric = points_wgs84.to_crs(args.metric_crs)

    # Strict interior test for centers (boundary excluded)
    inside = points_metric.within(poly_for_test).values  # shape (N,)

    # Temporal coherence threshold
    tc_ok = (tc2d.ravel() >= float(args.tc_thr))

    # Combined selection
    sel = inside & tc_ok
    kept = int(sel.sum())
    print(f"Selected pixels: {kept} / {N} (inside sections & TC >= {args.tc_thr})")

    # Build outputs
    sel2d = sel.reshape(R, C)
    vel_out = vel.copy()
    vel_out[~sel2d] = np.nan
    out_mask = sel2d.astype(np.uint8)

    # Write output HDF5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing filtered velocity to: {out_path}")
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset(args.dataset_name, data=vel_out, compression="gzip", shuffle=True)
        # copy attrs to dataset and file
        for k, v in atr.items():
            try:
                dset.attrs[k] = v
            except Exception:
                pass
        for k, v in atr.items():
            try:
                f.attrs[k] = v
            except Exception:
                pass
        f.create_dataset("mask", data=out_mask, compression="gzip", shuffle=True)

    print("Done.")


if __name__ == "__main__":
    main()
