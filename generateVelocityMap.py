#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot velocity.h5 pixels clipped to GeoJSON polygons (no H5 output, image only).

Produces a single image (PNG/SVG/PDF) with:
  - velocity pixels (only where pixel centers are strictly inside polygons)
  - polygon outlines on top

Dependencies:
    pip install mintpy h5py numpy geopandas shapely matplotlib scipy pyproj

Example:
    python plot_velocity_sections.py \
      --velocity /insar-data/PROJECT/miaplpy2/network_delaunay_4/velocity.h5 \
      --polygons /insar-data/PinheirosSP.geojson \
      --geometry /insar-data/PROJECT/miaplpy2/network_delaunay_4/inputs/geometryRadar.h5 \
      --metric-crs EPSG:31983 \
      --shrink-eps-m 0.2 \
      --out /insar-data/PROJECT/figs/velocity_sections.png \
      --cmap RdBu_r --vmin -0.02 --vmax 0.02 --dpi 300
"""
import argparse
import itertools
import warnings
from pathlib import Path

import h5py
import numpy as np
import geopandas as gpd
from shapely import make_valid
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter  # <-- for lon/lat tick labels
from mintpy.utils import readfile
from pyproj import Transformer                # <-- projection to/from metric CRS
from scipy.interpolate import griddata        # <-- resampling


# ----------------- I/O helpers -----------------
def read_2d_dataset(path: Path, dataset_candidates):
    """Read a 2D dataset via MintPy readfile with name guessing."""
    try:
        arr, atr = readfile.read(str(path))
        if np.asarray(arr).ndim == 2:
            return np.array(arr), dict(atr)
    except Exception:
        pass
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


# ----------------- polygon loading -----------------
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

    def _valid(g):
        try:
            return make_valid(g)
        except Exception:
            return g.buffer(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["geometry"] = gdf.geometry.apply(_valid)

    gdf = gdf[gdf.geometry.notna() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise RuntimeError("No polygon geometry found in sections file.")
    gdf = gdf.dissolve().to_crs(metric_crs).reset_index(drop=True)
    return gdf.geometry.iloc[0]  # shapely (Multi)Polygon in metric CRS


# ----------------- helpers -----------------
def project_to_metric(lon2d, lat2d, metric_crs):
    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    X, Y = transformer.transform(lon2d, lat2d)
    return np.asarray(X), np.asarray(Y)

def inverse_to_lonlat(metric_crs):
    """Return a function (x,y)->(lon,lat) for tick label formatting."""
    inv = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    def fwd(x, y):
        lon, lat = inv.transform(x, y)
        return lon, lat
    return fwd

def estimate_native_spacing(X, Y):
    """Robust median spacing along rows/cols (meters)."""
    dx = np.nanmedian(np.diff(X, axis=1))
    dy = np.nanmedian(np.diff(Y, axis=0))
    return abs(dx), abs(dy)

def resample_to_square_grid(X, Y, Z, cell_size_m=None, bounds=None, method="nearest"):
    """
    Resample scattered (X,Y,Z) on a curvilinear grid to a regular square grid.

    Returns Xg, Yg, Zg suitable for pcolormesh (centers).
    """
    dx, dy = estimate_native_spacing(X, Y)
    if cell_size_m is None:
        cell_size_m = float(np.sqrt(dx * dy))

    if bounds is None:
        minx, maxx = np.nanmin(X), np.nanmax(X)
        miny, maxy = np.nanmin(Y), np.nanmax(Y)
    else:
        minx, miny, maxx, maxy = bounds

    xs = np.arange(minx, maxx + cell_size_m, cell_size_m)
    ys = np.arange(miny, maxy + cell_size_m, cell_size_m)
    Xg, Yg = np.meshgrid(xs, ys)

    points = np.column_stack([X.ravel(), Y.ravel()])
    values = np.asarray(Z).ravel()
    good = np.isfinite(values)
    # Use 'nearest' so we don't blur; will fill everywhere in convex hull.
    Zg = griddata(points[good], values[good], (Xg, Yg), method=method)
    return Xg, Yg, Zg


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Plot velocity pixels inside GeoJSON polygons (single image).")
    ap.add_argument("--velocity", required=True, type=Path, help="Path to velocity.h5")
    ap.add_argument("--polygons", required=True, type=Path, help="Sections GeoJSON (CRS84/EPSG:4326 or other)")
    ap.add_argument("--out", required=True, type=Path, help="Output figure path (e.g., .png, .pdf, .svg)")
    ap.add_argument("--geometry", type=Path, default=None, help="Optional MintPy geometry*.h5 with lon/lat")
    ap.add_argument("--dataset-name", type=str, default="velocity", help="Dataset name to read (default: velocity)")

    # Robust polygon selection options
    ap.add_argument("--polygons-crs", type=str, default="EPSG:4326",
                    help="CRS if the file lacks one (CRS84 ≈ EPSG:4326).")
    ap.add_argument("--metric-crs", type=str, default="EPSG:3857",
                    help="Projected CRS for geometry ops (e.g., EPSG:31983 for São Paulo).")
    ap.add_argument("--shrink-eps-m", type=float, default=0.2,
                    help="Negative buffer (meters) to shrink polygons before selection (0 to disable).")

    # Plot styling
    ap.add_argument("--cmap", type=str, default="RdBu_r", help="Matplotlib colormap name.")
    ap.add_argument("--vmin", type=float, default=None, help="Colorbar min (auto if omitted).")
    ap.add_argument("--vmax", type=float, default=None, help="Colorbar max (auto if omitted).")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving raster outputs.")
    ap.add_argument("--figsize", type=float, nargs=2, default=(8, 6), help="Figure size inches: W H.")
    ap.add_argument("--title", type=str, default=None, help="Optional custom plot title.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Alpha for inside-polygon pixels.")
    ap.add_argument("--poly-edgecolor", type=str, default="k", help="Polygon edge color.")
    ap.add_argument("--poly-linewidth", type=float, default=1.2, help="Polygon line width.")
    ap.add_argument("--square-cell-m", type=float, default=None,
                    help="Target square cell size in meters (default: geometric mean of native spacings).")
    args = ap.parse_args()

    # Read velocity
    vel, atr = read_2d_dataset(
        args.velocity,
        dataset_candidates=(args.dataset_name, f"/{args.dataset_name}", "velocity", "/velocity")
    )
    R, C = vel.shape
    N = R * C

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
    import geopandas as gpd
    points_wgs84 = gpd.GeoDataFrame(
        {"idx": np.arange(N, dtype=np.int64)},
        geometry=gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()),
        crs="EPSG:4326",
    )
    points_metric = points_wgs84.to_crs(args.metric_crs)

    # Strict interior test for centers (boundary excluded)
    inside = points_metric.within(poly_for_test).values  # shape (N,)
    inside2d = inside.reshape(R, C)

    # Mask outside polygons and convert to mm/yr
    vel_plot = np.ma.masked_where(~inside2d | ~np.isfinite(vel), vel) * 1000.0

    # Project lon/lat grid to metric CRS for plotting
    X, Y = project_to_metric(lon2d, lat2d, args.metric_crs)

    # --- RESAMPLE to a square-grid in metric CRS ---
    # Limit resample domain to polygon bounds (+pad) for efficiency
    minx, miny, maxx, maxy = poly_metric.bounds
    pad_x = (maxx - minx) * 0.05 if np.isfinite(maxx - minx) else 0.0
    pad_y = (maxy - miny) * 0.05 if np.isfinite(maxy - miny) else 0.0
    bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

    Xg, Yg, Zg = resample_to_square_grid(X, Y, vel_plot, cell_size_m=args.square_cell_m, bounds=bounds, method="nearest")

    # ---------- MASK BACKGROUND OUTSIDE THE POLYGON (to avoid colored background) ----------
    # Make a boolean mask of grid nodes inside the polygon (boundary excluded)
    try:
        # Fast path if shapely.vectorized is available
        from shapely import vectorized
        inside_poly = vectorized.contains(poly_for_test, Xg, Yg)
    except Exception:
        # Fallback via GeoPandas point-in-polygon
        grid_pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(Xg.ravel(), Yg.ravel()),
            crs=poly_for_test.geom_type  # placeholder
        )
        grid_pts.set_crs(poly_metric.crs, inplace=True, allow_override=True)
        inside_poly = grid_pts.within(poly_for_test).values.reshape(Xg.shape)

    # Apply mask: outside polygon becomes NaN (transparent in pcolormesh)
    Zg = np.where(inside_poly, Zg, np.nan)
    Zg = np.ma.masked_invalid(Zg)

    # Plot
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    pm = ax.pcolormesh(
        Xg, Yg, Zg,
        shading="nearest", cmap=args.cmap,
        vmin=args.vmin, vmax=args.vmax,
        rasterized=True, alpha=args.alpha,
    )

    # Overlay polygon outline in metric CRS
    gpd.GeoSeries(poly_metric).boundary.plot(ax=ax, color=args.poly_edgecolor, linewidth=args.poly_linewidth)

    # -------------------- ADD SECTION NAME LABELS (slightly above each polygon) --------------------
    try:
        # Read original features (not dissolved) to get per-section names
        gdf_lbl = gpd.read_file(args.polygons)
        # Normalize CRS similarly to loader
        if gdf_lbl.crs is None:
            gdf_lbl = gdf_lbl.set_crs(args.polygons_crs if hasattr(args, "polygons_crs") else "EPSG:4326")
        else:
            if "CRS84" in str(gdf_lbl.crs).upper():
                gdf_lbl = gdf_lbl.set_crs("EPSG:4326")
        gdf_lbl = gdf_lbl.to_crs(args.metric_crs)

        # Choose a name field (fallback to index)
        name_fields = ["Name", "name", "section", "Section", "id", "ID"]
        def pick_name(props):
            for nf in name_fields:
                if nf in props and props[nf] not in [None, ""]:
                    return str(props[nf])
            return str(props.get("index", ""))

        # Vertical offset ~ 2% of plot height
        y_off = 0.04 * (bounds[3] - bounds[1])

        for _, row in gdf_lbl.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # representative_point() is safer than centroid for concave polygons
            rp = geom.representative_point()
            label = pick_name(row)
            ax.text(rp.x, rp.y + y_off, label, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="black",
                    path_effects=None)
    except Exception as _e:
        # Silently skip labeling if anything goes wrong to avoid breaking the plot
        pass
    # ------------------------------------------------------------------------------------------------

    # Axes extent and aspect (equal meters)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal', adjustable='box')

    # -------------- Axis labels & ticks shown in lon/lat degrees --------------
    # Keep metric coords for plotting, but format tick labels in lon/lat
    to_lonlat = inverse_to_lonlat(args.metric_crs)

    def fmt_x(x, pos):
        lon, _ = to_lonlat(x, (bounds[1]+bounds[3])/2.0)
        return f"{lon:.5f}°"
    def fmt_y(y, pos):
        _, lat = to_lonlat((bounds[0]+bounds[2])/2.0, y)
        return f"{lat:.5f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # --------------------------------------------------------------------------

    cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("velocity [mm/year]")

    # Title
    if args.title:
        ax.set_title(args.title)
    else:
        proj = atr.get("PROJECT_NAME", "") or atr.get("PROJECT", "") or ""
        sat = atr.get("PLATFORM", "") or atr.get("SENSOR", "") or ""
        tit = f"Velocity inside sections"
        extra = " • ".join([s for s in (sat, proj) if s])
        if extra:
            tit += f" ({extra})"
        ax.set_title(tit)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to: {args.out}")
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
