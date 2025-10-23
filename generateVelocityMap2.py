#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generateVelocityMap_pro.py
--------------------------
"Best practice" velocity map generator with robust pixel filtering and clean plotting.
(See header in previous attempt for full description.)
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely import make_valid
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from mintpy.utils import readfile
from pyproj import Transformer
from scipy.interpolate import griddata


# ============================== Utilities ==============================

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    if isinstance(x, (dict, list, tuple, set)):
        return ""
    return str(x)


def read_2d(path: Path, candidates: Iterable[str]):
    try:
        arr, atr = readfile.read(str(path))
        if np.asarray(arr).ndim == 2:
            return np.array(arr), dict(atr)
    except Exception:
        pass
    for name in candidates:
        try:
            arr, atr = readfile.read(str(path), datasetName=name)
            if np.asarray(arr).ndim == 2:
                return np.array(arr), dict(atr)
        except Exception:
            continue
    raise RuntimeError(f"Could not read a 2D dataset from {path} using candidates {list(candidates)}")


def read_lonlat_from_geometry(geom: Path):
    if geom is None or not Path(geom).exists():
        return None, None
    lon_cands = ("longitude", "/longitude", "lon", "/lon")
    lat_cands = ("latitude", "/latitude", "lat", "/lat")
    lon = lat = None
    for nm in lon_cands:
        try:
            lon, _ = readfile.read(str(geom), datasetName=nm)
            break
        except Exception:
            continue
    for nm in lat_cands:
        try:
            lat, _ = readfile.read(str(geom), datasetName=nm)
            break
        except Exception:
            continue
    if lon is not None and lat is not None and np.ndim(lon) == 2 and np.ndim(lat) == 2:
        return np.array(lon, float), np.array(lat, float)
    return None, None


def lonlat_from_corners(h5_path: Path, shape):
    R, C = shape
    _, atr = readfile.read(str(h5_path))
    at = {k.lower(): v for k, v in dict(atr).items()}
    try:
        lats = [float(at[f"lat_ref{i}"]) for i in range(1, 5)]
        lons = [float(at[f"lon_ref{i}"]) for i in range(1, 5)]
    except Exception as e:
        raise RuntimeError("lat_ref*/lon_ref* not found; provide geometry*.h5 for lon/lat.") from e

    v = np.linspace(0.0, 1.0, R)[:, None]
    u = np.linspace(0.0, 1.0, C)[None, :]

    def bilinear(c00, c10, c01, c11):
        return ((1 - u) * (1 - v) * c00 +
                u * (1 - v) * c10 +
                (1 - u) * v * c01 +
                u * v * c11)

    best = None
    corners = list(zip(lons, lats))
    for perm in itertools.permutations(corners, 4):
        (lon00, lat00), (lon10, lat10), (lon01, lat01), (lon11, lat11) = perm
        LON = bilinear(lon00, lon10, lon01, lon11)
        LAT = bilinear(lat00, lat10, lat01, lat11)
        dlon = np.nanmedian(np.diff(LON, axis=1))
        dlat = np.nanmedian(np.diff(LAT, axis=0))
        flips = (np.sum(np.sign(np.diff(LON, axis=1)) != np.sign(dlon)) +
                 np.sum(np.sign(np.diff(LAT, axis=0)) != np.sign(dlat)))
        score = (flips, -(abs(dlon) + abs(dlat)))
        if best is None or score < best[0]:
            best = (score, (LON, LAT))
    if best is None:
        raise RuntimeError("Failed to infer lon/lat from corners.")
    return best[1]


def load_sections_metric(path: Path, declared_crs: str, metric_crs: str):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(declared_crs)
    else:
        crs_str = str(gdf.crs).upper()
        if "CRS84" in crs_str:
            gdf = gdf.set_crs("EPSG:4326")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None else None)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise RuntimeError("No polygon geometry found in sections file.")
    return gdf.dissolve().to_crs(metric_crs).geometry.iloc[0]


def to_metric(lon2d, lat2d, metric_crs):
    tr = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    X, Y = tr.transform(lon2d, lat2d)
    return np.asarray(X), np.asarray(Y)


def inverse_to_lonlat(metric_crs):
    inv = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    def f(x, y):
        return inv.transform(x, y)
    return f


def estimate_spacing(X, Y):
    dx = np.nanmedian(np.diff(X, axis=1))
    dy = np.nanmedian(np.diff(Y, axis=0))
    return abs(dx), abs(dy)


def resample_square(X, Y, Z, cell_size_m=None, bounds=None, method="nearest"):
    dx, dy = estimate_spacing(X, Y)
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
    pts = np.column_stack([X.ravel(), Y.ravel()])
    vals = Z.ravel()
    good = np.isfinite(vals)
    if not np.any(good):
        return Xg, Yg, np.full_like(Xg, np.nan, dtype=float)
    Zg = griddata(pts[good], vals[good], (Xg, Yg), method=method)
    return Xg, Yg, Zg


@dataclass
class PlotOptions:
    cmap: str
    vmin: float | None
    vmax: float | None
    vcenter: float | None
    alpha: float
    figsize: Tuple[float, float]
    dpi: int
    poly_edge: str
    poly_lw: float


def build_keep_mask(inside, vel, tc, ps_mask, coh_thr, exclude_tc_eq1):
    keep = np.ones_like(vel, dtype=bool)
    keep &= inside
    keep &= np.isfinite(vel)
    if ps_mask is not None:
        keep &= ps_mask.astype(bool)
    if tc is not None:
        if coh_thr is not None:
            keep &= (tc >= float(coh_thr))
        if exclude_tc_eq1:
            keep &= (tc != 1.0)
    return keep


def make_title(atr, explicit, show_filters):
    base = explicit or "Velocity inside sections"
    proj = safe_str(atr.get("PROJECT_NAME", atr.get("PROJECT", "")))
    sat  = safe_str(atr.get("PLATFORM", atr.get("SENSOR", "")))
    extra = " • ".join([s for s in (sat, proj) if s])
    if extra:
        base += f" ({extra})"
    if show_filters:
        base += f" — filters: {show_filters}"
    return base


def build_filter_string(has_ps, tc, coh_thr, excl_tc1):
    bits = []
    if has_ps:
        bits.append("PS")
    if tc is not None and coh_thr is not None:
        bits.append(f"TC≥{coh_thr:g}")
    if tc is not None and excl_tc1:
        bits.append("TC≠1.0")
    return ", ".join(bits)


def main():
    p = argparse.ArgumentParser(description="Best-practice velocity map with PS/TC filtering and masked plotting.")
    p.add_argument("--velocity", required=True, type=Path)
    p.add_argument("--polygons", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--geometry", type=Path, default=None)
    p.add_argument("--dataset-name", type=str, default="velocity")
    p.add_argument("--tc", type=Path, default=None)
    p.add_argument("--mask-ps", type=Path, default=None)
    p.add_argument("--coh-thr", type=float, default=None)
    p.add_argument("--exclude-tc-eq1", action="store_true")
    p.add_argument("--polygons-crs", type=str, default="EPSG:4326")
    p.add_argument("--metric-crs", type=str, default="EPSG:3857")
    p.add_argument("--shrink-eps-m", type=float, default=0.2)
    p.add_argument("--units", choices=["m", "mm"], default="mm")
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--resample", action="store_true")
    p.add_argument("--square-cell-m", type=float, default=None)
    p.add_argument("--points", action="store_true")
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--vcenter", type=float, default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--figsize", type=float, nargs=2, default=(9, 7))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--poly-edgecolor", type=str, default="k")
    p.add_argument("--poly-linewidth", type=float, default=1.2)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--abs-velocity", action="store_true")
    p.add_argument("--out-abs", type=Path, default=None)
    p.add_argument("--green-red-cmap", action="store_true")
    args = p.parse_args()

    vel, atr = read_2d(args.velocity, candidates=(args.dataset_name, f"/{args.dataset_name}", "velocity", "/velocity"))
    R, C = vel.shape
    N = R * C

    tc = None
    ps = None
    if args.tc and Path(args.tc).exists():
        tc, _ = read_2d(args.tc, candidates=("/temporalCoherence", "temporalCoherence"))
        if tc.shape != (R, C):
            raise ValueError(f"TC shape {tc.shape} != velocity shape {(R,C)}")
    if args.mask_ps and Path(args.mask_ps).exists():
        ps, _ = read_2d(args.mask_ps, candidates=("/mask", "mask"))
        ps = ps.astype(bool)
        if ps.shape != (R, C):
            raise ValueError(f"maskPS shape {ps.shape} != velocity shape {(R,C)}")

    lon2d = lat2d = None
    if args.geometry and Path(args.geometry).exists():
        lon2d, lat2d = read_lonlat_from_geometry(args.geometry)
        if lon2d is not None:
            print("Using lon/lat from geometry file.")
    if lon2d is None or lat2d is None:
        print("Reconstructing lon/lat from corner references...")
        lon2d, lat2d = lonlat_from_corners(args.velocity, (R, C))

    poly_metric = load_sections_metric(args.polygons, args.polygons_crs, args.metric_crs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poly_for_test = poly_metric.buffer(-float(args.shrink_eps_m)) if args.shrink_eps_m > 0 else poly_metric

    X, Y = to_metric(lon2d, lat2d, args.metric_crs)

    idx = np.arange(N, dtype=np.int64).reshape(R, C)[::args.step, ::args.step].ravel()
    pts = gpd.GeoDataFrame(
        {"i": idx},
        geometry=gpd.points_from_xy(X.ravel()[idx], Y.ravel()[idx]),
        crs=args.metric_crs,
    )
    inside_small = pts.within(poly_for_test).values
    inside = np.zeros(R * C, dtype=bool)
    inside[idx] = inside_small
    inside = inside.reshape(R, C)

    keep = build_keep_mask(inside, vel, tc, ps, args.coh_thr, args.exclude_tc_eq1)

    if args.scale is not None:
        scale = float(args.scale)
    else:
        scale = 1000.0 if args.units == "mm" else 1.0
    field = np.full_like(vel, np.nan, dtype=float)
    field[keep] = vel[keep] * scale

    use_abs_in_main = args.abs_velocity and (args.out_abs is None)
    if use_abs_in_main:
        field = np.abs(field)

    minx, miny, maxx, maxy = poly_metric.bounds
    pad_x = (maxx - minx) * 0.05 if np.isfinite(maxx - minx) else 0.0
    pad_y = (maxy - miny) * 0.05 if np.isfinite(maxy - miny) else 0.0
    bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

    if args.vcenter is not None and args.vmin is not None and args.vmax is not None:
        norm = TwoSlopeNorm(vcenter=args.vcenter, vmin=args.vmin, vmax=args.vmax)
    else:
        norm = Normalize(vmin=args.vmin, vmax=args.vmax)

    filter_str = build_filter_string(ps is not None, tc, args.coh_thr, args.exclude_tc_eq1)
    title_main = make_title(atr, args.title, filter_str)

    def add_common_axes_bits(ax):
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        to_ll = inverse_to_lonlat(args.metric_crs)
        def fmt_x(x, pos):
            lon, _ = to_ll(x, (bounds[1] + bounds[3]) / 2.0)
            return f"{lon:.5f}°"
        def fmt_y(y, pos):
            _, lat = to_ll((bounds[0] + bounds[2]) / 2.0, y)
            return f"{lat:.5f}°"
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        gpd.GeoSeries(poly_metric).boundary.plot(ax=ax,
                                                 color=args.poly_edgecolor,
                                                 linewidth=args.poly_linewidth)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        
        # -------------------- ADD SECTION NAME LABELS (slightly above each polygon) --------------------
        try:
            # Read original features (not dissolved) to get per-section names
            gdf_lbl = gpd.read_file(args.polygons)
            # Normalize CRS similarly to loader
            if gdf_lbl.crs is None:
                gdf_lbl = gdf_lbl.set_crs(args.polygons_crs)
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
            y_off = 0.06 * (bounds[3] - bounds[1])

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

    def plot_curvilinear(Z, outpath, title, label_abs):
        Zmask = np.ma.array(Z, mask=~np.isfinite(Z))
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        # Set vmin to 0 and vmax to norm of negative values if label_abs is true
        plot_norm = norm
        if label_abs:
            # For absolute velocity, vmin should be 0 and vmax should be the norm of the most negative value
            if hasattr(norm, 'vmin') and norm.vmin is not None and hasattr(norm, 'vmax') and norm.vmax is not None:
                # Use the larger of the absolute values of vmin and vmax
                abs_vmin = abs(norm.vmin)
                abs_vmax = abs(norm.vmax)
                plot_norm = Normalize(vmin=0, vmax=max(abs_vmin, abs_vmax))
            else:
                # If no explicit bounds, use the data range
                valid_data = Z[np.isfinite(Z)]
                if len(valid_data) > 0:
                    max_abs = np.max(np.abs(valid_data))
                    plot_norm = Normalize(vmin=0, vmax=max_abs)
        pm = ax.pcolormesh(X, Y, Zmask, shading="nearest", cmap=args.cmap, norm=plot_norm, alpha=args.alpha)
        add_common_axes_bits(ax)
        cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
        unit_lbl = f"[{args.units}/year]" if args.units in ("m", "mm") else "[units]"
        cb.set_label("|velocity| " + unit_lbl if label_abs else "velocity " + unit_lbl)
        ax.set_title(title)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving figure to: {outpath}")
        fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_points(Z, outpath, title, label_abs):
        Zflat = Z.ravel()
        good = np.isfinite(Zflat)
        xi = X.ravel()[good][::args.step]
        yi = Y.ravel()[good][::args.step]
        zi = Zflat[good][::args.step]
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        # Set vmin to 0 and vmax to norm of negative values if label_abs is true
        plot_norm = norm
        if label_abs:
            # For absolute velocity, vmin should be 0 and vmax should be the norm of the most negative value
            if hasattr(norm, 'vmin') and norm.vmin is not None and hasattr(norm, 'vmax') and norm.vmax is not None:
                # Use the larger of the absolute values of vmin and vmax
                abs_vmin = abs(norm.vmin)
                abs_vmax = abs(norm.vmax)
                plot_norm = Normalize(vmin=0, vmax=max(abs_vmin, abs_vmax))
            else:
                # If no explicit bounds, use the data range
                valid_data = Z[np.isfinite(Z)]
                if len(valid_data) > 0:
                    max_abs = np.max(np.abs(valid_data))
                    plot_norm = Normalize(vmin=0, vmax=max_abs)
        sc = ax.scatter(xi, yi, c=zi, s=args.point_size, cmap=args.cmap, norm=plot_norm, alpha=args.alpha, linewidths=0)
        add_common_axes_bits(ax)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        unit_lbl = f"[{args.units}/year]" if args.units in ("m", "mm") else "[units]"
        cb.set_label("|velocity| " + unit_lbl if label_abs else "velocity " + unit_lbl)
        ax.set_title(title)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving figure to: {outpath}")
        fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_resampled(Z, outpath, title, label_abs):
        Xg, Yg, Zg = resample_square(X, Y, Z, cell_size_m=args.square_cell_m, bounds=bounds, method="nearest")
        try:
            from shapely import vectorized
            inside_poly = vectorized.contains(poly_for_test, Xg, Yg)
        except Exception:
            grid_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(Xg.ravel(), Yg.ravel()), crs=args.metric_crs)
            inside_poly = grid_pts.within(poly_for_test).values.reshape(Xg.shape)
        Zg = np.where(inside_poly, Zg, np.nan)
        Zmask = np.ma.array(Zg, mask=~np.isfinite(Zg))
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        # Set vmin to 0 and vmax to norm of negative values if label_abs is true
        plot_norm = norm
        if label_abs:
            # For absolute velocity, vmin should be 0 and vmax should be the norm of the most negative value
            if hasattr(norm, 'vmin') and norm.vmin is not None and hasattr(norm, 'vmax') and norm.vmax is not None:
                # Use the larger of the absolute values of vmin and vmax
                abs_vmin = abs(norm.vmin)
                abs_vmax = abs(norm.vmax)
                plot_norm = Normalize(vmin=0, vmax=max(abs_vmin, abs_vmax))
            else:
                # If no explicit bounds, use the data range
                valid_data = Z[np.isfinite(Z)]
                if len(valid_data) > 0:
                    max_abs = np.max(np.abs(valid_data))
                    plot_norm = Normalize(vmin=0, vmax=max_abs)
        pm = ax.pcolormesh(Xg, Yg, Zmask, shading="nearest", cmap=args.cmap, norm=plot_norm, alpha=args.alpha)
        add_common_axes_bits(ax)
        cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
        unit_lbl = f"[{args.units}/year]" if args.units in ("m", "mm") else "[units]"
        cb.set_label("|velocity| " + unit_lbl if label_abs else "velocity " + unit_lbl)
        ax.set_title(title)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving figure to: {outpath}")
        fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    label_abs = use_abs_in_main
    if args.points:
        plot_points(field, args.out, title_main, label_abs)
    elif args.resample:
        plot_resampled(field, args.out, title_main, label_abs)
    else:
        plot_curvilinear(field, args.out, title_main, label_abs)

    if args.out_abs is not None:
        abs_field = np.abs(field)
        cmap_abs = LinearSegmentedColormap.from_list("green_red", ["#00FF00", "#FFFF00", "#FF0000"], N=256) \
                   if args.green_red_cmap else args.cmap
        title_abs = title_main.replace("Velocity", "Absolute Velocity")

        # Temporarily override cmap/norm for abs plot
        old_cmap = args.cmap
        args.cmap = cmap_abs
        # If using green-red and mostly positive values, a simple Normalize may be better.
        abs_norm = Normalize(vmin=args.vmin, vmax=args.vmax) if args.green_red_cmap else norm
        old_norm = globals().get("norm", None)
        globals()["norm"] = abs_norm
        try:
            if args.points:
                plot_points(abs_field, args.out_abs, title_abs, True)
            elif args.resample:
                plot_resampled(abs_field, args.out_abs, title_abs, True)
            else:
                plot_curvilinear(abs_field, args.out_abs, title_abs, True)
        finally:
            args.cmap = old_cmap
            if old_norm is not None:
                globals()["norm"] = old_norm

if __name__ == "__main__":
    main()
