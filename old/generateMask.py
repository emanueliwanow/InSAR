#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a MintPy-style mask from a lon/lat polygon, optionally combining
with an existing MintPy mask (e.g., output of generate_mask.py).

Outputs an HDF5 with dataset 'mask' (uint8, 1=valid/inside, 0=masked out).

Example:
    python generate_mask_with_polygon.py \
        --geom geometryGeo.h5 \
        --polygon "9.92540667693568 44.17437471448552,9.927486204677724 44.17417644184879,9.927505198552097 44.17432955344993,9.925425961586596 44.17451301708271,9.923174837691484 44.17465053547857,9.923160920209696 44.17453930372317,9.92540667693568 44.17437471448552" \
        --in-mask mask.h5 \
        -o mask_polygon.h5

Notes:
- Requires: numpy, h5py
- Uses a pure-numpy point-in-polygon (no shapely/matplotlib needed).
- The polygon must be in lon lat order. WKT POLYGON((...)) strings are also accepted.
"""


import argparse, sys, re, h5py, numpy as np
from typing import Tuple, List, Dict

# ---------------- existing helpers (unchanged) ----------------
# parse_polygon_string(...), points_in_polygon(...), read_lat_lon(...), read_existing_mask(...)

def parse_polygon_string(poly_str: str):
    s = poly_str.strip()
    wkt = re.match(r"^\s*POLYGON\s*\(\((.*)\)\)\s*$", s, flags=re.IGNORECASE)
    if wkt:
        s = wkt.group(1)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 3:
        raise ValueError("Polygon must have at least 3 vertices.")
    lons, lats = [], []
    for p in parts:
        toks = p.split()
        if len(toks) != 2:
            raise ValueError(f"Cannot parse vertex '{p}'. Expected 'lon lat'.")
        lon, lat = float(toks[0]), float(toks[1])
        lons.append(lon); lats.append(lat)
    poly_lon = np.asarray(lons, np.float64)
    poly_lat = np.asarray(lats, np.float64)
    if poly_lon[0] != poly_lon[-1] or poly_lat[0] != poly_lat[-1]:
        poly_lon = np.r_[poly_lon, poly_lon[0]]
        poly_lat = np.r_[poly_lat, poly_lat[0]]
    if poly_lon.size < 4:
        raise ValueError("Polygon must have at least 3 distinct vertices.")
    return poly_lon, poly_lat

def points_in_polygon(lon_grid, lat_grid, poly_lon, poly_lat):
    if lon_grid.shape != lat_grid.shape:
        raise ValueError("lon_grid and lat_grid must have the same shape.")
    x, y = lon_grid, lat_grid
    xv, yv = poly_lon, poly_lat
    n = xv.size
    inside = np.zeros(x.shape, dtype=bool)
    eps = np.finfo(np.float64).eps
    j = n - 1
    for i in range(n):
        xi, yi = xv[i], yv[i]
        xj, yj = xv[j], yv[j]
        m = ((yi > y) != (yj > y))
        x_cross = (xj - xi) * (y - yi) / ((yj - yi) + eps) + xi
        inside ^= m & (x < x_cross)
        j = i
    return inside

def read_lat_lon(geom_file: str):
    with h5py.File(geom_file, "r") as f:
        if "latitude" in f and "longitude" in f:
            lat = f["latitude"][()]
            lon = f["longitude"][()]
        elif "lat" in f and "lon" in f:
            lat = f["lat"][()]
            lon = f["lon"][()]
        else:
            lat = lon = None
            for k in f.keys():
                g = f[k]
                if isinstance(g, h5py.Dataset):
                    continue
                if "latitude" in g and "longitude" in g:
                    lat = g["latitude"][()]; lon = g["longitude"][()]; break
                if "lat" in g and "lon" in g:
                    lat = g["lat"][()]; lon = g["lon"][()]; break
            if lat is None or lon is None:
                raise KeyError("Could not find 'latitude'/'longitude' (or 'lat'/'lon') in the geometry file.")
    lat = np.array(lat, dtype=np.float64)
    lon = np.array(lon, dtype=np.float64)
    if lat.shape != lon.shape:
        raise ValueError("latitude and longitude arrays have different shapes.")
    return lon, lat

def read_existing_mask(mask_file: str, dset: str = "mask"):
    with h5py.File(mask_file, "r") as f:
        if dset in f:
            arr = f[dset][()]
        else:
            found = None
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    continue
                if dset in f[k]:
                    found = f[k][dset][()]
                    break
            if found is None:
                raise KeyError(f"Dataset '{dset}' not found in '{mask_file}'.")
            arr = found
    arr = np.array(arr)
    return (arr != 0) if arr.dtype != bool else arr

# ---------------- new: metadata helpers + fixed writer ----------------

META_KEYS_TO_COPY = [
    "WIDTH", "LENGTH",
    "X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP",
    "X_UNIT", "Y_UNIT", "EPSG",
    "REF_X", "REF_Y", "REF_LAT", "REF_LON",
    "REF_DATE", "DATE",
    "PROJECTION", "DATUM", "azimuthPixelSize", "rangePixelSize", "rangeResolution", "azimuthResolution"
]

def collect_root_attrs(h5_path: str) -> Dict[str, object]:
    """Collects common MintPy root attributes if present."""
    out = {}
    try:
        with h5py.File(h5_path, "r") as f:
            for k in META_KEYS_TO_COPY:
                if k in f.attrs:
                    out[k] = f.attrs[k]
    except Exception:
        pass
    return out

def write_mask_h5(out_file: str, mask: np.ndarray, dset: str = "mask",
                  template_attrs: Dict[str, object] = None) -> None:
    """Write mask and ensure MintPy-friendly root attrs exist."""
    length, width = map(int, mask.shape)
    with h5py.File(out_file, "w") as f:
        d = f.create_dataset(dset, data=mask.astype(np.uint8), compression="gzip", shuffle=True, fletcher32=True)
        # dataset attrs
        d.attrs["DESCRIPTION"] = np.string_("Binary mask (1=valid, 0=masked)")
        d.attrs["UNIT"] = np.string_("1 or 0")
        d.attrs["DATA_TYPE"] = np.string_("byte")

        # root attrs (copy, then enforce required)
        if template_attrs:
            for k, v in template_attrs.items():
                try:
                    f.attrs[k] = v
                except Exception:
                    # ignore attrs that can't be written cleanly
                    pass

        # these must match the actual dataset
        f.attrs["WIDTH"]  = width
        f.attrs["LENGTH"] = length

        # helpful identifiers
        f.attrs["FILE_TYPE"] = np.string_("mask")
        f.attrs["SOURCE"]    = np.string_("generate_mask_with_polygon.py")

# ------------------------------ main (updated) ------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate a MintPy-style mask from a lon/lat polygon, optionally combining with an existing mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--geom", required=True, help="Geocoded MintPy geometry file (HDF5) containing 'latitude' and 'longitude'.")
    ap.add_argument("--polygon", required=True,
                    help="Polygon as 'lon lat,lon lat,...' or WKT 'POLYGON((lon lat, ...))'.")
    ap.add_argument("--in-mask", default=None,
                    help="Optional existing MintPy mask HDF5 to AND with the polygon mask (also used as metadata template if given).")
    ap.add_argument("--invert", action="store_true",
                    help="If set, keep OUTSIDE the polygon instead of inside.")
    ap.add_argument("-o", "--out", default="mask_polygon.h5", help="Output HDF5 mask file.")
    ap.add_argument("--dset", default="mask", help="Dataset name for output mask.")
    args = ap.parse_args()

    # read geometry + metadata template
    lon, lat = read_lat_lon(args.geom)
    template_attrs = collect_root_attrs(args.in_mask) if args.in_mask else collect_root_attrs(args.geom)

    # polygon -> inside mask
    poly_lon, poly_lat = parse_polygon_string(args.polygon)
    inside = points_in_polygon(lon, lat, poly_lon, poly_lat)
    mask_poly = inside if not args.invert else ~inside

    # combine with existing mask if provided
    if args.in_mask:
        base_mask = read_existing_mask(args.in_mask, dset=args.dset)
        if base_mask.shape != mask_poly.shape:
            raise ValueError(f"Shape mismatch: base mask {base_mask.shape} vs geometry {mask_poly.shape}")
        final_mask = base_mask & mask_poly
    else:
        final_mask = mask_poly

    # write with proper metadata
    write_mask_h5(args.out, final_mask, dset=args.dset, template_attrs=template_attrs)

    total = final_mask.size
    kept = int(final_mask.sum())
    print(f"Wrote {args.out} with dataset '{args.dset}'.")
    print(f"Pixels kept: {kept}/{total} ({100.0*kept/float(total):.2f}%).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr); sys.exit(1)
