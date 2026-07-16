"""Reconstruct the REAL fire's hourly progression by combining the two ground truths.

The burned-area polygon says WHERE it burned (92%-accuracy classification) but not
when; VIIRS says WHEN (per-detection acquisition time) but is sparse (375 m pixels,
a few overpasses per day). Combining them: every burned cell gets the acquisition
time of its NEAREST VIIRS detection (nearest-neighbour time interpolation on a
metric grid, clipped to the scar) -> an "arrival time" surface, from which the real
fire's extent at ANY hour can be cut (`arrival <= t`).

Caveat: between satellite overpasses there is no information, so the reconstruction
advances in STEPS at each overpass (e.g. the ~9 h overnight gap 01:02 -> 10:43 on
day 1). Treat hourly extents as overpass-resolution stairs, not smooth truth.
See [[Validation]].

Run standalone for a quick self-check:
    python scripts/cell2fire/validation/real_progression.py
"""

import sys
from pathlib import Path

# Lives in the validation/ subfolder; `_paths` is one directory UP, in
# scripts/cell2fire/ proper (same bootstrap as validate_overlay.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio import transform as rtransform
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import unary_union

from _paths import DATA_DIR

RFD = DATA_DIR / "Real_Fire_Data"
GRID_M = 200.0        # reconstruction cell (m); ~half a VIIRS pixel
# The DATA seam to the live pipeline: the dashboard reads THIS FILE, never this
# module (validation code stays fully decoupled from the live run_scenario loop).
HOURLY_GEOJSON = RFD / "real_progression_hourly.geojson"


def reconstruct(grid_m=GRID_M):
    """Arrival-time surface of the real fire.

    Returns (arrival, transform, t0): `arrival` = 2-D float array of hours since
    `t0` (NaN outside the burned scar) on a `grid_m` EPSG:2100 grid; `t0` = the
    first VIIRS detection, floored to the hour (2021-08-05 23:00 UTC)."""
    b = gpd.read_file(RFD / "Burned_Area" / "Burned_area_20210829.shp").to_crs(2100)
    scar = unary_union(b.geometry)
    v = gpd.read_file(RFD / "VIIRS" / "VIRS_dataset_egsa_dhmos.shp").to_crs(2100)
    v["t"] = pd.to_datetime(v["acq_at_s"])
    t0 = v["t"].min().floor("h")

    minx, miny, maxx, maxy = scar.bounds
    ncols = int(np.ceil((maxx - minx) / grid_m))
    nrows = int(np.ceil((maxy - miny) / grid_m))
    tr = rtransform.from_origin(minx, maxy, grid_m, grid_m)
    mask = features.rasterize([(scar, 1)], out_shape=(nrows, ncols), transform=tr,
                              fill=0, dtype="uint8").astype(bool)
    rr, cc = np.where(mask)
    xs, ys = rtransform.xy(tr, rr, cc)          # cell centres
    tree = cKDTree(np.c_[v.geometry.x.values, v.geometry.y.values])
    _, idx = tree.query(np.c_[xs, ys])
    hours = ((v["t"].iloc[idx] - t0).dt.total_seconds() / 3600.0).to_numpy()
    arrival = np.full((nrows, ncols), np.nan)
    arrival[rr, cc] = hours
    return arrival, tr, t0


def extent_at(arrival, tr, t0, when):
    """Real fire extent (MultiPolygon or None) at timestamp `when` (EPSG:2100)."""
    h = (pd.to_datetime(when) - t0).total_seconds() / 3600.0
    m = (~np.isnan(arrival)) & (arrival <= h)
    if not m.any():
        return None
    polys = [shape(g) for g, val in
             features.shapes(m.astype("uint8"), mask=m, transform=tr) if val == 1]
    return unary_union(polys)


def export_hourly(out_path=HOURLY_GEOJSON):
    """Precompute the real fire's extent at EVERY hour of the event into a static
    GeoJSON - the file the live dashboard reads instead of importing this module.

    One feature per hour: `time_utc` (the lookup key), `km2` (area BEFORE the
    display simplification, so the chart curve is exact), geometry = the extent
    simplified at ~20 m (display-grade) and stored in EPSG:4326 (RFC 7946).
    The 2021 event is static ground truth, so this runs ONCE, manually."""
    arrival, tr, t0 = reconstruct()
    hmax = int(np.ceil(np.nanmax(arrival)))       # last hour anything new burned
    rows = []
    for h in range(hmax + 1):
        e = extent_at(arrival, tr, t0, t0 + pd.Timedelta(hours=h))
        if e is None:                             # before the first detection
            continue
        rows.append({"hour": h,
                     "time_utc": (t0 + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M"),
                     "km2": round(e.area / 1e6, 2),   # exact area, pre-simplify
                     "geometry": e.simplify(20)})     # display-grade geometry
    gdf = gpd.GeoDataFrame(rows, crs=2100).to_crs(4326)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path, len(rows)


if __name__ == "__main__":
    arrival, tr, t0 = reconstruct()
    burned = ~np.isnan(arrival)
    print(f"t0 = {t0} UTC | scar cells {int(burned.sum())} "
          f"({burned.sum() * GRID_M**2 / 1e6:.1f} km^2 at {GRID_M:.0f} m)")
    for h in (3, 12, 16, 24, 48, 144):
        e = extent_at(arrival, tr, t0, t0 + pd.Timedelta(hours=h))
        print(f"  +{h:3d} h -> {0 if e is None else e.area / 1e6:6.1f} km^2")
    path, n = export_hourly()                     # the dashboard's data seam
    print(f"Exported {n} hourly extents -> {path}")
