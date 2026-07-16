"""Fire -> road-blocking helpers (shared library, imported by the evacuation pipeline).

The engine-agnostic "network/exposure" building blocks, shared (DRY) by
`evacuation.py`, `evacuate_routes.py` and `fire_timesteps.py`:

  * `buffer_width_grid()`     per-cell VARIABLE buffer width (m) from an intensity
        proxy = fuel-class x slope (PLACEHOLDER weights, uncalibrated);
  * `blocked_zone(fire, ...)` the polygon of cells within their local buffer width
        of a fire perimeter -> the road edges to remove;
  * `fuel_hazard_overlay()`   a coloured fuel-danger image for the map.

All analysis is in EPSG:2100 (Greek Grid, m). The intensity proxy is a PLACEHOLDER
(fuel x slope); swap `buffer_width_grid` for a real FLI raster and everything
downstream (block + reroute) is unchanged.

NOTE: input rasters are tagged as a generic LOCAL_CS (Greek Grid without an EPSG
code); we re-label them EPSG:2100 (coordinates already correct, no reprojection).

Library only - no entry point. The fire perimeter now comes from the real Cell2Fire
run (`cell2fire_adapter.py`), never a synthetic circle.
"""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from _paths import FUEL, SLOPE
from route_shortest_path import POINT_CRS

# --- Configuration -----------------------------------------------------------
ANALYSIS_CRS = "EPSG:2100"   # rasters/fire analysed and saved in Greek Grid (m)

# --- PLACEHOLDER intensity -> buffer model (UNCALIBRATED) --------------------
# SVM land-cover/fuel classes (Forest_Fuel/SVM_HUA.tif), weighted by flammability.
# Provisional weights -- calibrate with the domain expert.
FUEL_WEIGHT = {
    0: 1.00,   # Κωνοφόρα (conifer) - resin, crown fire -> most dangerous
    1: 0.85,   # Μεικτό δάσος (mixed forest)
    2: 0.55,   # Πλατύφυλλα (broadleaf) - higher moisture -> less flammable
    3: 0.70,   # Λοιπές μορφές δάσους (other forest forms)
    4: 0.30,   # Μη δάσος (non-forest)
}
FUEL_WEIGHT_DEFAULT = 0.30   # nodata (7) / outside the fuel raster's extent -> low
SLOPE_CAP_DEG = 45.0           # slope factor saturates here
BUFFER_FLOOR_M = 100.0         # per-cell minimum buffer
BUFFER_MAX_M = 600.0           # per-cell cap
WIDTH_SCALE = 300.0            # metres of extra buffer per unit of intensity proxy
INTENSITY_REF = 0.3            # intensity at which width = floor


def _slope_factor(slope_deg):
    """PLACEHOLDER monotone slope factor in [1, 2] (not Rothermel-calibrated)."""
    return 1.0 + np.clip(slope_deg, 0.0, SLOPE_CAP_DEG) / SLOPE_CAP_DEG   # steeper slope -> fire spreads faster uphill


# --- Raster helpers ----------------------------------------------------------
def _read_reference():
    """Read the slope raster as the reference grid (treated as EPSG:2100)."""
    with rasterio.open(SLOPE) as ds:
        slope = ds.read(1).astype("float64")
        nodata = ds.nodata
        grid = {"transform": ds.transform, "shape": (ds.height, ds.width),   # the MASTER grid everything else resamples onto
                "res": float(ds.res[0])}
    if nodata is not None:
        slope = np.where(slope == nodata, 0.0, slope)   # nodata slope -> flat
    return slope, grid


def _read_fuel_on(grid):
    """Nearest-resample the fuel raster onto the reference (slope) grid via affine
    math -- a pure regrid (same Greek Grid), which also sidesteps rasterio's CRS
    parser choking on the LOCAL_CS tag. The fuel raster is finer (3 m) and covers
    only the study area, so reference cells whose centre falls outside its extent
    are returned as nodata (-> default weight downstream). Kept as uint8 (no
    float64 cast) to avoid blowing up memory on the ~150M-pixel raster."""
    with rasterio.open(FUEL) as ds:
        fuel = ds.read(1)
        ft, fh, fw = ds.transform, ds.height, ds.width
        fnodata = ds.nodata if ds.nodata is not None else 7
        left, bottom, right, top = ds.bounds
    height, width = grid["shape"]
    st = grid["transform"]
    # world coords of reference-grid cell centres (north-up rasters: no rotation)
    xs = st.c + st.a * (np.arange(width) + 0.5)               # reference-grid column centres, in map X
    ys = st.f + st.e * (np.arange(height) + 0.5)              # reference-grid row centres, in map Y
    fcol = np.clip(np.round((xs - ft.c) / ft.a - 0.5).astype(int), 0, fw - 1)   # -> fuel raster column (nearest)
    frow = np.clip(np.round((ys - ft.f) / ft.e - 0.5).astype(int), 0, fh - 1)   # -> fuel raster row (nearest)
    sampled = fuel[np.ix_(frow, fcol)]                          # nearest-neighbour resample, fine grid -> coarse grid
    in_extent = np.outer((ys >= bottom) & (ys <= top), (xs >= left) & (xs <= right))   # reference cells outside fuel coverage
    return np.where(in_extent, sampled, fnodata)


def buffer_width_grid():
    """PLACEHOLDER variable buffer-width raster (metres) = f(fuel, slope).

    Swap this whole function for a real FLI raster (Cell2Fire / fuel model);
    everything downstream (block + reroute) stays the same.
    """
    slope, grid = _read_reference()
    fuel = _read_fuel_on(grid)

    weight = np.full(grid["shape"], FUEL_WEIGHT_DEFAULT)        # default: low-flammability / unknown fuel
    for cls, w in FUEL_WEIGHT.items():
        weight[fuel == cls] = w                                 # reclassify: fuel class -> flammability weight

    intensity = weight * _slope_factor(slope)            # proxy, dimensionless
    width = BUFFER_FLOOR_M + (intensity - INTENSITY_REF) * WIDTH_SCALE   # intensity -> metres of buffer, linearly
    width = np.clip(width, BUFFER_FLOOR_M, BUFFER_MAX_M)         # never below the floor, never above the cap
    return width, grid


def fuel_hazard_overlay(max_px=1200):
    """Downsampled RGBA image of the fuel raster coloured by FLAMMABILITY
    (FUEL_WEIGHT: green = low danger -> red = high danger), plus its lat/lon
    bounds, for a folium ImageOverlay. Lets the map show WHY the buffer/friction
    behave as they do. Returns (rgba uint8 [H, W, 4], [[south, west], [north, east]]).

    The fuel raster is huge (~150M px) so it is read decimated; bounds are the
    Greek-Grid extent reprojected corner-wise to lat/lon (a small skew is fine for
    a hazard backdrop)."""
    from rasterio.enums import Resampling
    with rasterio.open(FUEL) as ds:
        scale = max(1, round(max(ds.width, ds.height) / max_px))   # decimation factor -> keeps the overlay light
        oh, ow = ds.height // scale, ds.width // scale
        fuel = ds.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)   # downsampled read, not full-res then shrink
        left, bottom, right, top = ds.bounds
    danger = np.full(fuel.shape, np.nan, dtype="float32")
    for cls, w in FUEL_WEIGHT.items():
        danger[fuel == cls] = w
    valid = ~np.isnan(danger)
    t = np.clip((danger - 0.3) / 0.7, 0.0, 1.0)          # 0.30..1.00 -> 0..1
    rgba = np.zeros((*fuel.shape, 4), dtype="uint8")
    rgba[..., 0] = np.where(valid, 255 * t, 0)           # red rises with danger
    rgba[..., 1] = np.where(valid, 255 * (1 - t), 0)     # green falls with danger
    rgba[..., 3] = np.where(valid, 150, 0)               # transparent where nodata
    corners = gpd.GeoSeries(                              # bbox corners -> WGS84, so folium can place the image overlay
        [Point(left, bottom), Point(right, bottom), Point(right, top), Point(left, top)],
        crs=ANALYSIS_CRS).to_crs(POINT_CRS)
    xs, ys = [p.x for p in corners], [p.y for p in corners]
    return rgba, [[min(ys), min(xs)], [max(ys), max(xs)]]


# --- Fire -> blocked zone ----------------------------------------------------
def blocked_zone(fire, width, grid):
    """Blocked = cells within their local buffer width of the fire.

    Returns a single (Multi)Polygon (EPSG:2100), or None if nothing is blocked.
    """
    fire_mask = features.rasterize(                       # fire polygon -> boolean raster on the reference grid
        [(fire, 1)], out_shape=grid["shape"], transform=grid["transform"],
        fill=0, dtype="uint8",
    ).astype(bool)
    dist_m = distance_transform_edt(~fire_mask) * grid["res"]   # metres to fire
    blocked = dist_m <= width                                   # variable width -- NOT a uniform buffer() call
    geoms = [
        shape(geom)
        for geom, val in features.shapes(                  # blocked mask -> vector polygon(s)
            blocked.astype("uint8"), mask=blocked, transform=grid["transform"]
        )
        if val == 1
    ]
    return unary_union(geoms) if geoms else None            # dissolve into one (Multi)Polygon
