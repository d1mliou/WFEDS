"""Build a Cell2Fire instance - PARAMETERIZED (user inputs OR a named scenario).

The deterministic pipeline's entry gate. Two ways in, one code path:

  * USER INPUTS (operational use - what the LLM agent tool passes through):
        build_instance(ignition_points=[(lat, lon), ...],   # WGS84
                       window_km=..., horizon_h=..., start_time="YYYY-MM-DDTHH:MM")
    EVERY pin is an OBSERVATION of fire with a physical footprint: each point is
    buffered by half a VIIRS pixel (SEED_RADIUS_M = 187.5 m), the buffers are
    unioned, rasterized onto the grid, and the burnable cells under the footprint
    become InitialBurned.csv (the `--InitialBurned` engine mod ignites each cell
    at period 0). Pins <= ~375 m apart merge into ONE front purely by geometry;
    farther pins stay disjoint polygons = independent fires. Same rule for any
    number of pins - no special cases.
    The window is a square of `window_km` centred on the points' centroid.

  * NAMED SCENARIO (fixed parameter sets - e.g. the validation reference):
        build_instance(scenario="north_evia_2021")
    reproduces the 2021 North Evia case (VIIRS-bbox window, real ignition, real
    weather, 24 h). Its observed front - the VIIRS detections up to `seed_time` -
    goes through the SAME seeding rule automatically (one function, both paths).

Everything is written onto ONE grid (EPSG:2100, the slope grid's 30 m cells):

    Forest.asc            FBP fuel codes  (SVM 5-class -> FBP, PLACEHOLDER mapping)
    elevation.asc         from DEM.tif
    slope.asc             from Slope_Degrees.tif (degrees)
    saz.asc               aspect / slope azimuth (deg) computed from the DEM
    fbp_lookup_table.csv  canonical Canadian FBP fuel defs (bundled, copied in)
    Weather.csv           REAL Open-Meteo/ERA5 weather + FWI codes computed from it
    IgnitionPoints.csv    one ignition cell (engine-mandatory anchor)
    InitialBurned.csv     the seeded footprint cells (always written)
    scenario_params.json  provenance: which inputs produced this instance

GUARDRAILS (refuse clearly instead of failing weirdly): ignition must fall inside
the data coverage (fuel raster); window/horizon bounded; start_time+horizon must
not extend further than FORECAST_MAX_DAYS_AHEAD into the future (the weather
comes from Open-Meteo's ARCHIVE for older dates and FORECAST for recent/future
ones, stitched transparently by meteo.fetch_weather_wind/_grid - no lower bound,
archive goes back decades).

PROVISIONAL: the SVM->FBP mapping is an uncalibrated placeholder (calibrate with
the domain expert).

Run (CLI):
    python scripts/cell2fire/build_cell2fire_instance.py                # the 2021 scenario
    python scripts/cell2fire/build_cell2fire_instance.py \
        --points "38.86,23.23;38.87,23.25" --window-km 16 --horizon 12 \
        --start 2024-08-10T12:00 [--out DIR]
"""

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window
from shapely.ops import unary_union

import fwi                                              # FWI codes (fire-weather indices)
from _paths import DATA_DIR, DEM, FUEL, SLOPE           # data catalog: terrain + fuel rasters
from meteo import (WeatherGapError, assert_contiguous_hourly, fetch_weather_grid,
                   fetch_weather_wind, idw_series, _archive_cutoff)   # weather fetch + IDW

ANALYSIS_CRS = "EPSG:2100"                              # Greek Grid, metric
NODATA = -9999                                          # nodata value for the .asc grids

# --- Guardrails (bounds a user/LLM request must respect) ----------------------
WINDOW_KM_MIN, WINDOW_KM_MAX = 6.0, 40.0     # runtime scales with cell count
HORIZON_H_MIN, HORIZON_H_MAX = 1, 48
# Open-Meteo's forecast ceiling is 16 days (verified live 2026-07-07); 1 day of
# margin absorbs the +24h pad the weather fetch below adds beyond the horizon.
# No lower bound - the archive goes back decades; meteo.fetch_weather_* routes
# each date to whichever source (archive/forecast) actually has it.
FORECAST_MAX_DAYS_AHEAD = 15
FWI_SPINUP_DAYS = 65         # DC has ~50-day memory; spin-up starts this far back

# --- The ONE seeding rule: every observed point gets its physical footprint ----
# A point (user pin OR VIIRS detection) is an OBSERVATION whose reported position
# is the CENTRE of what was seen. A VIIRS pixel is 375 m wide, so each observation
# is buffered by half a pixel (187.5 m) - a pin claims no more positional accuracy
# than a satellite detection. Buffers <= 375 m apart merge into ONE front
# (unary_union); farther apart they stay disjoint polygons = independent fires.
# The unioned footprint is rasterized and only BURNABLE cells become the engine
# seed (InitialBurned.csv) - rivers/roads/non-fuel never ignite.
VIIRS_PIXEL_M = 375.0
SEED_RADIUS_M = VIIRS_PIXEL_M / 2.0

# --- SVM 5-class -> Canadian FBP fuel code (PLACEHOLDER, uncalibrated) --------
# Reclassify our land-cover classes into the engine's fuel model (like a raster
# reclassify: each SVM class -> one FBP fuel type).
SVM_TO_FBP = {
    0: 2,    # Κωνοφόρα conifer      -> C-2 (boreal spruce, high ROS)
    1: 3,    # Μεικτό mixed forest   -> C-3 (mature pine)  [C-types avoid M pc/pdf params]
    2: 11,   # Πλατύφυλλα broadleaf  -> D-1 (aspen)
    3: 7,    # Λοιπές δασικές        -> C-7 (ponderosa / douglas-fir)
    4: 101,  # Μη δάσος non-forest   -> Non-fuel
}
FBP_NONFUEL = 101
FUEL_CODES = sorted(set(SVM_TO_FBP.values()) - {FBP_NONFUEL})   # burnable codes

REAL_IGNITION = DATA_DIR / "Real_Fire_Data" / "Ignition" / "Fire_ignition_point.shp"   # real 2021 ignition point
VIIRS = DATA_DIR / "Real_Fire_Data" / "VIIRS" / "VIRS_dataset_egsa_dhmos.shp"          # satellite active-fire detections
DEFAULT_OUT = DATA_DIR / "Fire" / "cell2fire" / "instance"
LOOKUP_SRC = Path(__file__).with_name("fbp_lookup_table.csv")   # canonical FBP fuel defs (bundled)

# --- Named scenarios: fixed parameter sets over the SAME code path ------------
# The validation reference. Window = bbox of ALL VIIRS detections + margin (adopted
# 2026-07-02 after the centred box clipped the real N/NW run); ignition = the real
# 2021 point; weather = the real event hours. The LLM can invoke it by name
# ("τρέξε επικύρωση για τη φωτιά της Β. Εύβοιας") with no geometry needed.
SCENARIOS = {
    "north_evia_2021": dict(
        description="North Evia 2021 wildfire - the validation reference event",
        window_mode="viirs_bbox",
        window_margin_m=400.0,
        ignition_source="real_2021",          # REAL_IGNITION shapefile
        start_time="2021-08-05T23:00",        # UTC; first VIIRS detection ~23:22
        seed_time="2021-08-06 02:00",         # observed front "as detected" (first ~3 VIIRS overpasses) -> seeded via the ONE rule
        horizon_h=24,
        wind_dir_source="era5",               # "viirs" kept as legacy analyst option
        wind_dir_obs_hours=3,                 # ("viirs" only) leakage-safe window
        fwi_spinup_start="2021-06-01",
    ),
}


def viirs_wind_dir(centre, start_iso, hours):
    """Wind DIRECTION (deg, coming-from) implied by the observed VIIRS first-day spread.

    The fire blows TOWARD the mean bearing of the first-day VIIRS detections from the
    ignition; the wind comes FROM the opposite (= bearing + 180). Returns (wd_from,
    blows_to). Legacy analyst option (vs ERA5's per-hour direction)."""
    v = gpd.read_file(VIIRS).to_crs(ANALYSIS_CRS)        # load detections, reproject to Greek Grid
    v["t"] = pd.to_datetime(v["acq_at_s"])
    t0 = pd.to_datetime(start_iso.replace("T", " "))
    v1 = v[(v["t"] >= t0) & (v["t"] <= t0 + pd.Timedelta(hours=hours))]   # keep only the first-day detections
    c = v1.geometry.union_all().centroid                 # their centroid = where the fire ran to
    blows_to = (math.degrees(math.atan2(c.x - centre.x, c.y - centre.y)) + 360) % 360   # bearing ignition -> spread
    return (blows_to + 180) % 360, blows_to             # wind comes FROM the opposite bearing


def aspect_deg(dem, cellsize):
    """Compass azimuth (deg from N, clockwise) of the uphill direction; flat -> 0."""
    gy, gx = np.gradient(dem.astype("float64"), cellsize)   # gy:+row(S), gx:+col(E)
    dz_de, dz_dn = gx, -gy                                  # rows increase southward
    asp = (np.degrees(np.arctan2(dz_de, dz_dn)) + 360.0) % 360.0
    return np.where((dz_de == 0) & (dz_dn == 0), 0.0, asp)  # flat cells -> aspect 0


def write_asc(path, arr, wt, fmt):
    """Write an Arc/Info ASCII grid (no CRS) using the window transform `wt`."""
    nrows, ncols = arr.shape
    cell = wt.a
    xll = wt.c
    yll = wt.f + nrows * wt.e        # wt.f = top, wt.e < 0 -> bottom-left corner
    header = (f"ncols {ncols}\nnrows {nrows}\nxllcorner {xll:.6f}\n"
              f"yllcorner {yll:.6f}\ncellsize {cell:.6f}\nNODATA_value {NODATA}\n")
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, arr, fmt=fmt, delimiter=" ")


class InstanceInputError(ValueError):
    """A user/LLM input the builder refuses - message is meant to be shown as-is."""


def _coverage_bounds_wgs84():
    """The data-coverage rectangle (fuel raster extent) in lat/lon, for messages."""
    with rasterio.open(FUEL) as ds:
        b = ds.bounds
    c = gpd.GeoSeries(gpd.points_from_xy([b.left, b.right], [b.bottom, b.top]),
                      crs=ANALYSIS_CRS).to_crs("EPSG:4326")   # corners -> lat/lon for the message
    return (round(c.y.min(), 3), round(c.x.min(), 3),
            round(c.y.max(), 3), round(c.x.max(), 3))


def _validate_user_inputs(points_ll, window_km, horizon_h, start_time):
    """Guardrails for the operational path. Raises InstanceInputError with a clear,
    user-facing message (the LLM relays it verbatim)."""
    if not points_ll:
        raise InstanceInputError(
            "Χρειάζομαι τουλάχιστον ένα σημείο έναρξης (1 σημείο = σημειακή έναυση, "
            "περισσότερα = μέτωπο).")
    if window_km is None or not (WINDOW_KM_MIN <= window_km <= WINDOW_KM_MAX):   # window within safe bounds
        raise InstanceInputError(
            f"Το παράθυρο προσομοίωσης πρέπει να είναι {WINDOW_KM_MIN:.0f}-"
            f"{WINDOW_KM_MAX:.0f} km (ζητήθηκε: {window_km}). Μεγαλύτερο παράθυρο = "
            f"πολύ μεγαλύτερος χρόνος εκτέλεσης.")
    if not (HORIZON_H_MIN <= horizon_h <= HORIZON_H_MAX):                       # horizon within safe bounds
        raise InstanceInputError(
            f"Ο ορίζοντας πρέπει να είναι {HORIZON_H_MIN}-{HORIZON_H_MAX} ώρες "
            f"(ζητήθηκε: {horizon_h}).")
    if start_time is None:
        raise InstanceInputError("Χρειάζομαι ημερομηνία/ώρα έναρξης (UTC).")
    t0 = datetime.fromisoformat(start_time)
    # UTC "now" (start_time is documented/used as UTC throughout) - a naive
    # host-local datetime.now() would drift against t0 by the local UTC offset,
    # a latent bug that only started mattering once "now" became a valid input.
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    horizon_end = t0 + timedelta(hours=horizon_h)
    furthest_ok = now_utc + timedelta(days=FORECAST_MAX_DAYS_AHEAD)
    if horizon_end > furthest_ok:                        # can't simulate past the weather forecast ceiling
        raise InstanceInputError(
            f"Η πρόγνωση καιρού φτάνει έως ~{FORECAST_MAX_DAYS_AHEAD} μέρες "
            f"μπροστά (έως {furthest_ok:%Y-%m-%d}). Δώσε έναρξη+ορίζοντα πριν "
            f"από αυτό.")
    # ignition must fall inside the data coverage (fuel raster extent)
    with rasterio.open(FUEL) as ds:
        fb = ds.bounds
    pts = gpd.GeoSeries(gpd.points_from_xy([p[1] for p in points_ll],
                                           [p[0] for p in points_ll]),
                        crs="EPSG:4326").to_crs(ANALYSIS_CRS)   # ignition points -> Greek Grid
    outside = [(round(ll[0], 4), round(ll[1], 4))
               for ll, p in zip(points_ll, pts)
               if not (fb.left <= p.x <= fb.right and fb.bottom <= p.y <= fb.top)]   # point-in-extent test
    if outside:
        s, w, n, e = _coverage_bounds_wgs84()
        raise InstanceInputError(
            f"Σημεία εκτός κάλυψης δεδομένων (καύσιμο/DEM υπάρχουν μόνο για τη "
            f"Β. Εύβοια, lat {s}-{n}, lon {w}-{e}): {outside}")
    return pts, t0


def _nearest_burnable_cell(x, y, wt, br, bc, ncols):
    """Row/col + 1-based Ncell of the burnable cell nearest to map coords (x, y).
    `br`, `bc` = row/col arrays of ALL burnable cells (np.where(burnable)),
    computed ONCE by the caller and passed in - not recomputed per call."""
    r = int(round((y - wt.f) / wt.e))                   # map coords -> grid row
    c = int(round((x - wt.c) / wt.a))                   # map coords -> grid col
    nearest = ((br - r) ** 2 + (bc - c) ** 2).argmin()  # nearest burnable cell (squared distance)
    ir, ic = int(br[nearest]), int(bc[nearest])
    return ir, ic, ir * ncols + ic + 1                  # 1-based row-major cell id


def seed_cells_from_points(points_2100, wt, nrows, ncols, burnable):
    """The ONE seeding rule (user pins AND VIIRS detections alike).

    Every observed point -> its physical footprint: buffer by SEED_RADIUS_M
    (half a VIIRS pixel - the reported position is the pixel centre), union the
    buffers, rasterize the footprint onto the window grid, keep only burnable
    cells. Nearby points merge into one front purely by geometry (buffers touch
    at <= VIIRS_PIXEL_M apart); far-apart points stay disjoint = independent
    fires - no clustering heuristic, no drawn polyline.

    Returns (ncell_ids, n_fronts): sorted 1-based row-major cell ids (the
    `--InitialBurned` file format) + the number of disjoint footprint polygons.
    """
    if not points_2100:
        return [], 0
    footprint = unary_union([p.buffer(SEED_RADIUS_M) for p in points_2100])   # each observation -> its footprint disc; touching discs merge
    n_fronts = 1 if footprint.geom_type == "Polygon" else len(footprint.geoms)   # disjoint polygons = independent fires
    mask = features.rasterize([(footprint, 1)], out_shape=(nrows, ncols),
                              transform=wt, fill=0, dtype="uint8").astype(bool)   # footprint -> boolean grid mask
    rr, cc = np.where(mask & burnable)                  # only burnable cells can be "already burning"
    return sorted((rr * ncols + cc + 1).tolist()), n_fronts   # 1-based row-major Ncell ids


def build_instance(ignition_points=None, window_km=None, horizon_h=24,
                   start_time=None, out_dir=None, scenario=None,
                   wind_dir_source="era5"):
    """Build a Cell2Fire instance from USER INPUTS or a NAMED SCENARIO.

    ignition_points : [(lat, lon), ...] WGS84. Every pin is an observation with a
                      ~375 m footprint (SEED_RADIUS_M buffer, unioned); far-apart
                      pins become independent fires (see the module docstring).
    window_km       : square window side (km), centred on the points' centroid.
    horizon_h       : hours to simulate (= hourly weather rows).
    start_time      : "YYYY-MM-DDTHH:MM" UTC (archive weather; see guardrails).
    out_dir         : instance folder (default: the canonical instance dir).
    scenario        : name in SCENARIOS - overrides all of the above.

    Returns a dict summary (also written as scenario_params.json for provenance).
    Raises InstanceInputError with a user-facing message on invalid input.
    """
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT

    if scenario is not None:                            # PATH A: named scenario (fixed params)
        if scenario not in SCENARIOS:
            raise InstanceInputError(
                f"Άγνωστο σενάριο '{scenario}'. Διαθέσιμα: {sorted(SCENARIOS)}")
        sc = SCENARIOS[scenario]
        start_time = sc["start_time"]
        horizon_h = sc["horizon_h"]
        wind_dir_source = sc["wind_dir_source"]
        spinup_start = sc["fwi_spinup_start"]
        centre = gpd.read_file(REAL_IGNITION).to_crs(ANALYSIS_CRS).geometry.iloc[0]   # real 2021 ignition point
        pts_2100 = [centre]
        window_mode = sc["window_mode"]
        window_margin_m = sc["window_margin_m"]
        src = f"scenario '{scenario}' (REAL ignition)"
    else:                                               # PATH B: user/LLM inputs (validated)
        if wind_dir_source == "viirs":
            raise InstanceInputError(
                "Η κατεύθυνση ανέμου 'viirs' είναι διαθέσιμη μόνο για το σενάριο "
                "αναφοράς (VIIRS = ground truth της φωτιάς 2021, όχι είσοδος "
                "χρήστη). Χρησιμοποίησε 'era5' ή δώσε σταθερή γωνία.")
        pts_2100_series, t0 = _validate_user_inputs(
            ignition_points, window_km, horizon_h, start_time)
        pts_2100 = list(pts_2100_series)
        centre = pts_2100_series.union_all().centroid   # window centre = centroid of all points
        spinup_start = (t0 - timedelta(days=FWI_SPINUP_DAYS)).strftime("%Y-%m-%d")
        window_mode = "centered"
        window_margin_m = 0.0
        src = f"user input ({len(pts_2100)} pin(s))"
    cx, cy = float(centre.x), float(centre.y)
    print(f"Window centre ({src}): ({cx:.0f}, {cy:.0f}) Greek Grid")

    # --- crop the window on the slope grid ------------------------------------
    with rasterio.open(SLOPE) as s:                     # slope raster = the master grid (30 m cells)
        st = s.transform
        res = st.a
        if window_mode == "viirs_bbox":                 # scenario: window = bbox of all VIIRS detections + margin
            vb = gpd.read_file(VIIRS).to_crs(ANALYSIS_CRS).total_bounds
            x0, y0 = vb[0] - window_margin_m, vb[1] - window_margin_m
            x1, y1 = vb[2] + window_margin_m, vb[3] + window_margin_m
            c0 = max(0, int((x0 - st.c) / st.a))        # bbox -> pixel col/row indices (clamped to raster)
            c1 = min(s.width, int(math.ceil((x1 - st.c) / st.a)))
            r0 = max(0, int((y1 - st.f) / st.e))          # st.e negative: top row
            r1 = min(s.height, int(math.ceil((y0 - st.f) / st.e)))
            print(f"Window mode viirs_bbox: {(c1 - c0)}x{(r1 - r0)} cells "
                  f"({(c1 - c0) * res / 1000:.1f} x {(r1 - r0) * res / 1000:.1f} km, "
                  f"margin {window_margin_m:.0f} m)")
        else:                                           # user path: square window centred on the centroid
            col = int((cx - st.c) / st.a)
            row = int((cy - st.f) / st.e)        # st.e negative
            half = int(round(window_km * 1000.0 / res / 2))
            c0, r0 = max(0, col - half), max(0, row - half)
            c1, r1 = min(s.width, col + half), min(s.height, row + half)
            want = 2 * half
            got = min(c1 - c0, r1 - r0)
            if got < want:                              # window hit the data edge -> warn (fire can't leave data)
                print(f"NB: window clipped by the data extent "
                      f"({got * res / 1000:.1f} km on the narrow side, "
                      f"asked {window_km:.1f}) - the fire cannot leave the data.")
        win = Window(c0, r0, c1 - c0, r1 - r0)          # the raster read-window
        slope_w = s.read(1, window=win).astype("float64")   # read slope inside the window
        wt = s.window_transform(win)                    # window's affine transform (pixel<->map coords)
        snod = s.nodata
    nrows, ncols = slope_w.shape
    if snod is not None:
        slope_w = np.where(slope_w == snod, 0.0, slope_w)   # nodata slope -> flat (0)
    print(f"Window grid: {nrows} x {ncols} cells @ {res:.2f} m "
          f"({nrows * ncols / 1000:.0f}k cells)")

    # --- guardrail: every point must fall inside the ACTUAL cropped window -----
    # _validate_user_inputs only checks the FUEL raster's full coverage bbox; the
    # window is cropped from SLOPE and may be smaller than the points' spread
    # (or clipped at the raster edge). Refuse clearly instead of silently
    # truncating the front.
    if scenario is None:
        xmin, xmax = wt.c, wt.c + ncols * wt.a          # window extent in map coords
        ymin, ymax = wt.f + nrows * wt.e, wt.f
        # Numeric check on pts_2100 (EPSG:2100 metres, matches xmin/xmax/ymin/
        # ymax); `ll` (WGS84, from ignition_points) is used ONLY for the
        # human-readable message - never mix the two CRS in the comparison.
        outside_win = [(round(ll[0], 4), round(ll[1], 4))
                       for ll, p in zip(ignition_points, pts_2100)
                       if not (xmin <= p.x <= xmax and ymin <= p.y <= ymax)]   # any point outside the window?
        if outside_win:
            raise InstanceInputError(
                f"Σημεία εκτός του επιλεγμένου παραθύρου {window_km:.0f} km "
                f"(κεντραρισμένο στο κέντρο βάρους όλων των σημείων): "
                f"{outside_win}. Αύξησε το παράθυρο (έως {WINDOW_KM_MAX:.0f} km) "
                f"ή, αν πρόκειται για άσχετες φωτιές, στείλε τα σε ξεχωριστά "
                f"αιτήματα.")

    # --- DEM (same grid) -> elevation + aspect ---------------------------------
    with rasterio.open(DEM) as d:
        dem_w = d.read(1, window=win).astype("float64")   # read elevation in the SAME window
        dnod = d.nodata
    if dnod is not None and (dem_w == dnod).any():
        fill = float(np.median(dem_w[dem_w != dnod])) if (dem_w != dnod).any() else 0.0
        dem_w = np.where(dem_w == dnod, fill, dem_w)      # fill nodata with the median elevation
    asp_w = aspect_deg(dem_w, res)                        # derive aspect (slope azimuth) from the DEM

    # --- fuel: nearest-sample the 3 m SVM raster onto the window grid ---------
    with rasterio.open(FUEL) as fz:
        ft = fz.transform
        fwid, fhei = fz.width, fz.height
        fuel = fz.read(1)
        fb = fz.bounds
    xs = wt.c + wt.a * (np.arange(ncols) + 0.5)           # cell-centre map X per window column
    ys = wt.f + wt.e * (np.arange(nrows) + 0.5)           # cell-centre map Y per window row
    fcol = np.clip(np.round((xs - ft.c) / ft.a - 0.5).astype(int), 0, fwid - 1)   # -> fuel raster col (nearest)
    frow = np.clip(np.round((ys - ft.f) / ft.e - 0.5).astype(int), 0, fhei - 1)   # -> fuel raster row (nearest)
    svm = fuel[np.ix_(frow, fcol)]                        # resample fuel onto the window grid
    in_ext = np.outer((ys >= fb.bottom) & (ys <= fb.top), (xs >= fb.left) & (xs <= fb.right))   # inside fuel coverage?
    fbp = np.full((nrows, ncols), FBP_NONFUEL, dtype=int)
    for k, v in SVM_TO_FBP.items():
        fbp[(svm == k) & in_ext] = v                     # reclassify SVM class -> FBP fuel code
    burnable = np.isin(fbp, FUEL_CODES)                  # boolean mask of burnable cells
    print(f"Fuel: {int(burnable.sum())} burnable / {nrows * ncols} cells "
          f"({100 * burnable.mean():.0f}%)")

    # --- ignition cell + the seeded footprint (the ONE rule, both paths) -------
    br, bc = np.where(burnable)                     # computed ONCE, reused below
    if not len(br):
        raise InstanceInputError(
            "Κανένα καύσιμο κελί στο παράθυρο - η περιοχή είναι μη-καύσιμη στα "
            "δεδομένα μας. Δοκίμασε άλλο σημείο/παράθυρο.")

    if scenario is None:
        # Anchor the mandatory single ignition cell to a REAL input point (the
        # first one), NOT the overall centroid: with multiple disconnected
        # fronts the centroid can fall in empty space BETWEEN two unrelated
        # fires, inventing a fake third ignition. (cx, cy stay untouched - the
        # window crop above must remain centred on the true centroid.)
        icx, icy = float(pts_2100[0].x), float(pts_2100[0].y)
        seed_pts = pts_2100                         # the user's observations seed the fire directly
        seed_src = f"{len(seed_pts)} pin(s)"
    else:
        icx, icy = cx, cy                           # scenario: the real ignition point
        # The scenario's observed front: VIIRS detections up to seed_time (the
        # fire "as detected") go through the SAME seeding rule as user pins.
        v = gpd.read_file(VIIRS).to_crs(ANALYSIS_CRS)
        v["t"] = pd.to_datetime(v["acq_at_s"])
        v = v[v["t"] <= pd.to_datetime(sc["seed_time"])]   # only what was OBSERVED by seed_time (no future leakage)
        seed_pts = list(v.geometry)
        seed_src = f"{len(seed_pts)} VIIRS detections <= {sc['seed_time']}"

    ir, ic, ncell = _nearest_burnable_cell(icx, icy, wt, br, bc, ncols)   # snap ignition to nearest burnable cell
    print(f"Ignition: row {ir}, col {ic} -> Ncell {ncell} (FBP {fbp[ir, ic]})")

    front_ncells, n_fronts = seed_cells_from_points(seed_pts, wt, nrows,
                                                    ncols, burnable)     # the ONE seeding rule
    if not front_ncells:
        raise InstanceInputError(
            f"Καμία παρατήρηση δεν ακουμπά καύσιμο κελί (αποτύπωμα "
            f"{SEED_RADIUS_M:.0f} m γύρω από κάθε σημείο) - έλεγξε τα σημεία.")
    print(f"Seed ({seed_src}): {n_fronts} front(s) -> "
          f"{len(front_ncells)} burnable cells (InitialBurned.csv)")

    # --- write the instance -----------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    write_asc(out_dir / "Forest.asc", fbp, wt, "%d")        # fuel-code grid
    write_asc(out_dir / "elevation.asc", dem_w, wt, "%.2f")  # elevation grid
    write_asc(out_dir / "slope.asc", slope_w, wt, "%.2f")    # slope grid (degrees)
    write_asc(out_dir / "saz.asc", asp_w, wt, "%.1f")        # aspect / slope-azimuth grid
    shutil.copy(LOOKUP_SRC, out_dir / "fbp_lookup_table.csv")   # bundle the FBP fuel definitions

    # Invalidate any stale Data.csv/.dat: the grid may have changed, but `--gen-data`
    # skips regeneration if Data.csv exists -> a stale one (from a different window)
    # has the WRONG row count and the C++ reads DF[i] out of bounds (SIGSEGV).
    for stale in ("Data.csv", "Data.dat"):
        (out_dir / stale).unlink(missing_ok=True)

    # IgnitionPoints.csv (Python side) + Ignitions.csv (read by the C++ binary)
    for fname in ("IgnitionPoints.csv", "Ignitions.csv"):
        with open(out_dir / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Year", "Ncell"])
            w.writerow([1, ncell])
    if front_ncells:                                    # the seeded footprint (engine ignites each cell at period 0)
        with open(out_dir / "InitialBurned.csv", "w", newline="") as f:
            f.write("Ncell\n")
            f.writelines(f"{n}\n" for n in front_ncells)

    # --- weather: REAL Open-Meteo/ERA5, FWI codes computed from it -------------
    ll = gpd.GeoSeries([centre], crs=ANALYSIS_CRS).to_crs("EPSG:4326").iloc[0]   # window centre -> lat/lon for the API
    ign_dt = datetime.fromisoformat(start_time)
    d0 = (ign_dt - timedelta(days=1)).strftime("%Y-%m-%d")            # fetch from 1 day before...
    d1 = (ign_dt + timedelta(hours=horizon_h + 24)).strftime("%Y-%m-%d")   # ...to horizon + 24 h pad
    cor = gpd.GeoSeries(
        gpd.points_from_xy([wt.c, wt.c + ncols * wt.a],
                           [wt.f + nrows * wt.e, wt.f]),
        crs=ANALYSIS_CRS).to_crs("EPSG:4326")           # window corners -> lat/lon (weather bbox)
    try:
        grid = fetch_weather_grid(cor.y.min(), cor.x.min(), cor.y.max(), cor.x.max(),
                                  nx=5, ny=5, start_date=d0, end_date=d1)   # 5x5 grid of ERA5 points over the window
    except WeatherGapError as e:
        raise InstanceInputError(str(e)) from e
    n_cells = len({(round(m["lat"], 3), round(m["lon"], 3)) for _, m in grid})   # distinct ERA5 cells hit
    era = idw_series(grid, ll.y, ll.x)                  # IDW the grid onto the window centre
    try:
        assert_contiguous_hourly(era, context="κύριο παράθυρο")   # guard: no hourly gaps
    except WeatherGapError as e:
        raise InstanceInputError(str(e)) from e
    wsrc = f"IDW over {n_cells} ERA5 cells @ window centre"
    era = [r for r in era if r["time"] >= start_time][:horizon_h]   # keep exactly the horizon hours
    if len(era) < horizon_h:
        raise InstanceInputError(
            f"Ο καιρός κάλυψε μόνο {len(era)}/{horizon_h} ώρες από {start_time} - "
            f"έλεγξε την ημερομηνία.")

    # wind DIRECTION: Open-Meteo per hour (default), analyst-fixed, or legacy VIIRS
    if wind_dir_source == "viirs":
        obs_h = SCENARIOS.get(scenario, {}).get("wind_dir_obs_hours", 3)
        wd_fixed, blows = viirs_wind_dir(centre, start_time, obs_h)   # legacy: direction from observed spread
        print(f"Wind DIR from VIIRS (legacy; first {obs_h} h only): fire blows "
              f"{blows:.0f} deg -> WD {wd_fixed:.0f} (from)")
    elif wind_dir_source == "era5":
        wd_fixed = None                                 # None = use ERA5's per-hour direction
    else:
        wd_fixed = float(wind_dir_source)               # a fixed analyst-supplied angle

    # FWI codes from the real weather: daily spin-up at the same point, then
    # per-hour ISI (day FFMC + hour wind) / BUI / FWI.
    try:
        spin_rows, _ = fetch_weather_wind(ll.y, ll.x, spinup_start, d1)   # long weather history for the spin-up
        assert_contiguous_hourly(spin_rows, context="FWI spin-up")
    except WeatherGapError as e:
        raise InstanceInputError(str(e)) from e
    daily = fwi.spinup_daily_codes(spin_rows)           # daily FFMC/DMC/DC built up over the spin-up
    dates = sorted(daily)

    def day_codes(t):
        date = t[:10]
        if date in daily:
            return daily[date]
        prev = [x for x in dates if x < date]           # no entry that day -> carry the last available codes
        return daily[prev[-1]] if prev else (fwi.FFMC_INIT, fwi.DMC_INIT,
                                             fwi.DC_INIT)

    csv_rows = []
    for r in era:                                       # per hour: combine daily codes + this hour's wind
        ws = r["ws_kmh"]
        wd = r["wd_from"] if wd_fixed is None else wd_fixed
        f_, p_, d_ = day_codes(r["time"])
        isi_h = fwi.isi(f_, ws)                          # Initial Spread Index (FFMC + wind)
        bui_h = fwi.bui(p_, d_)                          # Buildup Index (DMC + DC)
        fwi_h = fwi.fwi(isi_h, bui_h)                    # Fire Weather Index
        csv_rows.append(["NEVIA", r["time"].replace("T", " "),
                         round(r.get("precip_mm") or 0.0, 2),
                         r["temp_c"], r["rh_pct"], round(ws, 1), round(wd),
                         round(f_, 2), round(p_, 2), round(d_, 2),
                         round(isi_h, 2), round(bui_h, 2), round(fwi_h, 2)])
    wss = [r[5] for r in csv_rows]
    print(f"Weather: {wsrc}, {len(csv_rows)} h | WS {min(wss):.0f}-{max(wss):.0f} "
          f"km/h | DIR '{wind_dir_source}'")
    print(f"FWI computed from weather (spin-up {spinup_start}): "
          f"FFMC {csv_rows[0][7]:.1f} DMC {csv_rows[0][8]:.0f} DC "
          f"{csv_rows[0][9]:.0f} | ISI {min(r[10] for r in csv_rows):.1f}-"
          f"{max(r[10] for r in csv_rows):.1f}")

    cols = ["Scenario", "datetime", "APCP", "TMP", "RH", "WS", "WD",
            "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    with open(out_dir / "Weather.csv", "w", newline="") as f:   # the engine's hourly weather table
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(csv_rows)

    # --- provenance: which inputs produced this instance ------------------------
    params = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenario": scenario,
        "ignition_points_wgs84": ignition_points if scenario is None else "scenario",
        "mode": ("scenario" if scenario else            # user path: geometry decides, no hidden point-count rule
                 ("multi_front" if n_fronts >= 2
                  else ("front" if len(pts_2100) >= 2 else "point_ignition"))),
        "window_mode": window_mode, "window_km": window_km,
        "grid": {"nrows": nrows, "ncols": ncols, "cell_m": res},
        "start_time_utc": start_time, "horizon_h": horizon_h,
        "wind_dir_source": str(wind_dir_source), "fwi_spinup_start": spinup_start,
        "weather_source": ("archive" if d1 <= _archive_cutoff() else
                           "forecast" if d0 > _archive_cutoff() else
                           "archive+forecast"),
        "ignition_ncell": ncell, "front_cells": len(front_ncells),
        "n_fronts_detected": n_fronts,                  # disjoint footprint polygons (both paths)
        "seed_radius_m": SEED_RADIUS_M,                 # the ONE buffer radius (half a VIIRS pixel)
        "note": "any calibrated/scenario-fitted parameters are EVENT-SPECIFIC; "
                "do not reuse across events",
    }
    (out_dir / "scenario_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")   # provenance sidecar

    print(f"\nInstance -> {out_dir}")
    print("Next: run Cell2Fire (WSL) with --gen-data, and CRUCIALLY a SHORT fire period:")
    print("  --Fire-Period-Length 1 --gridsStep 60   (1-min physics step, hourly grids).")
    if front_ncells:
        print(f"  + --InitialBurned {out_dir / 'InitialBurned.csv'}  (the seeded footprint)")
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None,
                    help=f"named scenario {sorted(SCENARIOS)}; default when no "
                         f"--points are given: north_evia_2021")
    ap.add_argument("--points", default=None,
                    help='ignition "lat,lon[;lat,lon...]" (1 = point, >=2 = front)')
    ap.add_argument("--window-km", type=float, default=None,
                    help=f"square window side, {WINDOW_KM_MIN:.0f}-{WINDOW_KM_MAX:.0f} km")
    ap.add_argument("--horizon", type=int, default=24,
                    help=f"hours to simulate ({HORIZON_H_MIN}-{HORIZON_H_MAX})")
    ap.add_argument("--start", default=None, help='start "YYYY-MM-DDTHH:MM" UTC')
    ap.add_argument("--out", default=None, help="instance output dir")
    a = ap.parse_args()

    pts = None
    if a.points:                                        # parse "lat,lon;lat,lon" -> list of (lat, lon)
        pts = [tuple(float(x) for x in p.split(","))
               for p in a.points.split(";") if p.strip()]
    scenario = a.scenario if (a.scenario or pts) else "north_evia_2021"   # no input at all -> the 2021 reference
    try:
        build_instance(ignition_points=pts, window_km=a.window_km,
                       horizon_h=a.horizon, start_time=a.start,
                       out_dir=a.out, scenario=scenario)
    except InstanceInputError as e:
        raise SystemExit(f"INPUT ERROR: {e}")           # show the guardrail message cleanly, no traceback


if __name__ == "__main__":
    main()
