"""Phase 4.5: adapter from Cell2Fire outputs -> the WFEDS pipeline.

Cell2Fire (run in WSL) writes, per simulation, a sequence of cumulative burned-cell
grids and a propagation-message file. This module converts those into exactly what
the deterministic pipeline already consumes, WITHOUT changing anything downstream:

  * per-timestep fire PERIMETER as a (Multi)Polygon  (what `fire_timesteps.py` /
    `route_with_fire.blocked_zone` expect), and
  * a per-cell rate-of-spread (ROS) grid -> the basis for the intensity field that
    drives the variable road-block buffer (currently a fuel x slope PLACEHOLDER).

Expected layout of a Cell2Fire run:
    <instance>/Forest.asc                         # georeferencing (.asc header) + shape
    <output>/Grids/Grids<sim>/ForestGrid<NN>.csv  # 0/1 burned grids, cumulative, per period
    <output>/Messages/MessagesFile<sim>.csv       # rows: from_cell,to_cell,period,ROS

The `.asc`/CSV grids carry NO CRS, so we ASSIGN one (EPSG:2100 for the North Evia
study area). Cell ids are 1-based, row-major into the grid.

TIME AXIS (verified 2026-07-01):  `ForestGrid<NN>.csv` index NN IS the fire period.
A grid is written every `--gridsStep` minutes and, with `--gridsStep == --Fire-Period-
Length == 60`, that is exactly once per fire period, so `grid NN == period NN == hour
NN`. Proven by set-equality: the burned-cell SET of `ForestGrid<NN>` equals the
cumulative burned set the `MessagesFile` reports at period NN, for every NN where both
exist. So `cell2fire_perimeters()` (from the ForestGrid dumps) is the CANONICAL
per-timestep (hourly) perimeter source, and the perimeter sequence is a monotonically
growing front - the shape `fire_timesteps.py` expects.

CAVEAT - do NOT derive perimeters from the messages: the `MessagesFile` stops logging
after ~15 periods (its max `period` is NOT the fire duration), so it undercounts the
scar (~0.5 vs ~3 km^2 here). Use it only for per-cell ROS (`ros_grid_from_messages`),
which is a max-per-cell field and does not need completeness. See [[Cell2Fire]],
[[Network and exposure analysis]].

Usage (smoke test on a finished run):
    python scripts/cell2fire/cell2fire_adapter.py <instance_dir> <output_dir> [EPSG:2100]
"""

import csv
import glob
import os
import re
import sys

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import unary_union

BURNED_VALUES = (1,)   # ForestGrid: 1 = burned (0 = unburned/non-burnable)

# Suppressability class thresholds on FREE Byram fireline intensity (kW/m):
# class 1 <350 (direct attack), 2 = 350-1750 (mechanical/aerial), 3 = 1750-3500
# (serious control problems), 4 >3500 (indirect only). The engine emits the raw
# intensity grid (--out-intensity); classification is done HERE (downstream
# diagnostic only - no fire-behaviour effect). Standard fireline-intensity /
# flame-length suppressability classes (Rothermel 1983; Canadian FBP handbook).
SUPP_THRESHOLDS = (350.0, 1750.0, 3500.0)

# Below this a burned polygon part is a raster/spot speck, not a real front - drop
# it (and its interior holes) when building the outer isochrone. Matches the
# dashboard `_clean` min_part.
SPECK_MIN_M2 = 2e4


def read_asc_header(asc_path):
    """Parse an Arc/Info ASCII-grid header -> (Affine transform, nrows, ncols, nodata).

    Handles both `xllcorner`/`yllcorner` and `xllcenter`/`yllcenter`. The raster
    origin is top-left, so the transform's y-resolution is negative and the top
    edge is `yll + nrows*cellsize`.
    """
    hdr = {}
    with open(asc_path) as f:
        for _ in range(6):                                # the 6 fixed .asc header lines
            parts = f.readline().split()
            if len(parts) != 2:
                break
            hdr[parts[0].lower()] = float(parts[1])
    ncols, nrows = int(hdr["ncols"]), int(hdr["nrows"])
    cell = hdr["cellsize"]
    if "xllcorner" in hdr:
        x0, y0 = hdr["xllcorner"], hdr["yllcorner"]        # already the cell CORNER
    else:                                            # *center -> shift to corner
        x0, y0 = hdr["xllcenter"] - cell / 2, hdr["yllcenter"] - cell / 2
    transform = Affine(cell, 0, x0, 0, -cell, y0 + nrows * cell)   # pixel->map; top-left origin, negative y-step
    return transform, nrows, ncols, hdr.get("nodata_value")


def read_grid_csv(path):
    """Read a headerless Cell2Fire ForestGrid CSV -> 2-D float array."""
    return np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=float))


def period_grids(output_dir, sim=1):
    """Sorted [(period, path)] for the ForestGrid<NN>.csv of one simulation."""
    pattern = os.path.join(output_dir, "Grids", f"Grids{sim}", "ForestGrid*.csv")
    found = []
    for path in glob.glob(pattern):
        m = re.search(r"ForestGrid(\d+)\.csv$", os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))         # NN parsed from the filename = the fire period
    return sorted(found)


def perimeter_from_grid(grid, transform, burned_values=BURNED_VALUES):
    """Polygonise the burned cells of one grid -> a CLEAN (Multi)Polygon, or None.

    Diagonally-touching cells are the same fire (connectivity=8), and the union
    gets a LIGHT clean at HALF-CELL resolution: morphological closing glues the
    raster-artifact fragments (a large fire otherwise polygonises into ~1000+
    pieces / ~15k vertices that make every downstream `distance()` crawl - the
    2026-07-02 per-timestep hang, see [[Decision log]]), then `simplify()` drops
    the staircase vertices. Both act below the data's own 1-cell resolution, so
    no real information is lost (area delta <1%; the honest burned area remains
    `n_cells` x cell^2)."""
    mask = np.isin(grid, burned_values)
    if not mask.any():
        return None
    geoms = [shape(geom) for geom, val in features.shapes(                  # raster -> vector polygons
        mask.astype("uint8"), mask=mask, transform=transform, connectivity=8)   # 8-connectivity: diagonal cells join too
        if val == 1]
    if not geoms:
        return None
    c = transform.a / 2.0                             # half a cell, in map units (metres)
    u = unary_union(geoms)                             # dissolve all burned-cell polygons into one fire shape
    return u.buffer(c, join_style=2).buffer(-c, join_style=2).simplify(c)   # morphological CLOSE (+c then -c) + simplify


def cell2fire_perimeters(instance_dir, output_dir, crs="EPSG:2100", sim=1):
    """Per-timestep cumulative fire perimeters from a Cell2Fire run.

    Returns a GeoDataFrame [period, n_cells, geometry] in `crs`. Empty grids (e.g.
    period 0 before ignition spreads) are skipped.
    """
    transform, _nrows, _ncols, _nodata = read_asc_header(
        os.path.join(instance_dir, "Forest.asc"))          # the grid's own georeferencing (no CRS on the raster itself)
    rows = []
    for period, path in period_grids(output_dir, sim):
        grid = read_grid_csv(path)
        poly = perimeter_from_grid(grid, transform)
        if poly is not None:
            rows.append({"period": period,
                         "n_cells": int(np.isin(grid, BURNED_VALUES).sum()),   # the RAW cell count (ground truth area)
                         "geometry": poly})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)   # CRS assigned here (the raster carried none)


def intensity_grids(output_dir, sim=1):
    """Sorted [(period, path)] for the Intensity<NN>.csv of one simulation.

    Written by the WFEDS suppression engine patch (`--out-intensity`): same folder
    and numbering as the ForestGrid dumps, so period NN aligns 1:1. Cell value =
    max FREE Byram head fireline intensity (kW/m) while the cell was actively
    burning up to period NN; 0 = never burned / never active. Empty list = the run
    was made without the flag (old runs keep working)."""
    pattern = os.path.join(output_dir, "Grids", f"Grids{sim}", "Intensity*.csv")
    found = []
    for path in glob.glob(pattern):
        m = re.search(r"Intensity(\d+)\.csv$", os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def _segment_classes(boundary, fi, transform):
    """Suppressability class for each ~cell-length segment of a perimeter boundary.

    The boundary sits between burned and unburned cells, so the class at each
    segment midpoint is sampled as the MAX free FI over the 3x3 cell window around
    it (catches the burned side). Returns [(coords, classes)] per ring: `coords`
    the densified vertex array, `classes[i]` the class of segment coords[i]->
    coords[i+1] (0 = no burning cell nearby, e.g. artifacts)."""
    import shapely
    dense = shapely.segmentize(boundary, max_segment_length=transform.a)   # add a vertex every ~1 cell along the line
    lines = getattr(dense, "geoms", [dense])           # MultiLineString -> its parts, or a single LineString as-is
    nrows, ncols = fi.shape
    inv = ~transform                                   # map coords -> pixel (row, col), the inverse affine
    out = []
    for line in lines:
        coords = np.asarray(line.coords)
        if len(coords) < 2:
            continue
        mid = (coords[:-1] + coords[1:]) / 2.0         # one sample point per segment (its midpoint)
        fc, fr = inv * (mid[:, 0], mid[:, 1])            # fractional (col, row)
        cc, rr = fc.astype(int), fr.astype(int)
        best = np.zeros(len(mid))
        for dr in (-1, 0, 1):                            # 3x3 max around midpoint
            for dc in (-1, 0, 1):
                r2 = np.clip(rr + dr, 0, nrows - 1)      # clamp so edge segments don't index out of bounds
                c2 = np.clip(cc + dc, 0, ncols - 1)
                best = np.maximum(best, fi[r2, c2])
        classes = np.where(best > 0, np.digitize(best, SUPP_THRESHOLDS) + 1, 0)   # intensity -> suppressability class 1-4
        out.append((coords, classes))
    return out


def _outer_front(geom):
    """The clean OUTER advancing front of one period's perimeter.

    Keeps only polygon parts above a speck threshold and drops their interior
    holes (unburned islands are NOT the front - they inflate line-work and, being
    ringed by hot burned cells, misclass as class 4). A fire that is ENTIRELY
    below the speck threshold (barely spread) is still a real front - keep its
    largest part instead of dropping everything. Returns the boundary of the
    hole-filled outer exterior(s) as a (Multi)LineString, or None."""
    from shapely.geometry import Polygon

    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    parts = [Polygon(p.exterior) for p in polys if p.area >= SPECK_MIN_M2]   # exterior ring only -> holes dropped
    if not parts:
        biggest = max(polys, key=lambda p: p.area, default=None)   # nothing survives the threshold -> keep the largest piece
        if biggest is None or biggest.is_empty:
            return None
        parts = [Polygon(biggest.exterior)]
    return unary_union(parts).boundary                # polygon(s) -> just their boundary LINE (the front itself)


def isochrone_fronts(perims, output_dir, transform, sim=1):
    """Per-hour OUTER fire front (isochrone) + its suppressability breakdown.

    For each period: the clean hole-filled outer front (`_outer_front`), and - if
    an Intensity grid exists - the class shares (`pct1..pct4`, `dom` = dominant
    class) sampled along THAT exterior only (so holes no longer bias the numbers).
    Returns rows [period, geometry(MultiLineString), pct1..pct4, dom]; periods with
    no burned area are skipped. Runs made without `--out-intensity` still get the
    geometry (pct all 0, dom 0)."""
    fi_by_period = dict(intensity_grids(output_dir, sim))
    rows = []
    for _, p in perims.iterrows():
        front = _outer_front(p.geometry)
        if front is None or front.is_empty:
            continue
        pct = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        path = fi_by_period.get(p["period"])
        if path is not None:
            fi = read_grid_csv(path)
            length = dict.fromkeys((1, 2, 3, 4), 0.0)
            for coords, classes in _segment_classes(front, fi, transform):
                seg_len = np.hypot(*(coords[1:] - coords[:-1]).T)   # per-segment length (map units)
                for cls in (1, 2, 3, 4):
                    length[cls] += float(seg_len[classes == cls].sum())   # length-weighted, not segment-count-weighted
            total = sum(length.values())
            if total:
                pct = {c: 100.0 * length[c] / total for c in length}   # class share of the TOTAL front length
        dom = max(pct, key=pct.get) if any(pct.values()) else 0   # the dominant (most-common-by-length) class
        rows.append({"period": int(p["period"]), "geometry": front,
                     "pct1": pct[1], "pct2": pct[2], "pct3": pct[3], "pct4": pct[4],
                     "dom": dom})
    if not rows:      # fire never produced a front (e.g. did not spread at all)
        return gpd.GeoDataFrame(
            {"period": [], "pct1": [], "pct2": [], "pct3": [], "pct4": [],
             "dom": []}, geometry=gpd.GeoSeries([], crs=perims.crs))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=perims.crs)


def ros_grid_from_messages(messages_csv, nrows, ncols):
    """Per-cell ROS grid (m/min) from a MessagesFile (rows: from,to,period,ROS).

    The recorded ROS is the spread rate of the edge that REACHED the target cell;
    we keep the max per target. Cell ids are 1-based, row-major. Cells never
    reached stay 0. This is the raw field from which a fireline-intensity proxy
    (Byram, using fuel load) is later built to drive the buffer width.
    """
    ros = np.zeros((nrows, ncols), dtype=float)
    with open(messages_csv) as f:
        for r in csv.reader(f):
            if len(r) < 4:
                continue
            try:
                target, value = int(r[1]), float(r[3])
            except ValueError:
                continue
            rr, cc = divmod(target - 1, ncols)          # 1-based row-major id -> (row, col), 0-based
            if 0 <= rr < nrows and 0 <= cc < ncols:
                ros[rr, cc] = max(ros[rr, cc], value)     # keep the FASTEST edge that ever reached this cell
    return ros


def find_messages(output_dir, sim=1):
    """Path to MessagesFile<sim>.csv, or None."""
    hits = glob.glob(os.path.join(output_dir, "Messages", f"MessagesFile{sim:02d}.csv"))
    hits = hits or glob.glob(os.path.join(output_dir, "Messages", "MessagesFile*.csv"))
    return sorted(hits)[0] if hits else None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("error: need <instance_dir> <output_dir>")
        return
    instance_dir, output_dir = sys.argv[1], sys.argv[2]
    crs = sys.argv[3] if len(sys.argv) > 3 else None    # e.g. EPSG:2100 - the raster itself carries no CRS

    transform, nrows, ncols, _ = read_asc_header(os.path.join(instance_dir, "Forest.asc"))
    print(f"Instance grid: {nrows} x {ncols}, cell {transform.a:.1f}, crs={crs}")

    # Canonical per-timestep perimeters = the ForestGrid dumps: grid NN = fire
    # period NN = hour NN (gridsStep == Fire-Period-Length == 60 min). See the
    # module docstring for why the messages are NOT used for perimeters.
    perims = cell2fire_perimeters(instance_dir, output_dir, crs=crs)
    print(f"\nPer-timestep perimeters (grid NN = fire period NN = hour NN): "
          f"{len(perims)} periods")
    cell_area = transform.a * transform.a
    for _, r in perims.iterrows():
        raw = r["n_cells"] * cell_area                  # the raster's true area
        dpct = 100.0 * (r.geometry.area - raw) / raw    # how much the morphological clean shifted the polygon area
        print(f"   period {r['period']:>2}  burned cells {r['n_cells']:>5}  "
              f"area {r.geometry.area:,.0f} m^2 (clean delta {dpct:+.2f}%)")

    msg = find_messages(output_dir)
    if msg:
        ros = ros_grid_from_messages(msg, nrows, ncols)
        hit = ros > 0
        print(f"\nROS grid from {os.path.basename(msg)}: {int(hit.sum())} reached cells, "
              f"ROS min/mean/max = {ros[hit].min():.2f} / {ros[hit].mean():.2f} / "
              f"{ros.max():.2f} m/min")
    else:
        print("\n(no MessagesFile found - skipping ROS grid)")

    if len(perims):
        out = os.path.join(output_dir, "perimeters.geojson")
        perims.to_file(out, driver="GeoJSON")
        print(f"\nWrote per-timestep perimeters -> {out}")

    # Per-hour OUTER fronts (isochrones) + suppressability breakdown on the
    # exterior only (holes excluded). WFEDS suppression patch (`--out-intensity`);
    # old runs without Intensity grids still get the geometry (classes all 0).
    iso = isochrone_fronts(perims, output_dir, transform)
    if len(iso):
        out = os.path.join(output_dir, "isochrones.geojson")
        iso.to_file(out, driver="GeoJSON")
        print(f"\nWrote per-hour outer fronts (isochrones) -> {out}")
        print("   exterior-only suppressability class-4 % per period "
              "(vs the old hole-inflated ~96%):")
        for r in iso.itertuples():
            print(f"   period {r.period:>2}  class-4 {r.pct4:>4.0f}%  "
                  f"(dominant class {r.dom})")


if __name__ == "__main__":
    main()
