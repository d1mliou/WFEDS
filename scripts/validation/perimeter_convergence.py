"""Perimeter-to-perimeter geometric convergence and rate diagnostic (axis 1,
part 1b - see the vault's Validation note).

The question: at which simulated hour does the free-burning perimeter come
GEOMETRICALLY CLOSEST to the final satellite-mapped burned-area perimeter,
and where along the boundary does the simulation over-extend or lag? This is
a geometric convergence and rate diagnostic, NEVER an accuracy score: the
reference scar spans ~6 days and active suppression, while the run is a 24 h
free-burning worst case.

Protocol (all choices pre-declared, not tuned to the outcome):
  * Step 0 window check: the final scar is intersected with the model window
    (from instance/Forest.asc) ONLY to report how much of it lies outside the
    reachable domain. The scar polygon itself is never clipped.
  * Holes (unburned islands) are filled in BOTH datasets - out of scope.
  * Points are sampled every SAMPLE_SPACING_M along each source boundary
    ring; distances are measured to the CONTINUOUS boundary line of the other
    dataset, never to its sampled points.
  * Exclusions (counted and reported): real-boundary points outside the
    window or within FRAME_TOL_M of the window frame; simulated-boundary
    points within FRAME_TOL_M of the frame (a domain edge, not a fire front)
    - excluded as source points AND stripped from the real->sim target line.
  * Per direction per hour: mean / median / p95 / max (the directed
    Hausdorff), plus the symmetric Hausdorff and HD95. Primary indicator
    D_t = (mean sim->real + mean real->sim) / 2.
  * t* = argmin D_t = the HOUR OF CLOSEST GEOMETRIC APPROACH (ties: earliest
    hour, declared). If t* falls on the first or last evaluated hour the
    result is CENSORED: convergence was not reached within the horizon.
  * Stability: the argmin of every alternative indicator is reported next to
    t* as a result in its own right; alternatives can NEVER override t*.
  * Signed distances at t*: positive ALWAYS means the simulation went beyond
    reality (sim point outside the scar / real point inside the sim).
  * Zonation: maximal contiguous same-sign runs along each ring with
    |signed d| > ZONE_THRESHOLD_M (strict); runs of >= ZONE_MIN_POINTS
    points are zones, shorter ones isolated extremes. Rings wrap around at
    closure; an excluded point breaks contiguity. NO spatial statistics
    (no Moran's I, no LISA, no Gi*): on a boundary-distance profile sampled
    every 30 m a permutation null is geometrically impossible, so any
    significance test is foreordained - descriptive zonation replaces it.
  * Rate statement: the first hour the simulated area exceeds the final scar
    area. Areas here are the RAW (unfilled) burned areas - hole filling
    applies to the boundary comparison only - and the crossing is decided on
    UNROUNDED values.

t* is the hour of closest geometric approach of a free-burning run to a
suppressed multi-day scar; it is not the hour at which the model was correct.
The sentence is carried by the report and the metrics JSON (the two narrative
outputs); CSV/PNG/GeoJSON are data files and do not repeat it.

Scope: the reference scar is PRE-DECLARED and fixed to the 2021 North Evia
case (SCAR_SHP below). The run directory is selectable, but this diagnostic
is case-study-specific by design - pointing it at a run outside that case
would compare against the wrong scar.

Run (a finished run directory containing perimeters.geojson + instance/):
    python scripts/validation/perimeter_convergence.py <run_dir>
Default run_dir: the newest under DATA_DIR/Fire/cell2fire/runs/ with
perimeters.geojson ("newest" = greatest directory name; run ids sort
lexicographically). Chart/GIS/overlay generation is wrapped in try/except
like the 1a validation, so a plotting failure never aborts the analysis -
the console prints what was skipped.

Outputs (into <run_dir>): perimeter_metrics.json, perimeter_hourly.csv,
perimeter_zones.csv, perimeter_report.md, perimeter_dt_curve.png,
perimeter_signed_map.png, perimeter_overlay.html (optional, folium), and a
QGIS-ready bundle perimeter_gis/ (EPSG:2100 GeoJSON, every file prefixed
perimeter_ so the flat Exports copy cannot collide with the 1a bundle). The
whole set is also published to DATA_DIR/Exports/<run_id>/validation/ - the
same deliverable convention as the 1a VIIRS validation.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Lives in scripts/validation/ (its own evaluation-axis package); the pipeline
# imports below (_paths, cell2fire_adapter, route_shortest_path) live in the
# SIBLING package scripts/cell2fire/.
sys.path.insert(0, str(Path(__file__).parent.parent / "cell2fire"))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from _paths import DATA_DIR

# --- pre-declared protocol constants (do NOT tune these to the outcome) --------
SAMPLE_SPACING_M = 30.0       # boundary sampling step = one model cell
FRAME_TOL_M = 30.0            # "coincident with the window frame" = one cell
ZONE_THRESHOLD_M = 60.0       # |signed d| that counts as divergence = two cells
ZONE_MIN_POINTS = 5           # minimum contiguous run for a zone. Length convention:
                              # length_m = n_points x 30 m (each sample REPRESENTS one
                              # cell of boundary); the connecting polyline arc spans
                              # n_points-1 intervals, i.e. 5 points = 150 m represented,
                              # 120 m of arc. Declared, not an off-by-one.
SEED_MIN_CELLS = 1000         # period-0 bigger than this => front-seeded run (context)

RFD = DATA_DIR / "Real_Fire_Data"
RUNS_DIR = DATA_DIR / "Fire" / "cell2fire" / "runs"
SCAR_SHP = RFD / "Burned_Area" / "Burned_area_20210829.shp"

CENSORED_STATEMENT = "convergence not reached within the simulated horizon"
FIXED_SENTENCE = ("t* is the hour of closest geometric approach of a "
                  "free-burning run to a suppressed multi-day scar; it is not "
                  "the hour at which the model was correct.")


# --------------------------------------------------------------------------------
# pure helpers (unit-tested; no I/O)
# --------------------------------------------------------------------------------
def fill_holes(geom):
    """(Multi)Polygon with every interior ring (unburned island) removed.
    Filling a hole can merge parts that surrounded each other, so the filled
    parts are unioned. Empty/None input is returned unchanged."""
    if geom is None or geom.is_empty:
        return geom
    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return unary_union([Polygon(p.exterior) for p in parts if not p.is_empty])


def window_box_from_header(transform, nrows, ncols):
    """Model-window polygon from an .asc header's affine (top-left origin,
    negative y-step) - the exact bounds recipe the 1a overlay map uses."""
    return box(transform.c, transform.f + nrows * transform.e,
               transform.c + ncols * transform.a, transform.f)


def boundary_rings(filled):
    """Exterior rings of each polygon part as closed LineStrings (ring id =
    list index). Interiors are gone after fill_holes; ignored defensively."""
    if filled is None or filled.is_empty:
        return []
    parts = filled.geoms if filled.geom_type == "MultiPolygon" else [filled]
    return [LineString(p.exterior.coords) for p in parts if not p.is_empty]


def sample_ring(ring, spacing=SAMPLE_SPACING_M):
    """Points every `spacing` m along one closed ring, ordered by chainage,
    WITHOUT the duplicate closing point (np.arange stops before ring.length).
    A ring shorter than `spacing` still yields its start point.
    Returns (list of Points, chainage ndarray)."""
    chain = np.arange(0.0, ring.length, spacing)
    return [ring.interpolate(c) for c in chain], chain


def sample_boundary(filled, spacing=SAMPLE_SPACING_M):
    """30 m sample points over every exterior ring of a (Multi)Polygon.
    Columns: ring_id, idx (position within the ring), n_ring (points on that
    ring - needed for wrap-around), chainage_m, geometry. EPSG:2100."""
    rows = {"ring_id": [], "idx": [], "n_ring": [], "chainage_m": []}
    geoms = []
    for rid, ring in enumerate(boundary_rings(filled)):
        pts, chain = sample_ring(ring, spacing)
        n = len(pts)
        rows["ring_id"] += [rid] * n
        rows["idx"] += list(range(n))
        rows["n_ring"] += [n] * n
        rows["chainage_m"] += [float(c) for c in chain]
        geoms += pts
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=2100)


def exclusion_masks_real(pts, window, tol=FRAME_TOL_M):
    """Real-boundary exclusions: (outside_window, near_frame) boolean Series.
    The two partition the excluded set (near_frame only counts in-window
    points) so the reported counts never double-count a point."""
    outside = ~pts.geometry.within(window)
    near = pts.geometry.distance(window.exterior) <= tol
    return outside, near & ~outside


def exclusion_mask_sim(pts, window, tol=FRAME_TOL_M):
    """Sim-boundary samples sitting on the window frame: there the 'perimeter'
    is the clipped domain edge, not a simulated fire front."""
    return pts.geometry.distance(window.exterior) <= tol


def strip_frame_from_line(line, window, tol=FRAME_TOL_M):
    """The real->sim distance target with frame-coincident segments removed
    (same physical reason as exclusion_mask_sim, applied to the TARGET side)."""
    return line.difference(window.exterior.buffer(tol))


def _segment_tree(line):
    """STRtree over the individual 2-point SEGMENTS of a (Multi)LineString.
    Exact nearest-distance to the continuous line via tree traversal - plain
    `distance()` against the whole scar boundary is O(all segments) per point
    and took the first golden-run attempt past 10 minutes."""
    parts = line.geoms if hasattr(line, "geoms") else [line]
    segs = []
    for p in parts:
        if p.is_empty or p.geom_type not in ("LineString", "LinearRing"):
            continue
        c = np.asarray(p.coords)
        if len(c) >= 2:
            segs.append(shapely.linestrings(
                np.stack([c[:-1], c[1:]], axis=1)))
    if not segs:
        return None
    return shapely.STRtree(np.concatenate(segs))


def directed_distances(points, target_line):
    """Distances (m) from each sample point to the CONTINUOUS target boundary
    line - never to its sampled points. None for a degenerate target.
    Implemented as an exact nearest-segment query on an STRtree (identical
    values to point.distance(line), just indexed)."""
    if target_line is None or target_line.is_empty or len(points) == 0:
        return None
    tree = _segment_tree(target_line)
    if tree is None:
        return None
    geoms = np.asarray(points.values)
    idx, dist = tree.query_nearest(geoms, return_distance=True,
                                   all_matches=False)
    out = np.empty(len(geoms), dtype=float)
    out[idx[0]] = dist
    return out


def direction_stats(d):
    """mean / median / p95 / max (m) of one directed distance set. max IS the
    directed Hausdorff distance on the declared 30 m sampling."""
    if d is None or len(d) == 0:
        return None
    s = pd.Series(d, dtype=float)
    return {"mean_m": round(float(s.mean()), 1),
            "median_m": round(float(s.median()), 1),
            "p95_m": round(float(s.quantile(0.95)), 1),
            "max_m": round(float(s.max()), 1)}


def hour_indicators(s2r, r2s):
    """D_t (primary), symmetric Hausdorff and HD95 from the two directed stat
    dicts. Any missing direction -> all None (degenerate hour)."""
    if not s2r or not r2s:
        return {"d_t_m": None, "hausdorff_sym_m": None, "hd95_m": None}
    return {"d_t_m": round((s2r["mean_m"] + r2s["mean_m"]) / 2, 1),
            "hausdorff_sym_m": max(s2r["max_m"], r2s["max_m"]),
            "hd95_m": max(s2r["p95_m"], r2s["p95_m"])}


def find_t_star(hourly):
    """argmin of D_t over the evaluated hours. Ties -> earliest hour
    (declared; the full tie list is reported). CENSORED when the argmin sits
    on the first or last evaluated hour: the curve may still be falling (or
    already rising) there, so the minimum is not an interior convergence
    point. Returns None if no hour has a computable D_t."""
    ok = hourly.dropna(subset=["d_t_m"])
    if ok.empty:
        return None
    dmin = ok["d_t_m"].min()
    ties = sorted(int(h) for h in ok.loc[ok["d_t_m"] == dmin, "hour"])
    t_star = ties[0]
    censored = t_star in (int(ok["hour"].min()), int(ok["hour"].max()))
    return {"hour": t_star, "d_t_m": float(dmin), "censored": censored,
            "ties": ties[1:],
            "statement": (CENSORED_STATEMENT if censored
                          else "hour of closest geometric approach")}


STABILITY_COLS = ("mean_s2r_m", "mean_r2s_m", "median_s2r_m", "median_r2s_m",
                  "p95_s2r_m", "p95_r2s_m", "max_s2r_m", "max_r2s_m",
                  "hd95_m", "hausdorff_sym_m")


def stability_table(hourly, t_star_hour):
    """argmin hour of every alternative indicator, reported next to t* as a
    result in its own right. Alternatives can NEVER override t* (D_t is the
    sole pre-declared selector); agreement or disagreement is the finding."""
    out = []
    for col in STABILITY_COLS:
        if col not in hourly.columns:
            continue
        ok = hourly.dropna(subset=[col])
        if ok.empty:
            continue
        m = ok[col].min()
        h = int(ok.loc[ok[col] == m, "hour"].min())
        out.append({"indicator": col, "argmin_hour": h,
                    "min_value_m": float(m),
                    "agrees_with_t_star": h == t_star_hour})
    return out


def apply_sign(d, inside, positive_when_inside):
    """Signed distances. Convention (identical meaning in both directions):
    POSITIVE always = the simulation went beyond reality.
      sim->real: positive when the sim point is OUTSIDE the filled scar
                 -> positive_when_inside=False.
      real->sim: positive when the real point is INSIDE the filled sim
                 -> positive_when_inside=True.
    A point exactly on the other boundary has d = 0, so its sign is moot."""
    d = np.asarray(d, dtype=float)
    inside = np.asarray(inside, dtype=bool)
    return np.where(inside == positive_when_inside, 1.0, -1.0) * d


def extract_runs(signed, valid=None, threshold=ZONE_THRESHOLD_M,
                 min_points=ZONE_MIN_POINTS):
    """Maximal contiguous same-sign runs along ONE closed ring.

    signed: signed distances in ring order (NaN allowed - treated as class 0).
    valid: usable-point mask; an excluded point breaks contiguity (a zone can
    never lean on points the protocol excluded). Class per point: +1 if
    signed > threshold, -1 if signed < -threshold (STRICT inequalities),
    else 0. The ring wraps around: a same-class run crossing the closure is
    ONE run. Runs with >= min_points points are zones, shorter nonzero runs
    are isolated extremes. Returns (zones, extremes) as dicts with
    start_idx / n_points / sign / mean_abs_m / max_abs_m."""
    signed = np.asarray(signed, dtype=float)
    n = len(signed)
    if n == 0:
        return [], []
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    with np.errstate(invalid="ignore"):                # NaN compares -> False
        cls = np.zeros(n, int)
        cls[valid & (signed > threshold)] = 1
        cls[valid & (signed < -threshold)] = -1

    runs_idx = []                                       # [start, length, class]
    i = 0
    while i < n:
        c = cls[i]
        j = i
        while j < n and cls[j] == c:
            j += 1
        runs_idx.append([i, j - i, c])
        i = j
    # wrap-around: first and last run of the same class join across the closure
    if len(runs_idx) > 1 and runs_idx[0][2] == runs_idx[-1][2]:
        s, ln, c = runs_idx.pop()
        runs_idx[0] = [s, ln + runs_idx[0][1], c]

    zones, extremes = [], []
    absd = np.abs(signed)
    for s, ln, c in runs_idx:
        if c == 0:
            continue
        seg = absd[np.arange(s, s + ln) % n]
        run = {"start_idx": int(s), "n_points": int(ln), "sign": int(c),
               "mean_abs_m": round(float(seg.mean()), 1),
               "max_abs_m": round(float(seg.max()), 1)}
        (zones if ln >= min_points else extremes).append(run)
    return zones, extremes


def zone_table(samples):
    """Zones + isolated extremes over every (direction, ring) of the t*
    sample frame (columns: direction, ring_id, idx, signed_m, excluded).
    Returns the full zone DataFrame - every run is reported (rule: report
    everything computed)."""
    rows = []
    zid = 0
    for (direction, rid), grp in samples.groupby(["direction", "ring_id"]):
        grp = grp.sort_values("idx")
        zones, extremes = extract_runs(grp["signed_m"].to_numpy(),
                                       valid=(~grp["excluded"]).to_numpy())
        for run, kind in ([(z, "zone") for z in zones]
                          + [(e, "isolated_extreme") for e in extremes]):
            zid += 1
            rows.append({"zone_id": zid, "direction": direction,
                         "ring_id": int(rid), "kind": kind,
                         "sign": run["sign"],
                         "meaning": ("over-extension" if run["sign"] > 0
                                     else "lag"),
                         "n_points": run["n_points"],
                         "length_m": round(run["n_points"] * SAMPLE_SPACING_M),
                         "start_idx": run["start_idx"],
                         "mean_abs_m": run["mean_abs_m"],
                         "max_abs_m": run["max_abs_m"]})
    return pd.DataFrame(rows)


def first_hour_area_exceeds(areas_km2, scar_km2):
    """Rate-statement input: the first hour (dict/Series key order) whose
    simulated area exceeds the final scar area, or None if never."""
    for h, a in areas_km2.items():
        if a > scar_km2:
            return int(h)
    return None


# --------------------------------------------------------------------------------
# run-directory resolution + I/O
# --------------------------------------------------------------------------------
def resolve_run_dir(arg=None):
    """The run to diagnose: explicit CLI path, else the newest run under
    RUNS_DIR that has perimeters.geojson."""
    if arg:
        rd = Path(arg)
        if not (rd / "perimeters.geojson").exists():
            sys.exit(f"ERROR: {rd} has no perimeters.geojson - not a finished run dir")
        return rd
    if RUNS_DIR.exists():
        candidates = sorted((d for d in RUNS_DIR.iterdir()
                             if (d / "perimeters.geojson").exists()),
                            key=lambda d: d.name)
        if candidates:
            return candidates[-1]
    sys.exit("ERROR: no run dir given and no finished runs found under "
             f"{RUNS_DIR} - run e.g. `python scripts/cell2fire/run_scenario.py "
             "--scenario north_evia_2021` first")


# --------------------------------------------------------------------------------
# report pieces
# --------------------------------------------------------------------------------
LIMITATIONS = """\
* The reference is the FINAL burned scar (~6 days, actively suppressed); the
  run is a 24 h FREE-BURNING simulation. Over-extension zones are the
  expected signature of unmodelled suppression, and lag zones may simply be
  sectors the real fire reached only after the simulated horizon. Neither is
  scored as error - the vocabulary is over-extension / lag, never accuracy.
* This diagnostic uses ONLY the final scar polygon. No VIIRS points enter the
  main analysis (kept independent of the 1a per-pass validation by design).
* Boundaries are raster-derived (30 m cells, staircase edges, simplified at
  half a cell); distance differences of one to two cells are within
  discretization noise. This is stated as context, not applied as a formal
  plateau rule.
* Any part of the scar outside the model window could never be reached by the
  simulation; affected boundary samples are excluded and counted, and the
  scar-outside-window share is reported up front (step 0).
* No spatial statistics (Moran's I / LISA / Gi*) are computed: on a
  boundary-distance profile sampled every 30 m, neighbouring values cannot
  differ by more than the spacing while the values span hundreds of metres,
  so a permutation null is geometrically impossible and any significance test
  is foreordained. Descriptive same-sign zonation replaces it.
* Interior unburned islands (holes) are filled in both datasets and are out
  of this diagnostic's scope.
* Single-event case study; placeholder SVM->FBP fuel map; ERA5/Open-Meteo
  reanalysis weather. The 1a report's limitations apply unchanged.
"""


def md_table(df):
    """Plain markdown table (pandas' own .to_markdown needs the optional
    `tabulate` package - not worth a dependency for one table)."""
    if df.empty:
        return "(none)"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def write_report(run_dir, meta, hourly, ts, stab, signed_summary, zones,
                 rate):
    """perimeter_report.md - the thesis-facing writeup: declared protocol,
    step-0 window check, hourly table, t* + stability, signed distances,
    zones, rate statement, interpretation, limitations."""
    hour_cols = ["hour", "sim_km2", "mean_s2r_m", "mean_r2s_m", "d_t_m",
                 "hd95_m", "n_sim_excluded_frame", "n_real_used"]
    stab_df = pd.DataFrame(stab)
    zshow = zones.drop(columns=["start_idx"]) if not zones.empty else zones
    lines = [
        "# Perimeter-to-perimeter geometric convergence and rate diagnostic",
        "",
        f"Run: `{meta['run_id']}` - {meta['run_kind']}. Horizon "
        f"{meta['horizon']} h. Reference: final burned scar "
        f"`{SCAR_SHP.name}` ({meta['scar_km2']} km2, ~6 days, actively "
        "suppressed). EPSG:2100.",
        "",
        f"**{FIXED_SENTENCE}**",
        "",
        "## Declared protocol",
        f"* Boundary sampling every **{int(SAMPLE_SPACING_M)} m** per ring; "
        "distances measured to the CONTINUOUS boundary line of the other "
        "dataset, never to its sampled points.",
        "* Holes (unburned islands) filled in BOTH datasets - out of scope.",
        f"* Exclusions: real points outside the window or within "
        f"{int(FRAME_TOL_M)} m of the frame; sim points within "
        f"{int(FRAME_TOL_M)} m of the frame excluded as source AND stripped "
        "from the real->sim target (a domain edge is not a fire front). All "
        "counts reported.",
        "* Primary indicator D_t = (mean sim->real + mean real->sim) / 2; "
        "t* = argmin D_t (ties: earliest hour). Censoring: t* on the first "
        "or last evaluated hour = convergence not reached.",
        "* Stability of t* under every alternative indicator reported as a "
        "result; alternatives never override t*.",
        "* Sign convention: positive ALWAYS = the simulation went beyond "
        "reality.",
        f"* Zonation: contiguous same-sign runs with |signed d| > "
        f"{int(ZONE_THRESHOLD_M)} m (two cells, strict); >= "
        f"{ZONE_MIN_POINTS} points (>= {int(ZONE_MIN_POINTS * SAMPLE_SPACING_M)} m) "
        "= a zone, shorter = an isolated extreme. Wrap-around at ring "
        "closure; an excluded point breaks contiguity. Length convention: "
        "length_m = n_points x 30 m (each sample represents one cell; the "
        "connecting arc spans n_points-1 intervals).",
        "* No Moran's I / LISA / Gi* and no overlap metrics (IoU, precision, "
        "recall) by pre-declared decision - see limitations.",
        "",
        "## Step 0 - window check",
        "",
        f"* Model window: {meta['window_km']} km (from the run's own "
        "instance grid). The scar polygon is NOT clipped.",
        f"* Scar outside the window: **{meta['scar_outside_km2']} km2 "
        f"({meta['scar_outside_pct']}%)**.",
        f"* Real-boundary samples: {meta['n_real_pts']} total, "
        f"{meta['n_real_outside']} outside the window, "
        f"{meta['n_real_near_frame']} within {int(FRAME_TOL_M)} m of the "
        f"frame -> {meta['n_real_used']} used.",
        "",
        "## Hourly convergence",
        "",
        md_table(hourly[hour_cols]),
        "",
        "(Full indicator set per hour: perimeter_hourly.csv.)",
        "",
        "## Hour of closest geometric approach",
        "",
    ]
    if ts is None:
        lines.append("* No hour had a computable D_t (degenerate run).")
    elif ts["censored"]:
        lines += [
            f"* D_t is minimised at the boundary of the horizon (hour "
            f"+{ts['hour']}, D_t {ts['d_t_m']} m): **{CENSORED_STATEMENT}**. "
            "Only the shape of the two directed curves is interpreted.",
        ]
    else:
        lines += [
            f"* **t\\* = +{ts['hour']} h** (D_t {ts['d_t_m']} m)."
            + (f" Ties: {ts['ties']} (earliest declared winner)."
               if ts["ties"] else ""),
            f"* {FIXED_SENTENCE}",
        ]
    lines += [
        "",
        "## Stability of the argmin (alternatives never override t*)",
        "",
        md_table(stab_df),
        "",
        "## Signed distances at t* (positive = sim beyond reality)",
        "",
    ]
    for d in ("sim_to_real", "real_to_sim"):
        s = signed_summary.get(d)
        if s:
            lines.append(
                f"* {d}: {s['n_points']} points, {s['share_positive_pct']}% "
                f"positive, mean signed {s['mean_signed_m']} m, median "
                f"signed {s['median_signed_m']} m.")
    lines += [
        "",
        "## Divergence zones at t*",
        "",
        f"* {meta['n_zones']} zones and {meta['n_extremes']} isolated "
        "extremes (full table: perimeter_zones.csv).",
        "",
        md_table(zshow),
        "",
        "## Rate statement",
        "",
        (f"* The simulated area first exceeds the final scar area "
         f"({meta['scar_km2']} km2) at hour **+{rate}** of a ~6-day real "
         "event - a statement about the free-burning rate, not an accuracy "
         "score." if rate is not None else
         "* The simulated area never exceeds the final scar area within the "
         "horizon."),
        "",
        "## Interpretation",
        "",
        meta["interpretation"],
        "",
        "## Limitations",
        "",
        LIMITATIONS,
    ]
    out = run_dir / "perimeter_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report -> {out}")


# --------------------------------------------------------------------------------
# charts + GIS + publish
# --------------------------------------------------------------------------------
def _plot_geom_lines(ax, geom, **kw):
    """Plot every LineString of a (Multi)LineString / boundary on ax."""
    if geom is None or geom.is_empty:
        return
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in parts:
        if p.is_empty or p.geom_type not in ("LineString", "LinearRing"):
            continue
        x, y = p.xy
        ax.plot(x, y, **kw)
        kw.pop("label", None)                    # only the first part labels


def make_charts(run_dir, hourly, ts, samples, scar_boundary, sim_boundary,
                window):
    """perimeter_dt_curve.png (D_t + both directed means, t* marked) and
    perimeter_signed_map.png (signed samples on a diverging scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hourly["hour"], hourly["mean_s2r_m"], color="#d62728", lw=1.2,
            label="sim -> real (mean)")
    ax.plot(hourly["hour"], hourly["mean_r2s_m"], color="#1f77b4", lw=1.2,
            label="real -> sim (mean)")
    ax.plot(hourly["hour"], hourly["d_t_m"], color="black", lw=2,
            label="D_t (primary)")
    if ts is not None:
        ax.axvline(ts["hour"], color="gray", ls="--", lw=1)
        tag = (f"censored (+{ts['hour']} h)" if ts["censored"]
               else f"t* = +{ts['hour']} h")
        ax.annotate(tag, (ts["hour"], ts["d_t_m"]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.set_xlabel("simulated hour")
    ax.set_ylabel("boundary distance (m)")
    ax.set_title("Directed mean boundary distances and D_t per hour")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "perimeter_dt_curve.png", dpi=150)
    plt.close(fig)

    if samples is None or len(samples) == 0:
        print(f"Saved chart -> {run_dir / 'perimeter_dt_curve.png'} "
              "(signed map skipped: no t* samples)")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    _plot_geom_lines(ax, scar_boundary, color="#555", lw=1,
                     label="final scar boundary")
    _plot_geom_lines(ax, sim_boundary, color="black", lw=1,
                     label="sim perimeter at t*")
    _plot_geom_lines(ax, window.exterior, color="#333", lw=1, ls="--",
                     label="model window")
    kept = samples[~samples["excluded"]]
    vmax = float(np.nanmax(np.abs(kept["signed_m"]))) if len(kept) else 0.0
    if vmax > 0:
        sc = ax.scatter(kept.geometry.x, kept.geometry.y, c=kept["signed_m"],
                        cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
                        s=2, zorder=3)
        fig.colorbar(sc, ax=ax, shrink=0.7,
                     label="signed distance (m; + = sim beyond reality)")
    ax.set_aspect("equal")
    ax.set_title("Signed boundary distances at t*")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(run_dir / "perimeter_signed_map.png", dpi=150)
    plt.close(fig)
    print(f"Saved charts -> {run_dir / 'perimeter_dt_curve.png'}, "
          f"{run_dir / 'perimeter_signed_map.png'}")


def zone_geometries(zones, samples):
    """Rebuild each zone/extreme as a geometry from its ordered sample points
    (a polyline through 30 m samples tracks the boundary arc; a single-point
    run stays a Point). Returns a GeoDataFrame aligned with `zones`."""
    geoms = []
    for _, z in zones.iterrows():
        grp = samples[(samples["direction"] == z["direction"])
                      & (samples["ring_id"] == z["ring_id"])].sort_values("idx")
        n_ring = int(grp["n_ring"].iloc[0])
        idxs = [(int(z["start_idx"]) + k) % n_ring
                for k in range(int(z["n_points"]))]
        pts = grp.set_index("idx").loc[idxs].geometry.tolist()
        geoms.append(pts[0] if len(pts) == 1 else LineString(pts))
    return gpd.GeoDataFrame(zones.copy(), geometry=geoms, crs=2100)


def export_gis(run_dir, samples, zones, window, scar, sim_star):
    """QGIS-ready bundle (EPSG:2100 GeoJSON) into <run_dir>/perimeter_gis/.
    Every filename is perimeter_-prefixed: the Exports publish step copies GIS
    files FLAT next to the 1a bundle, so unprefixed names would collide."""
    gis = run_dir / "perimeter_gis"
    gis.mkdir(exist_ok=True)

    if len(samples):
        kept = samples[~samples["excluded"]].copy()
        kept.to_file(gis / "perimeter_samples_tstar.geojson",
                     driver="GeoJSON")
        exc = samples[samples["excluded"]].copy()
        if len(exc):
            exc.to_file(gis / "perimeter_excluded_points.geojson",
                        driver="GeoJSON")
        if not zones.empty:
            zone_geometries(zones, samples).to_file(
                gis / "perimeter_zones_tstar.geojson", driver="GeoJSON")

    gpd.GeoDataFrame([{"name": "model window frame"}],
                     geometry=[window.exterior], crs=2100).to_file(
        gis / "perimeter_window_frame.geojson", driver="GeoJSON")
    # _boundary file carries LINES (matching its name); the sim file carries
    # the filled burned-area POLYGON and is named accordingly
    gpd.GeoDataFrame([{"name": "final scar boundary (holes filled)"}],
                     geometry=[scar.boundary], crs=2100).to_file(
        gis / "perimeter_scar_boundary.geojson", driver="GeoJSON")
    if sim_star is not None:
        gpd.GeoDataFrame([{"name": "sim burned area at t* (holes filled)"}],
                         geometry=[sim_star], crs=2100).to_file(
            gis / "perimeter_sim_tstar.geojson", driver="GeoJSON")
    print(f"Saved GIS bundle -> {gis}")
    return gis


def publish_exports(run_dir, gis_dir):
    """Copy the whole diagnostic set to DATA_DIR/Exports/<run_id>/validation/
    (the same per-run deliverable convention as the 1a validation)."""
    import shutil
    dest = DATA_DIR / "Exports" / run_dir.name / "validation"
    dest.mkdir(parents=True, exist_ok=True)
    names = ["perimeter_metrics.json", "perimeter_hourly.csv",
             "perimeter_zones.csv", "perimeter_report.md",
             "perimeter_dt_curve.png", "perimeter_signed_map.png",
             "perimeter_overlay.html"]
    for n in names:
        src = run_dir / n
        if src.exists():
            shutil.copy2(src, dest / n)
    if gis_dir and gis_dir.exists():
        for f in gis_dir.iterdir():
            shutil.copy2(f, dest / f.name)
    print(f"Published deliverables -> {dest}")


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main(run_dir_arg=None):
    run_dir = resolve_run_dir(run_dir_arg)
    perims = gpd.read_file(run_dir / "perimeters.geojson").to_crs(2100) \
        .sort_values("period").reset_index(drop=True)
    horizon = int(perims["period"].max())
    hours = [int(p) for p in perims["period"] if p >= 1]   # period 0 = initial condition
    seeded = int(perims["n_cells"].iloc[0]) >= SEED_MIN_CELLS
    run_kind = (f"front-seeded, hours 0..{horizon}" if seeded
                else f"point ignition, {horizon} h")

    # --- reference scar ---
    b = gpd.read_file(SCAR_SHP).to_crs(2100)
    scar_raw = unary_union(b.geometry)
    scar = fill_holes(scar_raw)
    scar_km2 = round(scar_raw.area / 1e6, 1)

    # --- step 0: window check (report only - the scar is never clipped) ---
    from cell2fire_adapter import read_asc_header
    forest_asc = run_dir / "instance" / "Forest.asc"
    if not forest_asc.exists():
        sys.exit(f"ERROR: {forest_asc} missing - cannot derive the model "
                 "window (protocol step 0)")
    wtr, wnr, wnc, _ = read_asc_header(str(forest_asc))
    window = window_box_from_header(wtr, wnr, wnc)
    outside_km2 = round(scar_raw.difference(window).area / 1e6, 1)
    outside_pct = round(100 * outside_km2 / scar_km2, 1) if scar_km2 else 0.0
    window_km = (f"{round((wnc * wtr.a) / 1000, 1)} x "
                 f"{round((wnr * abs(wtr.e)) / 1000, 1)}")

    print("=== Perimeter-to-perimeter geometric convergence diagnostic "
          f"({run_kind}) ===")
    print(f"Run: {run_dir.name} | reference scar {scar_km2} km2 (final, "
          "~6 days, suppressed)")
    print(f"Step 0 window check: window {window_km} km | scar outside the "
          f"window {outside_km2} km2 ({outside_pct}%)")

    # --- real side (constant across hours) ---
    real_pts = sample_boundary(scar)
    out_mask, near_mask = exclusion_masks_real(real_pts, window)
    real_pts["outside_window"] = out_mask
    real_pts["near_frame"] = near_mask
    real_pts["excluded"] = out_mask | near_mask
    kept_real = real_pts[~real_pts["excluded"]]
    scar_boundary = scar.boundary
    print(f"Real boundary: {len(real_pts)} samples ({int(out_mask.sum())} "
          f"outside window, {int(near_mask.sum())} near frame -> "
          f"{len(kept_real)} used)")

    # --- hourly loop: both directed distance sets per hour ---
    hour_records = []
    sim_by_hour = {}
    areas_raw_km2 = {}                      # unrounded, unfilled - the rate input
    for h in hours:
        sim_raw = perims.loc[perims["period"] == h].geometry.iloc[0]
        simf = fill_holes(sim_raw)
        sim_by_hour[h] = simf
        areas_raw_km2[h] = sim_raw.area / 1e6
        sim_pts = sample_boundary(simf)
        frame_mask = exclusion_mask_sim(sim_pts, window)
        kept_sim = sim_pts[~frame_mask]
        r2s_target = strip_frame_from_line(simf.boundary, window)
        d_s2r = directed_distances(kept_sim.geometry, scar_boundary)
        d_r2s = directed_distances(kept_real.geometry, r2s_target)
        s2r, r2s = direction_stats(d_s2r), direction_stats(d_r2s)
        ind = hour_indicators(s2r, r2s)
        rec = {"hour": h, "sim_km2": round(sim_raw.area / 1e6, 1),
               "n_sim_pts": len(sim_pts),
               "n_sim_excluded_frame": int(frame_mask.sum()),
               "n_sim_used": len(kept_sim),
               "n_real_used": len(kept_real),
               "mean_s2r_m": s2r["mean_m"] if s2r else None,
               "median_s2r_m": s2r["median_m"] if s2r else None,
               "p95_s2r_m": s2r["p95_m"] if s2r else None,
               "max_s2r_m": s2r["max_m"] if s2r else None,
               "mean_r2s_m": r2s["mean_m"] if r2s else None,
               "median_r2s_m": r2s["median_m"] if r2s else None,
               "p95_r2s_m": r2s["p95_m"] if r2s else None,
               "max_r2s_m": r2s["max_m"] if r2s else None,
               **ind,
               "degenerate": s2r is None or r2s is None}
        hour_records.append(rec)
        print(f"-- +{h:2d} h | sim {rec['sim_km2']:6.1f} km2 | "
              f"mean s->r {rec['mean_s2r_m']} m | "
              f"mean r->s {rec['mean_r2s_m']} m | D_t {rec['d_t_m']} m")
    hourly = pd.DataFrame(hour_records)

    # --- t*, censoring, stability ---
    ts = find_t_star(hourly)
    stab = stability_table(hourly, ts["hour"]) if ts else []
    if ts is None:
        print("!! no hour had a computable D_t - degenerate run")
    elif ts["censored"]:
        print(f"-- D_t minimised at the horizon boundary (+{ts['hour']} h, "
              f"{ts['d_t_m']} m): {CENSORED_STATEMENT}")
    else:
        print(f"-- t* = +{ts['hour']} h (D_t {ts['d_t_m']} m) - hour of "
              "closest geometric approach")
        if ts["ties"]:
            print(f"   ties at {ts['ties']} - earliest hour declared winner")
    agree = sum(1 for s in stab if s["agrees_with_t_star"])
    if stab:
        print(f"-- stability: {agree}/{len(stab)} alternative indicators "
              "agree with t* (they can never override it)")

    # --- signed distances + zonation at t* ---
    samples = gpd.GeoDataFrame()
    zones = pd.DataFrame()
    signed_summary = {}
    sim_star = None
    # rate: RAW (unfilled) burned areas, crossing decided on UNROUNDED values
    rate = first_hour_area_exceeds(areas_raw_km2, scar_raw.area / 1e6)
    if ts is not None:
        h_star = ts["hour"]
        sim_star = sim_by_hour[h_star]
        sim_pts = sample_boundary(sim_star)
        frame_mask = exclusion_mask_sim(sim_pts, window)
        sim_pts["excluded"] = frame_mask
        sim_pts["direction"] = "sim_to_real"
        r2s_target = strip_frame_from_line(sim_star.boundary, window)

        ksim = ~sim_pts["excluded"]
        d = directed_distances(sim_pts.loc[ksim].geometry, scar_boundary)
        inside = sim_pts.loc[ksim].geometry.within(scar).to_numpy()
        sim_pts["dist_m"] = np.nan
        sim_pts.loc[ksim, "dist_m"] = np.round(d, 1)
        sim_pts["signed_m"] = np.nan
        sim_pts.loc[ksim, "signed_m"] = np.round(
            apply_sign(d, inside, positive_when_inside=False), 1)

        rl = real_pts.copy()
        rl["direction"] = "real_to_sim"
        krl = ~rl["excluded"]
        d2 = directed_distances(rl.loc[krl].geometry, r2s_target)
        rl["dist_m"] = np.nan
        rl["signed_m"] = np.nan
        if d2 is not None:
            inside2 = rl.loc[krl].geometry.within(sim_star).to_numpy()
            rl.loc[krl, "dist_m"] = np.round(d2, 1)
            rl.loc[krl, "signed_m"] = np.round(
                apply_sign(d2, inside2, positive_when_inside=True), 1)

        cols = ["direction", "ring_id", "idx", "n_ring", "chainage_m",
                "dist_m", "signed_m", "excluded", "geometry"]
        samples = gpd.GeoDataFrame(
            pd.concat([sim_pts[cols], rl[cols]], ignore_index=True), crs=2100)

        for dname in ("sim_to_real", "real_to_sim"):
            sd = samples.loc[(samples["direction"] == dname)
                             & ~samples["excluded"], "signed_m"].dropna()
            if len(sd):
                signed_summary[dname] = {
                    "n_points": int(len(sd)),
                    "n_positive": int((sd > 0).sum()),
                    "n_negative": int((sd < 0).sum()),
                    "share_positive_pct": round(100 * float((sd > 0).mean())),
                    "mean_signed_m": round(float(sd.mean()), 1),
                    "median_signed_m": round(float(sd.median()), 1)}

        zones = zone_table(samples)
        n_zones = int((zones["kind"] == "zone").sum()) if not zones.empty else 0
        n_extr = (int((zones["kind"] == "isolated_extreme").sum())
                  if not zones.empty else 0)
        print(f"-- zones at t*: {n_zones} zones, {n_extr} isolated extremes "
              f"(threshold {int(ZONE_THRESHOLD_M)} m, min "
              f"{ZONE_MIN_POINTS} pts)")
        for dname, s in signed_summary.items():
            print(f"   {dname}: {s['share_positive_pct']}% positive "
                  f"(mean signed {s['mean_signed_m']} m)")
    else:
        n_zones = n_extr = 0

    if rate is not None:
        print(f"-- rate: simulated area first exceeds the final scar area at "
              f"+{rate} h (of a ~6-day real event)")
    else:
        print("-- rate: simulated area never exceeds the final scar area "
              "within the horizon")

    # --- metrics JSON ---
    metrics = {
        "protocol": {
            "name": "Perimeter-to-perimeter geometric convergence and rate "
                    "diagnostic",
            "framing": "geometric diagnostic of a free-burning run vs a "
                       "suppressed multi-day scar - never an accuracy score",
            "fixed_sentence": FIXED_SENTENCE,
            "sample_spacing_m": SAMPLE_SPACING_M,
            "frame_tol_m": FRAME_TOL_M,
            "zone_threshold_m": ZONE_THRESHOLD_M,
            "zone_min_points": ZONE_MIN_POINTS,
            "zone_length_convention": "length_m = n_points x 30 m (each "
                                      "sample represents one cell of "
                                      "boundary; the connecting arc spans "
                                      "n_points-1 intervals)",
            "tie_rule": "earliest hour",
            "censoring_rule": "t* at the first or last evaluated hour => "
                              + CENSORED_STATEMENT,
            "holes": "filled in both datasets (unburned islands out of scope)",
            "seed_handling": "full simulated perimeter compared; no seed "
                             "subtraction",
            "banned": "Moran's I / LISA / Gi* / IoU / precision / recall",
        },
        "run_id": run_dir.name, "run_kind": run_kind, "horizon_h": horizon,
        "crs": "EPSG:2100",
        "window": {"km": window_km,
                   "bounds": [round(v, 1) for v in window.bounds],
                   "scar_km2_raw": scar_km2,
                   "scar_outside_window_km2": outside_km2,
                   "scar_outside_window_pct": outside_pct},
        "real_boundary": {"n_sample_points": int(len(real_pts)),
                          "n_excluded_outside_window": int(out_mask.sum()),
                          "n_excluded_near_frame": int(near_mask.sum()),
                          "n_used": int(len(kept_real))},
        "hourly": hour_records,
        "t_star": ts,
        "stability": stab,
        "signed_at_t_star": {**signed_summary,
                             "convention": "positive always = simulation "
                                           "went beyond reality"},
        "zones_at_t_star": {"n_zones": n_zones,
                            "n_isolated_extremes": n_extr,
                            "zones": (zones.to_dict("records")
                                      if not zones.empty else [])},
        "rate": {"final_scar_km2": scar_km2,
                 "scar_km2_filled": round(scar.area / 1e6, 1),
                 "first_hour_sim_area_exceeds_scar": rate,
                 "convention": "raw (unfilled) burned areas, crossing decided "
                               "on unrounded values; hole filling applies to "
                               "the boundary comparison only"},
        "discretization_context": {
            "cell_m": 30.0,
            "note": "staircase raster boundaries simplified at half a cell; "
                    "distance differences of one to two cells are within "
                    "discretization noise - context only, not a plateau rule"},
    }
    mpath = run_dir / "perimeter_metrics.json"
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print(f"Saved metrics -> {mpath}")

    hourly.to_csv(run_dir / "perimeter_hourly.csv", index=False)
    print(f"Saved hourly table -> {run_dir / 'perimeter_hourly.csv'}")
    zones.to_csv(run_dir / "perimeter_zones.csv", index=False)
    print(f"Saved zone table -> {run_dir / 'perimeter_zones.csv'}")

    # --- the thesis-facing report ---
    interp = []
    if ts is not None and not ts["censored"]:
        interp.append(
            f"The free-burning perimeter comes geometrically closest to the "
            f"final scar at +{ts['hour']} h (D_t {ts['d_t_m']} m), with "
            f"{agree}/{len(stab)} alternative indicators agreeing on the "
            "hour.")
    elif ts is not None:
        interp.append(
            f"D_t is still minimised at the boundary of the horizon "
            f"(+{ts['hour']} h): {CENSORED_STATEMENT}.")
    if rate is not None:
        interp.append(
            f"The simulated area passes the final scar's total area at "
            f"+{rate} h, against ~6 days for the real suppressed event - the "
            "rate statement this diagnostic was designed to make.")
    interp.append(
        "Divergence zones are labelled over-extension (expected for a "
        "free-burning run vs a suppressed fire) and lag (possibly sectors "
        "the real fire reached only after the horizon); neither is scored "
        "as error. " + FIXED_SENTENCE)
    write_report(run_dir, {
        "run_id": run_dir.name, "run_kind": run_kind, "horizon": horizon,
        "scar_km2": scar_km2, "window_km": window_km,
        "scar_outside_km2": outside_km2, "scar_outside_pct": outside_pct,
        "n_real_pts": int(len(real_pts)),
        "n_real_outside": int(out_mask.sum()),
        "n_real_near_frame": int(near_mask.sum()),
        "n_real_used": int(len(kept_real)),
        "n_zones": n_zones, "n_extremes": n_extr,
        "interpretation": " ".join(interp),
    }, hourly, ts, stab, signed_summary, zones, rate)

    # --- charts + GIS bundle ---
    try:
        make_charts(run_dir, hourly, ts, samples, scar_boundary,
                    sim_star.boundary if sim_star is not None else None,
                    window)
    except Exception as e:
        print(f"(charts skipped: {e})")
    try:
        gis_dir = export_gis(run_dir, samples, zones, window, scar, sim_star)
    except Exception as e:
        gis_dir = None
        print(f"(GIS bundle skipped: {e})")

    # --- optional folium overlay ---
    try:
        import folium

        from route_shortest_path import BASEMAP, POINT_CRS
        c = window.centroid
        c_ll = gpd.GeoSeries([c], crs=2100).to_crs(POINT_CRS).iloc[0]
        fmap = folium.Map(location=[c_ll.y, c_ll.x], zoom_start=11,
                          tiles=BASEMAP)

        def layer(geom, name, color, fill_opacity=0.0, weight=2, dash=None,
                  show=True):
            if geom is None or geom.is_empty:
                return
            folium.GeoJson(
                gpd.GeoSeries([geom.simplify(20)], crs=2100)
                .to_crs(POINT_CRS).to_json(),
                name=name, show=show,
                style_function=lambda _f, cl=color, fo=fill_opacity,
                w=weight, d=dash: {"color": cl, "weight": w, "fillColor": cl,
                                   "fillOpacity": fo, "dashArray": d}
            ).add_to(fmap)

        layer(window.exterior, "model window frame", "#333", weight=1.5,
              dash="4 7")
        layer(scar, "final scar (holes filled) - reference", "#555", 0.08)
        if sim_star is not None and ts is not None:
            layer(sim_star, f"sim perimeter at t* (+{ts['hour']} h)", "red",
                  0.10)
        if not zones.empty:
            zg = zone_geometries(zones, samples)
            for _, z in zg.iterrows():
                col = "#d62728" if z["sign"] > 0 else "#1f77b4"
                layer(z.geometry, f"{z['kind']} #{z['zone_id']} "
                      f"({z['meaning']}, {int(z['length_m'])} m)", col,
                      weight=4, show=(z["kind"] == "zone"))
        folium.LayerControl(collapsed=False).add_to(fmap)
        out_html = run_dir / "perimeter_overlay.html"
        fmap.save(str(out_html))
        print(f"Saved overlay map -> {out_html}")
    except ImportError:
        print("(folium not installed - skipping the overlay map)")
    except Exception as e:
        print(f"(overlay map skipped: {e})")

    publish_exports(run_dir, gis_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
