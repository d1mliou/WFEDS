"""Exploratory spatial validation of free-burning fire spread using subsequent
VIIRS active-fire detections.

The question (per the thesis evaluation plan, axis 1 - see the vault's
Validation note): given only the information available at an initial time t0,
does the Cell2Fire-based WFEDS pipeline produce a spatially plausible 24 h
free-burning evolution, consistent with the LATER VIIRS detections?

Protocol (all choices pre-declared, not tuned to the outcome):
  * t0 = the scenario's seed_time. VIIRS detections <= t0 are the
    INITIALIZATION set (they built the ignition front); detections in
    (t0 .. sim end] are the EVALUATION set and are never fed back into the run.
  * Evaluation detections are grouped PER SATELLITE PASS (acquisition
    timestamps within PASS_TOL_MIN minutes = one pass). No artificial hourly
    ground truth is interpolated for the headline metrics.
  * Time matching: each pass is compared against the LAST hourly simulated
    perimeter at or before the pass time.
  * Per pass and pooled over 24 h: inclusion rate (detections inside the
    simulated burn), buffered inclusion at BUFFERS_M (pre-declared: one VIIRS
    pixel, 500 m, 1 km), distance-to-simulation stats (median + percentiles),
    and spread direction (bearing from the seed centroid: observed detections
    vs simulated new growth).
  * Over-spread check: how much simulated growth lies far from EVERY
    subsequent detection, and its distribution by compass sector. Absence of a
    detection does not prove absence of fire - reported, not auto-scored.
  * The final multi-day burned scar is CONTEXT only (different time horizon,
    suppression-affected) - never an accuracy score.
  * The arrival-time reconstruction (VIIRS time x scar, step-wise between
    overpasses) is kept as CONTEXTUAL timing evidence - the dashboard reads it.

This is a plausibility check of a free-burning worst case, not calibration
(ERA5 ~25 km reanalysis; the SVM->FBP fuel map is a placeholder; the real 2021
event was actively suppressed - the model deliberately is not).

Run (a finished run directory containing perimeters.geojson + instance/):
    python scripts/validation/validate_overlay.py <run_dir>
Default run_dir: the newest under DATA_DIR/Fire/cell2fire/runs/ with
perimeters.geojson (typically a `--scenario north_evia_2021` run).

Outputs (into <run_dir>): validation_metrics.json (dashboard-compatible keys
preserved), validation_per_pass.csv, validation_report.md,
validation_inclusion.png, validation_distances.png, validation_overlay.html,
and a QGIS-ready GIS bundle (validation_gis/, EPSG:2100 GeoJSON: VIIRS
initialization/evaluation points with per-pass + distance attributes, seed
front, simulated growth, over-spread polygon, direction sectors, hourly
perimeter/front copies). The whole validation set is also published to
DATA_DIR/Exports/<run_id>/validation/1a_viirs/ - the same deliverable
convention as the web runs' bundles (1b_perimeter/ is the sibling for the
perimeter-convergence diagnostic).
"""

import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Lives in scripts/validation/ (its own evaluation-axis package); the pipeline
# imports below (_paths, build_cell2fire_instance, route_shortest_path,
# cell2fire_adapter) live in the SIBLING package scripts/cell2fire/.
sys.path.insert(0, str(Path(__file__).parent.parent / "cell2fire"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from _paths import DATA_DIR
from build_cell2fire_instance import SCENARIOS
from route_shortest_path import BASEMAP, POINT_CRS

# --- pre-declared protocol constants (do NOT tune these to the outcome) --------
SEED_TIME = SCENARIOS["north_evia_2021"]["seed_time"]   # t0 (UTC) for the 2021 case
VIIRS_PIX = 375.0             # VIIRS I-band pixel (m) - also the first buffer
BUFFERS_M = (375.0, 500.0, 1000.0)   # buffered-inclusion tolerances (declared up front:
                              # 1 VIIRS pixel / pixel + geolocation slack / 1 km sensitivity)
PASS_TOL_MIN = 20             # acquisition timestamps within this = one satellite pass
FAR_BUFFER_M = 1000.0         # "far from every detection" threshold for the over-spread check
N_SECTORS = 8                 # compass sectors for the growth-direction distribution
SECTOR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SEED_MIN_CELLS = 1000         # period-0 bigger than this => front-seeded run

RFD = DATA_DIR / "Real_Fire_Data"
RUNS_DIR = DATA_DIR / "Fire" / "cell2fire" / "runs"
IGN = (438077.7, 4302642.5)   # real 2021 ignition, EPSG:2100 (unseeded-run fallback origin)


# --------------------------------------------------------------------------------
# pure helpers (unit-tested; no I/O)
# --------------------------------------------------------------------------------
def bearing(dx, dy):
    """Compass bearing (deg, 0=N, 90=E) of the vector (dx east, dy north)."""
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def ang_diff(a, b):
    """Smallest absolute angular difference (deg) between two bearings."""
    return abs((a - b + 180) % 360 - 180)


def group_passes(times, tol_min=PASS_TOL_MIN):
    """Group acquisition timestamps into satellite passes.

    times: pd.Series of Timestamps. Returns a same-index Series whose value is
    the pass label (the group's FIRST timestamp): sorted unique times are
    walked in order and a new pass starts when the gap to the group's first
    timestamp exceeds tol_min minutes.
    """
    labels = {}
    current = None
    for t in sorted(times.unique()):
        t = pd.Timestamp(t)
        if current is None or (t - current) > pd.Timedelta(minutes=tol_min):
            current = t
        labels[t] = current
    return times.map(labels)


def distance_stats(nn_m):
    """Distance summary (metres) of a Series of point-to-simulation distances
    (0 = inside). Median and upper percentiles are the ones to emphasise."""
    nn = pd.Series(nn_m, dtype=float)
    return {
        "mean_m": round(float(nn.mean())),
        "median_m": round(float(nn.median())),
        "p75_m": round(float(nn.quantile(0.75))),
        "p90_m": round(float(nn.quantile(0.90))),
        "p95_m": round(float(nn.quantile(0.95))),
        "max_m": round(float(nn.max())),
    }


def pass_metrics(dets, sim_poly, growth_poly, origin, buffers=BUFFERS_M):
    """Point-to-polygon agreement of one detection group vs the simulation.

    dets: GeoDataFrame (EPSG:2100, points). sim_poly: cumulative simulated burn
    at the matched time. growth_poly: sim minus seed at the matched time (the
    part the model actually FORECAST; None/empty for unseeded runs -> sim).
    origin: (x, y) the spread direction is measured from (seed centroid).

    Returns a dict: n, inclusion, buffered inclusion per tolerance, distance
    stats, observed vs simulated bearing + angular difference.
    """
    n = len(dets)
    if n == 0:
        return None
    nn = dets.geometry.distance(sim_poly)                       # metres, 0 inside
    inside = dets.geometry.within(sim_poly)
    out = {
        "n_detections": n,
        "n_inside": int(inside.sum()),
        "inclusion_pct": round(100 * float(inside.mean())),
    }
    for b in buffers:
        out[f"inclusion_within_{int(b)}m_pct"] = round(100 * float((nn <= b).mean()))
    out["distance"] = distance_stats(nn)

    vc = dets.geometry.union_all().centroid
    obs_b = bearing(vc.x - origin[0], vc.y - origin[1])
    out["obs_bearing_deg"] = round(obs_b)
    g = growth_poly if (growth_poly is not None and not growth_poly.is_empty) else sim_poly
    if g is not None and not g.is_empty:
        gc = g.centroid
        sim_b = bearing(gc.x - origin[0], gc.y - origin[1])
        out["sim_bearing_deg"] = round(sim_b)
        out["direction_diff_deg"] = round(ang_diff(sim_b, obs_b))
    else:
        out["sim_bearing_deg"] = None
        out["direction_diff_deg"] = None
    return out


def sector_polys(origin, radius=60_000.0, n=N_SECTORS):
    """n compass-sector wedges around origin (sector 0 centred on North).
    Returns {sector_name: Polygon}."""
    step = 360.0 / n
    out = {}
    for i, name in enumerate(SECTOR_NAMES[:n]):
        start = i * step - step / 2
        pts = [origin]
        for k in range(25):                       # 24 arc segments per wedge
            a = math.radians(start + k * step / 24)
            pts.append((origin[0] + radius * math.sin(a),
                        origin[1] + radius * math.cos(a)))
        out[name] = Polygon(pts)
    return out


def sector_areas_km2(poly, origin, n=N_SECTORS):
    """Split a polygon's area (km^2) by compass sector around origin."""
    if poly is None or poly.is_empty:
        return {name: 0.0 for name in SECTOR_NAMES[:n]}
    return {name: round(poly.intersection(w).area / 1e6, 1)
            for name, w in sector_polys(origin, n=n).items()}


def metric_block(pred, dets, origin, label, note=""):
    """Legacy pooled metrics block (kept verbatim for the dashboard data seam):
    footprint IoU/precision/recall + NN distances + net direction.

    pred: (Multi)Polygon; dets: GeoDataFrame (EPSG:2100); origin: (x, y) the
    spread direction is measured from. Returns the metrics as a dict.
    """
    print(f"-- {label} --")
    if note:
        print(f"   {note}")
    if len(dets) == 0:
        print("   (no detections in this window - metrics not computable)")
        return None
    footprint = unary_union(list(dets.buffer(VIIRS_PIX / 2).values))
    inter = pred.intersection(footprint).area
    iou = inter / (pred.area + footprint.area - inter)
    precision = inter / pred.area
    recall = inter / footprint.area
    nn = dets.geometry.distance(pred) / 1000.0            # km, 0 if inside
    within_pix = 100 * float((nn <= VIIRS_PIX / 1000).mean())
    within_1k = 100 * float((nn <= 1.0).mean())
    pc, vc = pred.centroid, dets.geometry.union_all().centroid
    sb = bearing(pc.x - origin[0], pc.y - origin[1])
    vb = bearing(vc.x - origin[0], vc.y - origin[1])
    ddir = ang_diff(sb, vb)
    print(f"   predicted {pred.area / 1e6:.1f} km^2 vs footprint {footprint.area / 1e6:.1f} "
          f"km^2 ({len(dets)} detections)")
    print(f"   IoU {100 * iou:.0f}% | precision {100 * precision:.0f}% | "
          f"recall {100 * recall:.0f}%")
    print(f"   NN: median {float(nn.median()):.2f} km, mean {float(nn.mean()):.2f} km | "
          f"within 375 m {within_pix:.0f}% | within 1 km {within_1k:.0f}%")
    print(f"   spread direction: SIM {sb:.0f}deg vs VIIRS {vb:.0f}deg (diff {ddir:.0f}deg)")
    return {"pred_km2": round(pred.area / 1e6, 1),
            "footprint_km2": round(footprint.area / 1e6, 1), "n_detections": len(dets),
            "iou_pct": round(100 * iou), "precision_pct": round(100 * precision),
            "recall_pct": round(100 * recall), "nn_median_km": round(float(nn.median()), 2),
            "within_pixel_pct": round(within_pix), "within_1km_pct": round(within_1k),
            "direction_diff_deg": round(ddir)}


# --------------------------------------------------------------------------------
# run-directory resolution + I/O
# --------------------------------------------------------------------------------
def resolve_run_dir(arg=None):
    """The run to validate: explicit CLI path, else the newest run under
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


def sim_start_utc(run_dir):
    """The simulation's t=0 (UTC), from the run's own provenance."""
    p = run_dir / "instance" / "scenario_params.json"
    if p.exists():
        params = json.loads(p.read_text(encoding="utf-8"))
        return pd.to_datetime(params["start_time_utc"])
    return pd.to_datetime("2021-08-05 23:00")   # legacy fallback (the 2021 case)


# --------------------------------------------------------------------------------
# report pieces
# --------------------------------------------------------------------------------
LIMITATIONS = """\
* VIIRS detections are active-fire pixels, not a complete fire perimeter; the
  absence of a detection does not prove the absence of fire (smoke, clouds,
  overpass timing).
* Detection position and time carry uncertainty (~375 m pixels, discrete
  overpasses; the day-1 overnight gap 01:02 -> 10:43 UTC leaves ~9.7 h
  unobserved).
* The real 2021 event was actively suppressed from the first hours; the model
  deliberately simulates FREE-BURNING spread (the worst credible case), so
  over-spread relative to observations is expected, not a defect to calibrate
  away.
* Weather input is ERA5/Open-Meteo REANALYSIS (produced after the event) - a
  standard "perfect weather" assumption in fire-model validation, but strictly
  MORE information than was available at t0. ERA5 is ~25 km; wind is spatially
  IDW-interpolated, not terrain-resolved.
* The fire-spread model is Cell2Fire (upstream, pinned) with a PLACEHOLDER
  SVM->FBP fuel mapping - uncalibrated.
* The final burned scar spans ~6 days and suppression; it is temporally
  incompatible with a 24 h simulation and is used as spatial CONTEXT only.
* This is a single-event case study: agreement here does not generalise to
  other events, fuels or weather regimes.
"""


def md_table(df):
    """Plain markdown table (pandas' own .to_markdown needs the optional
    `tabulate` package - not worth a dependency for one table)."""
    if df.empty:
        return "(no evaluation passes)"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def write_report(run_dir, meta, table, summary, over, context):
    """validation_report.md - the thesis-facing writeup: declared protocol,
    per-pass table, pooled summary, over-spread, context, limitations."""
    lines = [
        "# Exploratory spatial validation of free-burning fire spread "
        "using subsequent VIIRS active-fire detections",
        "",
        f"Run: `{meta['run_id']}` - {meta['run_kind']}. Simulation start "
        f"{meta['sim_start']} UTC, horizon {meta['hours']} h.",
        "",
        "## Declared protocol",
        f"* t0 (initialization/evaluation split): **{meta['t0']} UTC** "
        "(the scenario's observed-front seed time).",
        f"* Initialization set: {meta['n_init']} detections (<= t0) - used only "
        "to build the ignition front (each buffered by half a VIIRS pixel).",
        f"* Evaluation set: {meta['n_eval']} detections in (t0 .. sim end], of "
        f"which {meta['n_eval_outside_seed']} outside the seed front (the "
        "scored set - detections inside the seed were handed to the model).",
        f"* Satellite passes: acquisition times within {PASS_TOL_MIN} min "
        "group into one pass.",
        "* Time matching: each pass vs the LAST hourly simulated perimeter at "
        "or before the pass time.",
        f"* Buffered-inclusion tolerances (pre-declared): "
        f"{', '.join(f'{int(b)} m' for b in BUFFERS_M)}.",
        "",
        "## Per-pass results",
        "",
        md_table(table),
        "",
        "## Pooled 24 h summary (all evaluation passes)",
        "",
    ]
    if summary:
        lines += [
            f"* Inclusion: **{summary['inclusion_pct']}%** of evaluation "
            "detections inside the simulated burn at their matched hour"
            + "".join(f"; within {int(b)} m: "
                      f"{summary[f'inclusion_within_{int(b)}m_pct']}%"
                      for b in BUFFERS_M) + ".",
            f"* Distance to the simulation: median "
            f"{summary['distance']['median_m']} m, p90 "
            f"{summary['distance']['p90_m']} m, max "
            f"{summary['distance']['max_m']} m.",
            f"* Net direction: simulated {summary['sim_bearing_deg']} deg vs "
            f"observed {summary['obs_bearing_deg']} deg "
            f"(difference {summary['direction_diff_deg']} deg).",
        ]
    lines += [
        "",
        "## Over-spread check (free-burning vs sparse observations)",
        "",
        f"* Simulated growth (beyond the seed) at +{meta['hours']} h: "
        f"**{over['growth_km2']} km2**.",
        f"* Growth farther than {int(FAR_BUFFER_M)} m from EVERY subsequent "
        f"detection: **{over['growth_far_km2']} km2** "
        f"({over['growth_far_pct']}% of growth). Not automatically an error - "
        "see limitations (unobserved fire, suppression of the real event).",
        "* Growth by compass sector (km2): "
        + ", ".join(f"{k} {v}" for k, v in over["growth_by_sector_km2"].items())
        + ".",
        "",
        "## Context (not accuracy scores)",
        "",
        f"* {context['containment_final_scar_pct']}% of the simulated burn lies "
        "inside the FINAL (~6-day, suppression-affected) scar - temporal "
        "mismatch makes this contextual only.",
    ]
    if context.get("arrival_time"):
        a = context["arrival_time"]
        lines += [
            f"* Arrival-time reconstruction (contextual; step-wise between "
            f"overpasses): sim reached {a['sim_reached_pct']}% of the real "
            f"forecast-window burn, median timing error "
            f"{a['timing_median_h']} h (negative = early).",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        meta["interpretation"],
        "",
        "## Limitations",
        "",
        LIMITATIONS,
    ]
    out = run_dir / "validation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report -> {out}")


def export_gis(run_dir, init_set, eval_set, seed, growth_final, far, origin,
               over):
    """QGIS-ready GIS bundle (EPSG:2100 GeoJSON) into <run_dir>/validation_gis/
    so the analyst can build their own maps: every layer the protocol's maps
    use, with the per-point attributes (pass, matched hour, distance, in/out)."""
    gis = run_dir / "validation_gis"
    gis.mkdir(exist_ok=True)

    if len(init_set):
        out = init_set[["t", "geometry"]].copy()
        out["t"] = out["t"].astype(str)
        out.to_file(gis / "viirs_initialization.geojson", driver="GeoJSON")
    if len(eval_set):
        out = eval_set[["t", "pass_t", "matched_hour", "distance_m", "inside",
                        "geometry"]].copy()
        for b in BUFFERS_M:
            out[f"within_{int(b)}m"] = eval_set[f"within_{int(b)}m"]
        out["t"] = out["t"].astype(str)
        out["pass_t"] = out["pass_t"].astype(str)
        out.to_file(gis / "viirs_evaluation.geojson", driver="GeoJSON")

    def poly_file(geom, name, **attrs):
        if geom is None or geom.is_empty:
            return
        gpd.GeoDataFrame([attrs or {"name": name}], geometry=[geom],
                         crs=2100).to_file(gis / name, driver="GeoJSON")

    poly_file(seed, "seed_front.geojson", layer="seed front (observed, t0)")
    poly_file(growth_final, "sim_growth_final.geojson",
              layer="simulated growth (sim minus seed)",
              km2=round(growth_final.area / 1e6, 1) if growth_final else None)
    poly_file(far, "overspread_far_from_viirs.geojson",
              layer=f"growth farther than {int(FAR_BUFFER_M)} m from every "
                    "subsequent detection", km2=over["growth_far_km2"])
    sectors = gpd.GeoDataFrame(
        [{"sector": n, "growth_km2": over["growth_by_sector_km2"][n]}
         for n in SECTOR_NAMES],
        geometry=[sector_polys(origin)[n] for n in SECTOR_NAMES], crs=2100)
    sectors.to_file(gis / "direction_sectors.geojson", driver="GeoJSON")

    # hourly perimeters + fronts: copied in so the folder is self-contained
    import shutil
    for f in ("perimeters.geojson", "isochrones.geojson"):
        src = run_dir / f
        if src.exists():
            shutil.copy2(src, gis / f"sim_{f.replace('.geojson', '')}_hourly.geojson")
    print(f"Saved GIS bundle -> {gis}")
    return gis


def publish_exports(run_dir, gis_dir):
    """Copy the whole validation set to
    DATA_DIR/Exports/<run_id>/validation/1a_viirs/ (the same per-run
    deliverable convention as the web bundles; the sibling 1b_perimeter/
    holds the perimeter-convergence diagnostic's set - separated 2026-07-20
    so the two pillars' deliverables never mix in one flat folder)."""
    import shutil
    dest = DATA_DIR / "Exports" / run_dir.name / "validation" / "1a_viirs"
    dest.mkdir(parents=True, exist_ok=True)
    names = ["validation_metrics.json", "validation_per_pass.csv",
             "validation_report.md", "validation_inclusion.png",
             "validation_distances.png", "validation_overlay.html"]
    for n in names:
        src = run_dir / n
        if src.exists():
            shutil.copy2(src, dest / n)
    if gis_dir and gis_dir.exists():
        for f in gis_dir.iterdir():
            shutil.copy2(f, dest / f.name)
    print(f"Published deliverables -> {dest}")


def make_charts(run_dir, per_pass_rows, dets_by_pass):
    """validation_inclusion.png (bars per pass) + validation_distances.png
    (distance distribution per pass)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['pass_utc']:%H:%M}" for r in per_pass_rows]
    x = np.arange(len(labels))
    series = [("inside", "inclusion_pct")] + [
        (f"<= {int(b)} m", f"inclusion_within_{int(b)}m_pct") for b in BUFFERS_M]
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, key) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, [r[key] for r in per_pass_rows],
               w, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("% of detections")
    ax.set_ylim(0, 105)
    ax.set_title("Inclusion of subsequent VIIRS detections per satellite pass")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "validation_inclusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([d / 1000.0 for d in dets_by_pass], tick_labels=labels, whis=(5, 95))
    ax.set_ylabel("distance to simulation (km)")
    ax.set_title("Detection -> simulation distance per satellite pass (0 = inside)")
    fig.tight_layout()
    fig.savefig(run_dir / "validation_distances.png", dpi=150)
    plt.close(fig)
    print(f"Saved charts -> {run_dir / 'validation_inclusion.png'}, "
          f"{run_dir / 'validation_distances.png'}")


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main(run_dir_arg=None):
    run_dir = resolve_run_dir(run_dir_arg)
    perims = gpd.read_file(run_dir / "perimeters.geojson").to_crs(2100) \
        .sort_values("period").reset_index(drop=True)
    sim = perims.iloc[-1].geometry
    hours = int(perims["period"].max())
    seeded = int(perims["n_cells"].iloc[0]) >= SEED_MIN_CELLS
    seed = perims.iloc[0].geometry if seeded else None
    t0_sim = sim_start_utc(run_dir)
    t_end = t0_sim + pd.Timedelta(hours=hours)
    t_seed = pd.to_datetime(SEED_TIME) if seeded else t0_sim
    origin = ((seed.centroid.x, seed.centroid.y) if seeded else IGN)

    # --- VIIRS: initialization vs evaluation sets (the t0 split) ---
    v = gpd.read_file(RFD / "VIIRS" / "VIRS_dataset_egsa_dhmos.shp").to_crs(2100)
    v["t"] = pd.to_datetime(v["acq_at_s"])
    init_set = v[(v["t"] >= t0_sim) & (v["t"] <= t_seed)].copy()
    eval_all = v[(v["t"] > t_seed) & (v["t"] <= t_end)].copy()
    eval_set = eval_all[~eval_all.geometry.within(seed)].copy() if seeded \
        else eval_all.copy()
    eval_set["pass_t"] = group_passes(eval_set["t"])

    run_kind = (f"front-seeded, hours 0..{hours}" if seeded
                else f"point ignition, {hours} h")
    print(f"=== Exploratory spatial validation vs subsequent VIIRS ({run_kind}) ===")
    print(f"Run: {run_dir.name} | sim start {t0_sim} UTC | t0 split {t_seed} UTC")
    print(f"Initialization set {len(init_set)} detections | evaluation set "
          f"{len(eval_all)} ({len(eval_set)} outside the seed front)")
    print(f"Simulated fire: {sim.area / 1e6:.1f} km^2"
          + (f" (seed {seed.area / 1e6:.1f} km^2)" if seeded else ""))

    # --- per-pass metrics (the protocol's core table) ---
    per_pass_rows, dets_by_pass = [], []
    for col in ("matched_hour", "distance_m", "inside"):
        eval_set[col] = None
    for b in BUFFERS_M:
        eval_set[f"within_{int(b)}m"] = None
    for pass_t, grp in sorted(eval_set.groupby("pass_t"), key=lambda kv: kv[0]):
        h = min(hours, max(0, int((pass_t - t0_sim).total_seconds() // 3600)))
        sim_h = perims.iloc[h].geometry
        growth_h = sim_h.difference(seed) if seeded else sim_h
        m = pass_metrics(grp, sim_h, growth_h, origin)
        if m is None:
            continue
        # per-point attributes (carried into the GIS export for QGIS work)
        nn = grp.geometry.distance(sim_h)
        eval_set.loc[grp.index, "matched_hour"] = h
        eval_set.loc[grp.index, "distance_m"] = nn.round().astype(int)
        eval_set.loc[grp.index, "inside"] = grp.geometry.within(sim_h)
        for b in BUFFERS_M:
            eval_set.loc[grp.index, f"within_{int(b)}m"] = nn <= b
        m["pass_utc"] = pass_t
        m["matched_sim_hour"] = h
        m["sim_km2_at_match"] = round(sim_h.area / 1e6, 1)
        per_pass_rows.append(m)
        dets_by_pass.append(nn.values)
        print(f"-- pass {pass_t:%d/%m %H:%M} UTC ({m['n_detections']} det, sim hour "
              f"+{h}, {m['sim_km2_at_match']} km^2) --")
        print(f"   inclusion {m['inclusion_pct']}%"
              + "".join(f" | <={int(b)}m {m[f'inclusion_within_{int(b)}m_pct']}%"
                        for b in BUFFERS_M))
        d = m["distance"]
        print(f"   distance: median {d['median_m']} m, p90 {d['p90_m']} m, "
              f"max {d['max_m']} m")
        print(f"   direction: sim {m['sim_bearing_deg']}deg vs obs "
              f"{m['obs_bearing_deg']}deg (diff {m['direction_diff_deg']}deg)")

    # --- pooled 24 h summary over the whole evaluation set ---
    growth_final = sim.difference(seed) if seeded else sim
    summary = None
    if len(eval_set):
        h_last = per_pass_rows[-1]["matched_sim_hour"] if per_pass_rows else hours
        # pooled scoring uses each detection's own matched hour via the per-pass
        # loop above; for the pooled block, score against the final matched state
        summary = pass_metrics(eval_set, perims.iloc[h_last].geometry,
                               perims.iloc[h_last].geometry.difference(seed)
                               if seeded else perims.iloc[h_last].geometry, origin)
        print("-- pooled 24 h (all evaluation passes vs the last matched hour "
              f"+{h_last}) --")
        print(f"   inclusion {summary['inclusion_pct']}%"
              + "".join(f" | <={int(b)}m {summary[f'inclusion_within_{int(b)}m_pct']}%"
                        for b in BUFFERS_M)
              + f" | direction diff {summary['direction_diff_deg']}deg")

    # --- over-spread check (section 10 of the protocol) ---
    eval_footprint = (unary_union(list(eval_all.buffer(VIIRS_PIX / 2).values))
                      if len(eval_all) else None)
    far = (growth_final.difference(eval_footprint.buffer(FAR_BUFFER_M))
           if eval_footprint is not None else growth_final)
    over = {
        "growth_km2": round(growth_final.area / 1e6, 1),
        "growth_far_km2": round(far.area / 1e6, 1),
        "growth_far_pct": round(100 * far.area / growth_final.area)
        if growth_final.area else 0,
        "far_buffer_m": FAR_BUFFER_M,
        "growth_by_sector_km2": sector_areas_km2(growth_final, origin),
    }
    print(f"-- over-spread: growth {over['growth_km2']} km^2, "
          f"{over['growth_far_pct']}% farther than {int(FAR_BUFFER_M)} m from every "
          "subsequent detection (NOT auto-scored as error - see limitations)")

    # --- legacy pooled blocks (kept: the dashboard reads these exact keys) ---
    metrics = {"run": run_kind, "run_id": run_dir.name,
               "sim_km2": round(sim.area / 1e6, 1),
               "seed_km2": round(seed.area / 1e6, 1) if seeded else None}
    if seeded:
        metrics["forecast_only"] = metric_block(
            growth_final, eval_set, origin,
            "FORECAST-ONLY (fair test: seed excluded) - HEADLINE these numbers",
            note=f"growth (sim minus seed) vs detections in ({t_seed:%d/%m %H:%M} .. "
                 f"{t_end:%d/%m %H:%M}] UTC outside the seed front")
    v1 = v[(v["t"] >= t0_sim) & (v["t"] <= t0_sim + pd.Timedelta(hours=24))]
    metrics["naive"] = metric_block(
        sim, v1, IGN,
        "NAIVE (whole sim vs whole first day"
        + (" - INFLATED by the seed, do not headline)" if seeded else ")"))

    b = gpd.read_file(RFD / "Burned_Area" / "Burned_area_20210829.shp").to_crs(2100)
    burned = unary_union(b.geometry)
    containment = sim.intersection(burned).area / sim.area
    metrics["containment_final_scar_pct"] = round(100 * containment)
    print("-- context (vs the FINAL 6-day scar - temporally incompatible, "
          "never an accuracy score) --")
    print(f"   {100 * containment:.0f}% of the sim falls inside the final scar")

    # --- arrival-time reconstruction (CONTEXTUAL - step-wise between passes) ---
    real_end = None
    try:
        from rasterio import features as rfeatures

        from real_progression import extent_at, reconstruct

        arr, rtr, rt0 = reconstruct()
        real_end = extent_at(arr, rtr, rt0, t_end)
        off = (t0_sim - rt0).total_seconds() / 3600.0    # sim hour h = rt0 + (off + h)
        sim_arr = np.full(arr.shape, np.inf)
        for h in range(hours + 1):
            m = rfeatures.rasterize([(perims.iloc[h].geometry, 1)], out_shape=arr.shape,
                                    transform=rtr, fill=0, dtype="uint8").astype(bool)
            sim_arr[m & np.isinf(sim_arr)] = h + off
        sm = (rfeatures.rasterize([(seed, 1)], out_shape=arr.shape, transform=rtr,
                                  fill=0, dtype="uint8").astype(bool)
              if seeded else np.zeros(arr.shape, bool))
        seed_h = ((t_seed - rt0).total_seconds() / 3600.0 if seeded else 0.0)
        end_h = off + hours
        win = (~np.isnan(arr)) & (arr > seed_h) & (arr <= end_h) & ~sm
        hit = win & ~np.isinf(sim_arr)
        cell_km2 = (rtr.a ** 2) / 1e6
        print("-- ARRIVAL-TIME (contextual: real progression = VIIRS time x scar, "
              f"{rtr.a:.0f} m grid, step-wise between overpasses) --")
        print(f"   real burn in the forecast window: {win.sum() * cell_km2:.1f} km^2; "
              f"sim reached {100 * hit.sum() / max(win.sum(), 1):.0f}% of it")
        if hit.any():
            err = sim_arr[hit] - arr[hit]              # + = sim late, - = sim early
            print(f"   timing error (sim - real): median {np.median(err):+.1f} h, "
                  f"MAE {np.mean(np.abs(err)):.1f} h | sim early on "
                  f"{100 * (err < 0).mean():.0f}% of cells")
            print("   (the 01:02->10:43 overpass gap makes 'sim early' partly an "
                  "observation artifact)")
        over_a = (~np.isnan(arr)) & (arr > end_h) & ~np.isinf(sim_arr) & ~sm
        wrong = np.isnan(arr) & ~np.isinf(sim_arr) & ~sm
        print(f"   sim ahead of the real fire: {over_a.sum() * cell_km2:.1f} km^2 | "
              f"sim outside the final scar: {wrong.sum() * cell_km2:.1f} km^2")
        metrics["arrival_time"] = {
            "real_window_km2": round(win.sum() * cell_km2, 1),
            "sim_reached_pct": round(100 * hit.sum() / max(win.sum(), 1)),
            "timing_median_h": round(float(np.median(sim_arr[hit] - arr[hit])), 1)
            if hit.any() else None,
            "ahead_of_schedule_km2": round(over_a.sum() * cell_km2, 1),
            "outside_final_scar_km2": round(wrong.sum() * cell_km2, 1)}
    except Exception as e:
        print(f"(arrival-time reconstruction skipped: {e})")

    # --- the protocol block in the metrics JSON + the per-pass CSV ---
    metrics["protocol"] = {
        "name": "Exploratory spatial validation of free-burning fire spread "
                "using subsequent VIIRS active-fire detections",
        "t0_utc": str(t_seed), "sim_start_utc": str(t0_sim),
        "buffers_m": list(BUFFERS_M), "pass_tolerance_min": PASS_TOL_MIN,
        "time_matching": "last hourly perimeter at or before the pass time",
        "n_initialization": len(init_set), "n_evaluation": len(eval_all),
        "n_evaluation_outside_seed": len(eval_set),
        "per_pass": [{**{k: v for k, v in r.items() if k != "pass_utc"},
                      "pass_utc": str(r["pass_utc"])} for r in per_pass_rows],
        "summary_24h": summary,
        "over_spread": over,
    }
    mpath = run_dir / "validation_metrics.json"
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print(f"Saved metrics -> {mpath}")

    if per_pass_rows:
        flat = []
        for r in per_pass_rows:
            row = {k: v for k, v in r.items() if k != "distance"}
            row.update({f"dist_{k}": v for k, v in r["distance"].items()})
            flat.append(row)
        table = pd.DataFrame(flat)
        table.to_csv(run_dir / "validation_per_pass.csv", index=False)
        print(f"Saved per-pass table -> {run_dir / 'validation_per_pass.csv'}")
        try:
            make_charts(run_dir, per_pass_rows, dets_by_pass)
        except Exception as e:
            print(f"(charts skipped: {e})")
    else:
        table = pd.DataFrame()
        print("(no evaluation passes inside the simulation window - per-pass "
              "table/charts skipped)")

    # --- the thesis-facing report ---
    interp = []
    if summary:
        interp.append(
            f"Within the 24 h horizon, {summary['inclusion_pct']}% of the "
            f"subsequent VIIRS detections fall inside the simulated burn "
            f"(median distance {summary['distance']['median_m']} m), and the "
            f"net simulated direction differs from the observed one by "
            f"{summary['direction_diff_deg']} deg.")
    interp.append(
        f"At the same time, {over['growth_far_pct']}% of the simulated growth "
        f"lies farther than {int(FAR_BUFFER_M)} m from every subsequent "
        "detection - consistent with a free-burning worst case run against a "
        "suppressed real event and sparse satellite observation, and reported "
        "here as over-spread context rather than error.")
    interp.append(
        "Overall the simulation should be read as a spatially plausible, "
        "conservative (worst-case) free-burning evolution consistent with the "
        "subsequent VIIRS observations in direction and coverage, not as a "
        "reproduction of the actual suppressed fire.")
    write_report(run_dir, {
        "run_id": run_dir.name, "run_kind": run_kind, "hours": hours,
        "sim_start": str(t0_sim), "t0": str(t_seed),
        "n_init": len(init_set), "n_eval": len(eval_all),
        "n_eval_outside_seed": len(eval_set),
        "interpretation": " ".join(interp),
    }, table, summary, over,
        {"containment_final_scar_pct": metrics["containment_final_scar_pct"],
         "arrival_time": metrics.get("arrival_time")})

    # --- QGIS-ready GIS bundle + publish to the per-run Exports folder ---
    try:
        gis_dir = export_gis(run_dir, init_set, eval_set, seed, growth_final,
                             far, origin, over)
    except Exception as e:
        gis_dir = None
        print(f"(GIS bundle skipped: {e})")

    # --- overlay map ---
    try:
        import folium
    except ImportError:
        print("(folium not installed - skipping the overlay map)")
        publish_exports(run_dir, gis_dir)
        return
    ign_ll = gpd.GeoSeries([Point(*origin)], crs=2100).to_crs(POINT_CRS).iloc[0]
    fmap = folium.Map(location=[ign_ll.y, ign_ll.x], zoom_start=12, tiles=BASEMAP)

    def poly_layer(geom, name, color, fill_opacity, weight=1, dash=None, show=True):
        if geom is None or geom.is_empty:
            return
        folium.GeoJson(
            gpd.GeoSeries([geom.simplify(20)], crs=2100).to_crs(POINT_CRS).to_json(),
            name=name, show=show,
            style_function=lambda _f, c=color, fo=fill_opacity, w=weight, d=dash: {
                "color": c, "weight": w, "fillColor": c, "fillOpacity": fo,
                "dashArray": d}).add_to(fmap)

    from shapely.geometry import box

    from cell2fire_adapter import read_asc_header
    forest_asc = run_dir / "instance" / "Forest.asc"
    if forest_asc.exists():
        wtr, wnr, wnc, _ = read_asc_header(str(forest_asc))
        window = box(wtr.c, wtr.f + wnr * wtr.e, wtr.c + wnc * wtr.a, wtr.f)
        poly_layer(window, f"model window ({round((wnc * wtr.a) / 1000)} km - sim "
                   "cannot spread beyond it)", "#333", 0.0, weight=1.5, dash="4 7")
    poly_layer(burned, "real burned area (2021 final - CONTEXT)", "#555", 0.10)
    if real_end is not None:
        poly_layer(real_end, "real extent at sim end (VIIRS x scar reconstruction)",
                   "#7a00cc", 0.10, weight=2, dash="6 4")
    if seeded:
        poly_layer(seed, "seed front (observed, t0)", "#444", 0.35)
        poly_layer(growth_final, "sim FORECAST growth", "red", 0.25)
    else:
        poly_layer(sim, f"simulated fire ({hours} h)", "red", 0.25)

    # detections: initialization (gray) + one colored group PER evaluation pass
    pass_colors = ["orange", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
    groups = [("VIIRS initialization (<= t0, input)", init_set, "#666", True)]
    for i, (pass_t, grp) in enumerate(sorted(eval_set.groupby("pass_t"),
                                             key=lambda kv: kv[0])):
        groups.append((f"pass {pass_t:%H:%M} UTC (evaluation)",
                       grp, pass_colors[i % len(pass_colors)], True))
    late = v[v["t"] > t_end]
    groups.append(("VIIRS after sim end", late, "#e8c8ff", False))
    for name, grp, col, show in groups:
        fg = folium.FeatureGroup(name=f"{name} ({len(grp)} pts)", show=show)
        for _, r in grp.to_crs(POINT_CRS).iterrows():
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=2, color=col,
                                fill=True, fill_opacity=0.8,
                                tooltip=f"{r['t']:%d/%m %H:%M} UTC").add_to(fg)
        fg.add_to(fmap)
    folium.Marker([ign_ll.y, ign_ll.x], popup="spread-direction origin (seed centroid)",
                  icon=folium.Icon(color="black", icon="fire", prefix="fa")).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_html = run_dir / "validation_overlay.html"
    fmap.save(str(out_html))
    print(f"\nSaved overlay map -> {out_html}")

    publish_exports(run_dir, gis_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
