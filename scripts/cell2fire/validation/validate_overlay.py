"""Validation: overlay the simulated fire on the REAL VIIRS detections +
the 2021 burned-area ground truth, and print agreement metrics.

Two metric blocks:
  * NAIVE - the whole sim vs the whole first-day VIIRS footprint. For a FRONT-SEEDED
    run this is INFLATED: the seed cells are part of the "prediction", so the sim gets
    credit for detections it was handed as the initial condition.
  * FORECAST-ONLY (the fair test, seeded runs) - evaluates only what the model actually
    FORECAST: sim growth (final minus seed) vs the detections AFTER the seed window
    (and outside the seed front). This is the number to headline.

This is a plausibility check, not calibration-grade accuracy
(ERA5 is ~25 km, the SVM->FBP fuel map is a placeholder).

Run:
    python scripts/cell2fire/validation/validate_overlay.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Lives one level down (scripts/cell2fire/validation/); its sibling imports below
# (_paths, build_cell2fire_instance, route_shortest_path, real_progression,
# cell2fire_adapter) are all one directory UP, in scripts/cell2fire/ proper.
sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from _paths import DATA_DIR
from build_cell2fire_instance import SCENARIOS
from route_shortest_path import BASEMAP, POINT_CRS

# Observed-front cutoff: single source of truth is the scenario config (the old
# standalone seed_from_viirs.py was folded into build_cell2fire_instance's ONE
# seeding rule - every observed point buffered by half a VIIRS pixel).
SEED_TIME = SCENARIOS["north_evia_2021"]["seed_time"]

RFD = DATA_DIR / "Real_Fire_Data"
PERIMS = DATA_DIR / "Fire" / "cell2fire" / "perimeters.geojson"
INSTANCE = DATA_DIR / "Fire" / "cell2fire" / "instance"
OUT_HTML = DATA_DIR / "Evacuation_c2f" / "validation_overlay.html"
IGN = (438077.7, 4302642.5)              # real ignition, EPSG:2100
SIM_START = "2021-08-05 23:00"           # sim first hour (UTC, matches VIIRS)
SIM_HOURS = 24                           # naive first-day comparison window
VIIRS_PIX = 375.0                        # VIIRS I-band pixel (m)
SEED_MIN_CELLS = 1000                    # period-0 bigger than this => front-seeded run


def bearing(dx, dy):
    """Compass bearing (deg, 0=N, 90=E) of the vector (dx east, dy north)."""
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def metric_block(pred, dets, origin, label, note=""):
    """Print agreement metrics of predicted burn `pred` vs VIIRS points `dets`.

    pred: (Multi)Polygon; dets: GeoDataFrame (EPSG:2100); origin: (x, y) the spread
    direction is measured from. Returns the metrics as a dict (for the JSON).
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
    ddir = abs((sb - vb + 180) % 360 - 180)
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


def main():
    # --- simulated fire (final = last hourly perimeter) ---
    perims = gpd.read_file(PERIMS).to_crs(2100).sort_values("period").reset_index(drop=True)
    sim = perims.iloc[-1].geometry
    hours = int(perims["period"].max())
    seeded = int(perims["n_cells"].iloc[0]) >= SEED_MIN_CELLS
    seed = perims.iloc[0].geometry if seeded else None

    # --- VIIRS, first-day window (naive comparison, as before) ---
    v = gpd.read_file(RFD / "VIIRS" / "VIRS_dataset_egsa_dhmos.shp").to_crs(2100)
    v["t"] = pd.to_datetime(v["acq_at_s"])
    t0 = pd.to_datetime(SIM_START)
    t_end = t0 + pd.Timedelta(hours=hours)             # sim really ends here
    v1 = v[(v["t"] >= t0) & (v["t"] <= t0 + pd.Timedelta(hours=SIM_HOURS))].copy()

    # --- 2021 burned-area ground truth (context only) ---
    b = gpd.read_file(RFD / "Burned_Area" / "Burned_area_20210829.shp").to_crs(2100)
    burned = unary_union(b.geometry)

    run_kind = f"front-seeded, hours 0..{hours}" if seeded else f"point ignition, {hours} h"
    print(f"=== Fire-model validation ({run_kind}) ===")
    print(f"Simulated fire: {sim.area / 1e6:.1f} km^2"
          + (f" (seed {seed.area / 1e6:.1f} km^2)" if seeded else ""))
    metrics = {"run": run_kind, "sim_km2": round(sim.area / 1e6, 1),
               "seed_km2": round(seed.area / 1e6, 1) if seeded else None}

    if seeded:
        # ---- FORECAST-ONLY: what the model predicted beyond its initial condition ----
        t_seed = pd.to_datetime(SEED_TIME)
        growth = sim.difference(seed)
        tgt = v1[(v1["t"] > t_seed) & (v1["t"] <= t_end)].copy()
        tgt = tgt[~tgt.geometry.within(seed)]
        metrics["forecast_only"] = metric_block(
            growth, tgt, (seed.centroid.x, seed.centroid.y),
            "FORECAST-ONLY (fair test: seed excluded) - HEADLINE these numbers",
            note=f"growth (sim minus seed) vs detections in ({t_seed:%d/%m %H:%M} .. "
                 f"{t_end:%d/%m %H:%M}] UTC outside the seed front",
        )

    # ---- NAIVE: whole sim vs whole first-day footprint (inflated when seeded) ----
    metrics["naive"] = metric_block(
        sim, v1, IGN,
        "NAIVE (whole sim vs whole first day"
        + (" - INFLATED by the seed, do not headline)" if seeded else ")"),
    )
    containment = sim.intersection(burned).area / sim.area
    metrics["containment_final_scar_pct"] = round(100 * containment)
    print("-- loose context (vs the FINAL 6-day scar - too lenient, do not headline) --")
    print(f"   {100 * containment:.0f}% of the sim falls inside the final ~165 km^2 scar")

    # ---- ARRIVAL-TIME: sim timing vs the real progression (VIIRS time x burned scar).
    #      Combines the two ground truths: the scar says WHERE, VIIRS says WHEN.
    real_end = None
    try:
        import numpy as np
        from rasterio import features as rfeatures

        from real_progression import extent_at, reconstruct

        arr, rtr, rt0 = reconstruct()
        real_end = extent_at(arr, rtr, rt0, t_end)
        off = (t0 - rt0).total_seconds() / 3600.0      # sim hour h = rt0 + (off + h)
        sim_arr = np.full(arr.shape, np.inf)
        for h in range(hours + 1):
            m = rfeatures.rasterize([(perims.iloc[h].geometry, 1)], out_shape=arr.shape,
                                    transform=rtr, fill=0, dtype="uint8").astype(bool)
            sim_arr[m & np.isinf(sim_arr)] = h + off
        sm = (rfeatures.rasterize([(seed, 1)], out_shape=arr.shape, transform=rtr,
                                  fill=0, dtype="uint8").astype(bool)
              if seeded else np.zeros(arr.shape, bool))
        seed_h = ((pd.to_datetime(SEED_TIME) - rt0).total_seconds() / 3600.0
                  if seeded else 0.0)
        end_h = off + hours
        win = (~np.isnan(arr)) & (arr > seed_h) & (arr <= end_h) & ~sm
        hit = win & ~np.isinf(sim_arr)
        cell_km2 = (rtr.a ** 2) / 1e6
        print("-- ARRIVAL-TIME (real progression = VIIRS time x burned scar, "
              f"{rtr.a:.0f} m grid) --")
        print(f"   real burn in the forecast window: {win.sum() * cell_km2:.1f} km^2; "
              f"sim reached {100 * hit.sum() / max(win.sum(), 1):.0f}% of it")
        if hit.any():
            err = sim_arr[hit] - arr[hit]              # + = sim late, - = sim early
            print(f"   timing error (sim - real): median {np.median(err):+.1f} h, "
                  f"MAE {np.mean(np.abs(err)):.1f} h | sim early on "
                  f"{100 * (err < 0).mean():.0f}% of cells")
            print("   (reconstruction is step-wise between overpasses - the 01:02->10:43 "
                  "gap makes 'sim early' partly an observation artifact)")
        over = (~np.isnan(arr)) & (arr > end_h) & ~np.isinf(sim_arr) & ~sm
        wrong = np.isnan(arr) & ~np.isinf(sim_arr) & ~sm
        print(f"   sim ahead of the real fire (area it only reached later): "
              f"{over.sum() * cell_km2:.1f} km^2 | sim outside the final scar "
              f"(never burned): {wrong.sum() * cell_km2:.1f} km^2")
        metrics["arrival_time"] = {
            "real_window_km2": round(win.sum() * cell_km2, 1),
            "sim_reached_pct": round(100 * hit.sum() / max(win.sum(), 1)),
            "timing_median_h": round(float(np.median(sim_arr[hit] - arr[hit])), 1)
            if hit.any() else None,
            "ahead_of_schedule_km2": round(over.sum() * cell_km2, 1),
            "outside_final_scar_km2": round(wrong.sum() * cell_km2, 1)}
    except Exception as e:
        print(f"(arrival-time reconstruction skipped: {e})")

    import json
    mpath = OUT_HTML.parent / "validation_metrics.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved metrics -> {mpath}")

    # --- overlay map ---
    try:
        import folium
    except ImportError:
        print("(folium not installed - skipping the overlay map)")
        return
    sc = sim.centroid
    ign_ll = gpd.GeoSeries([sc.__class__(*IGN)], crs=2100).to_crs(POINT_CRS).iloc[0]
    fmap = folium.Map(location=[ign_ll.y, ign_ll.x], zoom_start=12, tiles=BASEMAP)

    def poly_layer(geom, name, color, fill_opacity, weight=1, dash=None, show=True):
        folium.GeoJson(
            gpd.GeoSeries([geom.simplify(20)], crs=2100).to_crs(POINT_CRS).to_json(),
            name=name, show=show,
            style_function=lambda _f, c=color, fo=fill_opacity, w=weight, d=dash: {
                "color": c, "weight": w, "fillColor": c, "fillOpacity": fo,
                "dashArray": d}).add_to(fmap)

    from shapely.geometry import box

    from cell2fire_adapter import read_asc_header
    wtr, wnr, wnc, _ = read_asc_header(str(INSTANCE / "Forest.asc"))
    window = box(wtr.c, wtr.f + wnr * wtr.e, wtr.c + wnc * wtr.a, wtr.f)
    poly_layer(window, f"model window ({round((wnc * wtr.a) / 1000)} km - sim cannot "
               "spread beyond it)", "#333", 0.0, weight=1.5, dash="4 7")
    poly_layer(burned, "real burned area (2021 final)", "#555", 0.10)
    if real_end is not None:
        poly_layer(real_end, "real extent at sim end (VIIRS x scar reconstruction)",
                   "#7a00cc", 0.10, weight=2, dash="6 4")
    if seeded:
        poly_layer(seed, "seed front (observed, t0)", "#444", 0.35)
        poly_layer(sim.difference(seed), "sim FORECAST growth", "red", 0.25)
    else:
        poly_layer(sim, f"simulated fire ({hours} h)", "red", 0.25)

    # VIIRS detections: seed-window (gray) vs forecast-window targets (orange) vs later (pale)
    t_seed = pd.to_datetime(SEED_TIME)
    groups = [
        ("VIIRS seed window (input)", v1[v1["t"] <= t_seed], "#666", True),
        ("VIIRS forecast window (target)",
         v1[(v1["t"] > t_seed) & (v1["t"] <= t_end)], "orange", True),
        ("VIIRS after sim end", v1[v1["t"] > t_end], "#e8c8ff", False),
    ] if seeded else [("VIIRS first-day", v1, "orange", True)]
    for name, grp, col, show in groups:
        fg = folium.FeatureGroup(name=f"{name} ({len(grp)} pts)", show=show)
        for _, r in grp.to_crs(POINT_CRS).iterrows():
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=2, color=col,
                                fill=True, fill_opacity=0.8,
                                tooltip=f"{r['t']:%d/%m %H:%M} UTC").add_to(fg)
        fg.add_to(fmap)
    folium.Marker([ign_ll.y, ign_ll.x], popup="ignition 2021-08-05 23:22",
                  icon=folium.Icon(color="black", icon="fire", prefix="fa")).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(OUT_HTML))
    print(f"\nSaved overlay map -> {OUT_HTML}")


if __name__ == "__main__":
    main()
