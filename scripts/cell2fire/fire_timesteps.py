"""Phase 4.5 - THESIS CORE: dynamic evacuation over the REAL fire's timesteps.

For EACH hourly Cell2Fire perimeter (`perimeters.geojson`, period 0..N) the whole
evacuation problem is re-solved: exposure (which settlements are at risk NOW) ->
friction routing (each to its nearest safe refuge on the fire-cut, friction-
weighted graph). This demonstrates the thesis claim end-to-end: as the fire
advances, roads close, routes lengthen and settlements get cut off.

Reuses, with no duplication (DRY):
  * exposure          evacuation.exposure_for_fire (atrisk + width grid loaded ONCE)
  * routing           evacuate_routes.route_to_refuges (graph loaded ONCE, copied per step)
  * buffer/intensity  route_with_fire.buffer_width_grid

Outputs (DATA_DIR/Evacuation_c2f/):
    timestep_evacuation.gpkg  layers (all periods stacked, `period` column):
                              at_risk (status/refuge/route_km), routes  (EPSG:2100)
    timestep_summary.json     per-period stats: time, at-risk count + population,
                              routed / cut_off / impacted, edges removed, longest km

Run (after the Cell2Fire run + adapter):
    python scripts/cell2fire/fire_timesteps.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from _paths import DATA_DIR
from evacuation import POP_COL, exposure_for_fire
from settlements import load_atrisk
from evacuate_routes import route_to_refuges
from route_shortest_path import load_graph
from route_with_fire import ANALYSIS_CRS, buffer_width_grid

# Scenario override: set WFEDS_SCENARIO_DIR to a run folder holding the scenario's
# perimeters.geojson (+ optionally its own instance/Weather.csv) and the evacuation
# outputs are written back into it - so a scenario run never clobbers the canonical
# outputs (and each run gets its own folder). Default = canonical.
_SCEN = os.environ.get("WFEDS_SCENARIO_DIR")
if _SCEN:
    _SCEN = Path(_SCEN)
    PERIMS = _SCEN / "perimeters.geojson"             # THIS run's hourly perimeters, not the canonical ones
    OUT_GPKG = _SCEN / "timestep_evacuation.gpkg"
    OUT_JSON = _SCEN / "timestep_summary.json"
    _w = _SCEN / "instance" / "Weather.csv"      # the run's own weather (t0!)
    WEATHER = _w if _w.exists() else (                # fall back to canonical weather if the run has none of its own
        DATA_DIR / "Fire" / "cell2fire" / "instance" / "Weather.csv")
else:
    PERIMS = DATA_DIR / "Fire" / "cell2fire" / "perimeters.geojson"
    OUT_GPKG = DATA_DIR / "Evacuation_c2f" / "timestep_evacuation.gpkg"
    OUT_JSON = DATA_DIR / "Evacuation_c2f" / "timestep_summary.json"
    WEATHER = DATA_DIR / "Fire" / "cell2fire" / "instance" / "Weather.csv"


def main():
    perims = gpd.read_file(PERIMS).to_crs(ANALYSIS_CRS).sort_values(   # the hourly fire polygons, period 0..N
        "period").reset_index(drop=True)
    w = pd.read_csv(WEATHER)
    t0 = pd.to_datetime(w["datetime"].iloc[0])          # sim start time -> period NN becomes a real UTC timestamp
    atrisk = load_atrisk()             # official + OSM, whole study area
    width, grid = buffer_width_grid()  # fuel x slope buffer-width raster -- LOADED ONCE, reused every period
    graph = load_graph()               # the road network -- LOADED ONCE; route_to_refuges COPIES it per step

    at_all, rt_all, summary = [], [], []
    print(f"{'t':>3} {'UTC':<12} {'fire_km2':>8} {'at_risk':>7} {'pop':>7} "
          f"{'ok':>3} {'cut':>4} {'imp':>4} {'edges':>6} {'longest_km':>10}")
    for _, prow in perims.iterrows():                   # ---- re-solve the WHOLE evacuation problem, once per hour ----
        p = int(prow["period"])
        when = t0 + pd.Timedelta(hours=p)
        scen = exposure_for_fire(prow.geometry, atrisk, width, grid)   # who's at risk from THIS hour's fire shape
        res = route_to_refuges(scen, graph=graph)        # route each at-risk settlement on the fire-cut graph
        iz, rts = res.in_zone.copy(), res.routes.copy()
        iz["period"], rts["period"] = p, p                # tag both layers so all hours can be stacked into one file
        at_all.append(iz)
        if len(rts):
            rt_all.append(rts)

        pop = int(iz[POP_COL].fillna(0).sum()) if POP_COL in iz else 0
        n_ok = int((iz["status"] == "ok").sum())          # routed to a refuge
        n_cut = int((iz["status"] == "cut_off").sum())    # at risk, no route left
        n_imp = int((iz["status"] == "impacted").sum())   # inside the fire perimeter itself
        longest = round(float(rts["length_m"].max()) / 1000, 2) if len(rts) else None
        summary.append({
            "period": p, "time_utc": when.isoformat(),
            "fire_km2": round(prow.geometry.area / 1e6, 2),
            "at_risk": len(iz), "population": pop,
            "routed": n_ok, "cut_off": n_cut, "impacted": n_imp,
            "edges_removed": int(res.n_removed), "longest_route_km": longest,
        })
        print(f"{p:>3} {when:%d/%m %H:%M}  {prow.geometry.area / 1e6:>8.1f} "
              f"{len(iz):>7} {pop:>7,} {n_ok:>3} {n_cut:>4} {n_imp:>4} "
              f"{res.n_removed:>6} {longest if longest is not None else '--':>10}")

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()                                # start clean -- to_file(layer=...) would otherwise APPEND
    gpd.GeoDataFrame(pd.concat(at_all, ignore_index=True), crs=ANALYSIS_CRS).to_file(   # all periods, one "at_risk" layer
        OUT_GPKG, layer="at_risk", driver="GPKG")
    if rt_all:
        gpd.GeoDataFrame(pd.concat(rt_all, ignore_index=True), crs=ANALYSIS_CRS).to_file(   # all periods, one "routes" layer
            OUT_GPKG, layer="routes", driver="GPKG")
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\nSaved -> {OUT_GPKG} (layers: at_risk, routes; `period` column)")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
