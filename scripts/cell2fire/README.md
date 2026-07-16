# Deterministic pipeline (`scripts/cell2fire/`)

**This is THE deterministic geospatial core** of the thesis: a real **Cell2Fire**
fire simulation coupled to dynamic road-network evacuation. (The old synthetic-circle
`stub/` snapshot was removed 2026-07-03 - this folder is now the single pipeline.)

Cell2Fire itself is built and run inside the Docker image (on the legacy
pre-Docker path it runs in WSL/Ubuntu; engine mods in `engine_patch/`). Shared one-off
data acquisition lives in `scripts/data_prep/`. All analysis is EPSG:2100 (Greek Grid, m).

## Components & data flow

The fire is modelled **free-burning** (worst credible case = the evacuation planning
basis); suppression/evacuation *measures* are reasoned by the LLM agent layer, not the
engine. Stages, in order (▶ = run directly, · = library imported by others):

**A. Data prep** (`scripts/data_prep/`, run once) → road graph, settlements, refuges.

**B. Fire simulation + adapter**
- ▶ `build_cell2fire_instance.py` - crop the window, write the Cell2Fire instance
  (`Forest.asc` fuel, `Weather.csv`, ignition + `InitialBurned.csv`). Uses
  `fwi.py` + `meteo.py`. **ONE seeding rule** (`seed_cells_from_points`): every
  observed point - user pin OR VIIRS detection - is buffered by half a VIIRS
  pixel (187.5 m), the buffers are unioned, rasterized, and the burnable cells
  become the seed; the `north_evia_2021` scenario auto-seeds the VIIRS
  detections up to its `seed_time` through the same function.
- *(WSL)* Cell2Fire → `Grids/ForestGrid<NN>.csv` (+ `Intensity<NN>.csv` with
  `--out-intensity`, the `cell2fire_intensity.patch` diagnostic).
- ▶ `cell2fire_adapter.py <inst> <out> EPSG:2100` - grids → `perimeters.geojson`
  (per-hour fire) + `isochrones.geojson` (outer front + fireline-intensity class).
- · `fwi.py`, `meteo.py` - FWI codes / ERA5 wind (imported by the builder).

**C. Evacuation** (the coupling - fire changes road availability → routes change)
- · `route_with_fire.py` - fire→road-blocking helpers (`buffer_width_grid`,
  `blocked_zone`, `fuel_hazard_overlay`). Library only.
- · `route_shortest_path.py` - routing primitives (`load_graph`, `route_to_line`, …).
- · `evacuation.py` - exposure (`exposure_for_fire`: which settlements are at risk).
- · `evacuate_routes.py` - friction routing (`route_to_refuges`: each to a safe refuge).
- ▶ `fire_timesteps.py` - **THESIS CORE**: re-solve the whole evacuation at EACH hourly
  perimeter → `timestep_evacuation.gpkg` + `timestep_summary.json`.

**D. Visualization / validation** (evaluation scripts live in their OWN package
`scripts/validation/`, fully decoupled from the live loop - the live pipeline
never imports validation code)
- ▶ `scripts/validation/real_progression.py` - the real fire's arrival-time surface
  (burned scar WHERE × VIIRS WHEN), the validation ground truth. Run once manually:
  also exports `Real_Fire_Data/real_progression_hourly.geojson` (the **data seam** -
  the dashboard reads this static file for its real-fire overlay, never this module).
- ▶ `scripts/validation/validate_overlay.py` - the per-pass VIIRS validation protocol
  → `validation_metrics.json` + per-pass table/report/charts/GIS bundle.
  Manual/offline (not part of the live `run_scenario.py` loop).
- ▶ `visualize_fire.py` - the timeline dashboard `fire_timesteps.html`. Reads the two
  validation ARTIFACTS above (hourly extents + metrics) as data, if present.

**E. Orchestration** (the seam the LLM agent actually calls)
- ▶ `run_scenario.py` - ONE call = build_instance → Cell2Fire engine (WSL) →
  `cell2fire_adapter.py` → `fire_timesteps.py` → `visualize_fire.py` → `snapshot_map.py`
  → `result.json` (the LLM-facing summary). Every run isolated under
  `Fire/cell2fire/runs/<run_id>/` - never touches the canonical outputs.
  *(An older `run_workflow.py`/`export_geojson.py` JSON-contract seam was
  removed 2026-07-09 - confirmed unused by anything, superseded by this.)*

**F. Shared** - `_paths.py` (resolves `DATA_DIR` from the OneDrive env var + dataset paths).

## Run
```
python scripts/cell2fire/build_cell2fire_instance.py   # writes InitialBurned.csv too (the seeded footprint)
# (WSL) python -m cell2fire.main --input-instance-folder <inst> --output-folder <out> \
#        --Fire-Period-Length 1 --gridsStep 60 --out-intensity --InitialBurned <inst>/InitialBurned.csv ...
python scripts/cell2fire/cell2fire_adapter.py <inst> <out> EPSG:2100
python scripts/cell2fire/fire_timesteps.py             # per-timestep evacuation
python scripts/validation/validate_overlay.py <run_dir>  # metrics + overlay map (manual/offline)
python scripts/cell2fire/visualize_fire.py             # timeline dashboard
```

Scenario runs (e.g. a different fire) can be isolated via `WFEDS_SCENARIO_DIR` (read by
`fire_timesteps.py` + `visualize_fire.py`) so they never overwrite the canonical outputs.
Operationally, `run_scenario.py` does exactly this in one call - see Stage E above.

(The author's full design/decision notes are kept in a local project vault,
outside this repository.)
