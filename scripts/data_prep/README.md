# Data preparation (shared, run-once)

Acquires the geospatial inputs the rest of the pipeline reads from `DATA_DIR`
(OneDrive). Run once; not part of the stub-vs-Cell2Fire split, so it lives here on
its own rather than in `../stub/` or `../cell2fire/`.

- `_paths.py` - resolves `DATA_DIR` machine-independently (from the OneDrive env var).
- `download_road_network.py` - OSMnx road graph for the WHOLE study area (both municipalities).
- `download_settlements.py` - OSM settlements for the Mantoudi-Limni-Agia Anna
  municipality only (the Istiaia side comes from the official ELSTAT layer);
  written cleaned, they ARE the evacuation refuge candidates. The official +
  Mantoudi layers are unioned at read time by `cell2fire/settlements.py` (plain
  concatenation - the two cover disjoint ground, so no de-dup step is needed).

Run (from the repo root):

    python scripts/data_prep/download_road_network.py
    python scripts/data_prep/download_settlements.py

Vault: `LLM-WFEDS/Datasets.md`.
