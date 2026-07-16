"""Resolve the WFEDS data directory in a machine-independent way.

The project data lives in OneDrive so it syncs across the two work computers,
but the absolute path is NOT portable: the Windows user folder differs per
machine, so a hard-coded
``C:/Users/<name>/...`` breaks on the other computer. OneDrive, however, exports
its synced root as an environment variable on every machine where it is set up,
so we resolve the data folder from that instead of hard-coding a username.

Resolution order (first hit wins):
  1. ``WFEDS_DATA_DIR``     explicit override -- full path to the Data folder
  2. ``OneDriveCommercial`` set by OneDrive for Business
  3. ``OneDrive``           generic OneDrive root
For 2/3 the relative tail ``Working progress/Scripts/Data`` is appended.

Import this from any script:
    from _paths import DATA_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

_DATA_REL = Path("Working progress") / "Scripts" / "Data"   # fixed sub-path to the Data workspace inside OneDrive


def get_data_dir() -> Path:
    """Return the absolute path to the OneDrive ``Scripts/Data`` folder.

    Works on any machine where OneDrive is configured, regardless of the
    Windows username. **Verifies the folder actually exists on disk** and fails
    with a clear, actionable message if not -- otherwise a missing/unsynced
    OneDrive would surface much later as a cryptic FileNotFoundError deep inside
    the first data read. Raises RuntimeError (never returns a bad path).
    """
    override = os.environ.get("WFEDS_DATA_DIR")             # 1st choice: explicit override, if you set it
    if override:
        path, how = Path(override), "the WFEDS_DATA_DIR override"
    else:                                                   # else: find OneDrive's root from the system
        path = how = None
        for var in ("OneDriveCommercial", "OneDrive"):      # Business account first, then generic OneDrive
            root = os.environ.get(var)
            if root:
                path, how = Path(root) / _DATA_REL, f"the {var} environment variable"   # OneDrive root + Data sub-path
                break

    if path is None:                                        # no env var found -> stop with a clear message
        raise RuntimeError(
            "Could not locate the WFEDS data directory: none of WFEDS_DATA_DIR / "
            "OneDriveCommercial / OneDrive is set.\n"
            "Fix: sign in to OneDrive, or set WFEDS_DATA_DIR to the full path of\n"
            "  '<OneDrive>/Working progress/Scripts/Data'."
        )

    if not path.exists():                                   # path built but folder not on disk (OneDrive not synced)
        raise RuntimeError(
            f"The WFEDS data directory does not exist on disk:\n"
            f"  {path}\n"
            f"(resolved from {how}.)\n"
            "The path is built correctly but the folder is missing -- most likely "
            "OneDrive is signed in but this folder has not synced/downloaded on "
            "this machine (or its files are set to 'online-only').\n"
            "Fix: open OneDrive and let 'Working progress/Scripts/Data' sync, or "
            "point WFEDS_DATA_DIR at wherever the Data folder actually is."
        )

    return path                                             # verified, absolute Data path


DATA_DIR = get_data_dir()                                   # resolve the workspace ONCE, at import time


# --- Canonical dataset paths (single source of truth) ------------------------
# The project's data catalog: if the Data folders move, edit ONLY here (not 6 scripts).
STUDY_AREA = DATA_DIR / "Boundaries" / "Study_Area.geojson"       # study-area boundary polygon (AOI / clip extent)
ROAD_GRAPH = DATA_DIR / "Roads" / "road_graph.graphml"            # road network as a routable graph (nodes + edges)
DEM = DATA_DIR / "DEM" / "DEM.tif"                                # elevation raster (terrain)
SLOPE = DATA_DIR / "DEM" / "Slope_Degrees.tif"                    # slope raster, in degrees
FUEL = DATA_DIR / "Forest_Fuel" / "SVM_HUA.tif"                   # land-cover/fuel raster (5-class SVM) -- drives flammability
ATRISK_SETTLEMENTS = DATA_DIR / "Settlements" / "Settlements_Istiaia.geojson"  # official settlements + census2021 = evacuation sources
OSM_SETTLEMENTS = DATA_DIR / "Settlements" / "settlements.gpkg"   # OSM settlements (Mantoudi) = refuge / destination candidates
