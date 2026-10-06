"""Repository-root conftest.py -- MUST stay plain top-level code, not a fixture.

Why: pytest COLLECTS (imports) every test module before running ANY fixture, even
session-scoped ones. `scripts/cell2fire/_paths.py` and `scripts/data_prep/_paths.py`
both do `DATA_DIR = get_data_dir()` as a bare module-level statement, which runs the
FIRST time either module is imported and raises RuntimeError if WFEDS_DATA_DIR /
OneDriveCommercial / OneDrive can't be resolved to a real, existing folder. By the
time a fixture-based fix would run, collection has already imported (and crashed)
these modules. conftest.py files, by contrast, are imported during pytest's
plugin-discovery phase, which happens BEFORE collection of anything below them in
the directory tree -- root-level conftest.py is discovered first of all -- so setting
the env var here, as plain code, actually beats the import.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TEST_DATA_DIR = _ROOT / "tests" / "fixtures" / "data"
_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# setdefault (not `=`): a developer with a real OneDrive folder configured can still
# override by exporting WFEDS_DATA_DIR themselves before running pytest.
os.environ.setdefault("WFEDS_DATA_DIR", str(_TEST_DATA_DIR))

# --- Environment landmine found while building tests/helpers/geodata_factories.py ---
# On at least this dev machine, a machine-wide PROJ_LIB env var (set by a PostgreSQL /
# PostGIS install: "...\PostgreSQL\<ver>\share\contrib\postgis-<ver>\proj") shadows
# rasterio's own bundled, version-matched PROJ database. That makes ANY
# `rasterio.open(..., crs=...)` WRITE (e.g. tests/helpers/geodata_factories.py's
# make_tiny_raster) raise `rasterio.errors.CRSError: The WKT could not be parsed.`
# even for an ordinary EPSG code like "EPSG:2100" -- GDAL's own PROJ context honours
# PROJ_LIB, unlike geopandas/shapely's pyproj, which ships its own bundled data and is
# unaffected (`gpd.GeoSeries(...).to_crs(...)` works fine either way). This is a
# machine/environment issue, not a bug in any WFEDS script (production code only ever
# READS pre-existing rasters and never hit this), but it silently breaks raster-writing
# test helpers unless fixed before rasterio's/GDAL's PROJ context is ever touched --
# i.e. here, in the root conftest.py, before collection imports anything.
os.environ.pop("PROJ_LIB", None)

# Both scripts/cell2fire/_paths.py and scripts/data_prep/_paths.py are modules named
# `_paths` (byte-identical files). Only ONE can win the `import _paths` ambiguity --
# whichever directory lands first on sys.path. We want cell2fire before data_prep: it
# is imported by far more of the test suite.
#
# NB: repeatedly doing `sys.path.insert(0, x)` in a plain loop would put the LAST
# item inserted FIRST (each insert(0, ...) shifts everything else, including the
# previous insertion, one slot to the right) -- that would silently reverse the
# intended priority and hand the ambiguity to data_prep instead. Insert at an
# increasing index so the listed order is preserved.
for _i, _p in enumerate(("scripts/cell2fire", "scripts/data_prep", "scripts/agent",
                         "scripts/validation")):
    sys.path.insert(_i, str(_ROOT / _p))
