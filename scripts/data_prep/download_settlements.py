"""Download the OSM settlements of the Mantoudi-Limni-Agia Anna municipality
(the study area's OTHER half) and save them CLEANED as a GeoPackage -- these ARE
the evacuation refuge candidates, no separate build step needed.

Why only one municipality: the official ELSTAT layer
(`Settlements_Istiaia.geojson`) already covers Istiaia-Aidipsos WITH census
population. Pulling OSM for the WHOLE study area duplicated every Istiaia village
and forced a fragile "is this the same place?" distance de-dup. Restricting the
OSM pull to the Mantoudi municipality makes the two layers cover DISJOINT ground
by administrative boundary, so the union (`cell2fire/settlements.py`) is a plain
concatenation and no de-dup is needed.

Reuses the boundary loader from download_road_network (DRY), passing a
single-municipality filter. NB the ROADS download still uses BOTH municipalities
(its default filter is None), so cross-boundary evacuation/exit routes stay intact.

Cleaning is done HERE so downstream reads a ready schema: reduce each feature to a
representative point, keep the best name (name:el -> name), parse population to a
number, drop unnamed places.

Output (EPSG:2100), columns: name, place_type, population, source
    Data/Settlements/settlements.gpkg   layer "settlements"

Run:
    python scripts/data_prep/download_settlements.py
"""

import geopandas as gpd
import osmnx as ox
import pandas as pd

from _paths import OSM_SETTLEMENTS                       # output path (data catalog)
from download_road_network import _atomic_write, load_boundary_4326   # DRY: reuse the safe-write + boundary loader

ANALYSIS_CRS = "EPSG:2100"                               # output in metres (Greek Grid)
MUNICIPALITY_FILTER = ("Municipality", "Mantoudi - Limni - Aghia Anna Municipality")   # pull OSM only for the half NOT in the ELSTAT layer
PLACE_TAGS = {"place": ["city", "town", "village", "hamlet", "isolated_dwelling"]}      # inhabited place types (excludes locality/suburb/quarter)

OUT_GPKG = OSM_SETTLEMENTS                               # single source of truth (scripts/_paths.py)


def _first(gdf, *names):
    """First present column among `names`, else an all-NA Series."""
    for n in names:
        if n in gdf.columns:
            return gdf[n]                                # first column that exists -> return it whole
    return pd.Series(pd.NA, index=gdf.index)             # none present -> all-NA column


def _coalesce(gdf, *names):
    """PER-ROW first non-empty value across `names` (not first-present-column).

    E.g. prefer `name:el` where a settlement has it, else fall back to `name` for
    that same row. (`_first` returns a whole column, which would drop every row
    whose preferred tag is missing.)"""
    out = pd.Series(pd.NA, index=gdf.index, dtype="string")
    for n in names:
        if n in gdf.columns:
            col = gdf[n].astype("string").str.strip()
            out = out.where(out.notna() & (out.str.len() > 0), col)   # keep what's set, fill blanks from this column (SQL COALESCE)
    return out


def main():
    aoi = load_boundary_4326(MUNICIPALITY_FILTER)        # AOI = just the Mantoudi municipality, in WGS84
    print("Downloading OSM settlements (place=city/town/village/hamlet/...) "
          "within the Mantoudi-Limni-Agia Anna municipality...")
    try:
        gdf = ox.features_from_polygon(aoi, tags=PLACE_TAGS)   # download OSM place features inside the AOI
    except Exception as e:
        raise RuntimeError(                              # wrap network failures clearly; existing file untouched
            "Failed to download settlements from OpenStreetMap -- usually no "
            "internet, the Overpass API being down, or rate-limiting. The existing "
            "settlements file was left untouched; just re-run when the connection "
            f"is back.\nOriginal error: {type(e).__name__}: {e}"
        ) from e

    gdf = gdf.to_crs(ANALYSIS_CRS).reset_index()         # reproject to Greek Grid (metres)
    pts = gdf.geometry.representative_point()            # polygon -> a point guaranteed INSIDE it (not centroid)

    name = _coalesce(gdf, "name:el", "name")             # best name per row: Greek if present, else default name
    out = gpd.GeoDataFrame({
        "name": name,
        "place_type": _first(gdf, "place"),              # OSM place class (village/hamlet/...)
        "population": pd.to_numeric(_first(gdf, "population"),
                                    errors="coerce").astype("Int64"),   # population -> number (bad values -> NA)
        "source": "osm",
    }, geometry=pts, crs=ANALYSIS_CRS)
    before = len(out)
    out = out[out["name"].notna() & (out["name"].str.len() > 0)].reset_index(drop=True)   # drop unnamed places
    dropped = before - len(out)

    _atomic_write(OUT_GPKG, lambda p: out.to_file(p, driver="GPKG", layer="settlements"))   # safe write to GeoPackage

    by_type = out["place_type"].value_counts().to_dict()
    print(f"Saved {len(out)} settlements -> {OUT_GPKG}")
    print(f"  by type: {by_type}")
    print(f"  with population: {int(out['population'].notna().sum())}")
    if dropped:
        print(f"  dropped {dropped} unnamed place(s)")


if __name__ == "__main__":
    main()
