"""The ONE settlements source for the whole pipeline (maps AND analysis).

Two layers that cover DISJOINT ground, so the union is a plain concatenation -
no distance de-dup needed (unlike the old whole-study-area OSM pull):
  * the OFFICIAL Istiaia-Aidipsos layer (Settlements_Istiaia.geojson) - carries
    the census2021 population;
  * the OSM places of the Mantoudi-Limni-Agia Anna municipality
    (settlements.gpkg, built CLEANED by download_settlements.py) - the study
    area's other half; these are ALSO the evacuation REFUGE candidates.

`all_settlements()`  -> columns  name, pop, geometry            (display)
`load_atrisk()`      -> columns  NAME_OIK, census2021, geometry (analysis -
                        the schema evacuation.py / fire_timesteps.py expect)
`refuges()`          -> columns  name, place_type, population, source, geometry
                        (the OSM/Mantoudi side only = evacuation destinations)
"""

import geopandas as gpd
import pandas as pd

from _paths import ATRISK_SETTLEMENTS, OSM_SETTLEMENTS

ANALYSIS_CRS = "EPSG:2100"


def _official():
    """The ELSTAT Istiaia layer, normalised to name/pop/geometry."""
    st = gpd.read_file(ATRISK_SETTLEMENTS).to_crs(ANALYSIS_CRS)
    return gpd.GeoDataFrame({
        "name": [str(n).strip() for n in st["NAME_OIK"]],   # official ELSTAT name field -> the common "name" schema
        "pop": st["census2021"].tolist(),                    # official census population
    }, geometry=st.geometry, crs=ANALYSIS_CRS)


def refuges():
    """Refuge CANDIDATES = the OSM (Mantoudi-side) settlements, ready-cleaned by
    download_settlements.py. Columns: name, place_type, population, source,
    geometry."""
    return gpd.read_file(OSM_SETTLEMENTS).to_crs(ANALYSIS_CRS)   # already clean (see download_settlements.py) - no further work needed


def all_settlements():
    """Official (Istiaia) + OSM (Mantoudi) settlements. Plain union - the two
    layers are geographically disjoint. Columns: name, pop, geometry."""
    official = _official()
    try:
        osm = refuges()
        osm2 = gpd.GeoDataFrame({
            "name": osm["name"].astype(str).str.strip(),
            "pop": osm["population"] if "population" in osm.columns else pd.NA,   # OSM schema -> the same name/pop columns as official
        }, geometry=osm.geometry, crs=ANALYSIS_CRS)
        merged = pd.concat([official, osm2], ignore_index=True)   # DISJOINT ground by admin boundary -> plain concat, no spatial de-dup
        return gpd.GeoDataFrame(merged, geometry="geometry", crs=ANALYSIS_CRS)
    except Exception as e:
        # Loud, not silent: falling back to official-only would quietly make the
        # Mantoudi side invisible again (the exact bug this module fixes).
        print(f"WARNING settlements: OSM layer unavailable ({e}); "
              f"using the official Istiaia layer only.")
        return official


def load_atrisk():
    """The settlements layer FOR THE ANALYSIS (exposure/evacuation), in the
    schema the pipeline has always used: NAME_OIK + census2021 (numeric or NaN,
    parsed from the OSM population where present)."""
    st = all_settlements()
    return gpd.GeoDataFrame({
        "NAME_OIK": st["name"],                               # re-expose under the LEGACY column names -- downstream
        "census2021": pd.to_numeric(                          # scripts (evacuation.py, fire_timesteps.py) never had to change
            st["pop"].astype(str).str.replace(",", "", regex=False),   # OSM population can arrive as "1,234" -> strip before parsing
            errors="coerce"),                                  # unparseable/missing population -> NaN, not a crash
    }, geometry=st.geometry, crs=ANALYSIS_CRS)
