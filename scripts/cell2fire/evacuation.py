"""Evacuation scenario (STUB fire): a fire inside Δ. Ιστιαίας-Αιδηψού threatens
settlements; the at-risk settlements (+ census-2021 population) are flagged.

This is the EXPOSURE step -- the structured list the LLM will later present and
turn into a recommendation (the exposure -> narration seam).

Data roles:
  * AT-RISK = official Istiaia-Aidipsos settlements (Settlements_Istiaia.geojson,
    with census2021) -> exposure + (later) the evacuation origin.
  * refuges (OSM) + routing + friction -> next step.

Reuses (DRY): buffer_width_grid + blocked_zone from route_with_fire.

STUB: the fire is a synthetic circle, auto-placed inland in the municipality;
the intensity proxy = fuel x slope (PLACEHOLDER). Cell2Fire replaces both later
with no change to this exposure logic.

Outputs (DATA_DIR/Evacuation/):
    scenario.gpkg   layers: fire, blocked_zone, at_risk, safe   (EPSG:2100)
    scenario.html   folium map (red = at-risk, grey = safe)

Run:
    python scripts/cell2fire/evacuation.py
"""

from collections import namedtuple

import geopandas as gpd
from shapely.geometry import Point

from _paths import ATRISK_SETTLEMENTS, DATA_DIR
from route_shortest_path import BASEMAP, POINT_CRS
from route_with_fire import ANALYSIS_CRS, blocked_zone, buffer_width_grid
from settlements import load_atrisk

ATRISK_PATH = ATRISK_SETTLEMENTS   # official layer (kept for reference/compat);
                                   # the ANALYSIS now uses settlements.load_atrisk()
                                   # = official + OSM for the whole study area
NAME_COL = "NAME_OIK"
POP_COL = "census2021"

# --- STUB fire (auto-placed inland in the municipality) ----------------------
# 1.5 km radius -> threatens a settlement cluster without engulfing it (matches
# the original stub-fire size); leaves a meaningful evacuation-routing problem.
FIRE_RADIUS_M = 1500.0
INLAND_PCTL = 0.60   # consider the higher-elevation (inland) settlements for placement
# Optional manual ignition point (lat, lon) in EPSG:4326; None -> auto-place.
FIRE_CENTER_LATLON = None

# A settlement is stored as a POINT, but a village has extent: flag it at-risk if
# it is inside the fire zone OR within this distance of it (village extent +
# margin). Provisional -- calibrate with the expert.
EXPOSURE_MARGIN_M = 300.0

OUT_GPKG = DATA_DIR / "Evacuation_c2f" / "scenario.gpkg"
OUT_HTML = DATA_DIR / "Evacuation_c2f" / "scenario.html"


def auto_fire(atrisk):
    """Synthetic fire that THREATENS a settlement cluster without sitting on top
    of one.

    A circular fire centred ON a settlement merely engulfs it; centred in open
    ground between settlements it threatens the ones in its ring. So we place the
    centre at the midpoint of the closest pair of (inland) settlements that are
    ~2.4x the fire radius apart -- both then fall just outside the flame front:
    inside the at-risk ring (flagged) but not engulfed (so they can be routed).
    Falls back to the inland centroid if no suitable pair exists. A manual
    FIRE_CENTER_LATLON overrides the whole heuristic. Returns (polygon, centre).
    """
    if FIRE_CENTER_LATLON is not None:                  # analyst-supplied override
        centre = gpd.GeoSeries(
            [Point(FIRE_CENTER_LATLON[1], FIRE_CENTER_LATLON[0])], crs=POINT_CRS
        ).to_crs(ANALYSIS_CRS).iloc[0]
        return centre.buffer(FIRE_RADIUS_M), centre

    inland = atrisk[atrisk["H"] >= atrisk["H"].quantile(INLAND_PCTL)] if "H" in atrisk else atrisk
    pts = list(inland.geometry)
    lo, hi, target = 2.1 * FIRE_RADIUS_M, 3.1 * FIRE_RADIUS_M, 2.4 * FIRE_RADIUS_M   # "just outside the ring" band
    best = None
    for i in range(len(pts)):                            # O(n^2) pairwise search -- fine at settlement-count scale
        for j in range(i + 1, len(pts)):
            d = pts[i].distance(pts[j])
            if lo <= d <= hi and (best is None or abs(d - target) < best[0]):
                best = (abs(d - target), pts[i], pts[j])
    if best is None:                                    # fallback: inland centroid
        centre = inland.geometry.union_all().centroid
    else:
        a, b = best[1], best[2]
        centre = Point((a.x + b.x) / 2, (a.y + b.y) / 2)   # midpoint of the chosen pair -> both fall just outside
    return centre.buffer(FIRE_RADIUS_M), centre


def save_html(fire, zone, at_risk, safe):
    try:
        import folium
    except ImportError:
        print("folium not installed - skipping HTML map.")
        return
    fire_w = gpd.GeoSeries([fire], crs=ANALYSIS_CRS).to_crs(POINT_CRS)   # reproject to WGS84 for the web basemap
    centre = fire_w.iloc[0].centroid
    fmap = folium.Map(location=[centre.y, centre.x], zoom_start=11, tiles=BASEMAP)
    if zone is not None:
        folium.GeoJson(gpd.GeoSeries([zone], crs=ANALYSIS_CRS).to_crs(POINT_CRS).to_json(),
                       style_function=lambda _f: {"color": "orange", "weight": 1,
                       "fillColor": "orange", "fillOpacity": 0.2}).add_to(fmap)
    folium.GeoJson(fire_w.to_json(), style_function=lambda _f: {"color": "red",
                   "weight": 1, "fillColor": "red", "fillOpacity": 0.4}).add_to(fmap)

    def dots(gdf, colour, group):
        if not len(gdf):
            return
        fg = folium.FeatureGroup(name=group, show=True)
        for _, r in gdf.to_crs(POINT_CRS).iterrows():     # per-feature reproject for marker placement
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=4, color=colour,
                                fill=True, fill_opacity=0.9,
                                popup=f"{r.get(NAME_COL,'?')} (pop {r.get(POP_COL,'?')})").add_to(fg)
        fg.add_to(fmap)

    dots(safe, "grey", "safe settlements")
    dots(at_risk, "red", "AT-RISK settlements")
    folium.LayerControl(collapsed=False).add_to(fmap)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(OUT_HTML))
    print(f"Saved HTML map -> {OUT_HTML}")


Scenario = namedtuple("Scenario", "atrisk fire centre zone in_zone safe")   # the exposure result, passed on to routing

# --- Cell2Fire fire -----------------------------------------------------------
# Perimeters produced by cell2fire_adapter from a real Cell2Fire run.
C2F_PERIMETERS = DATA_DIR / "Fire" / "cell2fire" / "perimeters.geojson"


def cell2fire_fire(period=None):
    """Cell2Fire fire perimeter (final period, or a given one) + its centroid,
    EPSG:2100. Drop-in replacement for the stub `auto_fire()` - reads the
    `perimeters.geojson` written by `cell2fire_adapter` from a Cell2Fire run.
    """
    gdf = gpd.read_file(C2F_PERIMETERS).to_crs(ANALYSIS_CRS)
    row = gdf.iloc[-1] if period is None else gdf[gdf["period"] == period].iloc[0]   # last period = final scar, or a specific hour
    return row.geometry, row.geometry.centroid


def exposure_for_fire(fire, atrisk, width, grid):
    """Exposure scenario for an ARBITRARY fire perimeter (the per-timestep seam).

    Pure: callers that loop over timesteps (fire_timesteps.py) load `atrisk` and
    the buffer-width grid ONCE and call this per perimeter. zone->settlements via
    the points' R-tree (exact distance only for candidates - same result)."""
    zone = blocked_zone(fire, width, grid)              # fire + variable buffer -> the danger polygon
    if zone is not None:
        try:
            near = atrisk.sindex.query(zone, predicate="dwithin",         # R-tree: candidate points within margin
                                       distance=EXPOSURE_MARGIN_M)
        except (TypeError, ValueError):                  # older geopandas
            near = atrisk.sindex.query(zone.buffer(EXPOSURE_MARGIN_M),    # pre-buffer the zone -> plain intersects works the same
                                       predicate="intersects")
        in_zone = atrisk.iloc[sorted(near)]              # sorted -> stable, reproducible row order
    else:
        in_zone = atrisk.iloc[0:0]                        # no fire footprint this period -> empty at-risk set
    safe = atrisk[~atrisk.index.isin(in_zone.index)]      # everyone else = safe (by exclusion, not a separate query)
    return Scenario(atrisk, fire, fire.centroid, zone, in_zone, safe)


def compute_exposure():
    """Run the Cell2Fire exposure scenario and return its geometries + tables.

    Shared seam (DRY): evacuate_routes.py imports this so the fire, the blocked
    zone and the at-risk set are defined in ONE place. The fire is now the real
    Cell2Fire perimeter (replacing the stub `auto_fire()` circle).
    """
    atrisk = load_atrisk()             # official + OSM, whole study area
    fire, _centre = cell2fire_fire()   # the real Cell2Fire fire
    width, grid = buffer_width_grid()
    return exposure_for_fire(fire, atrisk, width, grid)


def main():
    atrisk, fire, centre, zone, in_zone, safe = compute_exposure()
    have_pop = int(atrisk[POP_COL].notna().sum()) if POP_COL in atrisk else 0
    print(f"At-risk layer: {len(atrisk)} official settlements "
          f"({POP_COL} present for {have_pop})")
    print(f"Fire (stub) centre ~({centre.x:.0f}, {centre.y:.0f}) Greek Grid, "
          f"radius {FIRE_RADIUS_M:.0f} m")
    pop_at_risk = int(in_zone[POP_COL].fillna(0).sum()) if POP_COL in in_zone else 0

    print(f"\nEXPOSURE -- {len(in_zone)} settlements within {EXPOSURE_MARGIN_M:.0f} m "
          f"of the fire zone, ~{pop_at_risk:,} people (census2021):")
    cols = [c for c in (NAME_COL, POP_COL) if c in in_zone.columns]
    for _, r in in_zone[cols].sort_values(POP_COL, ascending=False).iterrows():
        # names may not render in this console, but they are correct in the file
        print(f"   pop {str(r.get(POP_COL,'?')):>6}   {r.get(NAME_COL,'?')}")

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    gpd.GeoDataFrame({"kind": ["fire"]}, geometry=[fire], crs=ANALYSIS_CRS).to_file(
        OUT_GPKG, layer="fire", driver="GPKG")
    if zone is not None:
        gpd.GeoDataFrame({"kind": ["blocked_zone"]}, geometry=[zone], crs=ANALYSIS_CRS).to_file(
            OUT_GPKG, layer="blocked_zone", driver="GPKG")
    if len(in_zone):
        in_zone.to_file(OUT_GPKG, layer="at_risk", driver="GPKG")
    if len(safe):
        safe.to_file(OUT_GPKG, layer="safe", driver="GPKG")
    print(f"\nSaved GeoPackage -> {OUT_GPKG}")

    save_html(fire, zone, in_zone, safe)


if __name__ == "__main__":
    main()
