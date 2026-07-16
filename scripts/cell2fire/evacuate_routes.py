"""Phase 4: route every AT-RISK settlement to its nearest SAFE refuge, on a road
network that is (a) CUT where it crosses the fire, and (b) made costlier the
closer it runs to the fire (friction).

This is the step that turns the exposure list into an actionable recommendation
("evacuate Voutas -> Asmini, 7.2 km"). The LLM (Phase 5) will narrate it.

Reuses, with no duplication (DRY):
  * exposure scenario (fire + blocked zone + at-risk set)  from evacuation.compute_exposure
  * routing primitives (graph load, snap, route->line)     from route_shortest_path

Friction model (resolves the earlier "remove vs penalise" question as BOTH, split
by how close to the fire a road is):
  * edges crossing the FIRE FRONT (the perimeter itself) -> REMOVED (impassable --
                                            you cannot drive through the flames);
  * every surviving edge -> PENALISED, weight = length x (1 + ...), the penalty
                                            highest at the flame edge and fading
                                            linearly to 0 at FRICTION_BAND_M.
The buffer around the fire is therefore PASSABLE but very costly, so a settlement
right beside the fire can still escape (through threatened roads), while routing
still keeps its distance from the fire wherever a cooler alternative exists. Only
a settlement whose roads all cross the flame front is truly CUT OFF.

Refuges = OSM settlements (settlements.refuges(), the Mantoudi side), kept only if
at least SAFE_MARGIN_M clear of the blocked zone. For each at-risk origin we run one Dijkstra on the
friction-weighted graph and pick the cheapest reachable refuge; if none is
reachable the settlement is CUT OFF (critical).

Outputs (DATA_DIR/Evacuation/):
    evacuation_routes.gpkg   layers: fire, blocked_zone, at_risk, refuges,
                             routes   (EPSG:2100)
    evacuation_routes.html   folium map

Run:
    python scripts/cell2fire/evacuate_routes.py
"""

from collections import namedtuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from _paths import DATA_DIR
from evacuation import NAME_COL, POP_COL, compute_exposure
from route_shortest_path import BASEMAP, POINT_CRS, load_graph, route_to_line
from route_with_fire import fuel_hazard_overlay
from settlements import refuges as osm_refuges

ANALYSIS_CRS = "EPSG:2100"

# --- Friction (PLACEHOLDER, uncalibrated -- tune with the expert) -------------
FRICTION_MAX = 4.0        # an edge at the flame front costs (1+MAX)x its length
FRICTION_BAND_M = 1500.0  # penalty fades linearly to 0 this far from the fire front
SAFE_MARGIN_M = 500.0     # a refuge must be at least this clear of the blocked zone
EVAC_WEIGHT = "evac_w"    # per-edge friction-weighted cost attribute

OUT_GPKG = DATA_DIR / "Evacuation_c2f" / "evacuation_routes.gpkg"
OUT_HTML = DATA_DIR / "Evacuation_c2f" / "evacuation_routes.html"


# --- Friction-weighted, fire-cut graph ---------------------------------------
def friction_graph(graph, fire):
    """Copy of `graph`: edges crossing the FIRE FRONT are REMOVED (impassable);
    every surviving edge gets `EVAC_WEIGHT` = length x proximity penalty (highest
    at the flame edge, fading to 0 at FRICTION_BAND_M). The buffer is thus passable
    but costly -- a settlement beside the fire can still escape, but a route only
    runs close to the flames when there is no cooler alternative.

    fire->roads via the edges' R-tree (built automatically by GeoPandas): the
    exact distance is computed ONLY for edges inside the friction band - beyond
    it the penalty is 0 by definition, so brute-forcing all ~15k edges against a
    large perimeter (the 2026-07-02 hang) bought nothing. Identical result."""
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()   # graph -> attribute table (one row per edge)
    length = edges.length.to_numpy(dtype=float)
    try:
        near = edges.sindex.query(fire, predicate="dwithin",          # R-tree: only edges within the friction band...
                                  distance=FRICTION_BAND_M)
    except (TypeError, ValueError):                      # older geopandas
        near = edges.sindex.query(fire.buffer(FRICTION_BAND_M),       # ...pre-buffer the fire so plain intersects works the same
                                  predicate="intersects")
    penalty = np.ones(len(edges))                          # default: no penalty (far from the fire)
    if len(near):
        d_near = edges.geometry.iloc[near].distance(fire).to_numpy()   # exact distance -- ONLY for the near candidates
        penalty[near] = 1.0 + FRICTION_MAX * np.clip(
            1.0 - d_near / FRICTION_BAND_M, 0.0, 1.0)      # linear fade: 1+MAX at the front -> 1 at FRICTION_BAND_M
    cost = length * penalty                                # the friction-weighted edge cost
    blocked = np.zeros(len(edges), dtype=bool)
    blocked[edges.sindex.query(fire, predicate="intersects")] = True   # edges that physically cross the fire polygon

    gf = graph.copy()                                      # never mutate the caller's graph -- fire_timesteps.py reuses it
    drop = []
    for i in range(len(edges)):
        u, v, k = int(edges.u[i]), int(edges.v[i]), int(edges.key[i])
        if blocked[i]:
            drop.append((u, v, k))                          # crosses the flame front -> remove entirely (impassable)
        else:
            gf[u][v][k][EVAC_WEIGHT] = float(cost[i])        # survives -> stamp its friction cost for Dijkstra
    gf.remove_edges_from(drop)
    return gf, len(drop)


# --- Nearest safe refuge ------------------------------------------------------
def snap_nodes(graph, gdf):
    """Nearest graph node for each point; returns {node: row_index} keeping the
    closest point per node (graph CRS == gdf CRS == EPSG:2100)."""
    nodes, dists = ox.distance.nearest_nodes(                # snap every point onto the road network
        graph, X=gdf.geometry.x.values, Y=gdf.geometry.y.values, return_dist=True)
    best = {}
    for idx, node, d in zip(gdf.index, np.atleast_1d(nodes), np.atleast_1d(dists)):
        if node not in best or d < best[node][1]:            # two points can snap to the SAME node -> keep the closer one
            best[node] = (idx, d)
    return {n: v[0] for n, v in best.items()}


def nearest_refuge(graph, origin_node, refuge_nodes):
    """Cheapest reachable refuge node from `origin_node` on the friction graph.

    Returns (refuge_node, cost, node_path) or None if none is reachable.
    """
    dist, paths = nx.single_source_dijkstra(graph, origin_node, weight=EVAC_WEIGHT)   # ONE Dijkstra reaches every refuge at once
    best = None
    for rn in refuge_nodes:
        c = dist.get(rn)                                      # None -> unreachable (fire-cut off from this origin)
        if c is not None and (best is None or c < best[1]):
            best = (rn, c, paths[rn])                          # keep the cheapest reachable refuge seen so far
    return best


# --- Output -------------------------------------------------------------------
def save_html(fire, zone, at_risk, refuges, routes):
    try:
        import folium
    except ImportError:
        print("folium not installed - skipping HTML map.")
        return
    fire_w = gpd.GeoSeries([fire], crs=ANALYSIS_CRS).to_crs(POINT_CRS)   # reproject to WGS84 for the web basemap
    centre = fire_w.iloc[0].centroid
    fmap = folium.Map(location=[centre.y, centre.x], zoom_start=11, tiles=BASEMAP)
    try:                                          # fuel hazard backdrop (green->red)
        rgba, bounds = fuel_hazard_overlay()
        folium.raster_layers.ImageOverlay(rgba, bounds=bounds, opacity=0.5,
                                          name="fuel hazard (flammability)").add_to(fmap)
    except Exception as e:
        print(f"  (fuel overlay skipped: {e})")
    if zone is not None:
        folium.GeoJson(gpd.GeoSeries([zone], crs=ANALYSIS_CRS).to_crs(POINT_CRS).to_json(),
                       name="blocked zone", style_function=lambda _f: {"color": "orange",
                       "weight": 1, "fillColor": "orange", "fillOpacity": 0.2}).add_to(fmap)
    folium.GeoJson(fire_w.to_json(), name="fire", style_function=lambda _f: {"color": "red",
                   "weight": 1, "fillColor": "red", "fillOpacity": 0.4}).add_to(fmap)

    if len(routes):
        folium.GeoJson(routes.to_crs(POINT_CRS).to_json(), name="evacuation routes",
                       style_function=lambda _f: {"color": "blue", "weight": 4,
                       "opacity": 0.8}).add_to(fmap)

    fg_ref = folium.FeatureGroup(name="refuges (safe)", show=True)
    for _, r in refuges.to_crs(POINT_CRS).iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=4, color="green",
                            fill=True, fill_opacity=0.9,
                            popup=f"refuge: {r['name']}").add_to(fg_ref)
    fg_ref.add_to(fmap)

    status_colour = {"ok": "blue", "cut_off": "red", "impacted": "black"}
    fg_risk = folium.FeatureGroup(name="AT-RISK settlements", show=True)
    for _, r in at_risk.to_crs(POINT_CRS).iterrows():
        st = r.get("status", "cut_off")
        extra = (f" -> {r['refuge']} ({r['route_km']} km)" if st == "ok"
                 else " [encircled]" if st == "cut_off" else " [inside fire front]")
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=5,
                            color=status_colour.get(st, "red"), fill=True, fill_opacity=0.95,
                            popup=f"{r.get(NAME_COL,'?')} (pop {r.get(POP_COL,'?')}){extra}").add_to(fg_risk)
    fg_risk.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(OUT_HTML))
    print(f"Saved HTML map -> {OUT_HTML}")


def refuge_candidates(safe_official):
    """Destinations = OSM refuges + the SAFE official settlements.

    The official settlements that are NOT at-risk are perfectly good shelters
    (e.g. Kokkinomilia, on the way out), and their points are well-located (ELSTAT
    centroids); the OSM namesakes can be mislocated kilometres off the road. So we
    add the safe official settlements to the OSM refuge pool rather than relying on
    OSM alone."""
    osm = osm_refuges().to_crs(ANALYSIS_CRS)[["name", "place_type",
                                              "population", "source", "geometry"]]
    official = gpd.GeoDataFrame({                            # reshape official settlements onto the SAME schema as OSM
        "name": safe_official[NAME_COL].astype(str),
        "place_type": "official",
        "population": safe_official[POP_COL],
        "source": "official",
    }, geometry=safe_official.geometry, crs=ANALYSIS_CRS)
    return gpd.GeoDataFrame(pd.concat([osm, official], ignore_index=True),   # plain concat -- disjoint sources, no de-dup needed
                            crs=ANALYSIS_CRS), len(osm), len(official)


EvacResult = namedtuple("EvacResult", "in_zone routes safe_ref n_removed n_osm n_off")


def route_to_refuges(scenario, graph=None):
    """Route every AT-RISK settlement to its nearest SAFE refuge on the
    friction-weighted, fire-cut graph. Pure (no printing / no file I/O) so every
    caller (the CLI `main()`, `fire_timesteps.py`) shares one code path.
    Pass a preloaded `graph` when looping over timesteps (it is copied, not
    mutated); None -> load it here.

    Returns an `EvacResult`; `in_zone` is annotated with `status`
    (ok / cut_off / impacted), `refuge`, and `route_km`.
    """
    _atrisk, fire, _centre, zone, in_zone, safe = scenario
    in_zone = in_zone.copy()
    impacted = in_zone.geometry.within(fire).to_numpy()   # point inside flame front
    threatened = in_zone[~impacted]                        # at-risk but NOT yet inside the fire -> these get routed

    candidates, n_osm, n_off = refuge_candidates(safe)
    if zone is not None:
        candidates = candidates.reset_index(drop=True)
        try:
            near = candidates.sindex.query(zone, predicate="dwithin",       # candidate refuges too close to the danger zone...
                                           distance=SAFE_MARGIN_M)
        except (TypeError, ValueError):                  # older geopandas
            near = candidates.sindex.query(zone.buffer(SAFE_MARGIN_M),
                                           predicate="intersects")
        safe_ref = candidates.drop(index=candidates.index[near])   # ...are excluded: a refuge must itself be genuinely safe
    else:
        safe_ref = candidates
    safe_ref = safe_ref.reset_index(drop=True)

    if graph is None:
        graph = load_graph()
    gf, n_removed = friction_graph(graph, fire)             # the fire-cut, friction-weighted graph for THIS perimeter

    refuge_nodes = snap_nodes(gf, safe_ref)        # {node: refuge row index}
    refuge_set = set(refuge_nodes)

    # Outcome per at-risk settlement (keyed by in_zone index); impacted ones are
    # inside the flames -> not routed.
    status = {i: "impacted" for i in in_zone.index[impacted]}
    refuge_of, km_of = {}, {}
    rows, geoms = [], []                           # routed lines only (layer "routes")

    o_nodes = (np.atleast_1d(ox.distance.nearest_nodes(       # snap each THREATENED settlement onto the fire-cut graph
        gf, X=threatened.geometry.x.values, Y=threatened.geometry.y.values))
        if len(threatened) else [])
    for o_idx, o_node in zip(threatened.index, o_nodes):       # one Dijkstra per origin -- the actual routing loop
        o = in_zone.loc[o_idx]
        best = nearest_refuge(gf, int(o_node), refuge_set)
        if best is None:
            status[o_idx] = "cut_off"                          # no reachable refuge on the fire-cut graph -> encircled
            continue
        r_node, cost, path = best
        r = safe_ref.loc[refuge_nodes[r_node]]
        line, length_m, _crs, _n = route_to_line(gf, path)      # node-id path -> an actual LineString geometry
        status[o_idx], refuge_of[o_idx], km_of[o_idx] = "ok", r["name"], length_m / 1000
        rows.append({"origin": o.get(NAME_COL), "origin_pop": o.get(POP_COL),
                     "refuge": r["name"], "length_m": round(length_m, 1),
                     "cost": round(float(cost), 1)})
        geoms.append(line)

    # Attach the outcome to the at-risk points (for the GeoPackage + map + contract).
    in_zone["status"] = in_zone.index.map(status)
    in_zone["refuge"] = in_zone.index.map(refuge_of)
    in_zone["route_km"] = in_zone.index.map(lambda i: round(km_of[i], 2) if i in km_of else None)

    routes = (gpd.GeoDataFrame(rows, geometry=geoms, crs=ANALYSIS_CRS)
              if rows else gpd.GeoDataFrame(geometry=[], crs=ANALYSIS_CRS))
    return EvacResult(in_zone, routes, safe_ref, n_removed, n_osm, n_off)


def save_artifacts(scenario, result):
    """Write the GeoPackage (fire, blocked_zone, at_risk, refuges, routes) and the
    folium HTML map for an `EvacResult`. Reused by `main()` and the orchestrator."""
    _atrisk, fire, _centre, zone, _in_zone0, _safe = scenario
    in_zone, routes, safe_ref = result.in_zone, result.routes, result.safe_ref

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    gpd.GeoDataFrame({"kind": ["fire"]}, geometry=[fire], crs=ANALYSIS_CRS).to_file(
        OUT_GPKG, layer="fire", driver="GPKG")
    if zone is not None:
        gpd.GeoDataFrame({"kind": ["blocked_zone"]}, geometry=[zone], crs=ANALYSIS_CRS).to_file(
            OUT_GPKG, layer="blocked_zone", driver="GPKG")
    in_zone.to_file(OUT_GPKG, layer="at_risk", driver="GPKG")
    safe_ref.to_file(OUT_GPKG, layer="refuges", driver="GPKG")
    if len(routes):
        routes.to_file(OUT_GPKG, layer="routes", driver="GPKG")
    print(f"Saved GeoPackage -> {OUT_GPKG}")
    save_html(fire, zone, in_zone, safe_ref, routes)
    return OUT_GPKG


def _print_summary(result):
    """Console table of the evacuation outcome (CLI only)."""
    in_zone = result.in_zone
    n_imp = int((in_zone["status"] == "impacted").sum())
    print(f"At-risk origins: {len(in_zone)} ({n_imp} inside the fire front, "
          f"{len(in_zone) - n_imp} threatened around it)")
    print(f"Refuges: {result.n_osm} OSM + {result.n_off} safe official -> "
          f"{len(result.safe_ref)} candidates (>= {SAFE_MARGIN_M:.0f} m clear of the zone)")
    print(f"Friction graph: {result.n_removed} edges removed (cross the fire front), "
          f"penalty up to {1 + FRICTION_MAX:.0f}x within {FRICTION_BAND_M:.0f} m of it")

    label = {"ok": "ok", "cut_off": "CUT OFF (encircled)", "impacted": "inside fire front"}
    print(f"\n{'origin':<22}{'refuge':<22}{'km':>8}  status")
    for _, r in in_zone.sort_values("status").iterrows():
        ref = r["refuge"] if pd.notna(r["refuge"]) else "--"
        km = f"{r['route_km']:.2f}" if pd.notna(r["route_km"]) else "--"
        print(f"{str(r.get(NAME_COL,'?')):<22}{str(ref):<22}{km:>8}  {label.get(r['status'], '?')}")

    n_ok = int((in_zone["status"] == "ok").sum())
    n_enc = int((in_zone["status"] == "cut_off").sum())
    longest = f"{result.routes['length_m'].max()/1000:.2f} km" if len(result.routes) else "--"
    print(f"\n{n_ok} routed (longest {longest}), {n_enc} encircled, {n_imp} inside fire front")


def main():
    scenario = compute_exposure()
    result = route_to_refuges(scenario)
    _print_summary(result)
    save_artifacts(scenario, result)


if __name__ == "__main__":
    main()
