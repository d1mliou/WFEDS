"""Shortest evacuation route between two points -- and the shared routing module.

This is BOTH a standalone entry point (run it to produce the baseline route
GeoPackage + HTML) AND the single home of the routing primitives that the
fire-blocking script reuses. `route_with_fire.py` imports from here instead of
duplicating the routing:

    from route_shortest_path import (
        ORIGIN, DESTINATION, load_graph, snap, route_or_none, route_to_line,
    )

so the routing code lives in one place.

Outputs (DATA_DIR/Routes/):
    shortest_path.gpkg   layers "route" (line) + "endpoints" (points), EPSG:2100
    shortest_path.html   interactive map (only if folium is installed)

Run:
    python scripts/cell2fire/route_shortest_path.py
"""

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
from shapely.ops import linemerge

from _paths import DATA_DIR

# --- Shared routing configuration (imported by the fire-blocking script) ------
GRAPH_PATH = DATA_DIR / "Roads" / "road_graph.graphml"

# Origin/destination as (name, latitude, longitude) in EPSG:4326. Two North Evia
# towns, one per municipality, so the route crosses the municipal boundary.
ORIGIN = ("Istiaia", 38.9520, 23.1469)
DESTINATION = ("Limni", 38.7660, 23.3180)

WEIGHT = "length"             # shortest path by road length (metres)
POINT_CRS = "EPSG:4326"       # CRS the ORIGIN/DESTINATION coordinates are in
BASEMAP = "CartoDB positron"  # license-friendly, clean backdrop for folium maps

ROUTES_DIR = DATA_DIR / "Routes"
ROUTE_GPKG = ROUTES_DIR / "shortest_path.gpkg"
ROUTE_HTML = ROUTES_DIR / "shortest_path.html"


# --- Routing primitives (shared; reused by route_with_fire.py) ----------------
def load_graph(path=GRAPH_PATH):
    """Load the saved road graph (EPSG:2100)."""
    graph = ox.load_graphml(path)                       # GraphML already carries its CRS + node/edge attributes
    print(
        f"Loaded graph: {graph.number_of_nodes()} nodes / "
        f"{graph.number_of_edges()} edges ({graph.graph.get('crs')})"
    )
    return graph


def snap(graph, name, lat, lon):
    """Snap a lon/lat point to the nearest graph node.

    Returns (node_id, input_point, snapped_point, snap_distance_m); all geometry
    in the graph CRS.
    """
    crs = graph.graph["crs"]                              # the graph's own CRS (EPSG:2100) -- reproject the input to match
    input_pt = gpd.GeoSeries([Point(lon, lat)], crs=POINT_CRS).to_crs(crs).iloc[0]   # WGS84 lat/lon -> Greek Grid
    node, dist = ox.distance.nearest_nodes(               # nearest road NODE (not the point on the edge itself)
        graph, X=input_pt.x, Y=input_pt.y, return_dist=True
    )
    snapped_pt = Point(graph.nodes[node]["x"], graph.nodes[node]["y"])   # where the route actually starts/ends
    print(f"  {name}: node {node} ({dist:,.0f} m away)")   # how far off-road the input point was
    return node, input_pt, snapped_pt, dist


def route_or_none(graph, orig_node, dest_node):
    """Shortest path as a list of node ids, or None if no path exists."""
    try:
        return ox.routing.shortest_path(graph, orig_node, dest_node, weight=WEIGHT)   # Dijkstra by edge length
    except nx.NetworkXNoPath:                             # e.g. a disconnected component -- no route exists at all
        return None


def route_to_line(graph, route):
    """Merge a node-id route into one (Multi)LineString.

    Returns (geometry, total_length_m, crs, n_edges).
    """
    edges = ox.routing.route_to_gdf(graph, route, weight=WEIGHT)   # node-id path -> the actual edge geometries it uses
    line = linemerge(list(edges.geometry.values))         # dissolve consecutive edge segments into one continuous line
    return line, float(edges[WEIGHT].sum()), edges.crs, len(edges)


# --- Baseline-route outputs --------------------------------------------------
def build_route_gdf(line, total_m, crs, n_edges):
    """Single-feature route line GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "origin": [ORIGIN[0]],
            "destination": [DESTINATION[0]],
            "length_m": [round(total_m, 1)],
            "n_edges": [n_edges],
        },
        geometry=[line],
        crs=crs,
    )


def build_endpoints_gdf(crs, snaps):
    """Points layer: input + snapped point for origin and destination."""
    rows, geoms = [], []
    for role, name, (node, input_pt, snapped_pt, dist) in snaps:
        rows.append({"role": role, "kind": "input", "name": name,     # the RAW requested coordinate...
                     "node": None, "snap_dist_m": None})
        geoms.append(input_pt)
        rows.append({"role": role, "kind": "snapped", "name": name,   # ...and where it actually landed on the graph
                     "node": int(node), "snap_dist_m": round(float(dist), 1)})
        geoms.append(snapped_pt)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def save_gpkg(route_gdf, endpoints_gdf):
    """Write the route and endpoints as layers in one GeoPackage (EPSG:2100)."""
    ROUTE_GPKG.parent.mkdir(parents=True, exist_ok=True)
    route_gdf.to_file(ROUTE_GPKG, driver="GPKG", layer="route")
    endpoints_gdf.to_file(ROUTE_GPKG, driver="GPKG", layer="endpoints")
    print(f"Saved GeoPackage -> {ROUTE_GPKG} (layers: route, endpoints)")


def save_html(route_gdf, endpoints_gdf):
    """Interactive folium map (skipped gracefully if folium is absent)."""
    try:
        import folium
    except ImportError:
        print("folium not installed - skipping HTML map (GeoPackage still written).")
        return

    route_wgs = route_gdf.to_crs(POINT_CRS)               # reproject analysis-CRS layers to WGS84 for the web basemap
    ends_wgs = endpoints_gdf.to_crs(POINT_CRS)

    centre = route_wgs.geometry.iloc[0].centroid
    fmap = folium.Map(location=[centre.y, centre.x], zoom_start=11, tiles=BASEMAP)

    def add_line(geom):
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]   # linemerge may not fully dissolve every case
        for part in parts:
            folium.PolyLine(
                [(y, x) for x, y in part.coords], color="red", weight=4, opacity=0.8   # shapely (x,y) -> folium (lat,lon)
            ).add_to(fmap)

    add_line(route_wgs.geometry.iloc[0])

    for _, row in ends_wgs[ends_wgs["kind"] == "input"].iterrows():   # markers at the RAW input points, not the snapped ones
        colour = "green" if row["role"] == "origin" else "blue"
        folium.Marker(
            [row.geometry.y, row.geometry.x],
            popup=f"{row['role']}: {row['name']}",
            icon=folium.Icon(color=colour),
        ).add_to(fmap)

    ROUTE_HTML.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(ROUTE_HTML))
    print(f"Saved HTML map -> {ROUTE_HTML}")


def main():
    G = load_graph()

    print("Snapping endpoints:")
    o_snap = snap(G, *ORIGIN)
    d_snap = snap(G, *DESTINATION)

    route = route_or_none(G, o_snap[0], d_snap[0])
    if route is None:
        raise RuntimeError(
            "No path between origin and destination (network may be disconnected)."
        )
    print(f"Route: {len(route)} nodes")

    line, total_m, crs, n_edges = route_to_line(G, route)
    route_gdf = build_route_gdf(line, total_m, crs, n_edges)
    endpoints_gdf = build_endpoints_gdf(
        G.graph["crs"],
        [("origin", ORIGIN[0], o_snap), ("destination", DESTINATION[0], d_snap)],
    )
    print(f"Shortest-path length: {total_m / 1000:.2f} km")

    save_gpkg(route_gdf, endpoints_gdf)
    save_html(route_gdf, endpoints_gdf)


if __name__ == "__main__":
    main()
