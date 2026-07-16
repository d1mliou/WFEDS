"""Download the drivable road network within the study-area boundary using OSMnx,
project it to the Greek Grid (EPSG:2100), save it as GraphML, and export the edges
as a GeoPackage for viewing in QGIS.

The study area is defined by the boundary layer and may span several municipalities
on purpose: evacuation/exit routes often leave the originating municipality, so the
network must not be clipped at a single municipal line.

All analysis and outputs are in EPSG:2100 (Greek Grid, metric). OpenStreetMap is
queried in EPSG:4326 — the only CRS the Overpass API accepts — so the boundary is
reprojected to 4326 *only* to run the download; the resulting graph is then
projected to EPSG:2100 before any measurement or saving.

Scope: data acquisition only — no routing, no fire perimeters, no LLM code.

Outputs (written to DATA_DIR/Roads in OneDrive, see config below):
    road_graph.graphml   full graph (nodes + edges), EPSG:2100
    road_edges.gpkg      edge geometries (layer "edges"), EPSG:2100

Run:
    python scripts/data_prep/download_road_network.py
"""

import geopandas as gpd
import osmnx as ox

from _paths import DATA_DIR, STUDY_AREA               # data catalog: base folder + boundary layer

# --- Configuration -----------------------------------------------------------
BOUNDARY_PATH = STUDY_AREA          # study-area boundary polygons (the AOI), supplied in EPSG:2100
SOURCE_CRS = "EPSG:2100"            # the boundary file's CRS (Greek Grid)
BOUNDARY_FILTER = None              # None = BOTH North Evia municipalities (keep cross-boundary exit routes); or ("Municipality", "<name>") to clip to one
NETWORK_TYPE = "drive"             # drivable public roads only (excludes tracks, paths, service, private)
ANALYSIS_CRS = "EPSG:2100"          # all measurement/output in metres (Greek Grid)

ROADS_DIR = DATA_DIR / "Roads"                        # outputs go to the OneDrive Data folder, NOT the repo
GRAPH_PATH = ROADS_DIR / "road_graph.graphml"         # routable graph (nodes + edges)
EDGES_PATH = ROADS_DIR / "road_edges.gpkg"            # edge geometries for QGIS


def _aoi_geometry(gdf):
    """Dissolve a GeoDataFrame's features into a single (multi)polygon."""
    geom = gdf.geometry
    return geom.union_all() if hasattr(geom, "union_all") else geom.unary_union   # dissolve all features into one AOI


def load_boundary_4326(boundary_filter=BOUNDARY_FILTER):
    """Return the study-area polygon in EPSG:4326 for an OSM query.

    `boundary_filter` = (field, value) restricts the AOI to a single
    municipality; it defaults to the module-level BOUNDARY_FILTER (None = BOTH
    municipalities, the right choice for the road network so cross-boundary exit
    routes stay intact). `download_settlements.py` passes a single-municipality
    filter to pull OSM settlements for only the Mantoudi side.
    """
    gdf = gpd.read_file(BOUNDARY_PATH)                # read the boundary layer
    if gdf.crs is None:
        gdf = gdf.set_crs(SOURCE_CRS)                 # stamp the CRS if the file didn't carry one
    if boundary_filter is not None:                  # optional attribute filter -> select-by-attribute to one municipality
        field, value = boundary_filter
        gdf = gdf[gdf[field] == value]
        if gdf.empty:
            raise ValueError(f"No boundary feature with {field} == {value!r}")
    print(f"Boundary: {len(gdf)} feature(s) from {BOUNDARY_PATH.name} ({gdf.crs})")
    return _aoi_geometry(gdf.to_crs("EPSG:4326"))    # reproject to WGS84 (the only CRS Overpass accepts) + dissolve


def _atomic_write(target, write_fn):
    """Write via a temp file, then swap it in, keeping the previous version as
    ``<name>.bak<ext>``. So a failed or partial write never destroys the last
    good file (unlike a plain overwrite). Shared by both download scripts."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
    if tmp.exists():
        tmp.unlink()
    write_fn(tmp)                                    # write the new output to a temp file first
    if target.exists():                             # keep the previous good file as .bak
        target.replace(target.with_name(f"{target.stem}.bak{target.suffix}"))
    tmp.replace(target)                             # atomic swap the temp into place


def download_graph(aoi_4326):
    """Download the road network within the AOI and project it to EPSG:2100."""
    print(f"Downloading '{NETWORK_TYPE}' network within boundary...")
    try:
        graph_4326 = ox.graph_from_polygon(          # download drivable roads clipped to the AOI...
            aoi_4326, network_type=NETWORK_TYPE, truncate_by_edge=True   # ...keeping edges that cross the boundary intact (both nodes)
        )
    except Exception as e:
        raise RuntimeError(                          # wrap network failures in a clear message; old files untouched
            "Failed to download the road network from OpenStreetMap -- usually no "
            "internet, the Overpass API being down, or rate-limiting. The existing "
            "data files were left untouched; just re-run when the connection is "
            f"back.\nOriginal error: {type(e).__name__}: {e}"
        ) from e
    return ox.project_graph(graph_4326, to_crs=ANALYSIS_CRS)   # reproject graph to metric Greek Grid before any measurement


def export_edges_gpkg(graph):
    """Export the road edges as a GeoPackage (EPSG:2100) for QGIS."""
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()   # graph edges -> GeoDataFrame (attribute table)
    for col in edges.columns:                        # GeoPackage can't store Python lists...
        if col != "geometry":
            edges[col] = edges[col].apply(
                lambda v: "; ".join(map(str, v)) if isinstance(v, list) else v   # ...so flatten any list-valued attribute to text
            )
    _atomic_write(EDGES_PATH, lambda p: edges.to_file(p, driver="GPKG", layer="edges"))   # safe write to GeoPackage
    print(f"Exported {len(edges)} edges -> {EDGES_PATH}")


def main():
    aoi_4326 = load_boundary_4326()                  # 1) AOI polygon in WGS84
    graph = download_graph(aoi_4326)                 # 2) download + reproject the road graph
    _atomic_write(GRAPH_PATH, lambda p: ox.save_graphml(graph, p))   # 3) save the routable graph (GraphML)
    print(
        f"Saved {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges "
        f"-> {GRAPH_PATH} ({ANALYSIS_CRS})"
    )
    export_edges_gpkg(graph)                          # 4) also export edges as GeoPackage for QGIS


if __name__ == "__main__":
    main()
