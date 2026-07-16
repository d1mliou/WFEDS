"""Small, reusable factory functions for building tiny synthetic geodata in tests.

No real data files are needed anywhere in this suite -- every raster/vector/graph a
test needs is built in-memory (or in a `tmp_path`) by these helpers, matching the
exact formats/attributes the production code reads. Import from test files with:

    from tests.helpers.geodata_factories import make_tiny_raster, ...
"""

from __future__ import annotations

import math

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from shapely.geometry import LineString, Point, Polygon


def make_tiny_raster(path, array, transform, crs="EPSG:2100", nodata=None):
    """Write `array` as a single-band GeoTIFF via rasterio.

    `transform` is a `rasterio.Affine` (or `affine.Affine`) mapping array (row, col)
    -> world (x, y), e.g. `rasterio.transform.from_origin(x0, y0, xres, yres)`.
    """
    array = np.asarray(array)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=array.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def make_tiny_asc(path, array, transform, fmt="%d", nodata=-9999):
    """Write an Arc/Info ASCII grid, mirroring
    `build_cell2fire_instance.write_asc`'s exact header layout and body format:
    ncols/nrows/xllcorner/yllcorner/cellsize/NODATA_value, then one row per line,
    space-delimited (via `np.savetxt`).

    `transform` is a `rasterio.Affine`-like object: `.a` = cellsize (x-res),
    `.c` = left x (xllcorner), `.f` = top y, `.e` = negative y-res (so
    `.f + nrows * .e` = bottom y = yllcorner) -- exactly as `write_asc` computes it.
    """
    array = np.asarray(array)
    nrows, ncols = array.shape
    cell = transform.a
    xll = transform.c
    yll = transform.f + nrows * transform.e
    header = (f"ncols {ncols}\nnrows {nrows}\nxllcorner {xll:.6f}\n"
              f"yllcorner {yll:.6f}\ncellsize {cell:.6f}\nNODATA_value {nodata}\n")
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, array, fmt=fmt, delimiter=" ")
    return path


def make_tiny_points_gdf(rows, crs="EPSG:2100"):
    """rows: list of dicts with 'geometry' (an (x, y) tuple -> a shapely Point)
    plus any other columns. Returns a geopandas.GeoDataFrame."""
    data, geoms = [], []
    for row in rows:
        row = dict(row)
        x, y = row.pop("geometry")
        geoms.append(Point(x, y))
        data.append(row)
    return gpd.GeoDataFrame(data, geometry=geoms, crs=crs)


def make_tiny_polygon_gdf(rows, crs="EPSG:2100"):
    """Same idea as `make_tiny_points_gdf`, but each row's 'geometry' is a list of
    (x, y) tuples forming a polygon ring."""
    data, geoms = [], []
    for row in rows:
        row = dict(row)
        ring = row.pop("geometry")
        geoms.append(Polygon(ring))
        data.append(row)
    return gpd.GeoDataFrame(data, geometry=geoms, crs=crs)


def make_tiny_road_graph(nodes, edges, crs="EPSG:2100", bidirectional=True):
    """nodes: dict {node_id: (x, y)}. edges: list of (u, v, length_m) or (u, v)
    (length computed from node coords when omitted).

    Builds and returns a networkx.MultiDiGraph with exactly the attributes
    OSMnx / route_shortest_path.py read off a graph (verified against
    scripts/cell2fire/route_shortest_path.py's load_graph/snap/route_or_none/
    route_to_line, and osmnx.routing.route_to_gdf / osmnx.distance.nearest_nodes):
      * per-node 'x' / 'y' (floats)               -- snap(), route_to_line()
      * graph.graph['crs']                        -- snap() reprojects into it
      * per-edge 'length' (float, metres)          -- WEIGHT = "length" (shortest
                                                       path weight + route length sum)
      * per-edge 'geometry' (shapely LineString)   -- route_to_line()'s linemerge
        (osmnx will auto-synthesize a straight-line geometry from endpoint x/y if
        omitted, but we always set it explicitly here to remove that ambiguity).

    `bidirectional=True` (default) adds both (u, v) and (v, u) for each edge, like
    a real two-way street in an OSMnx graph -- so routing actually works both ways.
    Pass False to build one-way edges (e.g. to test a disconnected/no-path case).
    """
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = crs
    for node_id, (x, y) in nodes.items():
        graph.add_node(node_id, x=float(x), y=float(y))

    def _add_edge(u, v, length):
        xu, yu = nodes[u]
        xv, yv = nodes[v]
        geom = LineString([(xu, yu), (xv, yv)])
        graph.add_edge(u, v, length=float(length), geometry=geom)

    for edge in edges:
        if len(edge) == 3:
            u, v, length = edge
        else:
            u, v = edge
            xu, yu = nodes[u]
            xv, yv = nodes[v]
            length = math.hypot(xv - xu, yv - yu)
        _add_edge(u, v, length)
        if bidirectional:
            _add_edge(v, u, length)

    return graph
