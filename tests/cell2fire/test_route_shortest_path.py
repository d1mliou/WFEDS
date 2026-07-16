"""Tests for scripts/cell2fire/route_shortest_path.py's routing primitives.

Covers: snap(), route_or_none(), route_to_line(), load_graph().

snap() reprojects an EPSG:4326 (lon, lat) point into the graph's CRS via
geopandas/pyproj, then calls osmnx.distance.nearest_nodes(). osmnx picks its
search strategy from whether the graph CRS is projected: a projected CRS uses
a scipy k-d tree, an unprojected (geographic) CRS uses an sklearn ball tree.
scikit-learn is not installed in this environment, so all snap() tests here
use a projected CRS (EPSG:2100, matching production) rather than EPSG:4326 --
using EPSG:4326 as the graph CRS raises ImportError before it even reaches
the logic under test.
"""

import osmnx as ox
import pytest
from pyproj import Transformer

import route_shortest_path as rsp
from tests.helpers.geodata_factories import make_tiny_road_graph

# Round-trip EPSG:4326 <-> EPSG:2100 transformers, used to build "a point
# exactly at a node's coordinates" test inputs: pick the node's (x, y) in the
# graph's projected CRS, convert to lon/lat, feed that lon/lat into snap()
# (which converts it right back to EPSG:2100 internally). This roundtrip
# introduces only sub-centimetre floating point noise.
_TO_4326 = Transformer.from_crs("EPSG:2100", "EPSG:4326", always_xy=True)
_TO_2100 = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)


def _base_xy():
    """A realistic EPSG:2100 basepoint (near the production ORIGIN town)."""
    return _TO_2100.transform(23.15, 38.95)


class TestSnap:
    def test_point_at_node_returns_that_exact_node(self):
        x0, y0 = _base_xy()
        nodes = {1: (x0, y0), 2: (x0 + 1000.0, y0), 3: (x0 + 2000.0, y0 + 500.0)}
        graph = make_tiny_road_graph(nodes, [(1, 2), (2, 3)], crs="EPSG:2100")

        lon, lat = _TO_4326.transform(*nodes[1])
        node, input_pt, snapped_pt, dist = rsp.snap(graph, "P1", lat, lon)

        # snap() returns a 4-tuple: (node_id, input_point, snapped_point, dist).
        assert node == 1
        assert snapped_pt.x == pytest.approx(x0)
        assert snapped_pt.y == pytest.approx(y0)
        # Only round-trip reprojection noise, not real distance -- sub-cm.
        assert dist == pytest.approx(0.0, abs=1e-2)

    def test_offset_point_returns_geometrically_nearest_node(self):
        x0, y0 = _base_xy()
        nodes = {1: (x0, y0), 2: (x0 + 1000.0, y0), 3: (x0 + 2000.0, y0 + 500.0)}
        graph = make_tiny_road_graph(nodes, [(1, 2), (2, 3)], crs="EPSG:2100")

        # A point 10m/5m off of node 2, far from nodes 1 and 3.
        offset_x, offset_y = x0 + 1010.0, y0 + 5.0
        lon, lat = _TO_4326.transform(offset_x, offset_y)
        node, input_pt, snapped_pt, dist = rsp.snap(graph, "P2", lat, lon)

        assert node == 2
        assert dist == pytest.approx((10.0**2 + 5.0**2) ** 0.5, abs=1e-2)


class TestRouteOrNone:
    def test_disconnected_components_returns_none(self):
        # Two separate clusters, no edge between them -> no path.
        nodes = {
            1: (0.0, 0.0), 2: (100.0, 0.0),        # cluster A
            10: (5000.0, 5000.0), 11: (5100.0, 5000.0),  # cluster B
        }
        edges = [(1, 2), (10, 11)]
        graph = make_tiny_road_graph(nodes, edges, crs="EPSG:2100")

        assert rsp.route_or_none(graph, 1, 10) is None

    def test_connected_graph_returns_route(self):
        # Sanity check: same helper, but with a path -- should NOT be None,
        # so test_disconnected_components_returns_none is actually exercising
        # "no path" and not some unrelated failure mode.
        nodes = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (200.0, 0.0)}
        edges = [(1, 2), (2, 3)]
        graph = make_tiny_road_graph(nodes, edges, crs="EPSG:2100")

        route = rsp.route_or_none(graph, 1, 3)
        assert route == [1, 2, 3]


class TestRouteToLine:
    def test_total_length_equals_sum_of_edge_lengths(self):
        nodes = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (100.0, 150.0)}
        edges = [(1, 2, 100.0), (2, 3, 150.0)]
        graph = make_tiny_road_graph(nodes, edges, crs="EPSG:2100")

        line, total_m, crs, n_edges = rsp.route_to_line(graph, [1, 2, 3])

        assert total_m == pytest.approx(100.0 + 150.0)
        assert n_edges == 2
        assert str(crs) == "EPSG:2100"
        # The merged line's own geometric length should agree too.
        assert line.length == pytest.approx(250.0)


class TestLoadGraph:
    def test_reads_back_a_saved_graph_with_matching_node_and_edge_counts(
        self, tmp_path, monkeypatch
    ):
        nodes = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (100.0, 150.0)}
        edges = [(1, 2, 100.0), (2, 3, 150.0)]
        graph = make_tiny_road_graph(nodes, edges, crs="EPSG:2100")

        graphml_path = tmp_path / "tiny.graphml"
        ox.save_graphml(graph, graphml_path)  # matches load_graph's ox.load_graphml

        # GOTCHA (confirmed by inspection and empirically): load_graph is
        # defined as `def load_graph(path=GRAPH_PATH):` -- the default value
        # is bound once, at module-import time, to whatever GRAPH_PATH was
        # then. monkeypatch.setattr(rsp, "GRAPH_PATH", ...) only rebinds the
        # *module attribute*; it does NOT touch load_graph.__defaults__, so a
        # bare `rsp.load_graph()` call keeps resolving to the ORIGINAL path.
        # Demonstrate that here, then pass the path explicitly (the only
        # reliable way to inject a test path into this function).
        original_default = rsp.load_graph.__defaults__[0]
        monkeypatch.setattr(rsp, "GRAPH_PATH", graphml_path)
        assert rsp.load_graph.__defaults__[0] == original_default
        assert rsp.load_graph.__defaults__[0] != graphml_path

        loaded = rsp.load_graph(graphml_path)

        assert loaded.number_of_nodes() == graph.number_of_nodes() == 3
        assert loaded.number_of_edges() == graph.number_of_edges() == 4  # bidirectional
