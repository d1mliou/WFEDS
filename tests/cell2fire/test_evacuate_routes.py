"""Tests for scripts/cell2fire/evacuate_routes.py.

THIS PASS covers the pure(ish) graph/geometry helpers that can be exercised on a
tiny synthetic road graph and tiny synthetic GeoDataFrames, with no real data
files: `friction_graph`, `nearest_refuge`, `refuge_candidates`, `snap_nodes`.
`route_to_refuges` / `save_artifacts` / `main` (full pipeline, real file I/O) are
left for a later pass.

Values below (distances, weights) were prototyped against the real functions
first, so the assertions reflect ACTUAL observed behaviour, not assumptions --
see the report for any surprises.
"""

import pytest
from shapely.geometry import Point

import evacuate_routes as evac
from tests.helpers.geodata_factories import (
    make_tiny_points_gdf,
    make_tiny_road_graph,
)


# --------------------------------------------------------------------------------
# friction_graph
# --------------------------------------------------------------------------------
class TestFrictionGraph:
    """Fire = a circle (radius 200) centred on the origin; FRICTION_BAND_M = 1500,
    FRICTION_MAX = 4.0 (module constants, unpatched).

    Edge A (nodes 1<->2, along y=0 from x=-300 to x=300, length 600) CROSSES the
    fire circle -> must be removed (both directions - `make_tiny_road_graph`
    default `bidirectional=True` adds both (1,2) and (2,1), each with its own
    geometry, so both intersect the fire and both get dropped).

    Edge B (nodes 3<->4, vertical at x=1000, length 100) passes through
    (1000, 0), ~800 m from the fire boundary (1000 - 200) -- inside the
    1500 m friction band but not intersecting the fire -> kept, with
    EVAC_WEIGHT > plain length.

    Edge C (nodes 5<->6, vertical at x=3000, length 100) is ~2800 m from the
    fire boundary -- outside the friction band -> kept, EVAC_WEIGHT == length
    (penalty factor 1, i.e. untouched)."""

    NODES = {
        1: (-300, 0),
        2: (300, 0),
        3: (1000, -50),
        4: (1000, 50),
        5: (3000, -50),
        6: (3000, 50),
    }
    EDGES = [(1, 2), (3, 4), (5, 6)]
    FIRE = Point(0, 0).buffer(200)

    def _build(self):
        return make_tiny_road_graph(self.NODES, self.EDGES)

    def test_crossing_edge_is_removed(self):
        graph = self._build()

        gf, _n_removed = evac.friction_graph(graph, self.FIRE)

        assert not gf.has_edge(1, 2)
        assert not gf.has_edge(2, 1)

    def test_near_edge_gets_friction_weight_greater_than_length(self):
        graph = self._build()

        gf, _n_removed = evac.friction_graph(graph, self.FIRE)

        length = gf[3][4][0]["length"]
        assert length == pytest.approx(100.0)
        assert gf[3][4][0][evac.EVAC_WEIGHT] > length
        # reverse direction gets the same treatment
        assert gf[4][3][0][evac.EVAC_WEIGHT] > gf[4][3][0]["length"]

    def test_far_edge_weight_equals_plain_length(self):
        graph = self._build()

        gf, _n_removed = evac.friction_graph(graph, self.FIRE)

        assert gf[5][6][0][evac.EVAC_WEIGHT] == pytest.approx(gf[5][6][0]["length"])
        assert gf[5][6][0][evac.EVAC_WEIGHT] == pytest.approx(100.0)

    def test_drop_count_matches_edges_actually_removed(self):
        graph = self._build()
        before = graph.number_of_edges()

        gf, n_removed = evac.friction_graph(graph, self.FIRE)

        after = gf.number_of_edges()
        assert before - after == n_removed
        # both directions of the crossing edge (1,2)/(2,1) are dropped
        assert n_removed == 2
        # the original graph passed in is untouched (friction_graph copies it)
        assert graph.number_of_edges() == before
        assert graph.has_edge(1, 2)


# --------------------------------------------------------------------------------
# nearest_refuge
# --------------------------------------------------------------------------------
class TestNearestRefuge:
    def test_picks_cheapest_reachable_refuge_not_nearest(self):
        """Node 1 sits geometrically close to origin (100 m) but is reached via
        an expensive edge (EVAC_WEIGHT 1000). Node 2 is geometrically far (via
        an intermediate node 3) but the friction-weighted path costs only
        50 + 50 = 100 total -- cheaper than node 1's 1000. `nearest_refuge` must
        pick node 2."""
        nodes = {0: (0, 0), 1: (100, 0), 3: (500, 0), 2: (1000, 0)}
        edges = [(0, 1, 100), (0, 3, 50), (3, 2, 50)]
        graph = make_tiny_road_graph(nodes, edges)
        for u, v in ((0, 1), (0, 3), (3, 2)):
            weight = 1000.0 if {u, v} == {0, 1} else 50.0
            graph[u][v][0][evac.EVAC_WEIGHT] = weight
            graph[v][u][0][evac.EVAC_WEIGHT] = weight

        best = evac.nearest_refuge(graph, 0, {1, 2})

        assert best is not None
        refuge_node, cost, path = best
        assert refuge_node == 2
        assert cost == pytest.approx(100.0)
        assert path == [0, 3, 2]

    def test_returns_none_when_no_refuge_is_reachable(self):
        """Node 'isolated' exists in the graph but has no edges at all (built by
        simply never referencing it in `edges`) -- Dijkstra from node 0 never
        reaches it, so `nearest_refuge` must signal "unreachable" as `None`
        (confirmed by reading the source: `dist.get(rn)` stays `None` for an
        unreachable node, so `best` is never assigned)."""
        nodes = {0: (0, 0), 1: (100, 0), "isolated": (5000, 5000)}
        edges = [(0, 1, 100)]
        graph = make_tiny_road_graph(nodes, edges)
        graph[0][1][0][evac.EVAC_WEIGHT] = 10.0
        graph[1][0][0][evac.EVAC_WEIGHT] = 10.0

        result = evac.nearest_refuge(graph, 0, {"isolated"})

        assert result is None


# --------------------------------------------------------------------------------
# refuge_candidates
# --------------------------------------------------------------------------------
class TestRefugeCandidates:
    def test_combines_osm_refuges_with_safe_official_settlements(self, monkeypatch):
        """`refuge_candidates` reads OSM refuges via the module-level
        `osm_refuges` name (the current import is `from settlements import
        refuges as osm_refuges`), so that is the name to monkeypatch -- patching
        a hypothetical `evac.refuges` would silently do nothing."""
        osm_gdf = make_tiny_points_gdf(
            [
                {"geometry": (0, 0), "name": "OSM_A", "place_type": "village",
                 "population": 50, "source": "osm"},
                {"geometry": (100, 100), "name": "OSM_B", "place_type": "village",
                 "population": 80, "source": "osm"},
            ],
            crs=evac.ANALYSIS_CRS,
        )
        safe_official = make_tiny_points_gdf(
            [{"geometry": (200, 200), "NAME_OIK": "Off_A", "census2021": 120}],
            crs=evac.ANALYSIS_CRS,
        )
        monkeypatch.setattr(evac, "osm_refuges", lambda: osm_gdf)

        result, n_osm, n_off = evac.refuge_candidates(safe_official)

        assert n_osm == 2
        assert n_off == 1
        assert len(result) == n_osm + n_off == 3
        assert set(result["name"]) == {"OSM_A", "OSM_B", "Off_A"}
        official_rows = result[result["source"] == "official"]
        assert len(official_rows) == 1
        assert official_rows.iloc[0]["name"] == "Off_A"
        assert official_rows.iloc[0]["place_type"] == "official"
        assert official_rows.iloc[0]["population"] == 120


# --------------------------------------------------------------------------------
# snap_nodes
# --------------------------------------------------------------------------------
class TestSnapNodes:
    def test_keeps_closer_point_when_two_points_snap_to_same_node(self):
        """Both points snap to node 1 (node 2 is 1000 m away, far too far to be
        nearest). Point at index 0 is 10 m from node 1; point at index 1 is
        5 m from node 1 (closer). Reading the source: `best[node] = (idx, d)`
        is only overwritten `if node not in best or d < best[node][1]` -- i.e.
        the CLOSER point's index wins, regardless of iteration/row order."""
        nodes = {1: (0, 0), 2: (1000, 0)}
        edges = [(1, 2, 1000)]
        graph = make_tiny_road_graph(nodes, edges)
        points = make_tiny_points_gdf(
            [{"geometry": (10, 0)}, {"geometry": (5, 0)}], crs=evac.ANALYSIS_CRS
        )

        result = evac.snap_nodes(graph, points)

        assert result == {1: 1}
