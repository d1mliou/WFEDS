"""Tests for scripts/data_prep/download_road_network.py.

Covers `_atomic_write` (the shared write-via-temp-then-swap helper),
`load_boundary_4326` (with real geodata read from a tmp file), and
`download_graph` (with `osmnx` fully mocked -- no network calls).
`export_edges_gpkg` / `main` (thin glue over the above, plus real OSMnx graph
objects) are left for a later pass.
"""

from unittest.mock import Mock

import pytest

import download_road_network as drn
from tests.helpers.geodata_factories import make_tiny_polygon_gdf


# --------------------------------------------------------------------------------
# _atomic_write
# --------------------------------------------------------------------------------
class TestAtomicWrite:
    def test_first_write_creates_target_with_no_bak_file(self, tmp_path):
        target = tmp_path / "foo.txt"

        drn._atomic_write(target, lambda p: p.write_text("v1", encoding="utf-8"))

        assert target.read_text(encoding="utf-8") == "v1"
        assert not (tmp_path / "foo.bak.txt").exists()

    def test_second_write_preserves_old_content_as_bak(self, tmp_path):
        target = tmp_path / "foo.txt"
        drn._atomic_write(target, lambda p: p.write_text("v1", encoding="utf-8"))

        drn._atomic_write(target, lambda p: p.write_text("v2", encoding="utf-8"))

        assert target.read_text(encoding="utf-8") == "v2"
        bak = tmp_path / "foo.bak.txt"
        assert bak.exists()
        assert bak.read_text(encoding="utf-8") == "v1"

    def test_write_fn_failure_leaves_target_completely_untouched(self, tmp_path):
        target = tmp_path / "foo.txt"
        target.write_text("original", encoding="utf-8")

        def _boom(p):
            raise RuntimeError("simulated write failure")

        with pytest.raises(RuntimeError, match="simulated write failure"):
            drn._atomic_write(target, _boom)

        assert target.read_text(encoding="utf-8") == "original"
        assert not (tmp_path / "foo.bak.txt").exists()


# --------------------------------------------------------------------------------
# load_boundary_4326
# --------------------------------------------------------------------------------
# Small, non-overlapping ~1km squares in EPSG:2100 (Greek Grid), roughly North
# Evia, so the .to_crs("EPSG:4326") reprojection inside the function is on
# realistic ground and doesn't produce garbage/NaN coordinates.
def _boundary_rows():
    def _square(x0, y0, side=1000.0):
        return [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]

    return [
        {"geometry": _square(480000, 4310000), "Municipality": "A"},
        {"geometry": _square(485000, 4310000), "Municipality": "A"},
        {"geometry": _square(480000, 4320000), "Municipality": "B"},
    ]


@pytest.fixture
def boundary_file(tmp_path):
    gdf = make_tiny_polygon_gdf(_boundary_rows())
    path = tmp_path / "boundary.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


class TestLoadBoundary4326:
    def test_no_filter_returns_geometry_covering_all_rows(self, monkeypatch, boundary_file):
        monkeypatch.setattr(drn, "BOUNDARY_PATH", boundary_file)

        geom = drn.load_boundary_4326(boundary_filter=None)

        assert geom is not None
        assert geom.bounds[2] > geom.bounds[0]   # non-degenerate: max_x > min_x

    def test_filter_matching_subset_gives_different_bounds_than_full_set(
            self, monkeypatch, boundary_file):
        monkeypatch.setattr(drn, "BOUNDARY_PATH", boundary_file)

        full = drn.load_boundary_4326(boundary_filter=None)
        subset = drn.load_boundary_4326(boundary_filter=("Municipality", "A"))

        assert subset.bounds != full.bounds
        # The "B" row sits further north (larger y) than either "A" row, so the
        # filtered-to-"A" union's northern edge must fall short of the full set's.
        assert subset.bounds[3] < full.bounds[3]

    def test_filter_matching_no_rows_raises_value_error(self, monkeypatch, boundary_file):
        monkeypatch.setattr(drn, "BOUNDARY_PATH", boundary_file)

        with pytest.raises(ValueError, match="Municipality"):
            drn.load_boundary_4326(boundary_filter=("Municipality", "nonexistent"))


# --------------------------------------------------------------------------------
# download_graph
# --------------------------------------------------------------------------------
class TestDownloadGraph:
    def test_success_calls_graph_from_polygon_then_project_graph(self, monkeypatch):
        aoi = object()
        graph_4326 = object()
        projected = object()
        fake_graph_from_polygon = Mock(return_value=graph_4326)
        fake_project_graph = Mock(return_value=projected)
        monkeypatch.setattr(drn.ox, "graph_from_polygon", fake_graph_from_polygon)
        monkeypatch.setattr(drn.ox, "project_graph", fake_project_graph)

        result = drn.download_graph(aoi)

        fake_graph_from_polygon.assert_called_once_with(
            aoi, network_type=drn.NETWORK_TYPE, truncate_by_edge=True)
        fake_project_graph.assert_called_once_with(graph_4326, to_crs=drn.ANALYSIS_CRS)
        assert result is projected

    def test_network_failure_wrapped_in_clear_runtime_error(self, monkeypatch):
        def _boom(*a, **k):
            raise Exception("Overpass down")

        monkeypatch.setattr(drn.ox, "graph_from_polygon", _boom)

        with pytest.raises(RuntimeError) as excinfo:
            drn.download_graph(object())

        msg = str(excinfo.value)
        assert "Failed to download the road network from OpenStreetMap" in msg
        assert "Exception: Overpass down" in msg
