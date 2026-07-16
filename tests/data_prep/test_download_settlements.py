"""Tests for scripts/data_prep/download_settlements.py.

Covers `_coalesce` (per-row name coalescing) and the `main()` flow with
`osmnx.features_from_polygon` mocked (no network calls).
"""

import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

import download_settlements as ds


# --------------------------------------------------------------------------------
# _coalesce
# --------------------------------------------------------------------------------
class TestCoalesce:
    def test_prefers_the_first_listed_column_when_both_present_and_nonempty(self):
        df = pd.DataFrame({"name:el": ["Ιστιαία"], "name": ["Istiaia"]})

        out = ds._coalesce(df, "name:el", "name")

        assert out.iloc[0] == "Ιστιαία"

    def test_empty_string_in_preferred_column_falls_back_to_next(self):
        """Not just missing/NaN -- a literal EMPTY (or whitespace-only) string
        in the preferred column must also be treated as "no value" and fall
        through to the next name, per `col.str.strip()` + `str.len() > 0`."""
        df = pd.DataFrame({"name:el": ["", "   "], "name": ["Istiaia", "Limni"]})

        out = ds._coalesce(df, "name:el", "name")

        assert list(out) == ["Istiaia", "Limni"]

    def test_falls_back_per_row_when_preferred_column_entirely_absent(self):
        """With `name:el` missing altogether as a column, `_coalesce` must
        still recover the `name` column's real values for every row."""
        df = pd.DataFrame({"name": ["Istiaia", "Limni"]})

        coalesced = ds._coalesce(df, "name:el", "name")

        assert list(coalesced) == ["Istiaia", "Limni"]

    def test_coalesce_recovers_per_row_where_first_would_drop_it(self):
        """The docstring's claimed distinction, demonstrated concretely:
        `name:el` IS present as a column (so `_first` picks it and returns it
        whole), but is missing for ONE row -- `_first` returns that row as NA
        (it never looks at `name` for that row), while `_coalesce` correctly
        falls back to `name` for that specific row."""
        df = pd.DataFrame({"name:el": ["Ιστιαία", None], "name": ["Istiaia", "Limni"]})

        coalesced = ds._coalesce(df, "name:el", "name")
        first = ds._first(df, "name:el", "name")

        assert list(coalesced) == ["Ιστιαία", "Limni"]
        assert pd.isna(first.iloc[1])   # _first drops the row _coalesce recovers

    def test_no_names_present_yields_all_na(self):
        df = pd.DataFrame({"other_col": [1, 2]})

        out = ds._coalesce(df, "name:el", "name")

        assert out.isna().all()


# --------------------------------------------------------------------------------
# main() flow
# --------------------------------------------------------------------------------
def _synthetic_osm_gdf():
    """A hand-built GeoDataFrame mixing polygon/point geometries and
    named/unnamed rows, in EPSG:4326 (matches what `ox.features_from_polygon`
    would return before `main()`'s own `.to_crs(ANALYSIS_CRS)`)."""
    import geopandas as gpd

    rows = [
        # polygon, named via plain "name" (no "name:el")
        {"geometry": Polygon([(23.10, 38.90), (23.11, 38.90), (23.11, 38.91),
                              (23.10, 38.91)]),
         "name": "Kokkinomilia", "place": "village", "population": "450"},
        # point, named via "name:el"
        {"geometry": Point(23.20, 38.80), "name:el": "Ασμήνη", "name": "Asmini",
         "place": "hamlet", "population": "120"},
        # unnamed point -- must be dropped
        {"geometry": Point(23.30, 38.70), "place": "isolated_dwelling"},
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


class TestMain:
    def test_polygon_rows_become_representative_points_and_unnamed_rows_dropped(
            self, monkeypatch, tmp_path, capsys):
        import geopandas as gpd

        synthetic = _synthetic_osm_gdf()
        monkeypatch.setattr(ds.ox, "features_from_polygon",
                            lambda aoi, tags: synthetic)
        monkeypatch.setattr(ds, "load_boundary_4326", lambda filt: object())
        out_path = tmp_path / "settlements.gpkg"
        monkeypatch.setattr(ds, "OUT_GPKG", out_path)

        ds.main()

        assert out_path.exists()
        result = gpd.read_file(out_path, layer="settlements")

        # 3 input rows, 1 unnamed -> 2 survive.
        assert len(result) == 2
        assert set(result["name"]) == {"Kokkinomilia", "Ασμήνη"}

        # The polygon row's output geometry is its representative_point(),
        # reprojected into ANALYSIS_CRS along with the rest of the frame.
        expected_pt = (gpd.GeoSeries([synthetic.geometry.iloc[0].representative_point()],
                                     crs="EPSG:4326")
                       .to_crs(ds.ANALYSIS_CRS).iloc[0])
        out_row = result[result["name"] == "Kokkinomilia"].iloc[0]
        assert out_row.geometry.x == pytest.approx(expected_pt.x, abs=1.0)
        assert out_row.geometry.y == pytest.approx(expected_pt.y, abs=1.0)

        captured = capsys.readouterr()
        assert "Saved 2 settlements" in captured.out
        assert "dropped 1 unnamed place" in captured.out

    def test_network_failure_wrapped_in_clear_runtime_error(self, monkeypatch, tmp_path):
        def _boom(aoi, tags):
            raise Exception("Overpass down")

        monkeypatch.setattr(ds.ox, "features_from_polygon", _boom)
        monkeypatch.setattr(ds, "load_boundary_4326", lambda filt: object())
        monkeypatch.setattr(ds, "OUT_GPKG", tmp_path / "settlements.gpkg")

        with pytest.raises(RuntimeError) as excinfo:
            ds.main()

        msg = str(excinfo.value)
        assert "Failed to download settlements from OpenStreetMap" in msg
        assert "Exception: Overpass down" in msg
