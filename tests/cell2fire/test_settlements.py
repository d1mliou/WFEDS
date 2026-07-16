"""Tests for scripts/cell2fire/settlements.py.

settlements.py was rewritten this session into a simple 2-source union: the
official ELSTAT Istiaia layer (`_official()`, reading ATRISK_SETTLEMENTS) plus
the OSM Mantoudi layer (`refuges()`, reading OSM_SETTLEMENTS). `all_settlements()`
plain-concatenates the two -- NO distance-based de-dup any more, because the two
layers are documented as geographically disjoint. If the OSM layer can't be read
(missing file, corrupt file, whatever), `all_settlements()` catches the broad
`Exception`, prints a `WARNING settlements: ...` line, and falls back to the
official-only rows. `load_atrisk()` builds on `all_settlements()` and coerces the
`pop` column (comma-thousands strings, "n/a", etc.) to numeric via
`pd.to_numeric(..., errors="coerce")`.

Both ATRISK_SETTLEMENTS and OSM_SETTLEMENTS are read as plain module globals
inside each function body (not bound default arguments), so monkeypatching
`settlements.ATRISK_SETTLEMENTS` / `settlements.OSM_SETTLEMENTS` before calling
a function works correctly.
"""

import pandas as pd
import pytest

import settlements

from tests.helpers.geodata_factories import make_tiny_points_gdf


class TestOfficialStripsWhitespace:
    def test_name_is_stripped(self, tmp_path, monkeypatch):
        official = make_tiny_points_gdf([
            {"geometry": (0, 0), "NAME_OIK": "  Village A  ", "census2021": 100},
        ])
        path = tmp_path / "official.geojson"
        official.to_file(path, driver="GeoJSON")
        monkeypatch.setattr(settlements, "ATRISK_SETTLEMENTS", path)

        result = settlements._official()

        assert result["name"].iloc[0] == "Village A"


class TestAllSettlementsUnion:
    def test_row_count_is_plain_sum_no_dedup(self, tmp_path, monkeypatch):
        official = make_tiny_points_gdf([
            {"geometry": (0, 0), "NAME_OIK": "Alpha", "census2021": 10},
            {"geometry": (100, 100), "NAME_OIK": "Beta", "census2021": 20},
        ])
        # osm[0] is a deliberate near-duplicate of official[0]: same name, same
        # point. If any dedup logic remained, the merged count would be 4, not 5.
        osm = make_tiny_points_gdf([
            {"geometry": (0, 0), "name": "Alpha", "population": 10},
            {"geometry": (200, 200), "name": "Gamma", "population": 30},
            {"geometry": (300, 300), "name": "Delta", "population": 40},
        ])
        official_path = tmp_path / "official.geojson"
        osm_path = tmp_path / "osm.gpkg"
        official.to_file(official_path, driver="GeoJSON")
        osm.to_file(osm_path, driver="GPKG")
        monkeypatch.setattr(settlements, "ATRISK_SETTLEMENTS", official_path)
        monkeypatch.setattr(settlements, "OSM_SETTLEMENTS", osm_path)

        result = settlements.all_settlements()

        assert len(result) == 2 + 3
        # Both the official "Alpha" and the OSM near-duplicate "Alpha" survive.
        assert (result["name"] == "Alpha").sum() == 2


class TestAllSettlementsOsmFallback:
    def test_missing_osm_file_falls_back_and_warns(self, tmp_path, monkeypatch, capsys):
        official = make_tiny_points_gdf([
            {"geometry": (0, 0), "NAME_OIK": "A", "census2021": 1},
            {"geometry": (1, 1), "NAME_OIK": "B", "census2021": 2},
        ])
        official_path = tmp_path / "official.geojson"
        official.to_file(official_path, driver="GeoJSON")
        missing_osm_path = tmp_path / "does_not_exist.gpkg"  # never created
        monkeypatch.setattr(settlements, "ATRISK_SETTLEMENTS", official_path)
        monkeypatch.setattr(settlements, "OSM_SETTLEMENTS", missing_osm_path)

        result = settlements.all_settlements()

        assert len(result) == 2
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_corrupt_osm_file_falls_back_and_warns(self, tmp_path, monkeypatch, capsys):
        official = make_tiny_points_gdf([
            {"geometry": (0, 0), "NAME_OIK": "A", "census2021": 1},
            {"geometry": (1, 1), "NAME_OIK": "B", "census2021": 2},
        ])
        official_path = tmp_path / "official.geojson"
        official.to_file(official_path, driver="GeoJSON")
        corrupt_osm_path = tmp_path / "corrupt.gpkg"
        corrupt_osm_path.write_bytes(b"not a real gpkg")
        monkeypatch.setattr(settlements, "ATRISK_SETTLEMENTS", official_path)
        monkeypatch.setattr(settlements, "OSM_SETTLEMENTS", corrupt_osm_path)

        result = settlements.all_settlements()

        assert len(result) == 2
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestLoadAtrisk:
    def test_population_parsing(self, tmp_path, monkeypatch):
        official = make_tiny_points_gdf([
            {"geometry": (0, 0), "NAME_OIK": "A", "census2021": "1,250"},
            {"geometry": (1, 1), "NAME_OIK": "B", "census2021": "n/a"},
        ])
        official_path = tmp_path / "official.geojson"
        official.to_file(official_path, driver="GeoJSON")
        # No OSM file at all -> all_settlements() falls back to official-only,
        # which is all this test needs (it only checks the official rows).
        missing_osm_path = tmp_path / "does_not_exist.gpkg"
        monkeypatch.setattr(settlements, "ATRISK_SETTLEMENTS", official_path)
        monkeypatch.setattr(settlements, "OSM_SETTLEMENTS", missing_osm_path)

        result = settlements.load_atrisk()

        row_a = result.loc[result["NAME_OIK"] == "A", "census2021"].iloc[0]
        row_b = result.loc[result["NAME_OIK"] == "B", "census2021"].iloc[0]
        assert row_a == pytest.approx(1250)
        assert pd.isna(row_b)
