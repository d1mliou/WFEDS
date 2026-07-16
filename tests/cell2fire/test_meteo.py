"""Tests for scripts/cell2fire/meteo.py.

Covers the pure interpolation/helper math (`idw_series`, `slice_hours`,
`blows_to`) AND `fetch_era5_wind` / `fetch_era5_grid` with
`urllib.request.urlopen` mocked (no real network calls). `windfield_to_geojson`
/ `wind_to_geojson` (file I/O only, no network) are still left for a later pass.
"""

import json
from unittest.mock import Mock

import pytest

import meteo


class _FakeResponse:
    """Minimal stand-in for the `http.client.HTTPResponse` `urlopen` returns:
    supports the `with ... as resp:` context-manager protocol plus `.read()`
    (which is all `json.load(resp)` needs)."""

    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._data


def _hourly_block(n=3, precipitation=True):
    """A minimal Open-Meteo `hourly` sub-dict with exactly the keys
    `meteo._row()` reads."""
    block = {
        "time": [f"2021-08-05T{h:02d}:00" for h in range(n)],
        "wind_speed_10m": [10.0 + h for h in range(n)],
        "wind_direction_10m": [45.0 + h for h in range(n)],
        "wind_gusts_10m": [15.0 + h for h in range(n)],
        "temperature_2m": [22.0 + h for h in range(n)],
        "relative_humidity_2m": [55.0 + h for h in range(n)],
    }
    if precipitation:
        block["precipitation"] = [0.1 * h for h in range(n)]
    return block


def _row(time="2021-01-01T00:00", ws_kmh=10.0, wd_from=0.0, gust_kmh=0.0,
         temp_c=20.0, rh_pct=50.0, precip_mm=0.0):
    """One hourly dict matching meteo._row()'s schema."""
    return {"time": time, "ws_kmh": ws_kmh, "wd_from": wd_from, "gust_kmh": gust_kmh,
            "temp_c": temp_c, "rh_pct": rh_pct, "precip_mm": precip_mm}


class TestIdwSeries:
    def test_returns_exact_cell_when_target_within_1km(self):
        """`idw_series` short-circuits (`if d < 1_000.0: return rows`) and returns
        the ORIGINAL rows list unchanged for a cell effectively on top of the
        target. Placing the near cell exactly at the target (d=0) and a second,
        very different cell far away (~2500 km) proves the near cell -- and only
        it -- was picked, not a blend."""
        rows_near = [_row(ws_kmh=999.0, wd_from=123.0)]   # distinctive marker values
        meta_near = {"lat": 5.0, "lon": 10.0, "elev": 0}
        rows_far = [_row(ws_kmh=5.0, wd_from=45.0)]
        meta_far = {"lat": 20.0, "lon": 30.0, "elev": 0}

        result = meteo.idw_series(
            [(rows_near, meta_near), (rows_far, meta_far)], lat=5.0, lon=10.0)

        assert result is rows_near

    def test_circular_safe_averaging_avoids_degree_wraparound(self):
        """Two grid cells, EQUAL distance/weight from the target, with wind
        directions either side of the 350deg/10deg compass wrap. A naive linear
        average of the raw degrees would give (350+10)/2 = 180 (the wrong,
        opposite direction) -- meteo.py's own docstring calls this out explicitly.
        The real (U/V-component) IDW average must land near 0/360 instead."""
        rows1 = [_row(ws_kmh=10.0, wd_from=350.0)]
        rows2 = [_row(ws_kmh=10.0, wd_from=10.0)]
        # Symmetric about the target so both cells get equal IDW weight.
        meta1 = {"lat": 0.1, "lon": 0.0, "elev": 0}
        meta2 = {"lat": -0.1, "lon": 0.0, "elev": 0}

        out = meteo.idw_series([(rows1, meta1), (rows2, meta2)], lat=0.0, lon=0.0)

        assert len(out) == 1
        wd = out[0]["wd_from"]
        # near 0/360, and specifically NOT anywhere near the naive-wrong 180.
        assert wd < 10.0 or wd > 350.0
        assert abs(((wd - 180.0 + 180.0) % 360.0) - 180.0) > 90.0   # far from 180

    def test_weights_closer_cell_more_heavily(self):
        """A cell much closer to the target should dominate the blend: with
        wd_from=0 (close) and wd_from=90 (far), the IDW result should sit much
        closer to 0 than to the midpoint (45) or to 90."""
        rows_close = [_row(ws_kmh=10.0, wd_from=0.0)]
        rows_far = [_row(ws_kmh=10.0, wd_from=90.0)]
        meta_close = {"lat": 0.05, "lon": 0.0, "elev": 0}   # ~5.5 km away
        meta_far = {"lat": 1.0, "lon": 0.0, "elev": 0}      # ~111 km away

        out = meteo.idw_series(
            [(rows_close, meta_close), (rows_far, meta_far)], lat=0.0, lon=0.0)

        wd = out[0]["wd_from"]
        assert wd < 45.0            # closer to the near cell's 0 than the midpoint
        assert wd == pytest.approx(0.0, abs=5.0)   # much closer to 0 than to 90

    def test_deduplicates_grid_points_that_snap_to_the_same_era5_cell(self):
        """`results` is deduped to DISTINCT ERA5 cells keyed on
        (round(lat,3), round(lon,3)) BEFORE weighting -- two request points that
        snap to the identical cell must not double-count that cell's weight."""
        rows_a = [_row(ws_kmh=10.0, wd_from=0.0)]
        rows_b = [_row(ws_kmh=999.0, wd_from=999.0)]   # would-be duplicate is DROPPED
        same_meta_a = {"lat": 5.0000, "lon": 10.0000, "elev": 0}
        same_meta_b = {"lat": 5.0001, "lon": 10.0001, "elev": 0}   # rounds to same key
        far_rows = [_row(ws_kmh=20.0, wd_from=180.0)]
        far_meta = {"lat": 20.0, "lon": 30.0, "elev": 0}

        out = meteo.idw_series(
            [(rows_a, same_meta_a), (rows_b, same_meta_b), (far_rows, far_meta)],
            lat=5.0, lon=10.0)

        # Only 2 distinct cells contribute (the near dupe collapses to ONE, keeping
        # whichever was inserted first -- `cells.setdefault(key, rows)` -- so
        # rows_a wins over rows_b). That surviving cell sits at d=0 from the
        # target, so it short-circuits to the raw `rows` list, proving the
        # duplicate slot (rows_b) never independently entered the weighted blend.
        assert out is rows_a


class TestSliceHours:
    def test_returns_only_rows_within_the_inclusive_iso_range(self):
        rows = [{"time": f"2021-01-0{d}T00:00"} for d in range(1, 6)]

        result = meteo.slice_hours(rows, "2021-01-02T00:00", "2021-01-04T00:00")

        assert [r["time"] for r in result] == [
            "2021-01-02T00:00", "2021-01-03T00:00", "2021-01-04T00:00"]

    def test_empty_when_no_rows_in_range(self):
        rows = [{"time": "2021-01-01T00:00"}]
        assert meteo.slice_hours(rows, "2021-06-01T00:00", "2021-06-02T00:00") == []


class TestBlowsTo:
    def test_opposite_of_wd_from(self):
        assert meteo.blows_to(0.0) == pytest.approx(180.0)
        assert meteo.blows_to(90.0) == pytest.approx(270.0)

    def test_wraps_around_360(self):
        assert meteo.blows_to(350.0) == pytest.approx(170.0)
        assert meteo.blows_to(270.0) == pytest.approx(90.0)


class TestFetchEra5Wind:
    def test_parses_response_into_expected_row_structure(self, monkeypatch):
        payload = {"hourly": _hourly_block(3), "latitude": 38.9, "longitude": 23.1,
                   "elevation": 12.0}
        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["url"] = url
            captured["timeout"] = timeout
            return _FakeResponse(payload)

        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        rows, meta = meteo.fetch_era5_wind(38.9, 23.1, "2021-08-05", "2021-08-06")

        assert len(rows) == 3
        assert meta == {"lat": 38.9, "lon": 23.1, "elev": 12.0}
        for i, row in enumerate(rows):
            assert set(row) == {"time", "ws_kmh", "wd_from", "gust_kmh", "temp_c",
                                "rh_pct", "precip_mm"}
            assert row["time"] == payload["hourly"]["time"][i]
            assert row["ws_kmh"] == payload["hourly"]["wind_speed_10m"][i]
            assert row["wd_from"] == payload["hourly"]["wind_direction_10m"][i]
            assert row["gust_kmh"] == payload["hourly"]["wind_gusts_10m"][i]
            assert row["temp_c"] == payload["hourly"]["temperature_2m"][i]
            assert row["rh_pct"] == payload["hourly"]["relative_humidity_2m"][i]
            assert row["precip_mm"] == payload["hourly"]["precipitation"][i]

    def test_builds_url_with_expected_query_params(self, monkeypatch):
        payload = {"hourly": _hourly_block(1), "latitude": 38.9, "longitude": 23.1,
                   "elevation": 0.0}
        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResponse(payload)

        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        meteo.fetch_era5_wind(38.9, 23.1, "2021-08-05", "2021-08-06")

        url = captured["url"]
        assert url.startswith(meteo.ARCHIVE_URL)
        assert "latitude=38.9" in url
        assert "longitude=23.1" in url
        assert "start_date=2021-08-05" in url
        assert "end_date=2021-08-06" in url

    def test_missing_precipitation_key_falls_back_to_zero(self, monkeypatch):
        """`h.get("precipitation") or [0.0] * len(h["time"])` -- when the
        Open-Meteo response omits `precipitation` entirely, every row's
        `precip_mm` must default to 0.0 rather than raising a KeyError."""
        payload = {"hourly": _hourly_block(2, precipitation=False),
                   "latitude": 38.9, "longitude": 23.1, "elevation": 0.0}
        monkeypatch.setattr(meteo.urllib.request, "urlopen",
                            lambda url, timeout=None: _FakeResponse(payload))

        rows, _meta = meteo.fetch_era5_wind(38.9, 23.1, "2021-08-05", "2021-08-06")

        assert [r["precip_mm"] for r in rows] == [0.0, 0.0]

    def test_network_failure_wrapped_in_clear_runtime_error(self, monkeypatch):
        """A network failure is wrapped in a clear RuntimeError (Open-Meteo down /
        no internet / rate-limited), not propagated raw -- consistent with the
        data_prep download scripts. The original exception is chained (`from e`)."""
        def fake_urlopen(url, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(RuntimeError, match="Open-Meteo") as exc:
            meteo.fetch_era5_wind(38.9, 23.1, "2021-08-05", "2021-08-06")
        assert isinstance(exc.value.__cause__, OSError)      # original error chained


class TestFetchEra5Grid:
    def test_makes_exactly_one_batched_request_for_the_whole_grid(self, monkeypatch):
        """Despite the name, `fetch_era5_grid` does NOT make one HTTP request
        per grid point -- Open-Meteo accepts comma-separated coordinate lists,
        so nx*ny points are fetched in a SINGLE request (see the function's own
        docstring). Confirm this concretely: exactly one `urlopen` call, whose
        query params carry all nx*ny coordinates comma-joined."""
        nx, ny = 2, 2
        n_points = nx * ny
        payload = [
            {"hourly": _hourly_block(2), "latitude": 38.0 + i * 0.1,
             "longitude": 23.0 + i * 0.1, "elevation": float(i)}
            for i in range(n_points)
        ]
        fake_urlopen = Mock(side_effect=lambda url, timeout=None: _FakeResponse(payload))
        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        out = meteo.fetch_era5_grid(38.0, 23.0, 38.1, 23.1, nx=nx, ny=ny,
                                    start_date="2021-08-05", end_date="2021-08-06")

        assert fake_urlopen.call_count == 1
        url = fake_urlopen.call_args[0][0]
        # urlencode() percent-encodes the comma separator (',' -> '%2C').
        lat_param = url.split("latitude=")[1].split("&")[0]
        lon_param = url.split("longitude=")[1].split("&")[0]
        assert len(lat_param.split("%2C")) == n_points
        assert len(lon_param.split("%2C")) == n_points

        assert len(out) == n_points
        for i, (rows, meta) in enumerate(out):
            assert meta == {"lat": payload[i]["latitude"], "lon": payload[i]["longitude"],
                            "elev": payload[i]["elevation"]}
            assert len(rows) == 2
            assert rows[0]["time"] == payload[i]["hourly"]["time"][0]

    def test_network_failure_wrapped_in_clear_runtime_error(self, monkeypatch):
        """Same as fetch_era5_wind -- network failure wrapped in a clear RuntimeError."""
        monkeypatch.setattr(meteo.urllib.request, "urlopen",
                            lambda url, timeout=None: (_ for _ in ()).throw(OSError("down")))

        with pytest.raises(RuntimeError, match="Open-Meteo"):
            meteo.fetch_era5_grid(38.0, 23.0, 38.1, 23.1, nx=2, ny=2,
                                  start_date="2021-08-05", end_date="2021-08-06")


class TestBaseUrlPassthrough:
    """`base_url=` is new on both low-level fetchers, defaulting to ARCHIVE_URL
    so all pre-existing tests above stay valid unmodified."""

    def test_fetch_era5_wind_base_url_defaults_to_archive(self, monkeypatch):
        payload = {"hourly": _hourly_block(1), "latitude": 38.9, "longitude": 23.1,
                   "elevation": 0.0}
        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResponse(payload)

        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        meteo.fetch_era5_wind(38.9, 23.1, "2021-08-05", "2021-08-06")

        assert captured["url"].startswith(meteo.ARCHIVE_URL)

    def test_fetch_era5_wind_base_url_kwarg_switches_to_forecast(self, monkeypatch):
        payload = {"hourly": _hourly_block(1), "latitude": 38.9, "longitude": 23.1,
                   "elevation": 0.0}
        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResponse(payload)

        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        meteo.fetch_era5_wind(38.9, 23.1, "2021-08-05", "2021-08-06",
                              base_url=meteo.FORECAST_URL)

        assert captured["url"].startswith(meteo.FORECAST_URL)

    def test_fetch_era5_grid_base_url_defaults_to_archive(self, monkeypatch):
        payload = [{"hourly": _hourly_block(1), "latitude": 38.9, "longitude": 23.1,
                    "elevation": 0.0}]
        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResponse(payload)

        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        meteo.fetch_era5_grid(38.0, 23.0, 38.1, 23.1, nx=1, ny=1,
                              start_date="2021-08-05", end_date="2021-08-06")

        assert captured["url"].startswith(meteo.ARCHIVE_URL)

    def test_fetch_era5_grid_base_url_kwarg_switches_to_forecast(self, monkeypatch):
        payload = [{"hourly": _hourly_block(1), "latitude": 38.9, "longitude": 23.1,
                    "elevation": 0.0}]
        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResponse(payload)

        monkeypatch.setattr(meteo.urllib.request, "urlopen", fake_urlopen)

        meteo.fetch_era5_grid(38.0, 23.0, 38.1, 23.1, nx=1, ny=1,
                              start_date="2021-08-05", end_date="2021-08-06",
                              base_url=meteo.FORECAST_URL)

        assert captured["url"].startswith(meteo.FORECAST_URL)


class TestShiftDate:
    def test_shifts_forward(self):
        assert meteo._shift_date("2021-08-05", 1) == "2021-08-06"

    def test_shifts_backward(self):
        assert meteo._shift_date("2021-08-05", -6) == "2021-07-30"

    def test_crosses_month_and_year_boundary(self):
        assert meteo._shift_date("2021-12-31", 1) == "2022-01-01"


class TestArchiveCutoff:
    def test_is_today_minus_archive_lag_days(self, monkeypatch):
        monkeypatch.setattr(meteo, "utc_today_str", lambda: "2021-08-11")

        assert meteo._archive_cutoff() == meteo._shift_date(
            "2021-08-11", -meteo.ARCHIVE_LAG_DAYS)


class TestStitchRows:
    def test_concatenates_and_sorts_by_time(self):
        a = [_row(time="2021-08-05T02:00"), _row(time="2021-08-05T00:00")]
        b = [_row(time="2021-08-05T01:00")]

        out = meteo._stitch_rows(a, b)

        assert [r["time"] for r in out] == [
            "2021-08-05T00:00", "2021-08-05T01:00", "2021-08-05T02:00"]

    def test_first_list_wins_on_overlapping_time(self):
        """Archive is passed first -> an archive row wins over a forecast row for
        the same boundary hour, per the function's own docstring."""
        archive_row = _row(time="2021-08-05T00:00", ws_kmh=111.0)
        forecast_row = _row(time="2021-08-05T00:00", ws_kmh=999.0)

        out = meteo._stitch_rows([archive_row], [forecast_row])

        assert len(out) == 1
        assert out[0]["ws_kmh"] == 111.0


class TestAssertContiguousHourly:
    def test_passes_silently_when_exactly_1h_apart(self):
        rows = [_row(time=f"2021-08-05T0{h}:00") for h in range(3)]

        meteo.assert_contiguous_hourly(rows)   # must not raise

    def test_passes_on_empty_or_single_row(self):
        meteo.assert_contiguous_hourly([])
        meteo.assert_contiguous_hourly([_row()])

    def test_raises_weather_gap_error_on_a_gap(self):
        rows = [_row(time="2021-08-05T00:00"), _row(time="2021-08-05T02:00")]

        with pytest.raises(meteo.WeatherGapError):
            meteo.assert_contiguous_hourly(rows)

    def test_gap_message_includes_context_when_given(self):
        rows = [_row(time="2021-08-05T00:00"), _row(time="2021-08-05T03:00")]

        with pytest.raises(meteo.WeatherGapError, match="κύριο παράθυρο"):
            meteo.assert_contiguous_hourly(rows, context="κύριο παράθυρο")


class TestFetchWeatherWindRouting:
    """`fetch_weather_wind` makes no network calls itself -- it delegates to
    `fetch_era5_wind` (already covered by TestFetchEra5Wind). Mock
    `meteo.fetch_era5_wind` directly to isolate the ROUTING logic (archive vs.
    forecast vs. split) from the HTTP layer, and pin the archive/forecast
    boundary via `_archive_cutoff` so routing doesn't depend on wall-clock time."""

    def test_range_entirely_before_cutoff_uses_archive_only(self, monkeypatch):
        monkeypatch.setattr(meteo, "_archive_cutoff", lambda: "2021-08-10")
        calls = []

        def fake_fetch(lat, lon, start_date, end_date, timezone, timeout, base_url):
            calls.append((start_date, end_date, base_url))
            return [_row(time="2021-08-05T00:00")], {"lat": lat, "lon": lon, "elev": 0}

        monkeypatch.setattr(meteo, "fetch_era5_wind", fake_fetch)

        rows, meta = meteo.fetch_weather_wind(38.9, 23.1, "2021-08-01", "2021-08-05")

        assert calls == [("2021-08-01", "2021-08-05", meteo.ARCHIVE_URL)]
        assert len(rows) == 1

    def test_range_entirely_after_cutoff_uses_forecast_only(self, monkeypatch):
        monkeypatch.setattr(meteo, "_archive_cutoff", lambda: "2021-08-10")
        calls = []

        def fake_fetch(lat, lon, start_date, end_date, timezone, timeout, base_url):
            calls.append((start_date, end_date, base_url))
            return [_row(time="2021-08-15T00:00")], {"lat": lat, "lon": lon, "elev": 0}

        monkeypatch.setattr(meteo, "fetch_era5_wind", fake_fetch)

        rows, meta = meteo.fetch_weather_wind(38.9, 23.1, "2021-08-11", "2021-08-15")

        assert calls == [("2021-08-11", "2021-08-15", meteo.FORECAST_URL)]
        assert len(rows) == 1

    def test_range_spanning_cutoff_splits_and_stitches_both_sources(self, monkeypatch):
        """`start_date <= cutoff < end_date` -> TWO calls: archive up to the
        cutoff, forecast from cutoff+1 day -- and the two results are stitched
        into one gap-free series."""
        monkeypatch.setattr(meteo, "_archive_cutoff", lambda: "2021-08-10")
        calls = []

        def fake_fetch(lat, lon, start_date, end_date, timezone, timeout, base_url):
            calls.append((start_date, end_date, base_url))
            if base_url == meteo.ARCHIVE_URL:
                return [_row(time="2021-08-10T00:00", ws_kmh=1.0)], {"lat": lat, "lon": lon, "elev": 0}
            return [_row(time="2021-08-11T00:00", ws_kmh=2.0)], {"lat": lat, "lon": lon, "elev": 0}

        monkeypatch.setattr(meteo, "fetch_era5_wind", fake_fetch)

        rows, meta = meteo.fetch_weather_wind(38.9, 23.1, "2021-08-08", "2021-08-12")

        assert calls == [
            ("2021-08-08", "2021-08-10", meteo.ARCHIVE_URL),
            ("2021-08-11", "2021-08-12", meteo.FORECAST_URL),
        ]
        assert [r["time"] for r in rows] == ["2021-08-10T00:00", "2021-08-11T00:00"]


class TestFetchWeatherGridRouting:
    """Same routing responsibilities as fetch_weather_wind, but merging two
    fetch_era5_grid calls by REQUEST INDEX rather than by meta -- mock
    `meteo.fetch_era5_grid` directly to isolate routing/merging from the HTTP
    layer."""

    def test_range_entirely_before_cutoff_uses_archive_only(self, monkeypatch):
        monkeypatch.setattr(meteo, "_archive_cutoff", lambda: "2021-08-10")
        calls = []

        def fake_fetch(min_lat, min_lon, max_lat, max_lon, **kw):
            calls.append((kw["start_date"], kw["end_date"], kw["base_url"]))
            return [([_row(time="2021-08-05T00:00")], {"lat": 1.0, "lon": 2.0, "elev": 0})]

        monkeypatch.setattr(meteo, "fetch_era5_grid", fake_fetch)

        out = meteo.fetch_weather_grid(38.0, 23.0, 38.1, 23.1, nx=2, ny=2,
                                       start_date="2021-08-01", end_date="2021-08-05")

        assert calls == [("2021-08-01", "2021-08-05", meteo.ARCHIVE_URL)]
        assert len(out) == 1

    def test_range_entirely_after_cutoff_uses_forecast_only(self, monkeypatch):
        monkeypatch.setattr(meteo, "_archive_cutoff", lambda: "2021-08-10")
        calls = []

        def fake_fetch(min_lat, min_lon, max_lat, max_lon, **kw):
            calls.append((kw["start_date"], kw["end_date"], kw["base_url"]))
            return [([_row(time="2021-08-15T00:00")], {"lat": 1.0, "lon": 2.0, "elev": 0})]

        monkeypatch.setattr(meteo, "fetch_era5_grid", fake_fetch)

        out = meteo.fetch_weather_grid(38.0, 23.0, 38.1, 23.1, nx=2, ny=2,
                                       start_date="2021-08-11", end_date="2021-08-15")

        assert calls == [("2021-08-11", "2021-08-15", meteo.FORECAST_URL)]
        assert len(out) == 1

    def test_range_spanning_cutoff_makes_two_matching_shape_calls_and_merges_by_index(
            self, monkeypatch):
        """Both calls MUST use an identical bbox/nx/ny (guaranteeing matching
        `coords` order/length in fetch_era5_grid) -- and the two grid points' row
        lists are merged pairwise BY POSITION, not by re-keyed lat/lon (the two
        APIs may snap the same nominal point to different model cells)."""
        monkeypatch.setattr(meteo, "_archive_cutoff", lambda: "2021-08-10")
        calls = []

        def fake_fetch(min_lat, min_lon, max_lat, max_lon, **kw):
            calls.append((min_lat, min_lon, max_lat, max_lon, kw["nx"], kw["ny"],
                         kw["start_date"], kw["end_date"], kw["base_url"]))
            if kw["base_url"] == meteo.ARCHIVE_URL:
                return [([_row(time="2021-08-10T00:00", ws_kmh=1.0)],
                         {"lat": 38.0, "lon": 23.0, "elev": 0}),
                        ([_row(time="2021-08-10T00:00", ws_kmh=10.0)],
                         {"lat": 38.1, "lon": 23.1, "elev": 5})]
            return [([_row(time="2021-08-11T00:00", ws_kmh=2.0)],
                     {"lat": 38.05, "lon": 23.05, "elev": 1}),
                    ([_row(time="2021-08-11T00:00", ws_kmh=20.0)],
                     {"lat": 38.15, "lon": 23.15, "elev": 6})]

        monkeypatch.setattr(meteo, "fetch_era5_grid", fake_fetch)

        out = meteo.fetch_weather_grid(38.0, 23.0, 38.1, 23.1, nx=2, ny=1,
                                       start_date="2021-08-08", end_date="2021-08-12")

        assert len(calls) == 2
        archive_call, forecast_call = calls
        assert archive_call[:6] == forecast_call[:6]          # identical bbox/nx/ny
        assert archive_call[6:] == ("2021-08-08", "2021-08-10", meteo.ARCHIVE_URL)
        assert forecast_call[6:] == ("2021-08-11", "2021-08-12", meteo.FORECAST_URL)

        assert len(out) == 2
        rows0, meta0 = out[0]
        assert [r["time"] for r in rows0] == ["2021-08-10T00:00", "2021-08-11T00:00"]
        assert meta0 == {"lat": 38.05, "lon": 23.05, "elev": 1}   # forecast meta wins
        rows1, meta1 = out[1]
        assert [r["time"] for r in rows1] == ["2021-08-10T00:00", "2021-08-11T00:00"]
