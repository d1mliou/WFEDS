"""Hourly surface wind - historical (ERA5 via the Open-Meteo ARCHIVE API) and,
for dates the archive doesn't have yet, FORECAST (the same provider's separate
forecast API - confirmed live: identical params/response shape, no key needed).

Imported as a module by `build_cell2fire_instance.py` to write a REAL time-varying
`Weather.csv` (one row per hour) instead of a constant PLACEHOLDER wind:

    from meteo import fetch_weather_wind, fetch_weather_grid

Source: https://open-meteo.com/  (ERA5 reanalysis, ~25 km grid, free, no key).

ARCHIVE vs FORECAST: the archive lags real time by ~`ARCHIVE_LAG_DAYS` (it is a
reanalysis product, not published instantly). `fetch_weather_wind`/
`fetch_weather_grid` are the entry points that route each date to whichever API
actually has it, transparently stitching the two into one gap-free hourly series
when a request spans the boundary (e.g. a "live" fire: recent past + the next few
days). `fetch_era5_wind`/`fetch_era5_grid` (below) still exist as the low-level,
single-source fetchers - `base_url=` picks which API each call hits.

WIND DIRECTION CONVENTION (critical - "coming from" vs "blowing to"):
  Both ERA5 `wind_direction_10m` AND Cell2Fire's `WD` column are METEOROLOGICAL:
  the direction the wind COMES FROM (0=N, 90=E, 180=S, 270=W). Fire spreads in the
  OPPOSITE, "blows-to" direction (= WD + 180). So a fire that runs NORTH needs a
  SOUTHERLY wind, WD ~= 180. We therefore pass ERA5 `wind_direction_10m` STRAIGHT
  THROUGH to Cell2Fire's `WD` (same convention - no +/-180 flip).
  Verified empirically: Cell2Fire WD=45 (from NE) -> the scar elongates toward SW.

TIME ZONE: request the archive in **UTC** by default so the hours line up with the
VIIRS `acq_at` timestamps (VIIRS is UTC). Do NOT mix a local-time wind series with
UTC satellite detections when checking spread direction.

Wind speed is returned in **km/h** (Cell2Fire's `WS` unit).
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone as _tz

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"     # reanalysis: past dates, published with a lag
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"        # same shape, recent/future dates
# The archive is a reanalysis product, not published instantly - dates newer than
# this many days back may not exist there yet (silently null-padded, not an
# error). Beyond this, fetch_weather_* route to FORECAST_URL instead. Verified
# live against both endpoints 2026-07-07 (params/response shape identical).
ARCHIVE_LAG_DAYS = 6
HOURLY_VARS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m",     # the variables pulled for every request
               "wind_direction_10m", "wind_gusts_10m", "precipitation")


def _fetch_json(url, timeout):
    """GET `url` and parse the JSON body, wrapping ANY network/parse failure in a
    clear RuntimeError (no internet, Open-Meteo down, rate-limited, or a malformed
    response) instead of a raw traceback. Mirrors the network-error wrapping added
    to the data_prep download scripts, so the one weather-fetching module fails as
    gracefully as the rest of the pipeline."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except Exception as e:
        raise RuntimeError(
            "Failed to fetch weather from the Open-Meteo API -- usually no "
            "internet, the API being down, or rate-limiting. Re-run when the "
            f"connection is back.\nOriginal error: {type(e).__name__}: {e}"
        ) from e


def utc_today_str():
    return datetime.now(_tz.utc).strftime("%Y-%m-%d")          # "now" always in UTC - never host-local time


def _shift_date(date_str, days):
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def _archive_cutoff():
    """Newest date (YYYY-MM-DD) the archive is trusted for."""
    return _shift_date(utc_today_str(), -ARCHIVE_LAG_DAYS)


def _stitch_rows(*row_lists):
    """Concatenate hourly row lists, keeping the FIRST row seen per `time` (pass
    archive before forecast so archive wins on any accidental boundary overlap),
    sorted by time."""
    seen = {}
    for rows in row_lists:
        for r in rows:
            seen.setdefault(r["time"], r)               # first-writer-wins de-dup on the boundary hour
    return [seen[t] for t in sorted(seen)]               # sorted -> a clean, gap-checkable hourly series


class WeatherGapError(RuntimeError):
    """A stitched hourly weather series has a gap other than exactly 1 hour."""


def assert_contiguous_hourly(rows, context=""):
    """Raise WeatherGapError if consecutive rows aren't exactly 1h apart - a gap
    would silently corrupt fwi.spinup_daily_codes' positional 24h rain-sum slice."""
    for a, b in zip(rows, rows[1:]):
        gap = datetime.fromisoformat(b["time"]) - datetime.fromisoformat(a["time"])
        if gap != timedelta(hours=1):                    # anything but exactly 1h = a hole in the series
            raise WeatherGapError(
                f"Weather data has a gap{f' ({context})' if context else ''} between "
                f"{a['time']} and {b['time']} - likely right at the archive/forecast "
                f"boundary. Try again shortly, or pick a start time a bit further from now.")


def _row(h, i):
    """One hourly dict from an Open-Meteo `hourly` block (shared by the fetchers)."""
    return {"time": h["time"][i],
            "ws_kmh": h["wind_speed_10m"][i],
            "wd_from": h["wind_direction_10m"][i],       # meteorological convention: degrees wind comes FROM
            "gust_kmh": h["wind_gusts_10m"][i],
            "temp_c": h["temperature_2m"][i],
            "rh_pct": h["relative_humidity_2m"][i],
            "precip_mm": (h.get("precipitation") or [0.0] * len(h["time"]))[i]}   # some responses omit precip entirely


def fetch_era5_wind(lat, lon, start_date, end_date, timezone="UTC", timeout=60,
                    base_url=ARCHIVE_URL):
    """Fetch hourly wind for a point + date range (inclusive, YYYY-MM-DD) from a
    SINGLE source (`base_url` - the archive by default, or FORECAST_URL). Prefer
    `fetch_weather_wind` (below) unless you specifically want one source only.

    Returns (rows, meta):
      rows: list of dicts {time, ws_kmh, wd_from, gust_kmh, temp_c, rh_pct}
            (`wd_from` = degrees the wind COMES FROM; see the module docstring).
      meta: {lat, lon, elev} of the actual model grid cell used (snaps to the grid).
    """
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "kmh", "timezone": timezone,
    })
    data = _fetch_json(f"{base_url}?{query}", timeout)
    h = data["hourly"]
    rows = [_row(h, i) for i in range(len(h["time"]))]
    meta = {"lat": data.get("latitude"), "lon": data.get("longitude"),   # the ACTUAL model cell (snapped, not the request point)
            "elev": data.get("elevation")}
    return rows, meta


def fetch_era5_grid(min_lat, min_lon, max_lat, max_lon, nx=4, ny=4,
                    start_date=None, end_date=None, timezone="UTC", timeout=90,
                    base_url=ARCHIVE_URL):
    """Fetch hourly wind over a GRID of nx*ny points spanning a bbox, from a
    SINGLE source (`base_url` - the archive by default, or FORECAST_URL). Prefer
    `fetch_weather_grid` (below) unless you specifically want one source only.

    Open-Meteo accepts comma-separated coordinates and returns one result per point,
    so this is a single request. Returns a list of (rows, meta) - one per grid point
    (meta carries the model cell lat/lon/elev the point snapped to). NB ERA5 is
    ~25 km, so several grid points may snap to the SAME cell over a ~20 km fire;
    dedupe on (meta['lat'], meta['lon']) if you want distinct cells."""
    lats = [min_lat + (max_lat - min_lat) * i / max(ny - 1, 1) for i in range(ny)]   # evenly spaced request grid...
    lons = [min_lon + (max_lon - min_lon) * j / max(nx - 1, 1) for j in range(nx)]   # ...over the bbox (nx*ny points)
    coords = [(la, lo) for la in lats for lo in lons]
    query = urllib.parse.urlencode({
        "latitude": ",".join(f"{la:.4f}" for la, _ in coords),          # one comma-separated multi-point request
        "longitude": ",".join(f"{lo:.4f}" for _, lo in coords),
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "kmh", "timezone": timezone,
    })
    data = _fetch_json(f"{base_url}?{query}", timeout)
    items = data if isinstance(data, list) else [data]    # a single-point request returns a dict, not a list
    out = []
    for d in items:
        h = d["hourly"]
        rows = [_row(h, i) for i in range(len(h["time"]))]
        out.append((rows, {"lat": d.get("latitude"), "lon": d.get("longitude"),
                           "elev": d.get("elevation")}))
    return out


def fetch_weather_wind(lat, lon, start_date, end_date, timezone="UTC", timeout=60):
    """Like `fetch_era5_wind`, but routes each date to whichever source actually
    has it (archive vs forecast), transparently stitching a gap-free hourly
    series when the range spans the boundary. This is the function
    build_cell2fire_instance.py should call."""
    cutoff = _archive_cutoff()
    if end_date <= cutoff:                                # whole range in the past -> archive only
        rows, meta = fetch_era5_wind(lat, lon, start_date, end_date, timezone,
                                     timeout, base_url=ARCHIVE_URL)
    elif start_date > cutoff:                             # whole range recent/future -> forecast only
        rows, meta = fetch_era5_wind(lat, lon, start_date, end_date, timezone,
                                     timeout, base_url=FORECAST_URL)
    else:                                                  # spans the boundary -> fetch both, stitch into one series
        a_rows, _ = fetch_era5_wind(lat, lon, start_date, cutoff, timezone,
                                    timeout, base_url=ARCHIVE_URL)
        f_rows, meta = fetch_era5_wind(lat, lon, _shift_date(cutoff, 1), end_date,
                                       timezone, timeout, base_url=FORECAST_URL)
        rows = _stitch_rows(a_rows, f_rows)
    return rows, meta


def fetch_weather_grid(min_lat, min_lon, max_lat, max_lon, nx=4, ny=4,
                       start_date=None, end_date=None, timezone="UTC", timeout=90):
    """Like `fetch_era5_grid`, but routes/stitches archive+forecast per date, same
    as `fetch_weather_wind`. Merges by REQUEST INDEX, not by re-keying on `meta`
    (the two APIs may snap the same nominal point to different underlying model
    cells) - both calls use IDENTICAL bbox/nx/ny, so `fetch_era5_grid`'s `coords`
    list (deterministic from those args) has the same order/length both times.
    This is the function build_cell2fire_instance.py should call; `idw_series`
    needs no changes - the output is the same `list[(rows, meta)]` shape."""
    cutoff = _archive_cutoff()
    kw = dict(nx=nx, ny=ny, timezone=timezone, timeout=timeout)
    if end_date <= cutoff:
        return fetch_era5_grid(min_lat, min_lon, max_lat, max_lon,
                               start_date=start_date, end_date=end_date,
                               base_url=ARCHIVE_URL, **kw)
    if start_date > cutoff:
        return fetch_era5_grid(min_lat, min_lon, max_lat, max_lon,
                               start_date=start_date, end_date=end_date,
                               base_url=FORECAST_URL, **kw)
    arch = fetch_era5_grid(min_lat, min_lon, max_lat, max_lon,          # same bbox/nx/ny on both calls...
                           start_date=start_date, end_date=cutoff,
                           base_url=ARCHIVE_URL, **kw)
    fcst = fetch_era5_grid(min_lat, min_lon, max_lat, max_lon,          # ...so zip() pairs up the SAME grid point
                           start_date=_shift_date(cutoff, 1), end_date=end_date,
                           base_url=FORECAST_URL, **kw)
    return [(_stitch_rows(a_rows, f_rows), f_meta)
            for (a_rows, _a_meta), (f_rows, f_meta) in zip(arch, fcst)]


def idw_series(results, lat, lon, power=2.0):
    """IDW-interpolate a grid fetch at (lat, lon) -> one hourly series.

    `results` = output of fetch_era5_grid. Deduplicates to DISTINCT ERA5 cells,
    weights each by 1/distance^power from the target, and blends per hour:
      * WIND via U/V COMPONENTS (circular-safe - never average raw degrees: the
        350/10 wrap would average to 180 instead of 0);
      * temperature / RH / gusts / precipitation linearly.
    Returns rows in the same schema as fetch_era5_wind. If the target sits on a
    cell (< ~1 km), that cell's series is returned as-is."""
    cells = {}
    for rows, meta in results:
        key = (round(meta["lat"], 3), round(meta["lon"], 3))    # de-dup: several request points can snap to one cell
        cells.setdefault(key, rows)
    m_per_deg = 111_320.0                                        # metres per degree latitude (constant)
    coslat = math.cos(math.radians(lat))                         # shrinks metres-per-degree-longitude at this latitude
    weights, series = [], []
    for (cla, clo), rows in cells.items():
        d = math.hypot((cla - lat) * m_per_deg, (clo - lon) * m_per_deg * coslat)   # planar distance target -> cell (m)
        if d < 1_000.0:                                          # target sits on a cell -> no interpolation needed
            return rows
        weights.append(1.0 / d ** power)                         # IDW weight
        series.append(rows)
    n = min(len(r) for r in series)
    out = []
    for i in range(n):
        wsum = usum = vsum = 0.0
        acc = {"temp_c": 0.0, "rh_pct": 0.0, "gust_kmh": 0.0, "precip_mm": 0.0}
        for w, rows in zip(weights, series):
            r = rows[i]
            if r["ws_kmh"] is None:
                continue
            th = math.radians(r["wd_from"])
            usum += w * (-r["ws_kmh"] * math.sin(th))   # +east flow component
            vsum += w * (-r["ws_kmh"] * math.cos(th))   # +north flow component
            for k in acc:
                acc[k] += w * (r[k] or 0.0)              # scalar fields: plain weighted sum (no wrap-around issue)
            wsum += w
        if wsum == 0.0:
            out.append(dict(series[0][i]))               # no valid cell this hour -> fall back to the first series
            continue
        u, v = usum / wsum, vsum / wsum                  # weighted-mean wind vector
        out.append({"time": series[0][i]["time"],
                    "ws_kmh": math.hypot(u, v),           # vector magnitude = blended speed
                    "wd_from": (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0,   # vector angle -> back to "from" convention
                    **{k: acc[k] / wsum for k in acc}})
    return out


def windfield_to_geojson(results, arrows_path, start_iso=None, end_iso=None,
                         arrow_m_per_kmh=40.0):
    """Grid wind FIELD -> GeoJSON arrows (one LineString per grid cell per hour).

    `results` = output of fetch_era5_grid. Deduplicates to DISTINCT ERA5 cells (so a
    fine request grid over a coarse ERA5 grid still yields one arrow per real cell).
    Optional [start_iso, end_iso] limits the hours. Animate on `time` in QGIS; each
    arrow points toward `blows_to`, length proportional to wind speed."""
    seen, cells = set(), []
    for rows, meta in results:
        key = (round(meta["lat"], 3), round(meta["lon"], 3))
        if key not in seen:                              # keep only the first request point per real ERA5 cell
            seen.add(key)
            cells.append((rows, meta))
    feats = []
    for rows, meta in cells:
        lat, lon = meta["lat"], meta["lon"]
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))   # longitude degrees shrink with latitude
        for r in rows:
            if start_iso and not (start_iso <= r["time"] <= (end_iso or r["time"])):
                continue
            length = (r["ws_kmh"] or 0.0) * arrow_m_per_kmh       # arrow length in metres, proportional to speed
            th = math.radians(blows_to(r["wd_from"]))             # arrow points where the wind BLOWS TO, not from
            dlon = (length * math.sin(th)) / m_per_deg_lon
            dlat = (length * math.cos(th)) / m_per_deg_lat
            feats.append({"type": "Feature",
                          "properties": {"time": r["time"], "ws_kmh": r["ws_kmh"],
                                         "wd_from": r["wd_from"],
                                         "blows_to": round(blows_to(r["wd_from"]), 1),
                                         "gust_kmh": r["gust_kmh"], "cell_lat": lat,
                                         "cell_lon": lon},
                          "geometry": {"type": "LineString",       # a short line FROM the cell TOWARD blows_to
                                       "coordinates": [[lon, lat],
                                                       [lon + dlon, lat + dlat]]}})
    os.makedirs(os.path.dirname(arrows_path), exist_ok=True)
    with open(arrows_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name",                        # explicit CRS84 tag (RFC 7946 default, stated for clarity)
                           "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                   "features": feats}, f, ensure_ascii=False)
    return arrows_path, len(cells)


def slice_hours(rows, start_iso, end_iso):
    """Sub-list of rows whose ISO `time` is in [start_iso, end_iso] (string compare
    works on ISO-8601). Use to pull just the fire's first day, etc."""
    return [r for r in rows if start_iso <= r["time"] <= end_iso]


def blows_to(wd_from):
    """Direction the wind blows TOWARD (degrees) = the fire's push direction."""
    return (wd_from + 180.0) % 360.0            # meteorological "from" -> geographic "to" (+180, wrapped)


def wind_to_geojson(rows, meta, points_path, arrows_path=None, arrow_m_per_kmh=40.0):
    """Write the ERA5 wind as GeoJSON for QGIS (EPSG:4326 / CRS84, RFC 7946).

    Two files (arrows optional):
      * points_path - one Point per hour at the ERA5 cell, attributes
        time / ws_kmh / wd_from / blows_to / gust_kmh / temp_c / rh_pct
        (animate with the QGIS Temporal Controller on `time`, or rotate a marker
        by `wd_from`).
      * arrows_path - one short LineString per hour FROM the cell toward `blows_to`
        (the fire-push direction), length = ws_kmh * arrow_m_per_kmh metres, so the
        fan of arrows shows the wind veering + its speed.
    """
    lon, lat = meta["lon"], meta["lat"]
    props = lambda r: {
        "time": r["time"], "ws_kmh": r["ws_kmh"], "wd_from": r["wd_from"],
        "blows_to": round(blows_to(r["wd_from"]), 1), "gust_kmh": r["gust_kmh"],
        "temp_c": r["temp_c"], "rh_pct": r["rh_pct"],
    }

    def _write(path, features):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fc = {"type": "FeatureCollection",
              "crs": {"type": "name",
                      "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
              "features": features}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)

    _write(points_path, [                                # one static point per hour (same cell, changing attributes)
        {"type": "Feature", "properties": props(r),
         "geometry": {"type": "Point", "coordinates": [lon, lat]}}
        for r in rows])

    if arrows_path:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        feats = []
        for r in rows:
            length = (r["ws_kmh"] or 0.0) * arrow_m_per_kmh
            th = math.radians(blows_to(r["wd_from"]))     # bearing wind blows toward
            dlon = (length * math.sin(th)) / m_per_deg_lon
            dlat = (length * math.cos(th)) / m_per_deg_lat
            feats.append({"type": "Feature", "properties": props(r),
                          "geometry": {"type": "LineString",
                                       "coordinates": [[lon, lat],
                                                       [lon + dlon, lat + dlat]]}})
        _write(arrows_path, feats)
    return points_path, arrows_path


if __name__ == "__main__":
    # Smoke test: North Evia 2021 fire location, first day (UTC, matches VIIRS).
    rows, meta = fetch_era5_wind(38.873, 23.288, "2021-08-05", "2021-08-06")
    print(f"ERA5 cell: {meta}")
    print(f"{len(rows)} hourly rows; sample:")
    for r in rows[:3]:
        print(f"  {r['time']}  WS {r['ws_kmh']:.1f} km/h  WD(from) {r['wd_from']:.0f} "
              f"-> blows to {blows_to(r['wd_from']):.0f} deg")
