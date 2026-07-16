"""Fire + evacuation TIMELINE DASHBOARD (custom HTML, from scratch).

Third redesign (2026-07-02). The folium/TimestampedGeoJson approach was judged
unreadable twice; this builds a self-contained HTML page (Leaflet from CDN + a
hand-rolled panel & chart - no folium) designed around the question "what would
someone need, hour by hour, to understand what is happening?":
  * MAP: the CURRENT state only - predicted fire (red), real fire so far
    (violet, VIIRS x scar reconstruction), evacuation routes + village status.
  * TIME: play button + scrubber + a big "+N ώρες" readout; ←/→ keys.
  * PANEL: this hour in words - fire size (+growth), real size, wind (arrow +
    Beaufort-style text), evacuation counts, expandable per-village list.
  * ΣΥΜΒΑΝΤΑ: what CHANGED this hour ("X αποκλείστηκε", "+12 κλειστά τμήματα").
  * CHART: predicted-vs-real burned area over time (the two curves ARE the
    validation story); the cursor is synced with the map and click-seeks.
  * Validation headline stats in a collapsible box (from validation_metrics.json).
Palette per the dataviz skill (validated: red #e34948 / violet #4a3aa7).

Run (after fire_timesteps.py and validate_overlay.py):
    python scripts/cell2fire/visualize_fire.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

from _paths import ATRISK_SETTLEMENTS, DATA_DIR
from cell2fire_adapter import read_asc_header

VIIRS = DATA_DIR / "Real_Fire_Data" / "VIIRS" / "VIRS_dataset_egsa_dhmos.shp"
BURNED = DATA_DIR / "Real_Fire_Data" / "Burned_Area" / "Burned_area_20210829.shp"
# The real fire's hourly extents arrive as STATIC DATA, precomputed once by the
# offline validation step (validation/real_progression.py -> export_hourly) -
# the live pipeline reads this file and never imports validation code. If the
# file is missing, the overlay is simply omitted (the dashboard still builds).
REAL_HOURLY = DATA_DIR / "Real_Fire_Data" / "real_progression_hourly.geojson"

# Scenario override (see fire_timesteps.py): render a run from its own folder into
# its own HTML, leaving the canonical dashboard intact. The run's own
# instance/ (weather t0 + window box) is used when present.
SCENARIO_LABEL = os.environ.get("WFEDS_SCENARIO_LABEL", "")
# The 2021 real-fire overlays (VIIRS, scar, arrival reconstruction, validation box)
# only make sense against the 2021 event - user/hypothetical runs set this to 0.
REAL_OVERLAY = os.environ.get("WFEDS_REAL_OVERLAY", "1") != "0"
_CANON_INST = DATA_DIR / "Fire" / "cell2fire" / "instance"
_SCEN = os.environ.get("WFEDS_SCENARIO_DIR")
if _SCEN:
    _SCEN = Path(_SCEN)
    PERIMS = _SCEN / "perimeters.geojson"
    ISOCHRONES = _SCEN / "isochrones.geojson"
    EVAC_DIR = _SCEN
    INSTANCE = _SCEN / "instance" if (_SCEN / "instance").exists() else _CANON_INST
else:
    PERIMS = DATA_DIR / "Fire" / "cell2fire" / "perimeters.geojson"
    ISOCHRONES = DATA_DIR / "Fire" / "cell2fire" / "output" / "isochrones.geojson"
    EVAC_DIR = DATA_DIR / "Evacuation_c2f"
    INSTANCE = _CANON_INST
WEATHER = INSTANCE / "Weather.csv"
TS_GPKG = EVAC_DIR / "timestep_evacuation.gpkg"
TS_JSON = EVAC_DIR / "timestep_summary.json"
VAL_JSON = EVAC_DIR / "validation_metrics.json"
OUT_HTML = EVAC_DIR / "fire_timesteps.html"

COMPASS = ["Β", "ΒΑ", "Α", "ΝΑ", "Ν", "ΝΔ", "Δ", "ΒΔ"]


def _round_coords(o, nd=5):
    if isinstance(o, (list, tuple)):
        return [_round_coords(x, nd) for x in o]
    if isinstance(o, float):
        return round(o, nd)
    return o


def _clean(geom, simp, min_part, min_hole, close=0):
    """Simplify + drop specks/micro-holes; `close` (m) first dissolves fragmented
    footprints into coherent patches (display-only; metrics use the originals)."""
    from shapely.geometry import MultiPolygon, Polygon

    if close:
        geom = geom.simplify(20).buffer(close, join_style=2).buffer(-close, join_style=2)
    geom = geom.simplify(simp)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    out = [Polygon(p.exterior,
                   [h for h in p.interiors if Polygon(h).area >= min_hole])
           for p in polys if p.area >= min_part]
    if not out:
        return geom
    return MultiPolygon(out) if len(out) > 1 else out[0]


def _geo(geom, simplify_m):
    """EPSG:2100 geometry -> compact GeoJSON geometry dict in EPSG:4326."""
    g = gpd.GeoSeries([geom.simplify(simplify_m)], crs=2100).to_crs(4326).iloc[0]
    d = g.__geo_interface__
    return {"type": d["type"], "coordinates": _round_coords(d["coordinates"])}


def _fill_holes(geom):
    """Drop interior holes (unburned islands) -> the solid outer footprint."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    filled = [Polygon(p.exterior) for p in polys]
    return unary_union(filled) if filled else geom


def _fmt_km(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 1)


def main():
    perims = gpd.read_file(PERIMS).to_crs(2100).sort_values("period").reset_index(drop=True)
    hours = int(perims["period"].max())
    w = pd.read_csv(WEATHER)
    t0 = pd.to_datetime(w["datetime"].iloc[0])
    areas = perims.geometry.area / 1e6

    # Per-hour OUTER fronts (isochrones) + exterior-only suppressability class
    # (WFEDS suppression patch) - optional: absent for runs without --out-intensity.
    iso = (gpd.read_file(ISOCHRONES).to_crs(2100).sort_values("period")
           if ISOCHRONES.exists() else None)

    summary = json.loads(TS_JSON.read_text(encoding="utf-8"))
    ts_at = gpd.read_file(TS_GPKG, layer="at_risk")
    ts_at_ll = ts_at.to_crs(4326)
    try:
        ts_rt = gpd.read_file(TS_GPKG, layer="routes").to_crs(2100)
    except Exception:
        ts_rt = None
    # {iso 'YYYY-MM-DDTHH:MM' -> (extent EPSG:2100, exact km2)} from the static
    # precomputed file; {} when absent or for user runs (overlay omitted).
    real_hourly = {}
    if REAL_OVERLAY:
        if REAL_HOURLY.exists():
            rg = gpd.read_file(REAL_HOURLY).to_crs(2100)
            # GDAL parses the ISO time_utc strings back into datetimes on read -
            # normalise to the lookup-key string format, else every .get() misses
            # and the overlay would be SILENTLY empty.
            real_hourly = {pd.to_datetime(r["time_utc"]).strftime("%Y-%m-%dT%H:%M"):
                           (r.geometry, float(r["km2"]))
                           for _, r in rg.iterrows()}
        else:
            print(f"NB: {REAL_HOURLY.name} not found - real-fire overlay omitted. "
                  "Generate it once: python scripts/validation/real_progression.py")

    hours_js, prev_state, prev_edges = [], {}, None
    for h in range(hours + 1):
        when = t0 + pd.Timedelta(hours=h)
        s = summary[h]
        wi = min(h, len(w) - 1)          # 25 perimeters (0..24) vs 24 weather rows
        ws, wd = float(w["WS"].iloc[wi]), float(w["WD"].iloc[wi])
        real, real_km2 = real_hourly.get(when.strftime("%Y-%m-%dT%H:%M"), (None, 0.0))

        at_rows, cur_state = [], {}
        sel = ts_at_ll[ts_at["period"] == h]
        for _, r in sel.iterrows():
            name = str(r.get("NAME_OIK", "?")).strip()
            km = _fmt_km(r.get("route_km"))
            ref = r.get("refuge")
            ref = None if (ref is None or (isinstance(ref, float) and math.isnan(ref))) else str(ref)
            st = r.get("status", "cut_off")
            pop = r.get("census2021")
            at_rows.append({"n": name, "p": None if pd.isna(pop) else int(pop),
                            "s": st, "r": ref, "k": km,
                            "lat": round(r.geometry.y, 5), "lon": round(r.geometry.x, 5)})
            cur_state[name] = (st, ref, km)

        routes = []
        if ts_rt is not None:
            for _, r in ts_rt[ts_rt["period"] == h].iterrows():
                routes.append({"g": _geo(r.geometry, 10), "o": str(r["origin"]),
                               "r": str(r["refuge"]), "k": round(r["length_m"] / 1000, 1)})

        events = []
        for name, (st, ref, km) in cur_state.items():
            if name not in prev_state:
                if st == "ok":
                    events.append({"c": "ok", "t": f"Σε κίνδυνο: {name} - εκκενώνεται προς "
                                                   f"{ref} ({km} km)"})
                elif st == "cut_off":
                    events.append({"c": "cut", "t": f"Σε κίνδυνο και ΑΠΟΚΛΕΙΣΜΕΝΟΣ: {name}"})
                else:
                    events.append({"c": "imp", "t": f"{name}: μέσα στο μέτωπο"})
            else:
                pst, pref, pkm = prev_state[name]
                if pst != "cut_off" and st == "cut_off":
                    extra = f" (είχε διαδρομή {pkm} km)" if pkm else ""
                    events.append({"c": "cut", "t": f"ΑΠΟΚΛΕΙΣΤΗΚΕ: {name}{extra}"})
                elif pst == "cut_off" and st == "ok":
                    events.append({"c": "ok", "t": f"Ξανά προσβάσιμος: {name} → {ref}"})
                elif st == "ok" and pref and ref and ref != pref:
                    events.append({"c": "ok", "t": f"{name}: αλλαγή καταφυγίου {pref} → {ref}"})
        if prev_edges is not None and s["edges_removed"] > prev_edges:
            events.append({"c": "road", "t": f"+{s['edges_removed'] - prev_edges} κλειστά "
                                             f"τμήματα δρόμου (σύνολο {s['edges_removed']})"})
        prev_state, prev_edges = cur_state, s["edges_removed"]

        hours_js.append({
            "t": when.strftime("%d/%m %H:%M"),
            "sim": _geo(_clean(perims.geometry[h], 40, 2e4, 2e4, close=150), 0),
            "simKm2": round(float(areas[h]), 1),
            "dKm2": round(float(areas[h] - areas[h - 1]), 1) if h else 0.0,
            "real": (_geo(_clean(real, 100, 8e4, 3e5, close=250), 0)
                     if real is not None else None),
            "realKm2": round(real_km2, 1),
            "ws": round(ws), "wd": round(wd),
            "wdName": COMPASS[round(wd / 45.0) % 8], "blow": round((wd + 180) % 360),
            "atRisk": at_rows, "routes": routes, "events": events,
            "stats": {"n": s["at_risk"], "pop": s["population"], "ok": s["routed"],
                      "cut": s["cut_off"], "imp": s["impacted"],
                      "edges": s["edges_removed"]},
        })

    if REAL_OVERLAY:
        b = gpd.read_file(BURNED).to_crs(2100)
        scar = _geo(_clean(b.union_all(), 150, 5e4, 5e4), 0)
        v = gpd.read_file(VIIRS).to_crs(4326)
        v["t"] = pd.to_datetime(v["acq_at_s"])
        vwin = v[(v["t"] >= t0) & (v["t"] <= t0 + pd.Timedelta(hours=hours))]
        viirs = [[round(p.y, 5), round(p.x, 5), int((t - t0).total_seconds() // 3600)]
                 for p, t in zip(vwin.geometry, vwin["t"])]
        val = json.loads(VAL_JSON.read_text(encoding="utf-8")) if VAL_JSON.exists() else {}
    else:
        scar, viirs, val = None, [], {}

    # the model window (the fire CANNOT leave this box - draw it so the straight
    # clipped edges are self-explanatory)
    from shapely.geometry import box

    wtr, wnrows, wncols, _ = read_asc_header(str(INSTANCE / "Forest.asc"))
    wx0, wy1 = wtr.c, wtr.f
    wx1, wy0 = wx0 + wncols * wtr.a, wy1 + wnrows * wtr.e
    window = box(wx0, wy0, wx1, wy1)

    # ALL settlements (official Istiaia layer + OSM places for the rest of the
    # study area; name + population) - permanent base layer; the per-hour at-risk
    # dots draw ON TOP with their status colours.
    from settlements import all_settlements
    st_all = all_settlements().to_crs(4326)
    settlements = []
    for _, r in st_all.iterrows():
        p = r.get("pop")
        try:
            p = None if p is None or pd.isna(p) else int(float(str(p).replace(",", "")))
        except (ValueError, TypeError):
            p = None
        settlements.append({"n": str(r.get("name", "?")).strip(), "p": p,
                            "lat": round(r.geometry.y, 5),
                            "lon": round(r.geometry.x, 5)})

    # per-hour suppressability class (for the panel stat only, exterior-only).
    iso_js = []
    if iso is not None:
        for _, r in iso.iterrows():
            iso_js.append({"p": int(r["period"]), "pct4": round(float(r["pct4"])),
                           "dom": int(r["dom"])})

    # arrival-time bands: the ring newly burned each hour (hole-filled outer
    # footprints, differenced) -> filled on a red ramp by arrival hour. The
    # perimeters nest cleanly, so the differences are clean rings.
    outers, bands_js = [], []
    for h in range(hours + 1):
        o = _fill_holes(_clean(perims.geometry[h], 40, 2e4, 2e4, close=150))
        outers.append(o)
        band = o if h == 0 else o.difference(outers[h - 1])
        if not band.is_empty:
            bands_js.append({"h": h, "g": _geo(band, 20)})

    centre_ll = gpd.GeoSeries([perims.geometry[hours].centroid], crs=2100).to_crs(4326).iloc[0]
    data = {"hours": hours_js, "scar": scar, "viirs": viirs, "val": val, "iso": iso_js,
            "bands": bands_js, "scenarioLabel": SCENARIO_LABEL,
            "settlements": settlements,
            "window": _geo(window, 0), "windowKm": round((wx1 - wx0) / 1000),
            "t0": t0.strftime("%d/%m/%Y %H:%M"),
            "centre": [round(centre_ll.y, 5), round(centre_ll.x, 5)]}

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                                   separators=(",", ":")))
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"Saved dashboard ({hours + 1} h, {size_mb:.1f} MB) -> {OUT_HTML}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Βόρεια Εύβοια 2021 - φωτιά & εκκένωση ανά ώρα</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {
    --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,.10);
    --sim: #e34948; --real: #4a3aa7; --route: #2a78d6;
    --ok: #2a78d6; --cut: #d03b3b; --imp: #0b0b0b; --viirs: #eda100;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font: 14px/1.45 system-ui, -apple-system,
               "Segoe UI", sans-serif; color: var(--ink); background: var(--page); }
  #app { display: flex; height: 100%; }
  #map { flex: 1; min-width: 0; }
  #side { width: 372px; flex: none; overflow-y: auto; padding: 14px;
          background: var(--page); border-left: 1px solid var(--border); }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; }
  h1 { font-size: 15px; margin: 0 0 2px; }
  .sub { color: var(--ink2); font-size: 12px; margin-bottom: 10px; }
  #clock { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }
  #clockSub { color: var(--ink2); font-size: 12px; margin-bottom: 8px; }
  #timeRow { display: flex; gap: 10px; align-items: center; }
  #play { width: 40px; height: 40px; border-radius: 50%; border: 1px solid var(--border);
          background: var(--surface); font-size: 16px; cursor: pointer; flex: none; }
  #play:hover { background: #f0efec; }
  #slider { flex: 1; accent-color: var(--ink); }
  .kv { display: flex; justify-content: space-between; margin: 3px 0;
        font-variant-numeric: tabular-nums; }
  .kv .k { color: var(--ink2); }
  .kv .v b { font-size: 15px; }
  .chip { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
          margin-right: 6px; vertical-align: baseline; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 6px; border: 1.5px solid #fff; box-shadow: 0 0 0 1px var(--border); }
  #events { max-height: 168px; overflow-y: auto; }
  #events div { padding: 3px 0 3px 2px; border-bottom: 1px dashed var(--grid);
                font-size: 13px; }
  #events div:last-child { border-bottom: 0; }
  .ev-cut { color: var(--cut); font-weight: 600; }
  .ev-ok { color: #1c5cab; }
  .ev-imp { font-weight: 600; }
  .ev-road { color: var(--ink2); }
  .secTitle { font-size: 12px; text-transform: uppercase; letter-spacing: .4px;
              color: var(--muted); margin: 0 0 6px; }
  #chartWrap { position: relative; }
  #chartTip { position: absolute; pointer-events: none; background: var(--surface);
              border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
              font-size: 12px; display: none; white-space: nowrap;
              box-shadow: 0 1px 4px rgba(0,0,0,.12); font-variant-numeric: tabular-nums; }
  .legend { font-size: 12px; color: var(--ink2); margin-bottom: 4px; }
  details { font-size: 13px; }
  summary { cursor: pointer; color: var(--ink2); }
  details table { width: 100%; border-collapse: collapse; margin-top: 6px;
                  font-variant-numeric: tabular-nums; }
  details td { padding: 2px 4px 2px 0; border-bottom: 1px solid var(--grid);
               font-size: 12.5px; vertical-align: top; }
  .toggles { position: absolute; z-index: 1000; top: 10px; right: 10px;
             background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: 8px 10px; font-size: 12.5px;
             min-width: 190px; box-shadow: 0 1px 4px rgba(0,0,0,.10); }
  .toggles label { display: block; cursor: pointer; margin: 2px 0; }
  .windArrow { display: inline-block; font-size: 15px; color: var(--ink2); }
  .leaflet-tooltip { font: 12.5px system-ui, sans-serif; }
  .settleTip { font: 10px system-ui, sans-serif; color: #52514e;
               background: rgba(255,255,255,.82); border: none; box-shadow: none;
               padding: 0 3px; }
  .settleTip::before { display: none; }
</style>
</head>
<body>
<div id="app">
  <div id="map"></div>
  <div id="side">
    <h1>Βόρεια Εύβοια 2021 - φωτιά &amp; εκκένωση</h1>
    <div class="sub">Πρόβλεψη Cell2Fire (σπορά από το παρατηρημένο μέτωπο) έναντι
      της πραγματικής εξέλιξης · έναρξη <span id="t0"></span> UTC</div>

    <div class="card">
      <div id="clock">+0 ώρες</div>
      <div id="clockSub"></div>
      <div id="timeRow">
        <button id="play" title="αναπαραγωγή (space)">&#9654;</button>
        <input id="slider" type="range" min="0" max="0" value="0" step="1">
      </div>
    </div>

    <div class="card">
      <div class="secTitle">Αυτή την ώρα</div>
      <div class="kv"><span class="k"><span class="chip" style="background:var(--sim)"></span>Πρόβλεψη</span>
        <span class="v"><b id="simKm2"></b> km² <span id="dKm2" style="color:var(--ink2)"></span></span></div>
      <div class="kv" id="realRow"><span class="k"><span class="chip" style="background:var(--real)"></span>Πραγματική φωτιά</span>
        <span class="v"><b id="realKm2"></b> km²</span></div>
      <div class="kv"><span class="k">Άνεμος</span>
        <span class="v"><span class="windArrow" id="wArrow">&#10148;</span>
        <span id="wind"></span></span></div>
      <div class="kv" id="pct4Row" style="display:none"><span class="k"><span class="chip"
        style="background:#7a1010"></span>Μέτωπο κλάσης 4 (μη αντιμετωπίσιμο)</span>
        <span class="v"><b id="pct4"></b>%</span></div>
      <div class="kv"><span class="k">Κλειστά τμήματα δρόμου</span><span class="v"><b id="edges"></b></span></div>
      <hr style="border:0;border-top:1px solid var(--grid);margin:8px 0">
      <div class="kv"><span class="k">Οικισμοί σε κίνδυνο</span>
        <span class="v"><b id="nRisk"></b> (<span id="popV"></span> κάτοικοι)</span></div>
      <div class="kv"><span class="k"><span class="dot" style="background:var(--ok)"></span>Εκκενώνονται</span><span class="v"><b id="nOk"></b></span></div>
      <div class="kv"><span class="k"><span class="dot" style="background:var(--cut)"></span>Αποκλεισμένοι</span><span class="v"><b id="nCut"></b></span></div>
      <details id="villDetails"><summary>αναλυτικά ανά οικισμό</summary>
        <table id="villTable"></table></details>
    </div>

    <div class="card">
      <div class="secTitle">Συμβάντα αυτή την ώρα</div>
      <div id="evList"></div>
    </div>

    <div class="card">
      <div class="secTitle">Καμένη έκταση στον χρόνο (km²)</div>
      <div class="legend"><span class="chip" style="background:var(--sim)"></span>πρόβλεψη
        <span id="realLegend">&nbsp;&nbsp;<span class="chip" style="background:var(--real)"></span>πραγματική</span>
        <span style="float:right;color:var(--muted)">κλικ = μετάβαση</span></div>
      <div id="chartWrap"><svg id="chart" width="316" height="150"></svg>
        <div id="chartTip"></div></div>
    </div>

    <div class="card">
      <details><summary><b>Επικύρωση</b> - συνολική αξιολόγηση της πρόβλεψης</summary>
        <div id="valBody" style="margin-top:8px"></div>
      </details>
    </div>
  </div>
</div>

<script>
const D = __DATA__;
const H = D.hours, N = H.length - 1;

/* ---------- map ---------- */
const map = L.map('map', { zoomControl: true, zoomSnap: 0.25 });
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  { attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 18 }).addTo(map);
['scar','real','sim','routes','vill'].forEach((p, i) => {
  map.createPane(p); map.getPane(p).style.zIndex = 380 + i * 10; });
map.createPane('band'); map.getPane('band').style.zIndex = 385;  // arrival fill: above scar, below real
map.createPane('settle'); map.getPane('settle').style.zIndex = 415;  // all settlements: under at-risk dots

/* ALL settlements: name (population), permanent labels, toggleable */
const settleLayer = L.layerGroup((D.settlements || []).map(s =>
  L.circleMarker([s.lat, s.lon], { pane:'settle', radius:3, color:'#8a897f',
      weight:1, fillColor:'#8a897f', fillOpacity:.85 })
    .bindTooltip(s.n + ' (' + (s.p === null ? '-' : s.p.toLocaleString('el')) + ')',
                 { permanent:true, direction:'right', offset:[4, 0],
                   className:'settleTip' }))).addTo(map);

L.geoJSON({type:'Feature',geometry:D.window}, { pane:'scar',
  style: {color:'#52514e', weight:1.5, dashArray:'4 7', fill:false} })
  .bindTooltip('Όριο παραθύρου προσομοίωσης (' + D.windowKm + '×' + D.windowKm +
    ' km) - η πρόβλεψη δεν μπορεί να βγει έξω από αυτό· όπου το μέτωπο το ' +
    'ακουμπά, κόβεται τεχνητά ίσιο', {sticky:true}).addTo(map);
/* the 2021 real-event overlays exist only when rendering the reference scenario */
const HAS_REAL = !!D.scar;
const scarLayer = HAS_REAL ? L.geoJSON({type:'Feature',geometry:D.scar}, { pane:'scar',
  style: {color:'#898781', weight:1, fillColor:'#898781', fillOpacity:.07, dashArray:'2 3'} })
  .bindTooltip('Τελικός πραγματικός κάμπος (29/8): 165.8 km²', {sticky:true}) : null;
const viirsLayer = L.layerGroup(D.viirs.map(p =>
  L.circleMarker([p[0], p[1]], { radius:2.5, color:'#b57b00', weight:1,
    fillColor:'var(--viirs)', fillOpacity:.75 })
   .bindTooltip('VIIRS +' + p[2] + ' ώρες', {sticky:true})));
if (scarLayer) scarLayer.addTo(map);

let simL = null, realL = null, routeL = null, villL = null, bandL = null;
const stName = { ok: 'εκκενώνεται', cut_off: 'ΑΠΟΚΛΕΙΣΜΕΝΟΣ', impacted: 'μέσα στο μέτωπο' };
const stCol = { ok: '#2a78d6', cut_off: '#d03b3b', impacted: '#0b0b0b' };
/* arrival-time BANDS: the ring newly burned each hour, filled on a RED sequential
   ramp (light salmon early -> dark red late); shown cumulatively up to the current
   hour, so the fire fills outward with a time gradient as you scrub. */
const ISO = D.iso || [], BANDS = D.bands || [], HAS_BANDS = BANDS.length > 0;
let showBands = true;
const _hx = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
function redRamp(p) {                                   // light salmon -> dark red by hour
  const a = _hx('#fee0d2'), b = _hx('#67000d'), t = N ? p / N : 0;
  return 'rgb(' + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',') + ')';
}

function setHour(h, keepView) {
  h = Math.max(0, Math.min(N, h)); cur = h;
  const d = H[h];
  [simL, realL, routeL, villL, bandL].forEach(l => l && map.removeLayer(l));
  realL = d.real ? L.geoJSON({type:'Feature',geometry:d.real}, { pane:'real',
      style: {color:'#4a3aa7', weight:2, dashArray:'6 4', fillColor:'#4a3aa7',
              fillOpacity:.10} })
      .bindTooltip('Πραγματική φωτιά έως τώρα: ' + d.realKm2 + ' km²', {sticky:true})
      .addTo(map) : null;
  const isoNow = ISO.length ? ISO.find(o => o.p === h) : null;
  bandL = (showBands && HAS_BANDS) ? L.layerGroup(
      BANDS.filter(bd => bd.h <= h).map(bd =>
        L.geoJSON({type:'Feature',geometry:bd.g}, { pane:'band',
          style: {stroke:false, fillColor: redRamp(bd.h), fillOpacity:.72} })
        .bindTooltip('Έφτασε +' + bd.h + (bd.h === 1 ? ' ώρα' : ' ώρες'),
                     {sticky:true}))).addTo(map) : null;
  simL = L.geoJSON({type:'Feature',geometry:d.sim}, { pane:'sim',
      style: {color:'#7a0a0a', weight:1.5, fill:false} })
      .bindTooltip('Πρόβλεψη έως +' + h + 'ω: ' + d.simKm2 + ' km² (+' + d.dKm2 +
                   ' αυτή την ώρα)', {sticky:true}).addTo(map);
  routeL = L.layerGroup(d.routes.flatMap(r => [
      L.geoJSON({type:'Feature',geometry:r.g}, { pane:'routes',
        style:{color:'#ffffff', weight:6, opacity:.9} }),
      L.geoJSON({type:'Feature',geometry:r.g}, { pane:'routes',
        style:{color:'#2a78d6', weight:3, opacity:.95} })
        .bindTooltip(r.o + ' → ' + r.r + ' (' + r.k + ' km)', {sticky:true}) ]))
    .addTo(map);
  villL = L.layerGroup(d.atRisk.map(a =>
      L.circleMarker([a.lat, a.lon], { pane:'vill', radius:7,
        color:'#ffffff', weight:1.5, fillColor: stCol[a.s], fillOpacity:.95 })
      .bindTooltip('<b>' + a.n + '</b>' + (a.p ? ' (' + a.p + ' κατ.)' : '') + '<br>' +
        stName[a.s] + (a.s === 'ok' && a.r ? ' προς ' + a.r + ' (' + a.k + ' km)' : ''),
        {sticky:true}))).addTo(map);

  /* panel */
  document.getElementById('clock').textContent = '+' + h + (h === 1 ? ' ώρα' : ' ώρες');
  document.getElementById('clockSub').textContent = d.t + ' UTC';
  document.getElementById('slider').value = h;
  simKm2.textContent = d.simKm2.toFixed(1);
  dKm2.textContent = h ? '(+' + d.dKm2.toFixed(1) + ')' : '(αρχικό μέτωπο)';
  realKm2.textContent = d.realKm2.toFixed(1);
  wind.textContent = d.ws + ' km/h από ' + d.wdName + ' (' + d.wd + '°)';
  wArrow.style.transform = 'rotate(' + (d.blow - 90) + 'deg)';
  edges.textContent = d.stats.edges;
  if (isoNow && isoNow.dom) {
    document.getElementById('pct4Row').style.display = '';
    pct4.textContent = isoNow.pct4;
  } else document.getElementById('pct4Row').style.display = 'none';
  nRisk.textContent = d.stats.n; popV.textContent = d.stats.pop.toLocaleString('el');
  nOk.textContent = d.stats.ok; nCut.textContent = d.stats.cut + (d.stats.imp ?
      ' (+' + d.stats.imp + ' στο μέτωπο)' : '');
  villTable.innerHTML = d.atRisk.slice().sort((a, b) => a.s.localeCompare(b.s))
    .map(a => '<tr><td><span class="dot" style="background:' + stCol[a.s] + '"></span>' +
      a.n + '</td><td style="text-align:right">' + (a.p ?? '-') + '</td><td>' +
      (a.s === 'ok' ? '→ ' + a.r + ' (' + a.k + ' km)' : stName[a.s]) + '</td></tr>')
    .join('');
  const evs = d.events.length ? d.events : [{c:'road', t:'- καμία αλλαγή'}];
  evList.innerHTML = evs.map(e => '<div class="ev-' + e.c + '">' + e.t + '</div>').join('');
  drawCursor();
}

/* ---------- time controls ---------- */
let cur = 0, playing = null;
const slider = document.getElementById('slider');
slider.max = N;
slider.addEventListener('input', () => setHour(+slider.value, true));
const playBtn = document.getElementById('play');
function stop() { clearInterval(playing); playing = null; playBtn.innerHTML = '&#9654;'; }
playBtn.addEventListener('click', () => {
  if (playing) { stop(); return; }
  if (cur >= N) setHour(0, true);
  playBtn.innerHTML = '&#10074;&#10074;';
  playing = setInterval(() => { cur < N ? setHour(cur + 1, true) : stop(); }, 1100);
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') setHour(cur + 1, true);
  else if (e.key === 'ArrowLeft') setHour(cur - 1, true);
  else if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
});

/* ---------- toggles ---------- */
const tg = L.control({position:'topright'});
tg.onAdd = () => {
  const div = L.DomUtil.create('div', 'toggles');
  div.innerHTML =
    (HAS_REAL ?
    '<label><input type="checkbox" id="tgReal" checked> πραγματική φωτιά</label>' +
    '<label><input type="checkbox" id="tgScar" checked> τελικός κάμπος</label>' +
    '<label><input type="checkbox" id="tgViirs"> ανιχνεύσεις VIIRS</label>' : '') +
    '<label><input type="checkbox" id="tgSettle" checked> οικισμοί + ονόματα/πληθυσμός</label>' +
    (HAS_BANDS ?
    '<label><input type="checkbox" id="tgBands" checked> ζώνες κατά χρόνο άφιξης</label>' +
    '<div style="color:#898781;max-width:200px;border-top:1px solid #e1e0d9;' +
    'margin-top:4px;padding-top:4px">' +
    '<div>χρώμα = πότε έφτασε η φωτιά:</div>' +
    '<div style="height:9px;border-radius:2px;margin:3px 0;background:linear-gradient(' +
    'to right,' + redRamp(0) + ',' + redRamp(N) + ')"></div>' +
    '<div style="display:flex;justify-content:space-between"><span>ώρα 0</span>' +
    '<span>ώρα ' + N + '</span></div></div>' : '') +
    '<div style="color:#898781;max-width:180px;border-top:1px solid #e1e0d9;' +
    'margin-top:4px;padding-top:4px"><span style="display:inline-block;width:14px;' +
    'height:9px;border:1.5px dashed #52514e;margin-right:5px"></span>όριο του μοντέλου ' +
    '- έξω από αυτό η πρόβλεψη δεν μπορεί να συνεχίσει</div>';
  L.DomEvent.disableClickPropagation(div);
  return div;
};
tg.addTo(map);
if (HAS_REAL) {
  document.getElementById('tgScar').addEventListener('change', e =>
    e.target.checked ? scarLayer.addTo(map) : map.removeLayer(scarLayer));
  document.getElementById('tgViirs').addEventListener('change', e =>
    e.target.checked ? viirsLayer.addTo(map) : map.removeLayer(viirsLayer));
  document.getElementById('tgReal').addEventListener('change', e => {
    showReal = e.target.checked; setHour(cur, true); });
}
if (HAS_BANDS) document.getElementById('tgBands').addEventListener('change', e => {
  showBands = e.target.checked; setHour(cur, true); });
document.getElementById('tgSettle').addEventListener('change', e =>
  e.target.checked ? settleLayer.addTo(map) : map.removeLayer(settleLayer));
let showReal = true;
const _setHour = setHour;
setHour = function(h, k) { _setHour(h, k); if (!showReal && realL) map.removeLayer(realL); };

/* ---------- chart ---------- */
const svg = document.getElementById('chart');
const CW = 316, CH = 150, ML = 34, MR = 10, MT = 8, MB = 20;
const maxY = Math.ceil(Math.max(...H.map(d => Math.max(d.simKm2, d.realKm2))) / 20) * 20;
const X = h => ML + (CW - ML - MR) * h / N;
const Y = v => MT + (CH - MT - MB) * (1 - v / maxY);
function path(key) { return H.map((d, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ',' +
  Y(d[key]).toFixed(1)).join(''); }
let sh = '';
[0, maxY / 2, maxY].forEach(v => { sh += '<line x1="' + ML + '" x2="' + (CW - MR) +
  '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="#e1e0d9"/>' +
  '<text x="' + (ML - 5) + '" y="' + (Y(v) + 4) + '" text-anchor="end" font-size="10"' +
  ' fill="#898781">' + v + '</text>'; });
for (let h = 0; h <= N; h += 4) sh += '<text x="' + X(h) + '" y="' + (CH - 5) +
  '" text-anchor="middle" font-size="10" fill="#898781">+' + h + 'ω</text>';
if (HAS_REAL) sh += '<path d="' + path('realKm2') + '" fill="none" stroke="#4a3aa7"' +
      ' stroke-width="2" stroke-dasharray="5 3"/>';
sh += '<path d="' + path('simKm2') + '" fill="none" stroke="#e34948" stroke-width="2"/>';
sh += '<line id="cursor" y1="' + MT + '" y2="' + (CH - MB) + '" stroke="#52514e"/>' +
      '<circle id="cSim" r="4" fill="#e34948" stroke="#fff" stroke-width="1.5"/>' +
      (HAS_REAL ? '<circle id="cReal" r="4" fill="#4a3aa7" stroke="#fff" stroke-width="1.5"/>' : '');
svg.innerHTML = sh;
function drawCursor() {
  const x = X(cur), d = H[cur];
  cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
  cSim.setAttribute('cx', x); cSim.setAttribute('cy', Y(d.simKm2));
  if (HAS_REAL) { cReal.setAttribute('cx', x); cReal.setAttribute('cy', Y(d.realKm2)); }
}
const tip = document.getElementById('chartTip');
svg.addEventListener('mousemove', e => {
  const r = svg.getBoundingClientRect();
  const h = Math.max(0, Math.min(N, Math.round((e.clientX - r.left - ML) /
            ((CW - ML - MR) / N))));
  const d = H[h];
  tip.style.display = 'block';
  tip.style.left = Math.min(X(h) + 8, CW - 130) + 'px';
  tip.style.top = '6px';
  tip.innerHTML = '<b>+' + h + ' ώρες</b><br>πρόβλεψη ' + d.simKm2.toFixed(1) +
                  (HAS_REAL ? ' · πραγματική ' + d.realKm2.toFixed(1) : '');
});
svg.addEventListener('mouseleave', () => tip.style.display = 'none');
svg.addEventListener('click', e => {
  const r = svg.getBoundingClientRect();
  setHour(Math.round((e.clientX - r.left - ML) / ((CW - ML - MR) / N)), true);
});

/* ---------- validation box ---------- */
(function() {
  const v = D.val || {}, f = v.forecast_only, a = v.arrival_time || {};
  if (!f) { valBody.innerHTML = '<i>δεν βρέθηκαν μετρικές</i>'; return; }
  valBody.innerHTML =
    '<div class="kv"><span class="k">Κάλυψη της νέας πραγματικής εξάπλωσης</span><b>' +
    f.recall_pct + '%</b></div>' +
    '<div class="kv"><span class="k">Πρόβλεψε ' + f.pred_km2 + ' km² · παρατηρήθηκαν ' +
    f.footprint_km2 + ' km²</span><b>precision ' + f.precision_pct + '%</b></div>' +
    '<div class="kv"><span class="k">Απόσταση από τις ανιχνεύσεις (διάμεσος)</span><b>' +
    f.nn_median_km + ' km</b></div>' +
    '<div class="kv"><span class="k">Χρονισμός (διάμεσος)</span><b>' +
    (a.timing_median_h ?? '-') + ' ώρες</b></div>' +
    '<div class="kv"><span class="k">Μέσα στον τελικό πραγματικό κάμπο</span><b>' +
    v.containment_final_scar_pct + '%</b></div>' +
    '<div style="margin-top:6px;color:var(--ink2)"><i>Συμπέρασμα: σωστή γεωγραφία ' +
    '(μόλις ' + (a.outside_final_scar_km2 ?? '-') + ' km² εκτός κάμπου), αλλά πολύ ' +
    'γρήγορος ρυθμός - η πραγματική φωτιά δεχόταν κατάσβεση που το μοντέλο δεν ' +
    'προσομοιώνει. Οι μετρικές είναι οι δίκαιες (χωρίς το αρχικό μέτωπο/σπόρο).</i></div>';
})();

/* ---------- init ---------- */
document.getElementById('t0').textContent = D.t0;
if (D.scenarioLabel) document.querySelector('.sub').innerHTML +=
  ' · <b style="color:#7a0a0a">' + D.scenarioLabel + '</b>';
if (!HAS_REAL) {
  document.getElementById('realRow').style.display = 'none';
  document.getElementById('realLegend').style.display = 'none';
}
const b = L.geoJSON({type:'Feature',geometry:D.window}).getBounds();
map.fitBounds(b.pad(0.04));
const mh = location.hash.match(/h=(\d+)/);
setHour(mh ? +mh[1] : 0, true);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
