"""Phase 2: FastAPI web backend over the UNMODIFIED WfedsAgent.

The web counterpart of scripts/agent/telegram_bot.py - same division of
labour, different channel:

    browser pin-drop  ->  POST /api/pins    (geometry, never guessed by the LLM)
    free text         ->  POST /api/chat    (202 + stream_id)
    live progress     ->  GET  /api/chat/{stream_id}/stream   (SSE)
    map.png / map.mp4 / fire_timesteps.html -> GET /files/{run_id}/{filename}

Design notes (kept short - the full reasoning lives in the Phase-2 plan):
  * One WfedsAgent per browser session (uuid cookie), constructed LAZILY on
    the first chat - pins + geometry disclosure work with no LLM key at all,
    mirroring telegram_bot._agent().
  * agent.chat() is blocking (1-15 min) -> runs on a daemon thread; progress
    messages hop to the event loop via loop.call_soon_threadsafe into a
    per-run asyncio.Queue the SSE endpoint drains. `stream_id` is the SSE
    token ONLY - it is NOT a pipeline run_id (one chat can trigger up to 4
    runs); file URLs are derived from each file's own real run directory.
  * A per-session threading.Lock serializes chats: two concurrent chats on
    one session would interleave WfedsAgent.messages (broken tool pairing
    that LLM providers reject) - the second request gets 409 instead.
  * In-memory state only, lost on restart - same documented tradeoff as the
    Telegram bot's `agents`/`pins` dicts.
  * Accepted prototype limitations (single-user, local): /files/ has no
    auth; SSE has no replay on reconnect; a stream whose client never
    connects leaks its queue entry (TODO: sweep or cap if it ever matters).

Runs as the image's default CMD:  /opt/venv-app/bin/python /app/docker/api.py
"""

import asyncio
import json
import math
import re
import sys
import threading
import uuid
from pathlib import Path

# Same bootstrap idiom as the repo's conftest.py (increasing index keeps the
# priority order); agent.py alone only inserts cell2fire, so scripts/agent
# must be added here, exactly as telegram_bot.py does for itself.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _i, _p in enumerate((_HERE,
                         _ROOT / "scripts" / "cell2fire",
                         _ROOT / "scripts" / "agent")):
    sys.path.insert(_i, str(_p))

import run_scenario                                    # noqa: E402
from engine_local_run import run_engine_docker         # noqa: E402

# The one intervention (same as run_container.py): container-native engine
# instead of the WSL bridge. Must happen BEFORE WfedsAgent is imported/used;
# sys.modules caching guarantees _run_tool's later `from run_scenario import
# run_scenario` resolves to this same, patched module object.
run_scenario.run_engine = run_engine_docker

from agent import WfedsAgent                           # noqa: E402
from run_scenario import RUNS_DIR                      # noqa: E402

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel                          # noqa: E402

app = FastAPI(title="WFEDS API", docs_url=None, redoc_url=None)


# --- pin-geometry helpers -----------------------------------------------------
# VERBATIM copies of telegram_bot._km_between/_n_fires. Importing them from
# telegram_bot is not possible here: that module imports the `telegram`
# package at module level, which is deliberately NOT installed in this image
# (web mode has no Telegram dependency), and telegram_bot.py is original repo
# code this branch must not modify (no extraction refactor allowed). Keep the
# three copies in sync manually if the ONE seeding rule ever changes.
def _km_between(p1, p2):
    """Distance (km) between two (lat, lon) WGS84 points - flat-Earth
    approximation; adequate at this scale."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    return math.hypot(dx, dy) / 1000.0


def _n_fires(points, join_m):
    """How many independent fires the engine will see for these pins: centres
    <= join_m apart merge into one front (transitive union-find), mirroring
    build_cell2fire_instance's buffer-union seeding rule."""
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _km_between(points[i], points[j]) * 1000.0 <= join_m:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(points))})


# --- polygon input ------------------------------------------------------------
# A user-drawn polygon is ONE MORE WAY to state the same thing pins state: "fire
# was observed here". It is converted to sample points and fed through the
# EXACT same seeding rule (each point buffered by half a VIIRS pixel, unioned) -
# no new engine path, no change to build_cell2fire_instance. Spacing 300 m
# (< 375 m) guarantees all of one polygon's points merge transitively into ONE
# front; the residual sub-100 m gaps between disc edges are immaterial (the
# engine ignites all seeded cells at t=0 and free-burn closes them instantly).
POLY_SPACING_M = 300.0
POLY_MAX_POINTS = 1200          # ~100 km2 - far beyond any sane observed front


def _polygon_seed_points(vertices):
    """[(lat, lon), ...] polygon -> seed points on the same flat-earth local
    frame _km_between uses. Returns (points, area_km2); raises ValueError on
    degenerate/oversized input."""
    from shapely.geometry import Point, Polygon  # heavy import, kept local

    lat0 = sum(v[0] for v in vertices) / len(vertices)
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(lat0))
    poly = Polygon([(lon * m_lon, lat * m_lat) for lat, lon in vertices])
    if not poly.is_valid:
        poly = poly.buffer(0)          # standard fix for self-intersections
    if poly.is_empty or poly.area == 0:
        raise ValueError("Μη έγκυρο πολύγωνο (μηδενικό εμβαδόν ή αυτοτεμνόμενο).")
    area_km2 = poly.area / 1e6

    pts_m = []
    # boundary, densified - guarantees the outline is fully claimed
    boundary = poly.exterior if poly.geom_type == "Polygon" else max(
        poly.geoms, key=lambda g: g.area).exterior
    n_seg = max(1, int(boundary.length // POLY_SPACING_M))
    for i in range(n_seg):
        p = boundary.interpolate(i / n_seg, normalized=True)
        pts_m.append((p.x, p.y))
    # interior grid
    minx, miny, maxx, maxy = poly.bounds
    y = miny + POLY_SPACING_M / 2
    while y < maxy:
        x = minx + POLY_SPACING_M / 2
        while x < maxx:
            if poly.contains(Point(x, y)):
                pts_m.append((x, y))
            x += POLY_SPACING_M
        y += POLY_SPACING_M

    if len(pts_m) > POLY_MAX_POINTS:
        raise ValueError(
            f"Πολύ μεγάλο πολύγωνο ({area_km2:.0f} km² → {len(pts_m)} σημεία "
            f"σποράς, όριο {POLY_MAX_POINTS}). Σχεδίασε μικρότερο μέτωπο.")
    return ([(round(y / m_lat, 5), round(x / m_lon, 5)) for x, y in pts_m],
            area_km2)


class _PinsView(list):
    """A real list of (lat, lon) seed points that PRINTS as a short summary.

    agent.chat() interpolates the pins object straight into the LLM prompt
    (f"[ΤΡΕΧΟΝΤΑ PINS ΧΡΗΣΤΗ: {pins}]") AND hands the same object to the tool.
    With polygon-derived geometry that raw interpolation would dump hundreds
    of coordinates into the prompt for no benefit - the LLM never needs the
    raw numbers (the tool gets them directly). Subclassing list keeps every
    consumer working (truthiness, iteration, len, json serialization for the
    provenance file) while __str__/__format__ give the LLM a compact note.
    Zero changes to agent.py."""

    def __init__(self, items, label):
        super().__init__(items)
        self._label = label

    def __str__(self):
        return self._label

    __repr__ = __str__


def _geometry_for_chat(session):
    """The session's full ignition geometry as the agent expects it: plain
    pins as-is; with polygons, everything concatenated behind a summarizing
    _PinsView. Returns None when there is no geometry at all."""
    pins, polys = session["pins"], session["polygons"]
    if not polys:
        return list(pins) or None
    allpts = list(pins) + [p for poly in polys for p in poly["points"]]
    label = []
    if pins:
        label.append(f"{len(pins)} pins {list(pins)}")
    label.append(f"{len(polys)} πολύγωνο/α χρήστη "
                 f"({sum(len(p['points']) for p in polys)} σημεία σποράς, "
                 f"{sum(p['area_km2'] for p in polys):.1f} km² συνολικά)")
    return _PinsView(allpts, " + ".join(label))

SESSION_COOKIE = "wfeds_session"
sessions: dict[str, dict] = {}          # sid -> {"agent", "pins", "lock"}
streams: dict[str, asyncio.Queue] = {}  # stream_id -> SSE queue (plumbing only)


def _session(request: Request, response: Response) -> dict:
    """Get-or-create the browser session (uuid cookie -> in-memory dict)."""
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid or sid not in sessions:
        sid = uuid.uuid4().hex
        sessions[sid] = {"agent": None, "pins": [], "polygons": [],
                         "lock": threading.Lock()}
        # No `secure=True`: this is a plain-HTTP local prototype - with it,
        # the browser would silently drop the cookie and every request would
        # look like a brand-new session.
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return sessions[sid]


class PinIn(BaseModel):
    lat: float
    lon: float


class PolygonIn(BaseModel):
    vertices: list[tuple[float, float]]   # [(lat, lon), ...], >= 3


class ChatIn(BaseModel):
    text: str


def _all_points(session):
    """Every seed point in the session (pins + polygon-derived), flat list."""
    return (list(session["pins"])
            + [p for poly in session["polygons"] for p in poly["points"]])


@app.post("/api/pins")
async def post_pin(body: PinIn, request: Request, response: Response):
    """Register a fire-observation pin; disclose the geometry AT PIN TIME
    (one front vs N separate fires) - the 2026-07-10 field-test lesson."""
    from build_cell2fire_instance import VIIRS_PIXEL_M, WINDOW_KM_MAX  # lazy, heavy deps

    session = _session(request, response)
    new_pin = (round(body.lat, 5), round(body.lon, 5))
    existing = _all_points(session)
    far_km = None
    if existing:
        d = max(_km_between(new_pin, p) for p in existing)
        if d > WINDOW_KM_MAX:
            far_km = round(d, 1)
    session["pins"].append(new_pin)
    allpts = _all_points(session)
    if len(allpts) == 1:
        geom = "Ένα pin = σημειακή έναυση."
    else:
        k = _n_fires(allpts, VIIRS_PIXEL_M)
        n = len(session["pins"])
        what = (f"{n} pins" if not session["polygons"] else
                f"{n} pins + {len(session['polygons'])} πολύγωνο/α")
        geom = (f"{what} = ένα ενιαίο μέτωπο." if k == 1 else
                f"{what} = {k} ξεχωριστές φωτιές - ενώνονται μόνο σημεία έως "
                f"~{VIIRS_PIXEL_M:.0f} μ. μεταξύ τους· για ενιαίο μέτωπο "
                f"στείλε ενδιάμεσα pins.")
    out = {"count": len(session["pins"]), "pins": session["pins"], "message": geom}
    if far_km is not None:
        out["warning"] = (f"Το pin απέχει {far_km} km από την υπόλοιπη γεωμετρία - "
                          f"πάνω από το μέγιστο παράθυρο ({WINDOW_KM_MAX:.0f} km). "
                          f"Έλεγξε μήπως είναι λάθος σημείο (/api/clear για εκκαθάριση).")
    return out


@app.post("/api/polygon")
async def post_polygon(body: PolygonIn, request: Request, response: Response):
    """Register a user-drawn observed-front polygon. Same seeding logic as
    pins underneath: the polygon becomes sample points 300 m apart, each one
    an observation with the standard half-VIIRS-pixel footprint."""
    from build_cell2fire_instance import VIIRS_PIXEL_M, WINDOW_KM_MAX  # lazy, heavy deps

    if len(body.vertices) < 3:
        raise HTTPException(400, "Ένα πολύγωνο θέλει τουλάχιστον 3 κορυφές.")
    session = _session(request, response)
    try:
        points, area_km2 = _polygon_seed_points(
            [(round(la, 5), round(lo, 5)) for la, lo in body.vertices])
    except ValueError as e:
        raise HTTPException(400, str(e))

    existing = _all_points(session)
    far_km = None
    if existing:
        d = max(_km_between(points[0], p) for p in existing)
        if d > WINDOW_KM_MAX:
            far_km = round(d, 1)
    session["polygons"].append({"vertices": body.vertices,
                                "points": points, "area_km2": round(area_km2, 2)})
    allpts = _all_points(session)
    k = _n_fires(allpts, VIIRS_PIXEL_M)
    n_poly, n_pins = len(session["polygons"]), len(session["pins"])
    what = f"{n_poly} πολύγωνο/α" + (f" + {n_pins} pins" if n_pins else "")
    geom = (f"{what} = ένα ενιαίο μέτωπο." if k == 1 else
            f"{what} = {k} ξεχωριστές φωτιές (ενώνονται μόνο σημεία έως "
            f"~{VIIRS_PIXEL_M:.0f} μ. μεταξύ τους).")
    out = {"polygons": n_poly, "area_km2": round(area_km2, 2),
           "seed_points": len(points),
           "message": f"Πολύγωνο {area_km2:.2f} km² → {len(points)} σημεία σποράς. {geom}"}
    if far_km is not None:
        out["warning"] = (f"Το πολύγωνο απέχει {far_km} km από την υπόλοιπη "
                          f"γεωμετρία - πάνω από το μέγιστο παράθυρο "
                          f"({WINDOW_KM_MAX:.0f} km).")
    return out


@app.post("/api/clear")
async def post_clear(request: Request, response: Response):
    session = _session(request, response)
    session["pins"].clear()
    session["polygons"].clear()
    session["agent"] = None
    return {"message": "Καθαρίστηκαν τα pins, τα πολύγωνα και η συνομιλία."}


@app.get("/api/state")
async def get_state(request: Request, response: Response):
    session = _session(request, response)
    return {"pins": session["pins"],
            "polygons": [{"vertices": p["vertices"], "area_km2": p["area_km2"],
                          "seed_points": len(p["points"])}
                         for p in session["polygons"]],
            "busy": session["lock"].locked()}


@app.post("/api/chat", status_code=202)
async def post_chat(body: ChatIn, request: Request, response: Response):
    session = _session(request, response)

    if session["agent"] is None:
        try:
            # Cheap (env/preset resolution only, no network) -> catch config
            # errors HERE with a clean HTTP status instead of a 202 followed
            # by an SSE error for something entirely predictable.
            session["agent"] = WfedsAgent()
        except (ValueError, RuntimeError) as e:   # unknown preset / missing API key
            raise HTTPException(500, f"Ρύθμιση LLM: {type(e).__name__}: {e}")

    if not session["lock"].acquire(blocking=False):
        raise HTTPException(409, "Τρέχει ήδη μια συνομιλία σε αυτή τη συνεδρία - περίμενε να ολοκληρωθεί.")

    stream_id = uuid.uuid4().hex
    queue: asyncio.Queue = asyncio.Queue()
    streams[stream_id] = queue
    loop = asyncio.get_running_loop()
    agent = session["agent"]
    pins = _geometry_for_chat(session)   # pins + polygon seed points (or None)

    def progress(msg):
        # Worker thread -> event loop: call_soon_threadsafe is the documented
        # cross-thread primitive; put_nowait then runs ON the loop thread.
        loop.call_soon_threadsafe(queue.put_nowait,
                                  {"event": "progress", "data": msg})

    def worker():
        try:
            reply, files, cards = agent.chat(body.text, pins, progress)
            # Each file's URL comes from its OWN run directory (one chat can
            # trigger several pipeline runs - never assume a single run_id).
            urls = [f"/files/{Path(f).parent.name}/{Path(f).name}" for f in files]
            payload = {"event": "done",
                       "data": {"reply": reply, "cards": cards, "files": urls}}
        except Exception as e:   # litellm auth/network/rate limit, bad tool JSON...
            payload = {"event": "error",
                       "data": f"{type(e).__name__}: {e}"}
        finally:
            session["lock"].release()
        loop.call_soon_threadsafe(queue.put_nowait, payload)
        loop.call_soon_threadsafe(queue.put_nowait, None)   # sentinel: end of stream

        # Pre-generate the GIS exports while the user is still reading the
        # reply - the first download click then serves instantly from disk
        # instead of paying the ~1 min generation on demand.
        if payload["event"] == "done":
            for rid in {Path(f).parent.name for f in files}:
                try:
                    with _export_lock(rid):
                        if not (_exports_dir(rid) / "evacuation_routes.geojson").exists():
                            _build_run_exports(RUNS_DIR / rid)
                    _publish_bundle(rid)   # full deliverable set -> DATA_DIR/Exports/<rid>/
                except Exception:
                    pass          # exports are a convenience; the run itself is done

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"stream_id": stream_id}, status_code=202)


@app.get("/api/chat/{stream_id}/stream")
async def stream_chat(stream_id: str):
    queue = streams.get(stream_id)
    if queue is None:
        raise HTTPException(404, "άγνωστο stream_id")

    async def gen():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield (f"event: {item['event']}\n"
                       f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n")
        finally:
            streams.pop(stream_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# What _run_tool actually surfaces to the channel layer - nothing more.
_RUN_ID_RE = re.compile(r"^[0-9A-Za-z_]{1,80}$")
_ALLOWED_FILES = {"map.png", "map.mp4", "fire_timesteps.html"}


@app.get("/files/{run_id}/{filename}")
async def get_file(run_id: str, filename: str):
    """Serve a run's output artifacts. Both path segments are untrusted:
    whitelist each, then a resolve+containment check as defense in depth."""
    if not _RUN_ID_RE.match(run_id) or filename not in _ALLOWED_FILES:
        raise HTTPException(400, "invalid run_id or filename")
    run_dir = (RUNS_DIR / run_id).resolve()
    target = (run_dir / filename).resolve()
    try:
        target.relative_to(run_dir)
    except ValueError:
        raise HTTPException(400, "path traversal rejected")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target)


# --- context map layers (study area boundary + fuel raster) -------------------
# Generated once on first request, cached in memory. Heavy imports stay lazy so
# server startup remains instant.
_layer_cache: dict = {}

# SVM 5-class fuel -> display RGBA. No prior convention exists (the old folium
# fuel overlay was removed 2026-07-03); a muted green flammability ramp,
# non-forest nearly neutral, nodata (7) fully transparent. Keep in sync with
# the legend hexes in web/app.js.
_FUEL_RGBA = {
    0: (45, 106, 47, 165),    # Κωνοφόρα - dark green
    1: (111, 174, 78, 165),   # Μεικτό - mid green
    2: (181, 209, 120, 165),  # Πλατύφυλλα - light green
    3: (138, 154, 91, 165),   # Λοιπές δασικές - olive
    4: (217, 212, 200, 110),  # Μη δάσος - pale neutral, fainter
}
_FUEL_MAX_PX = 1600           # longest output side; 3 m native is absurd overkill here


def _round_coords(obj, nd=5):
    """Round every coordinate in a GeoJSON-like structure in place. 5 decimals
    in 4326 is ~1 m - visually lossless on a web map, cuts payloads hugely
    (raw to_json() emits full float precision)."""
    if isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) for v in obj):
            return [round(v, nd) for v in obj]
        return [_round_coords(v, nd) for v in obj]
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "coordinates":
                obj[k] = _round_coords(v, nd)
            elif isinstance(v, (dict, list)):
                _round_coords(v, nd)
    return obj


@app.get("/api/layers/study-area")
async def layer_study_area():
    """Study-area boundary (two municipalities) as EPSG:4326 GeoJSON."""
    if "study_area" not in _layer_cache:
        import geopandas as gpd
        from _paths import STUDY_AREA
        gdf = gpd.read_file(STUDY_AREA).to_crs(2100)
        gdf["geometry"] = gdf.geometry.simplify(30)     # boundary context, not analysis
        _layer_cache["study_area"] = _round_coords(
            json.loads(gdf.to_crs(4326).to_json()))
    return _layer_cache["study_area"]


def _build_fuel_layer():
    """Reproject + downsample the SVM fuel raster to a web-ready RGBA PNG.
    The source GeoTIFF carries a LOCAL_CS Greek Grid definition rasterio can't
    reproject from - src_crs is overridden to EPSG:2100 (the documented
    'treat as 2100' rule, see Datasets)."""
    import io

    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    from _paths import FUEL

    with rasterio.open(FUEL) as src:
        with WarpedVRT(src, src_crs="EPSG:2100", crs="EPSG:4326",
                       resampling=Resampling.nearest) as vrt:
            scale = max(vrt.width, vrt.height) / _FUEL_MAX_PX
            out_w = max(1, int(vrt.width / scale))
            out_h = max(1, int(vrt.height / scale))
            data = vrt.read(1, out_shape=(out_h, out_w))
            b = vrt.bounds                       # lon/lat: left, bottom, right, top

    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for cls, col in _FUEL_RGBA.items():
        rgba[data == cls] = col                  # nodata (7) stays fully transparent

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    _layer_cache["fuel_png"] = buf.getvalue()
    _layer_cache["fuel_meta"] = {
        "url": "/api/layers/fuel.png",
        "bounds": [[b.bottom, b.left], [b.top, b.right]],   # Leaflet [[S,W],[N,E]]
    }


@app.get("/api/layers/fuel")
async def layer_fuel_meta():
    """Bounds + URL for the fuel image overlay (generated on first call)."""
    if "fuel_meta" not in _layer_cache:
        _build_fuel_layer()
    return _layer_cache["fuel_meta"]


@app.get("/api/layers/fuel.png")
async def layer_fuel_png():
    if "fuel_png" not in _layer_cache:
        _build_fuel_layer()
    from fastapi import Response as _Resp
    return _Resp(content=_layer_cache["fuel_png"], media_type="image/png",
                 headers={"Cache-Control": "public, max-age=86400"})


# --- per-run map overlay (the dashboard's layers, for the MAIN map) ----------
# Returns everything the in-map hourly playback needs, reprojected to 4326 and
# geometry-simplified for the browser: hourly perimeters, evacuation routes,
# at-risk settlement states, and the per-hour stats summary.
@app.get("/api/runs/{run_id}/overlay")
async def run_overlay(run_id: str):
    import geopandas as gpd

    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "invalid run_id")
    cache_key = ("overlay", run_id)
    if cache_key in _layer_cache:
        return _layer_cache[cache_key]

    run_dir = (RUNS_DIR / run_id).resolve()
    try:
        run_dir.relative_to(RUNS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "path traversal rejected")
    perims_path = run_dir / "perimeters.geojson"
    if not perims_path.exists():
        raise HTTPException(404, "run not found or has no perimeters")

    # Hourly perimeters: cumulative extents, one feature per period. Simplify
    # in the metric CRS; 40 m + 5-decimal coordinates keeps a 24 h golden run
    # around a few MB instead of 12+ (visually identical at web zooms).
    perims = gpd.read_file(perims_path).to_crs(2100).sort_values("period")
    perims["geometry"] = perims.geometry.simplify(40)
    perims = perims.to_crs(4326)

    out = {"perimeters": _round_coords(json.loads(perims.to_json())),
           "routes": None, "at_risk": None, "summary": []}

    gpkg = run_dir / "timestep_evacuation.gpkg"
    if gpkg.exists():
        at_risk = gpd.read_file(gpkg, layer="at_risk").to_crs(4326)
        out["at_risk"] = _round_coords(json.loads(
            at_risk[["NAME_OIK", "census2021", "status", "refuge",
                     "route_km", "period", "geometry"]].to_json()))
        try:
            routes = gpd.read_file(gpkg, layer="routes").to_crs(2100)
            routes["geometry"] = routes.geometry.simplify(10)
            routes = routes.to_crs(4326)
            out["routes"] = _round_coords(json.loads(
                routes[["origin", "refuge", "length_m", "period",
                        "geometry"]].to_json()))
        except Exception:
            pass                                  # a run may have zero routes

    summary_path = run_dir / "timestep_summary.json"
    if summary_path.exists():
        out["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    _layer_cache[cache_key] = out
    return out


# --- GIS exports (downloadable GeoJSON, EPSG:2100) ----------------------------
# Four analyst-facing files per run, generated once into run_dir/exports/ and
# then served from disk. All stay in the metric analysis CRS (EPSG:2100) - the
# GeoJSON driver records it in the legacy `crs` member, which QGIS reads.
_EXPORT_NAMES = ("isochrones_polygons", "isochrones_lines",
                 "settlements_evacuation", "evacuation_routes")
_export_locks: dict = {}          # run_id -> Lock (no duplicate generation)
_export_locks_guard = threading.Lock()


def _export_lock(run_id):
    with _export_locks_guard:
        if run_id not in _export_locks:
            _export_locks[run_id] = threading.Lock()
        return _export_locks[run_id]


def _road_edges():
    """The full road network as an edges GeoDataFrame, loaded once per
    process (the ~40 s graphml parse dominated first-download latency)."""
    if "road_edges" not in _layer_cache:
        import osmnx as ox
        from route_shortest_path import load_graph
        edges = ox.graph_to_gdfs(load_graph(), nodes=False, edges=True).reset_index()
        keep = [c for c in ("u", "v", "highway", "length") if c in edges.columns]
        edges = edges[keep + ["geometry"]].copy()
        if "highway" in edges.columns:             # OSM multi-values arrive as lists
            edges["highway"] = edges["highway"].apply(
                lambda v: ",".join(v) if isinstance(v, list) else str(v))
        _layer_cache["road_edges"] = edges
    return _layer_cache["road_edges"].copy()


def _fmt_local(iso_utc):
    """summary time_utc (naive-UTC ISO) -> Greece-local dd/mm/YYYY HH:MM.
    One datetime column only, per user request - hour int + local time; the
    UTC twin columns were redundant (same information three times)."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    dt = datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("Europe/Athens")).strftime("%d/%m/%Y %H:%M")


# Exports live on CONTAINER-LOCAL disk, not in the mounted data folder:
# profiling showed writing the ~13 MB network GeoJSON through the OneDrive
# bind mount takes ~49 s vs 0.3 s locally - the whole "downloads feel dead"
# problem was mount write latency, nothing else. Regeneration after a
# container restart costs only a few seconds, so nothing of value is lost.
import os as _os

_EXPORTS_ROOT = Path(_os.environ.get("WFEDS_EXPORT_DIR", "/tmp/wfeds_exports"))


def _exports_dir(run_id):
    return _EXPORTS_ROOT / run_id


def _build_run_exports(run_dir):
    import geopandas as gpd
    import pandas as pd

    exp = _exports_dir(run_dir.name)
    exp.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "timestep_summary.json").read_text(encoding="utf-8"))
    per_hour = {s["period"]: s for s in summary}

    def hour_cols(p):
        s = per_hour.get(p)
        loc = _fmt_local(s["time_utc"]) if (s and s.get("time_utc")) else None
        return {"hour": int(p), "datetime_local": loc,
                "settlements_at_risk": s["at_risk"] if s else None}

    # 1+2. Isochrones: polygons = the NEW ring reached during each hour
    # (cumulative perimeter minus the previous one; hour 0 = the seed);
    # lines = the outer front per hour (from the adapter's isochrones).
    perims = gpd.read_file(run_dir / "perimeters.geojson").to_crs(2100) \
        .sort_values("period").reset_index(drop=True)
    rows = []
    for i, r in perims.iterrows():
        geom = r.geometry if i == 0 else r.geometry.difference(perims.geometry.iloc[i - 1])
        if geom.is_empty:
            continue
        rows.append({**hour_cols(int(r["period"])), "geometry": geom})
    gpd.GeoDataFrame(rows, crs=2100).to_file(
        exp / "isochrones_polygons.geojson", driver="GeoJSON")

    iso = gpd.read_file(run_dir / "isochrones.geojson").to_crs(2100)
    iso_rows = [{**hour_cols(int(r["period"])), "pct4": r.get("pct4"),
                 "geometry": r.geometry} for _, r in iso.iterrows()]
    gpd.GeoDataFrame(iso_rows, crs=2100).to_file(
        exp / "isochrones_lines.geojson", driver="GeoJSON")

    # 3. ALL settlements, with the hour (and datetime) each one first needs
    # evacuation (first period it appears in the at_risk layer); never-at-risk
    # settlements keep null attributes.
    from settlements import all_settlements   # scripts/cell2fire is on sys.path
    allset = all_settlements()[["name", "pop", "geometry"]].copy()
    gpkg = run_dir / "timestep_evacuation.gpkg"
    first = {}
    if gpkg.exists():
        ar = gpd.read_file(gpkg, layer="at_risk")
        first = ar.groupby("NAME_OIK")["period"].min().to_dict()
    allset["evac_hour"] = allset["name"].map(first).astype("Int64")
    loc_map = {p: hour_cols(p)["datetime_local"] for p in per_hour}
    allset["evac_datetime_local"] = allset["evac_hour"].map(loc_map)
    allset.to_file(exp / "settlements_evacuation.geojson", driver="GeoJSON")

    # 4. The WHOLE road network, evacuation-route segments flagged so they can
    # be selected apart (evac_route=1 + the first hour each segment is used).
    edges = _road_edges()
    edges["evac_route"] = 0
    edges["evac_hour_first"] = pd.array([None] * len(edges), dtype="Int64")
    if gpkg.exists():
        try:
            routes = gpd.read_file(gpkg, layer="routes")
            # routes reuse the exact edge geometries, so a tiny buffer +
            # within-test identifies the network segments they run over
            for p in sorted(routes["period"].unique()):
                buf = routes[routes["period"] == p].union_all().buffer(3.0)
                idx = edges.sindex.query(buf, predicate="intersects")
                hit = [i for i in idx if edges.geometry.iloc[i].within(buf)]
                new = [i for i in hit if edges["evac_route"].iloc[i] == 0]
                edges.loc[edges.index[new], "evac_route"] = 1
                edges.loc[edges.index[new], "evac_hour_first"] = int(p)
        except Exception:
            pass                                   # zero-routes run
    edges.to_file(exp / "evacuation_routes.geojson", driver="GeoJSON")


def _publish_bundle(run_id):
    """Copy the full per-simulation deliverable set into the general export
    folder on the data volume: DATA_DIR/Exports/<run_id>/ = the 4 GIS files
    + map.png + map.mp4 + fire_timesteps.html. OneDrive-mount writes are slow
    (~50s/13MB, measured) - this runs ONLY on the background thread after a
    run completes, never on a request path. Per-file try/except: a missing
    piece (e.g. no mp4 on a barely-spreading run) never blocks the rest."""
    import shutil

    from _paths import DATA_DIR

    dest = DATA_DIR / "Exports" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    src_pairs = (
        [(_exports_dir(run_id) / f"{n}.geojson", f"{n}.geojson") for n in _EXPORT_NAMES]
        + [(RUNS_DIR / run_id / f, f)
           for f in ("map.png", "map.mp4", "fire_timesteps.html")]
    )
    for src, name in src_pairs:
        try:
            if src.exists() and not (dest / name).exists():
                shutil.copy2(src, dest / name)
        except Exception:
            pass


@app.get("/api/runs/{run_id}/export/{name}")
async def run_export(run_id: str, name: str):
    if not _RUN_ID_RE.match(run_id) or name not in _EXPORT_NAMES:
        raise HTTPException(400, "invalid run_id or export name")
    run_dir = (RUNS_DIR / run_id).resolve()
    try:
        run_dir.relative_to(RUNS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "path traversal rejected")
    if not (run_dir / "timestep_summary.json").exists():
        raise HTTPException(404, "run not found")
    target = _exports_dir(run_id) / f"{name}.geojson"
    if not target.exists():
        # Serialize with the post-run pre-generation thread: a click landing
        # mid-pregeneration waits for it instead of generating twice.
        with _export_lock(run_id):
            if not target.exists():
                _build_run_exports(run_dir)
    if not target.exists():
        raise HTTPException(500, "export generation failed")
    return FileResponse(target, media_type="application/geo+json",
                        filename=f"{run_id}_{name}.geojson")


# --- static frontend (Phase 3) -----------------------------------------------
# Mounted LAST so every /api/* and /files/* route above wins the match first.
# Same-origin serving keeps the session cookie story trivial (no CORS at all).
_WEB = _HERE / "web"
if _WEB.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_WEB), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0, not the default 127.0.0.1 - otherwise `-p 8000:8000` publishes
    # a port nothing inside the container is listening on externally.
    uvicorn.run(app, host="0.0.0.0", port=8000)
