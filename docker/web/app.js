/* WFEDS web UI. Vanilla JS + Leaflet, same-origin API (cookies flow freely).
   Colors mirror the existing outputs: sim fire #e34948, routes #2a78d6. */

"use strict";

/* ---------- map ---------- */

const map = L.map("map", { zoomControl: true }).setView([38.90, 23.30], 10);

const baseLight = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  maxZoom: 18,
}).addTo(map);
const baseSat = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles &copy; Esri &middot; Source: Esri, Maxar, Earthstar Geographics",
  maxZoom: 18,
});

const layersCtl = L.control.layers(
  { "Χάρτης": baseLight, "Δορυφόρος": baseSat }, {}, { collapsed: true }).addTo(map);

/* Study-area boundary: on by default. Fuel: registered in the control but
   NOT added to the map (off by default). Both non-interactive so map clicks
   still drop pins through them. */
fetch("/api/layers/study-area", { credentials: "include" })
  .then((r) => r.json())
  .then((gj) => {
    const lyr = L.geoJSON(gj, {
      interactive: false,
      style: { color: "#52514e", weight: 2, dashArray: "6 4", fill: false },
    }).addTo(map);
    layersCtl.addOverlay(lyr, "Όρια περιοχής μελέτης");
  })
  .catch(() => {});

const fuelLegend = document.createElement("div");
fuelLegend.className = "fuel-legend";
/* hexes mirror api.py's _FUEL_RGBA - keep in sync */
fuelLegend.innerHTML =
  "<b>Καύσιμη ύλη (SVM)</b>" +
  '<span><i style="background:#2d6a2f"></i>Κωνοφόρα</span>' +
  '<span><i style="background:#6fae4e"></i>Μεικτό δάσος</span>' +
  '<span><i style="background:#b5d178"></i>Πλατύφυλλα</span>' +
  '<span><i style="background:#8a9a5b"></i>Λοιπές δασικές</span>' +
  '<span><i style="background:#d9d4c8"></i>Μη δάσος</span>';
fuelLegend.hidden = true;
document.getElementById("map").appendChild(fuelLegend);

fetch("/api/layers/fuel", { credentials: "include" })
  .then((r) => r.json())
  .then((meta) => {
    const lyr = L.imageOverlay(meta.url, meta.bounds, { opacity: 0.75, interactive: false });
    layersCtl.addOverlay(lyr, "Καύσιμη ύλη");
    map.on("overlayadd", (e) => { if (e.layer === lyr) fuelLegend.hidden = false; });
    map.on("overlayremove", (e) => { if (e.layer === lyr) fuelLegend.hidden = true; });
  })
  .catch(() => {});

const pinLayer = L.layerGroup().addTo(map);     // committed pins
const polyLayer = L.layerGroup().addTo(map);    // committed polygons
const draftLayer = L.layerGroup().addTo(map);   // in-progress polygon draft
const resultLayer = L.layerGroup().addTo(map);  // per-run overlay (fire, routes, settlements)

const PIN_STYLE = { radius: 7, color: "#ffffff", weight: 1.5, fillColor: "#e34948", fillOpacity: 0.95 };
const POLY_STYLE = { color: "#e34948", weight: 2, fillColor: "#e34948", fillOpacity: 0.18 };
const DRAFT_STYLE = { color: "#e34948", weight: 2, dashArray: "6 4", fill: false };

/* ---------- state ---------- */

let mode = "pin";          // "pin" | "poly"
let draft = [];            // polygon-in-progress vertices [[lat, lon], ...]
let busy = false;          // a chat run is in flight

const $ = (id) => document.getElementById(id);
const geomStatus = $("geomStatus"), chatLog = $("chatLog"), chatText = $("chatText");

/* ---------- helpers ---------- */

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* **bold** -> <b>, newlines -> <br>; input is escaped first */
function renderText(s) {
  return escapeHtml(s).replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");
}

function setGeomStatus(message, warning) {
  geomStatus.classList.remove("idle", "warn");
  let html = escapeHtml(message || "");
  if (warning) {
    geomStatus.classList.add("warn");
    html += ' <span class="w">⚠ ' + escapeHtml(warning) + "</span>";
  }
  geomStatus.innerHTML = html;
}

function resetGeomStatus() {
  geomStatus.classList.add("idle");
  geomStatus.classList.remove("warn");
  geomStatus.textContent = "Κλικ στον χάρτη για να δηλώσεις πού παρατηρήθηκε η φωτιά.";
}

function addMsg(cls, html) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.innerHTML = html;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

async function api(path, body) {
  const opts = { method: body === undefined ? "GET" : "POST", credentials: "include" };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json; charset=utf-8" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch (_) { /* non-json error body */ }
  if (!r.ok) {
    const detail = (data && data.detail) ? data.detail : ("HTTP " + r.status);
    const err = new Error(detail);
    err.status = r.status;
    throw err;
  }
  return data;
}

/* ---------- geometry input ---------- */

function setMode(m) {
  if (busy && m !== mode) return;
  mode = m;
  $("modePin").classList.toggle("active", m === "pin");
  $("modePoly").classList.toggle("active", m === "poly");
  const drafting = m === "poly";
  $("btnFinishPoly").hidden = !drafting;
  $("btnCancelPoly").hidden = !drafting;
  if (!drafting) cancelDraft();
}

function drawDraft() {
  draftLayer.clearLayers();
  draft.forEach((v) => L.circleMarker(v, { ...PIN_STYLE, radius: 4 }).addTo(draftLayer));
  if (draft.length >= 2) L.polyline(draft, DRAFT_STYLE).addTo(draftLayer);
}

function cancelDraft() {
  draft = [];
  draftLayer.clearLayers();
}

async function commitPin(latlng) {
  try {
    const d = await api("/api/pins", { lat: latlng.lat, lon: latlng.lng });
    L.circleMarker(latlng, PIN_STYLE).addTo(pinLayer);
    setGeomStatus(d.message, d.warning);
  } catch (e) {
    setGeomStatus("Σφάλμα: " + e.message);
  }
}

async function commitPolygon() {
  if (draft.length < 3) {
    setGeomStatus("Ένα πολύγωνο θέλει τουλάχιστον 3 κορυφές. Συνέχισε να προσθέτεις σημεία.");
    return;
  }
  try {
    const d = await api("/api/polygon", { vertices: draft });
    L.polygon(draft, POLY_STYLE).addTo(polyLayer);
    cancelDraft();
    setGeomStatus(d.message, d.warning);
  } catch (e) {
    setGeomStatus("Σφάλμα: " + e.message);   // e.g. self-intersecting or oversized
  }
}

map.on("click", (ev) => {
  if (busy) return;
  if (mode === "pin") {
    commitPin(ev.latlng);
  } else {
    draft.push([+ev.latlng.lat.toFixed(5), +ev.latlng.lng.toFixed(5)]);
    drawDraft();
    setGeomStatus("Πολύγωνο σε εξέλιξη: " + draft.length + " κορυφές. " +
      (draft.length >= 3 ? "Πάτησε «Κλείσιμο πολυγώνου» για καταχώρηση." : "Πρόσθεσε τουλάχιστον 3."));
  }
});

$("modePin").addEventListener("click", () => setMode("pin"));
$("modePoly").addEventListener("click", () => setMode("poly"));
$("btnFinishPoly").addEventListener("click", commitPolygon);
$("btnCancelPoly").addEventListener("click", () => { cancelDraft(); resetGeomStatus(); });

$("btnClear").addEventListener("click", async () => {
  if (busy) return;
  try {
    const d = await api("/api/clear", {});
    pinLayer.clearLayers();
    polyLayer.clearLayers();
    cancelDraft();
    clearRunOverlay();
    resetGeomStatus();
    addMsg("agent", renderText(d.message));
  } catch (e) {
    addMsg("error", renderText("Σφάλμα: " + e.message));
  }
});

/* ---------- chat ---------- */

function setBusy(b) {
  busy = b;
  chatText.disabled = b;
  $("btnSend").disabled = b;
}

/* ---------- in-map hourly playback (the dashboard, on the main map) ---------- */

const STATUS_COL = { ok: "#2a78d6", cut_off: "#d03b3b", impacted: "#0b0b0b" };
const STATUS_LBL = { ok: "εκκενώνεται", cut_off: "ΑΠΟΚΛΕΙΣΜΕΝΟΣ", impacted: "στο μέτωπο" };

let overlay = null;       // fetched run data
let overlayHour = 0;
let playTimer = null;

const hourCtl = document.createElement("div");
hourCtl.className = "hour-ctl";
hourCtl.hidden = true;
hourCtl.innerHTML =
  '<button id="ovPlay" title="Αναπαραγωγή">▶</button>' +
  '<input id="ovSlider" type="range" min="0" max="0" value="0" step="1">' +
  '<span id="ovLabel"></span>' +
  '<button id="ovClose" title="Κλείσιμο αποτελεσμάτων">✕</button>';
document.getElementById("map").appendChild(hourCtl);
/* Slider drags must not fall through to the map (pin drops mid-scrub). */
L.DomEvent.disableClickPropagation(hourCtl);

function rampColor(t) {
  /* light salmon (early) -> dark red (late); same ramp as snapshot/dashboard */
  const a = [254, 224, 210], b = [103, 0, 13];
  return "rgb(" + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",") + ")";
}

function renderHour(h) {
  overlayHour = h;
  resultLayer.clearLayers();
  if (!overlay) return;
  const N = Math.max(1, overlay.periods[overlay.periods.length - 1]);

  overlay.perimeters.features
    .filter((f) => f.properties.period <= h)
    .forEach((f) => {
      L.geoJSON(f, { interactive: false, style: {
        color: f.properties.period === h ? "#67000d" : "transparent",
        weight: 1.2,
        fillColor: rampColor(f.properties.period / N),
        fillOpacity: 0.35,
      } }).addTo(resultLayer);
    });

  if (overlay.routes) {
    overlay.routes.features.filter((f) => f.properties.period === h).forEach((f) => {
      L.geoJSON(f, { interactive: false,
        style: { color: "#ffffff", weight: 5, opacity: 0.9 } }).addTo(resultLayer);
      L.geoJSON(f, { style: { color: "#2a78d6", weight: 2.5, opacity: 0.95 } })
        .bindTooltip(f.properties.origin + " → " + f.properties.refuge + " (" +
                     (f.properties.length_m / 1000).toFixed(1) + " km)")
        .addTo(resultLayer);
    });
  }

  if (overlay.at_risk) {
    overlay.at_risk.features.filter((f) => f.properties.period === h).forEach((f) => {
      const [lon, lat] = f.geometry.coordinates;
      L.circleMarker([lat, lon], {
        radius: 6, color: "#ffffff", weight: 1.5,
        fillColor: STATUS_COL[f.properties.status] || "#8a897f", fillOpacity: 0.95,
      }).bindTooltip(f.properties.NAME_OIK + " (" + (f.properties.census2021 ?? "χωρίς απογραφή") +
                     ") · " + (STATUS_LBL[f.properties.status] || f.properties.status))
        .addTo(resultLayer);
    });
  }

  const s = (overlay.summary || [])[h];
  $("ovLabel").textContent = s
    ? "+" + h + "ω · " + s.fire_km2 + " km² · σε κίνδυνο " + s.at_risk +
      " (🚗" + s.routed + " ⛔" + s.cut_off + " 🔥" + s.impacted + ")"
    : "+" + h + "ω";
  $("ovSlider").value = h;
}

function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; $("ovPlay").textContent = "▶"; }
}

async function showRunOverlay(runId) {
  try {
    const d = await api("/api/runs/" + runId + "/overlay");
    overlay = d;
    overlay.periods = d.perimeters.features.map((f) => f.properties.period)
      .sort((a, b) => a - b);
    const maxP = overlay.periods[overlay.periods.length - 1];
    $("ovSlider").max = maxP;
    hourCtl.hidden = false;
    renderHour(maxP);                       // land on the final hour, like the PNG
    const b = L.geoJSON(d.perimeters).getBounds();
    if (b.isValid()) map.fitBounds(b.pad(0.25));
  } catch (e) {
    /* overlay is a nicety - the cards/files already carry the result */
  }
}

function clearRunOverlay() {
  stopPlay();
  overlay = null;
  resultLayer.clearLayers();
  hourCtl.hidden = true;
}

$("ovSlider").addEventListener("input", (ev) => { stopPlay(); renderHour(+ev.target.value); });
$("ovClose").addEventListener("click", clearRunOverlay);
$("ovPlay").addEventListener("click", () => {
  if (playTimer) { stopPlay(); return; }
  $("ovPlay").textContent = "⏸";
  playTimer = setInterval(() => {
    const next = overlayHour + 1;
    if (!overlay || next > +$("ovSlider").max) { stopPlay(); return; }
    renderHour(next);
  }, 800);
});

function renderDone(data) {
  (data.cards || []).forEach((c) => {
    const div = document.createElement("div");
    div.className = "card";
    div.textContent = c;
    chatLog.appendChild(div);
  });

  if (data.reply) {
    /* The reply always ends with an advisory/limits statement, usually after
       a "---" separator; style that tail as a muted disclaimer. */
    const parts = data.reply.split(/\n-{3,}\n/);
    let html = renderText(parts[0]);
    if (parts.length > 1) {
      html += '<div class="disclaimer">' + renderText(parts.slice(1).join("\n")) + "</div>";
    }
    addMsg("agent", html);
  }

  if (data.files && data.files.length) {
    const wrap = document.createElement("div");
    wrap.className = "msg agent files";
    data.files.forEach((url) => {
      if (url.endsWith("map.png")) {
        const img = document.createElement("img");
        img.src = url; img.alt = "Τελικός χάρτης"; img.loading = "lazy";
        wrap.appendChild(img);
      } else if (url.endsWith("map.mp4")) {
        const v = document.createElement("video");
        v.src = url; v.controls = true; v.muted = true; v.playsInline = true;
        wrap.appendChild(v);
      } else if (url.endsWith("fire_timesteps.html")) {
        const a = document.createElement("a");
        a.className = "dash"; a.href = url; a.target = "_blank"; a.rel = "noopener";
        a.textContent = "📊 Πλήρες dashboard σε νέα καρτέλα";
        wrap.appendChild(a);
      }
    });
    chatLog.appendChild(wrap);

    /* The interactive result lives ON the main map now: hourly playback of
       fire, routes and settlement states. run_id comes from the file URLs
       (/files/<run_id>/...) - if several runs happened in one chat turn,
       show the last one. */
    const runIds = [...new Set(data.files.map((u) => u.split("/")[2]))];
    if (runIds.length) {
      const rid = runIds[runIds.length - 1];
      showRunOverlay(rid);

      /* GIS downloads: GeoJSON, EPSG:2100 (first click generates them). */
      const gis = document.createElement("div");
      gis.className = "msg agent gis";
      gis.innerHTML = "<b>Λήψεις GIS (GeoJSON · ΕΓΣΑ87/2100)</b>";
      [["isochrones_polygons", "Ζώνες ανά ώρα (πολύγωνα)"],
       ["isochrones_lines", "Μέτωπα ανά ώρα (γραμμές)"],
       ["settlements_evacuation", "Οικισμοί με ώρα εκκένωσης"],
       ["evacuation_routes", "Οδικό δίκτυο με διαδρομές εκκένωσης"],
      ].forEach(([name, label]) => {
        const a = document.createElement("a");
        a.href = "/api/runs/" + rid + "/export/" + name;
        a.textContent = "⬇ " + label;
        gis.appendChild(a);
      });
      chatLog.appendChild(gis);
    }
  }
  chatLog.scrollTop = chatLog.scrollHeight;
}

$("chatForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = chatText.value.trim();
  if (!text || busy) return;

  addMsg("user", renderText(text));
  chatText.value = "";
  setBusy(true);
  const progress = addMsg("progress", '<span class="spinner"></span><span>Αποστολή...</span>');

  let streamId;
  try {
    const d = await api("/api/chat", { text });
    streamId = d.stream_id;
  } catch (e) {
    progress.remove();
    setBusy(false);
    addMsg("error", renderText(
      e.status === 409 ? "Τρέχει ήδη μια συνομιλία. Περίμενε να ολοκληρωθεί." :
      "Σφάλμα: " + e.message));
    return;
  }

  const es = new EventSource("/api/chat/" + streamId + "/stream");

  es.addEventListener("progress", (ev2) => {
    const msg = JSON.parse(ev2.data);
    progress.innerHTML = '<span class="spinner"></span><span>' + renderText(msg) + "</span>";
  });

  es.addEventListener("done", (ev2) => {
    es.close();                       // IMPORTANT: otherwise EventSource auto-reconnects
    progress.remove();
    setBusy(false);
    renderDone(JSON.parse(ev2.data));
  });

  /* One handler for BOTH cases: the server's own `event: error` (has .data)
     AND the browser's native connection-error event (no .data) dispatch with
     the same event type "error" - keep a single listener to avoid double
     handling. There is no stream replay server-side, so any drop is final. */
  es.addEventListener("error", (ev2) => {
    es.close();
    if (!progress.isConnected) return;   // already finalized by "done"
    progress.remove();
    setBusy(false);
    const detail = ev2.data ? JSON.parse(ev2.data)
      : "Η ροή ενημέρωσης διακόπηκε. Το σενάριο ίσως ολοκληρώνεται στο παρασκήνιο.";
    addMsg("error", renderText("Σφάλμα: " + detail));
  });
});

/* Enter sends, Shift+Enter = newline */
chatText.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $("chatForm").requestSubmit();
  }
});

/* ---------- restore session on load ---------- */

(async function restore() {
  try {
    const s = await api("/api/state");
    (s.pins || []).forEach(([lat, lon]) => L.circleMarker([lat, lon], PIN_STYLE).addTo(pinLayer));
    (s.polygons || []).forEach((p) => L.polygon(p.vertices, POLY_STYLE).addTo(polyLayer));
    const n = (s.pins || []).length, k = (s.polygons || []).length;
    if (n || k) {
      setGeomStatus("Επαναφορά συνεδρίας: " + n + " pins, " + k + " πολύγωνα.");
    }
    if (s.busy) {
      setBusy(true);
      addMsg("progress", '<span class="spinner"></span><span>Ένα σενάριο τρέχει ήδη σε αυτή τη ' +
        "συνεδρία. Η σελίδα δεν μπορεί να επανασυνδεθεί στη ροή του, περίμενε λίγα λεπτά " +
        "και πάτησε ανανέωση.</span>");
    }
  } catch (_) { /* first visit, nothing to restore */ }
})();
