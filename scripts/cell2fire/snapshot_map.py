"""Static PNG + animated GIF snapshot of a scenario run (chat/mobile views).

The PNG is the final state with a full INFO PANEL (all run parameters + final
numbers) drawn on the map. The GIF animates hour by hour - fire bands growing,
settlement statuses updating - and plays inline on the phone.

ALL official settlements in the window are drawn with their name + population
(grey = safe; blue/red/black = εκκενώνεται / ΑΠΟΚΛΕΙΣΜΕΝΟΣ / στο μέτωπο at the
shown hour). Same red arrival ramp as the dashboard.

Run:  python scripts/cell2fire/snapshot_map.py <run_dir>   (or WFEDS_SCENARIO_DIR)
Writes <run_dir>/map.png + <run_dir>/map.gif
"""

import io
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from _paths import DATA_DIR
from cell2fire_adapter import read_asc_header
from settlements import all_settlements

ROADS = DATA_DIR / "Roads" / "road_edges.gpkg"

STATUS_COL = {"ok": "#2a78d6", "cut_off": "#d03b3b", "impacted": "#0b0b0b"}
STATUS_LBL = {"ok": "εκκενώνεται", "cut_off": "ΑΠΟΚΛΕΙΣΜΕΝΟΣ", "impacted": "στο μέτωπο"}
SAFE_COL = "#8a897f"
LABEL_FX = [pe.withStroke(linewidth=2, foreground="white")]
NO_SPREAD_MSG = "Η φωτιά δεν επεκτάθηκε σημαντικά"


def _fill_holes(geom):
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    parts = [Polygon(p.exterior) for p in polys if p.area >= 2e4]
    if not parts:
        parts = [Polygon(max(polys, key=lambda p: p.area).exterior)]
    return unary_union(parts)


def _red_ramp(t):
    """light salmon (early) -> dark red (late), same as the dashboard."""
    a, b = (254, 224, 210), (103, 0, 13)
    return tuple((a[i] + (b[i] - a[i]) * t) / 255 for i in range(3))


def _fmt_pop(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{int(float(str(v).replace(',', ''))):,}".replace(",", ".")
    except (ValueError, TypeError):
        return "—"


class RunContext:
    """Everything the frames need, loaded once."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        # A "fire that barely spread" writes a perimeters.geojson with ZERO
        # features; geopandas can then infer no properties schema, so the frame
        # comes back with ONLY a "geometry" column (no "period") and zero rows.
        # That is a legitimate, realistic outcome under mild weather - a
        # REPORTABLE RESULT, not a crash (mirrors cell2fire_adapter's
        # "barely-spreading fires no longer crash the adapter" handling). We keep
        # an EMPTY front and still draw the study window / roads / settlements,
        # with a clear "did not spread" note (see snapshot() / _no_spread_note).
        self.perims = gpd.read_file(self.run_dir / "perimeters.geojson").to_crs(2100)
        if "period" in self.perims.columns and len(self.perims):
            self.perims = self.perims.sort_values("period").reset_index(drop=True)
            self.hours = self.perims["period"].tolist()
            self.outers = [_fill_holes(g) for g in self.perims.geometry]
        else:
            self.perims = self.perims.reset_index(drop=True)
            self.hours = []
            self.outers = []

        wtr, wnr, wnc, _ = read_asc_header(str(self.run_dir / "instance" / "Forest.asc"))
        self.wx0, self.wy1 = wtr.c, wtr.f
        self.wx1, self.wy0 = self.wx0 + wnc * wtr.a, self.wy1 + wnr * wtr.e
        self.window = box(self.wx0, self.wy0, self.wx1, self.wy1)

        try:
            self.roads = gpd.read_file(
                ROADS, bbox=(self.wx0, self.wy0, self.wx1, self.wy1)).to_crs(2100)
        except Exception:
            self.roads = None

        # ALL settlements inside the window (official + OSM; name + population)
        st = all_settlements()
        self.settlements = st[st.geometry.within(self.window)].copy()

        self.at_risk, self.routes = None, None
        ts = self.run_dir / "timestep_evacuation.gpkg"
        if ts.exists():
            try:
                self.at_risk = gpd.read_file(ts, layer="at_risk").to_crs(2100)
            except Exception:
                pass
            try:
                self.routes = gpd.read_file(ts, layer="routes").to_crs(2100)
            except Exception:
                pass

        self.result = {}
        rf = self.run_dir / "result.json"
        if rf.exists():
            self.result = json.loads(rf.read_text(encoding="utf-8"))
        else:
            # In the run_scenario chain the snapshot runs BEFORE result.json is
            # written (step 5 vs step 6), so the info panel was silently empty
            # on every run. Assemble the SAME view from the run's own files -
            # identical values, no ordering dependency.
            pf = self.run_dir / "instance" / "scenario_params.json"
            sf = self.run_dir / "timestep_summary.json"
            inputs = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
            per_hour = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else []
            self.result = {"inputs": inputs, "per_hour": per_hour,
                           "final_hour": per_hour[-1] if per_hour else {}}
            iso = self.run_dir / "isochrones.geojson"        # fightability of the final front
            if iso.exists():
                try:
                    feats = json.loads(iso.read_text(encoding="utf-8"))["features"]
                    if feats:
                        last = max(feats, key=lambda f: f["properties"]["period"])
                        self.result["final_front_class4_pct"] = round(
                            float(last["properties"].get("pct4", 0)))
                except Exception:
                    pass
        self.summary = {r["period"]: r for r in self.result.get("per_hour", [])}

    def status_at(self, hour):
        """{NAME_OIK: status} at the given hour."""
        if self.at_risk is None or not len(self.at_risk):
            return {}
        sel = self.at_risk[self.at_risk["period"] == hour]
        return {str(r.get("NAME_OIK", "")).strip(): r.get("status", "cut_off")
                for _, r in sel.iterrows()}


def _draw_frame(ax, ctx, upto, label_fs=6.5):
    """One map frame: fire up to `upto` hours, settlement states at that hour."""
    ax.set_facecolor("#f9f9f7")
    if ctx.roads is not None:
        ctx.roads.plot(ax=ax, color="#c9c8c2", linewidth=0.5, zorder=1)

    idx = [i for i, h in enumerate(ctx.hours) if h <= upto]
    hmax = max(ctx.hours[-1], 1) if ctx.hours else 1     # no perimeters -> no fire band
    for i in reversed(idx):                       # dark (late) under, light on top
        gpd.GeoSeries([ctx.outers[i]], crs=2100).plot(
            ax=ax, color=_red_ramp(ctx.hours[i] / hmax), edgecolor="none", zorder=2)
    if idx:
        gpd.GeoSeries([ctx.outers[idx[-1]].boundary], crs=2100).plot(
            ax=ax, color="#67000d", linewidth=1.1, zorder=3)

    if ctx.routes is not None and len(ctx.routes):
        rsel = ctx.routes[ctx.routes["period"] == upto]
        if len(rsel):
            rsel.plot(ax=ax, color="#2a78d6", linewidth=1.8, zorder=4)

    status = ctx.status_at(upto)
    for _, r in ctx.settlements.iterrows():
        name = str(r.get("name", "")).strip()
        st = status.get(name)
        col = STATUS_COL.get(st, SAFE_COL)
        big = st is not None
        ax.scatter(r.geometry.x, r.geometry.y, s=42 if big else 16, c=col,
                   edgecolors="white", linewidths=1 if big else 0.6, zorder=5)
        ax.annotate(f"{name} ({_fmt_pop(r.get('pop'))})", (r.geometry.x, r.geometry.y),
                    fontsize=label_fs + (0.8 if big else 0),
                    fontweight="bold" if big else "normal",
                    xytext=(4, 3), textcoords="offset points",
                    color="#0b0b0b" if big else "#52514e",
                    path_effects=LABEL_FX, zorder=6)

    gpd.GeoSeries([ctx.window.boundary], crs=2100).plot(
        ax=ax, color="#52514e", linewidth=1, linestyle="--", zorder=3)
    ax.set_xlim(ctx.wx0, ctx.wx1)
    ax.set_ylim(ctx.wy0, ctx.wy1)
    ax.set_xticks([])
    ax.set_yticks([])


def _legend(ax):
    handles = [Patch(color=_red_ramp(0.1), label="κάηκε νωρίς"),
               Patch(color=_red_ramp(0.9), label="κάηκε αργά"),
               Line2D([], [], color="#2a78d6", lw=2, label="διαδρομές εκκένωσης"),
               Line2D([], [], marker="o", color="none", markerfacecolor=SAFE_COL,
                      markersize=6, label="οικισμός (ασφαλής)")]
    for st, c in STATUS_COL.items():
        handles.append(Line2D([], [], marker="o", color="none",
                              markerfacecolor=c, markersize=8, label=STATUS_LBL[st]))
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.92)
    leg.set_zorder(8)      # above the settlement labels (zorder 6), like the info panel - else they overwrite it


def _no_spread_note(ax):
    """Centre overlay for a run whose fire produced ZERO perimeters (barely
    spread): the map still shows the study window / roads / settlements, and this
    note makes the 'no significant spread' result explicit instead of leaving a
    blank map (or, before the fix, crashing)."""
    ax.text(0.5, 0.5, f"{NO_SPREAD_MSG}\n(καμία καταγεγραμμένη περίμετρος πυρκαγιάς)",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="center", ha="center", color="#67000d",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      alpha=0.92, edgecolor="#67000d"), zorder=9)


def _info_text(ctx):
    """The settings+result panel drawn ON the PNG (mirrors the chat cards)."""
    inp = ctx.result.get("inputs", {})
    fin = ctx.result.get("final_hour", {})
    grid = inp.get("grid", {})
    mode = {"point_ignition": "σημειακή έναυση", "front": "μέτωπο",
            "multi_front": f"{inp.get('n_fronts_detected', '?')} ανεξ. μέτωπα",
            "scenario": f"σενάριο: {inp.get('scenario')}"}.get(inp.get("mode"), "?")
    wkm = inp.get("window_km")
    lines = [
        "ΠΑΡΑΜΕΤΡΟΙ",
        f"τύπος: {mode}" + (f" ({inp.get('front_cells')} κελιά)" if inp.get("front_cells") else ""),
        f"παράθυρο: {f'{wkm:g} km' if wkm else 'VIIRS bbox'}"
        f" · {grid.get('nrows', '?')}×{grid.get('ncols', '?')} κελιά",
        f"ορίζοντας: {inp.get('horizon_h')} ώρες",
        f"έναρξη (UTC): {str(inp.get('start_time_utc', '')).replace('T', ' ')}",
        f"καιρός: ERA5 (άνεμος '{inp.get('wind_dir_source')}')",
        "μοντέλο: ελεύθερο κάψιμο (χειρότερο σενάριο)",
        "",
        "ΑΠΟΤΕΛΕΣΜΑ (τελική ώρα)",
        f"έκταση: {fin.get('fire_km2', '?')} km²",
        f"σε κίνδυνο: {fin.get('at_risk', 0)} οικισμοί (~{fin.get('population', 0)} κάτ.)",
        f"εκκενώνονται {fin.get('routed', 0)} · αποκλεισμένοι {fin.get('cut_off', 0)}"
        f" · στο μέτωπο {fin.get('impacted', 0)}",
        f"κλειστά τμήματα: {fin.get('edges_removed', 0)}",
    ]
    pct4 = ctx.result.get("final_front_class4_pct")
    if pct4 is not None:
        lines.append(f"μέτωπο κλάσης 4: {pct4}%")
    return "\n".join(lines)


def snapshot(run_dir):
    ctx = RunContext(run_dir)
    spread = bool(ctx.hours)                  # False = fire barely spread (0 perims)
    last = ctx.hours[-1] if spread else 0
    aspect = (ctx.wy1 - ctx.wy0) / (ctx.wx1 - ctx.wx0)

    # --- final PNG with the info panel ---------------------------------------
    fig, ax = plt.subplots(figsize=(11, 11 * aspect), dpi=150)
    _draw_frame(ax, ctx, last, label_fs=7)
    _legend(ax)
    ax.text(0.988, 0.988, _info_text(ctx), transform=ax.transAxes, fontsize=7.6,
            va="top", ha="right", family="DejaVu Sans", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      alpha=0.93, edgecolor="#c9c8c2"), zorder=8)
    if spread:
        ax.set_title(f"Πρόβλεψη εξάπλωσης +{last} ώρες", fontsize=12)
    else:
        ax.set_title(f"Πρόβλεψη εξάπλωσης: {NO_SPREAD_MSG}", fontsize=12)
        _no_spread_note(ax)
    plt.tight_layout()
    png = ctx.run_dir / "map.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved map snapshot -> {png}")

    # --- animated MP4, one frame per hour -------------------------------------
    # MP4 (H.264) instead of GIF: Telegram transcodes GIF animations DOWN hard,
    # but passes a well-formed H.264 through nearly untouched - and it autoplays
    # inline the same way. Frames must have EVEN dimensions for yuv420p.
    import imageio.v2 as imageio
    import numpy as np

    frames = []
    # No perimeters -> render ONE synthetic frame (the study window + settlements
    # with the "did not spread" note) so map.mp4 is still a valid, non-empty clip
    # and the `frames[-1]` hold below is safe.
    hours_to_render = ctx.hours if spread else [last]
    for h in hours_to_render:
        f2, ax2 = plt.subplots(figsize=(10, 10 * aspect), dpi=130)
        _draw_frame(ax2, ctx, h, label_fs=7)
        if spread:
            s = ctx.summary.get(h, {})
            ax2.set_title(f"+{h} ώρες · {s.get('fire_km2', '?')} km² · "
                          f"σε κίνδυνο {s.get('at_risk', 0)} "
                          f"(⛔ {s.get('cut_off', 0)})", fontsize=13)
        else:
            ax2.set_title(NO_SPREAD_MSG, fontsize=13)
            _no_spread_note(ax2)
        plt.tight_layout()
        buf = io.BytesIO()
        f2.savefig(buf, format="png", bbox_inches="tight")
        plt.close(f2)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"))
        frames.append(arr)
    # bbox_inches="tight" crops each frame to its own content, so frame sizes
    # can differ by a few pixels hour-to-hour (title/labels change) - but the
    # MP4 writer requires ALL frames identical. Pad every frame to the common
    # max on a white canvas, then trim to even dims (yuv420p requirement).
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)

    def _pad(f):
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        canvas[:f.shape[0], :f.shape[1]] = f
        return canvas[:max_h // 2 * 2, :max_w // 2 * 2]

    frames = [_pad(f) for f in frames]
    mp4 = ctx.run_dir / "map.mp4"
    with imageio.get_writer(mp4, fps=1.25, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)
        for _ in range(3):                              # hold the last frame
            w.append_data(frames[-1])
    print(f"Saved animation -> {mp4} ({len(frames)} frames, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]} px)")
    return png, mp4


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WFEDS_SCENARIO_DIR")
    if not rd:
        raise SystemExit("usage: snapshot_map.py <run_dir> (or WFEDS_SCENARIO_DIR)")
    snapshot(rd)
