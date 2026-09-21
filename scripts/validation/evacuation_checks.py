"""Axis 2: evacuation verification and plausibility. Protocol declared
2026-08-27, revised the same day (see Validation.md Section 2a, Decision log
2026-08-27).

Runs against an EXISTING scenario run's outputs (perimeters.geojson,
timestep_evacuation.gpkg, timestep_summary.json) - no new simulation, no LLM
call.

THE DESIGN RULE, and why "code checking code" is not circular here
------------------------------------------------------------------
A check is worth something ONLY if it reaches the answer by a DIFFERENT PATH
than the production code. Recomputing the same quantity the same way repeats
any mistake and proves nothing. (A first draft of this file contained exactly
that mistake: it recounted fire-intersecting edges with the same predicate
friction_graph() uses, which agrees by construction.)

Every check below therefore targets a property that is an EMERGENT CONSEQUENCE
of the pipeline, never something the pipeline computes and stores directly:

  C1 routes vs fire        production removes edges crossing the fire BEFORE
                           routing; it never inspects the FINAL assembled route.
                           Between the two lie node snapping, path assembly,
                           linemerge and the GeoPackage write.
  C2 monotone blockage     each hour is solved INDEPENDENTLY (fire_timesteps.py
                           loops); nothing enforces a relationship between
                           hours. Monotonicity must emerge from growing
                           perimeters.
  C3 label vs artefacts    status, refuge, route_km and the route geometry are
                           written to DIFFERENT layers through a .map()
                           indirection; cross-layer agreement is never checked.
  C4 stated km vs drawn km the stated length is a SUM OF EDGE `length`
                           ATTRIBUTES (route_to_line returns
                           edges["length"].sum()); the geometry is a LINEMERGE
                           OF EDGE SHAPES. Two independent sources, never
                           compared - this is the "the number must match the
                           picture" check.
  C5 route continuity      a route assembled from consecutive edges must merge
                           into ONE LineString; a MultiLineString means the
                           path fragmented somewhere.

Separately, and labelled as a DIFFERENT KIND of evidence:

  R1 reproducibility       recompute the whole evacuation per hour with the
                           CURRENT code and compare to the stored labels. This
                           is NOT an independent correctness check (it is the
                           same code path); it detects DRIFT between a
                           published deliverable and the code that is supposed
                           to produce it. Slow (~5-10 min) - opt in with
                           --reproduce.

And Chalkias's three scenario-delta METRICS (2026-08-04 email), which are
descriptive outputs, not pass/fail checks:

  B5 edges removed per hour
  B6 route-length delta per settlement per hour. NOTE: a NEGATIVE delta is
     LEGITIMATE, not a violation. Routing minimises friction-weighted COST,
     not length, so the least-cost PATH can change even to the SAME refuge:
     as the fire grows, proximity penalties shift across the network, a
     detour that was worth taking (longer but farther from the fire) can
     become expensive and the router falls back to a shorter direct path.
     Origin snapping can also move to a different node once edges are
     removed. (A refuge change is one cause among several - on the golden run
     all three negative deltas kept the SAME refuge.) The refuge at both ends
     and a refuge_changed flag are reported so each case can be explained.
  B7 settlements losing access (routed -> cut_off/impacted) per hour.

Outputs (in <run_dir>): evacuation_metrics.json, evacuation_hourly.csv,
evacuation_deltas.csv, evacuation_report.md. Published to
DATA_DIR/Exports/<run_id>/validation/2_evacuation/.

Run:
    python scripts/validation/evacuation_checks.py [run_dir] [--reproduce]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "cell2fire"))
from _paths import DATA_DIR                                    # noqa: E402

ANALYSIS_CRS = "EPSG:2100"
NAME_COL = "NAME_OIK"
RUNS_DIR = DATA_DIR / "Fire" / "cell2fire" / "runs"

# --- Declared tolerances (pre-declared, not tuned to the outcome) -------------
ROUTE_FIRE_OVERLAP_TOL_M = 5.0   # a route may TOUCH the front at a snapped node;
                                  # more than this much shared LENGTH means it
                                  # runs through the flames
LENGTH_MISMATCH_REL_TOL = 0.01   # stated km vs drawn km: 1% relative
LENGTH_MISMATCH_ABS_TOL_M = 25.0  # ...or 25 m absolute, whichever is larger


# --------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------
def resolve_run_dir(arg=None):
    """The run to check: explicit CLI path, else the newest run under RUNS_DIR
    that has all three per-timestep evacuation outputs."""
    def _complete(d):
        return ((d / "perimeters.geojson").exists()
                and (d / "timestep_evacuation.gpkg").exists()
                and (d / "timestep_summary.json").exists())
    if arg:
        rd = Path(arg)
        if not _complete(rd):
            sys.exit(f"ERROR: {rd} is missing perimeters.geojson / "
                     "timestep_evacuation.gpkg / timestep_summary.json")
        return rd
    if RUNS_DIR.exists():
        candidates = sorted((d for d in RUNS_DIR.iterdir() if _complete(d)),
                            key=lambda d: d.name)
        if candidates:
            return candidates[-1]
    sys.exit("ERROR: no run dir given and no finished run with per-timestep "
             f"evacuation outputs found under {RUNS_DIR}")


def load_run_outputs(run_dir):
    perims = (gpd.read_file(run_dir / "perimeters.geojson").to_crs(ANALYSIS_CRS)
              .sort_values("period").reset_index(drop=True))
    at_risk = gpd.read_file(run_dir / "timestep_evacuation.gpkg",
                            layer="at_risk").to_crs(ANALYSIS_CRS)
    try:
        routes = gpd.read_file(run_dir / "timestep_evacuation.gpkg",
                               layer="routes").to_crs(ANALYSIS_CRS)
    except Exception:              # layer absent when nothing was ever routed
        routes = gpd.GeoDataFrame({"period": [], "origin": [], "refuge": [],
                                   "length_m": []}, geometry=[], crs=ANALYSIS_CRS)
    summary = json.loads((run_dir / "timestep_summary.json").read_text(encoding="utf-8"))
    return perims, at_risk, routes, summary


# --------------------------------------------------------------------------------
# C1-C5: independent checks (stored data only; different path from production)
# --------------------------------------------------------------------------------
def c1_routes_vs_fire(perims, routes):
    """Production removes fire-crossing EDGES before routing and never inspects
    the FINAL route. Here the assembled route geometry is intersected with the
    hour's fire polygon."""
    fire_by_period = {int(r["period"]): r.geometry for _, r in perims.iterrows()}
    violations = []
    for _, r in routes.iterrows():
        fire = fire_by_period.get(int(r["period"]))
        if fire is None or r.geometry is None or r.geometry.is_empty:
            continue
        overlap = r.geometry.intersection(fire)
        overlap_m = 0.0 if overlap.is_empty else float(overlap.length)
        if overlap_m > ROUTE_FIRE_OVERLAP_TOL_M:
            violations.append({"period": int(r["period"]),
                               "origin": r.get("origin"),
                               "overlap_m": round(overlap_m, 1)})
    return violations


def c2_monotone_blockage(summary):
    """Each hour is solved independently; nothing links hours. Blocked-edge
    count must still never decrease, because perimeters only grow."""
    violations = []
    for i in range(1, len(summary)):
        prev, cur = summary[i - 1], summary[i]
        if cur["edges_removed"] < prev["edges_removed"]:
            violations.append({"period_from": prev["period"],
                               "period_to": cur["period"],
                               "edges_removed_from": prev["edges_removed"],
                               "edges_removed_to": cur["edges_removed"]})
    return violations


def c3_label_vs_artefacts(at_risk, routes):
    """status / refuge / route_km live on the at_risk layer, the geometry on the
    routes layer, joined only by (period, name). Check they agree:
    status 'ok' <=> refuge + route_km + exactly one route line."""
    route_keys = {}
    for _, r in routes.iterrows():
        route_keys.setdefault((int(r["period"]), r.get("origin")), 0)
        route_keys[(int(r["period"]), r.get("origin"))] += 1
    violations = []
    for _, s in at_risk.iterrows():
        key = (int(s["period"]), s.get(NAME_COL))
        n_routes = route_keys.get(key, 0)
        status = s.get("status")
        has_km = pd.notna(s.get("route_km"))
        has_ref = pd.notna(s.get("refuge"))
        if status == "ok":
            if not (has_km and has_ref and n_routes == 1):
                violations.append({"period": key[0], "settlement": key[1],
                                   "status": status, "has_route_km": bool(has_km),
                                   "has_refuge": bool(has_ref), "n_route_lines": n_routes,
                                   "problem": "status 'ok' without exactly one complete route"})
        else:
            if has_km or has_ref or n_routes:
                violations.append({"period": key[0], "settlement": key[1],
                                   "status": status, "has_route_km": bool(has_km),
                                   "has_refuge": bool(has_ref), "n_route_lines": n_routes,
                                   "problem": "non-routed status carries route artefacts"})
    return violations


def c4_stated_vs_drawn_length(routes):
    """The stated length is a SUM OF EDGE `length` ATTRIBUTES; the geometry is a
    LINEMERGE OF EDGE SHAPES. Two independent sources - they must agree, or the
    number the user reads does not describe the line the user sees."""
    violations, rows = [], []
    for _, r in routes.iterrows():
        if r.geometry is None or r.geometry.is_empty:
            continue
        stated = float(r["length_m"])
        drawn = float(r.geometry.length)
        diff = abs(stated - drawn)
        tol = max(LENGTH_MISMATCH_ABS_TOL_M, LENGTH_MISMATCH_REL_TOL * stated)
        rows.append({"period": int(r["period"]), "origin": r.get("origin"),
                     "stated_m": round(stated, 1), "drawn_m": round(drawn, 1),
                     "diff_m": round(diff, 1)})
        if diff > tol:
            violations.append({"period": int(r["period"]), "origin": r.get("origin"),
                               "stated_m": round(stated, 1), "drawn_m": round(drawn, 1),
                               "diff_m": round(diff, 1), "tol_m": round(tol, 1)})
    return violations, pd.DataFrame(rows)


def c5_route_continuity(routes):
    """Consecutive edges must merge into ONE LineString. A MultiLineString means
    the assembled path fragmented (a gap in the node sequence)."""
    violations = []
    for _, r in routes.iterrows():
        if r.geometry is None or r.geometry.is_empty:
            violations.append({"period": int(r["period"]), "origin": r.get("origin"),
                               "geom_type": "empty/None", "n_parts": 0})
        elif r.geometry.geom_type != "LineString":
            n = len(r.geometry.geoms) if hasattr(r.geometry, "geoms") else 1
            violations.append({"period": int(r["period"]), "origin": r.get("origin"),
                               "geom_type": r.geometry.geom_type, "n_parts": n})
    return violations


# --------------------------------------------------------------------------------
# B5-B7: scenario-delta metrics (descriptive, NOT pass/fail)
# --------------------------------------------------------------------------------
def scenario_delta_metrics(summary, at_risk):
    edges_removed_series = [{"period": r["period"], "edges_removed": r["edges_removed"]}
                            for r in summary]

    routed = at_risk[at_risk["status"] == "ok"]
    km = routed.pivot_table(index=NAME_COL, columns="period", values="route_km",
                            aggfunc="first")
    ref = routed.pivot_table(index=NAME_COL, columns="period", values="refuge",
                             aggfunc="first")
    status_wide = at_risk.pivot_table(index=NAME_COL, columns="period",
                                      values="status", aggfunc="first")

    periods = sorted(km.columns) if len(km.columns) else []
    delta_rows, losing_rows = [], []
    for i in range(1, len(periods)):
        p0, p1 = periods[i - 1], periods[i]
        for name in km.index:
            k0, k1 = km.at[name, p0], km.at[name, p1]
            if pd.notna(k0) and pd.notna(k1):
                r0 = ref.at[name, p0] if name in ref.index else None
                r1 = ref.at[name, p1] if name in ref.index else None
                delta_rows.append({
                    "period_from": p0, "period_to": p1, "settlement": name,
                    "route_km_from": round(float(k0), 2),
                    "route_km_to": round(float(k1), 2),
                    "delta_km": round(float(k1) - float(k0), 2),
                    "refuge_from": r0, "refuge_to": r1,
                    "refuge_changed": bool(r0 != r1),
                })
    route_delta = pd.DataFrame(delta_rows)

    s_periods = sorted(status_wide.columns) if len(status_wide.columns) else []
    for i in range(1, len(s_periods)):
        p0, p1 = s_periods[i - 1], s_periods[i]
        for name in status_wide.index:
            s0, s1 = status_wide.at[name, p0], status_wide.at[name, p1]
            if s0 == "ok" and s1 in ("cut_off", "impacted"):
                losing_rows.append({"period_from": p0, "period_to": p1,
                                    "settlement": name, "to_status": s1})
    return edges_removed_series, route_delta, pd.DataFrame(losing_rows)


# --------------------------------------------------------------------------------
# R1: reproducibility (slow, opt-in; SAME code path - not an independent check)
# --------------------------------------------------------------------------------
def r1_reproduce(perims, at_risk, summary):
    """Recompute the whole evacuation per hour with the CURRENT pipeline code and
    compare to the stored labels. Detects drift between a published deliverable
    and today's code/data - NOT independent correctness (same code path)."""
    from evacuation import exposure_for_fire
    from evacuate_routes import route_to_refuges
    from route_shortest_path import load_graph
    from route_with_fire import buffer_width_grid
    from settlements import load_atrisk

    graph = load_graph()
    atrisk_layer = load_atrisk()
    width, grid = buffer_width_grid()
    stored_by_period = {int(p): dict(zip(g[NAME_COL], g["status"]))
                        for p, g in at_risk.groupby("period")}
    stored_edges = {r["period"]: r["edges_removed"] for r in summary}

    status_violations, edge_violations = [], []
    for _, prow in perims.iterrows():
        p = int(prow["period"])
        scen = exposure_for_fire(prow.geometry, atrisk_layer, width, grid)
        res = route_to_refuges(scen, graph=graph)
        recomputed = dict(zip(res.in_zone[NAME_COL], res.in_zone["status"]))
        stored = stored_by_period.get(p, {})
        for name in set(recomputed) | set(stored):
            if recomputed.get(name) != stored.get(name):
                status_violations.append({"period": p, "settlement": name,
                                          "status_stored": stored.get(name),
                                          "status_recomputed": recomputed.get(name)})
        if int(res.n_removed) != int(stored_edges.get(p, -1)):
            edge_violations.append({"period": p, "stored": stored_edges.get(p),
                                    "recomputed": int(res.n_removed)})
        print(f"   period {p:>3}: {len(recomputed)} at risk "
              f"({int(res.n_removed)} edges removed)")
    return status_violations, edge_violations


# --------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------
def md_table(df, max_rows=25):
    if df is None or (hasattr(df, "empty") and df.empty) or (isinstance(df, list) and not df):
        return "(none)"
    df = pd.DataFrame(df) if isinstance(df, list) else df
    show = df.head(max_rows)
    cols = [str(c) for c in show.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n*(+{len(df) - max_rows} more rows - see the CSV)*")
    return "\n".join(lines)


def write_report(run_dir, meta, checks, length_df, edges_series, route_delta,
                 losing_access, repro):
    neg = route_delta[route_delta["delta_km"] < 0] if len(route_delta) else route_delta
    neg_explained = int(neg["refuge_changed"].sum()) if len(neg) else 0
    lines = [
        "# Evacuation verification and plausibility (axis 2)",
        "",
        f"Run: `{meta['run_id']}` - {meta['n_periods']} hourly periods, "
        f"{meta['n_routes']} stored routes, {meta['n_atrisk_rows']} settlement-hour rows. "
        "Protocol declared 2026-08-27 (Validation.md Section 2a). No new simulation, "
        "no LLM call.",
        "",
        "## Why these checks are not circular",
        "",
        "Each check below reaches its answer by a **different path** than the code that "
        "produced the run. A check that recomputes the same quantity the same way "
        "repeats any mistake and proves nothing; those were removed from this protocol. "
        "What is checked is always an *emergent consequence* of the pipeline, never a "
        "value the pipeline computes and stores directly.",
        "",
        "## Independent checks (stored data only)",
        "",
        f"**C1 - routes vs fire** ({len(checks['c1'])} violation(s)). Production removes "
        "fire-crossing edges *before* routing and never inspects the final assembled "
        f"route; here the route geometry is intersected with the hour's fire polygon "
        f"(tolerance {ROUTE_FIRE_OVERLAP_TOL_M} m of shared length, for a route that "
        "merely touches the front at a snapped node).",
        md_table(checks["c1"]),
        "",
        f"**C2 - monotone blockage** ({len(checks['c2'])} violation(s)). Each hour is "
        "solved independently and nothing links hours; the blocked-edge count must "
        "still never decrease, because perimeters only grow.",
        md_table(checks["c2"]),
        "",
        f"**C3 - label vs artefacts** ({len(checks['c3'])} violation(s)). `status`, "
        "`refuge` and `route_km` live on the at_risk layer, the geometry on the routes "
        "layer, joined only by (period, name): status `ok` must carry exactly one "
        "complete route, and any other status must carry none.",
        md_table(checks["c3"]),
        "",
        f"**C4 - stated km vs drawn km** ({len(checks['c4'])} violation(s)). The stated "
        "length is a sum of edge `length` **attributes**; the geometry is a linemerge of "
        "edge **shapes**. Two independent sources, never otherwise compared - the number "
        f"the reader sees must describe the line the reader sees (tolerance: "
        f"{LENGTH_MISMATCH_REL_TOL:.0%} or {LENGTH_MISMATCH_ABS_TOL_M} m, whichever is larger).",
        md_table(checks["c4"]),
        (f"\nObserved stated-vs-drawn difference across {len(length_df)} routes: "
         f"median {length_df['diff_m'].median():.1f} m, max {length_df['diff_m'].max():.1f} m."
         if len(length_df) else ""),
        "",
        f"**C5 - route continuity** ({len(checks['c5'])} violation(s)). A route assembled "
        "from consecutive edges must merge into one LineString; a MultiLineString means "
        "the path fragmented.",
        md_table(checks["c5"]),
        "",
        "## Scenario-delta metrics (named by the supervisor, 2026-08-04 email)",
        "",
        "These are **descriptive metrics, not pass/fail checks**.",
        "",
        f"**B5 - edges removed per hour:** {len(edges_series)} hourly values, "
        f"{edges_series[0]['edges_removed'] if edges_series else 0} at +0 h rising to "
        f"{edges_series[-1]['edges_removed'] if edges_series else 0} at the final hour "
        "(full series in `evacuation_hourly.csv`).",
        "",
        f"**B6 - route-length delta** ({len(route_delta)} settlement-hour pairs routed at "
        f"both ends; {len(neg)} negative, of which {neg_explained} involve a change of "
        "refuge). A negative delta is **legitimate, not a violation**: routing minimises "
        "friction-weighted *cost*, not length, so the least-cost path can change **even to "
        "the same refuge** - as the fire grows, proximity penalties shift across the "
        "network, a detour that was worth taking (longer but farther from the fire) can "
        "become expensive and the router falls back to a shorter direct path; origin "
        "snapping can also move to a different node once edges are removed. Judge these by "
        "MAGNITUDE: sub-km values are path detail, not a modelling anomaly. Full table in "
        "`evacuation_deltas.csv`.",
        md_table(neg[["period_from", "period_to", "settlement", "route_km_from",
                      "route_km_to", "delta_km", "refuge_from", "refuge_to"]]
                 if len(neg) else neg),
        "",
        f"**B7 - settlements losing access** ({len(losing_access)} routed -> "
        "cut_off/impacted transitions across the run) - the coupling result the "
        "supervisor asked to present.",
        md_table(losing_access),
        "",
        "## Reproducibility (different kind of evidence)",
        "",
        (f"**R1** - the whole evacuation was recomputed per hour with the current code: "
         f"{len(repro['status'])} settlement-hour status mismatch(es), "
         f"{len(repro['edges'])} hourly edge-count mismatch(es). This is **not** an "
         "independent correctness check - it is the same code path - it detects drift "
         "between the published deliverable and the code that should produce it."
         if repro else
         "**R1 was not run** (opt in with `--reproduce`, ~5-10 min). It recomputes the "
         "whole evacuation per hour with the current code to detect drift between the "
         "published deliverable and today's code. It is **not** an independent "
         "correctness check - it is the same code path."),
        md_table(repro["status"]) if repro else "",
        md_table(repro["edges"]) if repro else "",
        "",
        "## Limitations",
        "* No independent ground truth exists for this axis (no real 2021 road-closure "
        "record), so under this chapter's vocabulary these are **verification and "
        "plausibility** checks, never validation.",
        "* The checks establish internal consistency and emergent-property correctness; "
        "they cannot establish that the *modelling rules themselves* (300 m exposure "
        "margin, friction band, remove-vs-penalise split) are operationally right - that "
        "is what the domain-expert questionnaire (bonus) and the axis-4 sensitivity "
        "sweep address.",
        "* C1's tolerance admits a route that touches the front at a snapped node; a "
        "genuine drive-through-the-flames error is orders of magnitude larger.",
    ]
    (run_dir / "evacuation_report.md").write_text(
        "\n".join(l for l in lines if l is not None), encoding="utf-8")


def publish_exports(run_dir):
    dest = DATA_DIR / "Exports" / run_dir.name / "validation" / "2_evacuation"
    dest.mkdir(parents=True, exist_ok=True)
    for n in ("evacuation_metrics.json", "evacuation_hourly.csv",
              "evacuation_deltas.csv", "evacuation_report.md"):
        src = run_dir / n
        if src.exists():
            shutil.copy2(src, dest / n)
    print(f"Published deliverables -> {dest}")


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main(arg=None, reproduce=False):
    run_dir = resolve_run_dir(arg)
    print(f"=== Axis 2 - evacuation verification and plausibility: {run_dir.name} ===")
    perims, at_risk, routes, summary = load_run_outputs(run_dir)
    meta = {"run_id": run_dir.name, "n_periods": len(summary),
            "n_routes": len(routes), "n_atrisk_rows": len(at_risk)}
    print(f"{meta['n_periods']} periods, {meta['n_routes']} routes, "
          f"{meta['n_atrisk_rows']} settlement-hour rows")

    print("-- C1 routes vs fire --")
    c1 = c1_routes_vs_fire(perims, routes)
    print(f"   {len(c1)} violation(s)")
    print("-- C2 monotone blockage --")
    c2 = c2_monotone_blockage(summary)
    print(f"   {len(c2)} violation(s)")
    print("-- C3 label vs artefacts --")
    c3 = c3_label_vs_artefacts(at_risk, routes)
    print(f"   {len(c3)} violation(s)")
    print("-- C4 stated km vs drawn km --")
    c4, length_df = c4_stated_vs_drawn_length(routes)
    print(f"   {len(c4)} violation(s)")
    print("-- C5 route continuity --")
    c5 = c5_route_continuity(routes)
    print(f"   {len(c5)} violation(s)")

    print("-- B5-B7 scenario-delta metrics --")
    edges_series, route_delta, losing_access = scenario_delta_metrics(summary, at_risk)
    n_neg = int((route_delta["delta_km"] < 0).sum()) if len(route_delta) else 0
    print(f"   {len(route_delta)} delta rows ({n_neg} negative - legitimate: the "
          f"least-cost path can change), {len(losing_access)} losing-access transitions")

    repro = None
    if reproduce:
        print("-- R1 reproducibility (slow) --")
        s_v, e_v = r1_reproduce(perims, at_risk, summary)
        repro = {"status": pd.DataFrame(s_v), "edges": pd.DataFrame(e_v)}
        print(f"   {len(s_v)} status mismatch(es), {len(e_v)} edge-count mismatch(es)")

    checks = {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}
    metrics = {
        "meta": meta,
        "independent_checks": {k: {"n_violations": len(v), "violations": v}
                               for k, v in checks.items()},
        "stated_vs_drawn_length_m": {
            "median_diff": round(float(length_df["diff_m"].median()), 2) if len(length_df) else None,
            "max_diff": round(float(length_df["diff_m"].max()), 2) if len(length_df) else None,
        },
        "b5_edges_removed": edges_series,
        "b6_route_length_delta": {
            "n_pairs": len(route_delta), "n_negative": n_neg,
            "n_negative_with_refuge_change": int(
                route_delta[route_delta["delta_km"] < 0]["refuge_changed"].sum()) if n_neg else 0,
        },
        "b7_settlements_losing_access": {"n_transitions": len(losing_access),
                                         "transitions": losing_access.to_dict("records")},
        "r1_reproducibility": ({"ran": True,
                                "n_status_mismatch": len(repro["status"]),
                                "n_edge_mismatch": len(repro["edges"])}
                               if repro else {"ran": False}),
    }
    (run_dir / "evacuation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(summary).to_csv(run_dir / "evacuation_hourly.csv", index=False)
    route_delta.to_csv(run_dir / "evacuation_deltas.csv", index=False)
    write_report(run_dir, meta, checks, length_df, edges_series, route_delta,
                 losing_access, repro)
    for n in ("evacuation_metrics.json", "evacuation_hourly.csv",
              "evacuation_deltas.csv", "evacuation_report.md"):
        print(f"Saved -> {run_dir / n}")
    try:
        publish_exports(run_dir)
    except Exception as e:
        print(f"(publish skipped: {e})")

    total = sum(len(v) for v in checks.values())
    print(f"\n=== DONE - {total} total violation(s) across the 5 independent checks ===")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None, reproduce="--reproduce" in sys.argv)
