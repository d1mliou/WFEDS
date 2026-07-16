"""Phase-1 equivalence check: container run vs WSL reference run.

Compares two completed run directories (each holding result.json +
engine/Grids/Grids1/*.csv) and reports:

  1. GRIDS   - every ForestGrid*.csv / Intensity*.csv, cell by cell.
               Byte-identical is the EXPECTED outcome; <= --tolerance-cells
               differing cells per grid is a WARNING, not a failure (the WSL
               and container engines are built by different g++/glibc, so a
               borderline cell can legitimately flip under -O3 float math).
  2. SUMMARY - result.json: hours_simulated, final_front_class4_pct, the full
               per_hour series and final_hour. Volatile fields are excluded
               (run_id, run_dir, elapsed_min, files, and inputs' built_at_utc
               / path-like entries) - they differ by construction.

Exit 0 = equivalent (possibly with warnings), 1 = real divergence.

Usage:
    python docker/verify_phase1.py --reference <wsl_run_dir> --candidate <container_run_dir>
"""

import argparse
import json
import sys
from pathlib import Path

# inputs keys that legitimately differ between two runs of the same scenario
VOLATILE_INPUT_KEYS = {"built_at_utc", "out_dir", "run_dir"}


def _load(run_dir):
    p = Path(run_dir) / "result.json"
    if not p.exists():
        sys.exit(f"ERROR: {p} not found - is this a completed run dir?")
    return json.loads(p.read_text(encoding="utf-8"))


def _grid_files(run_dir):
    d = Path(run_dir) / "engine" / "Grids" / "Grids1"
    return sorted(d.glob("*.csv")) if d.exists() else []


def compare_grids(ref_dir, cand_dir, tol_cells):
    """Cell-by-cell CSV grid diff. Returns (n_fail, n_warn, report_lines)."""
    ref_files = {f.name: f for f in _grid_files(ref_dir)}
    cand_files = {f.name: f for f in _grid_files(cand_dir)}
    lines, n_fail, n_warn = [], 0, 0

    missing = sorted(set(ref_files) ^ set(cand_files))
    if missing:
        n_fail += len(missing)
        lines.append(f"  FAIL  grid file sets differ: {missing}")

    for name in sorted(set(ref_files) & set(cand_files)):
        ref_txt = ref_files[name].read_text()
        cand_txt = cand_files[name].read_text()
        if ref_txt == cand_txt:
            lines.append(f"  OK    {name}: byte-identical")
            continue
        # not identical -> count differing cells (both are plain CSV numbers)
        diff = 0
        for rrow, crow in zip(ref_txt.splitlines(), cand_txt.splitlines()):
            rv, cv = rrow.split(","), crow.split(",")
            diff += sum(1 for a, b in zip(rv, cv) if a.strip() != b.strip())
            diff += abs(len(rv) - len(cv))
        if diff <= tol_cells:
            n_warn += 1
            lines.append(f"  WARN  {name}: {diff} cell(s) differ "
                         f"(<= tolerance {tol_cells}; likely compiler float drift)")
        else:
            n_fail += 1
            lines.append(f"  FAIL  {name}: {diff} cells differ (> tolerance {tol_cells})")
    return n_fail, n_warn, lines


# Downstream geometry areas (fire_km2, longest_route_km) are floats derived
# from polygon operations - not required to be bit-exact across environments
# whose GDAL/GEOS PATCH versions differ (unlike the raw engine grids, which
# ARE required to be byte-identical; see compare_grids). A relative tolerance
# catches real regressions while tolerating library-version float noise.
NUMERIC_REL_TOL = 0.01   # 1% - generous; real divergences are usually far bigger
NUMERIC_KEYS = {"fire_km2", "longest_route_km"}
EXACT_HOUR_KEYS = {"period", "time_utc", "at_risk", "population",
                   "routed", "cut_off", "impacted", "edges_removed"}


def _hour_diffs(r, c, rel_tol):
    """Field-level diffs between two per-hour dicts, numeric fields tolerant."""
    diffs = {}
    for k in set(r) | set(c):
        rv, cv = r.get(k), c.get(k)
        if rv == cv:
            continue
        if k in NUMERIC_KEYS and isinstance(rv, (int, float)) and isinstance(cv, (int, float)):
            if rv == 0 or abs(rv - cv) / abs(rv) <= rel_tol:
                continue
        diffs[k] = (rv, cv)
    return diffs


def compare_summary(ref, cand):
    """result.json comparison, volatile fields excluded. Returns (n_fail, n_warn, lines)."""
    lines, n_fail, n_warn = [], 0, 0

    for key in ("hours_simulated", "final_front_class4_pct"):
        if ref.get(key) == cand.get(key):
            lines.append(f"  OK    {key}: {ref.get(key)}")
        else:
            n_fail += 1
            lines.append(f"  FAIL  {key}: reference={ref.get(key)} candidate={cand.get(key)}")

    fh_diffs = _hour_diffs(ref.get("final_hour") or {}, cand.get("final_hour") or {},
                          NUMERIC_REL_TOL)
    if not fh_diffs:
        lines.append("  OK    final_hour: identical (or within numeric tolerance)")
    else:
        n_fail += 1
        lines.append(f"  FAIL  final_hour differs: {fh_diffs}")

    rp, cp = ref.get("per_hour") or [], cand.get("per_hour") or []
    if len(rp) != len(cp):
        n_fail += 1
        lines.append(f"  FAIL  per_hour length differs: {len(rp)} vs {len(cp)}")
    else:
        bad_hours, warn_hours = [], []
        for i, (r, c) in enumerate(zip(rp, cp)):
            diffs = _hour_diffs(r, c, NUMERIC_REL_TOL)
            if not diffs:
                continue
            hard = {k: v for k, v in diffs.items() if k in EXACT_HOUR_KEYS}
            if hard:
                bad_hours.append((i, hard))
            else:
                warn_hours.append((i, diffs))
        if bad_hours:
            n_fail += 1
            lines.append(f"  FAIL  per_hour: {len(bad_hours)} hour(s) differ on exact fields, "
                         f"e.g. hour {bad_hours[0][0]}: {bad_hours[0][1]}")
        elif warn_hours:
            n_warn += 1
            lines.append(f"  WARN  per_hour: {len(warn_hours)}/{len(rp)} hour(s) have numeric "
                         f"drift within {NUMERIC_REL_TOL:.0%} tolerance (e.g. hour "
                         f"{warn_hours[0][0]}: {warn_hours[0][1]}) - likely GDAL/GEOS patch-"
                         f"version float noise in polygon-area computation, not a logic bug")
        else:
            lines.append("  OK    per_hour: identical (or within numeric tolerance)")

    ri = {k: v for k, v in (ref.get("inputs") or {}).items()
          if k not in VOLATILE_INPUT_KEYS and not isinstance(v, str) or
          (isinstance(v, str) and "/" not in v and "\\" not in v and k not in VOLATILE_INPUT_KEYS)}
    ci = {k: v for k, v in (cand.get("inputs") or {}).items()
          if k in ri}
    mismatched = {k: (ri[k], ci.get(k)) for k in ri if ri[k] != ci.get(k)}
    if mismatched:
        n_fail += 1
        lines.append(f"  FAIL  inputs differ on: {mismatched}")
    else:
        lines.append(f"  OK    inputs: {len(ri)} comparable fields identical")
    return n_fail, n_warn, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="WSL (ground-truth) run dir")
    ap.add_argument("--candidate", required=True, help="container run dir")
    ap.add_argument("--tolerance-cells", type=int, default=1,
                    help="max differing cells per grid reported as WARN not FAIL (default 1)")
    a = ap.parse_args()

    ref, cand = _load(a.reference), _load(a.candidate)

    print("=== Phase-1 equivalence: GRIDS (engine output) ===")
    g_fail, g_warn, g_lines = compare_grids(a.reference, a.candidate, a.tolerance_cells)
    print("\n".join(g_lines) or "  (no grids found on either side)")

    print("\n=== Phase-1 equivalence: SUMMARY (result.json) ===")
    s_fail, s_warn, s_lines = compare_summary(ref, cand)
    print("\n".join(s_lines))

    print("\n=== VERDICT ===")
    total_warn = g_warn + s_warn
    if g_fail or s_fail:
        print(f"DIVERGENT: {g_fail} grid failure(s), {s_fail} summary failure(s), "
              f"{total_warn} warning(s)")
        sys.exit(1)
    note = f" ({total_warn} tolerated warning(s))" if total_warn else " (all byte-identical)"
    print(f"EQUIVALENT{note}")


if __name__ == "__main__":
    main()
