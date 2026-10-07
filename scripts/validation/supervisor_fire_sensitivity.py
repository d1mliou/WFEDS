"""Read-only sensitivity checks for the supervisor's fire-evaluation comments.

This script reads saved model output. It does not rerun Cell2Fire or overwrite
the canonical validation outputs.
"""

from __future__ import annotations

import argparse
import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from scripts.validation import validate_overlay as vo
from scripts.validation import perimeter_convergence as pc


def temporal_check() -> dict:
    run_dir = vo.RUNS_DIR / "north_evia_2021_validation"
    perims = (
        gpd.read_file(run_dir / "perimeters.geojson")
        .to_crs(2100)
        .sort_values("period")
        .reset_index(drop=True)
    )
    start = vo.sim_start_utc(run_dir)
    seed_time = pd.Timestamp(vo.SEED_TIME)
    seed = perims.iloc[0].geometry
    end = start + pd.Timedelta(hours=int(perims["period"].max()))
    viirs = gpd.read_file(
        vo.RFD / "VIIRS" / "VIRS_dataset_egsa_dhmos.shp"
    ).to_crs(2100)
    viirs["t"] = pd.to_datetime(viirs["acq_at_s"])
    eval_all = viirs[(viirs["t"] > seed_time) & (viirs["t"] <= end)].copy()
    eval_set = eval_all[~eval_all.geometry.within(seed)].copy()
    eval_set["pass_t"] = vo.group_passes(eval_set["t"])

    output: dict = {
        "run": run_dir.name,
        "sim_start_utc": str(start),
        "seed_end_utc": str(seed_time),
        "n_after_seed": len(eval_all),
        "n_scored_outside_seed": len(eval_set),
        "alignments": {},
    }
    for label, anchor in (("reported", start), ("seed_shifted", seed_time)):
        rows = []
        pooled_inside = []
        pooled_dist = []
        for pass_t, group in sorted(eval_set.groupby("pass_t"), key=lambda pair: pair[0]):
            hour = max(0, min(len(perims) - 1, int((pass_t - anchor).total_seconds() // 3600)))
            poly = perims.iloc[hour].geometry
            distances = group.geometry.distance(poly)
            inside = group.geometry.within(poly)
            pooled_inside.extend(inside.tolist())
            pooled_dist.extend(distances.tolist())
            rows.append({
                "pass_utc": str(pass_t),
                "sim_hour": hour,
                "n": len(group),
                "inside": int(inside.sum()),
                "within_375m": int((distances <= 375).sum()),
                "within_500m": int((distances <= 500).sum()),
                "within_1000m": int((distances <= 1000).sum()),
                "median_distance_m": float(np.median(distances)),
                "max_distance_m": float(np.max(distances)),
            })
        d = np.asarray(pooled_dist)
        output["alignments"][label] = {
            "passes": rows,
            "pooled": {
                "n": len(d),
                "inside": int(sum(pooled_inside)),
                "within_375m": int((d <= 375).sum()),
                "within_500m": int((d <= 500).sum()),
                "within_1000m": int((d <= 1000).sum()),
                "median_distance_m": float(np.median(d)),
                "p90_distance_m": float(np.quantile(d, 0.9)),
                "max_distance_m": float(np.max(d)),
            },
        }
    return output


def spatial_check() -> dict:
    """Re-evaluate on Istiaia-Aidipsos sample support without clip-edge lines."""
    run_dir = vo.RUNS_DIR / "north_evia_2021_validation"
    perims = (
        gpd.read_file(run_dir / "perimeters.geojson")
        .to_crs(2100)
        .sort_values("period")
        .reset_index(drop=True)
    )
    municipality = (
        gpd.read_file(vo.DATA_DIR / "Boundaries" / "Study_Area.geojson")
        .to_crs(2100)
        .query("id == 143")
        .geometry.iloc[0]
    )
    scar_raw = gpd.read_file(
        vo.RFD / "Burned_Area" / "Burned_area_20210829.shp"
    ).to_crs(2100).geometry.iloc[0]
    scar = pc.fill_holes(unary_union([scar_raw]))
    raw_parts = list(scar_raw.geoms)

    samples = gpd.read_file(
        run_dir / "perimeter_gis" / "perimeter_samples_tstar.geojson"
    ).to_crs(2100)
    reference = samples[(samples["direction"] == "real_to_sim") & ~samples["excluded"]]
    # A 30 m border guard removes ambiguous points close to the municipal edge.
    reference_common = reference[
        reference.geometry.within(municipality)
        & (reference.geometry.distance(municipality.boundary) > 30)
    ]
    sim18 = samples[(samples["direction"] == "sim_to_real") & ~samples["excluded"]]
    sim18_common = sim18[
        sim18.geometry.within(municipality)
        & (sim18.geometry.distance(municipality.boundary) > 30)
    ]
    parts_ge_1ha = [part for part in raw_parts if part.area >= 10_000]
    scar_ge_1ha = pc.fill_holes(unary_union(parts_ge_1ha))

    def saved_stats(frame: gpd.GeoDataFrame) -> dict:
        d = frame["dist_m"].dropna().to_numpy(dtype=float)
        signed = frame["signed_m"].dropna().to_numpy(dtype=float)
        return {
            "n": len(d),
            "mean_m": round(float(d.mean()), 1),
            "median_m": round(float(np.median(d)), 1),
            "p95_m": round(float(np.quantile(d, 0.95)), 1),
            "positive_pct": round(100 * float((signed > 0).mean()), 1),
            "mean_signed_m": round(float(signed.mean()), 1),
        }

    out = {
        "municipality": "Istiaia-Aidipsos (Study_Area.geojson id=143)",
        "border_guard_m": 30,
        "target_lines": "original fire boundaries; no artificial municipal clip edges",
        "scar_raw_total_km2": round(scar_raw.area / 1e6, 3),
        "scar_raw_inside_municipality_km2": round(scar_raw.intersection(municipality).area / 1e6, 3),
        "scar_raw_parts": len(raw_parts),
        "scar_parts_lt_1ha": sum(part.area < 10_000 for part in raw_parts),
        "scar_lt_1ha_area_km2": round(sum(part.area for part in raw_parts if part.area < 10_000) / 1e6, 3),
        "scar_lt_1ha_exterior_km": round(sum(part.exterior.length for part in raw_parts if part.area < 10_000) / 1000, 1),
        "scar_all_exterior_km": round(sum(part.exterior.length for part in raw_parts) / 1000, 1),
        "saved_hour18": {
            "full_sim_to_real": saved_stats(sim18),
            "common_sim_to_real": saved_stats(sim18_common),
            "full_real_to_sim": saved_stats(reference),
            "common_real_to_sim": saved_stats(reference_common),
        },
        "hourly": [],
    }

    # Published area crossing uses raw, unfilled geometries; repeat with the
    # municipality on both sides of the comparison.
    target_area = scar_raw.intersection(municipality).area
    first_crossing = None
    for _, row in perims.iloc[1:].iterrows():
        hour = int(row["period"])
        raw = row.geometry
        common_area = raw.intersection(municipality).area
        if first_crossing is None and common_area >= target_area:
            first_crossing = hour
        out["hourly"].append({
            "hour": hour,
            "full_area_km2": round(raw.area / 1e6, 3),
            "common_area_km2": round(common_area / 1e6, 3),
        })
    out["first_common_area_crossing_hour"] = first_crossing

    # At h18, check whether thousands of sub-hectare scar components are
    # exerting disproportionate influence on the boundary statistic.
    sim18_poly = pc.fill_holes(perims.loc[perims["period"] == 18].geometry.iloc[0])
    sim18_target = sim18_poly.boundary
    small_filtered_reference = pc.sample_boundary(scar_ge_1ha)
    small_filtered_reference = small_filtered_reference[
        small_filtered_reference.geometry.within(municipality)
        & (small_filtered_reference.geometry.distance(municipality.boundary) > 30)
    ]
    d_sim_to_filtered = pc.directed_distances(sim18_common.geometry, scar_ge_1ha.boundary)
    d_filtered_to_sim = pc.directed_distances(small_filtered_reference.geometry, sim18_target)
    out["hour18_without_lt_1ha_scar_parts"] = {
        "sim_to_real": {"n": len(d_sim_to_filtered), **pc.direction_stats(d_sim_to_filtered)},
        "real_to_sim": {"n": len(d_filtered_to_sim), **pc.direction_stats(d_filtered_to_sim)},
        "d_t_m": pc.hour_indicators(
            pc.direction_stats(d_sim_to_filtered), pc.direction_stats(d_filtered_to_sim)
        )["d_t_m"],
    }
    return out


def hourly_common_boundary_check() -> dict:
    """Recalculate D_t for every hour using only municipality-inside origins."""
    run_dir = vo.RUNS_DIR / "north_evia_2021_validation"
    perims = (
        gpd.read_file(run_dir / "perimeters.geojson")
        .to_crs(2100)
        .sort_values("period")
        .reset_index(drop=True)
    )
    municipality = (
        gpd.read_file(vo.DATA_DIR / "Boundaries" / "Study_Area.geojson")
        .to_crs(2100)
        .query("id == 143")
        .geometry.iloc[0]
    )
    shapely.prepare(municipality)
    # This saved line is the canonical dissolved, hole-filled scar boundary.
    scar_boundary = gpd.read_file(
        run_dir / "perimeter_gis" / "perimeter_scar_boundary.geojson"
    ).to_crs(2100).geometry.iloc[0]
    scar_tree = pc._segment_tree(scar_boundary)

    sample_file = run_dir / "perimeter_gis" / "perimeter_samples_tstar.geojson"
    saved = gpd.read_file(sample_file).to_crs(2100)
    reference = saved[(saved["direction"] == "real_to_sim") & ~saved["excluded"]]
    reference_points = np.asarray(reference.geometry.values)
    reference_points = reference_points[shapely.contains(municipality, reference_points)]

    canonical = json.loads((run_dir / "perimeter_metrics.json").read_text(encoding="utf-8"))
    window = box(*canonical["window"]["bounds"])

    def tree_distances(points: np.ndarray, tree: shapely.STRtree) -> np.ndarray:
        indices, distances = tree.query_nearest(points, return_distance=True, all_matches=False)
        ordered = np.empty(len(points), dtype=float)
        ordered[indices[0]] = distances
        return ordered

    hourly = []
    for _, row in perims.iloc[1:].iterrows():
        h = int(row["period"])
        sim = pc.fill_holes(row.geometry)
        sim_sample = pc.sample_boundary(sim)
        sim_points = np.asarray(sim_sample.geometry.values)
        sim_points = sim_points[
            shapely.contains(municipality, sim_points)
            & (shapely.distance(sim_points, window.exterior) > 30)
        ]
        sim_target = pc.strip_frame_from_line(sim.boundary, window)
        sim_tree = pc._segment_tree(sim_target)
        d_s2r = tree_distances(sim_points, scar_tree)
        d_r2s = tree_distances(reference_points, sim_tree)
        s2r, r2s = pc.direction_stats(d_s2r), pc.direction_stats(d_r2s)
        indicator = pc.hour_indicators(s2r, r2s)
        record = {
            "hour": h,
            "n_sim_common": len(sim_points),
            "n_scar_common": len(reference_points),
            "mean_sim_to_scar_m": s2r["mean_m"],
            "mean_scar_to_sim_m": r2s["mean_m"],
            "d_t_m": indicator["d_t_m"],
            "original_d_t_m": canonical["hourly"][h - 1]["d_t_m"],
        }
        hourly.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
    return {
        "common_support": "origin samples inside Istiaia-Aidipsos; authentic full fire lines as targets",
        "hourly": hourly,
        "t_star_common": pc.find_t_star(pd.DataFrame(hourly)),
    }


def fragment_check(thresholds_ha: tuple[float, ...] = (0, 0.1, 1, 10)) -> dict:
    """At h18, vary only the minimum reference-scar component area."""
    run_dir = vo.RUNS_DIR / "north_evia_2021_validation"
    municipality = (
        gpd.read_file(vo.DATA_DIR / "Boundaries" / "Study_Area.geojson")
        .to_crs(2100)
        .query("id == 143")
        .geometry.iloc[0]
    )
    shapely.prepare(municipality)
    scar_raw = gpd.read_file(
        vo.RFD / "Burned_Area" / "Burned_area_20210829.shp"
    ).to_crs(2100).geometry.iloc[0]
    parts = list(scar_raw.geoms)
    perims = gpd.read_file(run_dir / "perimeters.geojson").to_crs(2100)
    sim = pc.fill_holes(perims.loc[perims["period"] == 18].geometry.iloc[0])
    saved = gpd.read_file(
        run_dir / "perimeter_gis" / "perimeter_samples_tstar.geojson"
    ).to_crs(2100)
    sim_samples = saved[(saved["direction"] == "sim_to_real") & ~saved["excluded"]]
    sim_points = np.asarray(sim_samples.geometry.values)
    sim_points = sim_points[shapely.contains(municipality, sim_points)]

    canonical = json.loads((run_dir / "perimeter_metrics.json").read_text(encoding="utf-8"))
    window = box(*canonical["window"]["bounds"])
    sim_target = pc.strip_frame_from_line(sim.boundary, window)
    result = []
    for threshold_ha in thresholds_ha:
        minimum_area = threshold_ha * 10_000
        kept = [part for part in parts if part.area >= minimum_area]
        # The unfiltered baseline must use the same one-feature union as the
        # canonical validation; filtered versions union their retained parts.
        source = unary_union([scar_raw]) if minimum_area == 0 else unary_union(kept)
        scar = pc.fill_holes(source)
        real_points = np.asarray(pc.sample_boundary(scar).geometry.values)
        real_points = real_points[
            shapely.contains(municipality, real_points)
            & shapely.contains(window, real_points)
            & (shapely.distance(real_points, window.exterior) > 30)
        ]
        d_s2r = pc.directed_distances(gpd.GeoSeries(sim_points, crs=2100), scar.boundary)
        d_r2s = pc.directed_distances(gpd.GeoSeries(real_points, crs=2100), sim_target)
        s2r = pc.direction_stats(d_s2r)
        r2s = pc.direction_stats(d_r2s)
        result.append({
            "minimum_component_area_ha": minimum_area / 10_000,
            "n_components_kept": len(kept),
            "raw_area_kept_km2": round(sum(part.area for part in kept) / 1e6, 3),
            "raw_exterior_kept_km": round(sum(part.exterior.length for part in kept) / 1000, 1),
            "n_real_samples_common": len(real_points),
            "n_sim_samples_common": len(sim_points),
            "mean_sim_to_real_m": s2r["mean_m"],
            "mean_real_to_sim_m": r2s["mean_m"],
            "d_t_m": pc.hour_indicators(s2r, r2s)["d_t_m"],
        })
        print(json.dumps(result[-1], ensure_ascii=True), flush=True)
    return {"hour": 18, "threshold_sensitivity": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("temporal", "spatial", "hourly", "fragments"), required=True)
    parser.add_argument("--fragment-thresholds-ha", nargs="+", type=float, default=[0, 0.1, 1, 10])
    args = parser.parse_args()
    section = args.section
    started = time.perf_counter()
    if section == "temporal":
        result = temporal_check()
    elif section == "spatial":
        result = spatial_check()
    elif section == "hourly":
        result = hourly_common_boundary_check()
    else:
        result = fragment_check(tuple(args.fragment_thresholds_ha))
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    print(f"Runtime: {time.perf_counter() - started:.1f} s", flush=True)
