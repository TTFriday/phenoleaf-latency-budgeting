"""Independent recompute of the JRTIP P1 multiscale quality + service-time benchmark.

CPU only. Run AFTER the benchmark queue exits: 18 label-map rebuilds, 54 COCO
evaluations and ~9.7k timing records must never compete for cores with the
saturated timing sessions, or the measured service times would be perturbed.

Recomputed here rather than trusted from the pilot:
  * the development selection (roles, valid masks, plant grouping, order) from the frozen
    inventory CSV, compared with the recorded stream digest and population;
  * leaf counts: every logged per-frame `gt_count` against the audited `n_instances`, and a
    seeded sample re-derived from the label-map PNG pixels;
  * per-genotype COCO mask AP/AP50/AP75 and the frame -> plant -> genotype -> equal-domain
    count-MAE hierarchy, compared against quality.json;
  * per-scale service-time statistics and the same equal-mean hierarchy over
    role/genotype/plant, plus session-to-session deviation, compared against scales.json;
  * the predeclared relative-to-S640 tolerance flags (<=1 pp AP, <=0.2 leaf count MAE);
  * checkpoint / frozen-input hashes, weights hash, and each session's config digest.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mva_model import COCO_V1_SHA256  # noqa: E402
from src.mva_pilot import coco_ground_truth, coco_mask_ap  # noqa: E402
from src.jrtip_pilot import (DEV_ROLES, SCALES, digest_json, file_sha256, load_inputs,  # noqa: E402
                             population, select_dev)

SCALE_TOL = 1e-9
TIME_FIELDS = ("decode_ms", "resize_transfer_ms", "network_sync_ms",
               "restore_scores_masks_rle_serialize_sync_ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=str(ROOT / "jrtip/baseline_benchmark_v1"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--gt-count-sample", type=int, default=400,
                        help="frames whose leaf count is re-derived from label-map PNG pixels")
    args = parser.parse_args()
    out = Path(args.benchmark).resolve()
    manifest = json.loads((out / "benchmark_config.json").read_text(encoding="utf-8-sig"))
    quality = json.loads((out / "quality.json").read_text(encoding="utf-8-sig"))
    final = json.loads((out / "scales.json").read_text(encoding="utf-8-sig"))
    config = manifest["training_config"]
    step = quality["optimizer_step"]
    checks, failures = [], []

    def note(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail if isinstance(detail, str)
                       else json.dumps(detail, ensure_ascii=False)})
        if not ok:
            failures.append(name)

    def near(name, mine, reported, tol, detail=""):
        ok = reported is not None and abs(float(mine) - float(reported)) <= tol
        note(name, ok, f"{name}: recomputed={mine!r} recorded={reported!r} tol={tol} {detail}".strip())

    # ---- frozen inputs and declared scope --------------------------------------------------
    for key in ("inventory", "instances", "readiness"):
        note(f"{key}_sha256_matches_training_config",
             file_sha256(config[key]) == config[key + "_sha256"], str(config[key]))
    note("checkpoint_sha256_matches", file_sha256(manifest["checkpoint"]) == manifest["checkpoint_sha256"],
         manifest["checkpoint"])
    note("coco_weights_sha256_matches", file_sha256(config["weights"]) == config["weight_sha256"],
         str(config.get("weights")))
    note("coco_weights_are_the_pinned_generic_v1",
         config["weight_sha256"] == COCO_V1_SHA256, config["weight_sha256"])
    note("frozen_training_config_declares_fp32_seed43_and_no_formal_run",
         config["precision"] == "FP32" and int(config["seed"]) == 43
         and config["formal_36_epoch_run"] is False)
    note("benchmark_did_not_reselect_weights", manifest["checkpoint_reselected"] is False)
    note("benchmark_checkpoint_is_the_training_best",
         Path(manifest["checkpoint"]).name == Path(manifest["selected_best"]["checkpoint"]).name
         and manifest["checkpoint_sha256"] == manifest["selected_best"]["checkpoint_sha256"])
    note("quality_pass_covers_full_development_set",
         manifest["quality_max_frames_per_plant"] == 0 and quality["diagnostic_subset"] is False)
    note("timing_is_not_a_diagnostic_subset",
         manifest["warmup"] >= 50 and manifest["timing_max_frames_per_plant"] == 20
         and final["diagnostic_timing"] is False,
         f"warmup={manifest['warmup']} timing_max={manifest['timing_max_frames_per_plant']}")
    note("tolerances_are_the_predeclared_ones",
         final["tolerances"] == {"AP_drop_pp": 1.0, "count_MAE_increase": 0.2})
    note("no_formal_36_epoch_and_no_confirmatory_claim",
         final["formal_36_epoch_run"] is False and final["confirmatory_quality_claim"] is False
         and "36-epoch" in final["predeclared_pilot_note"])
    note("shared_device_recorded_not_exclusive",
         final["exclusive_platform_guaranteed"] is False
         and manifest["exclusive_platform_guaranteed"] is False)

    # ---- rebuild the development selection from the CSV ------------------------------------
    rows, data, _ = load_inputs(config["inventory"], config["instances"], config["readiness"],
                                config["data_root"])
    selected = select_dev(rows, manifest["quality_max_frames_per_plant"])
    by_image = {row["image_id"]: row for row in selected}
    note("selected_stream_order_digest_matches",
         digest_json([r["image_id"] for r in selected]) == quality["selected_stream_sha256"])
    note("selected_population_matches", population(selected) == quality["selected_population"],
         json.dumps(population(selected)))
    note("dev_roles_only", {r["split_role"] for r in selected} == set(DEV_ROLES))
    groups = defaultdict(list)
    for row in selected:
        groups[(row["split_role"], row["genotype"])].append(row)

    # ---- per-frame logs vs the audit -------------------------------------------------------
    per_scale, by_key = {}, {}
    for scale in SCALES:
        records = [json.loads(line) for line in
                   (out / f"quality_{step:06d}_S{scale}.jsonl").read_text(encoding="utf-8").splitlines()]
        per_scale[scale] = records
        logged = defaultdict(list)
        for record in records:
            by_key[(scale, record["split_role"], record["genotype"], record["image_id"])] = record
            logged[(record["split_role"], record["genotype"])].append(record["image_id"])
        note(f"quality_log_S{scale}_stream_matches_selection",
             set(logged) == set(groups) and all(logged[key] == [r["image_id"] for r in groups[key]]
                                                for key in logged),
             f"rows={len(records)} expected={len(selected)}")
    count_vs_audit = [f"{scale}/{record['image_id']}" for scale in SCALES for record in per_scale[scale]
                      if int(record["gt_count"]) != int(by_image[record["image_id"]]["n_instances"])]
    note("every_logged_gt_count_equals_audit_instance_count", not count_vs_audit,
         f"mismatches={len(count_vs_audit)} first={count_vs_audit[:5]}")
    internal = [f"{scale}/{record['image_id']}" for scale in SCALES for record in per_scale[scale]
                if int(record["count_abs_error"]) != abs(int(record["count"]) - int(record["gt_count"]))]
    note("count_abs_error_equals_count_minus_gt", not internal, f"mismatches={len(internal)}")
    thresholds = {(record.get("ap_score_threshold"), record.get("count_score_threshold"),
                   record.get("mask_threshold")) for scale in SCALES for record in per_scale[scale]}
    note("score_and_mask_thresholds_are_the_declared_ones", thresholds == {(.001, .5, .5)},
         json.dumps(sorted(str(t) for t in thresholds)))
    note("no_locked_role_in_quality_logs",
         all(record["split_role"] in DEV_ROLES for scale in SCALES for record in per_scale[scale]))
    note("predictions_restored_to_original_frame_size",
         all(tuple(record["original_hw"]) == (by_image[record["image_id"]]["height"],
                                              by_image[record["image_id"]]["width"])
             for scale in SCALES for record in per_scale[scale]),
         "every logged mask raster is at the original RGB height/width")
    rng = np.random.default_rng(43)
    ids = [row["image_id"] for row in selected]
    picks = {ids[i] for i in rng.choice(len(ids), size=min(args.gt_count_sample, len(ids)), replace=False)}
    pixel_mismatches = []
    for row in selected:
        if row["image_id"] in picks:
            derived = int(data.label(row)["masks"].shape[0])
            if derived != int(row["n_instances"]):
                pixel_mismatches.append((row["image_id"], derived, int(row["n_instances"])))
    note("sampled_leaf_counts_rederived_from_label_pngs", not pixel_mismatches,
         f"sampled={len(picks)} mismatches={len(pixel_mismatches)} first={pixel_mismatches[:5]}")

    # ---- COCO AP and count MAE recomputation ----------------------------------------------
    gt_cache = {key: coco_ground_truth(frames, data) for key, frames in groups.items()}
    recomputed_quality, max_ap_gap, max_mae_gap = {}, 0.0, 0.0
    for scale in SCALES:
        reported_genotypes = quality["scales"][str(scale)]["per_genotype"]
        per_role = {}
        for (role, genotype), frames in sorted(groups.items()):
            predictions, errors = [], defaultdict(list)
            for index, row in enumerate(frames, 1):
                record = by_key[(scale, role, genotype, row["image_id"])]
                predictions.extend({"image_id": index, "category_id": 1, "score": item["score"],
                                    "segmentation": item["segmentation"]} for item in record["instances"])
                errors[row["biological_plant_id"]].append(record["count_abs_error"])
            metrics = coco_mask_ap(gt_cache[(role, genotype)], predictions)
            mae = float(np.mean([np.mean(v) for v in errors.values()]))
            reported = reported_genotypes[role][genotype]
            for key, value in (("AP", metrics["AP"]), ("AP50", metrics["AP50"]),
                               ("AP75", metrics["AP75"]), ("count_MAE", mae)):
                max_ap_gap = max(max_ap_gap, abs(value - float(reported[key]))) if key != "count_MAE" else max_ap_gap
                max_mae_gap = max(max_mae_gap, abs(value - float(reported[key]))) if key == "count_MAE" else max_mae_gap
            per_role.setdefault(role, {})[genotype] = {"AP": metrics["AP"], "AP50": metrics["AP50"],
                                                       "AP75": metrics["AP75"], "count_MAE": mae,
                                                       "plants": len(errors), "frames": len(frames),
                                                       "gt_instances": len(gt_cache[(role, genotype)]["annotations"])}
            note(f"S{scale}_{role}_{genotype}_population",
                 int(reported["frames"]) == len(frames) and int(reported["plants"]) == len(errors)
                 and int(reported["gt_instances"]) == len(gt_cache[(role, genotype)]["annotations"]),
                 f"frames={len(frames)} plants={len(errors)}")
        roles = {role: {key: float(np.mean([g[key] for g in per_genotype.values()]))
                        for key in ("AP", "AP50", "AP75", "count_MAE")}
                 for role, per_genotype in per_role.items()}
        equal = {key: float(np.mean([roles[r][key] for r in DEV_ROLES]))
                 for key in ("AP", "AP50", "AP75", "count_MAE")}
        recomputed_quality[str(scale)] = {"per_genotype": per_role, "role_macro": roles,
                                          "equal_domain_macro": equal,
                                          "genotypes_per_role": {r: len(v) for r, v in per_role.items()}}
        reported_scale = quality["scales"][str(scale)]
        for role in DEV_ROLES:
            for key in ("AP", "AP50", "AP75", "count_MAE"):
                near(f"role_macro_S{scale}_{role}_{key}", roles[role][key],
                     reported_scale["role_macro"][role][key], SCALE_TOL)
        for key in ("AP", "AP50", "AP75", "count_MAE"):
            near(f"equal_domain_macro_S{scale}_{key}", equal[key],
                 reported_scale["equal_domain_macro"][key], SCALE_TOL)
    note("per_genotype_AP_recomputed_exactly", max_ap_gap <= SCALE_TOL, f"max_abs_gap_pp={max_ap_gap:.3e}")
    note("per_genotype_count_MAE_recomputed_exactly", max_mae_gap <= SCALE_TOL,
         f"max_abs_gap_leaf={max_mae_gap:.3e}")

    # ---- service-time recomputation --------------------------------------------------------
    timing_records, session_objects = [], {}
    for session in (1, 2, 3):
        records = [json.loads(line) for line in
                   (out / f"timing_session_{session}.jsonl").read_text(encoding="utf-8").splitlines()]
        timing_records.extend(records)
        # Each scale replays the whole declared timing stream in manifest order; the flat
        # file is scale after scale, so the comparison must be done per scale.
        note(f"timing_session_{session}_stream_equals_manifest",
             all([r["image_id"] for r in records if r["scale"] == scale] == manifest["timing_image_ids"]
                 for scale in SCALES),
             f"requests={len(records)} per_scale={len(records) // len(SCALES)} "
             f"declared_stream={len(manifest['timing_image_ids'])}")
        obj = json.loads((out / f"session_{session}.json").read_text(encoding="utf-8-sig"))
        session_objects[session] = obj
        note(f"session_{session}_completed_and_bound_to_this_config",
             obj.get("completed") is True and obj["benchmark_config_sha256"] == digest_json(manifest))
        note(f"session_{session}_scale_order_rotates",
             obj["scale_order"] == list(SCALES[session - 1:] + SCALES[:session - 1]), str(obj["scale_order"]))
        # Protocol warmup: at least max(requested, 60) iterations, at most 200, and it stops
        # only when the medians of the last three groups of 20 sit within 5% of their mean.
        for scale in SCALES:
            warmed = obj["scales"][str(scale)]["warmup"]
            times = np.asarray(warmed["raw_ms"], dtype=np.float64)
            minimum = max(manifest["warmup"], 60)
            medians = [float(np.median(times[j:j + 20]))
                       for j in range(len(times) - 60, len(times), 20)]
            stability = (max(medians) - min(medians)) / max(float(np.mean(medians)), 1e-12)
            note(f"session{session}_S{scale}_warmup_bounds",
                 warmed["requested"] == manifest["warmup"] and len(times) == warmed["actual"]
                 and minimum <= warmed["actual"] <= warmed["maximum"] == 200,
                 f"requested={warmed['requested']} actual={warmed['actual']} min={minimum}")
            note(f"session{session}_S{scale}_warmup_stability_recomputed",
                 bool(stability <= .05) == warmed["stable_recent_three_groups20"]
                 and warmed["stable_recent_three_groups20"] is True
                 and warmed["jitter"] is False and warmed["diagnostic_warmup"] is False,
                 f"relative_spread={stability:.6f} flag={warmed['stable_recent_three_groups20']}")
            # The iterations the protocol discards must actually be slower than the settled
            # regime; no fixed multiplier is part of the predeclared rule.
            note(f"session{session}_S{scale}_warmup_discarded_cold_iterations",
                 bool(float(times[0]) > float(np.median(times[-60:]))),
                 f"first={float(times[0]):.1f}ms median_last60={float(np.median(times[-60:])):.1f}ms")

    def aggregate(records):
        values = np.asarray([r["service_ms"] for r in records], dtype=np.float64)
        per_image = defaultdict(list)
        for record in records:
            per_image[(record["split_role"], record["genotype"], record["plant"], record["image_id"])].append(
                record["service_ms"])
        plants = defaultdict(list)
        for (role, genotype, plant, _), times in per_image.items():
            plants[(role, genotype, plant)].append(float(np.mean(times)))
        genotypes = defaultdict(list)
        for (role, genotype, _), times in plants.items():
            genotypes[(role, genotype)].append(float(np.mean(times)))
        roles = {role: float(np.mean([np.mean(v) for (r, _), v in genotypes.items() if r == role]))
                 for role in DEV_ROLES}
        return {"requests_technical_repeats": int(len(values)), "mean_ms": float(values.mean()),
                "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "saturated_throughput_per_second": float(1000 * len(values) / values.sum()),
                "equal_domain_mean_ms": float(np.mean(list(roles.values()))),
                "role_genotype_plant_equal_mean_ms": roles,
                "stage_mean_ms": {key: float(np.mean([r[key] for r in records])) for key in TIME_FIELDS},
                "plant_units": len(plants), "genotype_units": len(genotypes),
                "image_units": len(per_image)}

    reported_rows = {int(row["scale"]): row for row in final["rows"]}
    session_deviation, stage_gap, timing_population = {}, 0.0, {}
    for scale in SCALES:
        records = [r for r in timing_records if r["scale"] == scale]
        mine = aggregate(records)
        row = reported_rows[scale]
        for key in ("requests_technical_repeats", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                    "saturated_throughput_per_second", "equal_domain_mean_ms"):
            near(f"timing_S{scale}_{key}", mine[key], row[key], 1e-9)
        for role in DEV_ROLES:
            near(f"timing_S{scale}_{role}_equal_mean_ms", mine["role_genotype_plant_equal_mean_ms"][role],
                 row["role_genotype_plant_equal_mean_ms"][role], 1e-9)
        for key in TIME_FIELDS:
            near(f"timing_S{scale}_stage_{key}", mine["stage_mean_ms"][key], row["stage_mean_ms"][key], 1e-9)
        stage_sum = sum(mine["stage_mean_ms"][key] for key in TIME_FIELDS)
        stage_gap = max(stage_gap, abs(stage_sum - mine["mean_ms"]) / mine["mean_ms"])
        timing_population[str(scale)] = {k: mine[k] for k in ("plant_units", "genotype_units", "image_units")}
        per_session = {str(s): aggregate([r for r in records if r["session"] == s]) for s in (1, 2, 3)}
        for s, sub in per_session.items():
            near(f"timing_S{scale}_session{s}_mean_ms", sub["mean_ms"],
                 row["per_session"][s]["mean_ms"], 1e-9)
        means = [sub["mean_ms"] for sub in per_session.values()]
        session_deviation[str(scale)] = {
            "per_session_mean_ms": {s: sub["mean_ms"] for s, sub in per_session.items()},
            "max_relative_deviation": (max(means) - min(means)) / float(np.mean(means))}
    note("stage_times_partition_the_service_time", stage_gap <= 1e-9, f"max_relative_gap={stage_gap:.3e}")
    note("session_to_session_stability_within_5_percent",
         all(v["max_relative_deviation"] <= 0.05 for v in session_deviation.values()),
         session_deviation)
    note("timing_image_stream_is_the_declared_dev_subset",
         set(manifest["timing_image_ids"]) <= set(by_image)
         and len(manifest["timing_image_ids"]) == len(set(manifest["timing_image_ids"])),
         f"declared={len(manifest['timing_image_ids'])} dev_selection={len(selected)}")
    note("timing_subset_reproduces_the_per_plant_cap_rule",
         manifest["timing_stream_sha256"] == digest_json(
             [r["image_id"] for r in select_dev(rows, manifest["timing_max_frames_per_plant"])]),
         f"timing_frames={len(manifest['timing_image_ids'])}")
    note("no_locked_role_in_timing_logs",
         all(record["split_role"] in DEV_ROLES for record in timing_records))

    # ---- predeclared tolerances ------------------------------------------------------------
    ref = recomputed_quality["640"]["equal_domain_macro"]
    tolerance_rows = []
    for scale in SCALES:
        equal = recomputed_quality[str(scale)]["equal_domain_macro"]
        entry = {"scale": scale, "AP_drop_pp": ref["AP"] - equal["AP"],
                 "count_MAE_increase": equal["count_MAE"] - ref["count_MAE"]}
        entry["point_tolerances_met"] = bool(entry["AP_drop_pp"] <= 1.0 and entry["count_MAE_increase"] <= 0.2)
        reported = reported_rows[scale]["relative_to_S640"]
        for key in ("AP_drop_pp", "count_MAE_increase"):
            near(f"tolerance_S{scale}_{key}", entry[key], reported["equal_domain_macro"][key], SCALE_TOL)
        note(f"tolerance_flag_S{scale}",
             entry["point_tolerances_met"] == bool(reported["equal_domain_macro"]["point_tolerances_met"]))
        for role in DEV_ROLES:
            mine_role = {"AP_drop_pp": recomputed_quality["640"]["role_macro"][role]["AP"]
                         - recomputed_quality[str(scale)]["role_macro"][role]["AP"],
                         "count_MAE_increase": recomputed_quality[str(scale)]["role_macro"][role]["count_MAE"]
                         - recomputed_quality["640"]["role_macro"][role]["count_MAE"]}
            entry_role = reported[role]
            for key in ("AP_drop_pp", "count_MAE_increase"):
                near(f"tolerance_S{scale}_{role}_{key}", mine_role[key], entry_role[key], SCALE_TOL)
            entry[role] = {**mine_role, "point_tolerances_met": bool(
                mine_role["AP_drop_pp"] <= 1.0 and mine_role["count_MAE_increase"] <= 0.2)}
        tolerance_rows.append(entry)
    note("reference_row_is_S640_only", [r["reference"] for r in final["rows"]] == [False, False, True])
    note("row_AP_columns_equal_recomputed_equal_domain",
         all(abs(float(reported_rows[s]["AP"])
                 - recomputed_quality[str(s)]["equal_domain_macro"]["AP"]) <= SCALE_TOL for s in SCALES))

    machine = {
        "quality_machine_start_contention":
            ((json.loads((out / "quality_machine_start.json").read_text(encoding="utf-8")).get("nvidia")) or {}).get("contention"),
        "quality_machine_end_contention":
            ((json.loads((out / "quality_machine_end.json").read_text(encoding="utf-8")).get("nvidia")) or {}).get("contention"),
        "timing_contention_per_scale": {str(s): reported_rows[s]["contention"] for s in SCALES},
        "timing_warmup_jitter_per_scale": {str(s): reported_rows[s]["jitter"] for s in SCALES},
        "quality_elapsed_seconds": quality["elapsed_seconds"]}

    result = {"verified_utc": datetime.now(timezone.utc).isoformat(), "benchmark": str(out),
              "checkpoint": manifest["checkpoint"], "checkpoint_step": step,
              "development_population": quality["selected_population"],
              "recomputed_quality": recomputed_quality,
              "recomputed_tolerances": tolerance_rows,
              "session_timing_deviation": session_deviation,
              "timing_population": timing_population, "machine_context": machine,
              "checks": checks, "checks_total": len(checks),
              "failed_checks": [c["check"] for c in checks if not c["pass"]],
              "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
              "scope_limits": [
                  "Development domains only (V_SOURCE/V_SHIFT); C_POLICY, T_ID, T_GENO and T_EXT untouched.",
                  f"Single seed 43 at the {step}-update pilot checkpoint; NOT the formal 36-epoch schedule.",
                  "Shared desktop GPU under WDDM: service times are relative bottlenecks, not deployment latency.",
                  "Point estimates only; tolerance flags are development feasibility, not noninferiority.",
                  "Ground truth is the discrete manual label layer; leaf identity is consistent within a plant only."]}
    destination = Path(args.out) if args.out else out / "benchmark_independent_verification.json"
    if destination.exists():
        raise FileExistsError(f"Verification record already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["status"], "| checks", result["checks_total"], "| failed", result["failed_checks"][:12])
    for entry in tolerance_rows:
        print(f"S{entry['scale']}: equal-domain AP drop={entry['AP_drop_pp']:.6f} pp, "
              f"count MAE increase={entry['count_MAE_increase']:.6f} leaf, "
              f"tolerances met={entry['point_tolerances_met']}")
    print("wrote", destination)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
