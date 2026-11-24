"""Independent CPU recheck of the JRTIP service-time decomposition run.

Why this exists: the decomposition script measures spans with CUDA events inside the
inference process, and its aggregation decides which frames count as an interference-free
window.  This verifier re-derives every reported number from the raw per-frame JSONL with
its own code paths (numpy and the standard library only; nothing imported from the measured
script and no CUDA), so an aggregation mistake cannot hide behind agreement with its producer.

Discipline carried over from the benchmark verifier: an assertion here must mirror the
implementation that produced the data, never a belief about how the data "should" look.  If a
check fails, first decide whose expectation is wrong before touching either side, and keep the
failing record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Field lists mirrored from src/jrtip_bottleneck_decomposition.py (STAGE_FIELDS /
# FRAME_FIELDS).  Spelled out here so a silent change there surfaces as a missing key.
STAGE_FIELDS = ("scores_to_cpu_ms", "mask_restore_interpolate_to_cpu_ms", "mask_threshold_ms",
                "boxes_labels_ms", "rle_encode_ms", "serialize_ms")
MODULE_SPANS = ("transform_ms", "backbone_ms", "rpn_ms", "roi_heads_ms", "eval_postprocess_ms")
FRAME_FIELDS = ("decode_ms", "resize_transfer_ms", "host_to_device_event_ms", "forward_wall_ms",
                "forward_second_wall_ms", "forward_event_ms", "transform_ms", "backbone_ms",
                "rpn_ms", "roi_heads_ms", "eval_postprocess_ms", "postprocess_ms",
                "service_reconstructed_ms", "serialized_bytes", "n_instances",
                "returned_detections", *STAGE_FIELDS)
SCALES = (320, 480, 640)
DEV_ROLES = ("V_SOURCE", "V_SHIFT")
LOCKED_ROLES = ("C_POLICY", "T_ID", "T_GENO", "T_EXT")
BIN_ORDER = ("0-2", "3-4", "5-6", "7-9", "10+")
KEYS = ("n", "mean", "median", "p05", "p95", "min", "max")

checks: list[dict] = []


def note(name, ok, detail=None):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f" | {detail}"))
    return bool(ok)


def digest_json(obj) -> str:
    """Same canonicalisation as jrtip_pilot.digest_json for JSON-native inputs."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {"n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "p05": float(np.percentile(array, 5)), "p95": float(np.percentile(array, 95)),
            "min": float(array.min()), "max": float(array.max())}


def same(first, second, tol=1e-6) -> bool:
    """Compare one quantile dictionary, absent-vs-absent included."""
    if first is None or second is None:
        return first is None and second is None
    if set(first) != set(second):
        return False
    for key in second:
        left, right = first[key], second[key]
        if left is None or right is None:
            if left != right:
                return False
        elif abs(float(left) - float(right)) > tol + tol * abs(float(right)):
            return False
    return True


def close(value, target, tol=1e-6) -> bool:
    if value is None or target is None:
        return value is None and target is None
    return abs(float(value) - float(target)) <= tol + tol * abs(float(target))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def instance_bins(rows):
    """Mirror of instance_bins() in the measured script, rebuilt from the raw records."""
    groups = {}
    for row in rows:
        n = row["n_instances"]
        key = ("0-2" if n <= 2 else "3-4" if n <= 4 else "5-6" if n <= 6
               else "7-9" if n <= 9 else "10+")
        groups.setdefault(key, []).append(row)
    return {key: {"frames": len(groups[key]),
                  "mask_restore_ms": stats(r["mask_restore_interpolate_to_cpu_ms"] for r in groups[key]),
                  "rle_encode_ms": stats(r["rle_encode_ms"] for r in groups[key]),
                  "serialize_ms": stats(r["serialize_ms"] for r in groups[key]),
                  "postprocess_ms": stats(r["postprocess_ms"] for r in groups[key]),
                  "serialized_bytes": stats(r["serialized_bytes"] for r in groups[key])}
            for key in BIN_ORDER if key in groups}


def scaled_pixels(height, width, long_edge) -> int:
    """Mirror of the integer arithmetic in jrtip_pilot.resize_long (no interpolation needed)."""
    factor = long_edge / max(height, width)
    return max(1, round(height * factor)) * max(1, round(width * factor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="jrtip/bottleneck_decomposition_v1")
    parser.add_argument("--benchmark", default="jrtip/baseline_benchmark_v1")
    parser.add_argument("--inventory", default="shared/pilot_inventory.csv")
    parser.add_argument("--fidelity", type=int, default=25,
                        help="fidelity frames per scale as launched; cross-checked against the queue record")
    parser.add_argument("--queue", default="stage3_decomposition_queue.json",
                        help="queue spec that recorded the launch command")
    parser.add_argument("--out", default=None, help="destination FILE for the verification record")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    source = Path(args.benchmark).resolve()
    out = (Path(args.out).resolve() if args.out
           else run / "decomposition_independent_verification.json")
    if out.exists():
        raise FileExistsError(f"Verification record already exists: {out}")

    config = load_json(run / "decomposition_config.json")
    result = load_json(run / "decomposition.json")
    manifest = load_json(source / "benchmark_config.json")
    frames = {scale: load_jsonl(run / f"frames_S{scale}.jsonl") for scale in SCALES}
    reported_probe_context: dict[str, dict] = {}
    subset_report: dict[str, dict] = {}
    stride = int(config["stride"])
    stream = manifest["timing_image_ids"][::stride]
    with Path(args.inventory).open(encoding="utf-8-sig", newline="") as handle:
        geometry = {row["image_id"]: (int(row["height"]), int(row["width"])) for row in csv.DictReader(handle)}

    # ---- A. what the run claimed about itself ---------------------------------
    note("config_matches_production_parameters",
         stride == 3 and config["warmup_calls"] == 60 and config["probe_every"] == 25
         and close(config["clean_probe_ratio"], 1.15) and list(config["scales"]) == list(SCALES)
         and config["detection_budget_sweep"] == [100, 20, 5]
         and config["detection_budget_frames"] == 40 and config["smoke_limited"] is False
         and config["diagnostic_only"] is True and config["protocol_timing_replacement"] is False
         and config["formal_36_epoch_run"] is False and config["locked_roles_accessed"] == []
         and config["device"] == "cuda" and config["precision"] == "FP32",
         f"stride={stride} warmup={config['warmup_calls']} probe_every={config['probe_every']} "
         f"clean_ratio={config['clean_probe_ratio']}")
    launched = _launched_job(Path(args.queue), run)
    flags = (launched or {}).get("flags", {})
    note("launch_command_agrees_with_recorded_parameters",
         launched is not None
         and str(launched.get("script", "")).replace("\\", "/").endswith("src/jrtip_bottleneck_decomposition.py")
         and flags.get("--stride") == str(stride)
         and flags.get("--warmup-calls") == str(config["warmup_calls"])
         and flags.get("--fidelity") == str(args.fidelity)
         and flags.get("--cap-sweep") == ",".join(str(c) for c in config["detection_budget_sweep"])
         and flags.get("--cap-frames") == str(config["detection_budget_frames"])
         and flags.get("--device") == config["device"]
         and Path(flags.get("--benchmark", "")).resolve() == source,
         f"job={launched and launched.get('name')} log={(launched or {}).get('log')}")
    note("config_frames_per_scale_equals_strided_timing_stream",
         config["frames_per_scale"] == len(stream)
         and len(stream) == len(manifest["timing_image_ids"]) // stride
         + (1 if len(manifest["timing_image_ids"]) % stride else 0),
         f"{config['frames_per_scale']} vs {len(stream)} of {len(manifest['timing_image_ids'])}")
    note("config_subsampled_stream_digest_recomputed",
         config["subsampled_stream_sha256"] == digest_json(stream), config["subsampled_stream_sha256"][:16])
    note("config_full_stream_digest_matches_benchmark_manifest",
         digest_json(manifest["timing_image_ids"]) == manifest["timing_stream_sha256"]
         and config["timing_stream_sha256_of_full_selection"] == manifest["timing_stream_sha256"],
         manifest["timing_stream_sha256"][:16])
    checkpoint = Path(manifest["checkpoint"])
    note("frozen_checkpoint_unchanged",
         file_sha256(checkpoint) == manifest["checkpoint_sha256"] == config["checkpoint_sha256"],
         config["checkpoint_sha256"][:16])
    note("weights_digest_constant_recorded",
         config["coco_weights_sha256_constant"] == manifest["training_config"].get("weight_sha256")
         or bool(config["coco_weights_sha256_constant"]) and bool(config["training_config_weight_sha256"]),
         f"constant {config['coco_weights_sha256_constant'][:12]} training {str(config['training_config_weight_sha256'])[:12]}")
    note("run_wrote_expected_artifacts",
         all((run / f"frames_S{s}.jsonl").is_file() for s in SCALES)
         and (run / "decomposition.json").is_file() and (run / "decomposition_config.json").is_file())
    note("scope_limits_present", len(result["scope_limits"]) >= 5
         and any("locked" in text.lower() or "C_POLICY" in text for text in result["scope_limits"]))

    # ---- B. per-frame record integrity ----------------------------------------
    for scale, rows in frames.items():
        note(f"S{scale}_stream_equals_strided_manifest",
             [r["image_id"] for r in rows] == stream, f"{len(rows)} records")
        note(f"S{scale}_ordinals_contiguous", [r["ordinal"] for r in rows] == list(range(len(rows))))
        roles = {r["split_role"] for r in rows}
        note(f"S{scale}_roles_development_only",
             roles <= set(DEV_ROLES) and not roles & set(LOCKED_ROLES), ",".join(sorted(roles)))
        note(f"S{scale}_plant_and_genotype_present",
             all(r["plant"] and r["genotype"] for r in rows),
             f"{len({r['plant'] for r in rows})} plants, {len({r['genotype'] for r in rows})} genotypes")
        note(f"S{scale}_second_pass_forward_recorded",
             all("forward_second_event_ms" in r and "forward_second_wall_ms" in r for r in rows))
        note(f"S{scale}_scaled_pixels_matches_resize_long_integer_math",
             all(r["image_id"] in geometry and r["scaled_pixels"]
                 == scaled_pixels(*geometry[r["image_id"]], scale) for r in rows),
             f"mean {np.mean([r['scaled_pixels'] for r in rows]) / 1e6:.3f} Mpx, "
             f"{len({r['image_id'] for r in rows} - set(geometry))} ids missing from inventory")
        residual_stage = max(abs(r["postprocess_ms"] - sum(r[f] for f in STAGE_FIELDS)) for r in rows)
        note(f"S{scale}_postprocess_equals_stage_sum", residual_stage <= 1e-6,
             f"max residual {residual_stage:.2e} ms")
        residual_service = max(abs(r["service_reconstructed_ms"] - (r["decode_ms"] + r["resize_transfer_ms"]
                               + r["forward_wall_ms"] + r["postprocess_ms"])) for r in rows)
        note(f"S{scale}_service_reconstruction_formula", residual_service <= 1e-6,
             f"max residual {residual_service:.2e} ms")
        note(f"S{scale}_no_negative_or_missing_fields",
             all(r[f] is not None and r[f] >= 0 for r in rows for f in FRAME_FIELDS))
        spans = [r["forward_event_ms"] - sum(r[f] for f in MODULE_SPANS) for r in rows]
        note(f"S{scale}_module_spans_tile_forward",
             abs(float(np.median(spans))) <= 5.0,
             f"median {float(np.median(spans)):.2f} ms, min {min(spans):.2f}, max {max(spans):.2f}")
        note(f"S{scale}_module_spans_inside_forward",
             all(r["eval_postprocess_ms"] <= r["forward_event_ms"] + 1e-6
                 and all(r[f] <= r["forward_event_ms"] + 1e-6 for f in MODULE_SPANS) for r in rows))
        note(f"S{scale}_count_le_instances_le_detections",
             all(r["count"] <= r["n_instances"] <= r["returned_detections"] for r in rows),
             f"median detections {float(np.median([r['returned_detections'] for r in rows])):.0f}")
        gaps = [r["forward_wall_ms"] - r["forward_event_ms"] for r in rows]
        relative = [abs(g) / max(r["forward_event_ms"], 1e-9) for g, r in zip(gaps, rows)]
        note(f"S{scale}_wall_and_event_clocks_agree_within_one_percent",
             all(value <= 0.01 for value in relative),
             f"max relative discrepancy {100 * max(relative):.2f}%, median wall-event "
             f"{float(np.median(gaps)):.3f} ms, {sum(1 for g in gaps if g < 0)}/{len(rows)} frames "
             f"with wall below event (clock granularity, not queueing)")
        repeat = [r["forward_second_event_ms"] / max(r["forward_event_ms"], 1e-9) for r in rows]
        note(f"S{scale}_second_forward_repeat_available",
             all(v > 0 for v in repeat),
             f"median second/first ratio {float(np.median(repeat)):.3f}")
        inflation = [r["probe_matrix_inflation"] for r in rows]
        note(f"S{scale}_inflation_at_least_one", all(v >= 1.0 for v in inflation),
             f"median {float(np.median(inflation)):.3f}, max {max(inflation):.3f}")
        changes = [r["ordinal"] for i, r in enumerate(rows) if i
                   and r["probe_matrix_inflation"] != rows[i - 1]["probe_matrix_inflation"]]
        note(f"S{scale}_inflation_only_changes_at_probe_boundaries",
             all(o % config["probe_every"] == 0 or o == len(rows) - 1 for o in changes),
             f"{len(changes)} changes over {len(rows)} frames")

    # ---- C. aggregation recomputed from the raw records ------------------------
    for scale, rows in frames.items():
        summary = result["scales"][str(scale)]
        note(f"S{scale}_frames_count_matches_summary", summary["frames"] == len(rows))
        note(f"S{scale}_all_quantiles_recomputed",
             all(same(summary["all"][field], stats(r[field] for r in rows)) for field in FRAME_FIELDS),
             f"{len(FRAME_FIELDS)} fields x {len(KEYS)} statistics")
        kept = [row for row in rows
                if row["ordinal"] in {r["ordinal"] for r in rows
                                      if (r["ordinal"] % config["probe_every"] == 0
                                          or r["ordinal"] == len(rows) - 1)
                                      and r["probe_matrix_inflation"] <= config["clean_probe_ratio"]}]
        interval = [row for row in rows if row["probe_matrix_inflation"] <= config["clean_probe_ratio"]]
        subset_report[str(scale)] = {
            "frames": len(rows), "probe_boundary_frames_clean": len(kept),
            "interval_frames_clean": len(interval),
            "definition_note": ("aggregate() keeps records whose ordinal is itself a probe ordinal in a "
                                "clean window, not every record inside a clean window. Both subsets are "
                                "recomputed here so the smallness of the reported clean-window medians "
                                "is visible; the all-frame medians are the robust column."),
            "median_service_ms": {"all": stats(r["service_reconstructed_ms"] for r in rows)["median"],
                                  "probe_boundary_clean": stats(r["service_reconstructed_ms"] for r in kept)["median"] if kept else None,
                                  "interval_clean": stats(r["service_reconstructed_ms"] for r in interval)["median"] if interval else None},
            "median_forward_event_ms": {"all": stats(r["forward_event_ms"] for r in rows)["median"],
                                        "probe_boundary_clean": stats(r["forward_event_ms"] for r in kept)["median"] if kept else None,
                                        "interval_clean": stats(r["forward_event_ms"] for r in interval)["median"] if interval else None},
            "median_eval_postprocess_ms": {"all": stats(r["eval_postprocess_ms"] for r in rows)["median"],
                                           "probe_boundary_clean": stats(r["eval_postprocess_ms"] for r in kept)["median"] if kept else None,
                                           "interval_clean": stats(r["eval_postprocess_ms"] for r in interval)["median"] if interval else None}}
        note(f"S{scale}_clean_window_membership_recomputed",
             summary["clean_probe_frames"] == len(kept)
             and close(summary["clean_probe_fraction"], round(len(kept) / len(rows), 6), 0.0),
             f"boundary-clean {len(kept)}/{len(rows)}, interval-clean {len(interval)}/{len(rows)}")
        note(f"S{scale}_clean_window_quantiles_recomputed",
             all(same(summary["clean_window"][field], stats(r[field] for r in kept) if kept else {"n": 0})
                 for field in FRAME_FIELDS))
        gaps = [r["forward_wall_ms"] - r["forward_event_ms"] for r in kept]
        note(f"S{scale}_wall_minus_event_median_recomputed",
             close(summary["wall_minus_event_median_ms"], float(np.median(gaps)) if gaps else None))
        stored, rebuilt = summary["output_bins_by_instances"], instance_bins(kept)
        note(f"S{scale}_instance_bins_recomputed_on_clean_subset",
             set(stored) == set(rebuilt)
             and all(stored[k]["frames"] == rebuilt[k]["frames"]
                     and all(same(stored[k][f], rebuilt[k][f]) for f in
                             ("mask_restore_ms", "rle_encode_ms", "serialize_ms",
                              "postprocess_ms", "serialized_bytes"))
                     for k in rebuilt),
             f"{ {k: v['frames'] for k, v in stored.items()} }")
        floors = summary["fixed_shape_floor"]
        note(f"S{scale}_fixed_shape_floor_recorded",
             set(floors) == {"64", "256"} and all(floors[k]["event_ms_per_forward"] > 0 for k in floors),
             " ".join(f"{k}:{floors[k]['event_ms_per_forward']:.1f}ms" for k in sorted(floors)))
        probe = summary["probe"]
        reported_probe_context[str(scale)] = {
            key: probe[key].get("median") for key in
            ("matrix_inflation", "matrix_ms", "launch_ms", "cpu_ms", "foreign_contexts")}
        stability = summary["post_run_stability"]
        if stability.get("available") is not False and "recent_three_groups20_relative_spread" in stability:
            tail = [r["service_reconstructed_ms"] for r in rows][-60:]
            medians = [float(np.median(tail[j:j + 20])) for j in range(0, 60, 20)]
            spread = (max(medians) - min(medians)) / float(np.mean(medians))
            note(f"S{scale}_post_run_stability_recomputed",
                 close(stability["recent_three_groups20_relative_spread"], spread, 1e-9)
                 and stability["within_five_percent"] == bool(spread <= .05), f"spread {spread:.4f}")

    def pick(scale, field):
        """Mirror of pick(): the clean window is reported when it has any frames."""
        summary = result["scales"][str(scale)]
        clean = summary["clean_window"][field]
        return clean if clean.get("n") else summary["all"][field]

    medians = {s: pick(s, "forward_event_ms")["median"] for s in SCALES}
    service = {s: pick(s, "service_reconstructed_ms")["median"] for s in SCALES}
    post = {s: pick(s, "postprocess_ms")["median"] for s in SCALES}
    response = result["resolution_response"]
    note("resolution_response_recomputed",
         all(close(response["median_forward_event_ms"][str(s)], medians[s]) for s in SCALES)
         and close(response["ratio_largest_over_smallest"], max(medians.values()) / min(medians.values()), 1e-9)
         and response["reference_scale"] == min(medians)
         and close(response["forward_ms_change_reference_to_640"], medians[640] - medians[min(medians)])
         and all(close(response["forward_ms_per_megapixel"][str(s)],
                       (medians[s] - medians[min(medians)]) / ((s ** 2 - min(medians) ** 2) / 1e6), 1e-9)
                 for s in SCALES if s != min(medians)),
         f"forward medians { {s: round(v, 1) for s, v in medians.items()} }, "
         f"ratio {response['ratio_largest_over_smallest']:.4f}")
    verdict, queue_gap = result["verdict"], {s: result["scales"][str(s)]["wall_minus_event_median_ms"] for s in SCALES}
    note("verdict_flags_recomputed",
         verdict["service_time_flat_across_scales"] == bool(max(service.values()) - min(service.values())
                                                            <= 0.02 * min(service.values()))
         and verdict["forward_device_time_scales_with_resolution"] == bool(response["ratio_largest_over_smallest"] >= 1.20)
         and verdict["output_side_cost_scale_independent"] == bool(max(post.values()) - min(post.values())
                                                                   <= 0.05 * min(post.values()))
         and verdict["queue_or_sync_above_quarter_of_forward"] == bool(
             all(g is not None and g > 0.25 * medians[s] for s, g in queue_gap.items())),
         json.dumps({k: v for k, v in verdict.items()}))

    # ---- D. instrumented restore vs the production restore_output -------------
    fidelity = result["fidelity_checks"]
    per_scale = defaultdict(int)
    for record in fidelity["records"]:
        per_scale[record["scale"]] += 1
    note("fidelity_frames_match_requested_count",
         fidelity["frames"] == args.fidelity * len(SCALES)
         and all(min(args.fidelity, len(frames[s])) == args.fidelity for s in SCALES),
         f"{fidelity['frames']} requested {args.fidelity} x {len(SCALES)}; stored samples "
         f"{ {k: v for k, v in per_scale.items()} }")
    note("fidelity_tolerant_match_complete",
         fidelity["tolerant_match"] == fidelity["frames"]
         and 0 <= fidelity["exact_digest_match"] <= fidelity["tolerant_match"],
         f"tolerant {fidelity['tolerant_match']}/{fidelity['frames']}, "
         f"exact {fidelity['exact_digest_match']}/{fidelity['frames']}")
    note("fidelity_samples_are_first_frames_with_right_ids",
         all(r["ordinal"] < args.fidelity and r["image_id"] == stream[r["ordinal"]]
             for r in fidelity["records"]), f"{len(fidelity['records'])} sampled records")

    # ---- E. detection-budget sweep invariants ---------------------------------
    for scale, rows in frames.items():
        sweep = result["scales"][str(scale)]["detection_cap_sweep"]
        caps = config["detection_budget_sweep"]
        note(f"S{scale}_sweep_budgets_and_reference",
             list(sweep["by_budget"]) == [str(c) for c in caps] and sweep["reference_budget"] == caps[0],
             f"model_default_detections_per_img={sweep['original_budget']}")
        note(f"S{scale}_sweep_frames_equal_requested",
             all(v["frames"] == v["compared_frames"] == config["detection_budget_frames"]
                 and 0 <= v["count_equal_to_reference"] <= v["compared_frames"]
                 and 0 <= v["instances_equal_to_reference"] <= v["compared_frames"]
                 for v in sweep["by_budget"].values()))
        note(f"S{scale}_sweep_detections_respect_budget",
             all(v["returned_detections"]["max"] <= int(cap) for cap, v in sweep["by_budget"].items()),
             {cap: v["returned_detections"]["max"] for cap, v in sweep["by_budget"].items()})
        note(f"S{scale}_sweep_medians_positive_and_postprocess_inside_forward",
             all(v["service_reconstructed_ms"]["median"] > 0 and v["forward_event_ms"]["median"] > 0
                 and v["eval_postprocess_ms"]["median"] <= v["forward_event_ms"]["median"]
                 for v in sweep["by_budget"].values()))

    # ---- F. context numbers, reported but not asserted ------------------------
    protocol = {scale: [] for scale in SCALES}
    for session in (1, 2, 3):
        for record in load_jsonl(source / f"timing_session_{session}.jsonl"):
            protocol[int(record["scale"])].append(float(record["service_ms"]))
    reported = {
        "note": ("The decomposition run subsamples the stream, forwards twice per frame, and times "
                 "inside a single process; the frozen protocol sessions remain the record for "
                 "service time. These values are shown for context and are NOT compared as equal."),
        "protocol_timing_mean_ms": {str(s): float(np.mean(v)) for s, v in protocol.items()},
        "decomposition_service_median_ms_all_frames":
            {str(s): stats(r["service_reconstructed_ms"] for r in frames[s])["median"] for s in SCALES},
        "decomposition_service_median_ms_clean_window":
            {str(s): result["scales"][str(s)]["clean_window"]["service_reconstructed_ms"].get("median")
             for s in SCALES},
        "decomposition_service_median_ms_inflation_le_1_30":
            {str(s): stats([r["service_reconstructed_ms"] for r in frames[s]
                            if r["probe_matrix_inflation"] <= 1.30])["median"] for s in SCALES},
        "eval_postprocess_share_of_forward":
            {str(s): float(np.median([r["eval_postprocess_ms"] / max(r["forward_event_ms"], 1e-9)
                                      for r in frames[s]])) for s in SCALES},
        "postprocess_share_of_service":
            {str(s): float(np.median([r["postprocess_ms"] / max(r["service_reconstructed_ms"], 1e-9)
                                      for r in frames[s]])) for s in SCALES},
        "forward_event_ms_all_frames": {str(s): stats([r["forward_event_ms"] for r in frames[s]])["median"]
                                        for s in SCALES},
        "probe_interference_context": reported_probe_context,
        "clean_subset_definitions": subset_report,
        "instance_bins_all_frames": {str(s): instance_bins(frames[s]) for s in SCALES},
        "detection_budget_sweep_service_median_ms":
            {str(s): {cap: v["service_reconstructed_ms"]["median"]
                      for cap, v in result["scales"][str(s)]["detection_cap_sweep"]["by_budget"].items()}
             for s in SCALES},
    }

    passed = sum(1 for item in checks if item["ok"])
    record = {"verdict": "PASS" if passed == len(checks) else "FAIL", "checks_total": len(checks),
              "checks_passed": passed, "checks_failed": len(checks) - passed,
              "failed": [item for item in checks if not item["ok"]],
              "reported_not_checked": reported, "run": str(run),
              "config_sha256": digest_json(config), "decomposition_sha256": digest_json(result),
              "frames_sha256": {str(s): digest_json(frames[s]) for s in SCALES},
              "method": ("Recomputed the strided stream identity and digest, resize_long integer "
                         "geometry, stage additivity, the service reconstruction formula, span "
                         "containment, detection-ordering invariants, probe-window membership, every "
                         "reported quantile, instance bins, the resolution response, the verdict flags, "
                         "the fidelity counts and the budget-sweep invariants from the raw JSONL using "
                         "numpy only.")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{record['verdict']}: {passed}/{len(checks)} checks | wrote {out}")
    return 0 if record["verdict"] == "PASS" else 1


def _launched_job(path: Path, out_dir: Path):
    """The queue entry that launched this run, located by its recorded --out directory."""
    if not path.is_file():
        return None
    for job in json.loads(path.read_text(encoding="utf-8")).get("jobs", []):
        command = [str(item) for item in job.get("command", [])]
        flags, index = {}, 0
        while index < len(command):
            if command[index].startswith("--") and index + 1 < len(command):
                flags[command[index]] = command[index + 1]
                index += 2
            else:
                index += 1
        if flags.get("--out") and Path(flags["--out"]).resolve() == out_dir:
            return {"name": job.get("name"), "script": command[2] if len(command) > 2 else None,
                    "log": job.get("log"), "flags": flags}
    return None


if __name__ == "__main__":
    raise SystemExit(main())
