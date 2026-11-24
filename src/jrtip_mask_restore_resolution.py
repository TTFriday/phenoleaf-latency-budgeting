"""Mask-restore resolution g in {1.0, 0.75, 0.5}: quality price and service-time gain (P1c(c)).

What is varied
--------------
``src/jrtip_pilot.py::restore_output`` interpolates every instance's mask probabilities from the
640-long input grid to the *original* frame, thresholds at 0.5, RLE-encodes the binary mask, and
expresses boxes in original-image coordinates.  That stage - not the network - is what the frozen
attribution table measured at 249-286 ms/frame.  This script declares a delivery frame that is a
factor ``g`` smaller and lets the untouched arithmetic run on it:

    meta_g = dict(meta, original_hw=(round(g*h), round(g*w)), scale_xy=[sx*g, sy*g])

so masks are rasterised and boxes expressed in one consistent g-frame.  No library file and no
frozen project function is modified; ``restore_output``'s arithmetic is reused verbatim, which is
why ``g = 1.0`` must reproduce the archived cap-100 stream bitwise (that is the gate).

Modes
-----
``quality``  one GPU pass over the frozen dev stream; per frame the model runs once and all three
             g payloads are derived from the same output tensors (the forward is identical across
             arms, so three separate passes would only re-measure the network).  Writes native
             g-frame streams.
``rescore``  CPU post-pass: verify the g=1.0 stream byte-for-byte against the archived cap-100
             stream, resample predictions back to the original frame (what a client that needs
             full resolution must do), then re-score with the frozen genotype-macro COCO pipeline
             (``jrtip_budget_policy.evaluate_arm``) so the numbers are comparable with the budget
             tables.  ``g=1.0`` is the control by construction.
``timing``   one session of fresh-process service requests per g over the frozen 1,073-frame
             timing stream, staged with ``jrtip_bottleneck_decomposition.staged_restore`` plus the
             client-side resample cost measured separately.  A twin ``service_g`` is gated against
             ``jrtip_pilot.service`` byte-for-byte before any timing is recorded.
``merge``    average sessions first, then compare g arms; time and bytes only.

Deliberately excluded: no new decision variable, no locked-role access, no retraining, no change
to the input scale (640 stays), and no claim that the offline cost model substitutes for measured
service time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_bottleneck_decomposition import (  # noqa: E402
    CLEAN_PROBE_RATIO, PROBE_EVERY, PROBE_MATRIX, PROBE_REPEATS, matrix_probe, staged_restore)
from src.jrtip_budget_policy import (  # noqa: E402
    AP_DROP_LIMIT_PP, COUNT_INCREASE_LIMIT_LEAF, Ledger, evaluate_arm, per_frame_consistency, regroup)
from src.jrtip_budget_rule_timing import (  # noqa: E402
    CONTROL, SCALE, compare_instances, install_signal_recorder, load_model, read_stream,
    session_order, utc)
from src.jrtip_pilot import (  # noqa: E402
    NOTE, build_jp_detector, decode, digest_json, encode_mask, file_sha256, input_bytes, jp_path,
    load_inputs, machine_snapshot, memory_snapshot, no_second_resize, resize_long, run_lock,
    seed_backend, select_dev, service, status, summarize_times, sync, versions, warmup)
from src.mva_pilot import append_jsonl, write_json  # noqa: E402

VERSION = "jrtip-mask-restore-resolution-v1"
GS = (1.0, 0.75, 0.5)
QUALITY_ROOT = ROOT / "jrtip/detection_budget_quality_v1"
CONTROL_STREAM = QUALITY_ROOT / f"cap{CONTROL:05d}" / f"quality_{CONTROL:06d}_S{SCALE}.jsonl"
STAGE_KEYS = ("decode_ms", "resize_transfer_ms", "network_sync_ms",
              "restore_scores_masks_rle_serialize_sync_ms")
SUB_STAGES = ("scores_to_cpu_ms", "mask_restore_interpolate_to_cpu_ms", "mask_threshold_ms",
              "boxes_labels_ms", "rle_encode_ms", "serialize_ms")


def tag(g: float) -> str:
    return f"{float(g):g}".replace(".", "p")


# ``meta_for_g`` appends exactly these three fields to the geometry block that ``staged_restore``
# embeds in its payload.  They are provenance, not delivered content, which is why the g = 1.0
# quality stream reproduced the archived cap-100 stream bitwise while its serialisation is longer
# than the frozen service path's.  The instrument gate below therefore demands more than the raw
# byte identity it was first written as: it strips precisely these keys, requires the remainder to
# be byte-identical, and requires the key set, the declared values and the byte delta to be the
# same on every gate frame.
PROVENANCE_KEYS = ("declared_hw", "declared_scale_xy", "restore_g")


def meta_for_g(meta, g: float) -> dict:
    """The same geometry, declared in a g-scaled delivery frame."""
    height, width = meta["original_hw"]
    scale_x, scale_y = meta["scale_xy"]
    delivered = (max(1, int(round(float(g) * int(height)))), max(1, int(round(float(g) * int(width)))))
    return dict(meta, original_hw=delivered, scale_xy=[float(scale_x) * float(g), float(scale_y) * float(g)],
                declared_hw=[int(height), int(width)], declared_scale_xy=[float(scale_x), float(scale_y)],
                restore_g=float(g))


@torch.inference_mode()
def service_g(model, blob_supplier, scale, device, g, expected_hw=None, clock=time.perf_counter,
              with_client=False):
    """Twin of jrtip_pilot.service whose restore stage honours the g-scaled delivery frame.

    Stage names match ``service`` so ``summarize_times`` aggregates both; the twin is gated against
    the frozen function byte-for-byte at g = 1.0 before any timing is trusted.
    """
    sync(device)
    start = clock()
    blob = blob_supplier()
    image = decode(blob, expected_hw)
    decoded = clock()
    scaled, meta = resize_long(image, scale)
    tensor = scaled.to(device)
    sync(device)
    resized = clock()
    with no_second_resize(model):
        output = model([tensor])[0]
    sync(device)
    network = clock()
    timings: dict[str, float] = {}
    payload, serialized = staged_restore(output, meta_for_g(meta, g), device, timings)
    sync(device)
    restored = clock()
    client_ms = None
    client_bytes = None
    if with_client and tuple(payload["original_hw"]) != tuple(meta["original_hw"]):
        payload_full = resample_payload(payload, tuple(meta["original_hw"]))
        client_bytes = len(json.dumps(payload_full, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        sync(device)
        client_ms = (clock() - restored) * 1000
    end = clock()
    stages = {"decode_ms": (decoded - start) * 1000, "resize_transfer_ms": (resized - decoded) * 1000,
              "network_sync_ms": (network - resized) * 1000,
              "restore_scores_masks_rle_serialize_sync_ms": (restored - network) * 1000}
    stages.update({f"stage_{key}": float(timings.get(key, 0.0)) for key in SUB_STAGES})
    measured = {key: stages[key] for key in STAGE_KEYS}
    stages["stage_sum_ms"] = sum(measured.values())
    stages["stage_gap_vs_service_ms"] = (end - start) * 1000 - stages["stage_sum_ms"]
    if client_ms is not None:
        stages["client_resample_to_original_ms"] = client_ms
    return serialized, {"service_ms": (end - start) * 1000, **stages,
                        "serialized_bytes": len(serialized),
                        "client_serialized_bytes": client_bytes,
                        "delivered_hw": list(payload["original_hw"]),
                        "declared_hw": list(meta["original_hw"]),
                        "geometry": meta}


def rle_decode(rle):
    return mask_utils.decode({"size": [int(rle["size"][0]), int(rle["size"][1])],
                              "counts": rle["counts"].encode("ascii")})


def resample_instance(instance, target_hw):
    """Client-side restoration of one delivered instance to the declared full frame."""
    height, width = int(target_hw[0]), int(target_hw[1])
    rle = instance["segmentation"]
    gh, gw = int(rle["size"][0]), int(rle["size"][1])
    if (gh, gw) == (height, width):
        return instance
    binary = rle_decode(rle)
    probs = torch.from_numpy(np.ascontiguousarray(binary)).float().view(1, 1, gh, gw)
    probs = F.interpolate(probs, size=(height, width), mode="bilinear", align_corners=False)
    mask = (probs[0, 0] >= 0.5).numpy().astype(np.uint8)
    factor_x, factor_y = width / gw, height / gh
    box = [instance["box_xyxy"][0] * factor_x, instance["box_xyxy"][1] * factor_y,
           instance["box_xyxy"][2] * factor_x, instance["box_xyxy"][3] * factor_y]
    box = [min(max(value, 0.0), float(width if index % 2 == 0 else height))
           for index, value in enumerate(box)]
    return {"score": instance["score"], "box_xyxy": box, "segmentation": encode_mask(mask)}


def resample_payload(payload, target_hw):
    out = dict(payload)
    out["instances"] = [resample_instance(instance, target_hw) for instance in payload["instances"]]
    out["original_hw"] = [int(target_hw[0]), int(target_hw[1])]
    return out


# ---------------------------------------------------------------- quality (GPU, one pass) ------
def stream_paths(out: Path, gs):
    return {g: out / f"quality_g{tag(g)}_S{SCALE}.jsonl" for g in gs}


def done_ids(paths):
    """Frames present in every g stream, so a resume never leaves a partial multi-arm frame."""
    sets = [({record["image_id"] for record in read_stream(path)} if path.exists() else set())
            for path in paths.values()]
    if not sets:
        return set()
    return set.intersection(*sets) if all(sets) else set()


def quality_pass(args):
    out = jp_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gs = [float(value) for value in args.gs.split(",") if value.strip()]
    paths = stream_paths(out, gs)
    benchmark = Path(args.benchmark).resolve()
    manifest = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    device = torch.device(args.device)
    seed_backend(config["seed"], device)
    torch.set_num_threads(args.threads)
    rows, data, _ = load_inputs(config["inventory"], config["instances"], config["readiness"],
                                config["data_root"])
    checkpoint_path = Path(manifest["checkpoint"])
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen benchmark checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_jp_detector(seed=config["seed"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    if int(model.roi_heads.detections_per_img) != CONTROL:
        raise ValueError("This pass prices the control budget only; detections_per_img must be 100")
    selected = select_dev(rows, args.max_frames)
    if {row["split_role"] for row in selected} != {"V_SHIFT", "V_SOURCE"}:
        raise ValueError("Both valid development domains required")
    already = done_ids(paths)
    write_json(out / "restore_config.json", {
        "version": VERSION, "created_utc": utc(), "benchmark": str(benchmark),
        "benchmark_config_sha256": digest_json(manifest), "checkpoint_sha256": checkpoint_sha,
        "scale": SCALE, "control_budget": CONTROL, "g_values": gs, "device": args.device,
        "threads": args.threads, "precision": "FP32", "selected_population": len(selected),
        "selected_stream_sha256": digest_json([row["image_id"] for row in selected]),
        "control_stream": str(CONTROL_STREAM),
        "control_stream_sha256": file_sha256(CONTROL_STREAM) if CONTROL_STREAM.exists() else None,
        "resumes_completed_frames": len(already), "versions": versions(), "note": NOTE})
    install_signal_recorder(out)
    started = time.perf_counter()
    written = 0
    with run_lock(out):
        model.eval()
        for index, row in enumerate(selected):
            if row["image_id"] in already:
                continue
            image = decode(input_bytes(row), (row["height"], row["width"]))
            scaled, meta = resize_long(image, SCALE)
            with no_second_resize(model):
                output = model([scaled.to(device)])[0]
            target = data.label(row)
            gt_count = len(target["masks"])
            for g in gs:
                timings: dict[str, float] = {}
                payload, serialized = staged_restore(output, meta_for_g(meta, g), device, timings)
                instances = payload["instances"]
                count = int(payload["count"])
                append_jsonl(paths[g], {
                    "image_id": row["image_id"], "split_role": row["split_role"],
                    "genotype": row["genotype"], "plant": row["biological_plant_id"],
                    "ordinal": index, "scale": SCALE, "restore_g": float(g),
                    "original_hw": list(meta["original_hw"]),
                    "delivered_hw": list(payload["original_hw"]),
                    "count": count, "gt_count": int(gt_count),
                    "count_abs_error": abs(count - int(gt_count)),
                    "ap_score_threshold": payload["ap_score_threshold"],
                    "count_score_threshold": payload["count_score_threshold"],
                    "mask_threshold": payload["mask_threshold"],
                    "serialized_bytes": len(serialized), "instances": instances})
            written += 1
            if index % 100 == 0:
                status(out, "mask_restore_quality", index, frames=len(selected), written=written,
                       g_values=gs)
                print(f"quality pass {index}/{len(selected)} frames, {written} written, "
                      f"{time.perf_counter() - started:.0f} s elapsed", flush=True)
    record = {"version": VERSION, "generated_utc": utc(), "frames_requested": len(selected),
              "frames_skipped_as_complete": len(already), "frames_written": written,
              "elapsed_seconds": round(time.perf_counter() - started, 1),
              "streams": {tag(g): {"path": str(paths[g]),
                                   "lines": sum(1 for _ in paths[g].open("rb")),
                                   "sha256": file_sha256(paths[g])} for g in gs}}
    write_json(out / "quality_pass.json", record)
    print("wrote", out / "quality_pass.json", flush=True)
    return 0


# ---------------------------------------------------------------- rescore (CPU) ------------------
def rescore(args):
    out = jp_path(args.out)
    ledger = Ledger()
    gs = [float(value) for value in args.gs.split(",") if value.strip()]
    paths = stream_paths(out, gs)
    for g in gs:
        if not paths[g].exists():
            raise FileNotFoundError(f"missing stream for g={g}: {paths[g]}")
    benchmark = Path(args.benchmark).resolve()
    manifest = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    rows, data, _ = load_inputs(config["inventory"], config["instances"], config["readiness"],
                                config["data_root"])
    stream = select_dev(rows, args.max_frames)
    rows_by_image_id = {row["image_id"]: row for row in stream}
    base = [record for record in read_stream(paths[1.0] if 1.0 in paths else paths[gs[0]])
            if record["image_id"] in rows_by_image_id]
    if not base:
        raise RuntimeError("no frames to rescore")

    # ---- gate: the g=1.0 delivery must be the archived cap-100 stream, byte for byte -----------
    archived = {record["image_id"]: record for record in read_stream(CONTROL_STREAM)}
    mismatches = {"length": 0, "score_or_box": 0, "mask": 0, "missing_reference": 0}
    compared = 0
    for record in base:
        reference = archived.get(record["image_id"])
        if reference is None:
            mismatches["missing_reference"] += 1
            continue
        compared += 1
        length_ok, score_box_bad, mask_bad = compare_instances(record["instances"], reference["instances"])
        mismatches["length"] += int(not length_ok)
        mismatches["score_or_box"] += score_box_bad
        mismatches["mask"] += mask_bad
    ledger.note("g1_stream_frames_have_archived_reference", mismatches["missing_reference"] == 0,
                f"{mismatches['missing_reference']} of {len(base)} frames unmatched")
    ledger.note("g1_stream_lengths_match_archived_cap100", mismatches["length"] == 0,
                f"{mismatches['length']} of {compared} frames")
    ledger.note("g1_stream_scores_boxes_bitwise_equal", mismatches["score_or_box"] == 0,
                f"{mismatches['score_or_box']} instances")
    ledger.note("g1_stream_mask_rle_bitwise_equal", mismatches["mask"] == 0,
                f"{mismatches['mask']} of {sum(len(r['instances']) for r in base)} instances")
    per_frame_consistency(base, CONTROL, ledger)

    control_report = json.loads((QUALITY_ROOT / f"cap{CONTROL:05d}" /
                                 f"budget_report_{CONTROL:06d}.json").read_text(encoding="utf-8"))
    control_masks = float(np.mean([len(record["instances"]) for record in base]))
    blocks = regroup(base, rows_by_image_id, ledger, CONTROL, stream)
    control_arm = evaluate_arm("g1", "control", lambda record: 10 ** 6, blocks, data,
                              rows_by_image_id, ledger,
                              {"macro": control_report["macro"], "macro_by_role": None,
                               "per_genotype": control_report["per_genotype"],
                               "mean_instances_masked_per_frame": control_masks})
    worst = max(abs(float(control_report["macro"][key]) - float(control_arm["macro"][key]))
                for key in ("AP", "AP50", "AP75", "count_MAE"))
    ledger.note("g1_reproduces_frozen_control_metrics", worst <= args.reproduce_tol,
                f"max |diff| {worst:.3e} pp")
    control = {"macro": control_report["macro"], "macro_by_role": None,
               "per_genotype": control_report["per_genotype"],
               "mean_instances_masked_per_frame": control_masks}

    # ---- arms: resample to the declared frame, then re-score with the frozen pipeline -----------
    g1_counts = {record["image_id"]: int(record["count"]) for record in base}
    results = {"g1": dict(control_arm, restore_g=1.0, resampled_client_side=False,
                          delivered_hw_equals_original=True,
                          mean_serialized_bytes_per_frame=round(float(np.mean(
                              [record["serialized_bytes"] for record in base])), 1),
                          mean_mask_pixels_per_frame=round(float(np.mean([
                              int(np.prod(record["delivered_hw"])) * len(record["instances"])
                              for record in base])), 1))}
    for g in gs:
        if g == 1.0:
            continue
        raw = [record for record in read_stream(paths[g]) if record["image_id"] in rows_by_image_id]
        if len(raw) != len(base):
            ledger.note(f"stream_g{tag(g)}_same_frame_set", False,
                        f"{len(raw)} vs {len(base)} frames")
        resampled, delivered_same = [], 0
        for record in raw:
            entry = dict(record)
            same = list(record["delivered_hw"]) == list(record["original_hw"])
            delivered_same += int(same)
            if not same:
                entry = resample_payload(entry, tuple(record["original_hw"]))
            resampled.append(entry)
        g_blocks = regroup(resampled, rows_by_image_id, ledger, CONTROL, stream)
        arm = evaluate_arm(f"g{tag(g)}", "mask_restore_resolution", lambda record: 10 ** 6,
                           g_blocks, data, rows_by_image_id, ledger, control)
        arm.update({"restore_g": float(g), "resampled_client_side": delivered_same < len(resampled),
                    "frames_delivered_at_full_resolution": delivered_same,
                    "frames_where_reported_count_differs_from_g1": int(sum(
                        1 for record in raw if g1_counts.get(record["image_id"]) != int(record["count"]))),
                    "mean_serialized_bytes_per_frame": round(float(np.mean(
                        [record["serialized_bytes"] for record in raw])), 1),
                    "mean_mask_pixels_per_frame": round(float(np.mean([
                        int(np.prod(record["delivered_hw"])) * len(record["instances"]) for record in raw])), 1)})
        results[f"g{tag(g)}"] = arm
        print(f"g {g}: AP {arm['macro']['AP']:.4f} ({-arm['ap_drop_vs_control_pp']:+.4f} pp) "
              f"count {-arm['count_MAE_increase_vs_control_leaf']:+.4f} leaf "
              f"bytes/frame {arm['mean_serialized_bytes_per_frame']:.0f}", flush=True)

    for name, arm in results.items():
        if name == "g1":
            continue
        ledger.note(f"{name}_ap_drop_within_1pp",
                    float(arm["ap_drop_vs_control_pp"]) <= AP_DROP_LIMIT_PP,
                    f"{arm['ap_drop_vs_control_pp']} pp")
        ledger.note(f"{name}_count_MAE_within_0_2_leaf",
                    float(arm["count_MAE_increase_vs_control_leaf"]) <= COUNT_INCREASE_LIMIT_LEAF,
                    f"{arm['count_MAE_increase_vs_control_leaf']} leaf")
        ledger.note(f"{name}_count_identical_to_control_by_construction",
                    int(arm["frames_where_reported_count_differs_from_g1"]) == 0,
                    f"{arm['frames_where_reported_count_differs_from_g1']} frames; g rescales the "
                    "mask frame only, scores and reported counts are untouched")

    payload = {"version": VERSION, "generated_utc": utc(), "out": str(out),
               "frames": len(base), "g_values": gs, "scale": SCALE, "control_budget": CONTROL,
               "gpu_used": False, "locked_roles_accessed": [],
               "control_stream": str(CONTROL_STREAM),
               "control_stream_sha256": file_sha256(CONTROL_STREAM),
               "streams": {tag(g): {"path": str(paths[g]), "lines": sum(1 for _ in paths[g].open("rb")),
                                    "sha256": file_sha256(paths[g])} for g in gs},
               "g1_gate_vs_archived_cap100": {"frames_compared": compared, **mismatches},
               "predeclared_tolerances": {"ap_drop_pp": AP_DROP_LIMIT_PP,
                                          "count_MAE_increase_leaf": COUNT_INCREASE_LIMIT_LEAF},
               "control": {key: control_report["macro"][key] for key in
                           ("AP", "AP50", "AP75", "count_MAE")} |
                          {"work_masks_per_frame": round(control_masks, 3),
                           "mean_serialized_bytes_per_frame":
                               results["g1"]["mean_serialized_bytes_per_frame"]},
               "arms": results, "checks": ledger.checks, "failures": ledger.failures,
               "client_resample_recipe": ("decode RLE -> float32 -> F.interpolate bilinear to the "
                                          "declared frame -> >= 0.5 -> re-encode; boxes scaled by "
                                          "(w/w_g, h/h_g) and re-clamped")}
    write_json(out / "restore_quality.json", payload)
    print("wrote", out / "restore_quality.json", flush=True)
    print("failures:", ledger.failures or "none", flush=True)
    return 0 if not ledger.failures else 2


# ---------------------------------------------------------------- timing (GPU) -------------------
def without_provenance(serialized: bytes, meta, g: float, expected_keys):
    """Re-serialise ``service_g`` output with the three provenance fields removed.

    Returns ``(bytes, notes)``.  The round-trip is byte-stable because the frozen path serialises with
    the same ``json.dumps(..., separators=(",", ":"))`` call on the same float objects, Python dicts
    preserve insertion order, and ``allow_nan=False`` rules out the values whose repr is not
    round-trip safe.  If that stability ever breaks, the comparison below fails loudly instead of
    drifting.  ``expected_keys`` is the frozen payload's geometry key order, which is what makes a
    fourth, undeclared field a named defect rather than just a byte mismatch.
    """
    payload = json.loads(serialized)
    geometry = payload["geometry"]
    notes = []
    if geometry.get("restore_g") != float(g):
        notes.append(f"restore_g={geometry.get('restore_g')!r}!={g!r}")
    if [int(v) for v in geometry.get("declared_hw", [])] != [int(v) for v in meta["original_hw"]]:
        notes.append(f"declared_hw={geometry.get('declared_hw')}!={meta['original_hw']}")
    if [float(v) for v in geometry.get("declared_scale_xy", [])] != \
            [float(v) for v in meta["scale_xy"]]:
        notes.append(f"declared_scale_xy={geometry.get('declared_scale_xy')}!={meta['scale_xy']}")
    kept = {key: value for key, value in geometry.items() if key not in PROVENANCE_KEYS}
    if sorted(kept) != sorted(expected_keys):
        notes.append(f"geometry key set differs: extra={sorted(set(kept) - set(expected_keys))} "
                     f"missing={sorted(set(expected_keys) - set(kept))}")
    payload["geometry"] = kept
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"), notes


def instrument_gate(model, blobs, rows, device, out, frames, args):
    """At g=1.0 the timed twin must deliver the frozen service path's payload, nothing else.

    The first version of this check compared raw bytes and so could never pass: ``meta_for_g`` adds
    three provenance fields to the embedded geometry block, which makes the twin's serialisation a
    constant ~100 bytes longer on every frame.  The replacement is stronger where it matters and
    honest about the one thing it no longer claims - it requires the delivered content to be identical
    byte-for-byte, pins the exact key set and values that may differ, and requires the byte delta to
    be the same on all gate frames (a frame-dependent delta would mean the twin's serialise stage is
    doing different amounts of work per frame, which would contaminate the timing).
    """
    checks = []
    warmed = warmup(model, blobs, rows, SCALE, device, min(args.warmup, 20))
    content_bad = byte_bad = 0
    notes, deltas, compared = [], set(), 0
    for row in rows[:frames]:
        reference, _ = service(model, lambda r=row: blobs[r["image_id"]], SCALE, device,
                               (row["height"], row["width"]))
        mine, record = service_g(model, lambda r=row: blobs[r["image_id"]], SCALE, device, 1.0,
                                 (row["height"], row["width"]))
        compared += 1
        deltas.add(len(mine) - len(reference))
        reference_payload, mine_payload = json.loads(reference), json.loads(mine)
        if mine_payload["instances"] != reference_payload["instances"]:
            content_bad += 1
        stripped, problem = without_provenance(mine, record["geometry"], 1.0,
                                               list(reference_payload["geometry"]))
        if problem:
            notes.append(f"{row['image_id']}: {'; '.join(problem)}")
        if stripped != reference:
            byte_bad += 1
        if byte_bad + content_bad == 1:
            print(f"instrument gate first failure at {row['image_id']}: {len(mine)} vs "
                  f"{len(reference)} bytes, stripped match {stripped == reference}, content match "
                  f"{content_bad == 0}", flush=True)
    checks.append({"name": "service_g_g1_instances_equal_frozen_service", "pass": content_bad == 0,
                   "detail": f"{content_bad} of {compared} frames differ in delivered instances"})
    checks.append({"name": "service_g_g1_payload_equal_frozen_service_after_provenance_strip",
                   "pass": byte_bad == 0 and not notes,
                   "detail": f"{byte_bad} of {compared} frames differ after stripping "
                             f"{list(PROVENANCE_KEYS)}; provenance defects: {notes[:3] or 'none'}"})
    checks.append({"name": "service_g_g1_provenance_byte_delta_is_constant",
                   "pass": len(deltas) == 1,
                   "detail": f"deltas {sorted(deltas)} bytes, expected one constant value"})
    all_pass = all(entry["pass"] for entry in checks)
    write_json(out / "instrument_gate.json", {
        "version": VERSION, "generated_utc": utc(), "frames": compared, "all_pass": all_pass,
        "checks": checks,
        "provenance_keys": list(PROVENANCE_KEYS),
        "provenance_byte_delta": sorted(deltas),
        "warmup": {"requested": warmed["requested"], "actual": warmed["actual"],
                   "stable": warmed["stable_recent_three_groups20"]},
        "meaning": ("the timed instrument is the frozen service path at g=1.0 plus three declared "
                    "provenance fields and nothing else; any later g arm therefore differs from the "
                    "control only by the declared delivery frame"),
        "history": ("written 2026-11-22 22:35: the original raw-byte gate failed 40/40 frames with a "
                    "constant +100 byte delta, which is what exposed the provenance fields; the "
                    "revision keeps the content requirement and pins the delta instead of pretending "
                    "the twin writes identical bytes")})
    return all_pass


def g_stats(name, records, loop_seconds, warmup_block, before, after):
    summary = summarize_times(records)
    # Counter keys are the formatted "HxW" strings themselves; unpacking them as (height, width)
    # pairs is what crashed session 1 of 2026-11-23 00:15 at the summary step (the requests were
    # already on disk in arm_g1.0.jsonl, so only the aggregation had to be re-run).
    delivered = Counter(f"{int(r['delivered_hw'][0])}x{int(r['delivered_hw'][1])}" for r in records)
    stages = {key: round(float(np.mean([r[key] for r in records])), 4) for key in STAGE_KEYS}
    sub = {key: round(float(np.mean([r.get(f"stage_{key}", 0.0) for r in records])), 4)
           for key in SUB_STAGES}
    client = [r["client_resample_to_original_ms"] for r in records
              if r.get("client_resample_to_original_ms") is not None]
    probes = [r["probe_matrix_ms"] for r in records if r.get("probe_matrix_ms")]
    inflation = np.asarray(probes, dtype=np.float64) / min(probes) if probes else None
    contentions = [snapshot for snapshot in (before, after)
                   if snapshot.get("nvidia") and snapshot["nvidia"].get("contention")]
    return {"arm": name, "summary": summary, "stage_means_ms": stages, "restore_substage_means_ms": sub,
            "serialized_bytes_mean": round(float(np.mean([r["serialized_bytes"] for r in records])), 1),
            "delivered_hw_hist": {key: count for key, count in sorted(delivered.items())
                                  if count >= 5}
            | {"distinct_shapes": len({tuple(r["delivered_hw"]) for r in records})},
            "client_resample": ({"requests": len(client), "mean_ms": round(float(np.mean(client)), 4),
                                 "p95_ms": round(float(np.percentile(client, 95)), 4),
                                 "bytes_after_resample_mean": round(float(np.mean(
                                     [r["client_serialized_bytes"] for r in records
                                      if r.get("client_serialized_bytes") is not None])), 1)}
                                if client else {"requests": 0}),
            "probes": ({"probes": len(probes), "matrix_ms_min": round(float(min(probes)), 4),
                        "matrix_ms_median": round(float(np.median(probes)), 4),
                        "matrix_ms_max": round(float(max(probes)), 4),
                        "inflation_median_vs_fastest": round(float(np.median(inflation)), 4),
                        "inflation_max_vs_fastest": round(float(inflation.max()), 4),
                        "probes_within_clean_ratio": int((inflation <= CLEAN_PROBE_RATIO).sum()),
                        "clean_probe_ratio_threshold": CLEAN_PROBE_RATIO,
                        "probe": f"{PROBE_MATRIX}^3 matmul x{PROBE_REPEATS} every {PROBE_EVERY} requests"}
                       if probes else {"probes": 0}),
            "warmup": {"requested": warmup_block["requested"], "actual": warmup_block["actual"],
                       "jitter": warmup_block["jitter"],
                       "stable": warmup_block["stable_recent_three_groups20"]},
            "contention": bool(contentions), "exclusive_platform_guaranteed": False,
            "loop_seconds": loop_seconds}


def timing_session(args):
    out = jp_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gs = [float(value) for value in args.gs.split(",") if value.strip()]
    names = [f"g{tag(g)}" for g in gs]
    benchmark = Path(args.benchmark).resolve()
    config_digest = digest_json({"version": VERSION, "arms": names, "benchmark": str(benchmark),
                                 "warmup": args.warmup, "scale": SCALE, "device": args.device,
                                 "threads": args.threads, "client": not args.no_client})
    marker = out / "session_config.json"
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded["config_digest"] != config_digest and not args.allow_config_change:
            raise ValueError("resume config differs from the recorded session config; pass "
                             "--allow-config-change only if that is intended")
    else:
        write_json(marker, {"version": VERSION, "config_digest": config_digest, "arms": names,
                            "created_utc": utc()})
    manifest, model, device, rows, blobs = load_model(benchmark, args.device, args.threads)
    if not args.skip_instrument_gate:
        if not instrument_gate(model, blobs, rows, device, out, args.instrument_gate_frames, args):
            print("INSTRUMENT GATE FAILED - timing not started", flush=True)
            return 2
    install_signal_recorder(out)
    per_arm = {}
    order = session_order(args.session, names)
    with run_lock(out):
        for position, name in enumerate(order):
            g = next(value for value in gs if f"g{tag(value)}" == name)
            log = out / f"arm_{name}.jsonl"
            records = list(read_stream(log)) if log.exists() else []
            if len(records) >= len(rows):
                per_arm[name] = g_stats(name, records, None, {"requested": args.warmup, "actual": 0,
                                                              "jitter": False,
                                                              "stable_recent_three_groups20": False},
                                        {"nvidia": None}, {"nvidia": None})
                per_arm[name]["resumed_without_new_requests"] = True
                write_json(out / f"arm_{name}.session{args.session}.json", per_arm[name])
                print(f"arm {name}: complete ({len(records)} rows), re-summarised only", flush=True)
                continue
            if records:
                log.unlink()
            status(out, "mask_restore_timing", args.session, arm=name, position=position,
                   order=order, frames=len(rows))
            block = {"warmup": warmup(model, blobs, rows, SCALE, device, args.warmup)}
            before = machine_snapshot(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for index, row in enumerate(rows):
                probe = matrix_probe(device) if index % PROBE_EVERY == 0 else None
                serialized, record = service_g(model, lambda r=row: blobs[r["image_id"]], SCALE,
                                               device, g, (row["height"], row["width"]),
                                               with_client=not args.no_client)
                append_jsonl(log, {"session": args.session, "arm": name, "restore_g": float(g),
                                   "scale": SCALE, "ordinal": index, "image_id": row["image_id"],
                                   "plant": row["biological_plant_id"], "genotype": row["genotype"],
                                   "split_role": row["split_role"],
                                   "emitted": len(json.loads(serialized)["instances"]),
                                   **record, "probe_matrix_ms": probe,
                                   "memory": memory_snapshot(device)})
            block.update({"before": before, "after": machine_snapshot(device),
                          "loop_seconds": round(time.perf_counter() - started, 1),
                          "requests": len(rows)})
            records = list(read_stream(log))
            per_arm[name] = g_stats(name, records, block["loop_seconds"], block["warmup"],
                                    block["before"], block["after"])
            write_json(out / f"arm_{name}.session{args.session}.json", per_arm[name])
            print(f"session {args.session} arm {name:6s} mean {per_arm[name]['summary']['mean_ms']:8.2f} ms "
                  f"restore {per_arm[name]['stage_means_ms']['restore_scores_masks_rle_serialize_sync_ms']:8.2f} ms "
                  f"bytes {per_arm[name]['serialized_bytes_mean']:9.1f} inflation "
                  f"{per_arm[name]['probes'].get('inflation_median_vs_fastest')} "
                  f"in {block['loop_seconds']:.0f} s", flush=True)
        write_json(out / f"session_{args.session}.json", {
            "version": VERSION, "session": args.session, "pid": os.getpid(), "fresh_process": True,
            "arm_order": order, "generated_utc": utc(), "benchmark": str(benchmark),
            "benchmark_config_sha256": digest_json(manifest), "frames_per_arm": len(rows),
            "device": args.device, "threads": args.threads, "precision": "FP32", "scale": SCALE,
            "warmup_requested": args.warmup, "timing_stream_sha256": manifest["timing_stream_sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"], "versions": versions(),
            "client_resample_included": not args.no_client,
            "instrument_gate": (json.loads((out / "instrument_gate.json").read_text(encoding="utf-8"))
                                ["all_pass"] if (out / "instrument_gate.json").exists() else None),
            "per_arm": per_arm, "note": NOTE})
    print("wrote", out / f"session_{args.session}.json", flush=True)
    return 0


def merge_sessions(args):
    sessions = [Path(item.strip()) for item in args.merge.split(",") if item.strip()]
    per_arm: dict[str, list[dict]] = {}
    for directory in sessions:
        # A session dir carries exactly one report, session_<N>.json, plus a session_config.json
        # resume marker that is not a report.  Deriving N from the directory rather than from
        # --session is what lets one merge command join sessions 1, 2 and 3.
        reports = [path for path in sorted(directory.glob("session_*.json"))
                   if path.name.removeprefix("session_").removesuffix(".json").isdigit()]
        if len(reports) != 1:
            raise ValueError(f"{directory} must hold exactly one session_<N>.json report, "
                             f"found {len(reports)}: {[p.name for p in reports]}")
        payload = json.loads(reports[0].read_text(encoding="utf-8"))
        for name, block in payload["per_arm"].items():
            per_arm.setdefault(name, []).append(block)
        if not all(not block.get("resumed_without_new_requests") for block in payload["per_arm"].values()):
            print(f"warning: {directory.name} contributed re-summarised arms without new requests",
                  flush=True)
    arms = {}
    for name, blocks in per_arm.items():
        means = lambda key: round(float(np.mean([float(b["summary"][key]) for b in blocks])), 4)
        arms[name] = {"sessions": len(blocks),
                      "mean_ms": means("mean_ms"), "p50_ms": means("p50_ms"),
                      "p95_ms": means("p95_ms"), "p99_ms": means("p99_ms"),
                      "saturated_throughput_per_second": means("saturated_throughput_per_second"),
                      "equal_domain_mean_ms": means("equal_domain_mean_ms"),
                      "stage_means_ms": {key: round(float(np.mean(
                          [b["stage_means_ms"][key] for b in blocks])), 4) for key in STAGE_KEYS},
                      "restore_substage_means_ms": {key: round(float(np.mean(
                          [b["restore_substage_means_ms"][key] for b in blocks])), 4)
                          for key in SUB_STAGES},
                      "serialized_bytes_mean": round(float(np.mean(
                          [b["serialized_bytes_mean"] for b in blocks])), 1),
                      "client_resample": {"requests": int(sum(b["client_resample"]["requests"]
                                                              for b in blocks)),
                                          "mean_ms": (round(float(np.mean([b["client_resample"]["mean_ms"]
                                                                           for b in blocks
                                                                           if b["client_resample"]["requests"]])), 4)
                                                      if any(b["client_resample"]["requests"] for b in blocks)
                                                      else None)},
                      "contention_in_any_session": any(b["contention"] for b in blocks),
                      "inflation_median_vs_fastest": [b["probes"].get("inflation_median_vs_fastest")
                                                      for b in blocks]}
    baseline = arms.get("g1")
    if baseline:
        for name, entry in arms.items():
            entry["service_ms_minus_g1"] = round(entry["mean_ms"] - baseline["mean_ms"], 4)
            entry["service_reduction_vs_g1"] = round(1 - entry["mean_ms"] / baseline["mean_ms"], 4)
            entry["restore_ms_minus_g1"] = round(
                entry["stage_means_ms"]["restore_scores_masks_rle_serialize_sync_ms"]
                - baseline["stage_means_ms"]["restore_scores_masks_rle_serialize_sync_ms"], 4)
            entry["serialized_bytes_minus_g1"] = round(entry["serialized_bytes_mean"]
                                                       - baseline["serialized_bytes_mean"], 1)
            client = entry["client_resample"]["mean_ms"]
            entry["service_ms_plus_client_resample"] = (
                round(entry["mean_ms"] + float(client), 4) if client else entry["mean_ms"])
            entry["net_reduction_if_client_restores_full_res"] = round(
                1 - entry["service_ms_plus_client_resample"] / baseline["mean_ms"], 4)
    payload = {"version": VERSION, "generated_utc": utc(), "sessions": [str(p) for p in sessions],
               "protocol": ("three fresh processes over the frozen 1,073-frame timing stream, rotated "
                            "arm order, warmup, matmul interference probe, per-arm resume"),
               "model_scale_input": SCALE, "control_budget": CONTROL,
               "arms": arms, "note": NOTE}
    out = jp_path(args.out)
    write_json(out / "restore_timing.json", payload)
    for name in sorted(arms):
        entry = arms[name]
        print(f"{name:6s} service {entry['mean_ms']:8.2f} ms  restore "
              f"{entry['stage_means_ms']['restore_scores_masks_rle_serialize_sync_ms']:8.2f} ms  "
              f"bytes {entry['serialized_bytes_mean']:9.1f}  client "
              f"{entry['client_resample']['mean_ms']}  net vs g1 "
              f"{entry.get('net_reduction_if_client_restores_full_res')}", flush=True)
    print("wrote", out / "restore_timing.json", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("quality", "rescore", "timing", "merge"))
    parser.add_argument("--benchmark", default=str(ROOT / "jrtip/baseline_benchmark_v1"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/mask_restore_resolution_v1"))
    parser.add_argument("--gs", default=",".join(f"{g:g}" for g in GS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--session", type=int, default=1, help="1..3, fresh process per session")
    parser.add_argument("--merge", default="", help="comma list of session dirs (mode=merge)")
    parser.add_argument("--reproduce-tol", dest="reproduce_tol", type=float, default=1e-6)
    parser.add_argument("--instrument-gate-frames", dest="instrument_gate_frames", type=int, default=40,
                        help="frames compared byte-for-byte against jrtip_pilot.service at g=1.0")
    parser.add_argument("--skip-instrument-gate", dest="skip_instrument_gate", action="store_true")
    parser.add_argument("--no-client", dest="no_client", action="store_true",
                        help="do not time the client-side resample inside the service request")
    parser.add_argument("--allow-config-change", dest="allow_config_change", action="store_true")
    args = parser.parse_args()
    if args.mode == "quality":
        return quality_pass(args)
    if args.mode == "rescore":
        return rescore(args)
    if args.mode == "merge":
        return merge_sessions(args)
    if not 1 <= args.session <= 3:
        raise ValueError("--session is 1..3 (three independent processes per plan v2 section 6)")
    return timing_session(args)


if __name__ == "__main__":
    sys.exit(main())
