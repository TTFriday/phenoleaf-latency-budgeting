"""Split the measured per-request service time into queue, compute, and output cost.

The three protocol timing sessions showed a flat service time across input scales. That
alone cannot say whether the network is resolution-insensitive or whether queueing and
scale-free output handling hide the difference, so this run records CUDA-event intervals
around the forward pass, a scheduling-latency probe, and the individual output stages.
Diagnostic decomposition only: it does not replace the protocol timing sessions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mva_model import COCO_V1_SHA256, file_sha256
from src.mva_pilot import append_jsonl, write_json
from src.jrtip_pilot import (NOTE, SCALES, build_jp_detector, decode, digest_json, encode_mask,
                             input_bytes, load_inputs, machine_snapshot, no_second_resize,
                             nvidia_snapshot, resize_long, restore_output, run_lock, seed_backend,
                             select_dev, status, sync)

VERSION = "jrtip-bottleneck-decomposition-v1"
WARMUP_CALLS = 60
PROBE_EVERY = 25
CLEAN_PROBE_RATIO = 1.15
PROBE_MATRIX = 256
PROBE_REPEATS = 20
STAGE_FIELDS = ("scores_to_cpu_ms", "mask_restore_interpolate_to_cpu_ms", "mask_threshold_ms",
                "boxes_labels_ms", "rle_encode_ms", "serialize_ms")
FRAME_FIELDS = ("decode_ms", "resize_transfer_ms", "host_to_device_event_ms", "forward_wall_ms",
                "forward_second_wall_ms", "forward_event_ms", "transform_ms", "backbone_ms",
                "rpn_ms", "roi_heads_ms", "eval_postprocess_ms", "postprocess_ms",
                "service_reconstructed_ms", "serialized_bytes", "n_instances",
                "returned_detections",
                *STAGE_FIELDS)


def pair(device):
    if device.type != "cuda":
        return None, None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def timed_forward(model, tensor, device):
    """Run the detector once, reporting both host wall time and device-clock time."""
    start, stop = pair(device)
    sync(device)
    if start is not None:
        start.record()
    wall_start = time.perf_counter()
    with no_second_resize(model):
        output = model([tensor])[0]
    sync(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000
    if start is None:
        return output, wall_ms, None
    stop.record()
    stop.synchronize()
    return output, wall_ms, float(start.elapsed_time(stop))


def launch_probe(device):
    """Empty-work round trip: the lower bound on submit-plus-complete latency right now."""
    if device.type != "cuda":
        return None
    start, stop = pair(device)
    scratch = torch.zeros(1, device=device)
    sync(device)
    start.record()
    scratch.fill_(0.0)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def matrix_probe(device):
    """Fixed GPU workload; inflation relative to the fastest window is the interference covariate."""
    if device.type != "cuda":
        return None
    start, stop = pair(device)
    left = torch.randn(PROBE_MATRIX, PROBE_MATRIX, device=device)
    right = torch.randn(PROBE_MATRIX, PROBE_MATRIX, device=device)
    result = torch.empty(PROBE_MATRIX, PROBE_MATRIX, device=device)
    sync(device)
    start.record()
    for _ in range(PROBE_REPEATS):
        torch.mm(left, right, out=result)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / PROBE_REPEATS


def cpu_probe():
    left = np.random.RandomState(0).rand(PROBE_MATRIX, PROBE_MATRIX)
    start = time.perf_counter()
    for _ in range(3):
        left = left @ left * 1e-3
    return (time.perf_counter() - start) * 1000


def staged_restore(output, meta, device, timings):
    """Same arithmetic and ordering as restore_output, with per-stage clocks."""
    height, width = meta["original_hw"]
    scaled_h, scaled_w = meta["scaled_hw"]
    mark = time.perf_counter()
    scores = output["scores"].float().cpu()
    sync(device)
    timings["scores_to_cpu_ms"] = (time.perf_counter() - mark) * 1000
    mark = time.perf_counter()
    probs = output["masks"].float()
    if probs.shape[-2:] != (scaled_h, scaled_w):
        raise RuntimeError("Model output not at scaled RGB size: implicit resize/padding "
                           "contract broken")
    if len(probs):
        probs = F.interpolate(probs, (height, width), mode="bilinear", align_corners=False)[:, 0].cpu()
    else:
        probs = torch.empty((0, height, width), dtype=torch.float32)
    sync(device)
    timings["mask_restore_interpolate_to_cpu_ms"] = (time.perf_counter() - mark) * 1000
    mark = time.perf_counter()
    masks = (probs >= .5).numpy()
    timings["mask_threshold_ms"] = (time.perf_counter() - mark) * 1000
    mark = time.perf_counter()
    boxes = output["boxes"].float().cpu() / torch.tensor(meta["scale_xy"] * 2)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    labels = output.get("labels", torch.ones(len(scores), dtype=torch.int64)).cpu()
    timings["boxes_labels_ms"] = (time.perf_counter() - mark) * 1000
    mark = time.perf_counter()
    instances = []
    for i in torch.where((scores >= .001) & (labels == 1))[0].tolist():
        instances.append({"score": float(scores[i]), "box_xyxy": boxes[i].tolist(),
                          "segmentation": encode_mask(masks[i])})
    timings["rle_encode_ms"] = (time.perf_counter() - mark) * 1000
    payload = {"original_hw": [height, width], "geometry": meta, "instances": instances,
               "count": sum(x["score"] >= .5 for x in instances),
               "ap_score_threshold": .001, "count_score_threshold": .5, "mask_threshold": .5}
    mark = time.perf_counter()
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    timings["serialize_ms"] = (time.perf_counter() - mark) * 1000
    return payload, serialized


def attach_callable_span(target, attr, name, device, spans):
    """Time a plain callable attribute (not a Module) on the device clock."""
    original = getattr(target, attr)
    pairs = []

    def wrapper(*args, **kwargs):
        if device.type != "cuda":
            return original(*args, **kwargs)
        start, stop = pair(device)
        start.record()
        result = original(*args, **kwargs)
        stop.record()
        pairs.append((start, stop))
        return result

    setattr(target, attr, wrapper)
    spans[name] = pairs
    return original


def attach_span_events(model, device):
    """Time the detector's three stages on the device clock without changing execution."""
    spans = {}
    handles = []
    targets = {"transform": model.transform, "backbone": model.backbone, "rpn": model.rpn,
               "roi_heads": model.roi_heads}
    for name, module in targets.items():
        pairs = []

        def pre(_module, _input, pairs=pairs):
            if device.type == "cuda":
                start, stop = pair(device)
                start.record()
                pairs.append((start, stop))

        def post(_module, _input, _output, name=name, pairs=pairs):
            if device.type == "cuda" and pairs:
                start, stop = pairs[-1]
                stop.record()

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
        spans[name] = pairs
    attach_callable_span(model.transform, "postprocess", "eval_postprocess", device, spans)
    return spans, handles


def read_spans(spans, device):
    result = {}
    for name, pairs in spans.items():
        result[f"{name}_ms"] = float(pairs[-1][0].elapsed_time(pairs[-1][1])) if pairs else None
        pairs.clear()
    return result


def fixed_shape_floor(model, device, reps=20, side=256):
    """Resolution-independent floor: the same network on one constant small tensor."""
    tensor = torch.zeros(3, side, side, device=device)
    for _ in range(5):
        with no_second_resize(model):
            model([tensor])
    sync(device)
    start, stop = pair(device)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(reps):
        with no_second_resize(model):
            model([tensor])
    stop.record()
    sync(device)
    return {"side": side, "reps": reps,
            "event_ms_per_forward": float(start.elapsed_time(stop)) / reps,
            "wall_ms_per_forward": (time.perf_counter() - wall_start) * 1000 / reps}


def instrumented_request(model, blob, row, scale, device, spans=None, second_pass=True):
    record = {"image_id": row["image_id"], "split_role": row["split_role"],
              "genotype": row["genotype"], "plant": row["biological_plant_id"]}
    sync(device)
    mark = time.perf_counter()
    image = decode(blob, (row["height"], row["width"]))
    record["decode_ms"] = (time.perf_counter() - mark) * 1000
    mark = time.perf_counter()
    scaled, meta = resize_long(image, scale)
    start, stop = pair(device)
    if start is not None:
        start.record()
    tensor = scaled.to(device)
    if start is not None:
        stop.record()
        stop.synchronize()
        record["host_to_device_event_ms"] = float(start.elapsed_time(stop))
    sync(device)
    record["resize_transfer_ms"] = (time.perf_counter() - mark) * 1000
    if spans:
        for pairs in spans.values():
            pairs.clear()
    output, first_wall_ms, first_event_ms = timed_forward(model, tensor, device)
    if spans:
        record.update(read_spans(spans, device))
    record["forward_wall_ms"] = first_wall_ms
    record["forward_event_ms"] = first_event_ms
    if second_pass:
        _repeat, second_wall_ms, second_event_ms = timed_forward(model, tensor, device)
        record["forward_second_wall_ms"] = second_wall_ms
        record["forward_second_event_ms"] = second_event_ms
    timings = {}
    payload, serialized = staged_restore(output, meta, device, timings)
    sync(device)
    record.update(timings)
    record["postprocess_ms"] = sum(timings.values())
    record["service_reconstructed_ms"] = (record["decode_ms"] + record["resize_transfer_ms"]
                                         + first_wall_ms + record["postprocess_ms"])
    record["returned_detections"] = int(output["scores"].numel())
    record["n_instances"] = len(payload["instances"])
    record["count"] = int(payload["count"])
    record["serialized_bytes"] = len(serialized)
    record["scaled_pixels"] = int(np.prod(meta["scaled_hw"]))
    return record, payload


def detection_cap_sweep(model, blobs, selected, scale, device, spans, caps, frames):
    """How much service time does the returned-detection budget actually buy back?"""
    roi_heads = model.roi_heads
    original = roi_heads.detections_per_img
    observations = {}
    for cap in caps:
        roi_heads.detections_per_img = cap
        rows = []
        for row in selected[:frames]:
            record, payload = instrumented_request(model, blobs[row["image_id"]], row, scale,
                                                   device, spans, second_pass=False)
            record["detections_per_img"] = cap
            record["count"] = int(payload["count"])
            rows.append(record)
        observations[str(cap)] = rows
        roi_heads.detections_per_img = original
    reference = {r["image_id"]: r for r in observations[str(caps[0])]}
    report = {}
    for cap, rows in observations.items():
        common = [r for r in rows if r["image_id"] in reference]
        report[cap] = {
            "frames": len(rows),
            "forward_event_ms": quantiles([r["forward_event_ms"] for r in rows]),
            "eval_postprocess_ms": quantiles([r["eval_postprocess_ms"] for r in rows]),
            "postprocess_ms": quantiles([r["postprocess_ms"] for r in rows]),
            "service_reconstructed_ms": quantiles([r["service_reconstructed_ms"] for r in rows]),
            "returned_detections": quantiles([r["returned_detections"] for r in rows]),
            "count_equal_to_reference": int(sum(r["count"] == reference[r["image_id"]]["count"]
                                                for r in common)),
            "instances_equal_to_reference": int(sum(r["n_instances"] == reference[r["image_id"]]
                                                    ["n_instances"] for r in common)),
            "compared_frames": len(common)}
    return {"reference_budget": caps[0], "original_budget": original, "by_budget": report}


def payloads_match(first, second):
    """Tolerant equivalence for the uninstrumented reference on the same frame."""
    if len(first["instances"]) != len(second["instances"]):
        return False
    for a, b in zip(first["instances"], second["instances"]):
        if abs(a["score"] - b["score"]) > 1e-6 or a["segmentation"]["counts"] != b["segmentation"]["counts"]:
            return False
        if max(abs(x - y) for x, y in zip(a["box_xyxy"], b["box_xyxy"])) > 1e-4:
            return False
    return first["count"] == second["count"]


def quantiles(values):
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {"n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "p05": float(np.percentile(array, 5)), "p95": float(np.percentile(array, 95)),
            "min": float(array.min()), "max": float(array.max())}


def pick(summary, field):
    clean = summary["clean_window"][field]
    return clean if clean.get("n") else summary["all"][field]


def med(summary, field):
    return pick(summary, field).get("median")


def aggregate(frames, probes, scale):
    """Split the stream into interference-qualified frames and summarize both views.

    Semantics kept as measured for bottleneck_decomposition_v1: a frame qualifies when its
    OWN ordinal is a probe ordinal whose fixed-work inflation is within the limit, so the
    clean subset is the set of probed boundary frames (2/16/12 of 358 at the three scales),
    not every frame inside a qualified interval (32/358/276).  Both counts are recomputed and
    reported by src/verify_jrtip_decomposition.py; the "all" column, which covers all 358
    frames per scale, is the robust one and is what conclusions should quote.  Changing this
    to interval semantics would invalidate that verifier's membership checks, so it has been
    left alone deliberately and the difference is documented instead.
    """
    clean = {p["ordinal"] for p in probes if p["matrix_inflation"] <= CLEAN_PROBE_RATIO}
    ordinal_of = {r["image_id"]: r["ordinal"] for r in frames}
    kept = [r for r in frames if ordinal_of[r["image_id"]] in clean]
    summary = {"scale": scale, "frames": len(frames), "clean_probe_frames": len(kept),
               "clean_probe_fraction": round(len(kept) / max(len(frames), 1), 6),
               "all": {f: quantiles([r.get(f) for r in frames]) for f in FRAME_FIELDS},
               "clean_window": {f: quantiles([r.get(f) for r in kept]) for f in FRAME_FIELDS},
               "probe": {key: quantiles([p[key] for p in probes])
                         for key in ("launch_ms", "matrix_ms", "matrix_inflation", "cpu_ms",
                                     "foreign_contexts")}}
    gaps = [r["forward_wall_ms"] - r["forward_event_ms"] for r in kept]
    summary["wall_minus_event_median_ms"] = float(np.median(gaps)) if gaps else None
    return summary, kept


def instance_bins(frames):
    bins = {}
    for row in frames:
        n = row["n_instances"]
        key = "0-2" if n <= 2 else "3-4" if n <= 4 else "5-6" if n <= 6 else "7-9" if n <= 9 else "10+"
        bins.setdefault(key, []).append(row)
    order = ["0-2", "3-4", "5-6", "7-9", "10+"]
    return {key: {"frames": len(bins[key]),
                  "mask_restore_ms": quantiles([r["mask_restore_interpolate_to_cpu_ms"] for r in bins[key]]),
                  "rle_encode_ms": quantiles([r["rle_encode_ms"] for r in bins[key]]),
                  "serialize_ms": quantiles([r["serialize_ms"] for r in bins[key]]),
                  "postprocess_ms": quantiles([r["postprocess_ms"] for r in bins[key]]),
                  "serialized_bytes": quantiles([r["serialized_bytes"] for r in bins[key]])}
            for key in order if key in bins}


def resolution_response(summaries):
    """Is the device-clock forward time sensitive to input pixels at all?"""
    medians = {int(scale): med(summary, "forward_event_ms") for scale, summary in summaries.items()}
    reference = min(medians)
    return {"available": True, "median_forward_event_ms": {str(k): v for k, v in medians.items()},
            "reference_scale": reference,
            "ratio_largest_over_smallest": max(medians.values()) / min(medians.values()),
            "forward_ms_change_reference_to_640": medians[640] - medians[reference],
            "forward_ms_per_megapixel": {str(s): (medians[s] - medians[reference])
                                         / ((s ** 2 - reference ** 2) / 1e6)
                                         for s in medians if s != reference}}


def run(args):
    source = Path(args.benchmark).resolve()
    manifest = json.loads((source / "benchmark_config.json").read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Decomposition output must start empty: {out}")
    config = manifest["training_config"]
    device = torch.device(args.device)
    seed_backend(config["seed"], device)
    rows, _, _ = load_inputs(config["inventory"], config["instances"], config["readiness"],
                             config["data_root"])
    selected = select_dev(rows, manifest["timing_max_frames_per_plant"])[::args.stride]
    if args.max_frames:
        selected = selected[:args.max_frames]
    path = Path(manifest["checkpoint"])
    checkpoint_sha = file_sha256(path)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen benchmark checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_jp_detector(seed=config["seed"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    blobs = {row["image_id"]: input_bytes(row) for row in selected}
    spans, handles = attach_span_events(model, device)
    write_json(out / "decomposition_config.json", {
        "version": VERSION, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_benchmark": str(source), "source_benchmark_config_sha256": digest_json(manifest),
        "checkpoint": str(path), "checkpoint_sha256": checkpoint_sha,
        "coco_weights_sha256_constant": COCO_V1_SHA256,
        "training_config_weight_sha256": config.get("weight_sha256"),
        "device": args.device, "precision": "FP32", "frames_per_scale": len(selected),
        "stride": args.stride, "warmup_calls": args.warmup_calls, "probe_every": PROBE_EVERY,
        "clean_probe_ratio": CLEAN_PROBE_RATIO, "scales": list(SCALES),
        "detection_budget_sweep": [int(x) for x in args.cap_sweep.split(",") if x.strip()],
        "detection_budget_frames": args.cap_frames,
        "smoke_limited": bool(args.stride != 3 or args.warmup_calls != WARMUP_CALLS or args.max_frames),
        "timing_stream_sha256_of_full_selection": manifest["timing_stream_sha256"],
        "subsampled_stream_sha256": digest_json([r["image_id"] for r in selected]),
        "diagnostic_only": True, "protocol_timing_replacement": False,
        "formal_36_epoch_run": False, "locked_roles_accessed": [],
        "note": "Two forwards per frame (first reported as the service forward, second as a "
                "device-clock repeat), a fixed call-count warmup, and a stride subsample of the "
                "protocol timing stream. Therefore not comparable as a timing session; the "
                "frozen checkpoint, role selection, and staging order are identical. " + NOTE})
    summaries, fidelity = {}, []
    with run_lock(out):
        for scale in SCALES:
            status(out, "decomposition", 0, scale=scale)
            for row in selected[:args.warmup_calls]:
                instrumented_request(model, blobs[row["image_id"]], row, scale, device, spans)
            floors = {str(side): fixed_shape_floor(model, device, side=side) for side in (64, 256)}
            probes, frames, stability = [], [], []
            fastest_matrix = math.inf
            for index, row in enumerate(selected):
                if index % PROBE_EVERY == 0 or index == len(selected) - 1:
                    matrix = matrix_probe(device)
                    fastest_matrix = min(fastest_matrix, matrix)
                    snapshot = nvidia_snapshot()
                    probes.append({"ordinal": index, "launch_ms": launch_probe(device),
                                   "matrix_ms": matrix, "cpu_ms": cpu_probe(),
                                   "matrix_inflation": matrix / fastest_matrix,
                                   "foreign_contexts": len(snapshot.get("other_compute_processes") or []),
                                   "telemetry": snapshot.get("telemetry", {}).get("stdout", "").strip()})
                    inflation = probes[-1]["matrix_inflation"]
                else:
                    inflation = probes[-1]["matrix_inflation"]
                record, payload = instrumented_request(model, blobs[row["image_id"]], row, scale,
                                                       device, spans)
                record["ordinal"] = index
                record["probe_matrix_inflation"] = inflation
                if index < args.fidelity:
                    image = decode(blobs[row["image_id"]], (row["height"], row["width"]))
                    scaled, meta = resize_long(image, scale)
                    with no_second_resize(model):
                        reference_output = model([scaled.to(device)])[0]
                    sync(device)
                    reference = restore_output(reference_output, meta)
                    fidelity.append({"scale": scale, "ordinal": index, "image_id": row["image_id"],
                                     "tolerant_match": payloads_match(payload, reference),
                                     "exact_digest_match": digest_json(payload) == digest_json(reference)})
                    if not payloads_match(payload, reference):
                        raise RuntimeError(f"Instrumented output differs from restore_output: "
                                           f"S{scale} frame {row['image_id']}")
                stability.append(record["service_reconstructed_ms"])
                frames.append(record)
                append_jsonl(out / f"frames_S{scale}.jsonl", record)
            window = np.asarray(stability[-60:], dtype=np.float64)
            if window.size >= 60:
                medians = [float(np.median(window[j:j + 20])) for j in range(0, 60, 20)]
                spread = (max(medians) - min(medians)) / float(np.mean(medians))
                stability_report = {"recent_three_groups20_relative_spread": spread,
                                    "within_five_percent": bool(spread <= .05)}
            else:
                stability_report = {"available": False, "frames": int(window.size)}
            summary, kept = aggregate(frames, probes, scale)
            summary["post_run_stability"] = stability_report
            summary["output_bins_by_instances"] = instance_bins(kept)
            summary["fixed_shape_floor"] = floors
            caps = [int(x) for x in args.cap_sweep.split(",") if x.strip()]
            if caps:
                summary["detection_cap_sweep"] = detection_cap_sweep(
                    model, blobs, selected, scale, device, spans, caps, args.cap_frames)
                for key, value in summary["detection_cap_sweep"]["by_budget"].items():
                    print(f"   detections {key}: service "
                          f"{value['service_reconstructed_ms']['median']:.1f} ms, forward "
                          f"{value['forward_event_ms']['median']:.1f} ms, count equal to "
                          f"{summary['detection_cap_sweep']['reference_budget']} on "
                          f"{value['count_equal_to_reference']}/{value['compared_frames']} frames",
                          flush=True)
            summary["machine_end"] = machine_snapshot(device)
            summaries[str(scale)] = summary
            print(f"S{scale}: clean window {summary['clean_probe_fraction']:.2%}, "
                  f"forward event {med(summary, 'forward_event_ms'):.1f} ms, "
                  f"wall {med(summary, 'forward_wall_ms'):.1f} ms, "
                  f"postprocess {med(summary, 'postprocess_ms'):.1f} ms | "
                  + " ".join(f"{name} {med(summary, name + '_ms'):.1f}"
                             for name in ("transform", "backbone", "rpn", "roi_heads",
                                          "eval_postprocess"))
                  + f" | floor64 {floors['64']['event_ms_per_forward']:.1f} "
                  f"floor256 {floors['256']['event_ms_per_forward']:.1f}", flush=True)
    service = {int(s): med(v, "service_reconstructed_ms") for s, v in summaries.items()}
    postprocess = {int(s): med(v, "postprocess_ms") for s, v in summaries.items()}
    response = resolution_response(summaries)
    queue_gap = {int(s): v["wall_minus_event_median_ms"] for s, v in summaries.items()}
    result = {"version": VERSION, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "frames_per_scale": {s: v["frames"] for s, v in summaries.items()},
              "fidelity_checks": {"frames": len(fidelity),
                                  "exact_digest_match": sum(f["exact_digest_match"] for f in fidelity),
                                  "tolerant_match": sum(f["tolerant_match"] for f in fidelity),
                                  "records": fidelity[:20]},
              "scales": summaries, "resolution_response": response,
              "verdict": {
                  "service_time_flat_across_scales":
                      bool(max(service.values()) - min(service.values())
                           <= 0.02 * min(service.values())),
                  "forward_device_time_scales_with_resolution":
                      bool(response["ratio_largest_over_smallest"] >= 1.20),
                  "output_side_cost_scale_independent":
                      bool(max(postprocess.values()) - min(postprocess.values())
                           <= 0.05 * min(postprocess.values())),
                  "queue_or_sync_above_quarter_of_forward":
                      bool(all(gap is not None and gap > 0.25 * med(summaries[str(s)],
                                                                    "forward_event_ms")
                               for s, gap in queue_gap.items()))},
              "scope_limits": [
                  "Development roles only; C_POLICY/T_ID/T_GENO/T_EXT untouched.",
                  "Single seed 43 pilot checkpoint at 1000 optimizer updates.",
                  "Shared desktop GPU: the fixed-work probe quantifies interference, it does not "
                  "remove it; the clean-window column keeps only frames whose own ordinal was "
                  "probed with inflation <=1.15, so it is a small boundary sample and the "
                  "all-frame column is the one to quote (see aggregate() and the verifier).",
                  "Subsampled and dual-forward instrumented run; the protocol timing sessions "
                  "remain the record for service-time claims.",
                  "The first forward of each pair feeds the reported output; the second is a "
                  "device-clock repeat and may differ slightly in cache state."]}
    write_json(out / "decomposition.json", result)
    print("wrote", out / "decomposition.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=str(ROOT / "jrtip/baseline_benchmark_v1"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/bottleneck_decomposition_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--warmup-calls", dest="warmup_calls", type=int, default=WARMUP_CALLS)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0,
                        help="cap the stream for a smoke check; recorded as smoke_limited")
    parser.add_argument("--cap-sweep", dest="cap_sweep", default="100,20,5",
                        help="roi_heads.detections_per_img values to time, empty to skip")
    parser.add_argument("--cap-frames", dest="cap_frames", type=int, default=40)
    parser.add_argument("--fidelity", type=int, default=25,
                        help="frames per scale compared against the uninstrumented restore_output")
    args = parser.parse_args()
    if not 0 < args.stride <= 10 or not 0 <= args.fidelity <= 100:
        raise ValueError("stride must be 1..10 and fidelity 0..100")
    if not 0 <= args.warmup_calls <= 200 or args.warmup_calls % 20:
        raise ValueError("warmup calls must be 0..200 and a multiple of 20")
    if not 0 <= args.max_frames <= 2000:
        raise ValueError("max frames must be 0..2000")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
