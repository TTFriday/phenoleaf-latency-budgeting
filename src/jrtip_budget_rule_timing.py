"""Measure the deployment-side service time of per-frame output budgets (JRTIP v2 P1c timing side).

Why this file exists.  The offline prefix pricing (src/jrtip_budget_policy.py, results in
jrtip/budget_policy_v1/) can price the *quality* of every "640 input, output-budget-only" arm from
one archived cap-100 pass, but it cannot price their *time*: plan v2 section 6 forbids substituting
an offline slope for a measured service time, and section 3.2 forbids pairing the time of the
synchronising implementation with the quality of the bucketed one.  This script measures time only.
Quality rows for the same arm names come from src/jrtip_budget_bucket_pricing.py, which imports the
arm definitions from here, so an arm name means one policy in both tables.

What is measured.  A budget arm is one of:

* ``native``      - the untouched torchvision path with ``roi_heads.detections_per_img = k``.
                    Static arms; the implementation a deployer gets for free today.
* ``wrap_const``  - the wrapper of this file degenerated to a constant k.  It prices the wrapper
                    itself: every rule arm below runs on top of this code path, so any statement
                    about controller overhead has to be measured against it.
* ``sync``        - rule arm, variant (i) of v2 section 3.2: one blocking device-to-host read of the
                    post-NMS count of score >= 0.5, decide k on the host, return k instances, and
                    only then let the mask head run.  Performance upper bound of adaptive budgeting,
                    including the synchronisation cost.
* ``readfull``    - same decision, but the whole score vector is transferred instead of one scalar.
                    Prices what a controller that needs richer candidate features pays at the same
                    point, which section 6 asks to report separately as the decision overhead.
* ``bucket``      - rule arm, variant (ii): the decided k is ceiled to BUCKET_GRID before the mask
                    head, so mask work happens on a shape a batch could be pooled by.  At batch 1
                    this cannot remove the device-to-host read - the read is what selects the bucket
                    - and the artifact records that instead of claiming a sync-free bucket grid.

Correctness gates, both required before any timing row is trusted:

* Gate A (v2 section 3.2, literal): with the rule degenerated to a constant budget, the per-frame
  output stream must equal the archived stream of a real run at that budget - score, box and mask
  RLE compared at byte level against jrtip/detection_budget_quality_v1/cap00100 and cap00020.
* Gate B (v2 section 4.1 extended to variable k): a wrapped variable-k arm must equal the matching
  *prefix* of the archived cap-100 stream, frame by frame.  Length, score and box must be byte-exact;
  mask strings are reported, not gated, because the known batch-dependent +-1 pixel rasterisation
  deviation (result doc section 9) is exactly what a changed mask-head batch size produces.

Protocol.  Frozen benchmark timing stream (manifest ``timing_max_frames_per_plant``, long edge 640,
FP32, batch 1, threads 4), one fresh process per session (``--session``), warm-up before every arm,
arm order rotated by session, the fixed compute probe of the bottleneck decomposition inserted on
the same cadence, and machine/nvidia snapshots before and after every arm.  Development roles only:
C_POLICY/T_ID/T_GENO/T_EXT are never loaded here.

Not measured here, and deliberately so: no queueing replay, no energy, no second device, no locked
test roles, no 36-epoch checkpoint.  These are pilot service times on a desktop GPU that carries ~34
third-party CUDA contexts, which is why probe inflation is recorded per arm and per session.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_bottleneck_decomposition import (  # noqa: E402
    CLEAN_PROBE_RATIO, PROBE_EVERY, PROBE_MATRIX, PROBE_REPEATS, matrix_probe)
from src.jrtip_pilot import (  # noqa: E402
    NOTE, build_jp_detector, child_manifest, digest_json, file_sha256, input_bytes, jp_path,
    load_inputs, machine_snapshot, memory_snapshot, run_lock, seed_backend, select_dev, service,
    status, summarize_times, versions, warmup)
from src.mva_pilot import append_jsonl, write_json  # noqa: E402

VERSION = "jrtip-budget-rule-timing-v2"
SCALE = 640
CONTROL = 100
COUNT_SCORE = 0.5
AP_SCORE = 0.001
BUCKET_GRID = (15, 20, 30, 50, 100)
QUALITY_ROOT = ROOT / "jrtip/detection_budget_quality_v1"
ARCHIVED = {budget: QUALITY_ROOT / f"cap{budget:05d}" / f"quality_{budget:06d}_S{SCALE}.jsonl"
            for budget in (100, 20, 15)}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rule_k(rule, c: int, floor: int, cap: int = CONTROL) -> int:
    """The two label-free families of plan v2 section 3.1, in host arithmetic on the read count."""
    kind, parameter = rule
    raw = c + int(parameter) if kind == "c_plus_m" else math.ceil(float(parameter) * int(c))
    return int(min(cap, max(floor, raw)))


def bucket_of(k: int, grid=BUCKET_GRID) -> int:
    return next((bucket for bucket in grid if bucket >= k), grid[-1])


def budget_for(arm, c):
    """The single definition of "what does this arm emit", shared by timing and offline pricing."""
    mode = arm["mode"]
    if mode in ("native", "wrap_const"):
        return int(arm["k"])
    k = rule_k(arm["rule"], c, int(arm["floor"]))
    return bucket_of(k) if mode == "bucket" else k


ARMS: dict[str, dict] = {
    "static100_native": {"mode": "native", "k": CONTROL, "family": "static",
                         "definition": "detections_per_img=100, unmodified torchvision path"},
    "static20_native": {"mode": "native", "k": 20, "family": "static",
                        "definition": "detections_per_img=20"},
    "static16_native": {"mode": "native", "k": 16, "family": "static",
                        "definition": "detections_per_img=16 (frozen static challenger)"},
    "static15_native": {"mode": "native", "k": 15, "family": "static",
                        "definition": "detections_per_img=15"},
    "static100_wrap": {"mode": "wrap_const", "k": CONTROL, "family": "wrapper_control",
                       "definition": "wrapper, rule degenerated to constant 100 (gate A arm)"},
    "static20_wrap": {"mode": "wrap_const", "k": 20, "family": "wrapper_control",
                      "definition": "wrapper, constant 20: prices wrapper overhead at small batch"},
    "sync_c_plus_10_f15": {"mode": "sync", "rule": ("c_plus_m", 10), "floor": 15,
                           "family": "rule_sync",
                           "definition": "k=clip(c+10,15,100), one blocking scalar D2H read"},
    "sync_ceil_1p5_f15": {"mode": "sync", "rule": ("ceil_mul", 1.5), "floor": 15,
                          "family": "rule_sync",
                          "definition": "k=clip(ceil(1.5*c),15,100), one blocking scalar D2H read"},
    "readfull_c_plus_10_f15": {"mode": "readfull", "rule": ("c_plus_m", 10), "floor": 15,
                               "family": "rule_readfull",
                               "definition": "as sync_c_plus_10_f15 but the whole score vector is "
                                             "transferred; prices candidate-feature controllers"},
    "bucket_c_plus_10_f15": {"mode": "bucket", "rule": ("c_plus_m", 10), "floor": 15,
                             "family": "rule_bucket",
                             "definition": "k=clip(c+10,15,100) ceiled to {15,20,30,50,100}"},
    "bucket_ceil_1p5_f15": {"mode": "bucket", "rule": ("ceil_mul", 1.5), "floor": 15,
                            "family": "rule_bucket",
                            "definition": "k=clip(ceil(1.5*c),15,100) ceiled to {15,20,30,50,100}"},
}
ARM_ORDER = list(ARMS)
VARIABLE_ARMS = [name for name, arm in ARMS.items() if arm["family"].startswith("rule_")]
CONST_ARMS = [name for name, arm in ARMS.items() if arm["mode"] in ("native", "wrap_const")]


class BudgetWrapper:
    """Instance-level wrap of RoIHeads.postprocess_detections that inserts the decision point.

    The wrapper calls the original method with the model default budget, so the per-image
    boxes/scores/labels it gets back are what an uncapped box head emits (truncated at the default
    100 candidates), decides k from scores still on device, and slices the three parallel lists.
    The mask head runs later, in ``RoIHeads.forward``, on ``[p["boxes"] for p in result]``, so this
    slice is applied before any mask work - which is the entire point of the arm and is what gate A
    verifies against a real ``keep[:k]`` run.

    Truncating the candidate list at 100 is decision-irrelevant for these arms: every rule here is
    monotone in c and saturates at 100, so a frame whose true post-NMS count exceeds 100 receives
    k=100 either way.  ``candidates_after_nms`` records the saturation so the claim stays checkable,
    and the capping caveat of v2 section 4.2 stays documented rather than being silently fixed here.
    """

    def __init__(self, roi_heads, arm, clock=time.perf_counter):
        self.roi = roi_heads
        self.arm = arm
        self.clock = clock
        self.original = roi_heads.postprocess_detections
        self.default_budget = int(roi_heads.detections_per_img)
        if self.default_budget != CONTROL:
            raise ValueError(f"wrapper assumes model default budget {CONTROL}, found {self.default_budget}")
        self.last: dict = {}
        self.installed = False

    def install(self):
        self.roi.postprocess_detections = self
        self.installed = True

    def remove(self):
        if self.installed:
            del self.roi.__dict__["postprocess_detections"]
            self.installed = False

    def __call__(self, class_logits, box_regression, proposals, image_shapes):
        arm = self.arm
        entered = self.clock()
        self.roi.detections_per_img = CONTROL
        try:
            boxes, scores, labels = self.original(class_logits, box_regression, proposals, image_shapes)
        finally:
            self.roi.detections_per_img = self.default_budget
        gathered = self.clock()
        out_boxes, out_scores, out_labels, trace = [], [], [], []
        for box_i, score_i, label_i in zip(boxes, scores, labels):
            candidates = int(score_i.numel())
            if arm["mode"] == "wrap_const":
                c, k, read_ms, decide_ms = None, int(arm["k"]), 0.0, 0.0
            elif arm["mode"] == "readfull":
                read_started = self.clock()
                host_scores = score_i.detach().float().cpu().numpy()
                read_ms = (self.clock() - read_started) * 1000
                decide_started = self.clock()
                c = int((host_scores >= COUNT_SCORE).sum())
                k = budget_for(arm, c)
                decide_ms = (self.clock() - decide_started) * 1000
            else:
                read_started = self.clock()
                c = int((score_i >= COUNT_SCORE).sum().item())
                read_ms = (self.clock() - read_started) * 1000
                decide_started = self.clock()
                k = budget_for(arm, c)
                decide_ms = (self.clock() - decide_started) * 1000
            out_boxes.append(box_i[:k])
            out_scores.append(score_i[:k])
            out_labels.append(label_i[:k])
            trace.append({"mode": arm["mode"], "c_ge_0_5": c, "k_decided": k,
                          "candidates_after_nms": candidates, "read_ms": round(read_ms, 5),
                          "decide_ms": round(decide_ms, 5)})
        left = self.clock()
        self.last = dict(trace[0], wrapper_total_ms=round((left - entered) * 1000, 5),
                         images=len(trace), box_head_ms=round((gathered - entered) * 1000, 5))
        return out_boxes, out_scores, out_labels


def install_arm(model, arm):
    """Return (wrapper, teardown): set the budget the way this arm's implementation would."""
    roi = model.roi_heads
    native_before = int(roi.detections_per_img)
    if arm["mode"] == "native":
        roi.detections_per_img = int(arm["k"])
        return None, lambda: setattr(roi, "detections_per_img", native_before)
    wrapper = BudgetWrapper(roi, arm)
    wrapper.install()
    return wrapper, wrapper.remove


def decision_fields(wrapper, arm, emitted):
    """Per-request decision accounting; ``emitted`` is the instance count actually returned."""
    if wrapper is None:
        base = {"mode": "native", "c_ge_0_5": None, "k_decided": int(arm["k"]),
                "candidates_after_nms": None, "read_ms": 0.0, "decide_ms": 0.0,
                "wrapper_total_ms": 0.0, "box_head_ms": None}
    else:
        base = {key: wrapper.last.get(key) for key in
                ("mode", "c_ge_0_5", "k_decided", "candidates_after_nms", "read_ms", "decide_ms",
                 "wrapper_total_ms", "box_head_ms")}
    base["emitted"] = int(emitted)
    return base


def payload_counts(serialized):
    """Instances and reported leaves of one service response, from the bytes the service returns."""
    payload = json.loads(serialized)
    instances = payload["instances"]
    below_ap = sum(1 for entry in instances if float(entry["score"]) < AP_SCORE)
    return {"emitted": len(instances), "count_reported": int(payload["count"]),
            "instances_below_ap_threshold": below_ap}


def timed_requests(model, name, arm, blobs, rows, device, log, session, args, wrapper):
    """One arm x one session over the frozen timing stream, with probes and per-frame records."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = machine_snapshot(device)
    warmed = warmup(model, blobs, rows, SCALE, device, args.warmup)
    started = time.perf_counter()
    count = 0
    for index, row in enumerate(rows):
        probe = matrix_probe(device) if index % PROBE_EVERY == 0 else None
        serialized, record = service(model, lambda r=row: blobs[r["image_id"]], SCALE, device,
                                     (row["height"], row["width"]))
        counts = payload_counts(serialized)
        entry = {"session": session, "arm": name, "scale": SCALE, "ordinal": index,
                 "image_id": row["image_id"], "plant": row["biological_plant_id"],
                 "genotype": row["genotype"], "split_role": row["split_role"],
                 **record, **decision_fields(wrapper, arm, counts["emitted"]),
                 "count_reported": counts["count_reported"],
                 "instances_below_ap_threshold": counts["instances_below_ap_threshold"],
                 "probe_matrix_ms": probe, "memory": memory_snapshot(device)}
        append_jsonl(log, entry)
        count += 1
    return {"warmup": warmed, "before": before, "after": machine_snapshot(device),
            "loop_seconds": round(time.perf_counter() - started, 1), "requests": count}


def arm_stats(name, records, loop_seconds, warmup_block, before, after):
    summary = summarize_times(records)
    emitted = np.asarray([r["emitted"] for r in records], dtype=np.float64)
    c_values = np.asarray([r["c_ge_0_5"] for r in records if r["c_ge_0_5"] is not None], dtype=np.float64)
    reads = np.asarray([r["read_ms"] for r in records], dtype=np.float64)
    wraps = np.asarray([r["wrapper_total_ms"] or 0.0 for r in records], dtype=np.float64)
    probes = [r["probe_matrix_ms"] for r in records if r.get("probe_matrix_ms")]
    inflation = np.asarray(probes, dtype=np.float64) / min(probes) if probes else None
    decision = {"frames": len(records), "emitted_instances_mean": round(float(emitted.mean()), 4),
                "emitted_instances_p50": float(np.median(emitted)),
                "emitted_instances_max": int(emitted.max()),
                "emitted_by_value": {str(int(value)): int((emitted == value).sum())
                                     for value in sorted(set(int(x) for x in emitted))},
                "k_decided_mean": round(float(np.mean([r["k_decided"] for r in records])), 4),
                "frames_where_emitted_lt_decided": int(sum(1 for r in records
                                                           if r["emitted"] < r["k_decided"])),
                "read_ms_mean": round(float(reads.mean()), 5),
                "read_ms_p95": round(float(np.percentile(reads, 95)), 5),
                "read_ms_max": round(float(reads.max()), 5),
                "wrapper_total_ms_mean": round(float(wraps.mean()), 5),
                "decide_ms_mean": round(float(np.mean([r["decide_ms"] for r in records])), 5),
                "reported_count_mean": round(float(np.mean([r["count_reported"] for r in records])), 4)}
    if c_values.size:
        decision.update({"c_mean": round(float(c_values.mean()), 4),
                         "c_p50": float(np.median(c_values)),
                         "c_p90": float(np.percentile(c_values, 90)),
                         "c_max": int(c_values.max()),
                         "frames_where_candidates_saturated_at_100":
                             int(sum(1 for r in records
                                     if (r["candidates_after_nms"] or 0) >= CONTROL))})
    contentions = [snapshot for snapshot in (before, after)
                   if snapshot.get("nvidia") and snapshot["nvidia"].get("contention")]
    return {"arm": name, "definition": ARMS[name]["definition"], "family": ARMS[name]["family"],
            "summary": summary, "decision": decision,
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
                       "jitter": warmup_block["jitter"], "stable": warmup_block["stable_recent_three_groups20"]},
            "contention": bool(contentions),
            "contention_assessment_unknown": any(snapshot.get("nvidia") and
                                                 not snapshot["nvidia"].get("compute_query_ok")
                                                 for snapshot in (before, after)),
            "exclusive_platform_guaranteed": False, "loop_seconds": loop_seconds}


def read_stream(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_archived(budgets):
    """Read only the archived GPU streams the requested gate arms actually reference.

    cap 15 was priced offline from the cap-100 prefix and has no archived GPU pass, so asking for it
    unconditionally would turn a bookkeeping choice into a gate failure.
    """
    streams, ledger = {}, []
    for budget in sorted(set(budgets)):
        path = ARCHIVED[budget]
        if not path.exists():
            ledger.append({"name": f"archived_stream_present_{budget}", "pass": False,
                           "detail": str(path)})
            continue
        streams[budget] = {record["image_id"]: record for record in read_stream(path)}
        ledger.append({"name": f"archived_stream_present_{budget}", "pass": True,
                       "detail": f"{len(streams[budget])} frames {path.name}"})
    return streams, ledger


def compare_instances(mine, theirs):
    """Byte-level comparison of two instance lists; returns (length_ok, score_box_bad, mask_bad)."""
    if len(mine) != len(theirs):
        return False, 0, 0
    score_box_bad = mask_bad = 0
    for a, b in zip(mine, theirs):
        if json.dumps([round(float(a["score"]), 9), [round(float(x), 6) for x in a["box_xyxy"]]]) \
                != json.dumps([round(float(b["score"]), 9), [round(float(x), 6) for x in b["box_xyxy"]]]):
            score_box_bad += 1
        if json.dumps(a["segmentation"], sort_keys=True) != json.dumps(b["segmentation"], sort_keys=True):
            mask_bad += 1
    return True, score_box_bad, mask_bad


def gate(args, model, blobs, rows, device, out):
    """Gate A (constant k vs a real cap-k pass) and gate B (variable k vs the cap-100 prefix)."""
    checks = []

    def note(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    requested = [entry.strip() for entry in args.gate_arms.split(",") if entry.strip()]
    unknown = [name for name in requested if name not in ARMS]
    if unknown:
        raise ValueError(f"unknown gate arms: {unknown}")
    needed = [int(ARMS[name]["k"]) if ARMS[name]["mode"] in ("native", "wrap_const") else CONTROL
              for name in requested]
    archived, ledger = load_archived(needed)
    checks.extend(ledger)
    if CONTROL not in archived:
        write_json(out / "identity_gate.json", {"checks": checks, "all_pass": False,
                                                "generated_utc": utc(), "version": VERSION})
        return False
    subset = rows[:max(1, args.gate_frames)]
    per_arm = {}
    for name in requested:
        arm = ARMS[name]
        constant = arm["mode"] in ("native", "wrap_const")
        budget = int(arm["k"]) if constant else CONTROL
        if budget not in archived:
            note(f"{name}_archived_reference_available", False, f"budget {budget} not archived")
            continue
        log = out / f"gate_{name}.jsonl"
        if log.exists():
            log.unlink()
        wrapper, teardown = install_arm(model, arm)
        produced, decisions = {}, {}
        try:
            for row in subset:
                serialized, record = service(model, lambda r=row: blobs[r["image_id"]], SCALE,
                                             device, (row["height"], row["width"]))
                counts = payload_counts(serialized)
                produced[row["image_id"]] = json.loads(serialized)
                decisions[row["image_id"]] = decision_fields(wrapper, arm, counts["emitted"])
                append_jsonl(log, dict(record, image_id=row["image_id"], **decisions[row["image_id"]]))
        finally:
            teardown()
        write_json(out / f"gate_decisions_{name}.json", decisions)
        lengths_bad = score_box_bad = mask_bad = compared = instances = missing = 0
        for image_id, record in produced.items():
            baseline = archived[budget].get(image_id)
            if baseline is None:
                missing += 1
                continue
            mine = record["instances"]
            theirs = (baseline["instances"] if constant
                      else baseline["instances"][:decisions[image_id]["k_decided"]])
            compared += 1
            instances += len(theirs)
            length_ok, box_bad, mask_entry_bad = compare_instances(mine, theirs)
            lengths_bad += int(not length_ok)
            score_box_bad += box_bad
            mask_bad += mask_entry_bad
        entry = {"arm": name, "mode": arm["mode"],
                 "gate": "A_constant_k" if constant else "B_variable_prefix",
                 "reference_budget": budget, "frames_compared": compared,
                 "frames_missing_reference": missing, "frames_with_length_mismatch": lengths_bad,
                 "instances_compared": instances, "score_or_box_mismatches": score_box_bad,
                 "mask_string_mismatches": mask_bad,
                 "mask_mismatch_rate": round(mask_bad / max(1, instances), 6),
                 "emitted_instances_mean": round(float(np.mean([d["emitted"] for d in decisions.values()])), 4),
                 "reference_stream": "real cap-k archived pass" if constant
                                     else "cap-100 archived prefix"}
        per_arm[name] = entry
        note(f"{name}_every_frame_has_reference", missing == 0, f"{missing} frames missing")
        note(f"{name}_lengths_match_reference", lengths_bad == 0, f"{lengths_bad} of {compared} frames")
        note(f"{name}_scores_and_boxes_bitwise_equal", score_box_bad == 0, f"{score_box_bad} instances")
        if constant:
            note(f"{name}_mask_rle_bitwise_equal", mask_bad == 0, f"{mask_bad} instances")
        else:
            note(f"{name}_mask_deviations_within_known_rate", entry["mask_mismatch_rate"] <= 0.001,
                 f"{mask_bad}/{instances} instances; the batch-size dependent +-1 pixel rasterisation "
                 f"deviation is documented in the result doc section 9, gated at <=0.1% and reported "
                 f"rather than hidden")
    payload = {"version": VERSION, "generated_utc": utc(), "gate_frames": len(subset),
               "gate_arms": args.gate_arms, "checks": checks, "per_arm": per_arm,
               "meaning": {"gate_A": "wrapped constant budget == real detections_per_img=k run, "
                                     "byte level, masks included",
                           "gate_B": "wrapped variable budget == prefix of length k of the cap-100 "
                                     "stream; length/score/box byte-exact, mask strings reported"},
               "failures": [check["name"] for check in checks if not check["pass"]]}
    payload["all_pass"] = not payload["failures"]
    write_json(out / "identity_gate.json", payload)
    return payload["all_pass"]


def load_model(benchmark: Path, device_name: str, threads: int):
    """Fresh-process model load against the frozen benchmark manifest, as timing_session does."""
    manifest = child_manifest(benchmark)
    config = manifest["training_config"]
    device = torch.device(device_name)
    seed_backend(config["seed"], device)
    torch.set_num_threads(threads)
    rows, _, _ = load_inputs(config["inventory"], config["instances"], config["readiness"],
                             config["data_root"])
    selected = select_dev(rows, manifest["timing_max_frames_per_plant"])
    if digest_json([r["image_id"] for r in selected]) != manifest["timing_stream_sha256"]:
        raise ValueError("Timing stream changed")
    path = Path(manifest["checkpoint"])
    if file_sha256(path) != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen benchmark checkpoint changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_jp_detector(seed=config["seed"]).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    del state
    blobs = {row["image_id"]: input_bytes(row) for row in selected}
    return manifest, model, device, selected, blobs


def session_order(session, names):
    shift = (session - 1) % len(names)
    return names[shift:] + names[:shift]


def run_session(args):
    benchmark = Path(args.benchmark).resolve()
    out = jp_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    marker = out / "session_config.json"
    config_digest = digest_json({"version": VERSION, "arms": names, "benchmark": str(benchmark),
                                 "warmup": args.warmup, "scale": SCALE, "device": args.device,
                                 "threads": args.threads})
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded["config_digest"] != config_digest and not args.allow_config_change:
            raise ValueError("resume config differs from the recorded session config; if that is "
                             "intended (e.g. more arms), pass --allow-config-change")
    else:
        write_json(marker, {"version": VERSION, "config_digest": config_digest, "arms": names,
                            "benchmark": str(benchmark), "warmup": args.warmup, "created_utc": utc()})
    manifest, model, device, rows, blobs = load_model(benchmark, args.device, args.threads)
    if args.gate_frames:
        if not gate(args, model, blobs, rows, device, out):
            print("GATE FAILED - timing not started, see identity_gate.json", flush=True)
            return 2
        print(f"gate passed on {args.gate_frames} frames, see identity_gate.json", flush=True)
        if args.gate_only:
            return 0
    elif not args.skip_gate:
        raise ValueError("run the bitwise gate first (--gate-frames N) or pass --skip-gate for a smoke")
    install_signal_recorder(out)
    order = session_order(args.session, names)
    per_arm = {}
    with run_lock(out):
        for position, name in enumerate(order):
            arm = ARMS[name]
            log = out / f"arm_{name}.jsonl"
            records = list(read_stream(log)) if log.exists() else []
            if len(records) >= len(rows):
                print(f"arm {name}: complete ({len(records)} rows), re-summarised only", flush=True)
                per_arm[name] = arm_stats(name, records, None, {"requested": args.warmup, "actual": 0,
                                                                "jitter": False,
                                                                "stable_recent_three_groups20": False},
                                          {"nvidia": None}, {"nvidia": None})
                per_arm[name]["resumed_without_new_requests"] = True
                write_json(out / f"arm_{name}.session{args.session}.json", per_arm[name])
                continue
            if records:
                log.unlink()
            status(out, "rule_timing", session=args.session, arm=name, position=position,
                   order=order, frames=len(rows))
            wrapper, teardown = install_arm(model, arm)
            try:
                block = timed_requests(model, name, arm, blobs, rows, device, log, args.session,
                                       args, wrapper)
            finally:
                teardown()
            records = list(read_stream(log))
            per_arm[name] = arm_stats(name, records, block["loop_seconds"], block["warmup"],
                                      block["before"], block["after"])
            write_json(out / f"arm_{name}.session{args.session}.json", per_arm[name])
            summary, decision = per_arm[name]["summary"], per_arm[name]["decision"]
            print(f"session {args.session} arm {name:24s} mean {summary['mean_ms']:8.2f} ms "
                  f"p95 {summary['p95_ms']:8.2f} ms emitted {decision['emitted_instances_mean']:6.2f}"
                  f"/frame read {decision['read_ms_mean']:7.4f} ms inflation "
                  f"{per_arm[name]['probes'].get('inflation_median_vs_fastest')} "
                  f"in {block['loop_seconds']:.0f} s", flush=True)
        write_json(out / f"session_{args.session}.json", {
            "version": VERSION, "session": args.session, "pid": os.getpid(), "fresh_process": True,
            "arm_order": order, "generated_utc": utc(), "benchmark": str(benchmark),
            "benchmark_config_sha256": digest_json(manifest), "frames_per_arm": len(rows),
            "device": args.device, "threads": args.threads, "precision": "FP32", "scale": SCALE,
            "warmup_requested": args.warmup, "timing_stream_sha256": manifest["timing_stream_sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"], "versions": versions(),
            "bucket_grid": list(BUCKET_GRID),
            "gate": (json.loads((out / "identity_gate.json").read_text(encoding="utf-8"))["all_pass"]
                     if (out / "identity_gate.json").exists() else None),
            "per_arm": per_arm, "note": NOTE})
    print("wrote", out / f"session_{args.session}.json", flush=True)
    return 0


def merge(args):
    """Average sessions first (v1 section 6 discipline), then compare arms. Time only."""
    roots = [Path(item.strip()).resolve() for item in args.merge.split(",") if item.strip()]
    per_arm = defaultdict(lambda: defaultdict(list))
    families, definitions, sessions = {}, {}, []
    for root in roots:
        # session_config.json is the resume marker, not a session report; only
        # session_<digits>.json carries the arm summaries.
        reports = [path for path in sorted(root.glob("session_*.json"))
                   if path.name.removeprefix("session_").removesuffix(".json").isdigit()]
        for report in reports:
            payload = json.loads(report.read_text(encoding="utf-8"))
            sessions.append({"dir": str(root), "session": payload["session"],
                             "arm_order": payload["arm_order"], "frames_per_arm": payload["frames_per_arm"],
                             "gate": payload["gate"], "warmup_requested": payload["warmup_requested"],
                             "timing_stream_sha256": payload["timing_stream_sha256"],
                             "checkpoint_sha256": payload["checkpoint_sha256"],
                             "versions": payload["versions"]})
            for name, block in payload["per_arm"].items():
                families[name] = block["family"]
                definitions[name] = block["definition"]
                per_arm[name]["mean_ms"].append(block["summary"]["mean_ms"])
                per_arm[name]["p50_ms"].append(block["summary"]["p50_ms"])
                per_arm[name]["p95_ms"].append(block["summary"]["p95_ms"])
                per_arm[name]["equal_domain_mean_ms"].append(block["summary"]["equal_domain_mean_ms"])
                per_arm[name]["emitted"].append(block["decision"]["emitted_instances_mean"])
                per_arm[name]["read_ms"].append(block["decision"]["read_ms_mean"])
                per_arm[name]["wrapper_total_ms"].append(block["decision"]["wrapper_total_ms_mean"])
                per_arm[name]["inflation"].append(block["probes"].get("inflation_median_vs_fastest"))
                per_arm[name]["loop_seconds"].append(block["loop_seconds"])
                per_arm[name]["contention"].append(block["contention"])
    reference = None
    arms = []
    for name in sorted(per_arm):
        block = per_arm[name]
        means = block["mean_ms"]
        row = {"arm": name, "family": families[name], "definition": definitions[name],
               "sessions": len(means), "session_means_ms": [round(x, 3) for x in means],
               "mean_ms_session_averaged": round(float(np.mean(means)), 3),
               "spread_of_session_means_ms": round(float(np.max(means) - np.min(means)), 3),
               "p95_ms_session_averaged": round(float(np.mean(block["p95_ms"])), 3),
               "equal_domain_mean_ms_session_averaged": round(float(np.mean(block["equal_domain_mean_ms"])), 3),
               "emitted_instances_mean": round(float(np.mean(block["emitted"])), 4),
               "read_ms_mean": round(float(np.mean(block["read_ms"])), 5),
               "wrapper_total_ms_mean": round(float(np.mean(block["wrapper_total_ms"])), 5),
               "probe_inflation_median_by_session": block["inflation"],
               "contention_any_session": any(block["contention"]),
               "arm_pass_seconds_by_session": [round(x, 1) for x in block["loop_seconds"]]}
        arms.append(row)
        if name == "static100_native":
            reference = row["mean_ms_session_averaged"]
    for row in arms:
        row["time_reduction_vs_static100_native"] = (round(1 - row["mean_ms_session_averaged"] / reference, 4)
                                                    if reference else None)
    wrapper_overhead = {}
    for suffix, pair in (("at_100", ("static100_wrap", "static100_native")),
                         ("at_20", ("static20_wrap", "static20_native"))):
        wrapped = next((row for row in arms if row["arm"] == pair[0]), None)
        native = next((row for row in arms if row["arm"] == pair[1]), None)
        if wrapped and native:
            wrapper_overhead[f"wrapper_minus_native_{suffix}_ms"] = round(
                wrapped["mean_ms_session_averaged"] - native["mean_ms_session_averaged"], 3)
    out = jp_path(args.out)
    write_json(out / "rule_timing.json", {
        "version": VERSION, "generated_utc": utc(), "arms": arms,
        "sessions_meta": sessions,
        "identical_timing_stream_across_sessions": len({s["timing_stream_sha256"] for s in sessions}) == 1,
        "identical_checkpoint_across_sessions": len({s["checkpoint_sha256"] for s in sessions}) == 1,
        "identical_versions_across_sessions": len({digest_json(s["versions"]) for s in sessions}) == 1,
        "all_gates_passed": all(bool(s["gate"]) for s in sessions),
        "aggregation": "per-arm session means first, then averaged over sessions (v1 section 6); "
                       "individual requests are never pooled across sessions",
        "implementation_overhead_decomposition": wrapper_overhead,
        "sync_vs_bucket": sync_bucket_summary(arms),
        "quality_side": "not measured here: the AP/count rows of these arm names live in "
                        "jrtip/budget_policy_v1/policy_eval.json (constant and sync rules) and "
                        "jrtip/budget_bucket_pricing_v1/policy_eval.json (bucketed rules)",
        "not_measured": ["queueing replay", "energy", "second hardware", "locked test roles",
                         "36-epoch checkpoint", "controller (learned) arms"],
        "note": NOTE})
    print(f"wrote {out / 'rule_timing.json'} with {len(arms)} arms from {len(sessions)} sessions",
          flush=True)
    return 0


def sync_bucket_summary(arms):
    """Variant (i) vs variant (ii) of v2 section 3.2, plus the bucket waste, per rule family."""
    by_name = {row["arm"]: row for row in arms}
    rows = []
    for base in ("c_plus_10_f15", "ceil_1p5_f15"):
        sync, bucket = by_name.get(f"sync_{base}"), by_name.get(f"bucket_{base}")
        if not sync or not bucket:
            continue
        rows.append({"rule_family": base,
                     "sync_mean_ms": sync["mean_ms_session_averaged"],
                     "bucket_mean_ms": bucket["mean_ms_session_averaged"],
                     "bucket_minus_sync_ms": round(bucket["mean_ms_session_averaged"]
                                                   - sync["mean_ms_session_averaged"], 3),
                     "sync_emitted": sync["emitted_instances_mean"],
                     "bucket_emitted": bucket["emitted_instances_mean"],
                     "bucket_waste_instances": round(bucket["emitted_instances_mean"]
                                                     - sync["emitted_instances_mean"], 4),
                     "ms_per_extra_masked_instance": round(
                         (bucket["mean_ms_session_averaged"] - sync["mean_ms_session_averaged"])
                         / max(1e-9, bucket["emitted_instances_mean"] - sync["emitted_instances_mean"]), 3)
                     if bucket["emitted_instances_mean"] != sync["emitted_instances_mean"] else None})
    return rows


def install_signal_recorder(out: Path):
    """Forensic marker for an external console control event (0xC000013A killed an earlier sweep)."""
    def handler(signum, frame):
        streams = {}
        for path in sorted(out.glob("arm_*.jsonl")):
            try:
                stat = path.stat()
                streams[path.name] = {"bytes": stat.st_size, "lines": sum(1 for _ in path.open("rb")),
                                      "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                     time.gmtime(stat.st_mtime))}
            except OSError as exc:
                streams[path.name] = {"error": repr(exc)}
        write_json(out / "signal_seen.json", {"signal": int(signum),
                                              "signal_name": signal.Signals(signum).name,
                                              "recorded_utc": utc(), "pid": os.getpid(),
                                              "streams": streams,
                                              "note": "handler installed by this script; the event "
                                                      "came from outside"})
        raise KeyboardInterrupt(signal.Signals(signum).name)

    for name in ("SIGBREAK", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=str(ROOT / "jrtip/baseline_benchmark_v1"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/rule_timing_v1/session1"))
    parser.add_argument("--session", type=int, default=1, help="1..3; a session is one fresh process")
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="reuse the arm logs this attempt left behind")
    parser.add_argument("--allow-config-change", dest="allow_config_change", action="store_true",
                        help="record that the arm list changed since the session config was written")
    parser.add_argument("--gate-frames", dest="gate_frames", type=int, default=0,
                        help="run gates A+B on this many frames before timing (0 = skip)")
    parser.add_argument("--gate-only", dest="gate_only", action="store_true",
                        help="exit after the gate; used to validate the implementation cheaply")
    parser.add_argument("--gate-arms", dest="gate_arms", default=",".join(
        ["static100_wrap", "static20_wrap"] + [name for name in ARMS if ARMS[name]["family"].startswith("rule_")]))
    parser.add_argument("--skip-gate", dest="skip_gate", action="store_true",
                        help="only for a smoke run; a protocol session must pass the gate")
    parser.add_argument("--merge", default="", help="comma list of session dirs to merge instead of timing")
    args = parser.parse_args()
    if args.merge:
        return merge(args)
    if not 1 <= args.session <= 3:
        raise ValueError("--session is 1..3 (three independent processes per plan v2 section 6)")
    if not 1 <= args.threads <= 32 or args.warmup < 1:
        raise ValueError("threads must be 1..32 and warmup >= 1")
    return run_session(args)


if __name__ == "__main__":
    sys.exit(main())
