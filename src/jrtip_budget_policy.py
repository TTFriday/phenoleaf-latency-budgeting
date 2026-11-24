"""Evaluate output-side detection-budget *policies* offline, by prefix-labeling one cap-100 pass.

Why this exists.  The static sweep in ``src/jrtip_detection_budget_quality.py`` measured seven
output-side budgets (100/50/30/25/20/10/5) on the frozen development-quality contract, one full GPU
pass each - about 7 x 10 minutes.  But ``torchvision.models.detection.roi_heads.RoIHeads`` truncates
*after* NMS and *before* the mask head (``keep = keep[: self.detections_per_img]``), the surviving
proposals keep the scores and boxes they had untruncated, and the mask head treats its proposals as
independent batch elements.  So the model output at budget ``k`` should be exactly the first ``k``
instances of the model output at budget 100, and a per-frame budget policy ``k(frame)`` is therefore
a per-frame prefix length of a *single* cap-100 pass.  If that holds, the whole policy space - static
budgets, count-proportional rules, and the quality/work frontier between them - is evaluable on CPU
with no further inference, and the formal matrix can price every budget arm from one pass per epoch.

This script checks the premise instead of assuming it, then uses it:

1. **Prefix identity lemma, measured.**  For every budget with an archived GPU pass (5/10/20/25/30/50)
   each frame's instance list is compared against the corresponding prefix of the cap-100 stream:
   length, score, box, and RLE-encoded mask.  Scores and boxes must match bit-for-bit because
   truncation happens downstream of the box head; masks are the only place where a batch-size
   dependent kernel could shift a borderline pixel, so mask agreement is reported as a rate rather
   than asserted as a precondition.
2. **Offline reproduction of the measured reports.**  Each archived budget is re-scored *from the
   cap-100 stream* through the same aggregation path (``src/verify_jrtip_budget_quality.aggregate``)
   and compared with its own GPU-measured macro AP/AP50/AP75/count MAE.  This is the load-bearing
   check: if offline prefix evaluation reproduces six independent measurements, offline policy
   comparison is sound.
3. **Policy arms.**  Static budgets plus two GT-free rule families that need only the post-NMS,
   pre-truncation observable ``count`` (the number of instances scoring >= 0.5):
   ``k = clip(count + m, floor, 100)`` and ``k = clip(ceil(r * count), floor, 100)``.  Reported per
   arm: macro metrics, mean instances actually masked per frame (the cost proxy), delta versus the
   cap-100 control, the predeclared gate (AP drop <= 1.0 pp AND count MAE increase <= 0.2 leaf), the
   number of frames whose reported count was truncated, and the worst per-genotype AP drop.

The equal-work comparison in the output is what H1 is about: a policy earns its keep only if, at
matched mean masked-instance count, it loses less quality than the best static budget.  No controller
is trained or compared here; this file only establishes that the offline evaluation is trustworthy and
what the static-versus-rule frontier looks like on development data.

Development roles V_SOURCE/V_SHIFT only; the checkpoint is never loaded and the GPU is never touched,
so this can run alongside GPU work.  ``gt_count`` appears in the streams and is used only for
reporting, never to set a budget: every arm here is computable from post-NMS observables alone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_pilot import load_inputs, select_dev  # noqa: E402
from src.verify_jrtip_budget_quality import (  # noqa: E402
    CONTROL, DEV_ROLES, Ledger, METRIC_KEYS, SCALE, aggregate, per_frame_consistency,
    read_stream, regroup,
)

VERSION = "jrtip-budget-policy-v1"
MEASURED = (100, 50, 30, 25, 20, 10, 5)  # budgets with an archived GPU pass
STATIC = (100, 50, 40, 30, 25, 20, 15, 10, 5)
RULE_M_FLOOR = ((2, 15), (5, 15), (10, 15), (2, 20), (5, 20), (10, 20))
RATIO_FLOOR = ((1.25, 15), (1.5, 15), (1.25, 20), (1.5, 20))
AP_DROP_LIMIT_PP, COUNT_INCREASE_LIMIT_LEAF = 1.0, 0.2
COUNT_SCORE = 0.5


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def instances_of(record) -> list:
    return record["instances"]


def prefix(record, k: int) -> list:
    return instances_of(record)[:max(0, int(k))]


def k_static(record, cap: int) -> int:
    return int(cap)


def k_rule(record, offset: int, floor: int, cap: int = CONTROL) -> int:
    return min(cap, max(floor, int(record["count"]) + int(offset)))


def k_ratio(record, ratio: float, floor: int, cap: int = CONTROL) -> int:
    return min(cap, max(floor, int(math.ceil(float(ratio) * int(record["count"])))))


def arm_records(records, selector) -> list:
    """Apply a per-frame budget selector to the cap-100 records, re-deriving count fields."""
    out = []
    for record in records:
        kept = selector(record)
        instances = prefix(record, kept)
        count = sum(1 for instance in instances if instance["score"] >= COUNT_SCORE)
        out.append(dict(record, instances=instances, count=count,
                        count_abs_error=abs(count - int(record["gt_count"])),
                        budget=kept, returned=len(instances),
                        count_truncated=count != int(record["count"])))
    return out


def score_order_ok(records) -> int:
    """Frames whose stored instance scores are not non-increasing (prefix == top-k would break)."""
    bad = 0
    for record in records:
        scores = [instance["score"] for instance in instances_of(record)]
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            bad += 1
    return bad


def mask_equal(left, right) -> bool:
    return left == right and left["counts"] == right["counts"]


def identity_check(base, other, cap) -> dict:
    """Compare an archived cap-C stream against the matching prefix of the cap-100 stream."""
    frames = len(base)
    length_mismatch = score_mismatch = box_mismatch = mask_mismatch = 0
    mask_frames_identical = 0
    count_mismatch = field_mismatch = 0
    expected_lengths = []
    for left, right in zip(base, other):
        if left["image_id"] != right["image_id"]:
            field_mismatch += 1
            continue
        expected = prefix(left, cap)
        expected_lengths.append(len(expected))
        if len(expected) != len(instances_of(right)):
            length_mismatch += 1
            continue
        clean = True
        for a, b in zip(expected, instances_of(right)):
            if float(a["score"]) != float(b["score"]):
                score_mismatch += 1
                clean = False
            if [round(float(x), 6) for x in a["box_xyxy"]] != [round(float(x), 6) for x in b["box_xyxy"]]:
                box_mismatch += 1
                clean = False
            if not mask_equal(a["segmentation"], b["segmentation"]):
                mask_mismatch += 1
                clean = False
        mask_frames_identical += int(clean)
        if int(right["count"]) != sum(1 for instance in instances_of(right)
                                      if instance["score"] >= COUNT_SCORE):
            count_mismatch += 1
    return {"frames": frames, "budget": int(cap),
            "image_id_order_matches": field_mismatch == 0,
            "frames_with_length_mismatch": length_mismatch,
            "instances_compared": int(sum(expected_lengths)),
            "score_mismatches": score_mismatch, "box_mismatches": box_mismatch,
            "mask_mismatches": mask_mismatch,
            "frames_fully_identical": mask_frames_identical,
            "frame_identity_rate": round(mask_frames_identical / max(1, frames), 6),
            "count_field_mismatches": count_mismatch}


def evaluate_arm(name, kind, selector, blocks, data, rows_by_image_id, ledger, control):
    transformed = {}
    masks = 0
    truncated = 0
    budgets = []
    for key, records in blocks.items():
        rows = []
        for record in records:
            kept = selector(record)
            instances = prefix(record, kept)
            count = sum(1 for instance in instances if instance["score"] >= COUNT_SCORE)
            masks += len(instances)
            truncated += int(count != int(record["count"]))
            budgets.append(kept)
            rows.append(dict(record, instances=instances, count=count,
                             count_abs_error=abs(count - int(record["gt_count"])),
                             budget=kept, returned=len(instances),
                             count_truncated=count != int(record["count"])))
        transformed[key] = rows
    started = time.perf_counter()
    genotype_metrics, role_macro, equal_domain, _, gt_mismatches, _ = aggregate(
        transformed, data, rows_by_image_id, 0, ledger, False)
    if gt_mismatches:
        ledger.note(f"arm_{name}_gt_count_matches_dataset", False, f"{gt_mismatches} frames")
    frames = sum(len(v) for v in transformed.values())
    reference = control.get("per_genotype") or {}
    drops = {f"{role}/{genotype}": round(metrics["AP"] - float(reference[role][genotype]["AP"]), 4)
             for role, per_genotype in genotype_metrics.items() for genotype, metrics
             in per_genotype.items() if role in reference and genotype in reference[role]}
    worst = min(drops.items(), key=lambda item: item[1]) if drops else (None, None)
    return {
        "name": name, "kind": kind,
        "macro": equal_domain,
        "macro_by_role": role_macro,
        "ap_drop_vs_control_pp": round(control["macro"]["AP"] - equal_domain["AP"], 4),
        "count_MAE_increase_vs_control_leaf": round(equal_domain["count_MAE"]
                                                    - control["macro"]["count_MAE"], 4),
        "gate_ap_drop_within_1pp": bool(control["macro"]["AP"] - equal_domain["AP"] <= AP_DROP_LIMIT_PP),
        "gate_count_MAE_within_0_2_leaf": bool(equal_domain["count_MAE"] - control["macro"]["count_MAE"]
                                               <= COUNT_INCREASE_LIMIT_LEAF),
        "mean_instances_masked_per_frame": round(masks / frames, 3),
        "mask_work_reduction_vs_control": round(1 - (masks / frames) / control["mean_instances_masked_per_frame"], 4),
        "frames_with_reported_count_truncated": truncated,
        "budget_mean": round(float(np.mean(budgets)), 3),
        "budget_median": float(np.median(budgets)),
        "budget_p90": float(np.percentile(budgets, 90)),
        "budget_max": int(np.max(budgets)),
        "frames": frames,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "worst_genotype": {"key": worst[0], "ap_drop_vs_control_pp": worst[1]},
        "ap_drop_by_genotype": drops,
    }


def cost_model(root: Path, budgets, masks_by_budget):
    """Regress the archived whole-pass wall time on mean masked instances per frame.

    ``quality()`` elapsed time is not a deployment latency: it includes byte retrieval, decode,
    resize, the convolutional trunk, mask restore, RLE, serialisation to the per-frame stream, and
    the per-genotype COCO evaluation.  But it is measured, on the same stream, for seven budgets, so
    its dependence on the number of masked instances can be read off directly.  A near-linear fit
    turns the offline evaluation into an offline time prediction - which is what an equal-work
    comparison needs - and its slope can be cross-checked against the CUDA-event attribution
    (jrtip/bottleneck_decomposition_v1), which measured the mask post-processing per instance.
    """
    points, elapsed = [], {}
    for cap in budgets:
        report_path = root / f"cap{cap:05d}" / f"budget_report_{cap:06d}.json"
        if not report_path.exists() or cap not in masks_by_budget:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        seconds = float(report["elapsed_seconds"])
        elapsed[str(cap)] = {"lines": int(report["quality_jsonl_lines"]),
                             "elapsed_seconds": round(seconds, 1),
                             "mean_instances_masked_per_frame": masks_by_budget[cap],
                             "seconds_per_frame": round(seconds / int(report["quality_jsonl_lines"]), 4),
                             "formal_36_epoch_run": report.get("formal_36_epoch_run")}
        points.append((masks_by_budget[cap], seconds))
    if len(points) < 3:
        return {"note": "not enough completed budgets to fit", "points": elapsed}
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    ss_total = float(np.sum((y - y.mean()) ** 2))
    return {"fit": "whole-pass wall seconds = slope * mean_masked_instances_per_frame + intercept",
            "slope_seconds_per_instance_per_frame_mean": round(float(slope), 4),
            "intercept_seconds": round(float(intercept), 3),
            "r_squared": round(1 - float(np.sum(residual ** 2)) / ss_total, 6),
            "max_abs_relative_residual": round(float(np.max(np.abs(residual / y))), 5),
            "frames_per_pass": sorted({int(entry["lines"]) for entry in elapsed.values()}),
            "ms_per_extra_masked_instance_per_frame": round(float(slope) / float(np.mean(
                [entry["lines"] for entry in elapsed.values()])) * 1000, 3),
            "points": elapsed,
            "caveat": ("wall seconds include decode, trunk, restore/RLE/serialisation and COCO "
                       "evaluation; measured on a desktop GPU shared with ~34 third-party CUDA "
                       "contexts, so the intercept is machine-bound and the slope is the part the "
                       "budget controls")}


def run(args):
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger()
    strict = args.max_frames == 0  # numeric gates compare against full-stream archived reports
    started = time.perf_counter()

    first = root / "cap00100" / "quality_000100_S640.jsonl"
    if not first.exists():
        raise FileNotFoundError(f"missing control stream: {first}")
    config_path = next((root / f"cap{c:05d}" / f"budget_quality_config.json" for c in MEASURED
                        if (root / f"cap{c:05d}" / "budget_quality_config.json").exists()), None)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    benchmark = Path(config["source_benchmark"])
    manifest = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))
    training = manifest["training_config"]
    rows, data, _ = load_inputs(training["inventory"], training["instances"], training["readiness"],
                                training["data_root"])
    stream = select_dev(rows, args.max_frames)
    rows_by_image_id = {row["image_id"]: row for row in stream}
    ledger.note("declared_stream_is_all_valid_dev_frames", len(stream) == 2947 or args.max_frames > 0,
                f"{len(stream)} frames, max_frames={args.max_frames}")

    base = [record for record in read_stream(first) if record["image_id"] in rows_by_image_id]
    blocks = regroup(base, rows_by_image_id, ledger, CONTROL, stream)
    per_frame_consistency(base, CONTROL, ledger)
    ledger.note("control_stream_scores_are_non_increasing", score_order_ok(base) == 0,
                f"{score_order_ok(base)} frames violate top-k == prefix")
    base_by_id = {record["image_id"]: record for record in base}

    # ---- 1. prefix identity lemma against every archived GPU pass -------------------------------
    identities = {}
    for cap in MEASURED:
        if cap == CONTROL:
            continue
        path = root / f"cap{cap:05d}" / f"quality_{cap:06d}_S{SCALE}.jsonl"
        if not path.exists():
            continue
        other = [record for record in read_stream(path) if record["image_id"] in rows_by_image_id]
        if args.max_frames == 0 and len(other) != len(base):
            identities[str(cap)] = {"skipped": "stream incomplete on disk", "lines": len(other),
                                    "expected_lines": len(base)}
            print(f"identity cap {cap}: skipped, {len(other)}/{len(base)} frames written", flush=True)
            continue
        result = identity_check(base, other, cap)
        identities[str(cap)] = result
        ledger.note(f"identity_cap{cap}_image_id_order", result["image_id_order_matches"])
        ledger.note(f"identity_cap{cap}_lengths_match_prefix",
                    result["frames_with_length_mismatch"] == 0,
                    f"{result['frames_with_length_mismatch']} frames")
        ledger.note(f"identity_cap{cap}_scores_bitwise_match", result["score_mismatches"] == 0,
                    f"{result['score_mismatches']} of {result['instances_compared']} instances")
        ledger.note(f"identity_cap{cap}_boxes_match", result["box_mismatches"] == 0,
                    f"{result['box_mismatches']} instances")
        ledger.note(f"identity_cap{cap}_masks_match", result["mask_mismatches"] == 0,
                    f"{result['mask_mismatches']} instances, "
                    f"{result['frames_fully_identical']}/{result['frames']} identical frames")
        print(f"identity cap {cap}: {result['frames_fully_identical']}/{result['frames']} frames "
              f"identical, {result['mask_mismatches']} mask / {result['score_mismatches']} score / "
              f"{result['box_mismatches']} box mismatches", flush=True)
        del other

    # ---- 2. offline re-scoring of the measured budgets, from the control stream -----------------
    control_report = json.loads((root / "cap00100" / "budget_report_000100.json").read_text(encoding="utf-8"))
    reproduction = {}
    for cap in MEASURED:
        report_path = root / f"cap{cap:05d}" / f"budget_report_{cap:06d}.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _, _, offline, _, _, _ = aggregate(
            {key: arm_records(records, lambda record, cap=cap: cap) for key, records in blocks.items()},
            data, rows_by_image_id, cap, ledger, False)
        reproduction[str(cap)] = {
            "measured": {key: report["macro"][key] for key in METRIC_KEYS},
            "offline_prefix": {key: offline[key] for key in METRIC_KEYS},
            "max_abs_difference_pp": max(abs(report["macro"][key] - offline[key]) for key in METRIC_KEYS),
        }
        for key in METRIC_KEYS:
            # A subset smoke re-scores fewer frames than the archived full-stream reports, so the
            # numeric agreement gates only mean something when the whole dev stream is in play.
            if strict:
                ledger.note(f"reproduce_cap{cap}_{key}",
                            abs(report["macro"][key] - offline[key]) <= args.reproduce_tol,
                            f"measured {report['macro'][key]!r} vs offline {offline[key]!r}")
        print(f"offline vs measured cap {cap}: max |diff| "
              f"{reproduction[str(cap)]['max_abs_difference_pp']:.3e} pp", flush=True)

    # ---- 3. the control arm, used as the reference for every comparison -------------------------
    control_masks = float(np.mean([len(instances_of(record)) for record in base]))
    _, control_role, control_domain, _, _, _ = aggregate(
        {key: arm_records(records, lambda record: CONTROL) for key, records in blocks.items()},
        data, rows_by_image_id, CONTROL, ledger, False)
    if strict:
        ledger.note("control_arm_reproduces_control_report",
                    all(abs(control_report["macro"][key] - control_domain[key]) <= args.reproduce_tol
                        for key in METRIC_KEYS),
                    str({key: (control_report["macro"][key], control_domain[key]) for key in METRIC_KEYS}))
    control = {"macro": control_domain, "macro_by_role": control_role,
               "per_genotype": control_report["per_genotype"],
               "mean_instances_masked_per_frame": control_masks}

    # ---- 4. policy arms ------------------------------------------------------------------------
    arms = []
    statics = sorted(set(STATIC) | {int(item) for item in args.extra_static.split(",") if item.strip()},
                     reverse=True)
    plan = [("static", str(cap), (lambda record, cap=cap: k_static(record, cap))) for cap in statics]
    plan += [("rule_count_plus_m", f"clip(count+{m},{floor},100)",
              (lambda record, m=m, floor=floor: k_rule(record, m, floor)))
             for m, floor in RULE_M_FLOOR]
    plan += [("rule_count_times_r", f"clip(ceil({r}*count),{floor},100)",
              (lambda record, r=r, floor=floor: k_ratio(record, r, floor)))
             for r, floor in RATIO_FLOOR]
    if args.arms:
        wanted = set(args.arms.split(","))
        plan = [entry for entry in plan if entry[1] in wanted or entry[0] in wanted]
    for kind, name, selector in plan:
        result = evaluate_arm(name, kind, selector, blocks, data, rows_by_image_id, ledger, control)
        result["control_masks_per_frame"] = round(control_masks, 3)
        if kind.startswith("rule"):
            # Structural claim these families rest on: k >= count by construction, so a rule arm can
            # shed tail candidates but can never drop a reported (score >= 0.5) leaf.
            ledger.note(f"arm_{name}_never_truncates_reported_count",
                        result["frames_with_reported_count_truncated"] == 0,
                        f"{result['frames_with_reported_count_truncated']} frames truncated")
        arms.append(result)
        print(f"arm {name:28s} AP {result['macro']['AP']:.4f} "
              f"({-result['ap_drop_vs_control_pp']:+.4f} pp) "
              f"countMAE {result['macro']['count_MAE']:.4f} "
              f"masks/frame {result['mean_instances_masked_per_frame']:.2f} "
              f"({-result['mask_work_reduction_vs_control'] * 100:+.1f}% work) "
              f"gate={result['gate_ap_drop_within_1pp'] and result['gate_count_MAE_within_0_2_leaf']}",
              flush=True)
        write_json(out / "policy_eval_partial.json", {
            "version": VERSION, "updated_utc": utc(), "arms_completed": [a["name"] for a in arms],
            "identities": identities, "reproduction": reproduction, "control": control, "arms": arms})

    # ---- 4b. cost model read off the archived sweep, using offline mask counts -----------------
    masks_by_budget = {int(arm["name"]): float(arm["mean_instances_masked_per_frame"])
                       for arm in arms if arm["kind"] == "static" and arm["name"].isdigit()}
    timing = cost_model(root, MEASURED, masks_by_budget)
    if "slope_seconds_per_instance_per_frame_mean" in timing:
        print(f"cost model: seconds = {timing['slope_seconds_per_instance_per_frame_mean']} x "
              f"masks/frame + {timing['intercept_seconds']}  R2={timing['r_squared']}  "
              f"max |rel residual| {timing['max_abs_relative_residual']}  "
              f"({timing['ms_per_extra_masked_instance_per_frame']} ms per extra masked instance "
              f"per frame)", flush=True)

    # ---- 5. equal-work frontier: does per-frame allocation beat equal allocation? ---------------
    passing = [arm for arm in arms if arm["name"] != "100"
               and arm["gate_ap_drop_within_1pp"] and arm["gate_count_MAE_within_0_2_leaf"]]
    static_passing = [arm for arm in passing if arm["kind"] == "static"]
    frontier = []
    for arm in passing:
        if arm["kind"] == "static":
            continue
        peers = [s for s in static_passing
                 if abs(s["mean_instances_masked_per_frame"]
                        - arm["mean_instances_masked_per_frame"]) <= 0.05
                 * arm["mean_instances_masked_per_frame"]]
        best = min(peers, key=lambda s: s["ap_drop_vs_control_pp"]) if peers else None
        frontier.append({
            "arm": arm["name"], "masks_per_frame": arm["mean_instances_masked_per_frame"],
            "ap_drop_pp": arm["ap_drop_vs_control_pp"],
            "count_truncated_frames": arm["frames_with_reported_count_truncated"],
            "matched_static": None if best is None else {
                "arm": best["name"], "masks_per_frame": best["mean_instances_masked_per_frame"],
                "ap_drop_pp": best["ap_drop_vs_control_pp"]},
            "ap_drop_gain_vs_matched_static_pp": None if best is None
            else round(best["ap_drop_vs_control_pp"] - arm["ap_drop_vs_control_pp"], 4),
            "equal_static_arm_exists": bool(peers),
        })
    # The tightest achievable work point of the static family, for reference.
    best_static = min((arm for arm in arms if arm["kind"] == "static" and arm["name"] != "100"),
                      key=lambda a: (not (a["gate_ap_drop_within_1pp"]
                                          and a["gate_count_MAE_within_0_2_leaf"]),
                                     a["mean_instances_masked_per_frame"]), default=None)
    payload = {
        "version": VERSION, "generated_utc": utc(), "root": str(root), "out": str(out),
        "source_benchmark": str(benchmark), "checkpoint_sha256": config["checkpoint_sha256"],
        "scale": SCALE, "max_frames_per_plant": args.max_frames,
        "smoke_limited": bool(args.max_frames),
        "numeric_gates_active": strict,
        "declared_static_budgets": list(STATIC), "extra_static_budgets": args.extra_static,
        "arm_filter": args.arms,
        "evaluation_contract": ("offline prefix-labeling of the archived cap-100 development stream, "
                                "scored with src/verify_jrtip_budget_quality.aggregate()"),
        "roles_opened": list(DEV_ROLES), "locked_roles_accessed": [], "gpu_used": False,
        "predeclared_tolerances": {"AP_drop_vs_control_pp": AP_DROP_LIMIT_PP,
                                   "count_MAE_increase_vs_control_leaf": COUNT_INCREASE_LIMIT_LEAF},
        "budgets_measured_on_gpu": list(MEASURED),
        "prefix_identity": identities,
        "offline_vs_measured": reproduction,
        "control": control,
        "control_report": {key: control_report["macro"][key] for key in METRIC_KEYS},
        "arms": arms,
        "cost_model": timing,
        "equal_work_comparison": frontier,
        "best_static_arm": None if best_static is None else {
            "arm": best_static["name"],
            "masks_per_frame": best_static["mean_instances_masked_per_frame"],
            "ap_drop_pp": best_static["ap_drop_vs_control_pp"]},
        "checks": ledger.checks, "failures": ledger.failures,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(out / "policy_eval.json", payload)
    write_json(out / "status.json", {"status": "complete" if not ledger.failures else "failed",
                                     "updated_utc": utc(), "failures": ledger.failures,
                                     "elapsed_seconds": payload["elapsed_seconds"]})
    print(f"\nwrote {out / 'policy_eval.json'}  checks={len(ledger.checks)} "
          f"failures={len(ledger.failures)}", flush=True)
    for name in ledger.failures[:20]:
        print("FAIL", name, flush=True)
    return 0 if not ledger.failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "jrtip/detection_budget_quality_v1"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_policy_v1"))
    parser.add_argument("--arms", default="", help="optional comma-separated arm filter")
    parser.add_argument("--extra-static", dest="extra_static", default="",
                        help="additional static budgets to append to the arm plan (equal-work "
                             "densification); combine with --arms to run only these")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0,
                        help="frames per plant for a smoke run; recorded as smoke_limited")
    parser.add_argument("--reproduce-tol", dest="reproduce_tol", type=float, default=1e-9,
                        help="tolerance for offline-vs-measured macro agreement, in percentage points")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
