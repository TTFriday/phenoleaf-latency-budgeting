"""Independent verification of a cap-100 feature/label archive (src/jrtip_budget_features.py).

Why this exists separately.  ``jrtip_budget_features.py`` turns one cap-100 development pass into the
Section 4.1 losses (ell_M, ell_N at every action arm) and the Section 4.2 per-frame features that the
Section 4.3 controller is fitted on.  Both sides of that file are small pieces of arithmetic where a
single transposed broadcast is silent: a first draft shipped a pairwise box IoU computed as
``x1[i] - x0[j]`` instead of ``min(x1[i], x1[j]) - max(x0[i], x0[j])``, which is asymmetric and
unbounded, and it surfaced only as ``pair_iou_mean = 12.05`` on a frame with seven instances.  Nothing
downstream of that feature would have failed loudly; the controller would simply have been fitted on
garbage.  So the checks below are written from the plan's definitions with independent code paths (pure
Python loops, full-frame mask intersections instead of box-cropped ones, scipy as the assignment
oracle) and run against the produced archive, and they gate the archive before any fit consumes it.

What is checked:

* contract - one record per selected frame, no duplicates, only the requested roles, every field of
  ``FEATURE_NAMES`` present exactly once and finite;
* ranges - every fraction-valued feature inside [0, 1], scores inside [0, 1], counts inside [0, 100],
  ell_M inside [0, 1] (the matched IoU sum can never exceed min(m, n) <= (m + n) / 2);
* internal consistency - ``n_05`` equals the label's ``count_ge_threshold``, ``m_pred_by_k[k]`` equals
  ``min(n_05, k)``, the cap-100 arm is exactly the reference (delta 0), and delta is exactly 0 whenever
  an arm truncates nothing (``m_pred_by_k[k] == m_pred_by_k[100]``);
* independent recomputation - for a deterministic sample of frames, the pairwise box statistics, the
  delivered mask count, the whole mask-IoU matrix, ell_M at every arm and ell_N are rebuilt from the
  archived RLE strings with separate code and compared;
* reported as diagnostics, not gates - how often marginal truncation *helps* (negative delta).  Plan v1
  Section 4.1 keeps negative values by design ("preserve negatives, a higher resolution can also be
  worse"), and here the analogous effect is a low-score false positive leaving the set.  A negative
  delta is therefore expected material for a learned ranker, and only its magnitude would be news.

Read-only over the archive and the cap-100 stream.  No GPU, no inference, no locked role.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as maskUtils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_features import (  # noqa: E402
    CONTROL, FEATURE_NAMES, MATCH_EDGE, SCALE,
)
from src.jrtip_budget_policy import instances_of  # noqa: E402

TOL = 1e-9
UNIT_INTERVAL = ("fg_area_fraction", "tiny_area_fraction", "edge_touch_fraction",
                 "pair_overlap_fraction", "pair_iou_mean")
IN_RANGE = ("score_mean", "score_min", "score_p10", "score_rank15", "score_rank20", "score_rank30")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def decode(rle: dict) -> np.ndarray:
    """Full-frame boolean mask straight from the archived RLE, no wrapper in between."""
    return np.asfortranarray(maskUtils.decode(
        {"size": [int(rle["size"][0]), int(rle["size"][1])],
         "counts": rle["counts"].encode("ascii")})).astype(bool)


def clip_boxes(boxes, width: int, height: int):
    return [[min(max(float(box[0]), 0.0), float(width)), min(max(float(box[1]), 0.0), float(height)),
             min(max(float(box[2]), 0.0), float(width)), min(max(float(box[3]), 0.0), float(height))]
            for box in boxes]


def box_pair_stats(boxes) -> tuple[float, float]:
    """Fraction of instance pairs whose boxes overlap, and the mean box IoU over all pairs.

    Textbook symmetric intersection, written as an explicit double loop so it shares no line of code
    with the vectorised implementation it is checking.
    """
    n = len(boxes)
    if n < 2:
        return 0.0, 0.0
    overlaps = ious = 0.0
    pairs = 0
    for i in range(n):
        ax0, ay0, ax1, ay1 = (float(v) for v in boxes[i])
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = (float(v) for v in boxes[j])
            w = min(ax1, bx1) - max(ax0, bx0)
            h = min(ay1, by1) - max(ay0, by0)
            inter = w * h if w > 0.0 and h > 0.0 else 0.0
            area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
            area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
            union = area_a + area_b - inter
            pairs += 1
            overlaps += float(inter > 0.0)
            ious += inter / union if union > 0.0 else 0.0
    return overlaps / pairs, ious / pairs


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two full-frame masks, intersected over the whole frame (the module crops to the box)."""
    inter = np.logical_and(a, b).sum()
    union = a.sum() + b.sum() - inter
    return float(inter) / float(union) if union else 0.0


def loss_m(iou: np.ndarray, m_pred: int, n_gt: int) -> float:
    """Section 4.1 ell_M: 1 - (max total IoU over IoU >= 0.5 edges) / ((|P| + |Y|) / 2)."""
    if m_pred == 0 and n_gt == 0:
        return 0.0
    if m_pred == 0 or n_gt == 0:
        return 1.0
    block = iou[:m_pred, :n_gt]
    allowed = block >= MATCH_EDGE
    if not allowed.any():
        return 1.0
    cost = np.where(allowed, -block, 0.0)
    rows, cols = linear_sum_assignment(cost)
    total = 0.0
    for r, c in zip(rows, cols):
        if allowed[r, c]:
            total += block[r, c]
    return 1.0 - total / ((m_pred + n_gt) / 2.0)


def recompute(record: dict, stream: dict, data, row_by_id: dict) -> dict:
    """Rebuild the checked quantities for one frame from the archived stream and ground truth."""
    iid = str(record["image_id"])
    archived = stream[iid]
    row = row_by_id[iid]
    height, width = int(row["height"]), int(row["width"])
    target = data.label(row)
    gt = [np.asarray(one, dtype=bool) for one in target["masks"].numpy()]
    instances = instances_of(archived)
    scores = np.asarray([float(inst["score"]) for inst in instances], dtype=np.float64)
    boxes = [[float(v) for v in inst["box_xyxy"]] for inst in instances]
    count_score = float(archived.get("count_score_threshold", 0.5))
    kept = [i for i in range(len(instances)) if scores[i] >= count_score]
    pred = [decode(instances[i]["segmentation"]) for i in kept]
    iou = np.zeros((len(pred), len(gt)))
    for i, one in enumerate(pred):
        for j, other in enumerate(gt):
            iou[i, j] = mask_iou(one, other)
    n_gt, m_full = len(gt), len(pred)
    full = loss_m(iou, m_full, n_gt)
    delivered_boxes = [boxes[i] for i in kept]
    overlap_fraction, iou_mean = box_pair_stats(clip_boxes(delivered_boxes, width, height))
    raw_overlap, raw_iou_mean = box_pair_stats(delivered_boxes)
    return {
        "n_05": float(m_full), "count_ge_threshold": int(sum(1 for s in scores if s >= count_score)),
        "masked_instances": min(len(instances), CONTROL),
        "lM_full": full, "lN_full": abs(m_full - n_gt), "gt_count": n_gt,
        "pair_overlap_fraction": overlap_fraction, "pair_iou_mean": iou_mean,
        "clipping_changes_pairs": abs(raw_iou_mean - iou_mean) > 1e-12,
        "raw_pair_iou_mean": raw_iou_mean,
        "by_k": {str(k): {"m_pred": min(m_full, int(k)), "lM": loss_m(iou, min(m_full, int(k)), n_gt)}
                 for k in record["labels"]["m_pred_by_k"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True,
                        help="directory written by jrtip_budget_features.py")
    parser.add_argument("--scope", choices=("TRAIN", "DEV"), required=True)
    parser.add_argument("--stream", required=True, help="the cap-100 JSONL the archive claims to came from")
    parser.add_argument("--benchmark", default=str(ROOT / "jrtip/baseline_benchmark_v1"))
    parser.add_argument("--frames", type=int, default=40, help="sample size for independent recomputation")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    fdir = Path(args.features)
    suffix = "train" if args.scope == "TRAIN" else "dev"
    path = fdir / f"frames_{suffix}.jsonl"
    report_path = fdir / f"features_{suffix}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    stream = {str(json.loads(line)["image_id"]): json.loads(line) for line in
              Path(args.stream).open(encoding="utf-8")}

    failures, notes = [], []
    # ---- contract -----------------------------------------------------------
    ids = [str(r["image_id"]) for r in records]
    if len(set(ids)) != len(ids):
        failures.append(f"duplicate image_id in archive ({len(ids) - len(set(ids))} repeats)")
    missing_in_stream = [i for i in ids if i not in stream]
    if missing_in_stream:
        failures.append(f"{len(missing_in_stream)} archived frames absent from the cap-100 stream")
    expected_roles = {"TRAIN": {"TRAIN"}, "DEV": {"V_SOURCE", "V_SHIFT"}}[args.scope]
    roles = {r["split_role"] for r in records}
    if not roles <= expected_roles:
        failures.append(f"unexpected roles in archive: {sorted(roles)}")
    if report.get("scope") != args.scope:
        failures.append(f"report scope {report.get('scope')} != requested {args.scope}")
    frozen = list(report.get("feature_names", []))
    if frozen and frozen != list(FEATURE_NAMES):
        failures.append("FEATURE_NAMES drifted from the archived report")
    if report.get("frames") != len(records):
        failures.append(f"report claims {report.get('frames')} frames, archive holds {len(records)}")
    for record in records:
        if int(record.get("scale", -1)) != SCALE or int(record.get("cap", -1)) != CONTROL:
            failures.append(f"{record['image_id']}: record is not scale {SCALE} / cap {CONTROL}")
            break

    # ---- ranges and internal consistency ------------------------------------
    n_negative_delta = n_negative_lM = 0
    worst_negative = 0.0
    for record in records:
        feats, labels = record["features"], record["labels"]
        if sorted(feats) != sorted(FEATURE_NAMES):
            failures.append(f"{record['image_id']}: feature keys != FEATURE_NAMES")
            continue
        for name, value in feats.items():
            if not np.isfinite(value):
                failures.append(f"{record['image_id']}: {name} is not finite")
        for name in UNIT_INTERVAL + IN_RANGE:
            if not -TOL <= feats[name] <= 1.0 + TOL:
                failures.append(f"{record['image_id']}: {name}={feats[name]:.6g} outside [0, 1]")
        if not -TOL <= labels["lM_full"] <= 1.0 + TOL:
            failures.append(f"{record['image_id']}: lM_full={labels['lM_full']:.6g} outside [0, 1]")
        for name in ("lN_full", "gt_count", "count_ge_threshold", "masked_instances"):
            if not float(labels[name]).is_integer():
                failures.append(f"{record['image_id']}: {name} is not integral")
        if feats["n_05"] != labels["count_ge_threshold"]:
            failures.append(f"{record['image_id']}: n_05 {feats['n_05']} != "
                            f"count_ge_threshold {labels['count_ge_threshold']}")
        if feats["n_emitted"] < feats["n_05"]:
            failures.append(f"{record['image_id']}: emitted {feats['n_emitted']} < delivered {feats['n_05']}")
        reference = labels["m_pred_by_k"][str(CONTROL)]
        for key, m_pred in labels["m_pred_by_k"].items():
            if m_pred != min(int(labels["count_ge_threshold"]), int(key)):
                failures.append(f"{record['image_id']}: m_pred({key})={m_pred} != min(n_05,{key})")
            if not -TOL <= labels["lM_by_k"][key] <= 1.0 + TOL:
                failures.append(f"{record['image_id']}: lM({key}) outside [0, 1]")
            if m_pred == reference and (labels["delta_M"][key] or labels["delta_N"][key]):
                failures.append(f"{record['image_id']}: arm {key} truncates nothing yet delta != 0")
            if int(key) == CONTROL and (labels["delta_M"][key] or labels["delta_N"][key]):
                failures.append(f"{record['image_id']}: cap-100 arm is not its own reference")
            if labels["delta_M"][key] < -TOL:
                n_negative_delta += 1
                worst_negative = min(worst_negative, float(labels["delta_M"][key]))
        if labels["lM_full"] < -TOL:
            n_negative_lM += 1

    # ---- independent recomputation ------------------------------------------
    from src.jrtip_pilot import load_inputs
    manifest = json.loads((Path(args.benchmark) / "benchmark_config.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    rows, data, _ = load_inputs(config["inventory"], config["instances"], config["readiness"],
                                config["data_root"])
    wanted = "TRAIN" if args.scope == "TRAIN" else None
    pool = [row for row in rows if (row["split_role"] == wanted if wanted
                                    else row["split_role"] in expected_roles)]
    row_by_id = {str(row["image_id"]): row for row in pool}
    rng = np.random.default_rng(np.random.PCG64(43))
    present = [i for i in ids if i in row_by_id]
    sample = sorted(rng.choice(len(present), size=min(args.frames, len(present)), replace=False).tolist())
    compared = clipping_matters = 0
    by_id = {str(r["image_id"]): r for r in records}
    for index in sample:
        iid = present[index]
        record = by_id[iid]
        mine = recompute(record, stream, data, row_by_id)
        clipping_matters += int(mine["clipping_changes_pairs"])
        feats, labels = record["features"], record["labels"]
        for name in ("n_05", "pair_overlap_fraction", "pair_iou_mean"):
            if abs(mine[name] - feats[name]) > 1e-6:
                failures.append(f"{iid}: {name} module {feats[name]:.6f} vs independent {mine[name]:.6f}")
        for name in ("count_ge_threshold", "masked_instances", "gt_count", "lN_full"):
            if int(mine[name]) != int(labels[name]):
                failures.append(f"{iid}: {name} module {labels[name]} vs independent {mine[name]}")
        if abs(mine["lM_full"] - labels["lM_full"]) > 1e-6:
            failures.append(f"{iid}: lM_full module {labels['lM_full']:.8f} vs independent "
                            f"{mine['lM_full']:.8f}")
        for key, value in mine["by_k"].items():
            if value["m_pred"] != int(labels["m_pred_by_k"][key]):
                failures.append(f"{iid}: m_pred({key}) {labels['m_pred_by_k'][key]} vs "
                                f"{value['m_pred']}")
            if abs(value["lM"] - labels["lM_by_k"][key]) > 1e-6:
                failures.append(f"{iid}: lM({key}) module {labels['lM_by_k'][key]:.8f} vs independent "
                                f"{value['lM']:.8f}")
            if abs((value["lM"] - mine["lM_full"]) - labels["delta_M"][key]) > 1e-6:
                failures.append(f"{iid}: delta_M({key}) is not lM(k) - lM(100)")
        compared += 1

    notes.append(f"{n_negative_delta} arm entries out of {len(records) * len(records[0]['labels']['delta_M'])} "
                 f"have a negative marginal delta_M (expected: plan v1 4.1 keeps negatives; "
                 f"most negative value seen {worst_negative:.4f})")
    notes.append(f"frame clipping changes the pairwise box statistics on {clipping_matters} of {compared} "
                 "sampled frames; the module clips, and this check compares against the clipped reading")
    if n_negative_lM:
        failures.append(f"{n_negative_lM} frames have lM < 0, which the definition forbids")

    passed = not failures
    result = {
        "version": "jrtip-budget-features-check-v1", "generated_utc": utc(),
        "features_dir": str(fdir), "scope": args.scope, "stream": args.stream,
        "frames_in_archive": len(records), "frames_recomputed_independently": compared,
        "expected_frames": report.get("frames"), "tolerance_abs": 1e-6,
        "checks": ["contract", "ranges", "internal_consistency", "independent_recomputation"],
        "notes": notes, "failures": failures[:200], "failure_count": len(failures), "passed": passed,
    }
    out = Path(args.out) if args.out else fdir / f"features_check_{suffix}.json"
    write_json(out, result)
    print(f"archive {path.name}: {len(records)} frames, {compared} recomputed independently, "
          f"{len(failures)} failures", flush=True)
    for line in failures[:20]:
        print("  FAIL", line, flush=True)
    for line in notes:
        print("  note", line, flush=True)
    print("features check:", "PASS" if passed else "FAIL", "->", out, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
