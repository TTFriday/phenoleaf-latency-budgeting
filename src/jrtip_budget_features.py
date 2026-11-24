"""Per-frame decision-point features and prefix loss labels for the budget controller (P1b).

Why this script exists
----------------------
Plan v2 section 4.2 fixes the information available when a frame's output budget is chosen, and
section 4.1 fixes the labels: for every budget ``k`` in the frozen action set, the marginal loss
``Delta_M(k) = l_M(P_k, Y) - l_M(P_100, Y)`` and ``Delta_N(k) = l_N(P_k, Y) - l_N(P_100, Y)``,
where ``P_k`` is the length-``k`` prefix of the cap-100 ordered instance sequence restricted to
``score >= count_score_threshold`` and ``Y`` is the ground-truth visible-instance set.  Because the
prefix lemma holds for this checkpoint (results doc section 9), the features and every ``k`` label
come out of *one* cap-100 pass - no second forward, no per-arm inference.

Single source of truth
----------------------
* ``l_M``/``l_N`` follow plan v1 section 4.1 verbatim: one-to-one maximum-total-IoU matching over
  edges with IoU >= 0.5, ``l_M = 1 - sum(matched IoU) / ((|P| + |Y|) / 2)``, ``l_N = ||P| - |Y|||``,
  both sets empty -> 0, exactly one side empty -> ``l_M = 1``.  Negative ``Delta`` is kept.
* The score thresholds come from the archived stream's own fields (``ap_score_threshold`` 0.001,
  ``count_score_threshold`` 0.5), so a stream written under a different contract cannot be silently
  re-read under this one.
* The cap constant comes from ``src/jrtip_budget_rule_timing.py``, the module that owns arm semantics.

Two documented contract limits (plan v2 section 4.2)
---------------------------------------------------
* ``n_0.05`` read from an archived cap-100 stream is ``min(true, 100)``; ``flag_n005_saturated``
  counts the frames where that clamp bites, so the deployment version's one-line change stays
  visible instead of being quietly baked into a feature.
* Every feature is a function of the delivered top-100 candidate set, so no feature may reference
  rank 101 or beyond; none does.

Locked roles stay closed: rows are filtered to TRAIN or to the two development roles, the guard runs
again per row inside ``data.label``, and the report records which roles were opened.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_policy import instances_of, read_stream
from src.jrtip_budget_rule_timing import CONTROL, SCALE
from src.jrtip_pilot import DEV_ROLES, digest_json, file_sha256, load_inputs, select_dev
from src.mva_pilot import append_jsonl, write_json  # noqa: E402

VERSION = "jrtip-budget-features-v1"
TRAIN_ROLE = "TRAIN"
MATCH_EDGE = 0.5            # matching edge threshold (plan v1 section 4.1)
TINY_AREA_FRACTION = 0.002  # "smaller than 0.2% of the frame" (plan v1 section 4.2 group 3)
RANK_SLOPE = (8, 40)        # 1-based inclusive rank window for the log-score slope
RANK_MARKS = (15, 20, 30)   # 1-based ranks whose scores are features
ACTION_DEFAULT = (100, 50, 30, 20, 15)

FEATURE_NAMES = (
    "raw_width", "raw_height", "raw_aspect",
    "n_emitted", "n_005", "n_05",
    "score_mean", "score_min", "score_p10",
    "log_slope_8_40", "score_rank15", "score_rank20", "score_rank30",
    "fg_area_fraction", "area_p10", "area_p50", "area_p90", "tiny_area_fraction",
    "edge_touch_fraction", "pair_overlap_fraction", "pair_iou_mean", "perimeter_sq_over_area_median",
    "flag_empty_output", "flag_slope_missing", "flag_rank15_missing", "flag_rank20_missing",
    "flag_rank30_missing", "flag_n005_saturated",
)
# Feature groups, used by the section 7.2 A1 ablation (drop the score-sequence group).
SCORE_SEQUENCE_GROUP = ("score_mean", "score_min", "score_p10", "log_slope_8_40",
                        "score_rank15", "score_rank20", "score_rank30")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_actions(spec: str) -> tuple:
    values = tuple(int(part) for part in spec.split(",") if part.strip())
    if not values or any(not 1 <= v <= CONTROL for v in values) or len(set(values)) != len(values):
        raise ValueError(f"action set must be unique budgets in 1..{CONTROL}: {spec!r}")
    return values


def rle_decode(rle):
    return mask_utils.decode({"size": [int(rle["size"][0]), int(rle["size"][1])],
                              "counts": rle["counts"].encode("ascii")})


def rle_area(rle) -> float:
    return float(mask_utils.area([{"size": [int(rle["size"][0]), int(rle["size"][1])],
                                   "counts": rle["counts"].encode("ascii")}])[0])


def quantile_or_zero(values, q) -> float:
    return float(np.quantile(values, q)) if len(values) else 0.0


def delivered(record, height: int, width: int) -> dict:
    """Split the cap-100 stream into the three sets the section 4.2 features are defined on."""
    instances = instances_of(record)
    count_score = float(record.get("count_score_threshold", 0.5))
    scores = np.asarray([float(item["score"]) for item in instances], dtype=np.float64)
    boxes = np.asarray([item["box_xyxy"] for item in instances], dtype=np.float64).reshape(-1, 4)
    kept = np.flatnonzero(scores >= count_score) if len(scores) else np.zeros(0, dtype=int)
    masks = [rle_decode(instances[i]["segmentation"]) for i in kept]
    areas = np.asarray([rle_area(instances[i]["segmentation"]) for i in kept], dtype=np.float64)
    return {"instances": instances, "scores": scores, "boxes": boxes, "kept": kept,
            "kept_boxes": boxes[kept] if len(kept) else boxes.reshape(0, 4),
            "kept_masks": masks, "kept_areas": areas, "count_score": count_score,
            "height": height, "width": width}


def boundary_length(mask) -> float:
    padded = np.pad(mask, 1)
    edge = ((padded[1:-1, :-2] != padded[1:-1, 2:]) | (padded[:-2, 1:-1] != padded[2:, 1:-1]))
    return float(np.count_nonzero(edge))


def features_for(frame: dict) -> dict:
    """Section 4.2 features; every input is post-NMS output or the raw frame geometry."""
    scores, kept_boxes, kept_masks = frame["scores"], frame["kept_boxes"], frame["kept_masks"]
    areas, height, width = frame["kept_areas"], frame["height"], frame["width"]
    instances = frame["instances"]
    frame_area = float(height) * float(width)
    n_kept = len(kept_masks)
    flags = {name: 0 for name in FEATURE_NAMES if name.startswith("flag_")}
    flags["flag_empty_output"] = int(len(instances) == 0)
    flags["flag_n005_saturated"] = int(len(instances) >= CONTROL)

    if n_kept:
        x0 = np.clip(kept_boxes[:, 0], 0.0, float(width))
        y0 = np.clip(kept_boxes[:, 1], 0.0, float(height))
        x1 = np.clip(kept_boxes[:, 2], 0.0, float(width))
        y1 = np.clip(kept_boxes[:, 3], 0.0, float(height))
        touches = float(np.mean((x0 <= 1.0) | (y0 <= 1.0) | (x1 >= width - 1.0)
                                | (y1 >= height - 1.0)))
        foreground = float(min(1.0, areas.sum() / frame_area)) if frame_area > 0 else 0.0
        perimeters = np.asarray([boundary_length(mask) for mask in kept_masks], dtype=np.float64)
        positive = areas > 0
        ratio_median = (float(np.median((perimeters[positive] ** 2) / areas[positive]))
                        if np.any(positive) else 0.0)
        tiny = float(np.mean(areas < TINY_AREA_FRACTION * frame_area)) if frame_area > 0 else 0.0
    else:
        touches = foreground = ratio_median = tiny = 0.0

    overlap_fraction = iou_mean = 0.0
    if n_kept > 1:
        inter_w = np.maximum(0.0, np.minimum(x1[:, None], x1[None, :])
                             - np.maximum(x0[:, None], x0[None, :]))
        inter_h = np.maximum(0.0, np.minimum(y1[:, None], y1[None, :])
                             - np.maximum(y0[:, None], y0[None, :]))
        inter = inter_w * inter_h
        box_area = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
        union = box_area[:, None] + box_area[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        upper = np.triu_indices(n_kept, 1)
        overlap_fraction = float(np.mean(inter[upper] > 0.0))
        iou_mean = float(np.mean(iou[upper]))

    if len(scores):
        window = scores[RANK_SLOPE[0] - 1:RANK_SLOPE[1]]
        slope_missing = int(len(window) < 2)
        if slope_missing:
            log_slope = 0.0
        else:
            ranks = np.arange(1, len(window) + 1, dtype=np.float64)
            log_slope = float(np.polyfit(ranks, np.log(np.maximum(window, 1e-8)), 1)[0])
    else:
        log_slope, slope_missing = 0.0, 1
    flags["flag_slope_missing"] = slope_missing

    marks = {}
    for rank in RANK_MARKS:
        if len(scores) >= rank:
            marks[rank], missing = float(scores[rank - 1]), 0
        else:
            marks[rank], missing = (float(scores[-1]) if len(scores) else 0.0), 1
        flags[f"flag_rank{rank}_missing"] = missing

    values = {
        "raw_width": float(width), "raw_height": float(height),
        "raw_aspect": float(width / height) if height else 0.0,
        "n_emitted": float(len(instances)),
        "n_005": float(np.count_nonzero(scores >= 0.05)) if len(scores) else 0.0,
        "n_05": float(n_kept),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_p10": quantile_or_zero(scores, 0.10),
        "log_slope_8_40": log_slope,
        "score_rank15": marks[15], "score_rank20": marks[20], "score_rank30": marks[30],
        "fg_area_fraction": foreground,
        "area_p10": quantile_or_zero(areas, 0.10), "area_p50": quantile_or_zero(areas, 0.50),
        "area_p90": quantile_or_zero(areas, 0.90), "tiny_area_fraction": tiny,
        "edge_touch_fraction": touches, "pair_overlap_fraction": overlap_fraction,
        "pair_iou_mean": iou_mean, "perimeter_sq_over_area_median": ratio_median,
        **flags}
    missing = [name for name in FEATURE_NAMES if name not in values]
    extra = [name for name in values if name not in FEATURE_NAMES]
    if missing or extra:
        raise ValueError(f"feature vector drifted: missing={missing} extra={extra}")
    return values


def iou_matrix(pred_masks, pred_boxes, gt_masks, gt_boxes) -> np.ndarray:
    """Mask IoU for every (delivered, ground-truth) pair, evaluated over the joint box window.

    Two shortcuts in the first draft were wrong, and both were silent:

    * pairs were skipped when their *box* IoU fell below ``MATCH_EDGE / 4``, but box IoU does not
      bound mask IoU from below - a mask is a subset of its own box, so a tight box against a loose one
      can give mask IoU 1.0 at any box IoU.  On 2 of the first 200 sampled development frames that
      prefilter deleted real section 4.1 matching edges and returned ``l_M`` 0.016 and 0.019 too high;
    * the intersection was taken over the *intersection* of the two boxes.  Two masks can overlap
      where the boxes' common area does not extend (mask pixels need not be confined to the reported
      box), so the union window is the sound speed-up.

    ``src/jrtip_budget_features_check.py`` recomputes these entries over the whole frame, which is what
    detected both effects; the union window is only used because that check says it is lossless.
    """
    out = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    pred_area = np.asarray([int(mask.sum()) for mask in pred_masks], dtype=np.float64)
    gt_area = np.asarray([int(mask.sum()) for mask in gt_masks], dtype=np.float64)
    for i, (mask, box) in enumerate(zip(pred_masks, pred_boxes)):
        for j, (other, gt_box) in enumerate(zip(gt_masks, gt_boxes)):
            if pred_area[i] <= 0.0 or gt_area[j] <= 0.0:
                continue
            top = int(max(0, min(box[1], gt_box[1])))
            left = int(max(0, min(box[0], gt_box[0])))
            bottom = int(min(mask.shape[0], max(box[3], gt_box[3])))
            right = int(min(mask.shape[1], max(box[2], gt_box[2])))
            if bottom <= top or right <= left:
                continue
            inter = float(np.logical_and(mask[top:bottom, left:right],
                                         other[top:bottom, left:right]).sum())
            union = pred_area[i] + gt_area[j] - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


def loss_for(m_pred: int, iou: np.ndarray, n_gt: int) -> float:
    """l_M of the first ``m_pred`` delivered instances against ``n_gt`` ground-truth instances."""
    if m_pred == 0 and n_gt == 0:
        return 0.0
    if m_pred == 0 or n_gt == 0:
        return 1.0
    block = iou[:m_pred, :n_gt]
    allowed = block >= MATCH_EDGE
    if not allowed.any():
        return 1.0
    rows, cols = linear_sum_assignment(np.where(allowed, -block, 0.0))
    total = float(sum(block[r, c] for r, c in zip(rows, cols) if allowed[r, c]))
    return 1.0 - total / ((m_pred + n_gt) / 2.0)


def labels_for(frame: dict, target, actions) -> dict:
    gt_masks = [np.asarray(mask, dtype=bool) for mask in target["masks"].numpy()]
    gt_boxes = [[float(v) for v in box] for box in target["boxes"].numpy()]
    iou = iou_matrix(frame["kept_masks"], [list(map(float, box)) for box in frame["kept_boxes"]],
                     gt_masks, gt_boxes)
    n_gt, m_full = len(gt_masks), len(frame["kept_masks"])
    base_m, base_n = loss_for(m_full, iou, n_gt), abs(m_full - n_gt)
    delta_m, delta_n, m_by_k, absolute = {}, {}, {}, {}
    for k in actions:
        m_pred = min(int(k), m_full)
        value = loss_for(m_pred, iou, n_gt)
        delta_m[str(k)] = round(value - base_m, 8)
        delta_n[str(k)] = int(abs(m_pred - n_gt) - base_n)
        m_by_k[str(k)] = m_pred
        absolute[str(k)] = round(value, 8)
    return {"lM_full": round(base_m, 8), "lN_full": int(base_n), "gt_count": int(n_gt),
            "count_ge_threshold": int(m_full), "masked_instances": int(len(frame["instances"])),
            "m_pred_by_k": m_by_k, "lM_by_k": absolute, "delta_M": delta_m, "delta_N": delta_n,
            "match_edges": int(np.count_nonzero(iou >= MATCH_EDGE))}


def selected_rows(rows, scope: str, max_frames: int):
    if scope == TRAIN_ROLE:
        from src.jrtip_budget_train_stream import select_train
        return select_train(rows, max_frames)
    if scope == "DEV":
        stream = select_dev(rows, max_frames)
        if {row["split_role"] for row in stream} - set(DEV_ROLES):
            raise PermissionError("DEV scope must stay inside the two development roles")
        return stream
    raise ValueError(f"scope must be {TRAIN_ROLE} or DEV, got {scope}")


def run(args):
    started = time.perf_counter()
    actions = resolve_actions(args.actions)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    benchmark = Path(args.benchmark).resolve()
    manifest = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))
    training = manifest["training_config"]
    rows, data, _ = load_inputs(training["inventory"], training["instances"], training["readiness"],
                                training["data_root"])
    selected = selected_rows(rows, args.scope, args.max_frames)
    rows_by_id = {row["image_id"]: row for row in selected}
    stream = Path(args.stream).resolve()
    if not stream.exists():
        raise FileNotFoundError(f"missing cap-{CONTROL} stream: {stream}")
    records = [record for record in read_stream(stream) if record["image_id"] in rows_by_id]
    if len(records) != len(rows_by_id):
        raise ValueError(f"stream covers {len(records)} of {len(rows_by_id)} selected frames; the "
                         "cap-100 pass is incomplete or --max-frames disagrees with --stream")
    target = out / f"frames_{args.scope.lower()}.jsonl"
    if target.exists():
        if not args.overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite or use a fresh --out")
        target.unlink()
    for index, record in enumerate(records, 1):
        row = rows_by_id[record["image_id"]]
        frame = delivered(record, int(row["height"]), int(row["width"]))
        append_jsonl(target, {
            "image_id": record["image_id"], "split_role": row["split_role"],
            "genotype": row["genotype"], "plant": row["biological_plant_id"],
            "scale": SCALE, "cap": CONTROL,
            "features": features_for(frame),
            "labels": labels_for(frame, data.label(row), actions)})
        if index % args.log_every == 0:
            rate = (time.perf_counter() - started) / index
            print(f"{args.scope} features {index}/{len(records)} frames, {rate:.3f} s/frame, "
                  f"eta {rate * (len(records) - index) / 60:.1f} min", flush=True)
    report = {
        "version": VERSION, "generated_utc": utc(), "scope": args.scope,
        "roles_opened": sorted({row["split_role"] for row in selected}),
        "locked_roles_accessed": [], "formal_36_epoch_run": False, "diagnostic_only": True,
        "benchmark": str(benchmark), "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "stream": str(stream), "stream_sha256": file_sha256(stream),
        "selected_ids_digest": digest_json([row["image_id"] for row in selected]),
        "action_set": list(actions), "feature_names": list(FEATURE_NAMES),
        "score_sequence_group": list(SCORE_SEQUENCE_GROUP),
        "match_edge_iou": MATCH_EDGE, "tiny_area_fraction": TINY_AREA_FRACTION,
        "rank_slope_window": list(RANK_SLOPE), "rank_marks": list(RANK_MARKS),
        "frames": len(records), "plants": len({row["biological_plant_id"] for row in selected}),
        "genotypes": len({row["genotype"] for row in selected}),
        "definitions": {
            "l_M": "1 - sum(IoU of a max-total-IoU one-to-one matching over edges IoU>=0.5) / "
                   "((|P_k|+|Y|)/2); both empty 0; exactly one side empty 1",
            "l_N": "abs(|P_k| - |Y|), P_k = first min(k, |P|) delivered instances with "
                   "score >= count_score_threshold",
            "delta": "loss at prefix k minus loss at the full delivered set (negative allowed)",
            "n_005": "min(true count, cap) as archived; see flag_n005_saturated",
            "geometry_group": "score >= count_score_threshold instances only (plan v1 section 3.2 "
                              "output thresholds)"},
        "seconds": round(time.perf_counter() - started, 1),
        "jsonl": target.name, "jsonl_sha256": file_sha256(target)}
    write_json(out / f"features_{args.scope.lower()}.json", report)
    print(f"{args.scope}: wrote {len(records)} frames in {report['seconds']} s -> {target}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=(TRAIN_ROLE, "DEV"))
    parser.add_argument("--stream", required=True, help=f"cap-{CONTROL} per-frame JSONL for --scope")
    parser.add_argument("--out", required=True)
    parser.add_argument("--benchmark", default=str(ROOT / "jrtip/baseline_benchmark_v1"))
    parser.add_argument("--actions", default=",".join(str(value) for value in ACTION_DEFAULT))
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0,
                        help="per-plant cap; 0 = the whole scope, positive = smoke subset")
    parser.add_argument("--log-every", dest="log_every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.log_every < 1:
        raise ValueError("log-every must be >= 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
