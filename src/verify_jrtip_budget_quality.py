"""Independently re-derive the detection-budget quality sweep from its raw per-frame streams.

Why this exists.  src/jrtip_detection_budget_quality.py measures one output-side detection budget
per process on the frozen development-quality contract, and
src/jrtip_detection_budget_merge.py does the control comparison.  Both trust
jrtip_pilot.quality() plus mva_pilot.coco_ground_truth/coco_mask_ap to aggregate correctly.  This
script never calls quality(): it reads the raw quality_NNNNNN_S640.jsonl streams back from disk,
rebuilds ground truth from the dataset, runs pycocotools itself, and re-implements the aggregation
chain (per-(role, genotype) collection -> genotype mean -> role macro -> equal dev domain) in this
file.  A number in the report is corroborated here or flagged.

Nothing here loads the checkpoint or touches the GPU, so the verification can run while other work
occupies the device.

What the sweep cannot see from inside, checked explicitly here: the headline AP is *not* an average
over the maxDets grid.  ``COCOeval.summarize`` reads ``stats[0] = _summarize(1)`` whose default is a
hard-coded ``maxDets=100``, and ``stats[1:3]`` use ``params.maxDets[2]``, so with
``maxDets=[1, 10, 100]`` the reported AP/AP50/AP75 are all evaluated at maxDets=100 and the first two
grid entries never enter them.  An early note in this project claimed the opposite - that a budget
below 10 is penalised by the metric grid independently of the instances it loses.  That claim is
false against this implementation, and the AP-by-maxDets extraction below is the measurement that
shows it: for a budget of 10 or 5 the maxDets=10 slice must equal the maxDets=100 slice exactly,
because no image carries more than 10 detections either way.  Reading the slices costs nothing - the
grid is already accumulated inside the single COCOeval call.

Development roles only; C_POLICY/T_ID/T_GENO/T_EXT are never read.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pycocotools import mask as mask_utils  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from src.jrtip_pilot import load_inputs, select_dev  # noqa: E402

SCALE = 640
DEV_ROLES = ("V_SOURCE", "V_SHIFT")
METRIC_KEYS = ("AP", "AP50", "AP75", "count_MAE")
MAXDET_SLOTS = ("1", "10", "100")
AP_DROP_LIMIT_PP, COUNT_INCREASE_LIMIT_LEAF = 1.0, 0.2
CONTROL = 100
TOL_METRIC = 1e-9


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def close(a, b, tol) -> bool:
    return abs(float(a) - float(b)) <= tol


def digest_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(mask) -> dict:
    rle = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def read_stream(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Ledger:
    def __init__(self):
        self.checks, self.notes = [], []

    def note(self, name, passed, detail=""):
        self.checks.append({"name": name, "pass": bool(passed), "detail": detail})
        if not passed:
            print(f"FAIL {name}: {detail}", flush=True)
        return bool(passed)

    def context(self, name, value):
        self.notes.append({"name": name, "value": value})

    @property
    def failures(self):
        return [check["name"] for check in self.checks if not check["pass"]]


def ground_truth(rows, data):
    """Own COCO ground-truth dict; image ids 1..n follow the order quality() enumerated."""
    images, annotations = [], []
    for image_id, row in enumerate(rows, 1):
        images.append({"id": image_id, "height": row["height"], "width": row["width"],
                       "source_image_id": row["image_id"]})
        target = data.label(row)
        masks = list(target["masks"].numpy())
        boxes = target["boxes"].tolist() if masks else []
        for index, mask in enumerate(masks):
            x0, y0, x1, y1 = boxes[index]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "segmentation": encode(mask), "area": int(mask.sum()),
                                "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0})
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "leaf"}]}


def coco_ap(gt_dataset, predictions, max_dets=(1, 10, 100)):
    """Own COCOeval call; empty prediction or empty GT set gives zeros, matching the contract.

    pycocotools' ``loadRes`` annotates the result records it is handed (it adds ``id``,
    ``area``, and a numpy ``bbox`` produced by ``maskUtils.toBbox``).  Reusing the same
    dicts for a second call would send them down the ``'bbox' in anns[0]`` branch of
    ``loadRes`` and raise, so every call gets shallow copies it may freely corrupt.

    Besides the three headline numbers, this returns the AP of every maxDets slot in the grid,
    read off ``eval['precision']`` with the same ``>-1`` masking pycocotools uses.  The
    maxDets=100 slot must reproduce ``stats[0]``; that identity is what proves the headline is a
    single-maxDets reading rather than an average over the grid.
    """
    if not predictions or not gt_dataset["annotations"]:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0,
                "AP_by_maxDets": {str(value): 0.0 for value in max_dets}}
    records = [{"image_id": ann["image_id"], "category_id": ann["category_id"],
                "score": ann["score"], "segmentation": dict(ann["segmentation"])}
               for ann in predictions]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dataset
        gt.createIndex()
        detected = gt.loadRes(records)
        evaluator = COCOeval(gt, detected, "segm")
        evaluator.params.imgIds = [image["id"] for image in gt_dataset["images"]]
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = list(max_dets)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    # precision has shape [IoU, recall, category, areaRng, maxDets]; the headline numbers average
    # over the IoU and recall axes at areaRng='all', so the area axis must be pinned to that label
    # rather than averaged over - averaging it too was this file's own first-attempt bug.
    area_index = list(evaluator.params.areaRngLbl).index("all")
    by_maxdet = {}
    for position, max_det in enumerate(max_dets):
        cell = precision[:, :, :, area_index, position]
        valid = cell[cell > -1]
        by_maxdet[str(max_det)] = float(np.mean(valid) * 100) if valid.size else 0.0
    return {name: float(max(0, evaluator.stats[index]) * 100)
            for index, name in enumerate(("AP", "AP50", "AP75"))} | {"AP_by_maxDets": by_maxdet}


def count_mae_chain(records):
    """frame -> plant -> genotype, exactly as the contract aggregates count MAE."""
    per_plant = defaultdict(list)
    for record in records:
        per_plant[record["plant"]].append(record["count_abs_error"])
    return float(np.mean([np.mean(values) for values in per_plant.values()])), len(per_plant)


def regroup(records, rows_by_image_id, ledger, budget, stream):
    """Rebuild quality()'s (role, genotype) blocks from the stream and check block order."""
    blocks, order = defaultdict(list), []
    for record in records:
        key = (record["split_role"], record["genotype"])
        if key not in blocks:
            order.append(key)
        blocks[key].append(record)
    ledger.note(f"cap{budget:03d}_blocks_are_sorted_contiguous", order == sorted(blocks),
                "file order " + str(order[:3]) + " vs sorted " + str(sorted(blocks)[:3]))
    expected_ids = [row["image_id"] for row in stream]
    ledger.note(f"cap{budget:03d}_stream_matches_declared_population",
                [record["image_id"] for record in records] == expected_ids,
                f"{len(records)} records vs {len(expected_ids)} selected dev frames")
    unknown = metadata_bad = size_bad = 0
    for record in records:
        row = rows_by_image_id.get(record["image_id"])
        if row is None:
            unknown += 1
            continue
        if (row["split_role"], row["genotype"], row["biological_plant_id"]) != (
                record["split_role"], record["genotype"], record["plant"]):
            metadata_bad += 1
        if list(record["original_hw"]) != [row["height"], row["width"]]:
            size_bad += 1
    ledger.note(f"cap{budget:03d}_every_record_maps_to_a_dev_row", unknown == 0, f"{unknown} records")
    ledger.note(f"cap{budget:03d}_role_genotype_plant_match_inventory", metadata_bad == 0,
                f"{metadata_bad} records")
    ledger.note(f"cap{budget:03d}_restored_masks_at_original_size", size_bad == 0, f"{size_bad} records")
    return blocks


def per_frame_consistency(records, budget, ledger):
    """Scores, counts and the budget cap, checked frame by frame against the stream itself."""
    bad_count = bad_error = 0
    returned = []
    for record in records:
        scores = [instance["score"] for instance in record["instances"]]
        returned.append(len(scores))
        if record["count"] != sum(score >= 0.5 for score in scores):
            bad_count += 1
        if record["count_abs_error"] != abs(record["count"] - record["gt_count"]):
            bad_error += 1
        if record["count"] > len(scores):
            bad_error += 1
    over_cap = sum(1 for value in returned if value > budget)
    ledger.note(f"cap{budget:03d}_count_equals_score_ge_half", bad_count == 0, f"{bad_count} frames")
    ledger.note(f"cap{budget:03d}_abs_error_and_count_internally_consistent", bad_error == 0,
                f"{bad_error} frames")
    ledger.note(f"cap{budget:03d}_no_frame_exceeds_the_budget", over_cap == 0, f"{over_cap} frames")
    return {"returned_instances_median": float(np.median(returned)),
            "returned_instances_max": int(np.max(returned)),
            "returned_instances_min": int(np.min(returned)),
            "frames_at_cap": int(sum(1 for value in returned if value == budget)),
            "frames": len(returned)}


def aggregate(blocks, data, rows_by_image_id, budget, ledger, with_extra_maxdets):
    genotype_metrics, extra, gt_mismatches = {}, {}, 0
    for (role, genotype), records in blocks.items():
        rows = [rows_by_image_id[record["image_id"]] for record in records]
        gt = ground_truth(rows, data)
        per_image_gt = defaultdict(int)
        for annotation in gt["annotations"]:
            per_image_gt[annotation["image_id"]] += 1
        gt_mismatches += sum(1 for index, record in enumerate(records, 1)
                             if per_image_gt.get(index, 0) != record["gt_count"])
        predictions = [{"image_id": index, "category_id": 1, "score": instance["score"],
                        "segmentation": instance["segmentation"]}
                       for index, record in enumerate(records, 1)
                       for instance in record["instances"]]
        metrics = coco_ap(gt, predictions)
        metrics["count_MAE"], metrics["plants"] = count_mae_chain(records)
        metrics["frames"] = len(records)
        metrics["gt_instances"] = len(gt["annotations"])
        genotype_metrics.setdefault(role, {})[genotype] = metrics
        if with_extra_maxdets:
            by_maxdet = metrics["AP_by_maxDets"]
            ledger.note(f"cap{budget:03d}_{role}/{genotype}_stats0_is_the_maxdets100_slice",
                        close(metrics["AP"], by_maxdet["100"], TOL_METRIC),
                        f"stats[0] {metrics['AP']!r} vs slice {by_maxdet['100']!r}")
            extra[f"{role}/{genotype}"] = by_maxdet
    role_macro = {role: {key: float(np.mean([entry[key] for entry in per_genotype.values()]))
                         for key in METRIC_KEYS} for role, per_genotype in genotype_metrics.items()}
    equal_domain = {key: float(np.mean([role_macro[role][key] for role in DEV_ROLES]))
                    for key in METRIC_KEYS}
    role_by_maxdet = {role: {slot: float(np.mean([entry["AP_by_maxDets"][slot]
                                                 for entry in per_genotype.values()]))
                             for slot in MAXDET_SLOTS}
                      for role, per_genotype in genotype_metrics.items()}
    domain_by_maxdet = {slot: float(np.mean([role_by_maxdet[role][slot] for role in DEV_ROLES]))
                        for slot in MAXDET_SLOTS}
    return genotype_metrics, role_macro, equal_domain, extra, gt_mismatches, domain_by_maxdet


def verify(root: Path, budgets, with_extra_maxdets, config):
    ledger = Ledger()
    benchmark = Path(config["source_benchmark"])
    manifest = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))
    frozen_quality = json.loads((benchmark / "quality.json").read_text(encoding="utf-8"))
    training = manifest["training_config"]
    rows, data, _ = load_inputs(training["inventory"], training["instances"], training["readiness"],
                                training["data_root"])
    stream = select_dev(rows, 0)
    rows_by_image_id = {row["image_id"]: row for row in stream}
    ledger.note("declared_stream_is_all_valid_dev_frames", len(stream) == 2947, f"{len(stream)} frames")
    ledger.note("stream_digest_matches_frozen_benchmark",
                digest_json([row["image_id"] for row in stream]) == frozen_quality["selected_stream_sha256"])
    ledger.note("locked_roles_absent_from_stream",
                {row["split_role"] for row in stream} == set(DEV_ROLES),
                str(sorted({row["split_role"] for row in stream})))

    summaries, per_budget = {}, {}
    for budget in budgets:
        directory = root / f"cap{budget:05d}"
        report_path = directory / f"budget_report_{budget:06d}.json"
        stream_path = directory / f"quality_{budget:06d}_S{SCALE}.jsonl"
        if not report_path.exists() or not stream_path.exists():
            ledger.note(f"cap{budget:03d}_artifacts_present", False, f"{report_path} / {stream_path}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        records = read_stream(stream_path)
        ledger.note(f"cap{budget:03d}_stream_digest_recorded_correctly",
                    sha256_file(stream_path) == report["quality_jsonl_sha256"])
        ledger.note(f"cap{budget:03d}_stream_line_count_recorded_correctly",
                    len(records) == report["quality_jsonl_lines"], f"{len(records)} lines")
        ledger.note(f"cap{budget:03d}_stream_is_the_declared_dev_stream",
                    report["selected_stream_sha256"]
                    == digest_json([row["image_id"] for row in stream]))
        blocks = regroup(records, rows_by_image_id, ledger, budget, stream)
        returned = per_frame_consistency(records, budget, ledger)
        need_extra = with_extra_maxdets and budget < 100
        genotype_metrics, role_macro, equal_domain, extra, gt_mismatches, domain_by_maxdet = aggregate(
            blocks, data, rows_by_image_id, budget, ledger, need_extra)
        ledger.note(f"cap{budget:03d}_gt_count_matches_dataset", gt_mismatches == 0,
                    f"{gt_mismatches} frames")
        if with_extra_maxdets:
            # The reported AP is read at maxDets=100, so a budget at or below 10 cannot be penalised
            # by the grid: with <=10 detections per image the maxDets=10 slice is the same set.
            ledger.note(f"cap{budget:03d}_ap_is_maxdets100_not_grid_average",
                        close(domain_by_maxdet["100"], equal_domain["AP"], 1e-9),
                        f"slice {domain_by_maxdet['100']!r} vs reported AP {equal_domain['AP']!r}")
            if budget <= 10:
                ledger.note(f"cap{budget:03d}_no_maxdets_grid_penalty",
                            close(domain_by_maxdet["10"], domain_by_maxdet["100"], 1e-9),
                            f"AP@maxDets10 {domain_by_maxdet['10']!r} vs "
                            f"AP@maxDets100 {domain_by_maxdet['100']!r}")
        for key in METRIC_KEYS:
            ledger.note(f"cap{budget:03d}_{key}_macro_reproduced",
                        close(report["macro"][key], equal_domain[key], TOL_METRIC),
                        f"report {report['macro'][key]!r} vs recomputed {equal_domain[key]!r}")
        for key in ("AP", "count_MAE"):
            for role in DEV_ROLES:
                ledger.note(f"cap{budget:03d}_{key}_{role}_reproduced",
                            close(report["macro_by_role"][key][role], role_macro[role][key], TOL_METRIC),
                            f"report {report['macro_by_role'][key][role]!r} vs "
                            f"recomputed {role_macro[role][key]!r}")
        mismatched = [f"{role}/{genotype}" for role, per_genotype in genotype_metrics.items()
                      for genotype, metrics in per_genotype.items()
                      if not all(close(report["per_genotype"][role][genotype][key], metrics[key],
                                       TOL_METRIC) for key in METRIC_KEYS)]
        ledger.note(f"cap{budget:03d}_per_genotype_table_reproduced", not mismatched,
                    ", ".join(mismatched[:6]))
        summaries[str(budget)] = {"macro": equal_domain, "returned_instance_shape": returned,
                                  "ap_by_maxdets_domain": domain_by_maxdet if with_extra_maxdets else None,
                                  "ap_by_maxdets_per_genotype": extra or None}
        per_budget[str(budget)] = report
        print(f"cap {budget}: verified macro AP {equal_domain['AP']:.4f}% "
              f"(report {report['macro']['AP']:.4f}%), count MAE {equal_domain['count_MAE']:.4f}, "
              f"returned/frame median {returned['returned_instances_median']:.0f}", flush=True)

    control = per_budget.get(str(CONTROL))
    if control is not None:
        ledger.note("control_reproduces_frozen_benchmark_bit_for_bit",
                    all(close(control["macro"][key],
                              frozen_quality["scales"][str(SCALE)]["equal_domain_macro"][key], 0.0)
                        for key in METRIC_KEYS),
                    str({key: (control["macro"][key],
                               frozen_quality["scales"][str(SCALE)]["equal_domain_macro"][key])
                         for key in METRIC_KEYS}))
        recomputed_control = summaries.get(str(CONTROL), {}).get("macro")
        if recomputed_control is not None:
            ledger.note("independently_recomputed_control_equals_frozen_benchmark",
                        all(close(recomputed_control[key],
                                  frozen_quality["scales"][str(SCALE)]["equal_domain_macro"][key],
                                  TOL_METRIC) for key in METRIC_KEYS),
                        str({key: (recomputed_control[key],
                                   frozen_quality["scales"][str(SCALE)]["equal_domain_macro"][key])
                             for key in METRIC_KEYS}))
    merged_path = root / "budget_quality.json"
    control_domain = summaries.get(str(CONTROL), {}).get("macro")
    merged_budget_count = None
    if merged_path.exists():
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        merged_budget_count = len(merged.get("budgets", {}))
    # The merged comparisons are only meaningful when this run covered exactly the budgets the
    # merged file covers; a partial --budgets run would otherwise report a spurious mismatch.
    if merged_budget_count is not None and control is not None and control_domain is not None \
            and merged_budget_count == len(summaries):
        bad = []
        for budget, report in per_budget.items():
            row = merged["budgets"][budget]
            drop = control["macro"]["AP"] - report["macro"]["AP"]
            increase = report["macro"]["count_MAE"] - control["macro"]["count_MAE"]
            acceptable = bool(drop <= AP_DROP_LIMIT_PP and increase <= COUNT_INCREASE_LIMIT_LEAF)
            if not close(row["AP_drop_vs_control_pp"], drop, TOL_METRIC):
                bad.append(f"{budget}:drop {row['AP_drop_vs_control_pp']} vs {drop}")
            if not close(row["count_MAE_increase_vs_control_leaf"], increase, TOL_METRIC):
                bad.append(f"{budget}:count {row['count_MAE_increase_vs_control_leaf']} vs {increase}")
            if row["quality_acceptable"] != acceptable:
                bad.append(f"{budget}:flag {row['quality_acceptable']} vs {acceptable}")
        ledger.note("merged_deltas_and_flags_reproduced", not bad, "; ".join(bad))
        expected_passing = sorted(
            [int(budget) for budget, summary in summaries.items()
             if summary["macro"]["AP"] is not None
             and control_domain["AP"] - summary["macro"]["AP"] <= AP_DROP_LIMIT_PP
             and summary["macro"]["count_MAE"] - control_domain["count_MAE"]
             <= COUNT_INCREASE_LIMIT_LEAF])
        ledger.note("merged_acceptable_set_reproduced_from_recomputed_metrics",
                    sorted(int(b) for b in merged["acceptable_budgets"]) == expected_passing,
                    f"merged {merged['acceptable_budgets']} vs recomputed {expected_passing}")
        ledger.note("merged_lowest_passing_budget_below_control",
                    merged["lowest_acceptable_budget_below_control"]
                    == min([b for b in expected_passing if b < CONTROL], default=None),
                    f"merged {merged['lowest_acceptable_budget_below_control']}")
        ledger.note("merged_verdict_matches_recomputed_set",
                    (merged["verdict"].endswith(", ".join(str(b) for b in expected_passing
                                                          if b < CONTROL))
                     if any(b < CONTROL for b in expected_passing)
                     else merged["verdict"] == "no_budget_below_the_control_passes_both_bars"),
                    f"verdict {merged['verdict']!r}, passing {expected_passing}")
    return ledger, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "jrtip/detection_budget_quality_v1"))
    parser.add_argument("--budgets", default="100,50,20,10,5")
    parser.add_argument("--maxdets-extra", dest="maxdets_extra", action="store_true",
                        help="record AP read at each maxDets slot (1/10/100) and assert the headline "
                             "is the maxDets=100 slice, i.e. that a budget <=10 is not penalised "
                             "by the metric grid")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = json.loads((root / "cap00100" / "budget_quality_config.json").read_text(encoding="utf-8"))
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    started = time.perf_counter()
    ledger, summaries = verify(root, budgets, args.maxdets_extra, config)
    result = {"version": "verify-jrtip-budget-quality-v1", "verified_utc": utc(),
              "root": str(root), "budgets": budgets,
              "checks": len(ledger.checks), "failures": ledger.failures,
              "passed": not ledger.failures,
              "maxdets_extra_computed": bool(args.maxdets_extra),
              "tolerance_metric": TOL_METRIC,
              "predeclared_bars": {"AP_drop_vs_control_pp": AP_DROP_LIMIT_PP,
                                   "count_MAE_increase_vs_control_leaf": COUNT_INCREASE_LIMIT_LEAF},
              "recomputed": summaries, "context": ledger.notes,
              "elapsed_seconds": time.perf_counter() - started,
              "scope_limits": [
                  "Ground truth and inventory metadata come from the frozen data layer "
                  "(load_inputs/select_dev); the metric aggregation and COCO calls are this file's "
                  "own implementation.",
                  "Development roles only; locked roles are never read, and no checkpoint or device "
                  "is used here.",
                  "maxDets=[1,10,100] is the frozen benchmark's grid, but the reported "
                  "AP/AP50/AP75 are read at maxDets=100 only (COCOeval.summarize hard-codes "
                  "maxDets=100 for stats[0] and uses params.maxDets[2] for stats[1:3]), so a budget "
                  "below 10 is NOT penalised by the grid independently of the instances it loses. An "
                  "earlier note in this project claimed the opposite; the ap_by_maxdets_* columns "
                  "here, and the cap<=10 checks that AP@maxDets10 == AP@maxDets100, are the machine "
                  "check of that correction."]}
    out = Path(args.out) if args.out else root / "budget_quality_independent_verification.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"],
                      "failures": ledger.failures[:20],
                      "elapsed_seconds": round(result["elapsed_seconds"], 1)},
                     indent=2, ensure_ascii=False), flush=True)
    print("wrote", out, flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
