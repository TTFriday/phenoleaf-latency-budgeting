"""Per-frame headroom of the output-side detection budget (CPU-only, 2026-11-22).

Reads the per-frame streams that ``jrtip_detection_budget_quality.py`` already wrote and asks
the question the formal protocol now hinges on: is the cost of ``detections_per_img`` uniform
across frames, or is it concentrated in a few dense frames?  If it is concentrated, a
per-frame budget is worth modelling; if every frame uses the whole budget, the only honest
result is "set the config lower" and the adaptive contribution dies.

Definitions used here (all from the score-ordered instance lists in the streams):

* ``returned``  - instances a frame actually got masks for at that budget (score >= 0.001,
                  capped by the budget).  This is what the mask head cost is proportional to.
* ``count``     - instances with score >= 0.5, i.e. the reported leaf count.
* ``safe_floor`` - the smallest budget whose top-k prefix still contains every score >= 0.5
                  instance.  Because the head ranks by score before truncating, this equals
                  ``count`` exactly; it is the oracle per-frame budget for the count metric.

Development roles only (V_SOURCE/V_SHIFT); no locked role is read.  Nothing here is a
quality verdict - the AP cost of a budget comes from the budget-quality pass, not from this
file.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

BUDGETS = (100, 50, 20, 10, 5)
K_GRID = (5, 10, 15, 20, 30, 50, 100)
ROOT = Path(__file__).resolve().parents[1] / "jrtip" / "detection_budget_quality_v1"


def stream(budget: int):
    path = ROOT / f"cap{budget:05d}" / f"quality_{budget:06d}_S640.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary(values) -> dict:
    ordered = sorted(values)
    return {"n": len(ordered), "mean": round(statistics.fmean(ordered), 4),
            "p50": ordered[len(ordered) // 2], "p90": ordered[int(0.90 * (len(ordered) - 1))],
            "p99": ordered[int(0.99 * (len(ordered) - 1))], "max": ordered[-1],
            "min": ordered[0]}


def main() -> int:
    streams = {budget: stream(budget) for budget in BUDGETS}
    ids = [record["image_id"] for record in streams[100]]
    for budget, rows in streams.items():
        assert [row["image_id"] for row in rows] == ids, f"budget {budget} stream is reordered"

    control = streams[100]
    returned = [len(row["instances"]) for row in control]
    counts = [row["count"] for row in control]
    gt_counts = [row["gt_count"] for row in control]

    out: dict = {
        "version": "jrtip-budget-headroom-v1",
        "stream_frames": len(control),
        "control_budget": 100,
        "returned_instances_at_control": summary(returned),
        "count_at_control": summary(counts),
        "gt_count": summary(gt_counts),
        "frames_saturating_control_budget": sum(1 for value in returned if value == 100),
        "oracle_safe_floor": summary(counts),
        "per_frame_masks_removed_by_fixed_budgets": {},
        "count_changed_vs_control": {},
        "count_error_changed_vs_control": {},
        "gt_recall_of_count_at_control": {},
    }

    # How much mask work a fixed budget removes, and whether the reported count survives it.
    for budget in BUDGETS:
        rows = streams[budget]
        out["per_frame_masks_removed_by_fixed_budgets"][str(budget)] = {
            "mean_returned": round(statistics.fmean([len(row["instances"]) for row in rows]), 4),
            "median_returned": statistics.median([len(row["instances"]) for row in rows]),
            "frames_at_cap": sum(1 for row in rows if len(row["instances"]) == budget),
        }
        out["count_changed_vs_control"][str(budget)] = sum(
            1 for row, base in zip(rows, control) if row["count"] != base["count"])
        out["count_error_changed_vs_control"][str(budget)] = sum(
            1 for row, base in zip(rows, control)
            if row["count_abs_error"] != base["count_abs_error"])

    # Which frames would a per-frame budget put at risk: dense frames need a large k.
    for k in K_GRID:
        risky = [row for row in control if row["count"] > k]
        dense_gt = [row for row in control if row["gt_count"] > k]
        out["gt_recall_of_count_at_control"][str(k)] = {
            "frames_with_count_above_k": len(risky),
            "share_of_frames": round(len(risky) / len(control), 4),
            "frames_with_gt_count_above_k": len(dense_gt),
            "share_of_frames_gt": round(len(dense_gt) / len(control), 4),
        }

    # Where the count cost of cap 20 actually lands, per genotype and per role.
    for key, index in (("by_role", "split_role"), ("by_genotype", "genotype")):
        buckets = defaultdict(lambda: {"frames": 0, "count_changed": 0, "returned_mean": 0.0,
                                       "count_mean": 0.0, "gt_mean": 0.0})
        for row, base in zip(streams[20], control):
            entry = buckets[row[index]]
            entry["frames"] += 1
            entry["count_changed"] += int(row["count"] != base["count"])
            entry["returned_mean"] += len(row["instances"])
            entry["count_mean"] += base["count"]
            entry["gt_mean"] += base["gt_count"]
        out[key] = {name: {"frames": value["frames"],
                           "frames_count_changed_at_cap20": value["count_changed"],
                           "mean_returned_at_cap20": round(value["returned_mean"] / value["frames"], 2),
                           "mean_count_at_control": round(value["count_mean"] / value["frames"], 2),
                           "mean_gt_count": round(value["gt_mean"] / value["frames"], 2)}
                    for name, value in sorted(buckets.items())}

    text = json.dumps(out, indent=2, ensure_ascii=False)
    (ROOT / "budget_headroom.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
