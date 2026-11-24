"""Structural audit of an already-written TRAIN cap-100 archive, without a GPU and without writing to it.

``src/jrtip_budget_train_stream.py`` wrote all 8,070 TRAIN frames on the first attempt and then died
in its own post-pass bookkeeping (a ``NameError`` in the image_id guard below the write loop), so the
archive exists but neither its completeness nor ``train_stream_report.json`` was ever asserted.  The
chain's stages 43-46 consume the archive directly, so "is the file whole?" has to be answered before
the controller fit is trusted - and answered without re-running a 57-minute inference pass.

Checks, all structural: the archive is exactly the deterministic TRAIN selection (same ids, same
order, no repeats, nothing from another role), every record carries the frozen schema, each record's
genotype/plant identity agrees with the inventory row for that image_id, the emitted instances
respect the cap and stay score-descending, the reported count equals the score >= 0.5 tally, and the
count error equals |count - gt_count|.  Quality - COCO mask AP and count MAE on TRAIN - is
deliberately left to ``train_pass``'s own ``score_genotypes``, because recomputing a metric here
would mean a second metric implementation, which is what section 9's byte gate exists to rule out.

Read-only over the archive; TRAIN role only (never a selection or effect-analysis role); no GPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_rule_timing import CONTROL, SCALE, read_stream  # noqa: E402
from src.jrtip_budget_train_stream import BENCHMARK, ROLE, select_train  # noqa: E402
from src.jrtip_pilot import load_inputs  # noqa: E402

REQUIRED = ("image_id", "split_role", "genotype", "plant", "count", "gt_count", "count_abs_error",
            "original_hw", "geometry", "instances", "ap_score_threshold", "count_score_threshold",
            "mask_threshold")
INSTANCE_KEYS = ("score", "box_xyxy", "segmentation")
MAX_REPORTED = 20


def note(failures, text):
    failures.append(text)
    print("FAIL", text, flush=True)


def main():
    stream = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "jrtip/budget_train_stream_v1" / f"quality_{CONTROL:06d}_S{SCALE}_TRAIN.jsonl")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else stream.parent / "train_stream_audit.json"
    manifest = json.loads((BENCHMARK / "benchmark_config.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    rows, _data, _ready = load_inputs(config["inventory"], config["instances"], config["readiness"],
                                      config["data_root"])
    selected = select_train(rows, 0)
    order = [str(row["image_id"]) for row in selected]
    by_id = {str(row["image_id"]): row for row in selected}

    failures, ids, records = [], [], 0
    counters = {key: 0 for key in ("schema", "identity", "role", "cap", "score_order", "count_mismatch",
                                   "count_error", "thresholds")}
    started = time.perf_counter()
    for line, record in enumerate(read_stream(stream), 1):
        records += 1
        missing = [key for key in REQUIRED if key not in record]
        if missing:
            counters["schema"] += 1
            if counters["schema"] <= 3:
                note(failures, f"line {line}: missing fields {missing}")
            continue
        if any(not set(INSTANCE_KEYS) <= set(item) for item in record["instances"]):
            counters["schema"] += 1
            if counters["schema"] <= 3:
                note(failures, f"line {line}: instance records lack {list(INSTANCE_KEYS)}")
        image_id = str(record["image_id"])
        ids.append(image_id)
        if record["split_role"] != ROLE:
            counters["role"] += 1
            if counters["role"] <= 3:
                note(failures, f"line {line}: split_role {record['split_role']} is not {ROLE}")
        row = by_id.get(image_id)
        if row is None:
            counters["identity"] += 1
            if counters["identity"] <= 3:
                note(failures, f"line {line}: image_id {image_id} is not in the TRAIN selection")
        elif (str(record["genotype"]) != str(row["genotype"])
              or str(record["plant"]) != str(row["biological_plant_id"])):
            counters["identity"] += 1
            if counters["identity"] <= 3:
                note(failures, f"line {line}: image_id {image_id} identifies "
                               f"{record['genotype']}/{record['plant']}, the inventory says "
                               f"{row['genotype']}/{row['biological_plant_id']}")
        if len(record["instances"]) > CONTROL:
            counters["cap"] += 1
            if counters["cap"] <= 3:
                note(failures, f"line {line}: {len(record['instances'])} instances exceed cap {CONTROL}")
        scores = [float(item["score"]) for item in record["instances"]]
        if any(b > a + 1e-12 for a, b in zip(scores, scores[1:])):
            counters["score_order"] += 1
            if counters["score_order"] <= 3:
                note(failures, f"line {line}: instance scores are not descending")
        above = sum(1 for value in scores if value >= float(record["count_score_threshold"]))
        if int(record["count"]) != above:
            counters["count_mismatch"] += 1
            if counters["count_mismatch"] <= 3:
                note(failures, f"line {line}: count {record['count']} disagrees with the "
                               f"score >= {record['count_score_threshold']} tally of {above}")
        if int(record["count_abs_error"]) != abs(int(record["count"]) - int(record["gt_count"])):
            counters["count_error"] += 1
            if counters["count_error"] <= 3:
                note(failures, f"line {line}: count_abs_error {record['count_abs_error']} is not "
                               f"|{record['count']} - {record['gt_count']}|")
        if (float(record["ap_score_threshold"]) != 0.001 or float(record["count_score_threshold"]) != 0.5
                or float(record["mask_threshold"]) != 0.5):
            counters["thresholds"] += 1
            if counters["thresholds"] <= 3:
                note(failures, f"line {line}: scoring thresholds drifted from 0.001/0.5/0.5")

    if len(ids) != len(set(ids)):
        note(failures, f"archive repeats image_ids: {len(ids)} rows, {len(set(ids))} distinct")
    if ids != order:
        outside = [item for item in ids if item not in set(order)]
        absent = [item for item in order if item not in set(ids)]
        note(failures, f"archive differs from the deterministic TRAIN selection: {len(ids)} rows vs "
                       f"{len(order)} selected, {len(outside)} outside "
                       f"{outside[:3]}, {len(absent)} missing {absent[:3]}")
    result = {
        "version": "jrtip-budget-train-stream-audit-v1", "generated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stream": str(stream), "seconds": round(time.perf_counter() - started, 1),
        "rows": records, "distinct_image_ids": len(set(ids)), "selected_frames": len(order),
        "selected_plants": len({row["biological_plant_id"] for row in selected}),
        "selected_genotypes": sorted({row["genotype"] for row in selected}),
        "role": ROLE, "locked_roles_accessed": [], "required_fields": list(REQUIRED),
        "instance_fields": list(INSTANCE_KEYS), "counters": counters,
        "structural_failure_count": len(failures), "structural_failures": failures[:MAX_REPORTED],
        "passed": not failures,
        "note": ("structural audit only; the TRAIN COCO mask AP / count MAE readout comes from "
                 "train_pass's score_genotypes, so this script never recomputes a metric"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({key: value for key, value in result.items() if key != "structural_failures"},
                     indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
