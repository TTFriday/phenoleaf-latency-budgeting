"""TRAIN-domain cap-100 pricing stream for the P1b controller fit (plan v2 sections 4.1-4.3).

Why this script exists
----------------------
Plan v2 section 4.3 fits the budget controller on out-of-fold data whose folds live *inside*
TRAIN, and section 4.1 needs one ordered instance sequence per frame so that every budget arm is a
prefix of that same sequence.  The archived ``jrtip/detection_budget_quality_v1/cap00100`` stream
covers only the development roles (V_SOURCE/V_SHIFT), so TRAIN has no cap-100 stream at all.  This
script produces it with the frozen P1 checkpoint, at long edge 640 and the model's default
``roi_heads.detections_per_img = 100``, and nothing else.

Single source of truth
----------------------
* The record schema is ``src/jrtip_pilot.quality()``'s: the same ``input_bytes``/``decode``/
  ``infer_image``/``restore_output`` call chain, and the same field set and field order, so the
  output is interchangeable with every archived stream the offline pricing already reads.
* The instrument gate proves it: the identical loop re-run on the first ``--gate-frames``
  development frames must reproduce the archived cap-100 records *byte-for-byte* (sorted-key JSON
  equality of the whole record, including scores, boxes and RLE strings).  A gate failure aborts
  before any TRAIN frame is written.

What is deliberately not done here
----------------------------------
* No locked role is opened.  ``rows`` is filtered to ``split_role == "TRAIN"``, the guard runs
  again per row inside the loop, and the report records ``roles_opened: ["TRAIN"]``.
* No new decision variable and no timing.  Service time for budget arms is measured per arm by
  ``src/jrtip_budget_rule_timing.py``; section 6 forbids substituting an offline slope.
* No claim that these frames are out-of-sample for the *backbone*: the P1 checkpoint was fit on
  TRAIN, so the OOF folds protect the regressor from frame-level over-fitting only.  The report
  states this limitation instead of hiding it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_rule_timing import CONTROL, SCALE, install_signal_recorder, read_stream
from src.jrtip_pilot import (
    NOTE, build_jp_detector, decode, digest_json, file_sha256, infer_image, input_bytes, jp_path,
    load_inputs, machine_snapshot, run_lock, seed_backend, select_dev, status, sync, versions)
from src.mva_pilot import append_jsonl, coco_ground_truth, coco_mask_ap, write_json  # noqa: E402

VERSION = "jrtip-budget-train-stream-v1"
ROLE = "TRAIN"
CAP = CONTROL
MACRO_KEYS = ("AP", "AP50", "AP75", "count_MAE")
ARCHIVED_DEV = ROOT / "jrtip/detection_budget_quality_v1/cap00100/quality_000100_S640.jsonl"
BENCHMARK = ROOT / "jrtip/baseline_benchmark_v1"


def select_train(rows, max_frames=0):
    """All valid TRAIN frames in a deterministic plant-major order (mirrors select_dev)."""
    if max_frames < 0:
        raise ValueError("Frame subset limit must be >=0")
    groups = defaultdict(list)
    for row in rows:
        if row["split_role"] == ROLE and row["mask_status"].lower() == "valid":
            groups[(row["genotype"], row["biological_plant_id"])].append(row)
    selected = []
    for key in sorted(groups):
        frames = sorted(groups[key], key=lambda r: (r["timestamp"], r["image_id"]))
        if max_frames and len(frames) > max_frames:
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
        selected.extend(frames)
    return selected


def record_for(model, row, data, device):
    """One frame, written exactly the way jrtip_pilot.quality() writes it."""
    if row["split_role"] not in (ROLE, *("V_SOURCE", "V_SHIFT")):
        raise PermissionError(f"Locked role refused: {row['split_role']}")
    target = data.label(row)
    image = decode(input_bytes(row), (row["height"], row["width"]))
    payload = infer_image(model, image, SCALE, device)
    return {"image_id": row["image_id"], "split_role": row["split_role"], "genotype": row["genotype"],
            "plant": row["biological_plant_id"],
            "count_abs_error": abs(payload["count"] - len(target["masks"])),
            "gt_count": len(target["masks"]), **payload}


def canon(record) -> str:
    return json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))


@torch.inference_mode()
def instrument_gate(model, rows, data, device, out: Path, frames: int) -> dict:
    """Reproduce archived development cap-100 records with this loop before trusting TRAIN."""
    archived = {str(record["image_id"]): record for record in read_stream(ARCHIVED_DEV)}
    subset = select_dev(rows, 0)[:frames]
    checked, mismatches = 0, []
    started = time.perf_counter()
    for row in subset:
        mine = record_for(model, row, data, device)
        theirs = archived.get(str(row["image_id"]))
        if theirs is None:
            raise ValueError(f"Archived stream has no frame {row['image_id']}")
        checked += 1
        if canon(mine) != canon(theirs):
            field = next((key for key in mine if json.dumps(mine[key], sort_keys=True)
                          != json.dumps(theirs.get(key), sort_keys=True)), "unknown")
            mismatches.append({"image_id": mine["image_id"], "first_field": field})
            if len(mismatches) >= 5:
                break
    report = {"gate": "archived_dev_stream_byte_identity", "frames_checked": checked,
              "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
              "passed": checked > 0 and not mismatches,
              "archived_stream": str(ARCHIVED_DEV),
              "archived_stream_sha256": file_sha256(ARCHIVED_DEV),
              "seconds": round(time.perf_counter() - started, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "instrument_gate.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Instrument gate failed: {report}")
    return report


def macro_from(report) -> dict:
    return {key: float(np.mean([g[key] for g in report.values()])) for key in MACRO_KEYS}


@torch.inference_mode()
def train_pass(model, rows, data, device, out: Path, selected, log_every: int) -> dict:
    stream = out / f"quality_{CAP:06d}_S{SCALE}_TRAIN.jsonl"
    done = {str(record["image_id"]) for record in read_stream(stream)} if stream.exists() else set()
    model.eval()
    original = model.roi_heads.detections_per_img
    if original != CAP:
        raise ValueError(f"Control cap {CAP} is not the model default {original}")
    started = time.perf_counter()
    written = 0
    with run_lock(out):
        for index, row in enumerate(selected, 1):
            if str(row["image_id"]) in done:
                continue
            record = record_for(model, row, data, device)
            append_jsonl(stream, record)
            written += 1
            if written % log_every == 0:
                elapsed = time.perf_counter() - started
                rate = elapsed / written
                print(f"train pass {index}/{len(selected)} frames, {len(done) + written} written, "
                      f"{elapsed:.0f} s elapsed, {rate:.3f} s/frame, "
                      f"eta {rate * (len(selected) - len(done) - written) / 60:.0f} min", flush=True)
                status(out, "train_stream", index, frames_written=len(done) + written)
        sync(device)
    written_ids = {str(record["image_id"]) for record in read_stream(stream)}
    selection_ids = {str(row["image_id"]) for row in selected}
    selection_count = len(selected)
    if len(selection_ids) != selection_count:
        raise ValueError(f"the TRAIN selection itself repeats image_ids "
                         f"({selection_count} rows, {len(selection_ids)} distinct)")
    if written_ids - selection_ids:
        raise ValueError(f"{stream} holds {len(written_ids - selection_ids)} frames outside this "
                         f"selection; use a fresh --out")
    missing = sorted(selection_ids - written_ids)
    if missing:
        raise ValueError(f"{stream} is missing {len(missing)} of {len(selection_ids)} selected "
                         f"frames (first three: {missing[:3]})")
    rows_total = sum(1 for _ in stream.open("rb"))
    if len(written_ids) != selection_count or rows_total != selection_count:
        raise ValueError(f"{stream} holds {rows_total} rows / {len(written_ids)} distinct "
                         f"image_ids for {selection_count} selected frames; the archive is "
                         f"duplicated or truncated")
    per_genotype = score_genotypes(out, selected, data)
    return {"stream": stream.name, "stream_sha256": file_sha256(stream),
            "frames_total": rows_total, "frames_written_this_run": written,
            "elapsed_seconds_this_run": round(time.perf_counter() - started, 1),
            "per_genotype": per_genotype, "macro": macro_from(per_genotype)}


def score_genotypes(out: Path, selected, data) -> dict:
    """TRAIN genotype-macro COCO mask AP / count MAE of the cap-100 stream (sanity readout)."""
    stream = out / f"quality_{CAP:06d}_S{SCALE}_TRAIN.jsonl"
    records = {str(record["image_id"]): record for record in read_stream(stream)}
    rows_by_id = {str(row["image_id"]): row for row in selected}
    groups = defaultdict(list)
    for image_id in sorted(records):
        groups[records[image_id]["genotype"]].append(image_id)
    result = {}
    for genotype in sorted(groups):
        frames = [rows_by_id[image_id] for image_id in groups[genotype]]
        gt = coco_ground_truth(frames, data)
        predictions, plants = [], defaultdict(list)
        for local_id, row in enumerate(frames, 1):
            record = records[str(row["image_id"])]
            predictions.extend({"image_id": local_id, "category_id": 1, "score": item["score"],
                                "segmentation": item["segmentation"]} for item in record["instances"])
            plants[row["biological_plant_id"]].append(record["count_abs_error"])
        metrics = coco_mask_ap(gt, predictions)
        metrics.update(
            count_MAE=float(np.mean([np.mean(values) for values in plants.values()])),
            frames=len(frames), plants=len(plants), gt_instances=len(gt["annotations"]),
            emitted_instances_mean=float(np.mean([len(records[str(row['image_id'])]['instances'])
                                                  for row in frames])),
            reported_count_mean=float(np.mean([records[str(row['image_id'])]['count']
                                               for row in frames])),
            gt_count_mean=float(np.mean([records[str(row['image_id'])]['gt_count']
                                         for row in frames])))
        result[genotype] = metrics
    return result


def build_report(args, manifest, config, checkpoint_sha, gate, pass_report, selected, ready) -> dict:
    return {
        "version": VERSION, "run_id": args.run_id, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "P1b controller fitting data: TRAIN-domain cap-100 ordered instance stream "
                   "(plan v2 sections 4.1-4.3)",
        "source_benchmark": str(jp_path(BENCHMARK)), "checkpoint": str(manifest["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha, "scale": SCALE, "cap": CAP,
        "model_default_detections_per_img": CAP, "precision": "FP32", "batch": 1,
        "threads": args.threads, "device": args.device,
        "roles_opened": [ROLE], "locked_roles_accessed": [],
        "evaluation_contract": "records written by src/jrtip_pilot.quality()'s own call chain "
                               "(input_bytes -> decode -> infer_image -> restore_output); schema "
                               "identity proven by the development-stream byte gate",
        "instrument_gate": gate, "train_pass": pass_report,
        "selected_population": {
            "frames": len(selected), "plants": len({r["biological_plant_id"] for r in selected}),
            "genotypes": len({r["genotype"] for r in selected})},
        "selected_stream_sha256": digest_json([r["image_id"] for r in selected]),
        "readiness_counts_by_role": {role: ready["counts_by_role"][role] for role in (ROLE,)},
        "formal_36_epoch_run": False, "diagnostic_only": True,
        "backbone_in_sample_on_train": True,
        "limitation": "The P1 checkpoint was fit on these TRAIN frames, so the section 4.3 OOF "
                      "folds remove regressor frame-level over-fitting but not backbone "
                      "in-sample optimism; the P2 three-fold OOF backbones are what address that.",
        "versions": versions(), "machine": args.machine, "note": NOTE}


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.run_id = f"{os.getpid()}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest = json.loads((BENCHMARK / "benchmark_config.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    install_signal_recorder(out)
    device = torch.device(args.device)
    seed_backend(config["seed"], device)
    torch.set_num_threads(args.threads)
    args.machine = machine_snapshot(device)
    rows, data, ready = load_inputs(config["inventory"], config["instances"], config["readiness"],
                                    config["data_root"])
    checkpoint_path = Path(manifest["checkpoint"])
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen benchmark checkpoint changed")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_jp_detector(seed=config["seed"]).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    del state
    gate_path = out / "instrument_gate.json"
    if args.skip_gate:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("passed"):
            raise ValueError(f"--skip-gate needs a stored gate that passed; {gate_path} says "
                             f"{gate.get('passed')!r}")
        gate = dict(gate, reused_by_this_pass=True)
        print(f"reusing the stored instrument gate ({gate['frames_checked']} development frames, "
              f"passed {gate['utc']}); this pass runs no development inference", flush=True)
    else:
        gate = instrument_gate(model, rows, data, device, out, args.gate_frames)
        print(f"gate passed on {gate['frames_checked']} development frames in {gate['seconds']} s", flush=True)
    if args.gate_only:
        print("gate only, TRAIN untouched", flush=True)
        return 0
    selected = select_train(rows, args.max_frames_per_plant)
    write_json(out / "train_stream_config.json", build_report(
        args, manifest, config, checkpoint_sha, gate, {"status": "in_progress"}, selected, ready))
    pass_report = train_pass(model, rows, data, device, out, selected, args.log_every)
    pass_report["status"] = "complete"
    if args.skip_gate:
        pass_report["completion_pass"] = {
            "instrument_gate_reused_from": str(gate_path), "instrument_gate_utc": gate.get("utc"),
            "frames_inferred_by_this_process": pass_report["frames_written_this_run"],
            "device_of_this_pass": str(device),
            "why": ("the archive was written whole by an earlier process that then died in its "
                    "post-pass bookkeeping; this pass re-derives only the readout and the report, "
                    "and the archive's completeness is asserted by train_pass's own id/order checks")}
    report = build_report(args, manifest, config, checkpoint_sha, gate, pass_report, selected, ready)
    write_json(out / "train_stream_report.json", report)
    write_json(out / "train_stream_config.json", report)
    macro = pass_report["macro"]
    print(f"TRAIN cap-100: AP {macro['AP']:.4f}% AP50 {macro['AP50']:.4f}% "
          f"AP75 {macro['AP75']:.4f}% count MAE {macro['count_MAE']:.4f} leaf "
          f"over {pass_report['frames_total']} frames in {pass_report['elapsed_seconds_this_run']:.0f} s",
          flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_train_stream_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gate-frames", dest="gate_frames", type=int, default=48,
                        help="development frames replayed against the archived cap-100 stream")
    parser.add_argument("--gate-only", dest="gate_only", action="store_true",
                        help="run the byte gate and exit without touching TRAIN")
    parser.add_argument("--skip-gate", dest="skip_gate", action="store_true",
                        help="reuse a stored instrument_gate.json instead of replaying development "
                             "frames; for completing a pass whose archive is already whole")
    parser.add_argument("--max-frames-per-plant", dest="max_frames_per_plant", type=int, default=0,
                        help="0 = every valid TRAIN frame; positive = smoke subset")
    parser.add_argument("--log-every", dest="log_every", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.threads <= 32 or args.gate_frames < 1 or args.log_every < 1:
        raise ValueError("threads, gate-frames and log-every must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
