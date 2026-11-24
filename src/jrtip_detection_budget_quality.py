"""Measure the development-quality cost of one output-side detection budget.

Context.  The three-scale resolution pilot showed that dropping the long edge from 640 to
320/480 costs 8.39/2.02 percentage points of development macro AP while the measured request
time barely moves, and the bottleneck decomposition attributed most of the forward stream to
torchvision's instance-mask post-processing rather than to the convolutional backbone.  That
moves the interesting control knob from input resolution to the number of detections the head
is allowed to emit (roi_heads.detections_per_img): with the default budget the head returns
about 100 instances per frame while only about 9 exceed score 0.5, so capping the budget
removes post-processing work that never reached a reported leaf.

This script measures the quality side of that trade-off on the frozen P1 checkpoint, with the
frozen development-quality contract from src/jrtip_pilot.quality(): 2,947 valid development
frames, genotype-macro COCO mask AP@[.5:.95] recomputed per collection, count MAE aggregated
frame -> plant -> genotype -> equal dev domain.  The budget sweep always runs at long edge 640,
the reference operating point, so nothing here re-litigates resolution.

Predeclared decision bars (fixed before any measurement in this file runs, and the same bars
the resolution protocol used): a budget is quality-acceptable when its AP drop versus the
budget-100 control is <=1.0 percentage point AND its count MAE increase is <=0.2 leaf.  Budget
100 is also re-measured here as a reproducibility control against the frozen benchmark's
quality.json; a control that does not reproduce is a finding, not noise.

One process per budget, by intent.  The first attempt at this sweep ran all five budgets in a
single process and was terminated by an external console control event (returncode
0xC000013A) 2.7 minutes into the control pass, which discarded the whole sweep.  This script
therefore takes the budgets to measure via --caps and writes one self-contained
budget_report_NNNNNN.json per budget, plus a progress record after each budget, so a launch can
be split into five queue jobs and an interruption costs at most one budget.  Comparison against
the control is deliberately NOT done here; src/jrtip_detection_budget_merge.py is the only place
that arithmetic lives, and it records whether the budgets it joined came from one process or
several.

Development roles only.  C_POLICY/T_ID/T_GENO/T_EXT are never loaded, never used for
selection, and never evaluated here.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_pilot import (  # noqa: E402
    NOTE, build_jp_detector, digest_json, file_sha256, load_inputs, quality, run_lock,
    seed_backend, status, versions, COCO_V1_SHA256,
)

VERSION = "jrtip-detection-budget-quality-v2"
SCALE = 640
CONTROL_CAP = 100
AP_DROP_LIMIT_PP, COUNT_INCREASE_LIMIT_LEAF = 1.0, 0.2
MACRO_KEYS = ("AP", "AP50", "AP75", "count_MAE")
EVALUATION_CONTRACT = ("src/jrtip_pilot.quality(): select_dev over V_SOURCE/V_SHIFT valid frames, "
                       "genotype-macro COCO mask AP recomputed per collection, count MAE "
                       "aggregated frame -> plant -> genotype -> equal dev domain")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def macro(report, key):
    return float(report["scales"][str(SCALE)]["equal_domain_macro"][key])


def per_role(report, key):
    return {role: float(values[key]) for role, values
            in report["scales"][str(SCALE)]["role_macro"].items()}


def install_signal_recorder(out: Path) -> str:
    """Record an external console control event before it terminates the process.

    The first sweep died with returncode 0xC000013A and an empty stdout log, which left no way
    to tell a delivered CTRL_C/CTRL_BREAK event from a hard kill.  The handlers below write a
    one-shot marker with the signal name and the per-budget stream sizes at that moment, then
    let the normal exception path print a traceback into the run log.
    """
    seen = ""

    def handler(signum, frame):
        nonlocal seen
        name = signal.Signals(signum).name
        seen = f"{name} at {utc()}"
        streams = {}
        for path in sorted(out.glob("quality_*_S*.jsonl")):
            try:
                stat = path.stat()
                streams[path.name] = {"bytes": stat.st_size,
                                      "modified_utc": utc_of(stat.st_mtime),
                                      "lines": sum(1 for _ in path.open("rb"))}
            except OSError as exc:  # pragma: no cover - forensic best effort
                streams[path.name] = {"error": repr(exc)}
        write_json(out / "signal_seen.json", {
            "signal": int(signum), "signal_name": name, "recorded_utc": utc(), "pid": os.getpid(),
            "note": "Handler installed by this script; the terminating event came from outside.",
            "streams": streams})
        raise KeyboardInterrupt(name)

    for name in ("SIGBREAK", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, handler)
    return seen


def utc_of(stamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))


def common_payload(args, source, manifest, checkpoint_sha, checkpoint_path, original, run_id):
    reference = json.loads((source / "quality.json").read_text(encoding="utf-8"))
    return {
        "version": VERSION, "run_id": run_id, "source_benchmark": str(source),
        "source_benchmark_config_sha256": digest_json(manifest),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "coco_weights_sha256_constant": COCO_V1_SHA256, "scale": SCALE,
        "model_default_detections_per_img": original, "control_cap": CONTROL_CAP,
        "predeclared_tolerances": {"AP_drop_vs_control_pp": AP_DROP_LIMIT_PP,
                                   "count_MAE_increase_vs_control_leaf": COUNT_INCREASE_LIMIT_LEAF},
        "device": args.device, "threads": args.threads, "precision": "FP32",
        "evaluation_contract": EVALUATION_CONTRACT,
        "max_frames_per_plant": args.max_frames, "smoke_limited": bool(args.max_frames),
        "formal_36_epoch_run": False, "locked_roles_accessed": [], "diagnostic_only": True,
        "frozen_benchmark_control": {key: macro(reference, key) for key in MACRO_KEYS},
        "frozen_benchmark_stream_sha256": reference.get("selected_stream_sha256"),
        "note": NOTE}


def run(args):
    source = Path(args.benchmark).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Detection-budget output must start empty: {out}")
    manifest = json.loads((source / "benchmark_config.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    if not caps or len(set(caps)) != len(caps) or any(not 1 <= c <= 1000 for c in caps):
        raise ValueError("caps must be unique integers in 1..1000")
    run_id = f"{os.getpid()}-{utc()}"
    install_signal_recorder(out)
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
    original = model.roi_heads.detections_per_img
    if original != CONTROL_CAP:
        raise ValueError(f"Control budget {CONTROL_CAP} is not the model default {original}")
    payload = common_payload(args, source, manifest, checkpoint_sha, checkpoint_path, original, run_id)
    write_json(out / "budget_quality_config.json", dict(payload, budgets=caps))

    with run_lock(out):
        for budget in caps:
            model.roi_heads.detections_per_img = budget
            status(out, "budget_quality", 0, budget=budget, scale=SCALE, run_id=run_id)
            report = quality(model, rows, data, [SCALE], device, max_frames=args.max_frames,
                             out=out, step=budget)
            model.roi_heads.detections_per_img = original
            record = dict(payload, budget=budget, generated_utc=utc(),
                          signal_seen_before_completion=payload_signal(out),
                          selected_stream_sha256=report["selected_stream_sha256"],
                          selected_population=report["selected_population"],
                          elapsed_seconds=report["elapsed_seconds"],
                          macro={key: macro(report, key) for key in MACRO_KEYS},
                          macro_by_role={key: per_role(report, key) for key in MACRO_KEYS},
                          per_genotype=report["scales"][str(SCALE)]["per_genotype"],
                          quality_jsonl=f"quality_{budget:06d}_S{SCALE}.jsonl",
                          quality_jsonl_sha256=str(file_sha256(
                              out / f"quality_{budget:06d}_S{SCALE}.jsonl")),
                          quality_jsonl_lines=sum(1 for _ in
                                                  (out / f"quality_{budget:06d}_S{SCALE}.jsonl")
                                                  .open("rb")))
            write_json(out / f"budget_report_{budget:06d}.json", record)
            done = sorted(p.name for p in out.glob("budget_report_*.json"))
            write_json(out / "budget_quality_partial.json", {
                "version": VERSION, "run_id": run_id, "budgets_requested": caps,
                "budgets_completed": [budget for budget in caps
                                      if (out / f"budget_report_{budget:06d}.json").exists()],
                "reports": done, "updated_utc": utc()})
            print(f"budget {budget}: AP {record['macro']['AP']:.4f}% "
                  f"AP50 {record['macro']['AP50']:.4f}% "
                  f"count MAE {record['macro']['count_MAE']:.4f} leaf "
                  f"({record['quality_jsonl_lines']} frames) in {record['elapsed_seconds']:.0f} s",
                  flush=True)
    print("wrote", ", ".join(str(out / f"budget_report_{b:06d}.json") for b in caps), flush=True)
    return 0


def payload_signal(out: Path) -> str:
    path = out / "signal_seen.json"
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("signal_name", "")
    except (OSError, ValueError):
        return "unreadable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=str(ROOT / "jrtip/baseline_benchmark_v1"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/detection_budget_quality_v1/cap00100"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--caps", default=str(CONTROL_CAP))
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0,
                        help="cap frames per plant for a smoke check; recorded as smoke_limited")
    args = parser.parse_args()
    if not 1 <= args.threads <= 32:
        raise ValueError("threads must be 1..32")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
