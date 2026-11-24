"""Join per-budget development-quality reports and apply the predeclared decision bars.

The producer (src/jrtip_detection_budget_quality.py) measures one or more output-side detection
budgets on the frozen P1 checkpoint and writes one budget_report_NNNNNN.json per budget, but it
performs no comparison.  This script is the only place the comparison arithmetic lives, so the
same numbers are produced whether the budgets came from one process or from five queue jobs.

Rules fixed in advance, none of them tuned to the data:
  * the control is budget 100, and it must be present;
  * every budget must have been evaluated on the identical development stream, and that stream
    must match the frozen benchmark's stream, otherwise the sweep is void rather than mixed;
  * a budget is quality-acceptable when AP drop versus control <= 1.0 percentage point AND count
    MAE increase versus control <= 0.2 leaf;
  * the control pass is additionally compared against the frozen benchmark's quality.json, and a
    non-reproducing control is reported as a finding.

No model is loaded, no dataset is read, and no device is touched: the predeclared bars and the
macro field names are imported from the producer module so the two files cannot drift, but this
script only turns budget_report_NNNNNN.json files into budget_quality.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_detection_budget_quality import (  # noqa: E402
    AP_DROP_LIMIT_PP, CONTROL_CAP, COUNT_INCREASE_LIMIT_LEAF, MACRO_KEYS, SCALE, VERSION, utc,
)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def collect(roots):
    reports, problems = {}, []
    for root in roots:
        for path in sorted(Path(root).rglob("budget_report_*.json")):
            record = load_json(path)
            budget = int(record["budget"])
            if budget in reports:
                problems.append(f"budget {budget} reported twice: "
                                f"{reports[budget]['_path']} and {path}")
                continue
            record["_path"] = str(path)
            if int(record["scale"]) != SCALE:
                problems.append(f"{path.name}: scale {record['scale']} is not the reference {SCALE}")
            stream = Path(path).parent / record["quality_jsonl"]
            if not stream.exists():
                problems.append(f"{path.name}: per-frame stream {stream.name} is gone")
            elif record.get("quality_jsonl_sha256") != sha256(stream):
                problems.append(f"{path.name}: quality jsonl no longer matches its recorded digest")
            elif record.get("quality_jsonl_lines") != sum(1 for _ in stream.open("rb")):
                problems.append(f"{path.name}: quality jsonl line count differs from the report")
            reports[budget] = record
    return reports, problems


def run(roots, out, expect):
    reports, problems = collect(roots)
    if CONTROL_CAP not in reports:
        problems.append(f"control budget {CONTROL_CAP} is missing; the sweep cannot be judged")
    streams = {budget: record["selected_stream_sha256"] for budget, record in reports.items()}
    frozen_streams = {record.get("frozen_benchmark_stream_sha256") for record in reports.values()}
    stream_consistent = (len(set(streams.values())) <= 1 and len(frozen_streams) == 1
                         and (not streams or next(iter(streams.values()))
                              == next(iter(frozen_streams))))
    checkpoints = {record["checkpoint_sha256"] for record in reports.values()}
    if len(checkpoints) > 1:
        problems.append("reports disagree on the checkpoint digest: " + ", ".join(sorted(checkpoints)))
    run_ids = sorted({record.get("run_id", "unknown") for record in reports.values()})
    control = reports.get(CONTROL_CAP)
    rows = {}
    for budget in sorted(reports):
        record = reports[budget]
        if control is None:
            drop = increase = None
            acceptable = None
        else:
            drop = control["macro"]["AP"] - record["macro"]["AP"]
            increase = record["macro"]["count_MAE"] - control["macro"]["count_MAE"]
            acceptable = bool(drop <= AP_DROP_LIMIT_PP and increase <= COUNT_INCREASE_LIMIT_LEAF)
        rows[str(budget)] = {
            "budget": budget, **{key: record["macro"][key] for key in MACRO_KEYS},
            "AP_by_role": record["macro_by_role"]["AP"],
            "count_MAE_by_role": record["macro_by_role"]["count_MAE"],
            "AP_drop_vs_control_pp": drop, "count_MAE_increase_vs_control_leaf": increase,
            "quality_acceptable": acceptable,
            "elapsed_seconds": record["elapsed_seconds"],
            "frames_evaluated": record["quality_jsonl_lines"],
            "selected_population": record["selected_population"],
            "selected_stream_sha256": record["selected_stream_sha256"],
            "run_id": record.get("run_id"), "report_path": record["_path"],
            "quality_jsonl": str(Path(record["_path"]).parent / record["quality_jsonl"]),
            "quality_jsonl_sha256": record.get("quality_jsonl_sha256"),
            "per_genotype": record["per_genotype"]}
    missing = [cap for cap in expect if cap not in reports]
    result = {
        "version": VERSION, "merged_utc": utc(), "scale": SCALE,
        "controls_and_bars": {"control_budget": CONTROL_CAP,
                              "AP_drop_limit_pp": AP_DROP_LIMIT_PP,
                              "count_MAE_increase_limit_leaf": COUNT_INCREASE_LIMIT_LEAF},
        "budgets_measured": sorted(int(b) for b in rows), "budgets_missing": missing,
        "same_process_for_all_budgets": bool(len(run_ids) == 1),
        "run_ids": run_ids,
        "checkpoint_sha256_identical_across_reports": bool(len(checkpoints) <= 1),
        "checkpoints": sorted(checkpoints),
        "dev_stream_identical_across_budgets_and_matches_frozen_benchmark": bool(stream_consistent),
        "selected_stream_sha256": next(iter(streams.values())) if streams else None,
        "control_vs_frozen_benchmark": (None if control is None else {
            "AP_this_pass": control["macro"]["AP"], "AP_frozen": control["frozen_benchmark_control"]["AP"],
            "difference_pp": control["macro"]["AP"] - control["frozen_benchmark_control"]["AP"],
            "AP50_difference_pp": control["macro"]["AP50"] - control["frozen_benchmark_control"]["AP50"],
            "count_MAE_this_pass": control["macro"]["count_MAE"],
            "count_MAE_frozen": control["frozen_benchmark_control"]["count_MAE"],
            "count_MAE_difference_leaf": control["macro"]["count_MAE"] - control["frozen_benchmark_control"]["count_MAE"],
            "control_reproduces_frozen_benchmark": bool(
                abs(control["macro"]["AP"] - control["frozen_benchmark_control"]["AP"]) <= 1e-6
                and abs(control["macro"]["count_MAE"] - control["frozen_benchmark_control"]["count_MAE"]) <= 1e-6)}),
        "budgets": rows,
        "acceptable_budgets": [int(b) for b, v in sorted(rows.items()) if v["quality_acceptable"]],
        "lowest_acceptable_budget_below_control": min(
            (int(b) for b, v in rows.items()
             if v["quality_acceptable"] and int(b) < CONTROL_CAP), default=None),
        "smoke_limited": bool(any(record["smoke_limited"] for record in reports.values())),
        "integrity_problems": problems,
        "verdict": None,
        "scope_limits": [
            "Development roles only; C_POLICY/T_ID/T_GENO/T_EXT never loaded or evaluated.",
            "Single seed 43 pilot checkpoint at 1000 optimizer updates, long edge 640 only; this is "
            "not the formal 36-epoch schedule.",
            "Point estimates on 2,947 development frames; no confirmatory confidence interval, no "
            "multiple-comparison control, and no locked-test evidence.",
            "Time savings per budget come from the separate bottleneck-decomposition run, which "
            "measured 40-frame medians on a device shared with desktop CUDA contexts; this merge "
            "reuses those timings as context and does not re-measure them.",
            "Counts are soft-mask instance counts of a fixed top-down field of view; no physical "
            "area or biomass calibration is implied."],
        "producer_version": sorted({record["version"] for record in reports.values()}),
        }
    if problems or control is None or not stream_consistent or missing:
        result["verdict"] = "incomplete_or_inconsistent"
    else:
        usable = [budget for budget in result["acceptable_budgets"] if budget < CONTROL_CAP]
        result["verdict"] = ("quality_preserved_below_control_at_budgets: "
                             + ", ".join(str(budget) for budget in usable)
                             if usable else "no_budget_below_the_control_passes_both_bars")
    write_json(out / "budget_quality.json", result)
    print(json.dumps({"budgets": {b: {k: v[k] for k in ("AP", "count_MAE", "AP_drop_vs_control_pp",
                                                        "count_MAE_increase_vs_control_leaf",
                                                        "quality_acceptable")}
                                   for b, v in sorted(rows.items())},
                      "verdict": result["verdict"],
                      "control_reproduces_frozen_benchmark":
                          (result["control_vs_frozen_benchmark"] or {}).get(
                              "control_reproduces_frozen_benchmark"),
                      "integrity_problems": problems}, indent=2, ensure_ascii=False), flush=True)
    print("wrote", out / "budget_quality.json", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", default=str(ROOT / "jrtip/detection_budget_quality_v1"),
                        help="directory (or comma-separated directories) holding budget_report_*.json")
    parser.add_argument("--out", default=str(ROOT / "jrtip/detection_budget_quality_v1"))
    parser.add_argument("--expect", default="100,50,20,10,5",
                        help="budgets the sweep promised; any that are absent are listed as missing")
    args = parser.parse_args()
    roots = [Path(item).resolve() for item in args.roots.split(",") if item.strip()]
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing report directories: " + ", ".join(missing))
    expect = [int(item) for item in args.expect.split(",") if item.strip()]
    run(roots, Path(args.out).resolve(), expect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
