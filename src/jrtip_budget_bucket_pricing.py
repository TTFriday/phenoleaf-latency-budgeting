"""Offline pricing of per-frame output-budget controllers (P1c, quality side, CPU only).

Why this script exists
----------------------
``src/jrtip_budget_rule_timing.py`` measures *service time* for controllers that choose each
frame's output budget from that frame's own post-NMS scores.  Plan v2 section 3.2 forbids pairing
the time of one arm with the quality of another, so every timed arm name also needs a quality
number computed under the contract of the frozen budget table.  The prefix identity lemma
(results doc section 9) makes that possible without a GPU pass: for a 640-input run whose only
change is the output cap, the cap-k output is the length-k prefix of the cap-100 output (lengths,
order and scores bitwise identical), so a controller's emitted budgets can be replayed against
the archived cap-100 stream and re-scored offline in about two CPU minutes per arm.

Single source of truth
----------------------
* Budget policy semantics come from the timing module (``ARMS``, ``budget_for``) - an arm name
  resolves to one policy in both tables.
* Aggregation arithmetic comes from ``src/jrtip_budget_policy.py`` (``evaluate_arm``), i.e. the
  same code path that produced every archived static and rule arm.

What is deliberately not done here
---------------------------------
* No GPU, no model, no inference.  Seconds appear only through the archived cost-model
  regression, flagged as expectation-only; section 6 requires the measured service times and
  forbids substituting this slope for them.
* No new decision variable.  ``c`` is the archived per-frame reported count (emitted instances
  with score >= 0.5), which is exactly what the wrapper reads, including the documented
  saturation at the 100-candidate cap.
* Locked roles stay closed: only the dev roles are regrouped, and the run asserts the frame
  count matches the frozen dev-stream contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_policy import (  # noqa: E402
    AP_DROP_LIMIT_PP, COUNT_INCREASE_LIMIT_LEAF, COUNT_SCORE, Ledger, arm_records, evaluate_arm,
    k_static, per_frame_consistency, read_stream, regroup, score_order_ok,
)
from src.jrtip_budget_rule_timing import (  # noqa: E402
    ARMS, BUCKET_GRID, CONTROL, QUALITY_ROOT, SCALE, bucket_of, budget_for,
)
from src.jrtip_pilot import load_inputs, select_dev  # noqa: E402
from src.verify_jrtip_budget_quality import DEV_ROLES, METRIC_KEYS, aggregate  # noqa: E402

VERSION = "jrtip-budget-bucket-pricing-v1"
RULE_MODES = ("sync", "readfull", "bucket")
QUALITY_KEYS = ("macro", "macro_by_role", "ap_drop_vs_control_pp",
                "count_MAE_increase_vs_control_leaf", "gate_ap_drop_within_1pp",
                "gate_count_MAE_within_0_2_leaf", "frames_with_reported_count_truncated",
                "worst_genotype", "mask_work_reduction_vs_control", "budget_mean", "budget_median",
                "budget_p90", "budget_max", "frames", "elapsed_seconds")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio_tag(ratio: float) -> str:
    """1.25 -> '1p25', 1.5 -> '1p5': the token the timing module uses inside arm names."""
    return f"{float(ratio):g}".replace(".", "p")


def rule_token(rule) -> str:
    kind, parameter = rule
    return f"c_plus_{parameter:g}" if kind == "c_plus_m" else f"ceil_{ratio_tag(parameter)}"


def arm_name(mode: str, rule, floor: int) -> str:
    return f"{mode}_{rule_token(rule)}_f{int(floor)}"


def arm_definition(mode: str, rule, floor: int) -> str:
    kind, parameter = rule
    raw = f"count+{parameter:g}" if kind == "c_plus_m" else f"ceil({ratio_tag(parameter)}*count)"
    core = f"clip({raw}, floor={floor}, cap={CONTROL})"
    if mode == "bucket":
        # Bucketing is applied to the decided k, after the floor, and can only raise it.
        core = f"grid_ceiling({core}, grid={list(BUCKET_GRID)})"
    return f"k = {core}"


def make_arm(mode: str, rule, floor: int) -> dict:
    """Build an arm dict in the timing module's schema so budget_for() consumes it verbatim."""
    return {"name": arm_name(mode, rule, floor), "mode": mode, "rule": [rule[0], rule[1]],
            "floor": int(floor), "family": f"{mode}_{rule_token(rule)}",
            "definition": arm_definition(mode, rule, floor)}


def emitted_vector(arm: dict, records) -> list[int]:
    return [budget_for(arm, int(record["count"])) for record in records]


def vector_digest(values) -> str:
    return hashlib.sha256(json.dumps([int(v) for v in values]).encode("ascii")).hexdigest()[:16]


def budget_histogram(values) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(int(value))] = counts.get(str(int(value)), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def cost_projection(work: float, model: dict):
    slope = model.get("slope_seconds_per_instance_per_frame_mean")
    intercept = model.get("intercept_seconds")
    if slope is None or intercept is None:
        return None
    return round(float(slope) * float(work) + float(intercept), 1)


def load_static_comparators(paths, ledger) -> dict:
    """Archived static arms (work, quality) so every rule arm is judged against them verbatim."""
    table: dict[str, dict] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            ledger.note("static_comparator_source_present", False, f"missing {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("smoke_limited"):
            ledger.note("static_comparator_source_is_full_stream", False, f"{path.name} is smoke-limited")
            continue
        for arm in payload.get("arms", []):
            if arm.get("kind") != "static":
                continue
            try:
                budget = int(arm["name"])
            except (TypeError, ValueError):
                continue
            table[str(budget)] = {
                "budget": budget,
                "work_masks_per_frame": arm["mean_instances_masked_per_frame"],
                "ap_drop_vs_control_pp": arm["ap_drop_vs_control_pp"],
                "count_MAE_increase_vs_control_leaf": arm["count_MAE_increase_vs_control_leaf"],
                "macro": {key: arm["macro"][key] for key in METRIC_KEYS if key in arm.get("macro", {})},
                "source": path.name,
            }
    return dict(sorted(table.items(), key=lambda item: -int(item[0])))


def nearest_statics(work: float, statics: dict, window: float) -> list[dict]:
    """Statics whose masked-instance work lies within +/- window of an arm's work."""
    out = []
    for entry in statics.values():
        if entry["budget"] == CONTROL:
            continue
        delta = float(entry["work_masks_per_frame"]) - float(work)
        if abs(delta) <= window:
            out.append({**entry, "work_delta_vs_arm": round(delta, 3),
                        "ap_drop_minus_arm_pp": round(
                            float(entry["ap_drop_vs_control_pp"]), 4)})
    return sorted(out, key=lambda item: abs(item["work_delta_vs_arm"]))


def collect_arms(args) -> list[dict]:
    """Timing-module rule arms, plus fill arms that land in the equal-work comparator gap."""
    arms: list[dict] = []
    seen: set[str] = set()
    for name, arm in ARMS.items():
        if arm.get("mode") not in RULE_MODES:
            continue
        if args.only_named and name not in args.only_named:
            continue
        arms.append(dict(arm, name=name))
        seen.add(name)
    if not args.only_named and args.fill:
        offsets = [int(v) for v in args.fill_m.split(",") if v.strip()]
        ratios = [float(v) for v in args.fill_ratio.split(",") if v.strip()]
        modes = ("sync", "bucket") if args.with_bucket_variants else ("sync",)
        for floor in [int(v) for v in args.fill_floor.split(",") if v.strip()]:
            for rule in [("c_plus_m", o) for o in offsets] + [("ceil_ratio", r) for r in ratios]:
                for mode in modes:
                    arm = make_arm(mode, rule, floor)
                    if arm["name"] not in seen:
                        arms.append(arm)
                        seen.add(arm["name"])
    return arms


def evaluate_control(blocks, data, rows_by_image_id, ledger):
    """Control reference: the cap-100 stream with everything kept, same aggregate() call as arms."""
    _, role_macro, equal_domain, _, _, _ = aggregate(
        {key: arm_records(records, lambda record: CONTROL) for key, records in blocks.items()},
        data, rows_by_image_id, CONTROL, ledger, False)
    return role_macro, equal_domain


def price_arms(arms, base, blocks, data, rows_by_image_id, ledger, control, cost, statics, args):
    entries: list[dict] = []
    by_name: dict[str, dict] = {}
    vectors: dict[str, list[int]] = {}
    first_by_digest: dict[str, dict] = {}
    for arm in arms:
        vector = emitted_vector(arm, base)
        digest = vector_digest(vector)
        entry = {
            "name": arm["name"], "mode": arm["mode"], "floor": int(arm.get("floor", 0)),
            "rule": list(arm.get("rule") or []), "definition": arm["definition"],
            "from_timing_module": arm["name"] in ARMS,
            "budget_vector_digest": digest,
            "work_masks_per_frame": round(float(np.mean(vector)), 3),
            "budget_histogram": budget_histogram(vector),
            "frames_at_floor": int(sum(1 for v in vector if v <= int(arm.get("floor", 0)))),
            "frames_at_cap": int(sum(1 for v in vector if v >= CONTROL)),
            "projected_wall_seconds_cost_model": cost_projection(float(np.mean(vector)), cost),
            "projection_is_expectation_only": True,
        }
        if digest in first_by_digest:
            twin = first_by_digest[digest]
            entry["reuses_quality_of"] = twin["name"]
            entry["note"] = ("identical emitted-budget vector, so the prefix - and therefore the "
                             "COCO re-scoring - is identical by construction; not recomputed")
            for key in QUALITY_KEYS:
                if key in twin:
                    entry[key] = twin[key]
            print(f"{entry['name']:26s} reuses {twin['name']}", flush=True)
        else:
            result = evaluate_arm(entry["name"], entry["mode"],
                                  (lambda record, arm=arm: budget_for(arm, int(record["count"]))),
                                  blocks, data, rows_by_image_id, ledger, control)
            entry.update({key: result[key] for key in QUALITY_KEYS if key in result})
            entry["ap_drop_by_genotype"] = result["ap_drop_by_genotype"]
            first_by_digest[digest] = entry
            print(f"{entry['name']:26s} work {entry['work_masks_per_frame']:7.3f} "
                  f"AP {entry['macro']['AP']:.4f} drop {-entry['ap_drop_vs_control_pp']:+.4f} pp "
                  f"count {-entry['count_MAE_increase_vs_control_leaf']:+.4f} leaf "
                  f"[{entry['elapsed_seconds']}s]", flush=True)
        if entry["mode"] in ("sync", "readfull"):
            ledger.note(f"arm_{entry['name']}_never_truncates_reported_count",
                        entry["frames_with_reported_count_truncated"] == 0,
                        f"{entry['frames_with_reported_count_truncated']} frames truncated")
        entry["comparators_within_work_window"] = nearest_statics(
            entry["work_masks_per_frame"], statics, args.work_window)
        if args.strict_gates:
            ledger.note(f"arm_{entry['name']}_ap_drop_within_1pp", entry["gate_ap_drop_within_1pp"],
                        f"drop {-entry['ap_drop_vs_control_pp']:.4f} pp")
            ledger.note(f"arm_{entry['name']}_count_MAE_within_0_2_leaf",
                        entry["gate_count_MAE_within_0_2_leaf"],
                        f"increase {-entry['count_MAE_increase_vs_control_leaf']:.4f} leaf")
        entries.append(entry)
        by_name[entry["name"]] = entry
        vectors[entry["name"]] = vector
    return entries, by_name, vectors


def bucket_vs_decision_pairs(entries, by_name, vectors, cost) -> list[dict]:
    """What rounding the budget up to a static grid costs, arm by arm, against its decision twin."""
    slope = cost.get("slope_seconds_per_instance_per_frame_mean")
    pairs = []
    for entry in entries:
        if entry["mode"] != "bucket":
            continue
        stem = entry["name"].split("bucket_", 1)
        if len(stem) != 2:
            continue
        for mode in ("sync", "readfull"):
            twin_name = f"{mode}_{stem[1]}"
            if twin_name not in by_name:
                continue
            left, right = vectors[twin_name], vectors[entry["name"]]
            delta = [int(b) - int(a) for a, b in zip(left, right)]
            pairs.append({
                "bucket_arm": entry["name"], "decision_arm": twin_name,
                "frames_where_bucket_emits_more": int(sum(1 for d in delta if d > 0)),
                "mean_extra_masked_instances_per_frame": round(float(np.mean(delta)), 3),
                "max_extra_masked_instances": int(max(delta)),
                "ap_drop_pp_bucket_minus_decision": round(
                    float(entry["ap_drop_vs_control_pp"])
                    - float(by_name[twin_name]["ap_drop_vs_control_pp"]), 4),
                "cost_model_seconds_bucket_minus_decision":
                    None if slope is None else round(float(slope) * float(np.mean(delta)), 2),
                "quality_identical_to_decision_arm": bool(delta and all(d == 0 for d in delta)) or (
                    entry["macro"] == by_name[twin_name]["macro"]),
            })
    return pairs


def run(args):
    started = time.perf_counter()
    ledger = Ledger()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    strict = args.max_frames == 0

    # ---- 1. frozen inputs ----------------------------------------------------------------------
    control_stream = QUALITY_ROOT / f"cap{CONTROL:05d}" / f"quality_{CONTROL:06d}_S{SCALE}.jsonl"
    if not control_stream.exists():
        raise FileNotFoundError(f"missing control stream: {control_stream}")
    config_path = next((QUALITY_ROOT / f"cap{c:05d}" / "budget_quality_config.json"
                        for c in (CONTROL, 50, 20, 15, 5)
                        if (QUALITY_ROOT / f"cap{c:05d}" / "budget_quality_config.json").exists()), None)
    if config_path is None:
        raise FileNotFoundError("no budget_quality_config.json under the archived budget passes")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    benchmark = Path(config["source_benchmark"])
    manifest = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))
    training = manifest["training_config"]
    rows, data, _ = load_inputs(training["inventory"], training["instances"], training["readiness"],
                                training["data_root"])
    stream = select_dev(rows, args.max_frames)
    rows_by_image_id = {row["image_id"]: row for row in stream}
    base = [record for record in read_stream(control_stream) if record["image_id"] in rows_by_image_id]
    blocks = regroup(base, rows_by_image_id, ledger, CONTROL, stream)
    per_frame_consistency(base, CONTROL, ledger)
    violations = score_order_ok(base)
    ledger.note("control_stream_scores_are_non_increasing", violations == 0, f"{violations} frames")
    if strict:
        ledger.note("stream_is_the_frozen_dev_contract", len(stream) == args.expected_frames,
                    f"{len(stream)} dev frames, expected {args.expected_frames}")
    ledger.note("no_gpu_used_by_this_script", not args.allow_gpu, "CPU re-scoring only")

    control_report = json.loads((QUALITY_ROOT / f"cap{CONTROL:05d}" /
                                 f"budget_report_{CONTROL:06d}.json").read_text(encoding="utf-8"))
    control_masks = float(np.mean([len(record["instances"]) for record in base]))
    control_role, control_domain = evaluate_control(blocks, data, rows_by_image_id, ledger)
    if strict:
        worst = max(abs(float(control_report["macro"][key]) - float(control_domain[key]))
                    for key in METRIC_KEYS)
        ledger.note("control_arm_reproduces_frozen_baseline", worst <= args.reproduce_tol,
                    f"max |diff| {worst:.3e} pp, control macro {control_domain}")
    control = {"macro": control_domain, "macro_by_role": control_role,
               "per_genotype": control_report["per_genotype"],
               "mean_instances_masked_per_frame": control_masks}

    # ---- 2. wiring: re-price an archived budget through this exact path ------------------------
    wiring = None
    if args.wiring_budget > 0:
        budget = int(args.wiring_budget)
        result = evaluate_arm(f"static_{budget}", "static_wiring_check",
                              lambda record, budget=budget: k_static(record, budget),
                              blocks, data, rows_by_image_id, ledger, control)
        report_path = QUALITY_ROOT / f"cap{budget:05d}" / f"budget_report_{budget:06d}.json"
        if report_path.exists():
            measured = json.loads(report_path.read_text(encoding="utf-8"))
            diffs = {key: round(float(measured["macro"][key]) - float(result["macro"][key]), 6)
                     for key in METRIC_KEYS}
            max_diff = max(abs(value) for value in diffs.values())
            # The archived GPU reports carry quality but no work field: the mask work of that pass is
            # recomputed from its own per-frame stream, exactly the way control_masks is derived above,
            # so the two sides of the comparison count instances with the same definition.
            work_key = "mean_instances_masked_per_frame"
            measured_work = measured.get(work_key)
            work_source = "archived_report_field"
            if measured_work is None:
                arm_stream = QUALITY_ROOT / f"cap{budget:05d}" / f"quality_{budget:06d}_S{SCALE}.jsonl"
                if arm_stream.exists():
                    frames = [record for record in read_stream(arm_stream)
                              if record["image_id"] in rows_by_image_id]
                    if len(frames) == len(base):
                        measured_work = float(np.mean([len(record["instances"]) for record in frames]))
                        work_source = f"recomputed_from:{arm_stream.name}"
                    else:
                        ledger.note("wiring_work_recompute_frame_count_matches", False,
                                    f"{arm_stream.name} held {len(frames)} scored frames, "
                                    f"the contract has {len(base)}")
            if measured_work is None:
                ledger.note("wiring_work_available", False,
                            f"neither {work_key} nor {arm_stream.name} could price cap {budget}")
                work_diff = None
            else:
                work_diff = abs(float(measured_work) - float(result[work_key]))
            wiring = {"budget": budget,
                      "measured": {key: measured["macro"][key] for key in METRIC_KEYS},
                      "offline_prefix": {key: result["macro"][key] for key in METRIC_KEYS},
                      "difference_pp": diffs, "max_abs_difference_pp": max_diff,
                      "work_source": work_source,
                      "work_measured": measured_work,
                      "work_offline": result[work_key],
                      "work_abs_difference": None if work_diff is None else round(work_diff, 4)}
            if strict:
                ledger.note(f"wiring_cap{budget}_reproduces_gpu_pass", max_diff <= args.reproduce_tol,
                            f"max |diff| {max_diff:.3e} pp")
                if work_diff is not None:
                    ledger.note(f"wiring_cap{budget}_work_matches", work_diff <= 0.01,
                                f"|diff| {work_diff:.4f} instances/frame")
            print(f"wiring cap {budget}: max |diff| {max_diff:.3e} pp, work diff "
                  f"{'n/a' if work_diff is None else format(work_diff, '.4f')}", flush=True)
        else:
            ledger.note("wiring_budget_archived_report_present", False, f"missing {report_path}")

    # ---- 3. price the arms ---------------------------------------------------------------------
    static_paths = [p.strip() for p in args.policy_eval.split(",") if p.strip()]
    statics = load_static_comparators(static_paths, ledger)
    cost = {}
    for path in static_paths:
        if Path(path).exists():
            cost = json.loads(Path(path).read_text(encoding="utf-8")).get("cost_model", {}) or {}
            if cost.get("slope_seconds_per_instance_per_frame_mean") is not None:
                break
    arms = collect_arms(args)
    print(f"pricing {len(arms)} arms on {len(base)} frames", flush=True)
    entries, by_name, vectors = price_arms(arms, base, blocks, data, rows_by_image_id, ledger,
                                           control, cost, statics, args)

    # ---- 4. frontier summary -------------------------------------------------------------------
    passing = [e for e in entries if e.get("gate_ap_drop_within_1pp")
               and e.get("gate_count_MAE_within_0_2_leaf")]
    frontier = sorted(passing, key=lambda e: e["work_masks_per_frame"])
    payload = {
        "version": VERSION, "generated_utc": utc(), "out": str(out),
        "control_stream": str(control_stream), "control_stream_sha256": sha256_file(control_stream),
        "source_benchmark": str(benchmark),
        "checkpoint_sha256": manifest.get("checkpoint_sha256") or config.get("checkpoint_sha256"),
        "scale": SCALE, "count_score_threshold": COUNT_SCORE, "bucket_grid": list(BUCKET_GRID),
        "gpu_used": False, "locked_roles_accessed": [], "roles_opened": sorted(DEV_ROLES),
        "frames": len(base), "max_frames": args.max_frames,
        "numeric_gates_active": bool(strict and args.strict_gates),
        "predeclared_tolerances": {"ap_drop_pp": AP_DROP_LIMIT_PP,
                                   "count_MAE_increase_leaf": COUNT_INCREASE_LIMIT_LEAF},
        "single_source_of_truth": {
            "budget_semantics": "src/jrtip_budget_rule_timing.py::budget_for",
            "aggregation": "src/jrtip_budget_policy.py::evaluate_arm",
            "timing_arms_priced_here": sorted(e["name"] for e in entries if e["from_timing_module"])},
        "control": {**{key: control_domain[key] for key in METRIC_KEYS},
                    "work_masks_per_frame": round(control_masks, 3)},
        "wiring_check": wiring,
        "cost_model": {key: cost.get(key) for key in
                       ("slope_seconds_per_instance_per_frame_mean", "intercept_seconds", "r_squared",
                        "max_abs_relative_residual", "ms_per_extra_masked_instance_per_frame")},
        "static_comparators": statics,
        "arms": entries,
        "bucket_vs_decision_pairs": bucket_vs_decision_pairs(entries, by_name, vectors, cost),
        "frontier_within_tolerances": [
            {"name": e["name"], "work_masks_per_frame": e["work_masks_per_frame"],
             "ap_drop_vs_control_pp": e["ap_drop_vs_control_pp"],
             "count_MAE_increase_vs_control_leaf": e["count_MAE_increase_vs_control_leaf"]}
            for e in frontier],
        "fill_gap_note": ("arms whose work lands between two archived statics are compared with "
                          "statics inside +/- "
                          f"{args.work_window} instances/frame; see comparators_within_work_window"),
        "checks": ledger.checks, "failures": ledger.failures,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(out / "bucket_pricing.json", payload)
    write_tsv(out / "arms.tsv", payload)
    write_json(out / "status.json", {"version": VERSION, "generated_utc": payload["generated_utc"],
                                     "failures": ledger.failures, "frames": len(base),
                                     "arms": len(entries),
                                     "elapsed_seconds": payload["elapsed_seconds"]})
    print(f"\nwrote {out / 'bucket_pricing.json'}", flush=True)
    print(f"failures: {ledger.failures if ledger.failures else 'none'}", flush=True)
    return 0 if not ledger.failures else 2


def write_tsv(path: Path, payload) -> None:
    header = ["name", "mode", "floor", "rule", "definition", "work_masks_per_frame", "AP",
              "ap_drop_vs_control_pp", "count_MAE", "count_MAE_increase_vs_control_leaf",
              "frames_truncated", "budget_mean", "budget_p90", "budget_max", "frames_at_floor",
              "frames_at_cap", "gate_ap", "gate_count", "projected_wall_seconds_cost_model",
              "reuses_quality_of"]
    lines = ["\t".join(header)]
    for arm in payload["arms"]:
        macro = arm.get("macro") or {}
        lines.append("\t".join([
            arm["name"], arm["mode"], str(arm.get("floor", "")), json.dumps(arm.get("rule", "")),
            arm["definition"], f"{arm['work_masks_per_frame']:.3f}",
            f"{macro.get('AP', float('nan')):.4f}",
            f"{arm.get('ap_drop_vs_control_pp', float('nan')):.4f}",
            f"{macro.get('count_MAE', float('nan')):.4f}",
            f"{arm.get('count_MAE_increase_vs_control_leaf', float('nan')):.4f}",
            str(arm.get("frames_with_reported_count_truncated", "")),
            f"{arm.get('budget_mean', float('nan')):.3f}", f"{arm.get('budget_p90', float('nan')):.1f}",
            str(arm.get("budget_max", "")), str(arm.get("frames_at_floor", "")),
            str(arm.get("frames_at_cap", "")), str(arm.get("gate_ap_drop_within_1pp")),
            str(arm.get("gate_count_MAE_within_0_2_leaf")),
            str(arm.get("projected_wall_seconds_cost_model")),
            str(arm.get("reuses_quality_of", ""))]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="jrtip/budget_bucket_pricing_v1")
    parser.add_argument("--policy-eval", default=",".join([
        "jrtip/budget_policy_v1/policy_eval.json",
        "jrtip/budget_policy_v1_extra/policy_eval.json"]),
        help="archived policy runs supplying the static comparators and the cost model")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--expected-frames", type=int, default=2947)
    parser.add_argument("--reproduce-tol", type=float, default=1e-6)
    parser.add_argument("--wiring-budget", type=int, default=20,
                        help="static budget re-priced through this path and compared with its GPU "
                             "pass; 0 skips")
    parser.add_argument("--work-window", type=float, default=1.0,
                        help="static comparators within this many masked instances/frame of an arm")
    parser.add_argument("--only-named", default="",
                        help="comma list: price just these timing-module arms (ignores the fill set)")
    parser.add_argument("--no-fill", dest="fill", action="store_false", default=True,
                        help="price only the timing-module arms")
    parser.add_argument("--fill-m", default="2,3,5,10")
    parser.add_argument("--fill-ratio", default="1.25,1.5")
    parser.add_argument("--fill-floor", default="17,18,19")
    parser.add_argument("--with-bucket-variants", action="store_true",
                        help="also price bucket_ versions of every fill arm (doubles the arm count)")
    parser.add_argument("--strict-gates", dest="strict_gates", action="store_true", default=True,
                        help="record a per-arm tolerance failure in the ledger (default on)")
    parser.add_argument("--no-strict-gates", dest="strict_gates", action="store_false",
                        help="report arm numbers without turning tolerance breaches into failures")
    parser.add_argument("--allow-gpu", action="store_true",
                        help="this script never touches the GPU; the flag only relaxes its own guard "
                             "record in the ledger")
    args = parser.parse_args()
    args.only_named = [name for name in args.only_named.split(",") if name.strip()]
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
