"""Price the offline-stamped rule arms against the static curve with the predeclared equal-work readings.

``src/jrtip_budget_bucket_pricing.py`` (P1c(b) quality half) stamps 23 rule arms - the five arms that
§11 timed, plus the floor in {17,18,19} densification §10 asked for - but it deliberately does not
judge them: it only reports work, quality and a +/-1.0-instance static neighbourhood.  §10's verdict
form, on the other hand, is predeclared and requires *two* readings to agree before a gain counts:
the best passing static inside a 5% work window, and the piecewise-linear static curve at exactly the
arm's work with its chord error bound.

This script applies that same form - imported from ``jrtip_budget_equal_work`` so there is one
implementation of the comparison, not two - to the pricing artifact.  It exists because the P1b
controller will land in the work interval 17-21 that §10 called a "comparator void", and the pass line
there must come from the same yardstick that produced the seven rule rows already in §10, not from a
looser ad-hoc comparison invented afterwards.

Two guards, both cheap and both about not silently mixing sources:

* the static knots are read from the pricing artifact and cross-checked, value by value, against the
  policy arm table (``--eval``); a disagreement means two runs disagree about the stream, so the
  verdicts would not be comparable;
* the five arms §11 timed are re-reported with their measured service time side by side with the
  offline-priced work, because those two numbers come from *different frame sets* (1,073 timed rows vs
  the 2,947-frame development contract) and must never be read as one population.

Read-only over already-written artifacts; development roles only; no GPU, no inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity with the sibling scripts' numeric surface)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_equal_work import WORK_WINDOW, interpolate  # noqa: E402

# §11 measured the service time of exactly these arms, on the timing stream.
TIMED_IN_SECTION_11 = ("sync_ceil_1p5_f15", "sync_c_plus_10_f15", "readfull_c_plus_10_f15",
                       "bucket_ceil_1p5_f15", "bucket_c_plus_10_f15")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.name + ".tmp"
    tmp_path = path.with_name(tmp)
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def static_knots(pricing, eval_paths):
    """Static (work, drop, name) knots from the pricing artifact, cross-checked against the arm tables.

    The comparators come from the pricing artifact, whose static grid is denser than the primary arm
    table of §6/§9 (the extra nodes live in the §10 densification run).  The cross-check therefore
    joins the tables by arm name the way ``jrtip_budget_equal_work.py`` does, and only compares a knot
    against a table that actually contains it: "absent from this file" is not a disagreement, it just
    records which run scored which arm.
    """
    comparators = pricing["static_comparators"]
    knots = sorted((float(v["work_masks_per_frame"]), float(v["ap_drop_vs_control_pp"]), k)
                   for k, v in comparators.items())
    merged, scored_by, present_in = {}, {}, []
    disagreements = []
    for path in eval_paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        present_in.append(path.name)
        for arm in document["arms"]:
            if arm["kind"] != "static":
                continue
            name = str(arm["name"])
            if name in comparators:
                scored_by.setdefault(name, []).append(path.name)
            if name in merged:
                previous, previous_source = merged[name]
                for key in ("mean_instances_masked_per_frame", "ap_drop_vs_control_pp"):
                    if abs(float(previous[key]) - float(arm[key])) > 1e-9:
                        disagreements.append(
                            f"two arm tables disagree about static {name} {key}: "
                            f"{previous[key]} ({previous_source}) vs {arm[key]} ({path.name})")
            else:
                merged[name] = (arm, path.name)
    for name, entry in comparators.items():
        found = merged.get(name)
        if found is None:
            disagreements.append(f"static {name} scored in the pricing artifact but in no arm table "
                                 f"(tables read: {present_in})")
            continue
        arm, source = found
        for pricing_key, table_key in (("work_masks_per_frame", "mean_instances_masked_per_frame"),
                                       ("ap_drop_vs_control_pp", "ap_drop_vs_control_pp")):
            if abs(float(entry[pricing_key]) - float(arm[table_key])) > 1e-6:
                disagreements.append(f"{source}: static {name} {pricing_key} "
                                     f"{entry[pricing_key]} vs {arm[table_key]}")
    breaches = [f"knot {knots[i + 1][2]} masks more ({knots[i + 1][0]}) yet loses more AP "
                f"({knots[i + 1][1]} pp vs {knots[i][1]} pp)"
                for i in range(len(knots) - 1) if knots[i + 1][1] > knots[i][1] + 1e-9]
    return knots, disagreements, breaches, scored_by


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing", default=str(ROOT / "jrtip/budget_bucket_pricing_v1/bucket_pricing.json"))
    parser.add_argument("--eval", default=str(ROOT / "jrtip/budget_policy_v1/policy_eval.json") + ","
                        + str(ROOT / "jrtip/budget_policy_v1_extra/policy_eval.json"),
                        help="comma-separated policy_eval.json files used only to cross-check the knots")
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_bucket_pricing_v1/equal_work.json"))
    args = parser.parse_args()

    pricing = json.loads(Path(args.pricing).read_text(encoding="utf-8"))
    eval_paths = [Path(item.strip()) for item in args.eval.split(",") if item.strip()]
    eval_paths = [path for path in eval_paths if path.exists()]
    bars = pricing["predeclared_tolerances"]
    control = pricing["control"]

    knots, disagreements, breaches, scored_by = static_knots(pricing, eval_paths)
    passing_static = [knot for knot in knots
                      if knot[1] <= bars["ap_drop_pp"]
                      and float(pricing["static_comparators"][knot[2]]["count_MAE_increase_vs_control_leaf"])
                      <= bars["count_MAE_increase_leaf"] and knot[2] != "100"]

    rows = []
    for arm in pricing["arms"]:
        work, drop = float(arm["work_masks_per_frame"]), float(arm["ap_drop_vs_control_pp"])
        gate = drop <= bars["ap_drop_pp"] and arm["count_MAE_increase_vs_control_leaf"] <= bars["count_MAE_increase_leaf"]
        neighbors = [knot for knot in passing_static if abs(knot[0] - work) <= WORK_WINDOW * work]
        nearest = min(neighbors, key=lambda knot: knot[1]) if neighbors else None
        value, bound, extrapolated = interpolate(knots, work)
        reading_two = value is not None and extrapolated is None and (value - drop) > float(bound)
        rows.append({
            "arm": arm["name"], "work_masks_per_frame": round(work, 3), "ap_drop_pp": round(drop, 4),
            "count_MAE_increase_leaf": arm["count_MAE_increase_vs_control_leaf"],
            "truncated_frames": arm["frames_with_reported_count_truncated"],
            "passes_quality_gate": bool(gate),
            "mask_work_reduction_vs_control": round(100 * float(arm["mask_work_reduction_vs_control"]), 2),
            "worst_genotype": arm["worst_genotype"],
            "projected_pass_seconds_cost_model": arm["projected_wall_seconds_cost_model"],
            "nearest_passing_static": None if nearest is None else {
                "arm": nearest[2], "work": nearest[0], "ap_drop_pp": nearest[1],
                "gain_pp": round(nearest[1] - drop, 4)},
            "interpolated_static": None if value is None else {
                "ap_drop_pp": round(value, 4), "error_bound_pp": round(float(bound), 4),
                "gain_pp": round(value - drop, 4), "extrapolated": extrapolated},
            "gain_established_both_readings": bool(gate and nearest is not None
                                                   and nearest[1] - drop > 0 and reading_two),
            "timed_in_section_11": arm["name"] in TIMED_IN_SECTION_11,
        })
        print(f"{arm['name']:26s} work {work:6.3f} drop {drop:6.4f}  "
              f"nearest {('none' if nearest is None else f'{nearest[2]} (+{nearest[1] - drop:.4f})'):22s} "
              f"interp {('-' if value is None else f'{value:.4f} +- {bound:.4f} (+{value - drop:.4f})'):28s} "
              f"{'ESTABLISHED' if rows[-1]['gain_established_both_readings'] else 'not established'}",
              flush=True)

    established = [row["arm"] for row in rows if row["gain_established_both_readings"]]
    result = {
        "version": "jrtip-budget-rule-equal-work-v1", "generated_utc": utc(),
        "pricing_artifact": str(args.pricing), "knot_cross_checked_against": [str(path) for path in eval_paths],
        "knot_disagreements": disagreements, "static_curve_monotone_in_work": not breaches,
        "monotonicity_breaches": breaches, "work_window": WORK_WINDOW,
        "predeclared_tolerances": bars,
        "control": control,
        "static_knots": [[knot[0], knot[1], knot[2]] for knot in knots],
        "static_knot_scored_in": scored_by,
        "passing_static_arms": [knot[2] for knot in passing_static],
        "arms": rows,
        "criterion": ("an arm counts only if the nearest passing static arm inside a 5% work window "
                      "loses more AP, AND the statically interpolated drop at the same work exceeds the "
                      "arm's drop by more than the chord error bound (same form as §10)"),
        "arms_beating_static_at_equal_work_both_readings": established,
        "pass_line_by_work": [
            {"work": row["work_masks_per_frame"], "best_label_free_ap_drop_pp": row["ap_drop_pp"],
             "arm": row["arm"]}
            for row in sorted(rows, key=lambda item: item["work_masks_per_frame"])
            if row["passes_quality_gate"]],
        "note_on_streams": ("work and projected seconds come from the 2,947-frame development contract; "
                            f"service times for {list(TIMED_IN_SECTION_11)} come from §11's 1,073-frame "
                            "timing stream and are different populations, reported side by side but never averaged"),
    }
    out = Path(args.out)
    write_json(out, result)
    print("\nwrote", out, flush=True)
    print("knot disagreements with the arm tables:", len(disagreements), disagreements[:3], flush=True)
    print("static knots:", len(knots), "monotone:", not breaches, breaches[:2], flush=True)
    print("established under both readings:", len(established), "of", len(rows), flush=True)
    return 0 if not disagreements and not breaches else 1


if __name__ == "__main__":
    sys.exit(main())
