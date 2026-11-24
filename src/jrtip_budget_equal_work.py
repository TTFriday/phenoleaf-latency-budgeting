"""Equal-work comparison of output-side budget arms, read off the offline arm table.

``src/jrtip_budget_policy.py`` scores every budget arm by prefix-truncating one cap-100 development
pass and reports, per arm, its macro quality and its mean masked instances per frame (the cost proxy
whose wall-time relation is fitted from the seven archived GPU passes).  Its built-in equal-work
comparison uses a +/-5% window around each arm's work point, which is fragile here for a concrete
reason: the rule families land *between* the static grid points (a rule with mean work 14.3 has no
static partner inside 5%, since static 15 is 5.1% away), so a window comparison can silently report
"no comparable static arm" for the pairs that matter most.

This script therefore prices each arm against the static curve in two ways and requires both to agree
before any equal-work gain is claimed:

1. ``nearest`` - the passing static arm whose work is within 5% of the arm's work, as predeclared;
2. ``interpolated`` - the static curve's AP drop at exactly the arm's work, from piecewise-linear
   interpolation between the two bracketing static arms, with the interpolation error bound reported
   (the largest one-sided AP gap per unit of work inside the bracket, so a 0.05 pp "win" smaller than
   that bound is not a win).

It also asserts the static curve is monotone in work (a smaller budget cannot be better than a larger
one on the same stream - if it ever is, the arm table or the prefix lemma is wrong, not the model),
and converts each arm's work into predicted whole-pass seconds using the fitted cost model.

Read-only over already-written artifacts; development roles only; no GPU, no inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORK_WINDOW = 0.05


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def static_curve(arms):
    """(work, AP drop) knots of the static cost curve, ascending in work, plus monotonicity breaches.

    Points are sorted by ascending work, and the curve must be non-increasing in drop: a larger
    budget masks at least as many instances, so it cannot lose more AP on the same stream.  A breach
    therefore means the arm table or the prefix lemma is wrong, not the model.
    """
    points = sorted((float(arm["mean_instances_masked_per_frame"]),
                     float(arm["ap_drop_vs_control_pp"]), arm["name"])
                    for arm in arms if arm["kind"] == "static")
    breaches = [f"budget {points[i + 1][2]} masks more per frame ({points[i + 1][0]}) than "
                f"budget {points[i][2]} ({points[i][0]}) "
                f"yet loses more AP ({points[i + 1][1]} pp vs {points[i][1]} pp)"
                for i in range(len(points) - 1) if points[i + 1][1] > points[i][1] + 1e-9]
    return points, breaches


def interpolate(points, work):
    """Piecewise-linear AP drop of the static curve at `work`, plus a local slope bound.

    Inside a bracket the bound is ``slope x max(work - low, high - work)``, which is exactly the
    distance from the chord to the farther knot: monotonicity alone says the true drop lies between
    the two knots, so a claimed gain smaller than that distance is not a gain.  Outside the knot
    range no value is invented; the nearest knot is reported with ``extrapolated`` set, and callers
    must not read a bracket-bound comparison as evidence there.
    """
    if not points:
        return None, None, None
    if work <= points[0][0]:
        return points[0][1], 0.0, "below_lowest_knot"
    if work >= points[-1][0]:
        return points[-1][1], 0.0, "above_highest_knot"
    for i in range(len(points) - 1):
        low, high = points[i], points[i + 1]
        if low[0] <= work <= high[0] and high[0] > low[0]:
            weight = (work - low[0]) / (high[0] - low[0])
            value = low[1] + weight * (high[1] - low[1])
            slope = abs(high[1] - low[1]) / (high[0] - low[0])
            return value, slope * max(work - low[0], high[0] - work), None
    return None, None, "no_bracket"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", default=str(ROOT / "jrtip/budget_policy_v1/policy_eval.json"),
                        help="comma-separated policy_eval.json files; arms are joined by name, so a "
                             "densification run can be priced against the primary arm table")
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_policy_v1/equal_work.json"))
    args = parser.parse_args()

    sources = [Path(item.strip()) for item in args.eval.split(",") if item.strip()]
    payload = json.loads(sources[0].read_text(encoding="utf-8"))
    arms, seen = [], {}
    for path in sources:
        document = payload if path == sources[0] else json.loads(path.read_text(encoding="utf-8"))
        for arm in document["arms"]:
            if arm["name"] in seen:
                # The same arm scored twice must agree exactly, or two runs disagree about the stream.
                assert abs(seen[arm["name"]]["macro"]["AP"] - arm["macro"]["AP"]) < 1e-12, arm["name"]
                continue
            seen[arm["name"]] = arm
            arm["scored_in"] = str(path)
            arms.append(arm)
    control_masks = float(payload["control"]["mean_instances_masked_per_frame"])
    cost = payload.get("cost_model", {})
    slope = cost.get("slope_seconds_per_instance_per_frame_mean")
    intercept = cost.get("intercept_seconds")
    bars = payload["predeclared_tolerances"]

    points, breaches = static_curve(arms)
    passing = {arm["name"]: arm for arm in arms
               if arm["ap_drop_vs_control_pp"] <= bars["AP_drop_vs_control_pp"]
               and arm["count_MAE_increase_vs_control_leaf"]
               <= bars["count_MAE_increase_vs_control_leaf"]}
    static_passing = [name for name, arm in passing.items() if arm["kind"] == "static" and arm["name"] != "100"]

    rows = []
    for arm in arms:
        if arm["kind"] == "static" or arm["name"] == "100":
            continue
        work = float(arm["mean_instances_masked_per_frame"])
        drop = float(arm["ap_drop_vs_control_pp"])
        neighbors = [passing[name] for name in static_passing
                     if abs(float(passing[name]["mean_instances_masked_per_frame"]) - work)
                     <= WORK_WINDOW * work]
        nearest = min(neighbors, key=lambda s: float(s["ap_drop_vs_control_pp"])) if neighbors else None
        interpolated, bound, extrapolated = interpolate(points, work)
        rows.append({
            "arm": arm["name"], "kind": arm["kind"], "work_masks_per_frame": round(work, 3),
            "ap_drop_pp": round(drop, 4),
            "count_MAE_increase_leaf": arm["count_MAE_increase_vs_control_leaf"],
            "count_truncated_frames": arm["frames_with_reported_count_truncated"],
            "passes_quality_gate": arm["name"] in passing,
            "worst_genotype": arm.get("worst_genotype"),
            "predicted_pass_seconds": None if slope is None else round(slope * work + intercept, 1),
            "nearest_static": None if nearest is None else {
                "arm": nearest["name"], "work": nearest["mean_instances_masked_per_frame"],
                "ap_drop_pp": nearest["ap_drop_vs_control_pp"],
                "gain_pp": round(float(nearest["ap_drop_vs_control_pp"]) - drop, 4)},
            "interpolated_static": None if interpolated is None else {
                "ap_drop_pp": round(interpolated, 4), "error_bound_pp": round(float(bound), 4),
                "gain_pp": round(interpolated - drop, 4), "extrapolated": extrapolated},
        })
        print(f"{arm['name']:28s} work {work:6.2f}  drop {drop:6.4f} pp  "
              f"interp-static {('-') if interpolated is None else f'{interpolated:6.4f} +- {bound:.4f}'}  "
              f"gain {('n/a') if interpolated is None else f'{interpolated - drop:+6.4f} pp'}  "
              f"nearest {('none in window') if nearest is None else nearest['name']}", flush=True)

    both_positive = [row["arm"] for row in rows if row["passes_quality_gate"]
                     and row["nearest_static"] and row["nearest_static"]["gain_pp"] > 0
                     and row["interpolated_static"]
                     and row["interpolated_static"]["extrapolated"] is None
                     and row["interpolated_static"]["gain_pp"] > row["interpolated_static"]["error_bound_pp"]]
    result = {
        "version": "jrtip-budget-equal-work-v1", "generated_utc": utc(), "source": [str(path) for path in sources],
        "work_window": WORK_WINDOW, "predeclared_tolerances": bars,
        "control_work_masks_per_frame": round(control_masks, 3),
        "static_curve_work_drop_points": [[p[0], p[1], p[2]] for p in points],
        "static_curve_monotone_in_work": not breaches, "monotonicity_breaches": breaches,
        "cost_model": {"slope_seconds_per_instance_per_frame_mean": slope,
                       "intercept_seconds": intercept, "r_squared": cost.get("r_squared"),
                       "max_abs_relative_residual": cost.get("max_abs_relative_residual")},
        "arms": rows,
        "gates": {
            "passing_static_arms": sorted(static_passing, key=lambda n: float(n)),
            "arms_beating_static_at_equal_work_both_readings": both_positive,
            "criterion": ("an arm counts only if the nearest passing static arm inside a 5% work "
                          "window loses more AP, AND the statically interpolated drop at the same "
                          "work exceeds the arm's drop by more than the interpolation error bound"),
        },
    }
    write_json(Path(args.out), result)
    print("\nwrote", args.out, flush=True)
    print("arms beating the equal-work static reference under both readings:",
          ", ".join(both_positive) if both_positive else "none", flush=True)
    print("static curve monotone:", result["static_curve_monotone_in_work"], breaches[:2], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
