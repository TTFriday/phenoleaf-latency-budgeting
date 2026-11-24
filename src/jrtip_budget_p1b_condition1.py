"""Execute stop-loss condition 1 of plan v2 section 9 against the label-free frontier.

Section 5's comparator amendment (written before any controller number existed) says the learned
controller must be judged not against the static ladder but against **the strongest label-free arm in
a +/-5% masked-instance window around its own work point**, static or rule.  Section 9's condition 1
then stops the formal matrix if *no* threshold candidate beats that comparator.

Two archived instruments already exist but neither is this judgment:

* ``jrtip_budget_equal_work.py`` compares non-static arms against the *static* curve in two readings.
  That is section 5's demoted secondary reading; feeding it the controller arms (stage 56 of the
  judgment queue) answers "beats the equal-work static", not condition 1.
* ``jrtip_budget_rule_equal_work.py`` extends the same two readings across the rule families, but its
  comparator set is still statics.

This script therefore builds the label-free comparator set - every *passing* arm of ``kind ==
"static"`` plus every passing rule object from the offline pricing tables - and prices each learned
policy in it.  It adds nothing to the predeclared machinery: the tolerances, the +/-5% window, the
definition of "passing", and the cost model are the frozen ones, read from the artifacts that already
carry them.

Boundary thresholds are reported, not hidden.  ``tau = -inf`` reproduces the control (work 89.519,
zero saving: there is no label-free comparator to beat because nothing is bought) and ``tau = +inf``
reproduces static 15, which fails the AP gate outright.  Neither is an allocation, so neither can
avert or trigger condition 1 on its own; the verdict line counts interior candidates only, and the
script prints how many candidates were interior so a collapse like C1's (63 -> 1 interior policy)
cannot be mistaken for a well-populated grid.

Read-only over already-written artifacts; pure arithmetic; no GPU, no COCO, no inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORK_WINDOW = 0.05          # section 5: +/-5% of the candidate's own masked-instance work
CONTROL_WORK = 89.519       # development contract, cap-100, masks per frame (frozen artifact value)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def work_of(arm: dict):
    for key in ("mean_instances_masked_per_frame", "work_masks_per_frame"):
        value = arm.get(key)
        if value is not None:
            return float(value)
    return None


def drop_of(arm: dict):
    for key in ("ap_drop_vs_control_pp",):
        value = arm.get(key)
        if value is not None:
            return float(value)
    return None


def count_of(arm: dict):
    for key in ("count_MAE_increase_vs_control_leaf", "count_mae_delta_vs_control"):
        value = arm.get(key)
        if value is not None:
            return float(value)
    return None


def passes(arm: dict) -> bool:
    ap_gate = arm.get("gate_ap_drop_within_1pp")
    count_gate = arm.get("gate_count_MAE_within_0_2_leaf")
    if ap_gate is None or count_gate is None:
        drop, count = drop_of(arm), count_of(arm)
        if drop is None or count is None:
            return False
        return drop <= 1.0 and abs(count) <= 0.2
    return bool(ap_gate and count_gate)


def label_free_tables(paths, kinds):
    """Return (comparators, skipped).  Each comparator: {source, name, kind, work, drop, count}."""
    comparators, skipped = [], []
    for path, default_kind in zip(paths, kinds):
        table = load(Path(path))
        for arm in table.get("arms", []):
            kind = str(arm.get("kind", default_kind))
            work, drop = work_of(arm), drop_of(arm)
            if work is None or drop is None:
                skipped.append({"source": str(path), "name": arm.get("name"), "why": "no work/AP field"})
                continue
            if kind == "static" and abs(work - CONTROL_WORK) < 1e-6:
                continue                     # the cap-100 control is the reference, not an arm
            if not passes(arm):
                skipped.append({"source": str(path), "name": arm.get("name"), "kind": kind,
                                "work": work, "drop_pp": drop, "why": "fails a predeclared gate"})
                continue
            comparators.append({"source": Path(path).name, "name": str(arm.get("name")), "kind": kind,
                                "work": work, "drop_pp": drop, "count_leaf": count_of(arm),
                                "reduction_vs_control": round(1.0 - work / CONTROL_WORK, 4)})
    return comparators, skipped


def learned_arms(paths):
    arms = []
    for path in paths:
        table = load(Path(path))
        tolerances = table.get("predeclared_tolerances", {})
        for arm in table.get("arms", []):
            kind = str(arm.get("kind", "learned"))
            if kind == "random_matched_work":
                continue                      # P1 is a control on allocation, not a candidate policy
            work, drop = work_of(arm), drop_of(arm)
            if work is None or drop is None:
                continue
            arms.append({"source": Path(path).name, "name": str(arm.get("name")), "kind": kind,
                         "work": work, "drop_pp": drop, "count_leaf": count_of(arm),
                         "truncated_frames": arm.get("frames_with_reported_count_truncated"),
                         "budget_max": arm.get("budget_max"), "budget_median": arm.get("budget_median"),
                         "budget_histogram": arm.get("budget_histogram"),
                         "gate": passes(arm), "tolerances": tolerances,
                         "worst_genotype": arm.get("worst_genotype")})
    return arms


def compare(candidate, comparators):
    low, high = candidate["work"] * (1.0 - WORK_WINDOW), candidate["work"] * (1.0 + WORK_WINDOW)
    in_window = [c for c in comparators if low <= c["work"] <= high]
    if not in_window:
        return {"window": [round(low, 3), round(high, 3)], "comparators_in_window": 0,
                "strongest_label_free": None, "beats_strongest": None,
                "why": "no passing label-free arm inside the +/-5% window"}
    strongest = min(in_window, key=lambda c: c["drop_pp"])
    gain = strongest["drop_pp"] - candidate["drop_pp"]
    strict_better = candidate["drop_pp"] < strongest["drop_pp"]
    count_ok = True
    if candidate["count_leaf"] is not None and strongest["count_leaf"] is not None:
        count_ok = candidate["count_leaf"] <= strongest["count_leaf"] + 1e-9
    return {"window": [round(low, 3), round(high, 3)], "comparators_in_window": len(in_window),
            "strongest_label_free": strongest, "label_free_gain_pp": round(gain, 4),
            "beats_strongest": bool(strict_better and count_ok),
            "ap_strictly_lower": bool(strict_better), "count_not_worse": bool(count_ok),
            "all_in_window": sorted(in_window, key=lambda c: c["drop_pp"])[:6]}


def classify(candidate):
    """Boundary and disqualified cases are stated, not silently counted as candidates."""
    if abs(candidate["work"] - CONTROL_WORK) < 1e-3:
        return "boundary_control"            # tau = -inf: buys nothing, so there is nothing to compare
    if not candidate["gate"]:
        return "disqualified_by_gate"        # cannot beat a comparator while failing a bar itself
    if candidate.get("budget_max") is not None and candidate["budget_max"] == candidate.get("budget_median"):
        return "boundary_uniform_ladder"     # one rung for every frame: a static arm, not an allocation
    return "interior"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-free", default=",".join([
        str(ROOT / "jrtip/budget_policy_v1/policy_eval.json"),
        str(ROOT / "jrtip/budget_policy_v1_extra/policy_eval.json"),
        str(ROOT / "jrtip/budget_bucket_pricing_v1/bucket_pricing.json")]))
    parser.add_argument("--controller", default=str(ROOT / "jrtip/budget_p1b_v1/arms_C1.json"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_p1b_v1/condition1_verdict.json"))
    args = parser.parse_args()

    comparators, skipped = label_free_tables(args.label_free.split(","), ["static", "static", "rule"])
    candidates = learned_arms(args.controller.split(","))
    if not candidates:
        raise SystemExit(f"no learned arms found in {args.controller}")

    rows = []
    for candidate in candidates:
        candidate["class"] = classify(candidate)
        candidate["comparison"] = compare(candidate, comparators)
        rows.append(candidate)

    interior = [r for r in rows if r["class"] == "interior"]
    winners = [r for r in interior if r["comparison"].get("beats_strongest")]
    verdict = {
        "version": "jrtip-budget-condition1-verdict-v1", "generated_utc": utc(),
        "criterion": ("plan v2 section 5 (comparator amendment, predeclared before any controller "
                      "number existed) and section 9 stop-loss condition 1: a learned threshold "
                      "candidate counts only if, at its own measured masked-instance work, it drops "
                      "less macro AP than the strongest passing label-free arm within +/-5% work "
                      "(static or rule) AND its count-MAE increase is no larger than that arm's."),
        "work_window": WORK_WINDOW, "control_work_masks_per_frame": CONTROL_WORK,
        "label_free_tables": args.label_free.split(","), "label_free_comparators": len(comparators),
        "label_free_skipped": skipped,
        "controller_tables": args.controller.split(","), "candidates": len(rows),
        "candidates_interior": len(interior),
        "candidates_boundary": [r["name"] for r in rows if r["class"] != "interior"],
        "interior_candidates": [r["name"] for r in interior],
        "distinct_policy_digests": sorted({r["name"].split("_")[-1] for r in rows}),
        "condition_1_triggered": bool(interior) and not winners,
        "no_interior_candidate": not interior,
        "winners": [r["name"] for r in winners],
        "rows": [{"name": r["name"], "class": r["class"], "work": r["work"],
                  "ap_drop_pp": r["drop_pp"], "count_leaf": r["count_leaf"],
                  "truncated_frames": r["truncated_frames"], "passes_gates": r["gate"],
                  "budget_median": r["budget_median"], "budget_max": r["budget_max"],
                  "budget_histogram": r["budget_histogram"], "comparison": r["comparison"]}
                 for r in sorted(rows, key=lambda x: x["work"])],
        "reading": ("condition 1 stops the formal matrix.  Boundary taus are listed with their class: "
                    "tau=-inf is the cap-100 control (no saving, nothing to compare against) and "
                    "tau=+inf collapses to static 15 (fails the AP gate); neither is an allocation.  "
                    "A grid that yields one interior policy is reported as such - the collapse is a "
                    "result, not a filtering artifact."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    print(f"comparators {len(comparators)}  candidates {len(rows)}  interior {len(interior)}  "
          f"winners {len(winners)}", flush=True)
    for r in sorted(rows, key=lambda x: x["work"]):
        comp = r["comparison"]
        best = comp.get("strongest_label_free")
        if best is None:
            reference = "-"
        else:
            reference = f"{best['name']} @ {best['work']:.3f} = {best['drop_pp']:.4f} pp"
        count = "-" if r["count_leaf"] is None else f"{r['count_leaf']:+.4f}"
        print(f"{r['name'][:34]:34s} [{r['class']:18s}] work {r['work']:7.3f} "
              f"dAP {r['drop_pp']:6.4f} dN {count}  vs label-free {reference}  "
              f"beats={comp.get('beats_strongest')}", flush=True)
    print("condition 1 triggered:", verdict["condition_1_triggered"], flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
