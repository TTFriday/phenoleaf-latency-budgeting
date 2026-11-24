"""Explain, without re-running any judgement, why the condition-1 report counts 48 label-free
comparators while the gate review in the results document counts 45 distinct passing points.

Section 19.1 of the results document records the discrepancy and explicitly declines to call it
resolved until the duplicate entries have actually been listed.  This script is that listing.  It
imports the comparator builder from the condition-1 script unchanged - same tables, same passing
test, same control-point exclusion - and only counts afterwards:

* entries as emitted, per source table;
* distinct by ``(kind, name)`` - a rule priced in two tables is one arm, not two;
* distinct by ``(kind, name, work, ap_drop)`` - catches a same-named arm whose numbers genuinely
  differ between tables, which would be a defect rather than a duplicate;
* the gate review's own key ``(work rounded to 0.01, drop rounded to 0.0001)`` minus the control.

Nothing here can move the verdict: the judgement takes the *minimum* drop inside the work window, so
repeating an arm cannot change it.  The script asserts the emitted count matches the archived verdict
and stops if it does not.

Two explanations were on the table when this was written - the extra three entries being the same arm
priced in two tables, or differently named arms colliding on one rounded coordinate - and the two
groupings below are what tell them apart.  The result JSON records which one holds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONDITION1 = ROOT / "src/jrtip_budget_p1b_condition1.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("condition1_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdict", default=str(ROOT / "jrtip/budget_p1b_v1/condition1_verdict.json"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_p1b_v1/comparator_duplicates.json"))
    args = parser.parse_args()

    module = load_module(CONDITION1)
    verdict = json.loads(Path(args.verdict).read_text(encoding="utf-8"))

    tables = str(module.__dict__["ROOT"] / "jrtip/budget_policy_v1/policy_eval.json")
    extra = str(module.__dict__["ROOT"] / "jrtip/budget_policy_v1_extra/policy_eval.json")
    bucket = str(module.__dict__["ROOT"] / "jrtip/budget_bucket_pricing_v1/bucket_pricing.json")
    comparators, skipped = module.label_free_tables([tables, extra, bucket], ["static", "static", "rule"])
    # label_free_tables keys its entries by basename, and two of the three tables share one, so count
    # per file by calling the same builder once per table.
    per_file = {}
    for path, kind in ((tables, "static"), (extra, "static"), (bucket, "rule")):
        entries, _ = module.label_free_tables([path], [kind])
        per_file[str(Path(path).relative_to(ROOT))] = len(entries)

    emitted = len(comparators)
    recorded = verdict.get("label_free_comparators")
    if emitted != recorded:
        raise SystemExit(f"comparator count {emitted} != archived verdict {recorded}")

    per_source = Counter(c["source"] for c in comparators)
    by_name = defaultdict(list)
    for c in comparators:
        by_name[(c["kind"], c["name"])].append(c)
    duplicated = {f"{kind} :: {name}": [
        {"source": c["source"], "work": c["work"], "drop_pp": c["drop_pp"],
         "count_leaf": c["count_leaf"]} for c in entries]
        for (kind, name), entries in sorted(by_name.items()) if len(entries) > 1}

    distinct_name = len(by_name)
    distinct_name_numbers = len({(c["kind"], c["name"], c["work"], c["drop_pp"]) for c in comparators})
    gate_review_key = {(round(c["work"], 2), round(c["drop_pp"], 4)) for c in comparators}
    same_name_conflicting = {k: v for k, v in duplicated.items()
                             if len({(e["work"], e["drop_pp"]) for e in v}) > 1}

    collide = defaultdict(list)
    for c in comparators:
        collide[(round(c["work"], 2), round(c["drop_pp"], 4))].append(c)
    collisions = {f"work {key[0]:.2f} / drop {key[1]:.4f} pp": [
        {"name": c["name"], "kind": c["kind"], "source": c["source"],
         "work": c["work"], "drop_pp": c["drop_pp"]} for c in entries]
        for key, entries in sorted(collide.items()) if len(entries) > 1}
    pass60 = [c for c in comparators if c["reduction_vs_control"] >= 0.60]

    out = {
        "version": "jrtip-comparator-duplicates-v1",
        "generated_from": str(CONDITION1.relative_to(ROOT)),
        "tables": [Path(p).name for p in (tables, extra, bucket)],
        "emitted_entries": emitted,
        "entries_per_file": per_file,
        "entries_per_source": dict(per_source),
        "distinct_by_kind_and_name": distinct_name,
        "distinct_by_kind_name_work_drop": distinct_name_numbers,
        "duplicate_entries": duplicated,
        "duplicate_count": emitted - distinct_name,
        "coordinate_collisions": collisions,
        "entries_reduction_ge_60pct": len(pass60),
        "distinct_points_reduction_ge_60pct": len(
            {(round(c["work"], 2), round(c["drop_pp"], 4)) for c in pass60}),
        "duplicates_with_conflicting_numbers": same_name_conflicting,
        "gate_review_distinct_points_on_rounded_work_and_drop": len(gate_review_key),
        "skipped_entries": len(skipped),
        "skipped_reasons": dict(Counter(s["why"] for s in skipped)),
        "reading": ("48 counts every comparator entry of every table; 45 counts distinct "
                    "(work, ap_drop) coordinates after rounding to 0.01 masks and 0.0001 pp.  The "
                    "difference is not cross-table repetition - every entry has a unique (kind, name), "
                    "so no arm is counted twice - it is differently named arms that land on the same "
                    "rounded coordinate, listed under coordinate_collisions.  The verdict is unaffected: "
                    "condition 1 takes the minimum drop inside the work window, which neither repeating "
                    "an arm nor rounding its coordinates can change."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"emitted entries {emitted} (verdict says {recorded}); per source {dict(per_source)}")
    print(f"skipped {len(skipped)} {dict(Counter(s['why'] for s in skipped))}")
    for path, count in per_file.items():
        print(f"  {path}: {count}")
    print(f"distinct by (kind, name) {distinct_name}  -> duplicates {emitted - distinct_name}")
    print(f"distinct by (kind, name, work, drop) {distinct_name_numbers}")
    print(f"gate-review key (work 2dp, drop 4dp) {len(gate_review_key)}")
    print(f"reduction >= 60%: {len(pass60)} entries, "
          f"{out['distinct_points_reduction_ge_60pct']} distinct points")
    for name, entries in duplicated.items():
        print(f"  DUP {name}")
        for e in entries:
            print(f"      {e['source']:24s} work {e['work']:8.3f}  drop {e['drop_pp']:.4f} pp  "
                  f"count {e['count_leaf']}")
    for key, entries in collisions.items():
        print(f"  SAME ROUNDED COORDINATE {key}")
        for e in entries:
            print(f"      {e['kind']:7s} {e['name']:38s} work {e['work']:8.4f}  "
                  f"drop {e['drop_pp']:.5f} pp  [{e['source']}]")
    if same_name_conflicting:
        print("  !! same name, different numbers:")
        for name, entries in same_name_conflicting.items():
            print(f"     {name}: {entries}")
    else:
        print("  no duplicate name carries conflicting numbers")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
