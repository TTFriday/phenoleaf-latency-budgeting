"""Machine-check that the timing module's decision function and the pricing module's are one function.

§10 priced ten rule arms through ``jrtip_budget_policy.k_rule`` / ``k_ratio``; §11 timed five arms
through ``jrtip_budget_rule_timing.rule_k`` / ``budget_for``; §14 stamped 23 arms' quality through
``budget_for`` as well.  Two implementations of "what budget does this frame emit" agreeing on the
*aggregate* (work to 3 decimals, AP to 6) is suggestive but not the claim we need - the claim is that
they agree **frame by frame**, because a per-frame disagreement that happened to average out would
still mean the timed object and the priced object are different arms.

This script reads the frozen cap-100 development stream once and asserts, over all contract frames:

1. ``record["count"] == #{score >= 0.5}`` - the scalar the GPU path reads off the device is the field
   the offline path uses (also checked inside the pricing run; re-checked here so this file is
   self-contained evidence);
2. for every arm the two modules share - the five rule arms §11 timed; the other eighteen of §14's
   twenty-three were born inside the pricing run and are listed under ``arms_priced_only`` rather than
   silently skipped: their ``(mode, rule, floor)`` must equal the timing module's ``ARMS`` entry, and
   the policy module's raw rule arithmetic, put through the timing module's bucket map when the arm is
   a bucketed one, must equal ``budget_for`` frame by frame.  For a bucketed arm the check additionally
   records how many frames the grid actually moved, so passing cannot be an artifact of comparing a
   value with itself;
3. the structural inequality every rule arm rests on - ``k >= count`` - plus a round trip from the
   recomputed budget vector to the artifact's ``budget_histogram`` and ``work_masks_per_frame``, so a
   mismatch between §14's table and the definition of the arms is caught here rather than in prose;
4. bucket rounding lands on the frozen grid and is never below the raw rule value.

What this does *not* buy: the two pipelines' quality numbers are not independent measurements of the
same arm.  Both call one evaluation engine, so once the budget vectors agree the AP agrees by
construction.  The value of the check is that "the arm §11 timed" and "the arm §14 priced" are proved
to be the same per-frame function, which is what makes the ms-to-pp readings legitimate.

Read-only; development roles only; no GPU, no inference, no COCO (runs in about a minute).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_policy import (  # noqa: E402
    COUNT_SCORE, k_ratio, k_rule, read_stream,
)
from src.jrtip_budget_rule_timing import (  # noqa: E402
    ARMS, BUCKET_GRID, CONTROL, QUALITY_ROOT, SCALE, bucket_of, budget_for, rule_k,
)
from src.jrtip_pilot import load_inputs, select_dev  # noqa: E402

TOLERANCE_ABS = 6e-4  # the artifact stores work rounded to 3 decimals


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalise_rule(rule):
    """JSON stores ``["c_plus_m", 10]``; keep the parameter's type aligned with the timing module."""
    kind, parameter = rule
    return (str(kind), int(parameter) if kind == "c_plus_m" else float(parameter))


def policy_k(rule, record, floor: int) -> int:
    kind, parameter = rule
    if kind == "c_plus_m":
        return k_rule(record, int(parameter), floor, CONTROL)
    return k_ratio(record, float(parameter), floor, CONTROL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing",
                        default=str(ROOT / "jrtip/budget_bucket_pricing_v1/bucket_pricing.json"))
    parser.add_argument("--out",
                        default=str(ROOT / "jrtip/budget_bucket_pricing_v1/rule_vector_check.json"))
    args = parser.parse_args()

    pricing = json.loads(Path(args.pricing).read_text(encoding="utf-8"))
    control_stream = QUALITY_ROOT / f"cap{CONTROL:05d}" / f"quality_{CONTROL:06d}_S{SCALE}.jsonl"
    if not control_stream.exists():
        raise FileNotFoundError(f"missing control stream: {control_stream}")
    config_path = QUALITY_ROOT / f"cap{CONTROL:05d}" / "budget_quality_config.json"
    benchmark = Path(json.loads(config_path.read_text(encoding="utf-8"))["source_benchmark"])
    training = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))["training_config"]
    rows, _data, _ = load_inputs(training["inventory"], training["instances"], training["readiness"],
                                 training["data_root"])
    stream = select_dev(rows, 0)
    rows_by_image_id = {row["image_id"]: row for row in stream}
    base = [record for record in read_stream(control_stream) if record["image_id"] in rows_by_image_id]
    if len(base) != len(stream):
        print(f"FAIL stream_length: {len(base)} records vs {len(stream)} contract rows", flush=True)
        return 1

    checks, failures = [], []

    def note(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)
            print(f"FAIL {name}: {detail}", flush=True)

    # 1. the device-side scalar is the archived count field
    bad_count = [record["image_id"] for record in base
                 if int(record["count"]) != sum(1 for instance in record["instances"]
                                                if float(instance["score"]) >= COUNT_SCORE)]
    note("count_field_equals_score_ge_half", not bad_count,
         f"{len(bad_count)} frames, first {bad_count[:3]}")

    counts = [int(record["count"]) for record in base]

    per_arm, only_priced = {}, []
    for arm in pricing["arms"]:
        name = arm["name"]
        mode, floor = str(arm["mode"]), int(arm["floor"])
        rule = normalise_rule(arm["rule"])
        if name not in ARMS:
            # Only the five arms section 11 timed have a second implementation to compare against;
            # the other eighteen were born inside the pricing run.  Recording them as uncovered is
            # the honest outcome - failing here would pretend the timing module was supposed to know
            # arms it was never asked to measure.
            only_priced.append(name)
            continue
        timing_arm = ARMS[name]
        note(f"arm_{name}_mode_rule_floor_match",
             (timing_arm["mode"], normalise_rule(timing_arm["rule"]), int(timing_arm["floor"]))
             == (mode, rule, floor),
             f"timing {timing_arm['mode']}/{timing_arm['rule']}/{timing_arm['floor']} "
             f"vs pricing {mode}/{rule}/{floor}")
        bucketed = mode == "bucket"
        policy_vector, timing_vector, raw_vector, scalar_mismatches = [], [], [], 0
        for record, c in zip(base, counts):
            k_policy = policy_k(rule, record, floor)
            raw_vector.append(k_policy)
            if k_policy != rule_k(rule, c, floor, CONTROL):
                scalar_mismatches += 1
            policy_vector.append(bucket_of(k_policy) if bucketed else k_policy)
            timing_vector.append(budget_for(timing_arm, c))
        frame_mismatches = sum(a != b for a, b in zip(policy_vector, timing_vector))
        note(f"arm_{name}_per_frame_vectors_identical",
             frame_mismatches == 0 and scalar_mismatches == 0,
             f"policy-vs-timing frame mismatches {frame_mismatches}, "
             f"record-count-vs-scalar mismatches {scalar_mismatches}")
        if bucketed:
            moved = sum(a != b for a, b in zip(raw_vector, policy_vector))
            note(f"arm_{name}_bucketing_bites", moved > 0,
                 f"the grid left every frame's k unchanged, so the composite comparison would be vacuous")
        dropped = sum(1 for k, c in zip(timing_vector, counts) if k < c)
        note(f"arm_{name}_never_drops_reported_leaf", dropped == 0, f"{dropped} frames where k < count")
        recomputed = {}
        for key, value in Counter(timing_vector).items():
            recomputed[str(key)] = value
        stored = {str(key): value for key, value in (arm.get("budget_histogram") or {}).items()}
        note(f"arm_{name}_budget_histogram_matches_artifact", recomputed == stored,
             "" if recomputed == stored
             else f"recomputed {sorted(recomputed.items())} vs stored {sorted(stored.items())}")
        recomputed_work = sum(timing_vector) / len(timing_vector)
        note(f"arm_{name}_mean_work_matches_artifact",
             abs(recomputed_work - float(arm["work_masks_per_frame"])) <= TOLERANCE_ABS,
             f"recomputed {recomputed_work:.4f} vs artifact {arm['work_masks_per_frame']}")
        per_arm[name] = {"frames": len(timing_vector), "mean_k": round(recomputed_work, 4),
                         "min_k": min(timing_vector), "max_k": max(timing_vector), "mode": mode,
                         "rule": list(rule), "floor": floor,
                         "frames_moved_by_the_grid": (sum(a != b for a, b in zip(raw_vector, policy_vector))
                                                      if bucketed else 0)}

    # 4. the bucket map itself
    off_grid = sorted({bucket_of(value) for value in counts} - set(BUCKET_GRID))
    note("bucket_of_lands_on_the_frozen_grid", not off_grid, f"off-grid values {off_grid[:5]}")
    note("bucket_grid_matches_pricing_run", list(BUCKET_GRID) == list(pricing.get("bucket_grid", BUCKET_GRID)),
         f"timing {list(BUCKET_GRID)} vs pricing {pricing.get('bucket_grid')}")
    below = [value for value in counts if bucket_of(value) < value]
    note("bucket_of_never_below_raw", not below, f"{len(below)} counterexamples")

    result = {
        "version": "jrtip-budget-rule-vector-check-v1", "generated_utc": utc(),
        "pricing_artifact": str(args.pricing), "stream": str(control_stream),
        "frames": len(base), "count_score_threshold": COUNT_SCORE, "bucket_grid": list(BUCKET_GRID),
        "arms_checked": len(per_arm), "arms_priced_only": only_priced,
        "checks_total": len(checks), "checks_failed": len(failures),
        "failures": failures, "checks": checks, "per_arm": per_arm,
        "claim": ("frame-by-frame identity, over the frozen development contract, between the policy "
                  "module's rule arithmetic plus the timing module's bucket map and the timing "
                  "module's own composite decision function - for the five rule arms section 11 timed "
                  "and section 14 priced.  This is a decision-function check, not a metric check: the "
                  "AP and count MAE of those arms come from one shared evaluation engine, so their "
                  "agreement across pipelines is definitional, not independent evidence."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    print(f"\narms {len(per_arm)}  checks {len(checks)}  failed {len(failures)} {failures[:4]}", flush=True)
    print("wrote", out, flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
