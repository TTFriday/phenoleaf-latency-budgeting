"""Compare the replicated rule-timing triple against the primary triple, arm by arm.

The replication (stage 63-65, ``jrtip/rule_timing_v1_rep/``) re-ran the eleven arms of
``jrtip/rule_timing_v1/`` under a byte-identical protocol: same frozen timing stream, same
checkpoint, same 1,073 frames per arm, rotating arm order, 24-frame bitwise identity gate before
timing.  The predeclared reading rules are the four written into section 21 of the results document
before session 2 landed, and this script implements exactly those and nothing else:

* judge every arm against *its own* cross-session spread, not a single global band;
* report both the absolute millisecond shift and the shift normalised by the same session's
  ``static100_native`` control, because the device is shared and machine state is not held fixed;
* the replication has no authority over any gate: the only verdict it can produce is whether an
  arm's time reduction stays at or above 60%;
* never quote whichever session happens to sit closest to the primary - all three sessions, their
  spread, and the interference probe are printed for every arm.

Read-only over both merged artifacts; writes one comparison JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE_ARMS = [
    "static20_native",
    "static16_native",
    "sync_ceil_1p5_f15",
    "sync_c_plus_10_f15",
    "readfull_c_plus_10_f15",
    "bucket_ceil_1p5_f15",
    "bucket_c_plus_10_f15",
]
CONTROL = "static100_native"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_arm(table: dict) -> dict:
    return {row["arm"]: row for row in table["arms"]}


def pct(new: float, old: float) -> float:
    return round(100.0 * (new - old) / old, 3)


def share(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2)


def control_series(rows: dict) -> list:
    return list(rows[CONTROL]["session_means_ms"])


def ratio_scatter(row: dict, control_ms: list) -> tuple:
    """Per-session share of the same session's cap-100 control, and how far each sits off the median.

    Excursions are flagged with one threshold applied symmetrically to the primary triple and the
    replication triple, so the rule cannot be tightened or loosened after seeing one of them.
    """
    ratios = [m / c for m, c in zip(row["session_means_ms"], control_ms)]
    med = statistics.median(ratios)
    dev = [round(100.0 * (r - med) / med, 2) for r in ratios]
    return ratios, dev


def per_session_reduction(row: dict, control_ms: list) -> list:
    return [round(100.0 * (1.0 - m / c), 2) for m, c in zip(row["session_means_ms"], control_ms)]


def static_line_residuals(p_rows: dict, r_rows: dict) -> dict:
    """Section 11's time-axis law re-tested on the replication: observed == static-15 anchor + 4.1161 ms per extra mask.

    The slope stays the frozen one from the primary static curve, and the anchor moves with the
    reading (primary mean, replication mean of session means, replication median of session means), so
    what is being tested is only whether the *rule* arms stay on the line that their own run's
    ``static15_native`` defines.
    """
    slope = 4.1161
    arms = ["sync_ceil_1p5_f15", "sync_c_plus_10_f15", "readfull_c_plus_10_f15",
            "bucket_ceil_1p5_f15", "bucket_c_plus_10_f15"]
    readings = {
        "primary_mean": (p_rows["static15_native"]["mean_ms_session_averaged"],
                         {a: p_rows[a]["mean_ms_session_averaged"] for a in arms}),
        "replication_mean": (r_rows["static15_native"]["mean_ms_session_averaged"],
                             {a: r_rows[a]["mean_ms_session_averaged"] for a in arms}),
        "replication_median_of_sessions": (statistics.median(r_rows["static15_native"]["session_means_ms"]),
                                           {a: statistics.median(r_rows[a]["session_means_ms"]) for a in arms}),
    }
    out = {"slope_ms_per_instance": slope, "anchor_arm": "static15_native", "readings": {}}
    for tag, (anchor, observed) in readings.items():
        res = {}
        for arm in arms:
            work = r_rows[arm]["emitted_instances_mean"]
            predicted = anchor + slope * (work - 15.0)
            res[arm] = {"work": round(work, 4), "predicted_ms": round(predicted, 3),
                        "observed_ms": round(observed[arm], 3),
                        "residual_ms": round(observed[arm] - predicted, 3)}
        out["readings"][tag] = {"anchor_ms": round(anchor, 3), "per_arm": res,
                                "max_abs_residual_ms": max(abs(v["residual_ms"]) for v in res.values())}
    return out


def paired_premium(prim: dict, rep: dict, sync_arm: str, bucket_arm: str, waste: float) -> dict:
    """bucket-minus-sync computed inside each session, which is the only legitimate pairing."""
    out = {"sync_arm": sync_arm, "bucket_arm": bucket_arm, "waste_instances": round(waste, 4)}
    for tag, table in (("primary", prim), ("replication", rep)):
        s = table[sync_arm]["session_means_ms"]
        b = table[bucket_arm]["session_means_ms"]
        prem = [round(bb - ss, 3) for ss, bb in zip(s, b)]
        out[f"{tag}_per_session_premium_ms"] = prem
        out[f"{tag}_median_premium_ms"] = statistics.median(prem)
        out[f"{tag}_ms_per_extra_mask"] = round(statistics.median(prem) / waste, 3)
    out["median_shift_pct"] = pct(out["replication_median_premium_ms"], out["primary_median_premium_ms"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default=str(ROOT / "jrtip/rule_timing_v1/rule_timing.json"))
    parser.add_argument("--replication", default=str(ROOT / "jrtip/rule_timing_v1_rep/rule_timing.json"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/rule_timing_v1_rep/replication_compare.json"))
    args = parser.parse_args()

    prim, rep = load(Path(args.primary)), load(Path(args.replication))
    p_rows, r_rows = by_arm(prim), by_arm(rep)
    if set(p_rows) != set(r_rows):
        raise SystemExit(f"arm sets differ: {sorted(set(p_rows) ^ set(r_rows))}")
    for name, table in (("primary", prim), ("replication", rep)):
        for key in ("identical_timing_stream_across_sessions", "identical_checkpoint_across_sessions",
                    "identical_versions_across_sessions", "all_gates_passed"):
            if not table[key]:
                raise SystemExit(f"{name}: {key} is not true")

    p_ctrl, r_ctrl = control_series(p_rows), control_series(r_rows)
    flag_at = 5.0
    rows = []
    for arm in [row["arm"] for row in rep["arms"]]:
        pr, rr = p_rows[arm], r_rows[arm]
        if len(pr["session_means_ms"]) != 3 or len(rr["session_means_ms"]) != 3:
            raise SystemExit(f"{arm}: expected 3 session means")
        if abs(pr["emitted_instances_mean"] - rr["emitted_instances_mean"]) > 1e-3:
            raise SystemExit(f"{arm}: emitted instances differ, replication is not the same work point")
        _, p_dev = ratio_scatter(pr, p_ctrl)
        _, r_dev = ratio_scatter(rr, r_ctrl)
        gate = arm in GATE_ARMS
        rows.append({
            "arm": arm,
            "primary_mean_ms": pr["mean_ms_session_averaged"],
            "replication_mean_ms": rr["mean_ms_session_averaged"],
            "mean_shift_pct": pct(rr["mean_ms_session_averaged"], pr["mean_ms_session_averaged"]),
            "median_of_rep_sessions_ms": round(statistics.median(rr["session_means_ms"]), 3),
            "median_shift_pct": pct(statistics.median(rr["session_means_ms"]),
                                    pr["mean_ms_session_averaged"]),
            "primary_session_means_ms": pr["session_means_ms"],
            "replication_session_means_ms": rr["session_means_ms"],
            "primary_spread_ms": pr["spread_of_session_means_ms"],
            "replication_spread_ms": rr["spread_of_session_means_ms"],
            "primary_spread_pct_of_own_mean": share(pr["spread_of_session_means_ms"],
                                                    pr["mean_ms_session_averaged"]),
            "replication_spread_pct_of_own_mean": share(rr["spread_of_session_means_ms"],
                                                        rr["mean_ms_session_averaged"]),
            "outside_own_primary_spread": bool(
                abs(rr["mean_ms_session_averaged"] - pr["mean_ms_session_averaged"])
                > pr["spread_of_session_means_ms"]),
            "control_normalised_shift_pct": pct(statistics.mean([m / c for m, c in zip(rr["session_means_ms"], r_ctrl)]),
                                                statistics.mean([m / c for m, c in zip(pr["session_means_ms"], p_ctrl)])),
            "primary_share_dev_pct_by_session": p_dev,
            "replication_share_dev_pct_by_session": r_dev,
            "primary_excursion_sessions": [i + 1 for i, d in enumerate(p_dev) if abs(d) > flag_at],
            "replication_excursion_sessions": [i + 1 for i, d in enumerate(r_dev) if abs(d) > flag_at],
            "probe_inflation_replication": rr["probe_inflation_median_by_session"],
            "probe_inflation_primary": pr["probe_inflation_median_by_session"],
            "contention_any_session_replication": rr["contention_any_session"],
            "emitted_instances": rr["emitted_instances_mean"],
            "reduction_vs_control_by_rep_session_pct": per_session_reduction(rr, r_ctrl),
            "reduction_vs_control_by_primary_session_pct": per_session_reduction(pr, p_ctrl),
            "time_gate_60pct_holds_in_every_rep_session": bool(
                gate and min(per_session_reduction(rr, r_ctrl)) >= 60.0) if gate else None,
            "worst_rep_session_margin_pp": round(min(per_session_reduction(rr, r_ctrl)) - 60.0, 2) if gate else None,
        })

    c10 = paired_premium(p_rows, r_rows, "sync_c_plus_10_f15", "bucket_c_plus_10_f15",
                         r_rows["bucket_c_plus_10_f15"]["emitted_instances_mean"]
                         - r_rows["sync_c_plus_10_f15"]["emitted_instances_mean"])
    p15 = paired_premium(p_rows, r_rows, "sync_ceil_1p5_f15", "bucket_ceil_1p5_f15",
                         r_rows["bucket_ceil_1p5_f15"]["emitted_instances_mean"]
                         - r_rows["sync_ceil_1p5_f15"]["emitted_instances_mean"])

    gate_rows = [r for r in rows if r["arm"] in GATE_ARMS]
    summary = {
        "version": "jrtip-rule-timing-replication-compare-v1",
        "primary": args.primary,
        "replication": args.replication,
        "arms": len(rows),
        "sessions_per_arm": 3,
        "frames_per_arm": rep["sessions_meta"][0]["frames_per_arm"],
        "gates_passed_before_timing": [s["gate"] for s in rep["sessions_meta"]],
        "mean_shift_pct_median_across_arms": round(statistics.median([r["mean_shift_pct"] for r in rows]), 3),
        "median_shift_pct_worst_across_arms": max(rows, key=lambda r: abs(r["median_shift_pct"]))["arm"],
        "arms_outside_own_primary_spread": [r["arm"] for r in rows if r["outside_own_primary_spread"]],
        "replication_spread_pct_of_own_mean_max": max(r["replication_spread_pct_of_own_mean"] for r in rows),
        "primary_spread_pct_of_own_mean_max": max(r["primary_spread_pct_of_own_mean"] for r in rows),
        "excursion_flag_threshold_pct_share": flag_at,
        "primary_excursions_arm_session": {r["arm"]: r["primary_excursion_sessions"] for r in rows
                                           if r["primary_excursion_sessions"]},
        "replication_excursions_arm_session": {r["arm"]: r["replication_excursion_sessions"] for r in rows
                                               if r["replication_excursion_sessions"]},
        "gate_arms": len(gate_rows),
        "gate_arms_pass_60pct_in_every_rep_session": all(
            r["time_gate_60pct_holds_in_every_rep_session"] for r in gate_rows),
        "smallest_worst_session_margin_pp": min(r["worst_rep_session_margin_pp"] for r in gate_rows),
        "paired_bucket_premium": [c10, p15],
    }

    out = {"summary": summary, "static_line_residuals": static_line_residuals(p_rows, r_rows),
           "per_arm": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'arm':26s} {'prim ms':>8s} {'rep ms':>8s} {'shift%':>7s} {'med%':>6s} "
          f"{'norm%':>6s} {'pSprd%':>7s} {'rSprd%':>7s} {'infl rep':>18s} "
          f"{'pEx':>5s} {'rEx':>5s} {'60gate':>7s}")
    for r in rows:
        gate = r["time_gate_60pct_holds_in_every_rep_session"]
        print(f"{r['arm']:26s} {r['primary_mean_ms']:8.3f} {r['replication_mean_ms']:8.3f} "
              f"{r['mean_shift_pct']:7.2f} {r['median_shift_pct']:6.2f} "
              f"{r['control_normalised_shift_pct']:6.2f} {r['primary_spread_pct_of_own_mean']:7.2f} "
              f"{r['replication_spread_pct_of_own_mean']:7.2f} "
              f"{('/'.join(f'{v:.2f}' for v in r['probe_inflation_replication'])):>18s} "
              f"{(','.join(map(str, r['primary_excursion_sessions'])) or '-'):>5s} "
              f"{(','.join(map(str, r['replication_excursion_sessions'])) or '-'):>5s} "
              f"{('-' if gate is None else ('PASS' if gate else 'FLIP')):>7s}")
    for prem in (c10, p15):
        print(f"bucket-minus-sync {prem['bucket_arm']} vs {prem['sync_arm']}: "
              f"primary per-session {prem['primary_per_session_premium_ms']} "
              f"(median {prem['primary_median_premium_ms']:.3f} ms), replication per-session "
              f"{prem['replication_per_session_premium_ms']} (median "
              f"{prem['replication_median_premium_ms']:.3f} ms), waste "
              f"{prem['waste_instances']:.4f} instances/frame")
    print("\nper-session share of that session's cap-100 control, deviation from the arm's own median (%):")
    for r in rows:
        print(f"  {r['arm']:26s} primary {r['primary_share_dev_pct_by_session']}   "
              f"replication {r['replication_share_dev_pct_by_session']}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("paired_bucket_premium",)}, indent=1, ensure_ascii=False))
    for tag, data in out["static_line_residuals"]["readings"].items():
        detail = " ".join(f"{a.split('_f15')[0].replace('_native','')}:{v['residual_ms']:+.3f}"
                          for a, v in data["per_arm"].items())
        print(f"static-line residuals, {tag}: anchor {data['anchor_ms']:.3f} ms, "
              f"max |residual| {data['max_abs_residual_ms']:.3f} ms  [{detail}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
