"""Where in each timing pass does the largest single-arm excursion sit?

The replication compare (jrtip_rule_timing_replication_compare.py) flags two
sessions whose arm means sit far outside their own cross-session band, and both
flagged arms happen to be the eighth arm of a rotated order.  Rotating the arm
order is supposed to spread any order-dependent drift evenly, so an excursion
that keeps landing on the same ordinal position across independent processes is
worth recording as an instrument observation rather than an arm property.

This script reads the six already-written session reports (three primary, three
replication), recomputes the control-normalised reading that section 21 rule 2
prescribes, and reports which ordinal slot holds the largest normalised
deviation in every pass.  Each flagged slot also gets a wall-clock window
reconstructed backwards from the session's own ``generated_utc`` stamp and the
recorded per-arm ``loop_seconds``, so the observation can be compared against
anything outside this harness.  It sends no requests; it only re-reads
artifacts.

    python src/jrtip_rule_timing_slot_probe.py

Writes jrtip/rule_timing_v1_rep/slot_probe.json.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASSES = {
    "primary": ROOT / "jrtip" / "rule_timing_v1",
    "replication": ROOT / "jrtip" / "rule_timing_v1_rep",
}
CONTROL = "static100_native"
OUT = PASSES["replication"] / "slot_probe.json"


def read_session(base: Path, session: int) -> dict:
    payload = json.loads(
        (base / f"session{session}" / f"session_{session}.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload["per_arm"]
    return {
        "generated_utc": payload["generated_utc"],
        "order": list(payload["arm_order"]),
        "means": {a: row["summary"]["mean_ms"] for a, row in rows.items()},
        "inflation": {a: row["probes"]["inflation_median_vs_fastest"] for a, row in rows.items()},
        "seconds": {a: row["loop_seconds"] for a, row in rows.items()},
    }


def reconstructed_window(sessions: list[dict], s_idx: int, slot: int) -> tuple[str, str]:
    """Best-effort arm window, anchored on the session end and summed backwards.

    Per-arm report writing is not included in ``loop_seconds``, so both bounds
    sit a little later than the truth by the accumulated writing time of the
    arms that follow; the window length itself is exact.  For the eighth arm of
    eleven that drift is well under a minute, which is all the observation
    needs.
    """
    sess = sessions[s_idx]
    end = datetime.strptime(sess["generated_utc"], "%Y-%m-%dT%H:%M:%SZ")
    tail = sess["order"][slot:]
    arm_end = end - timedelta(seconds=sum(sess["seconds"][a] for a in tail))
    arm_start = arm_end - timedelta(seconds=sess["seconds"][sess["order"][slot - 1]])
    return arm_start.strftime("%H:%M:%S"), arm_end.strftime("%H:%M:%S")


def main() -> None:
    report: dict = {"control_arm": CONTROL, "passes": {}}
    for tag, base in PASSES.items():
        sessions = [read_session(base, s) for s in (1, 2, 3)]
        orders = [s["order"] for s in sessions]
        if any(len(set(o)) != len(o) for o in orders):
            raise SystemExit(f"{tag}: duplicated arm in a recorded arm_order")
        means = [s["means"] for s in sessions]
        ratios = {
            arm: [
                means[s][arm] / means[s][CONTROL] for s in range(len(means))
            ]
            for arm in means[0]
        }
        rows = []
        for s in range(3):
            deviations = {}
            for arm, series in ratios.items():
                med = statistics.median(series)
                deviations[arm] = 100.0 * (series[s] - med) / med
            ranked = sorted(deviations, key=lambda a: abs(deviations[a]), reverse=True)
            top_slot = orders[s].index(ranked[0]) + 1
            window = reconstructed_window(sessions, s, top_slot)
            entries = []
            for arm in ranked[:3]:
                entries.append(
                    {
                        "arm": arm,
                        "slot": orders[s].index(arm) + 1,
                        "normalised_deviation_pct": round(deviations[arm], 2),
                        "session_mean_ms": round(means[s][arm], 3),
                        "probe_inflation": round(sessions[s]["inflation"][arm], 4),
                    }
                )
            rows.append(
                {
                    "session": s + 1,
                    "session_generated_utc": sessions[s]["generated_utc"],
                    "max_deviation_slot": top_slot,
                    "reconstructed_window_utc": list(window),
                    "top_by_abs_deviation": entries,
                }
            )
            print(
                f"{tag} session {s + 1}: max |normalised deviation| at slot {top_slot} "
                f"-> {entries[0]['arm']} ({entries[0]['normalised_deviation_pct']:+.2f}%), "
                f"window ~{window[0]}-{window[1]} UTC"
            )
            for item in entries:
                print(
                    f"    slot{item['slot']:2d} {item['arm']:<24s} "
                    f"{item['normalised_deviation_pct']:+6.2f}%  "
                    f"mean {item['session_mean_ms']:7.2f} ms  "
                    f"inflation {item['probe_inflation']}"
                )
        max_slots = [r["max_deviation_slot"] for r in rows]
        report["passes"][tag] = {
            "sessions": rows,
            "max_deviation_slot_per_session": max_slots,
            "all_three_sessions_same_slot": len(set(max_slots)) == 1,
            "largest_abs_deviation_pct": max(
                abs(r["top_by_abs_deviation"][0]["normalised_deviation_pct"])
                for r in rows
            ),
        }
        print(f"{tag}: max-deviation slots per session = {max_slots}")
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
