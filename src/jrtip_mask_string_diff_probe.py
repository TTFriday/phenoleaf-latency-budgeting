"""Explain the stored-mask mismatches that the prefix-identity check reports.

``src/jrtip_budget_policy.py`` compares every archived budget stream against the matching prefix of
the cap-100 development stream.  Scores, boxes, and per-frame lengths match exactly; a handful of
``segmentation`` RLE strings differ - and yet re-scoring the prefixes reproduces every measured macro
metric to zero.  Two readings are possible and they mean different things for the protocol: either the
mismatch is only in bytes the COCO decoder never looks at (run-end / padding after ``height*width``
bits), or the masks really differ and their contribution to AP is below double-precision visibility.

This probe decodes both RLEs for each mismatching instance and reports which case it is, so the claim
in the write-up can be made from a measurement rather than from an inference.

Read-only, CPU-only, development roles only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jrtip_budget_policy import instances_of, prefix  # noqa: E402
from src.verify_jrtip_budget_quality import read_stream  # noqa: E402


def decode(rle) -> np.ndarray:
    payload = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")
               if isinstance(rle["counts"], str) else rle["counts"]}
    return np.asarray(mask_utils.decode(payload), dtype=np.uint8)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "jrtip/detection_budget_quality_v1"))
    parser.add_argument("--caps", default="50,30,25,20,10,5")
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_policy_v1/mask_string_diff_probe.json"))
    args = parser.parse_args()

    root = Path(args.root)
    base = read_stream(root / "cap00100" / "quality_000100_S640.jsonl")
    report = {"control_stream_frames": len(base), "caps": {}}
    for cap in [int(item) for item in args.caps.split(",") if item.strip()]:
        path = root / f"cap{cap:05d}" / f"quality_{cap:06d}_S640.jsonl"
        if not path.exists():
            continue
        other = read_stream(path)
        cases, byte_only, geometric = [], 0, 0
        for left, right in zip(base, other):
            expected, actual = prefix(left, cap), instances_of(right)
            if len(expected) != len(actual):
                continue
            for index, (a, b) in enumerate(zip(expected, actual)):
                if a["segmentation"] == b["segmentation"]:
                    continue
                first, second = decode(a["segmentation"]), decode(b["segmentation"])
                same_pixels = bool(np.array_equal(first, second))
                overlap = iou(first, second)
                byte_only += int(same_pixels)
                geometric += int(not same_pixels)
                cases.append({"image_id": left["image_id"], "split_role": left["split_role"],
                              "genotype": left["genotype"], "instance_rank": index,
                              "score": float(a["score"]), "reported_leaf": bool(a["score"] >= 0.5),
                              "size_a": list(a["segmentation"]["size"]),
                              "size_b": list(b["segmentation"]["size"]),
                              "counts_len_a": len(str(a["segmentation"]["counts"])),
                              "counts_len_b": len(str(b["segmentation"]["counts"])),
                              "pixels_identical": same_pixels, "iou": round(overlap, 9),
                              "area_a": int(first.sum()), "area_b": int(second.sum())})
        report["caps"][str(cap)] = {
            "mismatching_instances": len(cases), "identical_after_decode": byte_only,
            "genuinely_different_masks": geometric,
            "min_iou_among_different": None if not cases else round(min(c["iou"] for c in cases), 9),
            "reported_leaves_affected": sum(1 for c in cases if c["reported_leaf"]),
            "cases": cases[:40]}
        print(f"cap {cap}: {len(cases)} RLE-string mismatches, {byte_only} identical after decode, "
              f"{geometric} geometrically different", flush=True)
        del other
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
