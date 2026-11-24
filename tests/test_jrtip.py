"""Synthetic CPU checks only: no real RGB/GT/checkpoints, network, CUDA, installs.

Run existing Python with -B. Native R50 uses random weights and a 64px synthetic
input with reduced proposals. These tests are NOT scientific pilot results.
"""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
from contextlib import nullcontext
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import roi_heads
from torchvision.ops import roi_align
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from src import jrtip_pilot as jp
from src.mva_pilot import SCHEMA

CPU = torch.device("cpu")


def fake_rows():
    rows = []
    for role, genotype, plant, count in [
        ("TRAIN", "g1", "t1", 6), ("TRAIN", "g1", "t2", 1), ("TRAIN", "g2", "t3", 3),
        ("V_SOURCE", "good", "s1", 2), ("V_SOURCE", "bad", "s2", 1),
        ("V_SHIFT", "good", "v1", 3)]:
        for i in range(count):
            rows.append(dict(image_id=f"{plant}_{i}", split_role=role, genotype=genotype,
                             biological_plant_id=plant, timestamp=f"{i:03d}", mask_status="valid",
                             height=8, width=12, rgb_path="synthetic/never_open.png",
                             label_path="synthetic/never_open_label.png"))
    return rows


def label(h=8, w=12):
    array = np.zeros((h, w), np.uint8)
    array[1:4, 2:6] = 1
    array[5:7, 8:11] = 31
    return array


def png_bytes(bright=.3, h=8, w=12):
    buffer = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), int(bright * 255), np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeModel(torch.nn.Module):
    def __init__(self, empty=False):
        super().__init__()
        # Stock transform would otherwise silently resize our tiny RGB to 640.
        self.transform = GeneralizedRCNNTransform(640, 1333, [0, 0, 0], [1, 1, 1])
        self.empty = empty
        self.seen = []

    def forward(self, images):
        transformed, _ = self.transform(images, None)
        self.seen.append((list(images[0].shape[-2:]), list(transformed.image_sizes[0]),
                          list(transformed.tensors.shape[-2:])))
        h, w = images[0].shape[-2:]
        empty = self.empty or float(images[0][0, 0, 0]) > .75
        n = 0 if empty else 1
        return [{"scores": torch.full((n,), .9), "labels": torch.ones(n, dtype=torch.int64),
                 "boxes": torch.tensor([[0., 0., float(w), float(h)]])[:n],
                 "masks": torch.ones((n, 1, h, w)) * .8}]


class GeometryAndInferenceTests(unittest.TestCase):
    def test_long_edge_rounding_ratios_and_padding(self):
        image = torch.rand(3, 7, 13)
        for scale in jp.SCALES:
            resized, meta = jp.resize_long(image, scale)
            self.assertEqual(max(resized.shape[-2:]), scale)
            self.assertEqual(meta["scaled_hw"], [round(7 * scale / 13), scale])
            self.assertEqual(meta["scale_xy"], [scale / 13, resized.shape[-2] / 7])
            self.assertTrue(all(v % 32 == 0 for v in meta["padded_hw"]))
        with self.assertRaises(ValueError):
            jp.resize_long(image, 0)

    def test_every_flip_rot_half_open_box_mask_mapping(self):
        target = jp.target_from_label(label())
        image = torch.rand(3, 8, 12)
        for horizontal in (False, True):
            for vertical in (False, True):
                for rotation in range(4):
                    params = dict(horizontal=horizontal, vertical=vertical, rot90=rotation, long_edge=12)
                    rgb, transformed, meta = jp.geometry(image, target, params)
                    reconstructed = np.zeros(rgb.shape[-2:], np.uint8)
                    for identity, mask in zip(transformed["leaf_ids"], transformed["masks"]):
                        reconstructed[mask.numpy().astype(bool)] = int(identity)
                    expected = jp.target_from_label(reconstructed)
                    torch.testing.assert_close(transformed["boxes"], expected["boxes"])
                    self.assertEqual(meta["long_edge"], 12)
        _, small, _ = jp.geometry(image, target, dict(horizontal=False, vertical=False, rot90=0, long_edge=1))
        self.assertEqual(len(small["labels"]), 2)
        self.assertEqual(int(small["masks"].sum()), 0)
        self.assertTrue(torch.all(small["boxes"][:, 2:] > small["boxes"][:, :2]))

    def test_probability_restored_before_threshold_and_box_inverse(self):
        _, meta = jp.resize_long(torch.zeros(3, 1, 4), 2)
        output = {"scores": torch.tensor([.9]), "labels": torch.tensor([1]),
                  "boxes": torch.tensor([[0., 0., 2., 1.]]),
                  "masks": torch.tensor([[[[.49, .9]]]])}
        payload = jp.restore_output(output, meta)
        from pycocotools import mask as mask_utils
        mask = mask_utils.decode(payload["instances"][0]["segmentation"])
        expected = torch.nn.functional.interpolate(output["masks"], (1, 4), mode="bilinear", align_corners=False)[0, 0] >= .5
        np.testing.assert_array_equal(mask.astype(bool), expected.numpy())
        wrong = torch.nn.functional.interpolate((output["masks"] >= .5).float(), (1, 4), mode="bilinear", align_corners=False)[0, 0] >= .5
        self.assertFalse(torch.equal(expected, wrong))
        self.assertEqual(payload["instances"][0]["box_xyxy"], [0., 0., 4., 1.])
        self.assertEqual(payload["count"], 1)

    def test_instance_local_no_second_resize_and_exception_restore(self):
        model, other = FakeModel().eval(), FakeModel().eval()
        other_resize = other.transform.resize
        before = model.transform.resize
        image = torch.zeros(3, 8, 12)
        payload = jp.infer_image(model, image, 12, CPU)
        self.assertEqual(model.seen, [([8, 12], [8, 12], [32, 32])])
        self.assertEqual(payload["original_hw"], [8, 12])
        self.assertEqual(model.transform.resize, before)
        self.assertEqual(other.transform.resize, other_resize)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with jp.no_second_resize(model):
                raise RuntimeError("synthetic")
        self.assertEqual(model.transform.resize, before)
        output = {"scores": torch.tensor([]), "boxes": torch.empty(0, 4), "masks": torch.empty(0, 1, 9, 9)}
        with self.assertRaisesRegex(RuntimeError, "implicit resize"):
            jp.restore_output(output, payload["geometry"])

    def test_empty_output_and_score_cutoffs(self):
        model = FakeModel(empty=True).eval()
        payload = jp.infer_image(model, torch.zeros(3, 8, 12), 12, CPU)
        self.assertEqual(payload["instances"], [])
        self.assertEqual(payload["count"], 0)
        _, meta = jp.resize_long(torch.zeros(3, 8, 12), 12)
        output = {"scores": torch.tensor([.0009, .001, .499, .5]),
                  "boxes": torch.tensor([[0., 0., 12., 8.]]).repeat(4, 1),
                  "masks": torch.ones(4, 1, 8, 12)}
        payload = jp.restore_output(output, meta)
        self.assertEqual(len(payload["instances"]), 3)
        self.assertEqual(payload["count"], 1)

    def test_service_clock_includes_postprocess_serialization_and_lookup(self):
        model = FakeModel().eval()
        t = [0.]
        def clock():
            return t[0]
        def supplier():
            t[0] += .002
            return png_bytes()
        restore = jp.restore_output
        def post(output, meta):
            t[0] += .007
            return restore(output, meta)
        dumps = json.dumps
        def serialize(*args, **kwargs):
            t[0] += .003
            return dumps(*args, **kwargs)
        with patch.object(jp, "restore_output", side_effect=post), patch.object(jp.json, "dumps", side_effect=serialize):
            data, record = jp.service(model, supplier, 12, CPU, (8, 12), clock)
        self.assertAlmostEqual(record["service_ms"], 12)
        self.assertAlmostEqual(record["decode_ms"], 2)
        self.assertAlmostEqual(record["restore_scores_masks_rle_serialize_sync_ms"], 10)
        self.assertEqual(record["serialized_bytes"], len(data))
        self.assertEqual(json.loads(data)["count"], 1)
        self.assertAlmostEqual(sum(v for k, v in record.items() if k.endswith("_ms") and k != "service_ms"), record["service_ms"])


class DataAndQualityTests(unittest.TestCase):
    def test_locked_and_quarantined_filtered_before_path_use(self):
        buffer = io.StringIO()
        fields = sorted(SCHEMA)
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for role in ("TRAIN", "V_SOURCE", "V_SHIFT", "C_POLICY", "T_ID", "T_GENO"):
            record = dict.fromkeys(fields, "")
            record.update(image_id=role, biological_plant_id=role, genotype="g", split_role=role,
                          width="12", height="8", rgb_path="synthetic/no.png",
                          label_path="synthetic/no_label.png", mask_status="valid")
            writer.writerow(record)
        record.update(image_id="quarantine", split_role="TRAIN", mask_status="quarantined")
        writer.writerow(record)
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "open", return_value=io.StringIO(buffer.getvalue())):
            rows = jp.read_inventory("synthetic/inventory.csv", data_root=Path("synthetic"))
        self.assertEqual({r["image_id"] for r in rows}, {"TRAIN", "V_SOURCE", "V_SHIFT"})
        for role in ("C_POLICY", "T_ID", "T_GENO"):
            with self.assertRaises(PermissionError):
                jp.ValidData.guard({"split_role": role, "mask_status": "valid"})
        with self.assertRaises(PermissionError):
            jp.input_bytes({"split_role": "TRAIN", "mask_status": "quarantined"})

    def test_dev_selection_time_order_subset_and_counts(self):
        rows = fake_rows()
        selected = jp.select_dev(rows, 1)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(r["split_role"] in jp.DEV_ROLES for r in selected))
        self.assertEqual(len(jp.select_dev(rows)), 6)
        frames = [dict(rows[0], image_id=f"i{i:02d}", timestamp=f"{i:03d}", split_role="V_SOURCE") for i in range(41)]
        timed = jp.select_dev(frames[::-1], 20)
        self.assertEqual(timed[0]["image_id"], "i00")
        self.assertEqual(timed[-1]["image_id"], "i40")
        self.assertEqual(len(timed), 20)
        self.assertEqual(len({r["image_id"] for r in timed}), 20)

    def test_readiness_requires_approved_matching_actual_valid_population(self):
        rows = fake_rows()
        counts = jp.population(rows)
        ready = {"approved_for_training": True, "pilot_inventory_sha256": "hash", "pilot_instances_sha256": "hash",
                 "counts_by_role": {r: {"frames": v["frames"], "plant_units": v["plants"]} for r, v in counts.items()}}
        with patch.object(Path, "read_text", return_value=json.dumps(ready)), \
                patch.object(jp, "file_sha256", return_value="hash"), \
                patch.object(jp, "read_inventory", return_value=rows), \
                patch.object(jp, "read_instances", return_value={}):
            actual, data, _ = jp.load_inputs("synthetic/i.csv", "synthetic/m.jsonl", "synthetic/r.json")
            self.assertEqual(actual, rows)
            self.assertIsInstance(data, jp.ValidData)
        ready["approved_for_training"] = False
        with patch.object(Path, "read_text", return_value=json.dumps(ready)):
            with self.assertRaisesRegex(ValueError, "approve"):
                jp.load_inputs("synthetic/i.csv", "synthetic/m.jsonl", "synthetic/r.json")

    def test_quality_full_collection_per_genotype_equal_domains_and_subset_flag(self):
        class Data:
            def label(self, row):
                return jp.target_from_label(np.ones((8, 12), np.uint8))
        model = FakeModel().train()
        state = torch.get_rng_state().clone()
        def blobs(row):
            return png_bytes(.9 if row["genotype"] == "bad" else .3)
        with patch.object(jp, "input_bytes", side_effect=blobs):
            report = jp.quality(model, fake_rows(), Data(), (12,), CPU)
            subset = jp.quality(model, fake_rows(), Data(), (12,), CPU, max_frames=1)
        q = report["scales"]["12"]
        self.assertAlmostEqual(q["role_macro"]["V_SOURCE"]["AP"], 50)
        self.assertAlmostEqual(q["role_macro"]["V_SHIFT"]["AP"], 100)
        self.assertAlmostEqual(q["equal_domain_macro"]["AP"], 75)
        self.assertAlmostEqual(q["equal_domain_macro"]["count_MAE"], .25)
        self.assertEqual(q["per_genotype"]["V_SHIFT"]["good"]["gt_instances"], 3)
        self.assertFalse(report["diagnostic_subset"])
        self.assertTrue(subset["diagnostic_subset"])
        self.assertEqual(len(subset["selected_image_ids"]), 3)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_coco_empty_predictions_ap_zero(self):
        gt = {"images": [{"id": 1, "height": 8, "width": 12}], "annotations": [{"id": 1}]}
        self.assertEqual(jp.coco_mask_ap(gt, []), {"AP": 0., "AP50": 0., "AP75": 0.})

    def test_predeclared_tolerances_are_domain_specific_and_equal(self):
        reference = {"AP": 80., "count_MAE": 1.}
        bad_source = {"AP": 78., "count_MAE": 1.3}
        good_shift = {"AP": 80., "count_MAE": 1.}
        q = {"diagnostic_subset": True, "scales": {}}
        for scale in jp.SCALES:
            q["scales"][str(scale)] = {"role_macro": {"V_SOURCE": reference, "V_SHIFT": reference},
                                      "equal_domain_macro": reference}
        q["scales"]["320"] = {"role_macro": {"V_SOURCE": bad_source, "V_SHIFT": good_shift},
                                    "equal_domain_macro": {"AP": 79., "count_MAE": 1.15}}
        rows = jp.scale_comparison(q, {str(s): {} for s in jp.SCALES})
        self.assertFalse(rows[0]["relative_to_S640"]["V_SOURCE"]["point_tolerances_met"])
        self.assertTrue(rows[0]["relative_to_S640"]["V_SHIFT"]["point_tolerances_met"])
        self.assertTrue(rows[0]["relative_to_S640"]["equal_domain_macro"]["point_tolerances_met"])
        self.assertTrue(rows[0]["diagnostic_subset"])
        self.assertEqual(rows[2]["relative_to_S640"]["equal_domain_macro"]["AP_drop_pp"], 0.)


class ResumeAndBenchmarkTests(unittest.TestCase):
    def test_defaults_schedule_and_out_guard(self):
        args = jp.parser().parse_args(["--action", "train", "--out", "jrtip/baseline_pilot_v1"])
        self.assertEqual((args.steps, args.seed, args.physical_batch, args.accumulation), (1000, 43, 1, 4))
        self.assertTrue(args.inventory.endswith("pilot_inventory.csv"))
        self.assertAlmostEqual(jp.lr_for_update(1), .0005)
        self.assertAlmostEqual(jp.lr_for_update(500), .005)
        self.assertAlmostEqual(jp.lr_for_update(1000), .005)
        self.assertAlmostEqual(jp.lr_for_update(3000), .005)
        self.assertIn("NOT the formal 36-epoch", jp.NOTE)
        self.assertEqual(jp.jp_path(args.out), ROOT / "jrtip/baseline_pilot_v1")
        for out in (str(ROOT / "mva/B1_pilot_v1"), str(ROOT / "shared"), "jrtip/../mva/new"):
            with self.assertRaises(ValueError):
                jp.jp_path(out)

    def test_small_cpu_optimizer_rng_sampler_augmentation_state_exact_resume(self):
        torch.manual_seed(43)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=1e-4)
        sampler, aug = jp.BalancedSampler(fake_rows()), jp.Augmentation()
        def update():
            row, params = sampler.draw(), aug.parameters()
            x = torch.rand(3, 4)
            extras = (random.random(), np.random.rand())
            optimizer.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            optimizer.step()
            return row["image_id"], params, extras
        update()
        config = {"steps_requested": 1000, "sources": "synthetic", "manifest": "synthetic"}
        checkpoint = jp.checkpoint_state(model, optimizer, sampler, aug, CPU, 1, config,
                                         {"score": None}, [.123], False)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        expected_draws = update()
        expected = copy.deepcopy(model.state_dict())
        jp.validate_resume(loaded, config, jp.jsonable(config))
        jp.restore_state(loaded, model, optimizer, sampler, aug, CPU)
        self.assertEqual(update(), expected_draws)
        for k, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[k]))
        self.assertEqual(loaded["next_optimizer_step"], 2)
        self.assertFalse(loaded["evaluation_complete"])
        self.assertEqual(loaded["timing_window"], [.123])
        for field in ("steps_requested", "sources", "manifest"):
            changed = dict(config, **{field: "changed"})
            with self.assertRaisesRegex(ValueError, "Exact resume refused"):
                jp.validate_resume(loaded, changed, jp.jsonable(config))

    def test_pending_checkpoint_saved_before_dev_failure(self):
        with patch.object(jp, "persist") as saved, patch.object(Path, "exists", return_value=False), \
                patch.object(jp, "quality", side_effect=RuntimeError("synthetic dev failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic dev failure"):
                jp.evaluate_boundary(ROOT / "jrtip/synthetic_never_created", None, None, None, None,
                                     CPU, [], None, 500, {"dev_max_frames_per_plant": 0}, {"score": None}, [])
            self.assertEqual(saved.call_count, 1)
            self.assertIs(saved.call_args.args[-1], False)

    def test_completed_self_hash_is_pointer_only_not_recursive_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.005)
        sampler, aug = jp.BalancedSampler(fake_rows()), jp.Augmentation()
        out = ROOT / "jrtip/synthetic_never_created"
        best = {"optimizer_step": 1, "score": 75., "checkpoint": str(out / "checkpoint_000001.pt"),
                "checkpoint_sha256": "stale"}
        with patch.object(jp, "save_state") as saved, patch.object(jp, "write_json"), \
                patch.object(jp, "file_sha256", return_value="fresh"):
            jp.persist(out, model, optimizer, sampler, aug, CPU, 1, {}, best, [], True)
        self.assertNotIn("checkpoint_sha256", saved.call_args.args[1]["best"])
        self.assertEqual(best["checkpoint_sha256"], "fresh")

    def test_synthetic_train_action_two_updates_accum4_final_dev_and_pointers(self):
        class Data:
            def label(self, row):
                return jp.target_from_label(np.ones((8, 12), np.uint8))
            def load(self, row):
                return torch.full((3, 8, 12), .3), self.label(row)
        class TrainModel(FakeModel):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))
                self.training_shapes = []
            def forward(self, images, targets=None):
                if self.training:
                    self.training_shapes.append(len(images))
                    return {"loss_classifier": self.weight.square() + images[0].mean() * self.weight}
                return super().forward(images)
        args = jp.parser().parse_args(["--action", "train", "--out", "jrtip/synthetic_never_created",
                                       "--steps", "2", "--device", "cpu", "--dev-max-frames-per-plant", "1"])
        model = TrainModel()
        config = {"version": jp.VERSION, "steps_requested": 2, "dev_max_frames_per_plant": 1}
        saves, writes, logs = [], [], []
        def saved(path, state):
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            saves.append(torch.load(buffer, weights_only=False))
        with patch.object(jp, "load_inputs", return_value=(fake_rows(), Data(), {})), \
                patch.object(jp, "file_sha256", return_value=jp.COCO_V1_SHA256), \
                patch.object(jp, "train_config", return_value=config), \
                patch.object(jp, "seed_backend"), patch.object(jp, "build_jp_detector", return_value=model), \
                patch.object(jp, "run_lock", return_value=nullcontext()), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(jp, "save_state", side_effect=saved), \
                patch.object(jp, "write_json", side_effect=lambda p, o: writes.append((Path(p).name, copy.deepcopy(o)))), \
                patch.object(jp, "append_jsonl", side_effect=lambda p, o: logs.append((Path(p).name, copy.deepcopy(o)))), \
                patch.object(jp, "status"), patch.object(jp, "machine_snapshot", return_value={"synthetic": True}), \
                patch.object(jp, "memory_snapshot", return_value={"synthetic": True}), \
                patch.object(jp, "input_bytes", side_effect=lambda row: png_bytes(.9 if row["genotype"] == "bad" else .3)):
            jp.train(args)
        self.assertEqual(model.training_shapes, [1] * 8)
        train_logs = [obj for name, obj in logs if name == "train.jsonl"]
        self.assertEqual([obj["optimizer_step"] for obj in train_logs], [1, 2])
        self.assertTrue(all(len(obj["raw_microbatch_losses"]) == 4 for obj in train_logs))
        self.assertTrue(all(len(obj["sample_stream"]) == 4 for obj in train_logs))
        self.assertEqual(saves[-1]["optimizer_step"], 2)
        self.assertTrue(saves[-1]["evaluation_complete"])
        self.assertTrue(any(s["evaluation_complete"] is False for s in saves))
        best = [obj for name, obj in writes if name == "best.json"][-1]
        self.assertEqual(best["optimizer_step"], 2)
        self.assertAlmostEqual(best["score"], 75.)
        self.assertTrue(best["diagnostic_subset"])
        self.assertEqual(best["checkpoint_sha256"], jp.COCO_V1_SHA256)
        self.assertTrue(any(name == "latest.json" for name, obj in writes))

    def test_three_session_aggregation_csv_json_and_plot_calls_in_memory_only(self):
        out = ROOT / "jrtip/synthetic_benchmark_never_created"
        manifest = {"warmup": 50, "timing_max_frames_per_plant": 20,
                    "timing_image_ids": ["s", "v"], "checkpoint": "synthetic/never_open.pt"}
        files = {"benchmark_config.json": json.dumps(manifest)}
        q = {"diagnostic_subset": False, "scales": {}}
        for scale in jp.SCALES:
            metrics = {"AP": 80., "count_MAE": 1.}
            q["scales"][str(scale)] = {"equal_domain_macro": metrics,
                                        "role_macro": {role: metrics for role in jp.DEV_ROLES}}
        files["quality.json"] = json.dumps(q)
        for session in (1, 2, 3):
            report = {"session": session, "completed": True, "benchmark_config_sha256": jp.digest_json(manifest),
                      "scales": {str(s): {"contention": False, "warmup": {"jitter": False}} for s in jp.SCALES}}
            files[f"session_{session}.json"] = json.dumps(report)
            records = []
            for scale in jp.SCALES:
                for role, identity in zip(jp.DEV_ROLES, ("s", "v")):
                    records.append(dict(session=session, scale=scale, image_id=identity, split_role=role,
                        genotype="g", plant=identity, service_ms=12., decode_ms=2., resize_transfer_ms=2.,
                        network_sync_ms=5., restore_scores_masks_rle_serialize_sync_ms=3.,
                        memory={"rss_bytes": 1, "peak_rss_bytes": 2, "cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0}))
            files[f"timing_session_{session}.jsonl"] = "\n".join(json.dumps(r) for r in records)
        writes = []
        csv_buffer = io.StringIO()
        csv_buffer.close = lambda: None
        fig, axes = MagicMock(), [MagicMock() for _ in range(3)]
        pyplot, matplotlib = MagicMock(), MagicMock()
        pyplot.subplots.return_value = fig, axes
        matplotlib.pyplot = pyplot
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "read_text", autospec=True, side_effect=lambda p, **kw: files[p.name]), \
                patch.object(Path, "open", return_value=csv_buffer), \
                patch.object(jp, "write_json", side_effect=lambda p, o: writes.append((Path(p).name, o))), \
                patch.object(jp, "status"), \
                patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
            jp.finalize_benchmark(out)
        report = [obj for name, obj in writes if name == "scales.json"][0]
        self.assertFalse(report["checkpoint_reselected"])
        self.assertEqual(len(report["rows"]), 3)
        self.assertTrue(all(r["requests_technical_repeats"] == 6 for r in report["rows"]))
        table = list(csv.DictReader(io.StringIO(csv_buffer.getvalue())))
        self.assertEqual([r["scale"] for r in table], ["320", "480", "640"])
        self.assertEqual(float(table[0]["peak_rss_bytes"]), 2.)
        self.assertEqual([call.args[0].suffix for call in fig.savefig.call_args_list], [".png", ".pdf"])

    def test_timing_percentiles_throughput_and_hierarchical_mean(self):
        records = []
        for session in (1, 2, 3):
            for role, genotype, plant, count, ms in [
                ("V_SOURCE", "g1", "p1", 4, 10.), ("V_SOURCE", "g1", "p2", 1, 30.),
                ("V_SOURCE", "g2", "p3", 1, 40.), ("V_SHIFT", "g3", "p4", 2, 60.)]:
                for i in range(count):
                    records.append(dict(session=session, split_role=role, genotype=genotype, plant=plant,
                        image_id=f"{plant}_{i}", service_ms=ms, decode_ms=ms / 4, resize_transfer_ms=ms / 4,
                        network_sync_ms=ms / 4, restore_scores_masks_rle_serialize_sync_ms=ms / 4,
                        memory={"rss_bytes": 1, "peak_rss_bytes": 2, "cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0}))
        result = jp.summarize_times(records)
        self.assertAlmostEqual(result["role_genotype_plant_equal_mean_ms"]["V_SOURCE"], 30.)
        self.assertAlmostEqual(result["equal_domain_mean_ms"], 45.)
        self.assertAlmostEqual(result["mean_ms"], 28.75)
        self.assertAlmostEqual(result["saturated_throughput_per_second"], 1000 / 28.75)
        self.assertEqual(result["p95_ms"], 60.)
        self.assertEqual(result["requests_technical_repeats"], 24)

    def test_warmup_smoke_flag_and_protocol_stability(self):
        model = FakeModel().eval()
        rows = jp.select_dev(fake_rows(), 1)
        blobs = {r["image_id"]: b"synthetic_not_decoded" for r in rows}
        with patch.object(jp, "service", return_value=(b"{}", {"service_ms": 5.})) as service:
            report = jp.warmup(model, blobs, rows, 320, CPU, 50)
            self.assertEqual(report["actual"], 60)
            self.assertTrue(report["stable_recent_three_groups20"])
            self.assertFalse(report["diagnostic_warmup"])
            smoke = jp.warmup(model, blobs, rows, 320, CPU, 2)
            self.assertTrue(smoke["diagnostic_warmup"])
            self.assertEqual(smoke["actual"], 2)
            self.assertEqual(service.call_count, 62)

    def test_nvidia_contention_records_other_process_without_killing(self):
        def command(args, **kwargs):
            if "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in args:
                return SimpleNamespace(returncode=0, stdout="GPU-test, 123456, existing_mva, 10000\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="synthetic telemetry", stderr="")
        with patch.object(jp.subprocess, "run", side_effect=command) as run:
            report = jp.nvidia_snapshot()
        self.assertTrue(report["contention"])
        self.assertFalse(report["exclusive_platform_guaranteed"])
        self.assertEqual(len(report["other_compute_processes"]), 1)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(c.args[0][0] == "nvidia-smi" for c in run.call_args_list))

    def test_benchmark_rejects_missing_developed_best_not_select_latest(self):
        values = [json.dumps({"version": jp.VERSION}), json.dumps({"score": None, "checkpoint": None}),
                  json.dumps({"checkpoint": "synthetic/latest.pt"})]
        with patch.object(Path, "read_text", side_effect=values), patch.object(jp.torch, "load") as loaded:
            with self.assertRaisesRegex(ValueError, "cannot select a new one"):
                jp.benchmark_checkpoint(ROOT / "jrtip/synthetic_never_created")
            loaded.assert_not_called()


class NativeArchitectureCPU(unittest.TestCase):
    def test_stock_native_roi_operator_and_loss_equivalence(self):
        feature = torch.rand(1, 2, 8, 8, requires_grad=True)
        boxes = [torch.tensor([[1., 2., 7., 6.]])]
        expected = roi_align(feature, boxes, (3, 3), sampling_ratio=2, aligned=False)
        actual = jp.native_roi_align(feature, boxes, (3, 3), sampling_ratio=2)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.)
        logits = torch.randn(2, 2, 28, 28, requires_grad=True)
        masks = [torch.stack([torch.ones(8, 8, dtype=torch.uint8), torch.zeros(8, 8, dtype=torch.uint8)])]
        boxes = [torch.tensor([[0., 0., 8., 8.], [0., 0., 8., 8.]])]
        labels, matched = [torch.ones(2, dtype=torch.int64)], [torch.arange(2)]
        expected = roi_heads.maskrcnn_loss(logits, boxes, masks, labels, matched)
        actual = jp._native_loss(logits, boxes, masks, labels, matched)
        torch.testing.assert_close(expected, actual)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[1].abs().sum()), 0.)  # zero-mass targets NOT skipped

    def test_random_r50_single_view_fp32_native_forward_backward_and_eval_no_resize(self):
        original_loss = roi_heads.maskrcnn_loss
        rng = torch.get_rng_state().clone()
        model = jp.build_jp_detector(seed=43, synthetic_smoke=True).train()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(roi_heads.maskrcnn_loss, original_loss)
        self.assertIsInstance(model.roi_heads, jp.JPRoIHeads)
        self.assertEqual(model.rpn.anchor_generator.sizes, ((8,), (16,), (32,), (64,), (128,)))
        self.assertFalse(hasattr(model, "auxiliary"))
        array = np.zeros((64, 64), np.uint8)
        array[8:36, 7:34] = 1
        array[45:52, 48:56] = 2
        image = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(43))
        target = jp.target_from_label(array)
        observed = []
        def hook(module, args, output):
            observed.append((output[0].image_sizes[0], output[0].tensors.shape[-2:]))
        handle = model.transform.register_forward_hook(hook)
        with jp.no_second_resize(model):
            losses = model([image], [target])
        self.assertEqual(set(losses), {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"})
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        sum(losses.values()).backward()
        self.assertIsNotNone(model.roi_heads.box_predictor.cls_score.weight.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.backbone.fpn.parameters() if p.grad is not None), 0.)
        model.eval()
        payload = jp.infer_image(model, image, 64, CPU)
        handle.remove()
        self.assertEqual(payload["original_hw"], [64, 64])
        self.assertEqual(observed, [((64, 64), torch.Size([64, 64])), ((64, 64), torch.Size([64, 64]))])
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertIs(roi_heads.maskrcnn_loss, original_loss)
        print("Synthetic CPU random R50-FPN 64px single-view forward/backward + native eval PASS; NOT scientific results.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    unittest.main(verbosity=2)
