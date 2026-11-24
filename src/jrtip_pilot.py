"""Independent COCO R50-FPN JRTIP P1 bottleneck pilot (not the formal 36 epochs).

No downloads, AMP, controllers, ROI routing, queue replay, or locked-role access.
CPU tests use synthetic data only. Real actions are explicit CLI invocations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import torchvision
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN, roi_heads
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops import poolers
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as TF
from src.mva_pilot import (BalancedSampler, LeafData, read_inventory, read_instances,
                           target_from_label, coco_ground_truth, coco_mask_ap, encode_mask,
                           global_rng_state, restore_global_rng, jsonable, write_json,
                           append_jsonl, trim_jsonl_for_resume, DEFAULT_COCO)
from src.mva_model import COCO_V1_SHA256, file_sha256, _ORIGINAL_MASK_LOSS

VERSION = "jrtip-p1-v1.1"
SCALES = (320, 480, 640)
DEV_ROLES = ("V_SOURCE", "V_SHIFT")
NOTE = ("Predeclared P1 bottleneck pilot: default 1000 optimizer updates, effective "
        "batch 4 (physical 1 x accumulation 4), seed 43, FP32, independent generic "
        "COCO R50-FPN initialization; multi-scale long edge 320/480/640. A shorter "
        "pilot first measures bottlenecks. SGD .005/.9/1e-4, 500-update linear "
        "warmup from .0005, then constant .005. Even a longer --steps run uses "
        "this pilot schedule, NOT the formal 36-epoch 24/32-epoch decay schedule. "
        "Formal 36-epoch training has NOT been run by this script's implementation; "
        "pilot outputs must not be described as full formal results. Selection "
        "uses only V_SOURCE/V_SHIFT at long edge 640; benchmark does not reselect "
        "weights. Relative S640 tolerances were fixed before measurement: AP "
        "drop <=1 percentage point and count MAE increase <=0.2. Development "
        "point-estimate feasibility is not a confirmatory noninferiority claim.")


def digest_json(obj):
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True,
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def jp_path(value):
    """All writable run paths stay within the dedicated JRTIP tree."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to((ROOT / "jrtip").resolve()) or path == (ROOT / "jrtip").resolve():
        raise ValueError("--out/--run must be a run subdirectory of phenoleaf_experiments/jrtip")
    return path


@contextmanager
def run_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".jp_lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"JP run already locked: {lock}; never stop another process") from None
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def status(out, phase, step=0, **extra):
    write_json(Path(out) / "status.json", dict(version=VERSION, phase=phase,
               optimizer_step=step, pid=os.getpid(), timestamp=time.time(),
               predeclared_pilot_note=NOTE, formal_36_epoch_run=False, **extra))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_backend(seed, device):
    # These flags belong only to this explicitly launched JP process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def nvidia_snapshot():
    commands = {
        "compute": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        "telemetry": ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw", "--format=csv,noheader,nounits"],
    }
    result = {}
    for key, command in commands.items():
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[key] = {"returncode": run.returncode, "stdout": run.stdout,
                           "stderr": run.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = {"error": str(exc)}
    others = []
    for line in result["compute"].get("stdout", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) != os.getpid():
            others.append({"gpu_uuid": parts[0], "pid": int(parts[1]), "name": parts[2]})
    result["other_compute_processes"] = others
    result["contention"] = bool(others)
    result["compute_query_ok"] = result["compute"].get("returncode") == 0
    result["exclusive_platform_guaranteed"] = False
    return result


def gpu_start_policy(snapshot, allow_shared_device=False):
    # On this Windows/WDDM workstation the compute-apps query also lists ordinary
    # browser/desktop contexts. It is not evidence that all listed apps train a
    # model. Do not request privileges or stop them to make the list empty.
    if not snapshot.get("compute_query_ok"):
        raise RuntimeError("Cannot record GPU process state; JP training refused")
    if snapshot.get("contention") and not allow_shared_device:
        raise RuntimeError("Shared GPU contexts detected; explicit shared-device mode required")
    return {"allow_shared_device": bool(allow_shared_device),
            "other_gpu_contexts_observed": bool(snapshot.get("contention")),
            "exclusive_platform_guaranteed": False,
            "note": "Shared desktop GPU; parent-queue guard separately prevents simultaneous project training."}


def memory_snapshot(device):
    """Process RSS/peak RSS without adding a psutil dependency."""
    rss, peak = None, None
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            rss, peak = int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    else:
        try:
            import resource
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
            if Path("/proc/self/statm").is_file():
                rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (ImportError, OSError, ValueError):
            pass
    return {"rss_bytes": rss, "peak_rss_bytes": peak,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}


def machine_snapshot(device):
    # Read-only OS/GPU telemetry. No process termination or environment installs.
    command = (["powershell.exe", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"]
               if os.name == "nt" else ["ps", "-eo", "pid,ppid,comm,args"])
    try:
        processes = subprocess.run(command, capture_output=True, text=True, timeout=15)
        processes = {"stdout": processes.stdout, "stderr": processes.stderr,
                     "returncode": processes.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        processes = {"error": str(exc)}
    return {"timestamp": time.time(), "pid": os.getpid(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "device": str(device),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "processes": processes, "memory": memory_snapshot(device),
            "nvidia": nvidia_snapshot() if device.type == "cuda" else None,
            "exclusive_platform_guaranteed": False}


def versions():
    return {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "PIL": Image.__version__, "cuda_runtime": torch.version.cuda,
            **{name: importlib.metadata.version(name) for name in ("scipy", "pycocotools")},
            "platform": platform.platform()}


def native_roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    """Native seeded, NOT bitwise CUDA ROIAlign; no torch.compile/Triton fallback."""
    if isinstance(boxes, (list, tuple)):
        boxes = poolers._convert_to_roi_format(boxes)
    size = (output_size, output_size) if isinstance(output_size, int) else output_size
    return torch.ops.torchvision.roi_align(input, boxes, float(spatial_scale),
                                         int(size[0]), int(size[1]), sampling_ratio, aligned)


class NativePool(MultiScaleRoIAlign):
    """Same torchvision FPN level mapping and aligned=False native ROI operator."""
    def forward(self, x, boxes, image_shapes):
        maps = poolers._filter_input(x, self.featmap_names)
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = poolers._setup_scales(maps, image_shapes,
                                                                self.canonical_scale, self.canonical_level)
        rois = poolers._convert_to_roi_format(boxes)
        if len(maps) == 1:
            return native_roi_align(maps[0], rois, self.output_size, self.scales[0], self.sampling_ratio)
        levels = self.map_levels(boxes)
        result = maps[0].new_zeros((len(rois), maps[0].shape[1], *self.output_size))
        for level, (feature, scale) in enumerate(zip(maps, self.scales)):
            ids = torch.where(levels == level)[0]
            if len(ids):
                pooled = native_roi_align(feature, rois[ids], self.output_size, scale, self.sampling_ratio)
                result = result.index_copy(0, ids, pooled.to(result.dtype))
        return result


def local_function(function, **replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


# Bind an isolated copy of the STOCK mask loss and RoIHeads.forward, changing
# only the ROIAlign dispatch. Zero-mass targets still participate in stock BCE.
# Do not monkeypatch torchvision or either MVA module/process.
_native_project = local_function(roi_heads.project_masks_on_boxes, roi_align=native_roi_align)
_native_loss = local_function(_ORIGINAL_MASK_LOSS, project_masks_on_boxes=_native_project)


class JPRoIHeads(roi_heads.RoIHeads):
    forward = local_function(roi_heads.RoIHeads.forward, maskrcnn_loss=_native_loss)


def build_jp_detector(weights_path=None, seed=43, synthetic_smoke=False):
    """Local COCO V1 -> fresh 2-class heads; no PhenoLeaf/MVA checkpoint accepted.

    None/synthetic_smoke exist exclusively for synthetic unit tests. CLI training
    always checks the full generic COCO SHA256 and uses the full candidate sizes.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = _resnet_fpn_extractor(resnet50(weights=None, norm_layer=FrozenBatchNorm2d), trainable_layers=3)
        kwargs = dict(min_size=640, max_size=640, _skip_resize=True,
                      box_score_thresh=.001, box_nms_thresh=.5, box_detections_per_img=100,
                      rpn_nms_thresh=.7, rpn_pre_nms_top_n_train=1000,
                      rpn_post_nms_top_n_train=1000, rpn_pre_nms_top_n_test=1000,
                      rpn_post_nms_top_n_test=1000)
        if synthetic_smoke:
            kwargs.update(rpn_pre_nms_top_n_train=32, rpn_post_nms_top_n_train=16,
                          rpn_pre_nms_top_n_test=16, rpn_post_nms_top_n_test=8,
                          box_batch_size_per_image=16, rpn_batch_size_per_image=32)
        model = MaskRCNN(backbone, num_classes=91, **kwargs)
        for module in model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0
        if weights_path is not None:
            if file_sha256(weights_path) != COCO_V1_SHA256:
                raise ValueError("Only SHA-pinned generic COCO V1 initialization is allowed")
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        model.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in (8, 16, 32, 64, 128)),
                                                     ((.5, 1., 2.),) * 5)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
        model.roi_heads.__class__ = JPRoIHeads
        model.roi_heads.box_roi_pool = NativePool(["0", "1", "2", "3"], 7, 2)
        model.roi_heads.mask_roi_pool = NativePool(["0", "1", "2", "3"], 14, 2)
    return model


@contextmanager
def no_second_resize(model):
    """Temporarily bypass only this transform's resize; normalize/pad remain."""
    transform = model.transform
    previous = transform.__dict__.get("resize")
    existed = "resize" in transform.__dict__
    transform.resize = lambda image, target=None: (image, target)
    try:
        yield
    finally:
        if existed:
            transform.resize = previous
        else:
            del transform.resize


def resize_long(image, long_edge):
    if long_edge < 1 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Positive long edge and CHW RGB required")
    h, w = image.shape[-2:]
    factor = long_edge / max(h, w)
    nh, nw = max(1, round(h * factor)), max(1, round(w * factor))
    resized = F.interpolate(image[None], (nh, nw), mode="bilinear", align_corners=False, antialias=True)[0]
    meta = {"original_hw": [h, w], "scaled_hw": [nh, nw],
            "scale_xy": [nw / w, nh / h], "long_edge": long_edge,
            "padding_right_bottom": [(-nw) % 32, (-nh) % 32],
            "padded_hw": [nh + (-nh) % 32, nw + (-nw) % 32]}
    return resized, meta


def geometry(image, target, params):
    image = image.clone()
    target = {k: v.clone() for k, v in target.items()}
    masks, boxes = target["masks"], target["boxes"]
    h, w = image.shape[-2:]
    if params["horizontal"]:
        image, masks = image.flip(-1), masks.flip(-1)
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if params["vertical"]:
        image, masks = image.flip(-2), masks.flip(-2)
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    for _ in range(params["rot90"]):
        image, masks = image.rot90(1, (-2, -1)), masks.rot90(1, (-2, -1))
        old = boxes.clone()
        boxes[:, 0], boxes[:, 2] = old[:, 1], old[:, 3]
        boxes[:, 1], boxes[:, 3] = w - old[:, 2], w - old[:, 0]
        h, w = w, h
    image, meta = resize_long(image, params["long_edge"])
    nh, nw = meta["scaled_hw"]
    target["masks"] = (F.interpolate(masks[:, None].float(), (nh, nw), mode="nearest")[:, 0].to(torch.uint8)
                       if len(masks) else torch.empty((0, nh, nw), dtype=torch.uint8))
    boxes *= boxes.new_tensor(meta["scale_xy"] * 2)
    target["boxes"] = boxes
    # Preserve half-open analytic boxes/classes even if a tiny resized mask vanishes.
    return image, target, meta


class Augmentation:
    def __init__(self, seed=43):
        self.rng = random.Random(seed + 101)

    def parameters(self):
        r = self.rng
        return {"horizontal": r.random() < .5, "vertical": r.random() < .5,
                "rot90": r.randrange(4), "long_edge": r.choice(SCALES),
                "color": r.random() < .8, "brightness": r.uniform(.8, 1.2),
                "contrast": r.uniform(.8, 1.2), "saturation": r.uniform(.8, 1.2),
                "hue": r.uniform(-.05, .05)}

    def apply(self, image, target):
        params = self.parameters()
        image, target, meta = geometry(image, target, params)
        if params["color"]:
            image = TF.adjust_brightness(image, params["brightness"])
            image = TF.adjust_contrast(image, params["contrast"])
            image = TF.adjust_saturation(image, params["saturation"])
            image = TF.adjust_hue(image, params["hue"])
        return image.clamp(0, 1), target, {"parameters": params, "geometry": meta}

    def state_dict(self):
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])


class ValidData(LeafData):
    @staticmethod
    def guard(row):
        if row["split_role"] not in ("TRAIN", *DEV_ROLES) or row["mask_status"].lower() != "valid":
            raise PermissionError("JP only opens valid TRAIN/V_SOURCE/V_SHIFT data")

    def label(self, row):
        self.guard(row)
        return super().label(row)

    def load(self, row):
        self.guard(row)
        return super().load(row)


def input_bytes(row):
    ValidData.guard(row)
    blob = Path(row["rgb_path"]).read_bytes()
    if row.get("rgb_sha256") and hashlib.sha256(blob).hexdigest() != row["rgb_sha256"]:
        raise ValueError(f"RGB hash changed: {row['image_id']}")
    return blob


def decode(blob, expected_hw=None):
    with Image.open(io.BytesIO(blob)) as im:
        array = np.asarray(im.convert("RGB")).copy()
    if expected_hw is not None and list(array.shape[:2]) != list(expected_hw):
        raise ValueError("RGB dimensions differ from inventory")
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def select_dev(rows, max_frames=0):
    if max_frames < 0:
        raise ValueError("Frame subset limit must be >=0")
    groups = defaultdict(list)
    for row in rows:
        if row["split_role"] in DEV_ROLES and row["mask_status"].lower() == "valid":
            groups[(row["split_role"], row["genotype"], row["biological_plant_id"])].append(row)
    selected = []
    for key in sorted(groups):
        frames = sorted(groups[key], key=lambda r: (r["timestamp"], r["image_id"]))
        if max_frames and len(frames) > max_frames:
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
        selected.extend(frames)
    return selected


def population(rows):
    result = {}
    for role in ("TRAIN", *DEV_ROLES):
        selected = [r for r in rows if r["split_role"] == role]
        result[role] = {"frames": len(selected), "plants": len({r["biological_plant_id"] for r in selected}),
                        "genotypes": len({r["genotype"] for r in selected})}
    return result


def load_inputs(inventory, instances, readiness, data_root=None):
    ready = json.loads(Path(readiness).read_text(encoding="utf-8"))
    if ready.get("approved_for_training") is not True:
        raise ValueError("Pilot readiness does not approve TRAIN")
    if file_sha256(inventory) != ready.get("pilot_inventory_sha256"):
        raise ValueError("Inventory differs from approved pilot readiness")
    if file_sha256(instances) != ready.get("pilot_instances_sha256"):
        raise ValueError("Instances differ from approved pilot readiness")
    rows = read_inventory(inventory, valid_status="valid", data_root=data_root)
    counts = population(rows)
    for role, count in counts.items():
        expected = ready["counts_by_role"][role]
        if (count["frames"], count["plants"]) != (expected["frames"], expected["plant_units"]):
            raise ValueError(f"Actual valid {role} population differs from readiness")
        if not count["frames"]:
            raise ValueError(f"No valid {role} rows")
    plants = defaultdict(set)
    for row in rows:
        plants[row["biological_plant_id"]].add(row["split_role"])
        if row.get("n_instances") and int(row["n_instances"]) > 100:
            raise ValueError("More than 100 leaves: predeclare a revised detector cap before training")
    if any(len(roles) != 1 for roles in plants.values()):
        raise ValueError("Biological plant crosses allowed roles")
    data = ValidData(rows, read_instances(instances, {r["image_id"] for r in rows if r["split_role"] == "TRAIN"}))
    return rows, data, ready


def restore_output(output, meta):
    h, w = meta["original_hw"]
    nh, nw = meta["scaled_hw"]
    scores = output["scores"].float().cpu()
    probs = output["masks"].float()
    if probs.shape[-2:] != (nh, nw):
        raise RuntimeError("Model output not at scaled RGB size: implicit resize/padding contract broken")
    if len(probs):
        probs = F.interpolate(probs, (h, w), mode="bilinear", align_corners=False)[:, 0].cpu()
    else:
        probs = torch.empty((0, h, w), dtype=torch.float32)
    masks = (probs >= .5).numpy()  # only AFTER soft probability restoration
    boxes = output["boxes"].float().cpu() / torch.tensor(meta["scale_xy"] * 2)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    labels = output.get("labels", torch.ones(len(scores), dtype=torch.int64)).cpu()
    instances = []
    for i in torch.where((scores >= .001) & (labels == 1))[0].tolist():
        instances.append({"score": float(scores[i]), "box_xyxy": boxes[i].tolist(),
                          "segmentation": encode_mask(masks[i])})
    return {"original_hw": [h, w], "geometry": meta, "instances": instances,
            "count": sum(x["score"] >= .5 for x in instances),
            "ap_score_threshold": .001, "count_score_threshold": .5, "mask_threshold": .5}


@torch.inference_mode()
def infer_image(model, image, scale, device):
    if model.training:
        raise ValueError("Inference requires eval model")
    scaled, meta = resize_long(image, scale)
    with no_second_resize(model):
        output = model([scaled.to(device)])[0]
    return restore_output(output, meta)


@torch.inference_mode()
def service(model, blob_supplier, scale, device, expected_hw=None, clock=time.perf_counter):
    """Byte retrieval -> decode -> resize -> network -> restore/RLE -> JSON -> sync.

    No GT, disk reads, result cache, or controller. Staged sync costs are included.
    """
    if model.training:
        raise ValueError("Service requires eval model")
    sync(device)
    start = clock()
    blob = blob_supplier()
    image = decode(blob, expected_hw)
    decoded = clock()
    scaled, meta = resize_long(image, scale)
    tensor = scaled.to(device)
    sync(device)
    resized = clock()
    with no_second_resize(model):
        output = model([tensor])[0]
    sync(device)
    network = clock()
    payload = restore_output(output, meta)
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    sync(device)
    end = clock()
    stages = {"decode_ms": (decoded - start) * 1000, "resize_transfer_ms": (resized - decoded) * 1000,
              "network_sync_ms": (network - resized) * 1000,
              "restore_scores_masks_rle_serialize_sync_ms": (end - network) * 1000}
    return serialized, {"service_ms": (end - start) * 1000, **stages,
                        "serialized_bytes": len(serialized), "geometry": meta}


@torch.inference_mode()
def quality(model, rows, data, scales, device, max_frames=0, out=None, step=0):
    selected = select_dev(rows, max_frames)
    if {r["split_role"] for r in selected} != set(DEV_ROLES):
        raise ValueError("Both valid development domains required")
    state, training = global_rng_state(device), model.training
    model.eval()
    groups = defaultdict(list)
    for row in selected:
        groups[(row["split_role"], row["genotype"])].append(row)
    result = {str(s): {} for s in scales}
    started = time.perf_counter()
    try:
        for (role, genotype), frames in sorted(groups.items()):
            if out:
                status(out, "quality_genotype", step, role=role, genotype=genotype)
            gt = coco_ground_truth(frames, data)  # full set AP, never frame AP mean
            predictions = {s: [] for s in scales}
            plants = {s: defaultdict(list) for s in scales}
            for image_id, row in enumerate(frames, 1):
                target = data.label(row)
                image = decode(input_bytes(row), (row["height"], row["width"]))
                for scale in scales:
                    payload = infer_image(model, image, scale, device)
                    predictions[scale].extend({"image_id": image_id, "category_id": 1,
                        "score": p["score"], "segmentation": p["segmentation"]} for p in payload["instances"])
                    error = abs(payload["count"] - len(target["masks"]))
                    plants[scale][row["biological_plant_id"]].append(error)
                    if out:
                        append_jsonl(Path(out) / f"quality_{step:06d}_S{scale}.jsonl",
                                     {"image_id": row["image_id"], "split_role": role, "genotype": genotype,
                                      "plant": row["biological_plant_id"], "count_abs_error": error,
                                      "gt_count": len(target["masks"]), **payload})
            for scale in scales:
                metrics = coco_mask_ap(gt, predictions[scale])
                metrics.update(count_MAE=float(np.mean([np.mean(v) for v in plants[scale].values()])),
                               frames=len(frames), plants=len(plants[scale]), gt_instances=len(gt["annotations"]))
                result[str(scale)].setdefault(role, {})[genotype] = metrics
        for scale in scales:
            per_genotype = result[str(scale)]
            roles = {role: {key: float(np.mean([g[key] for g in per_genotype[role].values()]))
                            for key in ("AP", "AP50", "AP75", "count_MAE")} for role in DEV_ROLES}
            result[str(scale)] = {"per_genotype": per_genotype, "role_macro": roles,
                                 "equal_domain_macro": {key: float(np.mean([roles[r][key] for r in DEV_ROLES]))
                                                        for key in ("AP", "AP50", "AP75", "count_MAE")}}
        return {"optimizer_step": step, "scales": result, "diagnostic_subset": bool(max_frames),
                "quality_max_frames_per_plant": max_frames, "selected_population": population(selected),
                "selected_image_ids": [r["image_id"] for r in selected],
                "selected_stream_sha256": digest_json([r["image_id"] for r in selected]),
                "AP_unit": "percentage_points", "count_aggregation": "frame -> plant -> genotype -> equal dev domain",
                "AP_aggregation": "COCO collection recomputed per genotype -> equal dev domain",
                "elapsed_seconds": time.perf_counter() - started}
    finally:
        model.train(training)
        restore_global_rng(state, device)


def lr_for_update(update):
    if update < 1:
        raise ValueError("Optimizer update is one-based")
    return .0005 + (.005 - .0005) * min(update - 1, 499) / 499


def train_config(args, rows):
    sources = {str(p.resolve()): {"sha256": file_sha256(p), "source": p.read_text(encoding="utf-8")}
               for p in (Path(__file__), ROOT / "src/mva_model.py", ROOT / "src/mva_pilot.py")}
    return {"version": VERSION, "run": str(jp_path(args.out)), "seed": args.seed,
            "device": args.device, "precision": "FP32", "steps_requested": args.steps,
            "physical_batch": args.physical_batch, "accumulation": args.accumulation,
            "allow_shared_device": bool(getattr(args, "allow_shared_device", False)),
            "effective_batch": 4, "views_per_image": 1, "predeclared_pilot_note": NOTE,
            "formal_36_epoch_run": False, "schedule": {"type": "linear_warmup_then_constant",
                "warmup_updates": 500, "first_lr": .0005, "base_lr": .005,
                "budget_updates": args.steps, "formal_epoch_decay": False},
            "optimizer": {"type": "SGD", "lr": .005, "momentum": .9, "weight_decay": 1e-4},
            "geometry": {"long_edges": list(SCALES), "bilinear_antialias": True,
                "rounding": "nearest integer; actual x/y ratios", "pad_right_bottom_stride": 32,
                "training_masks": "nearest", "restore": "soft bilinear original HW THEN >=.5",
                "implicit_second_resize": False},
            "augmentation": {"horizontal": .5, "vertical": .5, "rot90": [0, 1, 2, 3],
                             "color_probability": .8, "brightness_contrast_saturation": .2,
                             "hue": .05, "label_assisted_background": False},
            "detector": {"name": "independent MaskRCNN R50 FPN P2-P6", "trainable_backbone_layers": 3,
                "frozen_BN": True, "anchors": [8, 16, 32, 64, 128], "anchor_ratios": [.5, 1, 2],
                "rpn_pre_post_train_test": 1000, "rpn_nms": .7, "box_nms": .5,
                "max_instances": 100, "AP_score": .001, "count_score": .5, "mask_threshold": .5,
                "mask_loss": "stock BCE including zero-mass targets; native ROIAlign dispatch only"},
            "determinism": {"algorithms": True, "warn_only": True, "bitwise_replay_guaranteed": False,
                "ROIAlign": "native atomic CUDA backward, aligned=False; no compiled Python/Triton fallback",
                "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
                "threads": 4, "interop_threads": 1},
            "evaluation_interval_updates": 500, "selection_long_edge": 640,
            "dev_max_frames_per_plant": args.dev_max_frames_per_plant,
            "selection_diagnostic_subset": bool(args.dev_max_frames_per_plant),
            "selection": "equal domains of genotype macro COCO AP, ties earlier update",
            "inventory": str(Path(args.inventory).resolve()), "inventory_sha256": file_sha256(args.inventory),
            "instances": str(Path(args.instances).resolve()), "instances_sha256": file_sha256(args.instances),
            "readiness": str(Path(args.readiness).resolve()), "readiness_sha256": file_sha256(args.readiness),
            "data_root": str(Path(args.data_root).resolve()) if args.data_root else None,
            "allowed_population": population(rows), "valid_mask_status": "valid",
            "roles": {"train": ["TRAIN"], "selection_and_benchmark": list(DEV_ROLES),
                      "locked_never_opened": ["C_POLICY", "T_ID", "T_GENO"]},
            "weights": str(Path(args.weights).resolve()), "weight_sha256": file_sha256(args.weights),
            "initialization": "public generic torchvision COCO V1; NOT MVA task weights",
            "versions": versions(), "sources": sources}


def checkpoint_state(model, optimizer, sampler, augmentation, device, step, config,
                     best, timing_window, evaluation_complete=True):
    return {"version": VERSION, "optimizer_step": step, "next_optimizer_step": step + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(), "augmentation": augmentation.state_dict(),
            "global_rng": global_rng_state(device), "config": config, "best": best,
            "timing_window": timing_window, "evaluation_complete": evaluation_complete,
            "optimizer_boundary": True, "pending_gradients": False}


def save_state(path, state):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def validate_resume(checkpoint, config, recorded):
    if jsonable(checkpoint["config"]) != recorded or jsonable(config) != recorded:
        raise ValueError("Exact resume refused: run/config/source/manifest/weights/versions changed")
    step = checkpoint["optimizer_step"]
    if (checkpoint.get("version") != VERSION or checkpoint.get("next_optimizer_step") != step + 1
            or checkpoint.get("optimizer_boundary") is not True or checkpoint.get("pending_gradients") is not False
            or not 0 <= step <= config["steps_requested"]):
        raise ValueError("Not an exact JP optimizer-boundary state")


def restore_state(checkpoint, model, optimizer, sampler, augmentation, device):
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    sampler.load_state_dict(checkpoint["sampler"])
    augmentation.load_state_dict(checkpoint["augmentation"])
    restore_global_rng(checkpoint["global_rng"], device)


def persist(out, model, optimizer, sampler, augmentation, device, step, config, best, window, complete):
    path = out / f"checkpoint_{step:06d}.pt"
    embedded_best = dict(best)
    if best.get("optimizer_step") == step:
        embedded_best.pop("checkpoint_sha256", None)  # avoid a recursive self-hash
    save_state(path, checkpoint_state(model, optimizer, sampler, augmentation, device, step,
                                     config, embedded_best, window, complete))
    checkpoint_hash = file_sha256(path)
    write_json(out / "latest.json", {"checkpoint": str(path), "optimizer_step": step,
        "evaluation_complete": complete, "config_sha256": digest_json(config),
        "checkpoint_sha256": checkpoint_hash})
    if best.get("optimizer_step") == step and complete:
        best["checkpoint_sha256"] = checkpoint_hash
    write_json(out / "best.json", best)
    return path


def evaluate_boundary(out, model, optimizer, sampler, augmentation, device, rows, data,
                      step, config, best, window):
    persist(out, model, optimizer, sampler, augmentation, device, step, config, best, window, False)
    # A failed/interrupted dev pass is repeated from this saved optimizer boundary.
    log = out / f"quality_{step:06d}_S640.jsonl"
    if log.exists():
        archive = log.with_name(log.stem + f".interrupted_{time.time_ns()}" + log.suffix)
        if archive.exists():
            raise FileExistsError("Interrupted-evaluation archive already exists")
        os.replace(log, archive)  # preserve the incomplete evaluation for traceability
    report = quality(model, rows, data, (640,), device, config["dev_max_frames_per_plant"], out, step)
    write_json(out / f"dev_{step:06d}.json", report)
    score = report["scales"]["640"]["equal_domain_macro"]["AP"]
    if best["score"] is None or score > best["score"]:
        best = {"score": score, "optimizer_step": step, "checkpoint": str(out / f"checkpoint_{step:06d}.pt"),
                "diagnostic_subset": report["diagnostic_subset"], "selection_scale": 640,
                "selection_roles": list(DEV_ROLES), "config_sha256": digest_json(config)}
    persist(out, model, optimizer, sampler, augmentation, device, step, config, best, window, True)
    return best


def train(args):
    out, device = jp_path(args.out), torch.device(args.device)
    if args.steps < 1 or args.physical_batch * args.accumulation != 4 or args.accumulation < 1:
        raise ValueError("Positive steps, effective batch exactly 4 required")
    if args.dev_max_frames_per_plant < 0:
        raise ValueError("Invalid dev subset")
    rows, data, _ = load_inputs(args.inventory, args.instances, args.readiness, args.data_root)
    if file_sha256(args.weights) != COCO_V1_SHA256:
        raise ValueError("Generic COCO SHA mismatch; no task-trained initialization")
    config = train_config(args, rows)
    checkpoint = None
    if args.resume:
        resume = Path(args.resume).resolve()
        if not resume.is_relative_to(out) or resume.suffix != ".pt":
            raise ValueError("Resume must be the original JP run checkpoint")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        validate_resume(checkpoint, config, json.loads((out / "run_config.json").read_text(encoding="utf-8")))
    elif out.exists() and any(out.iterdir()):
        raise ValueError("New training requires an empty JP out directory")
    # The follow-up queue verifies completed MVA runs before this process starts.
    # WDDM desktop contexts are recorded, not mistaken for a permission to stop
    # other applications or for proof of exclusive GPU access.
    if device.type == "cuda":
        original_status = ROOT / "mva/pilot_queue_v1.status.json"
        if original_status.is_file():
            previous = json.loads(original_status.read_text(encoding="utf-8"))
            if previous.get("status") != "completed":
                raise RuntimeError("Original MVA queue not completed; no parallel project GPU training")
        gpu = nvidia_snapshot()
        gpu_start_policy(gpu, bool(getattr(args, "allow_shared_device", False)))
    seed_backend(args.seed, device)
    model = build_jp_detector(args.weights, args.seed).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=1e-4)
    sampler, augmentation = BalancedSampler(rows, args.seed), Augmentation(args.seed)
    step, best, window = 0, {"score": None, "optimizer_step": None, "checkpoint": None}, []
    if checkpoint is not None:
        restore_state(checkpoint, model, optimizer, sampler, augmentation, device)
        step, best, window = checkpoint["optimizer_step"], checkpoint["best"], checkpoint["timing_window"]
    with run_lock(out):
        write_json(out / "run_config.json", config)
        write_json(out / f"machine_start_{step:06d}.json", machine_snapshot(device))
        status(out, "initializing", step, actual_population=population(rows))
        if checkpoint is None:
            persist(out, model, optimizer, sampler, augmentation, device, 0, config, best, window, True)
        else:
            for name in ("train.jsonl", "timing.jsonl"):
                trim_jsonl_for_resume(out / name, step)
            persist(out, model, optimizer, sampler, augmentation, device, step, config, best, window,
                    checkpoint["evaluation_complete"])
            if not checkpoint["evaluation_complete"]:
                best = evaluate_boundary(out, model, optimizer, sampler, augmentation, device,
                                         rows, data, step, config, best, window)
        while step < args.steps:
            update = step + 1
            status(out, "training", step, next_optimizer_step=update)
            lr = lr_for_update(update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            start = time.perf_counter()
            raw, stream = [], []
            for micro in range(args.accumulation):
                images, targets = [], []
                for _ in range(args.physical_batch):
                    row = sampler.draw()
                    image, target = data.load(row)
                    image, target, record = augmentation.apply(image, target)
                    stream.append({"sample_ordinal": sampler.draws, "microbatch": micro,
                        "image_id": row["image_id"], "plant": row["biological_plant_id"],
                        "genotype": row["genotype"], **record})
                    images.append(image.to(device))
                    targets.append({k: v.to(device) for k, v in target.items()})
                with no_second_resize(model):
                    losses = model(images, targets)
                total = sum(losses.values()) / args.accumulation
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite JP loss; optimizer state not advanced")
                total.backward()
                raw.append({k: float(v.detach()) for k, v in losses.items()})
                del images, targets, losses, total
            # No clipping or AMP: the declared SGD baseline is unchanged.
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradients; optimizer state not advanced")
            optimizer.step()
            step = update
            sync(device)
            elapsed = time.perf_counter() - start
            window.append(elapsed)
            append_jsonl(out / "train.jsonl", {"optimizer_step": step, "lr": lr,
                "raw_microbatch_losses": raw, "sample_stream": stream, "sample_draws_total": sampler.draws,
                "update_seconds_decode_augment_forward_backward_optimizer_sync": elapsed,
                "memory": memory_snapshot(device)})
            if step % 100 == 0 or step == args.steps:
                append_jsonl(out / "timing.jsonl", {"optimizer_step": step, "window_updates": len(window),
                    "raw_update_seconds": window, "mean_update_seconds": float(np.mean(window)),
                    "p95_update_seconds": float(np.percentile(window, 95)),
                    "timing_excludes_dev_checkpoint_and_log_io": True})
                window = []
                print(json.dumps({"step": step, "lr": lr, "mean_loss": float(np.mean([sum(v.values()) for v in raw]))}), flush=True)
                if step % 500 or step == 0:
                    persist(out, model, optimizer, sampler, augmentation, device, step, config, best, window, True)
            if step % 500 == 0 or step == args.steps:
                best = evaluate_boundary(out, model, optimizer, sampler, augmentation, device,
                                         rows, data, step, config, best, window)
        write_json(out / "machine_end.json", machine_snapshot(device))
        status(out, "complete", step, best=best, diagnostic_subset=bool(args.dev_max_frames_per_plant))


def benchmark_checkpoint(run):
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    best = json.loads((run / "best.json").read_text(encoding="utf-8"))
    latest = json.loads((run / "latest.json").read_text(encoding="utf-8"))
    if config.get("version") != VERSION or not best.get("checkpoint") or best.get("score") is None:
        raise ValueError("No developed JP best checkpoint; benchmark cannot select a new one")
    path = Path(best["checkpoint"]).resolve()
    if not path.is_relative_to(run) or path.suffix != ".pt":
        raise ValueError("Best pointer escapes its JP run")
    if best.get("checkpoint_sha256") != file_sha256(path):
        raise ValueError("JP best checkpoint integrity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume(state, config, config)
    selected = state["best"]
    if (not state["evaluation_complete"] or state["optimizer_step"] != best["optimizer_step"]
            or selected.get("optimizer_step") != state["optimizer_step"]
            or selected.get("score") != best["score"] or best.get("selection_scale") != 640
            or best.get("config_sha256") != digest_json(config)):
        raise ValueError("Best checkpoint was not fully developed/selected by the frozen dev rule")
    # Source/versions/manifests are checked; never silently benchmark changed code.
    for source, entry in config["sources"].items():
        if file_sha256(source) != entry["sha256"]:
            raise ValueError("JP implementation/dependency source changed since training")
    if versions() != config["versions"]:
        raise ValueError("Backend versions differ from JP training")
    return config, best, latest, state, path


def warmup(model, blobs, rows, scale, device, requested):
    times, stable = [], False
    # Smoke can lower warmup; it is then explicitly non-protocol diagnostic.
    minimum = max(requested, 60) if requested >= 50 else requested
    maximum = 200 if requested >= 50 else requested
    for i in range(maximum):
        row = rows[i % len(rows)]
        _, record = service(model, lambda r=row: blobs[r["image_id"]], scale, device,
                            (row["height"], row["width"]))
        times.append(record["service_ms"])
        if len(times) >= minimum:
            if requested < 50:
                break
            medians = [np.median(times[j:j + 20]) for j in range(len(times) - 60, len(times), 20)]
            stable = (max(medians) - min(medians)) / max(np.mean(medians), 1e-12) <= .05
            if stable:
                break
    return {"requested": requested, "actual": len(times), "raw_ms": times,
            "stable_recent_three_groups20": stable, "jitter": not stable,
            "diagnostic_warmup": requested < 50, "maximum": maximum}


def child_manifest(out):
    manifest = json.loads((out / "benchmark_config.json").read_text(encoding="utf-8"))
    config = manifest["training_config"]
    for source, entry in config["sources"].items():
        if file_sha256(source) != entry["sha256"]:
            raise ValueError("Child implementation/source differs from frozen benchmark")
    for key in ("inventory", "instances", "readiness"):
        if file_sha256(config[key]) != config[key + "_sha256"]:
            raise ValueError("Child manifest differs from frozen benchmark")
    if versions() != manifest["versions"]:
        raise ValueError("Child backend differs from frozen benchmark")
    if file_sha256(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen benchmark checkpoint changed")
    return manifest


def quality_child(out):
    manifest = child_manifest(out)
    config = manifest["training_config"]
    device = torch.device(manifest["device"])
    seed_backend(config["seed"], device)
    rows, data, _ = load_inputs(config["inventory"], config["instances"], config["readiness"], config["data_root"])
    state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    model = build_jp_detector(seed=config["seed"]).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    step = state["optimizer_step"]
    del state
    write_json(out / "quality_machine_start.json", machine_snapshot(device))
    for scale in SCALES:
        log = out / f"quality_{step:06d}_S{scale}.jsonl"
        if log.exists():
            log.unlink()
    report = quality(model, rows, data, SCALES, device,
                     manifest["quality_max_frames_per_plant"], out, step)
    write_json(out / "quality.json", report)
    write_json(out / "quality_machine_end.json", machine_snapshot(device))


def timing_session(out, session):
    manifest = child_manifest(out)
    config = manifest["training_config"]
    device = torch.device(manifest["device"])
    seed_backend(config["seed"], device)
    rows, _, _ = load_inputs(config["inventory"], config["instances"], config["readiness"], config["data_root"])
    selected = select_dev(rows, manifest["timing_max_frames_per_plant"])
    if digest_json([r["image_id"] for r in selected]) != manifest["timing_stream_sha256"]:
        raise ValueError("Timing stream changed")
    path = Path(manifest["checkpoint"])
    if file_sha256(path) != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen benchmark checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_jp_detector(seed=config["seed"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model_residency = {"parameter_and_buffer_bytes": sum(t.numel() * t.element_size()
                       for t in (*model.parameters(), *model.buffers())),
                       "process_after_model_load_before_input_preload": memory_snapshot(device)}
    # Compressed inputs are pre-read, shared across scales, without annotations.
    blobs = {r["image_id"]: input_bytes(r) for r in selected}
    log = out / f"timing_session_{session}.jsonl"
    if log.exists() or (out / f"session_{session}.json").exists():
        raise ValueError("Timing session already has output; use a new benchmark out for retries")
    report = {"session": session, "pid": os.getpid(), "fresh_process": True,
              "model_residency": model_residency, "preloaded_compressed_bytes": sum(map(len, blobs.values())),
              "start": machine_snapshot(device), "scale_order": list(SCALES[session - 1:] + SCALES[:session - 1]),
              "scales": {}, "benchmark_config_sha256": digest_json(manifest)}
    for scale in report["scale_order"]:
        status(out, "timing", session=session, scale=scale)
        before = machine_snapshot(device)
        warmed = warmup(model, blobs, selected, scale, device, manifest["warmup"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        resident = memory_snapshot(device)
        start = time.perf_counter()
        for index, row in enumerate(selected):
            _, record = service(model, lambda r=row: blobs[r["image_id"]], scale, device,
                                (row["height"], row["width"]))
            append_jsonl(log, {"session": session, "scale": scale, "ordinal": index,
                "image_id": row["image_id"], "plant": row["biological_plant_id"],
                "genotype": row["genotype"], "split_role": row["split_role"],
                **record, "memory": memory_snapshot(device)})
        loop_seconds = time.perf_counter() - start
        after = machine_snapshot(device)
        contention = any(s["nvidia"] and s["nvidia"]["contention"] for s in (before, after))
        unknown = any(s["nvidia"] and not s["nvidia"]["compute_query_ok"] for s in (before, after))
        report["scales"][str(scale)] = {"warmup": warmed, "before": before, "after": after,
            "contention": contention, "contention_assessment_unknown": unknown,
            "exclusive_platform_guaranteed": False, "model_resident_memory": resident,
            "measurement_loop_seconds_including_log_io": loop_seconds}
    report["end"] = machine_snapshot(device)
    report["completed"] = True
    write_json(out / f"session_{session}.json", report)


def summarize_times(records):
    values = np.asarray([r["service_ms"] for r in records], dtype=np.float64)
    if not len(values):
        raise ValueError("No timed requests")
    per_image = defaultdict(list)
    for r in records:
        per_image[(r["split_role"], r["genotype"], r["plant"], r["image_id"])].append(r["service_ms"])
    plants = defaultdict(list)
    for (role, genotype, plant, _), times in per_image.items():
        plants[(role, genotype, plant)].append(float(np.mean(times)))
    genotypes = defaultdict(list)
    for (role, genotype, _), times in plants.items():
        genotypes[(role, genotype)].append(float(np.mean(times)))
    roles = {role: float(np.mean([np.mean(v) for (r, _), v in genotypes.items() if r == role])) for role in DEV_ROLES}
    fields = ("decode_ms", "resize_transfer_ms", "network_sync_ms", "restore_scores_masks_rle_serialize_sync_ms")
    memory = {key: max((r["memory"][key] for r in records if r["memory"][key] is not None), default=None)
              for key in ("rss_bytes", "peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")}
    role_raw = {}
    for role in DEV_ROLES:
        times = np.asarray([r["service_ms"] for r in records if r["split_role"] == role])
        role_raw[role] = {"mean_ms": float(times.mean()), "p95_ms": float(np.percentile(times, 95)),
                          "throughput_per_second": float(1000 * len(times) / times.sum())}
    return {"requests_technical_repeats": len(values), "mean_ms": float(values.mean()),
            "role_raw_request_distribution": role_raw,
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)), "saturated_throughput_per_second": float(1000 * len(values) / values.sum()),
            "throughput_denominator": "sum of full synchronized per-request wall times, excludes logging and warmup",
            "stage_mean_ms": {key: float(np.mean([r[key] for r in records])) for key in fields},
            "role_genotype_plant_equal_mean_ms": roles, "equal_domain_mean_ms": float(np.mean(list(roles.values()))),
            "per_genotype_plant_mean_ms": {f"{r}/{g}": float(np.mean(v)) for (r, g), v in genotypes.items()},
            "raw_p95_scope": "measured dev timing subset; includes three technical sessions, not extra plants",
            "memory": memory}


def scale_comparison(quality_report, timing):
    rows = []
    ref = quality_report["scales"]["640"]
    for scale in SCALES:
        quality_scale = quality_report["scales"][str(scale)]
        comparisons = {}
        for role in (*DEV_ROLES, "equal_domain_macro"):
            q = quality_scale["equal_domain_macro"] if role == "equal_domain_macro" else quality_scale["role_macro"][role]
            h = ref["equal_domain_macro"] if role == "equal_domain_macro" else ref["role_macro"][role]
            da, dn = h["AP"] - q["AP"], q["count_MAE"] - h["count_MAE"]
            comparisons[role] = {"AP_drop_pp": da, "count_MAE_increase": dn,
                                 "point_tolerances_met": da <= 1.0 and dn <= .2}
        equal = quality_scale["equal_domain_macro"]
        row = {"scale": scale, "reference": scale == 640, "AP": equal["AP"], "count_MAE": equal["count_MAE"],
               "diagnostic_subset": quality_report["diagnostic_subset"],
               "relative_to_S640": comparisons, **timing[str(scale)]}
        for role in DEV_ROLES:
            row[role + "_AP"] = quality_scale["role_macro"][role]["AP"]
            row[role + "_count_MAE"] = quality_scale["role_macro"][role]["count_MAE"]
        rows.append(row)
    return rows


def finalize_benchmark(out):
    sessions = []
    for session in (1, 2, 3):
        path = out / f"session_{session}.json"
        if not path.is_file():
            status(out, "awaiting_three_sessions", sessions_completed=[s["session"] for s in sessions])
            return
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not obj.get("completed"):
            raise ValueError("Incomplete session; keep failure logs, rerun paired group in a new out")
        sessions.append(obj)
    manifest = json.loads((out / "benchmark_config.json").read_text(encoding="utf-8"))
    if any(s["benchmark_config_sha256"] != digest_json(manifest) for s in sessions):
        raise ValueError("Mixed benchmark session configurations")
    quality_report = json.loads((out / "quality.json").read_text(encoding="utf-8"))
    all_records = []
    for session in (1, 2, 3):
        all_records.extend(json.loads(line) for line in (out / f"timing_session_{session}.jsonl").read_text(encoding="utf-8").splitlines())
    timing = {}
    for scale in SCALES:
        records = [r for r in all_records if r["scale"] == scale]
        expected = manifest["timing_image_ids"]
        for session in (1, 2, 3):
            got = [r["image_id"] for r in records if r["session"] == session]
            if got != expected:
                raise ValueError("Unpaired/incomplete timing stream; do not drop slow requests")
        timing[str(scale)] = summarize_times(records)
        timing[str(scale)]["per_session"] = {str(s): summarize_times([r for r in records if r["session"] == s]) for s in (1, 2, 3)}
        timing[str(scale)]["contention"] = any(s["scales"][str(scale)]["contention"] for s in sessions)
        timing[str(scale)]["jitter"] = any(s["scales"][str(scale)]["warmup"]["jitter"] for s in sessions)
    table = scale_comparison(quality_report, timing)
    report = {"predeclared_pilot_note": NOTE, "formal_36_epoch_run": False, "rows": table,
              "quality": quality_report, "benchmark_config": manifest,
              "checkpoint_reselected": False, "tolerances": {"AP_drop_pp": 1.0, "count_MAE_increase": .2},
              "exclusive_platform_guaranteed": False, "sessions": sessions,
              "diagnostic_subset": quality_report["diagnostic_subset"],
              "diagnostic_timing": manifest["warmup"] < 50 or manifest["timing_max_frames_per_plant"] != 20,
              "confirmatory_quality_claim": False}
    write_json(out / "scales.json", report)
    import csv
    fields = ("scale", "reference", "AP", "count_MAE", "V_SOURCE_AP", "V_SHIFT_AP",
              "V_SOURCE_count_MAE", "V_SHIFT_count_MAE", "mean_ms", "p95_ms", "saturated_throughput_per_second",
              "equal_domain_mean_ms", "AP_drop_pp", "count_MAE_increase", "point_tolerances_met",
              "decode_ms", "resize_transfer_ms", "network_sync_ms", "restore_scores_masks_rle_serialize_sync_ms",
              "rss_bytes", "peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
              "diagnostic_subset", "diagnostic_timing", "contention", "jitter")
    with (out / "scales.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in table:
            flat = {**row, **row["relative_to_S640"]["equal_domain_macro"],
                    **row["stage_mean_ms"], **row["memory"], "diagnostic_timing": report["diagnostic_timing"]}
            writer.writerow({key: flat[key] for key in fields})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    times = [r["equal_domain_mean_ms"] for r in table]
    for index, (field, title) in enumerate((("AP", "Development AP (%)"), ("count_MAE", "Development count MAE"))):
        axes[index].plot(times, [r[field] for r in table], "o-")
        for r in table:
            axes[index].annotate(f"S{r['scale']}", (r["equal_domain_mean_ms"], r[field]))
        axes[index].set(xlabel="Equal-domain plant/genotype mean service ms", ylabel=title)
    bottoms = np.zeros(3)
    for field in table[0]["stage_mean_ms"]:
        heights = [r["stage_mean_ms"][field] for r in table]
        axes[2].bar([str(s) for s in SCALES], heights, bottom=bottoms, label=field.replace("_ms", ""))
        bottoms += heights
    axes[2].set(xlabel="Long edge", ylabel="Raw request mean stage ms")
    axes[2].legend(fontsize=6)
    fig.suptitle("P1 PILOT / development only" + (" / diagnostic" if report["diagnostic_subset"] or report["diagnostic_timing"] else ""))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"scales.{suffix}", dpi=160)
    plt.close(fig)
    status(out, "complete", checkpoint=manifest["checkpoint"], checkpoint_reselected=False,
           diagnostic_subset=report["diagnostic_subset"], diagnostic_timing=report["diagnostic_timing"])


def benchmark(args):
    if not args.run:
        raise ValueError("Benchmark requires --run")
    out, run = jp_path(args.out), jp_path(args.run)
    if out == run or out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError("Benchmark output must be separate from training run")
    if args._timing_session:
        timing_session(out, args._timing_session)
        return
    if args._quality_pass:
        quality_child(out)
        return
    if (args.quality_max_frames_per_plant < 0 or not 1 <= args.timing_max_frames_per_plant <= 20
            or not 0 <= args.warmup <= 200):
        raise ValueError("Invalid quality subset, timing max (1..20), or warmup (0..200)")
    config, best, latest, state, path = benchmark_checkpoint(run)
    rows, data, _ = load_inputs(config["inventory"], config["instances"], config["readiness"], config["data_root"])
    for key in ("inventory", "instances", "readiness"):
        if file_sha256(config[key]) != config[key + "_sha256"]:
            raise ValueError(f"Frozen {key} changed")
    # Explicit CLI paths cannot silently substitute a different population.
    for key in ("inventory", "instances", "readiness"):
        if Path(getattr(args, key)).resolve() != Path(config[key]).resolve():
            raise ValueError(f"Benchmark --{key} must match training config")
    timed = select_dev(rows, args.timing_max_frames_per_plant)
    manifest = {"version": VERSION, "run": str(run), "out": str(out), "device": args.device,
                "training_config": config, "checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
                "selected_best": best, "latest_pointer_at_start": latest, "checkpoint_reselected": False,
                "quality_max_frames_per_plant": args.quality_max_frames_per_plant,
                "timing_max_frames_per_plant": args.timing_max_frames_per_plant, "warmup": args.warmup,
                "timing_image_ids": [r["image_id"] for r in timed],
                "timing_stream_sha256": digest_json([r["image_id"] for r in timed]),
                "timing_population": population(timed), "scales": list(SCALES), "precision": "FP32",
                "batch": 1, "threads": 4, "interop_threads": 1, "seed": config["seed"],
                "sessions": 3, "fresh_process_per_session": True, "scale_order_rotates": True,
                "session_spacing": "fresh sequential startups; --session 1/2/3 allows separate launch periods",
                "scope": "only development domains; fixed same-weight simple scale baseline",
                "tolerances_predeclared": {"AP_drop_pp": 1.0, "count_MAE_increase": .2},
                "timing": "pre-read compressed bytes; includes retrieval/decode/resize/model/softmask restore/scores/RLE/memory JSON serialization/sync",
                "versions": versions(), "exclusive_platform_guaranteed": False}
    with run_lock(out):
        existing = out / "benchmark_config.json"
        if existing.is_file():
            if json.loads(existing.read_text(encoding="utf-8")) != jsonable(manifest):
                raise ValueError("Benchmark configuration/checkpoint changed; use a new out")
        elif any(p.name != ".jp_lock" for p in out.iterdir()):
            raise ValueError("New benchmark requires empty out")
        else:
            write_json(existing, manifest)
        del state
        if not (out / "quality.json").is_file():
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--action", "benchmark",
                       "--run", str(run), "--out", str(out), "--_quality-pass"]
            # Parent never initializes CUDA: no lingering parent GPU context
            # or artificial contention in the later fresh timing processes.
            subprocess.run(command, check=True)
        sessions = [args.session] if args.session else [1, 2, 3]
        for session in sessions:
            if (out / f"session_{session}.json").exists():
                continue
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--action", "benchmark",
                       "--run", str(run), "--out", str(out), "--_timing-session", str(session)]
            # One child at a time; parent model is released. No parallel GPU jobs.
            subprocess.run(command, check=True)
        finalize_benchmark(out)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", required=True, choices=["train", "benchmark"])
    p.add_argument("--out", required=True, help="JP subdirectory, relative to experiment root or absolute")
    p.add_argument("--run", help="JP training run for benchmark")
    p.add_argument("--steps", type=int, default=1000, help="optimizer updates; immutable exact-resume budget")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--physical-batch", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--allow-shared-device", action="store_true", help="record ordinary desktop GPU contexts instead of requiring an empty WDDM process list; not an exclusive timing claim")
    p.add_argument("--resume", help="trusted .pt checkpoint in original --out; all config/source hashes must match")
    p.add_argument("--weights", default=DEFAULT_COCO)
    p.add_argument("--inventory", default=str(ROOT / "shared/pilot_inventory.csv"))
    p.add_argument("--instances", default=str(ROOT / "shared/pilot_instances.jsonl"))
    p.add_argument("--readiness", default=str(ROOT / "shared/pilot_readiness.json"))
    p.add_argument("--data-root")
    p.add_argument("--dev-max-frames-per-plant", type=int, default=0, help="train checkpoint development: 0=all; positive=diagnostic")
    p.add_argument("--quality-max-frames-per-plant", type=int, default=0, help="benchmark quality: 0=all; positive=diagnostic")
    p.add_argument("--timing-max-frames-per-plant", type=int, default=20, help="chronological equidistant dev frames, max20")
    p.add_argument("--warmup", type=int, default=50, help="per-scale per-session; <50 is timing diagnostic only")
    p.add_argument("--session", type=int, choices=[1, 2, 3], help="optional separate startup period; omit to run three fresh sequential children")
    p.add_argument("--_timing-session", type=int, choices=[1, 2, 3], dest="_timing_session", help=argparse.SUPPRESS)
    p.add_argument("--_quality-pass", action="store_true", dest="_quality_pass", help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    out = jp_path(args.out)
    try:
        if args.action == "train":
            train(args)
        else:
            benchmark(args)
    except Exception as exc:
        if out.is_dir():
            status(out, "failed", error=str(exc), traceback=traceback.format_exc(),
                   note="Retain logs; never delete slow requests or stop another process")
        raise


if __name__ == "__main__":
    main()
