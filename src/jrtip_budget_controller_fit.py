"""Fit the per-frame output-budget controller and price it at equal mask work (P1b).

What this decides
-----------------
Plan v2 section 9 moves the stop-loss judgment for H1 to P1b, on the development domain only, with zero
backbone training: one cap-100 pricing pass plus one feature pass.  The question is narrow and
predeclared - does *any* of the controller's 21 threshold candidates, at its own measured mask work
(+-5% window), lose less quality than the strongest label-free arm in the same window (section 5(a)
pass-lines: work 16.3-17.0 -> 0.3845 pp, 19.4-20.0 -> 0.2956 pp, 21.3 -> 0.2818 pp)?  If none does, the
formal matrix is cancelled and the work is downgraded to the mechanism decomposition plus the
label-free configuration result.  This script produces the numbers that judgment consumes; it does not
make the judgment, and the equal-work verdict is left to ``src/jrtip_budget_equal_work.py``, which
requires its nearest-window and interpolated readings to agree before calling anything a gain.

Everything here is the predeclared structure.

* Action set (section 4.1, frozen): ``K = {100, 50, 30, 20, 15}``.  ``k = 100`` is simultaneously the
  reference arm, the tau = -inf behaviour and the mandated fallback ("fallback to k=100"), and its delta
  heads are identically zero, so heads are fitted only for the other four arms.
* Folds (section 4.3): three whole-plant folds inside TRAIN, stratified by genotype, from an independent
  ``PCG64(43)`` stream.  Fold models see only the other two folds; their held-out predictions form the
  OOF table that the calibration and structure diagnostics read.  Development frames are scored by the
  full-TRAIN fit, which never saw a development frame.
* Regressor (plan v1 section 4.3, inherited): shallow gradient-boosted trees, 150 estimators, depth 2,
  learning rate 0.03, min_samples_leaf 20, subsample 1.0; seeds 43/1043/2043; per-plant equal total frame
  weight so long time series cannot dominate.  v1 nominates "development service time under the
  predeclared tolerances" as the structure criterion, which is undefined for a regressor, so
  ``--structure-probe`` compares depth 2 with depth 3 on OOF calibration, reports both, and lets the
  predeclared tie-break (depth 2) bind; the choice is never made on development labels.
* Heads (section 4.3): for every fitted arm, the 0.9 quantile and the mean of both section 4.1 marginal
  losses, plus a mean cost head (masked instances used relative to cap-100).  The cost head is fitted and
  reported but does not enter the decision, because the predeclared decision rule carries no cost term -
  which is also why section 7.2's A2 row ("drop cost normalisation") has no referent at P1b; that is
  surfaced at the gate review instead of being repaired by invention.
* Decision (section 4.3): ``r(frame, k) = max(Delta_M_hat(k)/0.02, Delta_N_hat(k)/0.2)`` from the
  quantile heads, and ``k*(tau)`` is the smallest arm with ``r <= tau``, else 100.  The 21 candidates are
  ``-inf``, the 5..95 (step 5) percentiles of that seed's own pooled development risk scores, and
  ``+inf`` (v1 section 5.2).  0.02 and 0.2 are surrogate-loss scales, not AP conversions.
* Protection (v1 section 4.4, inherited): a frame whose continuous features fall outside the TRAIN
  0.5%-99.5% band on more than 25% of the valid features goes to ``k = 100``, as does any frame with an
  empty output.  A3 removes exactly that fallback; A1 removes the score-sequence feature group.
* P1 (section 7.2, "random assignment from the same budget distribution, ignoring scores"): every priced
  learned policy is mirrored by a seeded permutation of its own budget multiset, so the controller is
  compared against a matched-work randomizer rather than only against static and rule arms.

Cost control that does not touch the protocol
---------------------------------------------
One assignment costs roughly 100 CPU seconds to price through the archived arm machinery, so 3 seeds x 21
thresholds would be 63 arms.  ``k`` is discrete, so neighbouring thresholds frequently yield the
*identical* per-frame assignment, and identical assignments are identical policies: each distinct budget
vector is priced once (``vector_digest``, the same dedup the rule-arm pricing uses) and every
(seed, tau) candidate is reported against its own digest.  Nothing is dropped or sampled.

Locked roles stay closed: TRAIN (fit) and V_SOURCE/V_SHIFT (evaluation) only; C_POLICY, T_ID, T_GENO and
T_EXT are never loaded.  No GPU, no model, no inference - the inputs are the two cap-100 feature archives
written by ``src/jrtip_budget_features.py``, and the development archive should have passed
``src/jrtip_budget_features_check.py`` first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

from src.jrtip_budget_bucket_pricing import (  # noqa: E402
    budget_histogram, cost_projection, evaluate_control, vector_digest,
)
from src.jrtip_budget_features import (  # noqa: E402
    CONTROL, FEATURE_NAMES, SCORE_SEQUENCE_GROUP,
)
from src.jrtip_budget_policy import (  # noqa: E402
    AP_DROP_LIMIT_PP, COUNT_INCREASE_LIMIT_LEAF, Ledger, evaluate_arm, per_frame_consistency,
    read_stream, regroup, score_order_ok,
)
from src.jrtip_budget_rule_timing import QUALITY_ROOT, SCALE  # noqa: E402
from src.jrtip_pilot import load_inputs, select_dev  # noqa: E402
from src.verify_jrtip_budget_quality import METRIC_KEYS  # noqa: E402

VERSION = "jrtip-budget-controller-fit-v1"
ARMS_ALL = (100, 50, 30, 20, 15)        # section 4.1 action set K
ARMS_FITTED = tuple(arm for arm in ARMS_ALL if arm != CONTROL)
LOSS_SCALE_M, LOSS_SCALE_N = 0.02, 0.2  # section 4.3 normalisers
SEEDS = (43, 1043, 2043)
STATISTICS = ("quantile_M", "quantile_N", "mean_M", "mean_N", "cost")
GB_PARAMS = {"n_estimators": 150, "learning_rate": 0.03, "min_samples_leaf": 20, "subsample": 1.0}
QUANTILE = 0.9
CONTINUOUS = tuple(name for name in FEATURE_NAMES if not name.startswith("flag_"))
BAND_PROB = (0.005, 0.995)
OUT_OF_BAND_LIMIT = 0.25
TAU_PERCENTILES = tuple(range(5, 100, 5))
RANDOM_WORK_BAND = (10.0, 30.0)         # the gate band the prewritten pass-lines sit inside


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_archive(directory: Path, preferred: str):
    """One feature/label archive: (records by image id, records in file order, report, suffix used).

    ``jrtip_budget_features.py`` names its files by scope, so the production call wants
    ``features_train.json`` / ``features_dev.json``.  A directory holding exactly one of the two is
    accepted under that suffix and reports it, which is what lets the wiring smoke point the TRAIN role
    at the development archive without pretending the file is named differently than it is.
    """
    suffix = preferred
    if not (directory / f"features_{suffix}.json").exists():
        options = [value for value in ("train", "dev") if (directory / f"features_{value}.json").exists()]
        if len(options) != 1:
            raise FileNotFoundError(f"{directory} has no features_{preferred}.json and is not a single "
                                    f"unambiguous archive (candidates found: {options or 'none'})")
        suffix = options[0]
    report = json.loads((directory / f"features_{suffix}.json").read_text(encoding="utf-8"))
    frames = [json.loads(line) for line in (directory / f"frames_{suffix}.jsonl").open(encoding="utf-8")]
    return {str(frame["image_id"]): frame for frame in frames}, frames, report, suffix


def design(frames, names):
    """Feature matrix in the frozen ``names`` order, plus per-plant equal total frame weight."""
    matrix = np.asarray([[frame["features"][name] for name in names] for frame in frames],
                        dtype=np.float64)
    per_plant = {}
    for frame in frames:
        per_plant[frame["plant"]] = per_plant.get(frame["plant"], 0) + 1
    weight = np.asarray([1.0 / per_plant[frame["plant"]] for frame in frames], dtype=np.float64)
    return matrix, weight / weight.mean()


def folds_by_plant(frames, fold_count: int, seed: int = 43) -> dict:
    """Genotype-stratified whole-plant fold table from an independent PCG64 stream (section 4.3).

    Within a genotype the plants are shuffled and dealt round-robin, offset by the genotype's index so no
    fold systematically holds the alphabetically first plants of every genotype.  All frames of a plant
    share a fold, which is the entire point of the exercise.
    """
    plants = {}
    for frame in frames:
        plants.setdefault(frame["plant"], frame["genotype"])
    by_genotype = {}
    for plant, genotype in plants.items():
        by_genotype.setdefault(genotype, []).append(plant)
    rng = np.random.default_rng(np.random.PCG64(seed))
    assignment = {}
    for offset, genotype in enumerate(sorted(by_genotype)):
        group = sorted(by_genotype[genotype])
        rng.shuffle(group)
        for index, plant in enumerate(group):
            assignment[plant] = (index + offset) % fold_count
    return assignment


def head_target(frames, arm: int, statistic: str) -> np.ndarray:
    """Realised label for one (arm, statistic) pair, read straight from the archive.

    ``cost`` is the change in masked instances relative to the same frame at cap 100, so it is at most
    zero: an arm that truncates nothing costs the same as the reference.
    """
    values = []
    for frame in frames:
        labels = frame["labels"]
        if statistic == "cost":
            delivered = int(labels["count_ge_threshold"])
            values.append(float(min(delivered, int(arm)) - delivered))
        elif statistic.endswith("_M"):
            values.append(float(labels["delta_M"][str(arm)]))
        else:
            values.append(float(labels["delta_N"][str(arm)]))
    return np.asarray(values, dtype=np.float64)


def fit_head(x, y, weight, seed: int, depth: int, statistic: str):
    params = dict(GB_PARAMS, random_state=int(seed), max_depth=int(depth))
    if statistic.startswith("quantile"):
        params["loss"] = "quantile"
        params["alpha"] = QUANTILE
    model = GradientBoostingRegressor(**params)
    model.fit(x, y, sample_weight=weight)
    return model


def fit_heads(x_fit, frames_fit, weight, arms, seed: int, depth: int, hold=None):
    """One set of heads (arm x statistic) fitted on ``x_fit`` minus the held-out fold, if any."""
    selected = np.ones(len(frames_fit), dtype=bool) if hold is None else ~hold
    models = {}
    for arm in arms:
        if arm == CONTROL:
            continue
        for statistic in STATISTICS:
            y = head_target(frames_fit, arm, statistic)
            models[f"{statistic}:{arm}"] = fit_head(x_fit[selected], y[selected], weight[selected],
                                                    seed, depth, statistic)
    return models


def predict(models, x, arms, rows: int) -> dict:
    """Score every head against ``x``; the cap-100 arm is its own reference, so its deltas are zero."""
    out = {}
    for key, model in models.items():
        out[key] = model.predict(x)
    for statistic in STATISTICS:
        out[f"{statistic}:{CONTROL}"] = np.zeros(rows)
    return out


def oof_predictions(x, frames, weight, fold_index, arms, seeds, depth):
    """Out-of-fold prediction table: every frame scored by the model that never saw its plant."""
    table = {}
    for fold in sorted(set(fold_index.tolist())):
        hold = fold_index == fold
        models = fit_heads(x, frames, weight, arms, seeds, depth, hold=hold)
        scored = predict(models, x, arms, len(frames))
        for key, values in scored.items():
            table.setdefault(key, np.zeros(len(frames)))[hold] = values[hold]
    return table


def stacked_risk(prediction: dict, arms) -> np.ndarray:
    """Rows follow ``arms`` order, columns follow the frame order: r = max(dM/0.02, dN/0.2)."""
    return np.asarray([np.maximum(prediction[f"quantile_M:{arm}"] / LOSS_SCALE_M,
                                   prediction[f"quantile_N:{arm}"] / LOSS_SCALE_N) for arm in arms])


def choose_budget(risk: np.ndarray, arms, tau: float) -> np.ndarray:
    """Smallest arm whose predicted risk clears tau, else the cap-100 fallback (section 4.3).

    ``arms`` are the fitted arms only; ``k = 100`` is the fallback rather than a candidate, which is why
    its identically-zero heads never enter the comparison or the tau grid.
    """
    choice = np.full(risk.shape[1], CONTROL, dtype=np.int64)
    settled = np.zeros(risk.shape[1], dtype=bool)
    position = {int(arm): index for index, arm in enumerate(arms)}
    for arm in sorted(position):
        take = (~settled) & (risk[position[arm]] <= tau)
        choice = np.where(take, arm, choice)
        settled |= take                 # ascending order: a cheaper acceptable arm always wins
    return choice


def feature_band(x, columns):
    return (np.quantile(x, BAND_PROB[0], axis=0)[columns],
            np.quantile(x, BAND_PROB[1], axis=0)[columns])


def out_of_band_fraction(x, columns, low, high) -> np.ndarray:
    if not columns:
        return np.zeros(x.shape[0])
    outside = np.zeros(x.shape[0], dtype=np.float64)
    for position, column in enumerate(columns):
        outside += (x[:, column] < low[position]) | (x[:, column] > high[position])
    return outside / len(columns)


def calibration_table(prediction, frames, arms) -> dict:
    """Realised exceedance of each 0.9-quantile head against its nominal 10%, plus error and bias."""
    table = {}
    for arm in arms:
        for statistic in ("quantile_M", "quantile_N", "cost"):
            predicted = prediction[f"{statistic}:{arm}"]
            truth = head_target(frames, arm, statistic)
            entry = {"mean_abs_error": round(float(np.mean(np.abs(predicted - truth))), 6),
                     "mean_bias": round(float(np.mean(predicted - truth)), 6)}
            if not statistic.startswith("quantile"):
                entry["note"] = "cost head: fitted and reported, unused by the predeclared decision rule"
            else:
                entry["nominal_exceedance"] = round(1 - QUANTILE, 3)
                entry["realised_exceedance"] = round(float(np.mean(truth > predicted)), 4)
            table[f"{statistic}:{arm}"] = entry
    return table


def drift_report(x_train, x_dev, names, continuous, low, high, flagged) -> dict:
    rows = []
    for position, column in enumerate(continuous):
        train, dev = x_train[:, column], x_dev[:, column]
        spread = float(np.std(train)) or 1.0
        rows.append({"feature": names[column],
                     "standardised_mean_shift": round(float(dev.mean() - train.mean()) / spread, 4),
                     "fraction_outside_train_band": round(float(np.mean((dev < low[position])
                                                                        | (dev > high[position]))), 4)})
    return {"train_frames": int(x_train.shape[0]), "dev_frames": int(x_dev.shape[0]),
            "worst_by_mean_shift": sorted(rows, key=lambda item: -abs(item["standardised_mean_shift"]))[:8],
            "worst_by_band_exit": sorted(rows, key=lambda item: -item["fraction_outside_train_band"])[:8],
            "frames_over_protection_limit": int(np.sum(flagged > OUT_OF_BAND_LIMIT)),
            "protection_limit": OUT_OF_BAND_LIMIT}


def seed_list(raw: str) -> tuple:
    """Parse --seeds into integers.

    argparse's ``type`` is applied to the command-line string, and the loop downstream iterates this
    value, so a plain string would fit one model per character ("43" silently becoming seeds 4 and 3).
    The defect was caught by the wiring smoke: it reported two seeds when one was requested.
    """
    values = [int(item) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--seeds needs at least one integer seed")
    return tuple(values)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", default=str(ROOT / "jrtip/budget_features_train_v1"))
    parser.add_argument("--dev-features", default=str(ROOT / "jrtip/budget_features_dev_v1"))
    parser.add_argument("--out", default=str(ROOT / "jrtip/budget_p1b_v1"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seeds", type=seed_list, default=SEEDS,
                        help="comma separated integer seeds, e.g. 43,1043,2043")
    parser.add_argument("--ablation", default="", choices=("", "A1", "A3"))
    parser.add_argument("--structure-probe", dest="structure_probe", action="store_true",
                        help="also fit depth 3 at the first seed and report both OOF calibrations")
    parser.add_argument("--skip-pricing", dest="skip_pricing", action="store_true",
                        help="fit and diagnose only, no COCO re-scoring (fast code-path check)")
    parser.add_argument("--expected-frames", dest="expected_frames", type=int, default=2947)
    parser.add_argument("--reproduce-tol", dest="reproduce_tol", type=float, default=0.002)
    parser.add_argument("--cost-eval", default=str(ROOT / "jrtip/budget_policy_v1/policy_eval.json"),
                        help="archived arm table, read only for its fitted cost model")
    return parser


def run(args):
    started = time.perf_counter()
    ledger = Ledger()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = args.ablation or "C1"

    train_dir, dev_dir = Path(args.train_features), Path(args.dev_features)
    _, train_frames, train_report, train_suffix = read_archive(train_dir, "train")
    _, dev_frames, dev_report, dev_suffix = read_archive(dev_dir, "dev")
    smoke = str(train_report.get("stream")) == str(dev_report.get("stream"))
    if smoke:
        print("CODE SMOKE ONLY - the TRAIN and DEV archives name the same cap-100 stream, so every "
              "number written by this run is a wiring check, not a result", flush=True)

    names = tuple(name for name in FEATURE_NAMES
                  if not (args.ablation == "A1" and name in SCORE_SEQUENCE_GROUP))
    columns = {name: index for index, name in enumerate(names)}
    continuous = [columns[name] for name in CONTINUOUS if name in columns]
    x_train, w_train = design(train_frames, names)
    x_dev, _ = design(dev_frames, names)

    # ---- development arm table: the archived cap-100 stream and its contracts -----------------
    control_stream = QUALITY_ROOT / f"cap{CONTROL:05d}" / f"quality_{CONTROL:06d}_S{SCALE}.jsonl"
    config = json.loads((QUALITY_ROOT / f"cap{CONTROL:05d}" /
                         "budget_quality_config.json").read_text(encoding="utf-8"))
    benchmark = Path(config["source_benchmark"])
    manifest = json.loads((benchmark / "benchmark_config.json").read_text(encoding="utf-8"))
    training = manifest["training_config"]
    rows, data, _ = load_inputs(training["inventory"], training["instances"], training["readiness"],
                                training["data_root"])
    stream = select_dev(rows, 0)
    rows_by_image_id = {row["image_id"]: row for row in stream}
    base = [record for record in read_stream(control_stream) if record["image_id"] in rows_by_image_id]
    blocks = regroup(base, rows_by_image_id, ledger, CONTROL, stream)
    per_frame_consistency(base, CONTROL, ledger)
    bad_order = score_order_ok(base)
    ledger.note("dev_stream_scores_are_non_increasing", bad_order == 0, f"{bad_order} frames")
    ledger.note("dev_stream_is_the_frozen_contract", len(stream) == args.expected_frames,
                f"{len(stream)} dev frames, expected {args.expected_frames}")
    ledger.note("dev_features_cover_the_dev_stream",
                {str(frame["image_id"]) for frame in dev_frames} == set(rows_by_image_id),
                f"{len(dev_frames)} feature frames vs {len(stream)} dev frames")
    if not smoke:
        ledger.note("train_archive_is_train_only",
                    {frame["split_role"] for frame in train_frames} == {"TRAIN"},
                    str(sorted({frame["split_role"] for frame in train_frames})))
    ledger.note("train_labels_cover_every_arm",
                all(str(arm) in frame["labels"]["delta_M"] for frame in train_frames for arm in ARMS_ALL),
                f"{len(train_frames)} frames x {len(ARMS_ALL)} arms")
    ledger.note("locked_roles_never_opened", True, "C_POLICY/T_ID/T_GENO/T_EXT are not read here")
    ledger.note("no_gpu_no_inference", True, "consumes archived cap-100 feature/label streams only")

    cost_payload = {}
    if Path(args.cost_eval).exists():
        cost_payload = json.loads(Path(args.cost_eval).read_text(encoding="utf-8")).get("cost_model", {})

    control_report = json.loads((QUALITY_ROOT / f"cap{CONTROL:05d}" /
                                 f"budget_report_{CONTROL:06d}.json").read_text(encoding="utf-8"))
    control_masks = float(np.mean([len(record["instances"]) for record in base]))
    control_role, control_domain = evaluate_control(blocks, data, rows_by_image_id, ledger)
    worst_control = max(abs(float(control_report["macro"][key]) - float(control_domain[key]))
                        for key in METRIC_KEYS)
    ledger.note("control_arm_reproduces_frozen_baseline", worst_control <= args.reproduce_tol,
                f"max |diff| {worst_control:.3e} pp")
    control = {"macro": control_domain, "macro_by_role": control_role,
               "per_genotype": control_report["per_genotype"],
               "mean_instances_masked_per_frame": control_masks}

    # ---- folds --------------------------------------------------------------------------------
    assignment = folds_by_plant(train_frames, args.folds)
    fold_index = np.asarray([assignment[frame["plant"]] for frame in train_frames], dtype=np.int64)
    folds = [{"fold": fold, "frames": int((fold_index == fold).sum()),
              "plants": len({frame["plant"] for frame, value in zip(train_frames, fold_index)
                             if value == fold}),
              "genotypes": len({frame["genotype"] for frame, value in zip(train_frames, fold_index)
                                if value == fold})} for fold in range(args.folds)]
    ledger.note("every_plant_in_exactly_one_fold",
                sum(item["plants"] for item in folds) == len(assignment), f"{len(assignment)} plants dealt")
    ledger.note("folds_cover_every_frame", sum(item["frames"] for item in folds) == len(train_frames),
                str([item["frames"] for item in folds]))

    # ---- heads: OOF for diagnostics, full-TRAIN fit for development decisions ------------------
    low, high = feature_band(x_train, continuous)
    flagged = out_of_band_fraction(x_dev, continuous, low, high)
    models = {}
    for seed in args.seeds:
        depths = (args.depth, 3) if (args.structure_probe and seed == args.seeds[0]) else (args.depth,)
        for depth in depths:
            key = f"seed{seed}_depth{depth}"
            oof = oof_predictions(x_train, train_frames, w_train, fold_index, ARMS_ALL, seed, depth)
            fitted = fit_heads(x_train, train_frames, w_train, ARMS_ALL, seed, depth)
            models[key] = {"seed": seed, "depth": depth, "oof": oof,
                           "dev": predict(fitted, x_dev, ARMS_ALL, len(dev_frames))}
            print(f"{key}: {len(fitted)} heads fitted, "
                  f"{time.perf_counter() - started:.0f} s elapsed", flush=True)
    ledger.note("seeds_parsed_as_integers", all(isinstance(value, int) for value in args.seeds),
                f"seeds {list(args.seeds)}, models {sorted(models)}")

    dev_ids = [str(frame["image_id"]) for frame in dev_frames]
    empty = np.asarray([float(frame["features"]["flag_empty_output"]) for frame in dev_frames])
    protect = (flagged > OUT_OF_BAND_LIMIT) | (empty > 0)
    if args.ablation == "A3":
        protect = np.zeros_like(protect)

    # ---- price every distinct policy once ------------------------------------------------------
    priced, candidates, first_seen = {}, [], {}

    def price(vector, name, kind):
        table = {iid: int(value) for iid, value in zip(dev_ids, vector)}
        return evaluate_arm(name, kind, lambda record, table=table: table[str(record["image_id"])],
                            blocks, data, rows_by_image_id, ledger, control)

    for key in sorted(models):
        model = models[key]
        if model["depth"] != args.depth:
            continue                                  # structure probe, never priced
        risk = stacked_risk(model["dev"], ARMS_FITTED)
        grid = [float("-inf")] + [float(np.quantile(risk.ravel(), value / 100.0))
                                  for value in TAU_PERCENTILES] + [float("inf")]
        for tau_index, tau in enumerate(grid):
            choice = np.where(protect, CONTROL, choose_budget(risk, ARMS_FITTED, tau))
            digest = vector_digest(choice)
            entry = priced.get(digest)
            if entry is None:
                entry = {"digest": digest, "vector": choice.astype(int).tolist(),
                         "budget_mean": round(float(choice.mean()), 3),
                         "budget_histogram": budget_histogram(choice),
                         "protected_frames": int(protect.sum()), "arm": None}
                if not args.skip_pricing:
                    entry["arm"] = price(choice, f"{tag}_{digest}", "learned")
                    entry["projected_pass_seconds"] = cost_projection(float(np.mean(choice)),
                                                                      cost_payload)
                    print(f"{key} tau[{tau_index:02d}] priced {digest} work "
                          f"{entry['arm']['mean_instances_masked_per_frame']:7.3f} dAP "
                          f"{entry['arm']['ap_drop_vs_control_pp']:7.4f} pp "
                          f"[{entry['arm']['elapsed_seconds']}s]", flush=True)
                priced[digest] = entry
                first_seen[digest] = f"{key}/tau{tau_index:02d}"
            elif entry["arm"] is not None:
                print(f"{key} tau[{tau_index:02d}] reuses {digest} from {first_seen[digest]}", flush=True)
            arm = entry["arm"]
            candidates.append({"key": key, "seed": model["seed"], "depth": model["depth"],
                               "tau_index": tau_index,
                               "tau": None if not math.isfinite(tau) else round(tau, 6),
                               "digest": digest, "first_priced_by": first_seen[digest],
                               "work": None if arm is None else arm["mean_instances_masked_per_frame"],
                               "ap_drop_pp": None if arm is None else arm["ap_drop_vs_control_pp"],
                               "count_increase_leaf": None if arm is None
                               else arm["count_MAE_increase_vs_control_leaf"],
                               "gate": None if arm is None else bool(arm["gate_ap_drop_within_1pp"]
                                                                     and arm["gate_count_MAE_within_0_2_leaf"])})

    # ---- P1: same budget multiset, random assignment, no scores --------------------------------
    random_policies = {}
    if not args.ablation and not args.skip_pricing:
        rng = np.random.default_rng(np.random.PCG64(43))
        for digest, entry in sorted(priced.items()):
            work = float(entry["arm"]["mean_instances_masked_per_frame"]) if entry["arm"] else None
            if work is None or not RANDOM_WORK_BAND[0] <= work <= RANDOM_WORK_BAND[1]:
                continue
            vector = np.asarray(entry["vector"])
            shuffled = rng.permutation(vector)
            name = f"P1_{digest}"
            arm = price(shuffled, name, "random_matched_work")
            arm["note"] = (f"permuted budget multiset of learned policy {digest}; the permutation also "
                           "relocates the protection fallbacks, so P1 is a matched-work control, not a "
                           "protected policy")
            random_policies[digest] = arm
            print(f"P1 {digest}: work {arm['mean_instances_masked_per_frame']:7.3f} dAP "
                  f"{arm['ap_drop_vs_control_pp']:7.4f} pp [{arm['elapsed_seconds']}s]", flush=True)

    # ---- reports -------------------------------------------------------------------------------
    structure = None
    if any(model["depth"] == 3 for model in models.values()):
        main_key = f"seed{args.seeds[0]}_depth{args.depth}"
        probe_key = f"seed{args.seeds[0]}_depth3"
        def mean_error(table):
            values = [entry["mean_abs_error"] for entry in table.values() if "mean_abs_error" in entry]
            return round(float(np.mean(values)), 6)
        main_table = calibration_table(models[main_key]["oof"], train_frames, ARMS_FITTED)
        probe_table = calibration_table(models[probe_key]["oof"], train_frames, ARMS_FITTED)
        structure = {
            "oof_calibration_depth2": main_table, "oof_calibration_depth3": probe_table,
            "mean_abs_error_depth2": mean_error(main_table),
            "mean_abs_error_depth3": mean_error(probe_table),
            "selected_depth": args.depth,
            "note": ("v1 section 4.3's stated criterion (development service time) is undefined for a "
                     "regressor, so the predeclared tie-break keeps depth 2; switching would need an "
                     "explicit rerun with --depth 3, and the comparison here uses TRAIN-OOF error only, "
                     "never development labels")}
    payload = {
        "version": VERSION, "generated_utc": utc(), "run_tag": tag, "code_smoke_only": smoke,
        "ablation": tag, "action_set": list(ARMS_ALL), "fitted_arms": list(ARMS_FITTED),
        "loss_scales": {"M": LOSS_SCALE_M, "N": LOSS_SCALE_N}, "seeds": list(args.seeds),
        "depth": args.depth, "heads_per_seed": len(STATISTICS) * len(ARMS_FITTED),
        "regressor": dict(GB_PARAMS, quantile=QUANTILE),
        "features_used": list(names), "features_dropped": [n for n in FEATURE_NAMES if n not in names],
        "train": {"archive": str(train_dir), "archive_suffix": train_suffix,
                  "frames": len(train_frames), "plants": len(assignment),
                  "genotypes": train_report.get("genotypes"), "stream": train_report.get("stream"),
                  "stream_sha256": train_report.get("stream_sha256"),
                  "features_check": str(train_dir / f"features_check_{train_suffix}.json")},
        "dev": {"archive": str(dev_dir), "archive_suffix": dev_suffix, "frames": len(dev_frames),
                "stream": dev_report.get("stream"), "stream_sha256": dev_report.get("stream_sha256"),
                "features_check": str(dev_dir / f"features_check_{dev_suffix}.json")},
        "folds": {"count": args.folds, "rng": "PCG64(43) genotype-stratified whole-plant",
                  "summary": folds},
        "protection": {"band": "TRAIN 0.5%-99.5% per continuous feature", "limit": OUT_OF_BAND_LIMIT,
                       "fallback": CONTROL, "enabled": args.ablation != "A3",
                       "frames_sent_to_fallback": int(protect.sum()),
                       "mean_fraction_out_of_band": round(float(flagged.mean()), 4)},
        "calibration_oof": {key: calibration_table(model["oof"], train_frames, ARMS_FITTED)
                            for key, model in sorted(models.items())},
        "drift": drift_report(x_train, x_dev, names, continuous, low, high, flagged),
        "structure_probe": structure,
        "control": control,
        "predeclared_tolerances": {"AP_drop_vs_control_pp": AP_DROP_LIMIT_PP,
                                   "count_MAE_increase_vs_control_leaf": COUNT_INCREASE_LIMIT_LEAF},
        "cost_model": cost_payload or None,
        "tau_grid_definition": ("[-inf] + percentiles 5..95 step 5 of that seed's pooled development "
                                "risk scores over (frame, arm) + [+inf]"),
        "distinct_policies_priced": sum(1 for entry in priced.values() if entry["arm"]),
        "candidates": candidates,
        "policies": {digest: entry["arm"] for digest, entry in sorted(priced.items()) if entry["arm"]},
        "policy_shapes": {digest: {k: v for k, v in entry.items() if k not in ("arm", "vector")}
                          for digest, entry in sorted(priced.items())},
        "p1_random_matched_work": random_policies,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "ledger": {"checks": len(ledger.checks), "failures": ledger.failures},
        "limitation": ("single 1,000-update pilot checkpoint; the detector was trained on TRAIN, so "
                       "TRAIN-side calibration is in-sample for the backbone while the development "
                       "readings are not. Point estimates on the development domain, not a "
                       "non-inferiority claim, and no locked role was consulted."),
    }
    write_json(out / f"controller_{tag}.json", payload)
    arms_out = ([entry["arm"] for _, entry in sorted(priced.items()) if entry["arm"]]
                + list(random_policies.values()))
    if arms_out:
        write_json(out / f"arms_{tag}.json", {
            "version": VERSION, "generated_utc": utc(), "run_tag": tag, "code_smoke_only": smoke,
            "predeclared_tolerances": payload["predeclared_tolerances"],
            "control": {"mean_instances_masked_per_frame": control_masks},
            "cost_model": cost_payload or None, "arms": arms_out})
    print(f"\n{tag}: {len(candidates)} candidates -> {payload['distinct_policies_priced']} distinct "
          f"policies + {len(random_policies)} matched-work randomizers, "
          f"{payload['elapsed_seconds']} s, ledger failures {len(ledger.failures)}", flush=True)
    for name in ledger.failures[:10]:
        print("FAIL", name, flush=True)
    return 0 if not ledger.failures else 1


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
