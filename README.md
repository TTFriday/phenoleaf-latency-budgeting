# PhenoLeaf latency budgeting

Code, derived features and archived results for output-side compute-budget
control in leaf instance segmentation.  The package separates per-frame
detection quality, service-time decomposition, rule/controller pricing,
mask-restore resolution and equal-work comparisons.

## Contents

- `src/`: detector pilot, budget-quality evaluation, pricing, controller,
  timing and independent verification programs.
- `jrtip/`: selected formal run outputs.  Large prediction streams,
  checkpoints, interrupted attempts and smoke runs are intentionally absent.
- `shared/pilot_inventory.csv`: development-role inventory with portable paths.
- `tests/`: synthetic CPU checks; these are not scientific results.
- `results/numbers.md`: readable result ledger.
- `SOURCE_MANIFEST.json`, `MANIFEST.json`, `MANIFEST.gitblob.sha1`: provenance
  and per-file SHA-256 / Git blob SHA-1 checks.

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -p "test_jrtip*.py"
```

The tests use synthetic inputs and do not open real media, train a model or
evaluate locked roles.  The programs in `src/verify_jrtip*.py` document the
independent checks used for the original runs.  Replaying them requires
omitted raw streams, benchmark configuration and, where applicable, the
original media or checkpoint; they cannot run from this package alone.
Their arguments are shown by `--help`.

## Scientific scope

The latency equation is a configuration-level mean model, not a per-frame
hard-deadline guarantee.  Reported tail latency is the arithmetic mean of
within-session p95 values.  Quality and timing results are development-role
pilot evidence.  No locked-role accuracy or sampling effect was inspected.

The public copies remove machine snapshots, embedded source-code dumps and
local absolute paths.  These normalization steps change file hashes but not
scientific table values; original and published hashes are listed in
`SOURCE_MANIFEST.json`.  Controller outputs retain the archived
`policy_shapes` evidence; per-frame `vector` fields and their fit ledgers
were not present in the final archived output and are not claimed to be.

## Re-running experiments

Re-running inference or training requires the original PhenoLeaf-TS media,
the COCO initialization checkpoint and a compatible CUDA environment.  Those
inputs are not included.  The packaged feature tables and aggregate outputs
support offline inspection and controller refitting without raw media.

No code licence is granted by this repository.  Third-party data and model
artefacts retain their own licences.
