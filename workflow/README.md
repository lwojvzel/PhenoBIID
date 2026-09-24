# Original experiment source

This directory preserves 236 Python modules in the dependency closure of the
paper experiment entrypoints. `entrypoints.json` groups them by task; the
source manifest records original and exported SHA-256 hashes. Only local root
literals were normalized during export, except for the explicitly recorded
optional GPP cohort-audit switch. Third-party license notices are retained.

These are research-workspace entrypoints, **not a verified one-command fresh
training pipeline**. Several trainers load selected recipes, earlier fitted
historical experts, or cached reference predictions. The guarded audit drivers
also expect registrations generated in the original workspace. Copying the
source alone does not satisfy those dependencies.

Use the release-level scripts rather than running the audit drivers directly:

- `scripts/reconstruct_raw.py`: independent raw-product stages.
- `scripts/build_features.py`: NumPy-only cohort and physical-feature stages.
- `evaluation/splits.py`: actual sample indices for the three paper blocks.
- `examples/replay_real_sample.py`: complete frozen inference on included data.
- `evaluation/rebuild_main_table.py`: annual-score aggregation, including SD.

`configs/selected_recipes/` provides crop/block/seed configurations for history,
state, and terminal refits. These configurations are fitted-recipe records, not
a substitute for generating the training arrays or rerunning model selection.
Some historical recipes contain a zero-epoch initialization component, as in
the released manuscript; this export does not silently retrain or replace it.

The original generic pipeline inference source covers 10/30/50%. The included
sample replay and annual main-table records additionally cover 70%. Do not
interpret the generic source's default ratios as the complete paper table.

See `docs/RECONSTRUCTION.md` for execution order and current verification limits.
