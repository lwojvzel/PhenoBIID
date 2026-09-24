# Reconstruction and verification

## Executable checks

From the repository root, with Python and dependencies installed:

```bash
python -m unittest discover -s tests -v
python evaluation/validate_reference.py
python evaluation/rebuild_main_table.py --output outputs/recomputed_main.csv
python examples/replay_real_sample.py --output outputs/sample_replay.json
```

The main table is reconstructed from 1,872 annual scores: four crops, three
methods, four cutoffs, three seeds and thirteen years. Annual spatial RMSE is
averaged equally over years within each seed, then the mean and sample SD
(`ddof=1`) are calculated across seeds. Agreement is checked against the
four-decimal published table. This checks aggregation, not a fresh training run.

## Full raw-product processing

Obtain the registered products under their upstream terms. Initialize an empty
workspace outside this repository; never point the script at your original
research project.

```bash
python scripts/reconstruct_raw.py --workspace /your/reconstruction --stage init
```

Place raw data inside that workspace with this layout:

```text
Data/GDHY/gdhy_v1.2_v1.3_20190128/<crop>/yield_<year>.nc4
Data/era5land/monthly/era5land_monthly_<year>.nc
Data/MIRCA-OS/Monthly Growing Area Grids/Monthly Growing Area Grids/
Data/GLASS_LAI_AVHRR_005D/<year>/*.hdf
Data/external/PKU_GIMMS_NDVI_v1p2/*consolidated*.zip
Data/external/reclue_monthly_gpp_probe/1982.zip
Data/external/reclue_monthly_gpp_v1/<year>.zip
```

GPP's latter directory contains 1983--2016. NDVI/GPP cover 1982--2016;
GDHY, ERA5-Land and legacy LAI inputs cover 1981--2016. Keep the original
provider metadata alongside archives. The MIRCA parser checks the archived
coordinate convention explicitly; a different product version must not be
silently substituted. The source registry still needs a precise public
MIRCA snapshot record for a fully self-service download path.

GDHY conversion requires the `h5dump` executable (typically `hdf5-tools` or
Conda's `hdf5` package). LAI reading requires HDF4 support through `pyhdf`.
The Python dependencies are in `requirements.txt`; optional experiment and
figure dependencies are in `requirements-workflow.txt`.

```bash
python scripts/reconstruct_raw.py --workspace /your/reconstruction --stage all
python scripts/build_features.py --workspace /your/reconstruction --stage cohort
python scripts/build_features.py --workspace /your/reconstruction --stage features
```

Stages can also be run individually in order: `gdhy`, `weather`, `calendar`,
`lai`, `ndvi`, `gpp`. `--dry-run` prints the plan without creating files.
Do not run competing stages in the same output directory. Raw processing
does not request credentials, download data, or read fitted model weights.
The GPP gate verifies full-period array hashes and rechecks every crop-slot
mapping; cohort-support diagnostics are deliberately deferred until cohorts
exist. The feature builder uses the original deterministic coverage calculation
instead of loading predictions from an earlier model.

Outputs include the original processed product layout and
`benchmark/cache/fresh_complete_inputs/<crop>/`. This last interface contains
physical trajectories, weather, identities, 289 metadata columns and training
statistics for each refit cutoff. It is not yet the complete training-package
interface consumed by every historical experiment runner.

The intermediate cache names `origin_2004`, `origin_2008`, `origin_2012` and
their legacy split labels belong to deterministic feature reconstruction.
The final paper evaluation uses the following blocks, not those cache labels:

| Final fitting through | Inner fitting through | Inner validation | Evaluation |
|---|---|---|---|
| 2001 | 1999 | 2000--2001 | 2002--2004 |
| 2005 | 2003 | 2004--2005 | 2006--2008 |
| 2009 | 2007 | 2008--2009 | 2010--2016 |

Given a sample CSV with a `year` column, generate positional indices with:

```bash
python evaluation/splits.py --samples samples.csv --output outputs/splits
```

Inner splits select settings; final training refits through the cutoff.
Earlier evaluation years become available history in later chronological
blocks. Normalization and climatological/support statistics must be fitted
using the corresponding block's training rows, never the evaluated rows.

## Verification performed for this update

- Main table: all 48 crop/method/cutoff entries reconstructed from annual data.
- Small sample: all 16 crop/cutoff combinations replayed on CPU; archived
  trajectory readout matched exactly. CPU FP32 recomputation is reported
  separately from the original mixed-precision trajectory predictions.
- Raw processing: 1982 GPP regenerated from its original ZIP in a new temporary
  workspace. All 17 arrays, including four crop mappings, matched original
  processed arrays exactly. This test reused existing crop-calendar mappings.
- NumPy-only feature modules: independent import test without loading Torch
  or fitted model runtimes. No full 85 GB rebuild was run in this update.
- Original workflow source: syntax and manifest checks, not execution of
  all trainers or auxiliary experiments.

## Remaining end-to-end release work

The repository now exposes raw processing, feature-building definitions,
selected recipes, original training/evaluation sources, frozen inference and
score aggregation. **Full fresh-clone training reproduction is not yet
verified or complete.** Remaining work is to connect the newly generated
feature interfaces to the fixed history/state/readout training packages
without old experiment caches, provide the generic 70% evaluation path, and
test that complete chain in an isolated environment. The exact public MIRCA
archive and version-specific LAI acquisition instructions also need resolution.

Do not use the source archive's presence, a passing lightweight CI run, or
the small example as a claim that all full-data experiments have been rerun.
