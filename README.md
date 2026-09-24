# PhenoBIID and CropDynamicsBench

Official release workspace for **PhenoBIID**, a weather-conditioned crop world
model, and **CropDynamicsBench**, its global in-season vegetation-to-yield
evaluation protocol.

The benchmark represents crop development with at most 12 crop-active slots.
At each cutoff, PhenoBIID keeps the observed vegetation prefix, forecasts the
hidden NDVI/GPP suffix under given weather, and sends the completed trajectory
to a crop-specific, history-anchored yield readout.

## Repository contents

- `src/phenobiid/`: model definitions used by the released pipeline.
- `evaluation/`: metric and reference-table validation utilities.
- `data/reference/`: machine-readable values reported in the manuscript.
- `data/sample/`: a 22 MB real-data sample with four crops and four cutoffs.
- `configs/`: fixed paper protocol and source-product registry.
- `configs/selected_recipes/`: history, state, and yield-head refit settings.
- `scripts/`: raw reconstruction and independent feature-building entrypoints.
- `reconstruction/`: dependency-light preprocessing definitions.
- `workflow/`: original experiment sources with provenance and licenses.
- `docs/`: dataset, evaluation, and reproducibility documentation.

## Reproduction guide

There are three distinct workflows. Start with **A** to run the model itself.

| Workflow | What it verifies | Full dataset needed? |
|---|---|---|
| **A. Real-data example inference** | Weather-conditioned BIID rollout, historical expert and final yield readout using released weights | No; inputs and weights are included |
| **B. Published-table reconstruction** | Aggregation of saved annual scores into the reported means and seed SDs | No; annual scores are included |
| **C. Full-data reconstruction** | Raw-product processing and physical feature construction | Yes; obtain upstream products first |

Workflow A is frozen inference, not training. Workflow B does not rerun models.
Workflow C provides preprocessing entrypoints, but the complete fresh-data
training chain still has integration requirements described below. None of
these checks alone constitutes a new full-benchmark training reproduction.

### 1. Download the repository and install dependencies

The commands below use a Linux/macOS-style shell. Run all subsequent commands
from the repository root, with the virtual environment activated.

```bash
git clone https://github.com/lwojvzel/PhenoBIID.git
cd PhenoBIID
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p outputs
```

The example explicitly runs on **CPU**; no GPU, CDS account, external dataset
download, or access to the original research server is required. PyTorch,
LightGBM, TabM and the other Python dependencies are installed by the command
above. The additional `requirements-workflow.txt` is for optional original
baselines and diagnostics, not required for this example.

### 2. Check the bundled files

```bash
python -m unittest discover -s tests -v
```

Expected outcome: the tests finish with `OK`. They check sample file hashes,
input shapes, hidden-slot masks, temporal partitions, metric pairing, source
manifests and table aggregation. They do not perform the neural-network
forward pass; run the next step for that.

### A. Run complete BIID-to-yield inference

#### Included data and weights

The approximately 22 MB example is in `data/sample/`. It contains **24 real
grid-cell/year samples per crop**: eight samples from each of 2006, 2007 and
2008. All included components were fitted through 2005. Each crop is evaluated
at nominal 10%, 30%, 50% and 70% unobserved fractions, giving **16 cases**.

| Crop | Vegetation product(s) | Final yield-head input |
|---|---|---|
| Maize | GPP | `[24, 501]` |
| Rice | NDVI | `[24, 501]` |
| Soybean | NDVI | `[24, 501]` |
| Wheat | NDVI and GPP | `[24, 537]` |

For each crop, `data/sample/assets/<crop>/` contains input files, separate
reference files, metadata, and the pretrained components. For example,
`cutoff_030_inputs.npz` supplies the 30% case; the matching
`cutoff_030_reference.npz` is used only after the forward prediction for
verification. The manifest lists the bundled assets and checksums.

#### What enters the model

Here `B=24` and the maximum sequence length is 12 **crop-active slots**, not
necessarily twelve consecutive January--December observations.

| Input | Shape | Purpose |
|---|---|---|
| Previous-year vegetation, per product | `[B, 12]` | Initialize the vegetation state together with context |
| Given weather, per product interface | `[B, 12, 13]` | Condition state evolution across crop-active slots |
| Observed vegetation prefix, per product | `[B, 12]` | Supply observation feedback; hidden/inactive entries are `NaN` |
| State context, per product | `[B, 5]` | Provide contextual features to the state model |
| Active and hidden-suffix masks | `[B, 12]` each | Identify valid slots and the forecast window |
| Historical-yield/context features | `[B, 20]` | Feed the frozen historical expert |
| Common yield-head features | `[B, 465]` | Combine history/context, metadata/support and weather |

The common features comprise 20 history/context columns, 289 metadata/support
columns, and 156 weather columns (`12 x 13`). Additional masks, quality arrays,
training climatologies and normalization parameters are supplied in the bundle.
The script validates these contracts before producing a prediction.

The weather inputs are realized full-season weather, as in the paper's
retrospective conditional setting. This example is not an operational weather
forecast demonstration.

#### What the script computes

```text
Previous vegetation + context
             |
             v
Weather-conditioned BIID state rollout
  observed window: use available vegetation observations as feedback
  hidden window:   use predicted vegetation as feedback
             |
             v
Observed prefix + predicted suffix -> completed trajectory [B, 12]
             |
             v
Trajectory values/statistics + local anomaly values/statistics
                         -> 36 features per vegetation product
             |
             v
Concatenate with 465 common features -> [B, 501] or [B, 537]
             |
             v
Frozen historical prediction + LightGBM residual correction
             |
             v
Annual grid-cell yield prediction [B], in t/ha
```

Wheat uses separate NDVI and GPP state models, then concatenates their
trajectory features. Maize averages two corrected historical branches;
the other crops use their configured single corrected branch. Soybean's
historical path also includes its saved TabM component.

The implementation is in
[`examples/replay_real_sample.py`](examples/replay_real_sample.py):
`cpu_forward()` calls `history()`, `state()` and `terminal()`.
It computes yield before opening the reference targets and predictions;
the generated yield is not copied from the reference files.

#### Run it

```bash
python examples/replay_real_sample.py --output outputs/sample_replay.json
```

The default data location is resolved relative to the repository, not the
original server. An alternative bundle can be supplied with `--root` but must
follow the same manifest, input and weight contracts.

#### Read the output and check success

The command prints one JSON record per crop/cutoff case and writes a combined
report to `outputs/sample_replay.json`. Successful completion should give:

- `archived_replay_passed: true`;
- `cpu_fp32_finite: true`;
- 16 entries in `cases`, each with `samples: 24`;
- feature shapes matching the crop table above.

Two different checks are reported:

| Report field | Interpretation |
|---|---|
| `archived_replay_max_error` | Yield-head replay using archived completed trajectories and anchors; expected to be zero or within the script's numerical tolerance |
| `cpu_fp32_yield_max_difference` | Maximum yield difference between the newly computed CPU pipeline and archived predictions |
| `cpu_fp32_state_max_difference` | Maximum hidden-slot vegetation difference for each product |
| `cpu_fp32_anchor_max_difference` | Maximum difference in historical-anchor predictions |

The original archived run used CUDA mixed precision, while this replay uses
CPU FP32. The latter three differences are reported rather than forced to
zero. They measure numerical replay differences, **not yield RMSE or model
improvement over a baseline**.

**Output boundary:** this CLI currently saves the verification report, not a
CSV of each sample's yield or a file of completed trajectories. Those arrays
are computed by `cpu_forward()`, which returns
`prediction, trajectories, anchors, features`. The small example checks the
complete inference path; it must not be used to estimate the paper's global
thirteen-year performance.

More sample details are in [`data/sample/README.md`](data/sample/README.md).

### B. Reconstruct the published tables

```bash
python evaluation/validate_reference.py
python evaluation/rebuild_main_table.py --output outputs/main_table.csv
```

The main-table reconstruction uses 1,872 saved annual RMSE scores: four crops,
three methods, four cutoffs, three seeds and thirteen years. It first averages
annual spatial RMSE equally across years within each seed, then calculates
the mean and sample standard deviation across seeds (`ddof=1`).

Expected outcome: all 48 crop/method/cutoff entries agree with the released
main table within four-decimal rounding. `validate_reference.py` also
reconstructs the seed-42 direct-predictor table and checks the vegetation table
schema. Source CSVs and provenance are in
[`data/reference/annual/`](data/reference/annual/).

#### Evaluate your own prediction CSV

Use one row per sample and prediction condition, with required columns
`crop`, `year`, `target`, and `prediction`. Targets and predictions must be
in the same units, here t/ha. Optional `method`, `seed`,
`unobserved_fraction`, and `product` columns separate evaluation conditions.

```bash
python evaluation/evaluate_predictions.py \
  --predictions path/to/predictions.csv \
  --output outputs/mean_annual_rmse.csv \
  --annual-output outputs/annual_rmse.csv
```

This computes spatial RMSE within each crop/year/condition and then averages
annual RMSE equally across years. It does not pool all years into one RMSE
or automatically average different seeds. See
[`docs/EVALUATION.md`](docs/EVALUATION.md) for metric definitions.

### C. Reconstruct full-data inputs

This path needs the upstream raw products and considerably more storage than
the example. **Do not use the original project directory as the output
workspace.** Follow the raw directory layout in
[`docs/RECONSTRUCTION.md`](docs/RECONSTRUCTION.md), then run:

```bash
# Initialize source scripts in a separate workspace.
python scripts/reconstruct_raw.py --workspace /your/reconstruction --stage init

# Place the registered raw products in that workspace as documented.
# Process GDHY, weather, crop calendars, LAI, NDVI and GPP in order.
python scripts/reconstruct_raw.py --workspace /your/reconstruction --stage all

# Build cohorts and physical feature interfaces without old model predictions.
python scripts/build_features.py --workspace /your/reconstruction --stage cohort
python scripts/build_features.py --workspace /your/reconstruction --stage features
```

The stages do not download raw products automatically. GDHY conversion needs
the system `h5dump` executable, and LAI processing needs HDF4/`pyhdf` support.
Individual stages and a non-writing `--dry-run` option are documented in the
reconstruction guide.

#### Temporal partitions

| Final fitting cutoff | Inner fitting through | Inner validation | Evaluation years |
|---|---|---|---|
| 2001 | 1999 | 2000--2001 | 2002--2004 |
| 2005 | 2003 | 2004--2005 | 2006--2008 |
| 2009 | 2007 | 2008--2009 | 2010--2016 |

Generate positional sample indices from a CSV containing a `year` column:

```bash
python evaluation/splits.py --samples path/to/samples.csv --output outputs/splits
```

The output contains one NPZ per fitting cutoff, with `inner_train`,
`inner_validation`, `final_train`, and `evaluation` index arrays. Keep the
sample row order unchanged when using these indices. Fixed component recipes
are recorded separately in `configs/selected_recipes/`.

#### Full retraining status

The original training sources and selected configurations are available, but
**the entire fresh-data-to-training-to-evaluation chain is not yet a verified
standalone workflow**. Some original training entrypoints require earlier
experiment caches. Linking the newly built features to every training
component, completing the generic 70% evaluation path, and resolving precise
MIRCA/LAI acquisition details remain necessary. Do not treat the commands above
as commands that reproduce every trained model or every paper experiment.

### Troubleshooting

| Symptom | What to check |
|---|---|
| Missing `torch`, `lightgbm`, `tabm` or another module | Activate the same environment used for dependency installation; install `requirements.txt` |
| Missing sample files or checksum failure | Use an intact checkout; do not edit or rename the bundled assets |
| Cannot write the output JSON | Create the parent directory first, for example `mkdir -p outputs` |
| CPU predictions differ slightly from saved predictions | Check the reported precision differences; exact equality is tested separately for the archived yield-head replay |
| Raw processing cannot find a source product | Check the documented raw layout and upstream version; the small sample does not contain full raw products |

## Data availability

The complete processed arrays are about 85 GB and combine products governed by
different upstream terms. They are therefore not committed to Git. Exact
versions, official records, transformations, and the expected layout are in
[docs/DATASET.md](docs/DATASET.md) and
[`configs/data_sources.json`](configs/data_sources.json). This repository ships
paper-level reference measurements but does not relabel upstream products under
a single project license.

### Real-data inference sample

The repository includes a deterministic sample from 2006--2008 for maize,
rice, soybean, and wheat at 10%, 30%, 50%, and 70% unobserved suffixes. It
separates model-visible inputs from reference targets and includes the frozen
state, historical, and yield-readout components. See
[`data/sample/README.md`](data/sample/README.md) for tensor shapes, selection
rules, limitations, and the optional complete CPU replay.

## Evaluation scope

The primary study uses four crops, thirteen evaluation years, seeds 42/45/48,
and nominal unobserved suffixes of 10%, 30%, 50%, and 70%. Realized full-season
ERA5-Land weather defines the main retrospective conditional setting. SEAS5 is
a separate nine-year sensitivity analysis, not an operational forecast claim.

See [docs/EVALUATION.md](docs/EVALUATION.md) for metric definitions and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the release boundary.
Raw-data layout, stage commands, temporal splits, and verified versus pending
steps are described in [docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md).
Full fresh-clone retraining is not yet verified: some original training
entrypoints still require earlier experiment caches. The small sample and
table reconstruction are executable checks, not replacements for that test.

## Citation

Citation metadata will be updated when the paper receives a public identifier.
