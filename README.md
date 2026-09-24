# PhenoBIID and CropDynamicsBench

Official release workspace for **PhenoBIID**, a weather-conditioned crop world
model, and **CropDynamicsBench**, its global in-season vegetation-to-yield
evaluation protocol.

The benchmark represents crop development with at most 12 crop-active slots.
At each cutoff, PhenoBIID keeps the observed vegetation prefix, forecasts the
hidden NDVI/GPP suffix under given weather, and sends the completed trajectory
to a crop-specific, history-anchored yield readout.

## Repository contents

- `paper/`: verified manuscript PDF and matching LaTeX source archive.
- `src/phenobiid/`: model definitions used by the released pipeline.
- `evaluation/`: metric and reference-table validation utilities.
- `data/reference/`: machine-readable values reported in the manuscript.
- `configs/`: fixed paper protocol and source-product registry.
- `scripts/`: source-product conversion and aggregation utilities.
- `docs/`: dataset, evaluation, and reproducibility documentation.

## Quick check

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evaluation/validate_reference.py
```

Evaluate a prediction CSV containing `crop`, `year`, `target`, and
`prediction`:

```bash
.venv/bin/python evaluation/evaluate_predictions.py \
  --predictions path/to/predictions.csv \
  --output metrics.csv
```

## Data availability

The complete processed arrays are about 85 GB and combine products governed by
different upstream terms. They are therefore not committed to Git. Exact
versions, official records, transformations, and the expected layout are in
[docs/DATASET.md](docs/DATASET.md) and
[`configs/data_sources.json`](configs/data_sources.json). This repository ships
paper-level reference measurements but does not relabel upstream products under
a single project license.

## Evaluation scope

The primary study uses four crops, thirteen evaluation years, seeds 42/45/48,
and nominal unobserved suffixes of 10%, 30%, 50%, and 70%. Realized full-season
ERA5-Land weather defines the main retrospective conditional setting. SEAS5 is
a separate nine-year sensitivity analysis, not an operational forecast claim.

See [docs/EVALUATION.md](docs/EVALUATION.md) for metric definitions and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the release boundary.

## Citation

Citation metadata will be updated when the paper receives a public identifier.
