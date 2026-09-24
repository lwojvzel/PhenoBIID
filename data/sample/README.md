# Real-data sample

This directory contains a compact, deterministic sample of the processed
CropDynamicsBench interface and the frozen PhenoBIID components needed to
replay it on CPU. It is intended for checking tensor contracts, cutoff logic,
and the complete state-to-yield inference path. It is not a substitute for the
full benchmark and must not be used to estimate the paper's reported gains.

## Scope

- Crops: maize, rice, soybean, and wheat.
- Years: 2006, 2007, and 2008.
- Samples: eight grid cells per crop and year, or 24 samples per crop.
- Cutoffs: nominal 10%, 30%, 50%, and 70% unobserved suffixes.
- Signals: maize uses GPP; rice and soybean use NDVI; wheat uses NDVI and GPP.
- Sampling: rows are sorted by grid coordinates and selected at equal
  intervals within each year. Targets and prediction errors are never used.
- Training cutoff: all bundled components were fitted through 2005.

Every `cutoff_*_inputs.npz` contains only model-visible data. Targets,
archived predictions, completed trajectories, and sample identities are kept
in the matching `cutoff_*_reference.npz`. The replay computes predictions
before opening the reference file.

## Main tensors

| Array | Shape | Meaning |
|---|---:|---|
| `history_x` | `[B, 20]` | Prepared historical-yield and context features |
| `trend` | `[B]` | Historical yield trend anchor |
| `<signal>_weather` | `[B, 12, 13]` | Given weather over crop-active slots |
| `<signal>_previous` | `[B, 12]` | Previous-year vegetation trajectory |
| `<signal>_context` | `[B, 5]` | State-model spatial and support context |
| `<signal>_prefix` | `[B, 12]` | Observed prefix; hidden and inactive slots are `NaN` |
| `active`, `tail` | `[B, 12]` | Crop-active and unobserved-suffix masks |
| `common` | `[B, 465]` | Yield-head history, metadata, support, and weather features |

Here `B=24`. One vegetation product contributes 36 completed-trajectory
features, producing a 501-column yield-head matrix. Wheat uses two products
and therefore produces 537 columns.

## Checks

The regular test suite validates all files without loading model checkpoints:

```bash
python -m unittest discover -s tests -v
```

The optional full CPU replay loads the frozen PyTorch and LightGBM models:

```bash
python examples/replay_real_sample.py --output sample_replay.json
```

The archived readout must reproduce exactly. Recomputing the state models on
CPU uses FP32, whereas the archived paper run used CUDA mixed precision, so
the script reports those numerical differences instead of claiming bitwise
identity.

## Data and licensing boundary

These files are small derived samples supplied for research reproducibility.
They do not redistribute complete upstream products and do not replace their
licenses or terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md),
the repository data registry, and the official source records before obtaining
or redistributing full products.
