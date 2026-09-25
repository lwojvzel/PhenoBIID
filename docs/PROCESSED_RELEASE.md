# Full processed numeric release

Dataset: https://huggingface.co/datasets/PHENOBIID/CropDynamicsBench

The release contains 2,691 files in 52 lossless shards: 5.48 GiB compressed
and 44.40 GiB extracted, including SEAS5. Reserve roughly 60 GiB for both the
download cache and extracted arrays. These counts include coordinate arrays
and variable-order files. Smoke-test duplicates and exploratory caches are not
part of the release.

Use `scripts/download_processed_data.py --workspace /your/benchmark`. The
default products are `crop_active era5_land gdhy lai ndvi gpp`; add
`--products all` to include native SEAS5 forecasts. `--products ndvi gpp`
downloads only those products. `--list-only` reports compressed sizes without
retrieving archives. `--revision` accepts an immutable dataset commit hash.
The default version and manifest checksum are pinned in
`configs/processed_release.json`; `--revision main` explicitly follows a newer
dataset branch instead.

## Extracted layout

```text
Data/
  GDHY/gdhy_v1.2_v1.3_20190128_npy_lon180/
  era5land/monthly_npy_lon180_0p5deg_by_var/
  processed/
    crop_yield_growing_season/
    glass_lai_avhrr_005d/
    pku_gimms_ndvi_v1p2/
    reclue_monthly_gpp_v1/
    seas5_weather_reliability_v1/native_1deg/  # optional
```

## Tensor interfaces

| Product | Shape | Coverage |
|---|---|---|
| Annual yield | `[360,720]` | Four crops, 1981--2016 |
| Monthly weather, each variable | `[12,360,720]` | 13 variables, 1981--2016 |
| Crop-active weather | `[13,12,360,720]` | Four crops, 1981--2016 |
| NDVI and support arrays | `[12,360,720]` | Monthly and four-crop aligned, 1982--2016 |
| GPP total/rate/support | `[12,360,720]` | Monthly and four-crop aligned, 1982--2016 |
| Legacy LAI | `[12,360,720]` | Monthly and four-crop aligned, 1981--2016 |
| Native SEAS5 ensemble mean | `[6,180,360,3]` | 432 initializations, 1981--2016 |

Read `lat.npy`, `lon.npy` and `variable_order.txt` rather than assuming axis or
variable ordering. The main grid uses ascending latitude and -180..180
longitude. Crop-active slots use the shipped MIRCA snapshot mappings; they
are not necessarily twelve consecutive months. Yield is t/ha. NDVI is
dimensionless. GPP daily mean rates are gC m-2 day-1. Use validity/support masks
and training-only normalization; do not silently replace all missing values
by zeros.

Each crop's `mirca/{2000,2005,2010,2015}/` folders include `[12,360,720]`
mapping arrays. `month_rel` contains natural month numbers 1--12 (0 padding),
`src_rel` contains indices 0--11 (255 padding), and `valid_rel` marks active
slots with 1. Valid months are packed in ascending calendar-month order.
`weight_rel` is each month's growing area divided by the sum over valid
months at the same cell. Use the snapshot-selection rule in the processing
code for each year; these are not a universal January-to-December input mask.

SEAS5 channels are `t2m,tp,ssrd` in K, m/day and J/m2/day. Unlike the main
grid, latitude descends from 89.5 to -89.5 and longitude runs 0.5..359.5.
These are ensemble means on the native grid, not already bias-corrected,
crop-aligned inputs. The separate crop-weather processing/calibration remains
necessary for weather-reliability experiments.

## Integrity and boundaries

The Hugging Face `manifest.json` records per-file and per-shard SHA-256, sizes,
shapes and dtypes. The downloader validates both levels and refuses to replace
an existing different file. It only extracts listed regular files inside the
workspace. The selected revision is recorded locally.

Arrays are byte-identical to the selected processed source files. Compression
does not modify values. The release excludes smoke tests, intermediate MIRCA
caches, exploratory CPC/crop-weighted products, raw provider archives, papers,
credentials, model weights and machine-specific audit JSON/TSV files. The
sample weights remain in the GitHub inference example.

This is the full selected numeric input release, not a portable copy of every
research-workspace cache. Some original scripts require audit gates or training
caches not supplied by the numeric download. The verified and pending
reconstruction stages are documented in RECONSTRUCTION.md; full fresh-data
retraining is not claimed solely on the basis of data publication.

Source records and attribution are in `configs/data_sources.json` and the
dataset's `sources.json` and `TERMS.md`. No blanket project license overrides
provider terms. See `configs/paper_protocol.json` for temporal splits, seeds
and cutoffs; it is mirrored as `protocol.json` with the dataset.
