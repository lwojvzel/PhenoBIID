# Reproducibility boundary

This repository fixes the model definitions, data registry, protocol, metric
implementation, and paper-level reference values. Full selected processed
numeric arrays are distributed through
[Hugging Face](https://huggingface.co/datasets/PHENOBIID/CropDynamicsBench),
not Git. See [PROCESSED_RELEASE.md](PROCESSED_RELEASE.md) for retrieval,
checksums, scope and the distinction from legacy training caches.

## Immediate checks

1. Metric behavior: `python -m unittest discover -s tests -v`.
2. Manuscript-table consistency: `python evaluation/validate_reference.py`.
3. Model definitions: `src/phenobiid/`.
4. Exact paper configuration: `configs/paper_protocol.json`.
5. Real-data interface checks: `data/sample/` and `tests/test_sample_data.py`.

`scripts/reconstruct_raw.py` provides staged GDHY, ERA5-Land, crop-calendar,
LAI, NDVI and GPP processing from user-acquired raw products. The independent
`scripts/build_features.py` builds cohorts and physical feature interfaces
without fitted model predictions. The original dependency closure is retained
under `workflow/`, and selected fitting recipes are in `configs/selected_recipes/`.
See [RECONSTRUCTION.md](RECONSTRUCTION.md) for commands, tests and remaining
training integration requirements. Source availability is not equivalent to
a verified fresh-clone training reproduction.

Full retraining and grid-level replay require the registered products, their
crop-active alignment, selected historical experts, state checkpoints, and
crop-specific LightGBM residual heads. Product licenses remain independent.
The manuscript and its LaTeX source are distributed separately from this repository.

The main table can now be recomputed from the included annual, seed-wise
scores using `evaluation/rebuild_main_table.py`. This checks means and sample
standard deviations rather than asserting that a particular method wins.

The paper's reference state inference used CUDA mixed precision. CPU FP32 or
different batching can produce small numerical differences; report rather than
silently replace those differences.

The bundled sample provides a narrower but executable check: it covers all
four crop-specific signal choices and 10%, 30%, 50%, and 70% cutoffs on fixed
samples from 2006--2008. `examples/replay_real_sample.py` runs the complete
frozen CPU forward pass. It is deliberately excluded from the lightweight CI
job because installing PyTorch, LightGBM, TabM, and their compiled dependencies
would dominate routine validation.
