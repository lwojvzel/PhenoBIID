# Reproducibility boundary

This repository fixes the manuscript, model definitions, data registry,
protocol, metric implementation, and paper-level reference values. It does not
place the full 85 GB processed benchmark or all third-party raw products in Git.

## Immediate checks

1. Metric behavior: `python -m unittest discover -s tests -v`.
2. Manuscript-table consistency: `python evaluation/validate_reference.py`.
3. Model definitions: `src/phenobiid/`.
4. Exact paper configuration: `configs/paper_protocol.json`.
5. Real-data interface checks: `data/sample/` and `tests/test_sample_data.py`.

The two standalone conversion utilities cover GDHY longitude normalization and
ERA5-Land aggregation. The retained NDVI/GPP preparation snapshots document the
exact paper transformations but still depend on the full crop-calendar
workspace; use them together with the source archive rather than treating them
as one-command downloaders.

Full retraining and grid-level replay require the registered products, their
crop-active alignment, selected historical experts, state checkpoints, and
crop-specific LightGBM residual heads. Product licenses remain independent.
The source ZIP in `paper/` records appendix-level training details.

The paper's reference state inference used CUDA mixed precision. CPU FP32 or
different batching can produce small numerical differences; report rather than
silently replace those differences.

The bundled sample provides a narrower but executable check: it covers all
four crop-specific signal choices and 10%, 30%, 50%, and 70% cutoffs on fixed
samples from 2006--2008. `examples/replay_real_sample.py` runs the complete
frozen CPU forward pass. It is deliberately excluded from the lightweight CI
job because installing PyTorch, LightGBM, TabM, and their compiled dependencies
would dominate routine validation.
