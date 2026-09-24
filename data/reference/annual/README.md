# Annual score records

These CSVs are copied from saved experiment summaries without changing values.
`provenance.json` records their original project-relative location and SHA-256.

- `paired_10_30_50.csv`: three-seed, thirteen-year paired comparison of PhenoBIID,
  Random Forest and LightGBM at 10/30/50% unobserved fractions.
- `paired_70.csv`: corresponding 70% comparison, including refit-cutoff metadata.
- `direct_seed42.csv`: seed-42 direct seasonal predictors at 10/30/50%, plus
  the retained historical reference and PhenoBIID. It is not a three-seed table.
- `history_and_extended_seed42.csv`: original thirteen-year historical-baseline
  and extended-recipe comparisons, plus the original `world_model` row.
  Extended recipe variants retain their original names and selection settings;
  they must not be relabeled as new independently tuned standard baselines.

The first two files reconstruct the main table through
`evaluation/rebuild_main_table.py`. The 10% slice of the third file reconstructs
`direct_baselines_seed42.csv` through `evaluation/validate_reference.py`.
Annual RMSE records do not contain grid-cell predictions and cannot establish
sample identity on their own. The full evaluators check sample identities;
the generic prediction-level metric API also requires paired years/targets
and, when supplied, spatial identifiers.
