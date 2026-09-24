# Evaluation protocol

## Yield metric

For each crop and year, RMSE is first computed across eligible grid cells. The
primary score is the arithmetic mean of annual RMSEs, so each year receives
equal weight regardless of its number of cells. Sample standard deviation over
seeds 42, 45, and 48 summarizes initialization variability.

For candidate error `E` and reference error `E_ref`, the reported reduction is
`100 * (1 - E / E_ref)`. Positive values favor the candidate.

## Vegetation metric

Vegetation RMSE is computed only on withheld active slots with finite product
references. NDVI and GPP are scored separately in physical units. The paper
also reports anomaly correlation as a complementary measure.

## Information boundary

At unobserved fraction `r`, the final `ceil(r * n)` of `n` active slots are
withheld. The model receives the observed vegetation prefix. The main study
also receives complete realized weather, including the hidden suffix, and is a
retrospective conditional evaluation. SEAS5 replacement is separate.

## Prediction CSV contract

`evaluation/evaluate_predictions.py` expects `crop`, `year`, `target`, and
`prediction`. Yield values are in t/ha. Optional `method`, `seed`,
`unobserved_fraction`, and `product` columns define separate scoring groups;
extra spatial identity columns are allowed.
