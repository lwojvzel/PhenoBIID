# CropDynamicsBench data

CropDynamicsBench aligns annual yield, monthly weather, vegetation products,
crop calendars, and data-support indicators on a common 0.5-degree grid. Each
crop-grid-year is packed into at most 12 crop-active slots rather than a fixed
January-to-December sequence.

The paper covers maize, rice, soybean, and wheat from 1982 to 2016. Annual yield
history may begin in 1981. Fixed evaluation years and temporal blocks are in
`configs/paper_protocol.json`.

## Products

| Product | Role | Paper transformation |
|---|---|---|
| GDHY v1.2/v1.3 | annual yield | longitude reordered to -180..180; fill excluded |
| ERA5-Land | 13 weather variables | monthly values aligned to crop-active slots |
| PKU GIMMS NDVI V1.2 | vegetation state | quality filtering, area aggregation, duration-weighted monthly mean |
| GLASS rEC-LUE GPP | productivity state | area aggregation; monthly total divided by calendar days |
| MIRCA-OS | crop calendar/area | nearest registered snapshot supplies active slots and area context |
| GLASS LAI | inherited cohort support | original sample filter and LAI comparison, not the selected main signal |
| ECMWF SEAS5 | weather sensitivity | bias-corrected replacement only in the hidden window |

Exact records and license notes are in `configs/data_sources.json`.

## Expected local layout

```text
data/
  raw/{gdhy,era5_land,pku_gimms_ndvi_v1p2,reclue_monthly_gpp_v1,mirca_os,glass_lai_v40,seas5}/
  processed/{gdhy_lon180,era5_land_monthly_0p5,pku_gimms_ndvi_v1p2,reclue_monthly_gpp_v1,crop_active}/
```

Large upstream and processed arrays are ignored by Git. Obtain each product
from its official record and comply with its terms. Project code terms do not
override MIRCA-OS, GLASS LAI, or Copernicus/ECMWF terms.

`data/reference/` contains manuscript-level aggregate measurements, not raw
Earth-observation pixels or full grid-level labels.

## Bundled real-data sample

`data/sample/` contains a compact sample of the processed interface for all
four crops, three years, and four observation cutoffs. It is selected by a
fixed coordinate-based rule without consulting targets or predictions. Input
arrays and reference arrays are separate. This sample supports contract tests
and frozen inference replay only; it cannot reproduce full-dataset scores.
