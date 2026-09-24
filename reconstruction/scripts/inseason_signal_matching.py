from __future__ import annotations
import numpy as np
from crop_signal_screen_data import GPP
from review_revision_data import sha256
def physical_gpp(raw, previous=False):
    values = np.full(raw['source_month'].shape, np.nan, np.float32)
    quality = np.zeros_like(values)
    sources = {}
    for year in np.unique(raw['year']):
        source_year = int(year)-int(previous)
        if source_year < 1982:
            continue
        take = np.flatnonzero(raw['year'] == year)
        mm = np.minimum(raw['source_month'][take], 11)
        rr, cc = raw['row'][take, None], raw['col'][take, None]
        active = raw['relative_valid'][take] > 0
        for field, prefix in (('value','gpp_daily_rate'),('quality','valid_area_fraction')):
            file = GPP/'monthly_0p5'/f'{prefix}_{source_year}.npy'
            sources[str(file)] = sha256(file)
            selected = np.load(file,mmap_mode='r')[mm,rr,cc]
            if field == 'value':
                values[take] = np.where(active,selected,np.nan)
            else:
                quality[take] = np.where(active,selected,0)
    quality = np.where(np.isfinite(values),quality,0).astype(np.float32)
    return values, quality, sources
