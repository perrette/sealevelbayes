#!/usr/bin/env python
import concurrent.futures
from pathlib import Path
import sys
sys.path.append('/p/projects/isipedia/perrette/sealevel/slr-matthias/notebooks')
from cmip6.regrid import cdo
from garnerkopp2022 import get_path
import garnerkopp2022, cmip6.api

def job(model, experiment, variable):
    path = get_path(model, experiment, variable)
    annual_path = str(path) + ".annual"
    if not Path(path).exists():
        return
    if not Path(annual_path).exists():
        cdo(f"yearmean {path} {annual_path}")

models = list(sorted(set(garnerkopp2022.get_models('zos', 'ssp585')).intersection(cmip6.api.get_all_models('tas', 'ssp585'))))
#models = ["CanESM5"]
    
with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
    futures = { executor.submit(job, m, x, v) : (m, x, v)
        for v in ['zos', 'zostoga','tas'] for x in ['ssp585', 'historical', 'ssp126', 'ssp245', 'ssp370','ssp119'] for m in models}

    for future in concurrent.futures.as_completed(futures):
        key = futures[future]
        try:
            future.result()
        except Exception as exc:
            print(exc)
            print("FAILED:", key)
