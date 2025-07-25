"""Data from Hermanns et al (2021) https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GL092064
"""
from pathlib import Path
import netCDF4 as nc
import numpy as np
import xarray as xa
import pandas as pd
from sealevelbayes.datasets.manager import register_dataset, get_datapath
from sealevelbayes.datasets.catalogue import require_hermans2021

_root = get_datapath("hermans2021")
root = _root / "Data"

def open_tas(full=False):
    if full:
        return xa.open_dataset(root/'gte_gsat/tas_full_CMIP6_n31_1986_2005ref_awm_1850_2100_am.nc')
    else:
        return xa.open_dataset(root/'gte_gsat/tas_CMIP6_n20_1986_2005ref_awm_1850_2100_am.nc')

def open_zostoga():
    return xa.open_dataset(root/'gte_gsat/zostoga_CMIP6_n20_1986_2005ref_qdedr_1850_2100_am.nc')

def load_tcr():
    return pd.read_csv(root/'ecs_tcr/tcr_cmip6.txt', sep='\t')

def load_ecs():
    return pd.read_csv(root/'ecs_tcr/ecs_cmip6.txt', sep='\t')


sources = "antdyn", "antnet", "antsmb", "expansion", "glacier", "GMSLR", "greendyn", "greennet", "greensmb", "landwater", "sheetdyn"
scenarios = "ssp126", "ssp245", "ssp585"
levels = "mid", "lower", "upper"

def load_gmsl(scenario, source='GMSLR', level=None, levermann14=False):
    """
    level: optional, mid, lower or upper. If not provided, the full ensemble is returned
    """

    if source not in sources:
        raise ValueError(f'source must be one of {", ".join(sources)}')
    if scenario not in scenarios:
        raise ValueError(f'scenario must be one of {", ".join(scenarios)}')
    if level and level not in levels:
        raise ValueError(f'level must be one of {", ".join(levels)}')

    if levermann14:
        return xa.load_dataset(root/f'cmip6_projections_levermann14/{scenario}_{source}_{level}.nc')[source]

    elif level is None:
        return xa.open_dataset(root/f'cmip6_mc_projections_im/{scenario}_{source}.nc')[source]

    else:
        return xa.load_dataset(root/f'cmip6_mc_projections/{scenario}_{source}_{level}.nc')[source]
