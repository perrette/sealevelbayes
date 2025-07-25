import numpy as np
import xarray as xa

from sealevelbayes.logs import logger
from sealevelbayes.datasets import get_datapath
from sealevelbayes.datasets.msl import open_pressure_data_merged
from sealevelbayes.datasets.maptools import interpolate


ib_datapath = get_datapath('savedwork/ERAmerged_1900_2018_ib.nc')


def prepare_ib_data():

    with xa.open_dataset("sealeveldata/savedwork/ERAmerged_1900_2018_bathymetry.nc") as bathy:
        ocean = bathy['elevation'].load() < 0

    with open_pressure_data_merged(mode='r') as ds:
        rho = 1025 # some reference value for ocean density
        g = 9.81
        ib = ds['msl'].load() / (rho * g)

        ib.values[~ocean.values] = np.nan
        cellw = np.cos(np.deg2rad(ib['latitude'].values))

        m = np.isfinite(ib.values).all(axis=-1)
        ib_mean = ((ib.values * cellw[:, None, None])[m]).sum(axis=0) / (m * cellw[:, None, None]).sum()

        ib_ano = - (ib - ib_mean)
        ib_ano.name = "ib"

    ib_ano.to_netcdf(ib_datapath)
    return ib_ano


def open_ib_dataset(**kw):
    if not ib_datapath.exists():
        prepare_ib_data()
    return xa.open_dataset(ib_datapath, **kw)


def load_ib(lons, lats):
    with open_ib_dataset() as ds:
        ib = ds['ib'].transpose("latitude", "longitude", "year").values
        return ds['year'].values, interpolate(ds.longitude.values, ds.latitude.values, ib, lons, lats, mask=np.isnan(ib).any(axis=-1))

def load_pressure(lons, lats):
    with open_pressure_data_merged() as ds:
        msl = ds['msl'].transpose("latitude", "longitude", "year").values
        return ds['year'].values, interpolate(ds.longitude.values, ds.latitude.values, msl, lons, lats, mask=np.isnan(msl).any(axis=-1))