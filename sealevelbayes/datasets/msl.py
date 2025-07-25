import xarray as xa
import numpy as np

from sealevelbayes.datasets import get_datapath
from sealevelbayes.logs import logger

msl_datapath = get_datapath('savedwork/ERAmerged_1900_2018.nc')

def _load_pressure_data_yearly(ds):
    return ds['msl'].transpose("latitude", "longitude", "time").groupby('time.year').mean('time') # time must be in the last dimension for use with interpolate

def prepare_pressure_data():

    with xa.open_dataset(get_datapath('Reanalyses/ERA20c/ERA20c_1900_1980_sfc.nc')) as ds:
        era20c = _load_pressure_data_yearly(ds).sel(year=slice(1900, 1978))

    with xa.open_dataset(get_datapath('Reanalyses/ERA5/ERA5.nc')) as ds:
        era5 = _load_pressure_data_yearly(ds)

    concat_ = xa.concat((era20c[:, :-1], era5), dim='year')

    logger.info(f"Saving merged data to {msl_datapath}")
    concat_.to_netcdf(msl_datapath)

    return concat_


def open_pressure_data_merged():
    if not msl_datapath.exists():
        prepare_pressure_data()

    return xa.open_dataset(msl_datapath)