"""Load data from https://zenodo.org/record/6419954

Reference
---------
Garner, Gregory G., & Kopp, Robert E. (2022). Framework for Assessing Changes To Sea-level (FACTS) modules, scripts, and data for the IPCC AR6 sea level projections (Version 20220406). Zenodo. https://doi.org/10.5281/zenodo.6419954
"""
from pathlib import Path
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xa

# from cmip6.regrid import cdo
from sealevelbayes.config import CONFIG
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.runutils import cdo

module_data = get_datapath("zenodo-6419954-garner_kopp_2022/modules-data")
module_data_zos = get_datapath("zenodo-6419954-garner_kopp_2022/modules-data-zos")

cmip6 = module_data_zos / "modules-data/tlm/oceandynamics/data/cmip6/"

exclude_models = {
    "tas": "MPI-ESM1-2-HR"  # weird value 150 something (instead of Kelvin) and too low warming. Use Hermann instead.
}

def get_models(variable="tas", experiment=None):
    if experiment is None:
        return list(sorted( [f.name for f in (cmip6 / variable).glob('*')] ))
    else:
        return list(sorted( [f.parent.name for f in (cmip6 / variable).glob(f'*/*{experiment}*.nc')] ))


def _yield_ensembles():
    for r in range(10):
        for i in range(10):
            for p in range(10):
                for f in range(10):
                    yield f"r{r}i{i}p{p}f{f}"


def get_path(model, experiment, variable, ensemble=None, ext=""):

    # if (model, experiment, variable) == ("GISS-E2-1-G", "historical", "zos") and ensemble is None:
    #     ensemble = "r1i1p1f2"  # dim0 = 3960 for r1i1p1f1 instead of time

    files = list((cmip6 / variable / model).glob(f'*{experiment}_{ensemble or ""}*.nc'+ext))

    if len(files) > 1:
        assert ensemble is None, 'ensemble is not None but several files were found !'
        for e in _yield_ensembles():
            try:
                path = get_path(model, experiment, variable, e)
                # print(f'INFO: {model}, {experiment}, {variable}: return ensemble {e} out of {len(files)} ensemble members')
                # logging.info(f'INFO: {model}, {experiment}, {variable}: return ensemble {e} out of {len(files)} ensemble members')
                return path
            except:
                continue
        raise ValueError(f'!! Several files found for {model}, {experiment}, {variable}, {ensemble}: {files}')

    elif len(files) == 0:
        raise ValueError(f'!! No file found for {model}, {experiment}, {variable}, {ensemble}')

    return files[0]



# def _annual_mean(x):
#     return x.reshape(x.size//12, 12).mean(axis=1)

def _calc_trend(zostoga0):
    "calculate past trend to fix artificial jumps"
    # mean of 1-year, 2-year, and 3-year trends
    return ((zostoga0[-1] - zostoga0[-2])
             + (zostoga0[-1] - zostoga0[-3])/2
             + (zostoga0[-1] - zostoga0[-4])/3)/3

# def _load_cmip6_annual_timeseries(path):
#     with nc.Dataset(path) as ds:
#         variable = ds.variable_id
#         data = _annual_mean(ds[variable][:]).filled(np.nan)
#         time = _annual_mean(ds['time'][:])
#         time = nc.num2date(time, ds['time'].units, ds['time'].calendar)
#         years = np.array([t.year for t in time])
#     return years, data

# def _load_cmip6_annual_xarray(path):
#     with xa.open_dataset(path) as ds:
#         variable = ds.variable_id
#         yearly = ds[variable].resample({"time":"Y"}).mean()
#         return yearly.time.values, yearly.values


def _open_cmip6_annual(model, experiment, variable, **kwargs):
    path = get_path(model, experiment, variable)
    annual_path = str(path) + ".annual"
    if not Path(annual_path).exists():
        cdo(f"yearmean {path} {annual_path}")
    return xa.open_dataset(annual_path, **kwargs)


def getyears(time):
    return np.array([t.year for t in time.astype('datetime64[s]').tolist()])

def load_cmip6(model, experiment, variable):
    """load CMIP6 future experiments, pre-pended by historical runs and fixed for known issues"""

    ds = _open_cmip6_annual(model, experiment, variable)
    ds0 = _open_cmip6_annual(model, 'historical', variable)

    years, zostoga = getyears(ds[variable].time.values), ds[variable].values.squeeze()
    years0, zostoga0 = getyears(ds0[variable].time.values), ds0[variable].values.squeeze()


    # fix problems from historical to future
    if variable == "zostoga" and model in ['GISS-E2-1-G', 'MRI-ESM2-0']:
        trend = _calc_trend(zostoga0[-4:])
        zostoga += zostoga0[-1] - zostoga[0] + trend

    elif (model, variable) == ('GISS-E2-1-G', 'tas'):
        # historical time-series doubled, for some reason
        zostoga0 = zostoga0[:years0.size//2]
        years0 = years0[:years0.size//2]

    assert years[0] - years0[-1] == 1
    assert (np.diff(years) == 1).all(), (model, experiment, variable)
    assert (np.diff(years0) == 1).all(), (model, experiment, variable)

    years = np.concatenate([years0, years], axis=0)
    zostoga = np.concatenate([zostoga0, zostoga], axis=0)

    # fix problem in 2100
    if variable == "zostoga" and (model, experiment) in [('MRI-ESM2-0', 'ssp585'), ('MRI-ESM2-0', 'ssp126')]:
        i = 2101-1850
        offset = zostoga[i-1] + _calc_trend(zostoga[i-4:i]) - zostoga[i]
        zostoga[i:] += offset

    return years, zostoga


experiments = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']


def load_ensemble(variable, experiment, models=None):

    if models is None:
        models = get_models(variable, experiment)

    data = {}
    for model in models:
        years, array = load_cmip6(model, experiment, variable)
        data[model] = pd.Series(array, index=years)

    return pd.DataFrame(data)
