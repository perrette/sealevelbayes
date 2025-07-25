## Additional code to laod the data
from pathlib import Path
import tqdm
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xa


from sealevelbayes.config import logger, get_datapath
from sealevelbayes.datasets.cmip6.globalmean import get_path, get_all_models as _get_all_models
import sealevelbayes.datasets.garnerkopp2022 as garnerkopp2022
from sealevelbayes.preproc.fingerprints import FINGERPRINTSDIR
from sealevelbayes.preproc.linalg import lsar1statsmodels as lsar1, _prepare_lstsq


def get_all_models(variable, experiments=None, database=None):
    if database == "ar6":
        if experiments is None or type(experiments) is str:
            return garnerkopp2022.get_models(variable, experiments)
        else:
            return list(sorted(set.intersection(*(garnerkopp2022.get_models(variable, x) for x in experiments))))
    else:
        return _get_all_models(variable, experiments, database)


def _load_globalmean(model, experiment, variable, database=None):
    "load yearly variable"
    f = get_path('globalmean', model, experiment, variable, database=database)
    with nc.Dataset(f) as ds:
        data = ds[variable][:]
        time = nc.num2date(ds['time'][:], ds['time'].units, ds['time'].calendar)
    years = np.array([t.year for t in time.astype('datetime64[s]').tolist()])
    return years, data.squeeze()


def load_cmip6(model, experiment, variable, database=None):

    if database == "ar6":
        return garnerkopp2022.load_cmip6(model, experiment, variable)

    years1, data1 = _load_globalmean(model, experiment, variable, database=database)
    years0, data0 = _load_globalmean(model, "historical", variable, database=database)
    
    if years0[-1] > 2014:
        print("!!",model,"historical run ends in", years0[-1],"=> truncate to 2014")
        idx = years0 <= 2014
        years0 = years0[idx] 
        data0 = data0[idx] 
    
    years = np.concatenate([years0, years1])
    data = np.concatenate([data0, data1])
    
    assert (np.diff(years) == 1).all()
    
    return years, data


def load_cmip6_ensemble(variable, experiment, models=None, database=None):
    if models is None:
        models = get_all_models(variable, experiments=[experiment, "historical"], database=database)
        
    ensemble = {}
    for model in models:
        try:
            years, data = load_cmip6(model, experiment, variable, database=database)
        except FileNotFoundError as error:
            print(error)
            print("Failed to load", model, experiment)
            continue
            
        ensemble[model] = pd.Series(data, index=years)
        
    return pd.DataFrame(ensemble)


def open_zos_dataset(model, experiment, database=None):
    if database == "ar6":
        return garnerkopp2022._open_cmip6_annual(model, experiment, 'zos')
    else:
        return xa.open_dataset(get_path("dedrifted_1850-2100", model, experiment, 'zos', database=database))


def _get_fingerprint_data(model, x, driver="tas", driver_database="cdc", zos_database="ar6"):

    # TODO add option not to append historical timeseries
    if x == "historical":
        years, values = load_cmip6(model, "ssp585", driver, database=driver_database)
        idx = (years < 2015) & (years >= 1900)
        years = years[idx]
        values = values[idx]
    else:
        years, values = load_cmip6(model, x, driver, database=driver_database)
        idx = (years >= 2015) & (years <= 2100)
        years = years[idx]
        values = values[idx]

    with open_zos_dataset(model, x, database=zos_database) as ds:
        zos_years = np.array([date.year for date in ds.time.values.astype("datetime64[Y]").tolist()])
        zos = ds['zos'].values
        zos[np.abs(zos)>1e5] = np.nan
        lat = ds.lat.values
        lon = ds.lon.values

    if np.any((zos_years < 1900) | (zos_years > 2100)):
        idx = (zos_years >= 1900) & (zos_years <= 2100)
        zos_years = zos_years[idx]
        zos = zos[idx]

    if (model, x, zos_database) == ("GISS-E2-1-G", 'historical', 'ar6'):
        n = zos_years.size
        zos_years = zos_years[:n//2]
        zos = zos[:n//2]
        assert np.all(years == zos_years), (years, zos_years)


    assert np.all(years == zos_years), (years, zos_years)

    return years, lat, lon, values, zos




def calc_zos_fingerprint(model, experiments=None, driver="tas", **kwargs):

    all_a = []
    all_b = []

    if experiments is None:
        experiments = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]

    actual_experiments = []
    all_drivers = {}

    for x in experiments:
        try:
            years, lat, lon, values, zos = _get_fingerprint_data(model, x, driver=driver, **kwargs)
        except (FileNotFoundError, ValueError):
            continue

        assert values.size == zos.shape[0]

        a, b = _prepare_lstsq(zos, values)

        all_a.append(a)
        all_b.append(b)
        actual_experiments.append(x)
        all_drivers[f"forcing_{driver}_{x}"] = xa.DataArray(values, coords={"years":years}, attrs={"experiment": x})

    if len(all_a) == 0:
        raise ValueError(f"{model}: no valid experiment found to calculate fingerprints (tried: {', '.join(experiments)})")

    elif len(all_a) == 1 and actual_experiments[0] == 'historical' and len(experiments) > 1:
        logger.warning(f"{model}: only historical could be used for the fingerprints")

    elif "ssp585" not in actual_experiments:
        logger.warning(f"{model}: ssp585 is not part of the fingerprints. Got: {', '.join(actual_experiments)})")

    experiments = actual_experiments


    a = np.concatenate(all_a, axis=0)
    b = np.concatenate(all_b, axis=0)


    # Prepare reshaped fingerprints and error
    n = 2   # slope and intercept

    finger = np.empty((n, lat.size*lon.size), dtype=float)
    finger.fill(np.nan)
    error = np.empty((n, lat.size*lon.size), dtype=float)
    error.fill(np.nan)
    # cov = np.empty((n, n, lat.size*lon.size), dtype=float)
    # cov.fill(np.nan)
    rmse = np.empty(lat.size*lon.size, dtype=float)
    rmse.fill(np.nan)

    # Loop over all grid points...
    valid = np.isfinite(b[0])

    for i in tqdm.tqdm(np.where(valid)[0]):
        B = b[:, i]
        x, stdx, rmse_, cov_ = lsar1(a, B)
        finger[:, i] = x
        error[:, i] = stdx
        # cov[..., i] = cov_
        rmse[i] = rmse_

    finger = finger.reshape(n, zos.shape[1], zos.shape[2])
    error = error.reshape(n, zos.shape[1], zos.shape[2])
    # cov = cov.reshape(n, n, zos.shape[1], zos.shape[2])
    rmse = rmse.reshape(zos.shape[1], zos.shape[2])

    finger_coef = xa.DataArray(finger[0], coords={"lat":lat, "lon":lon}, dims=["lat", "lon"])

    results = xa.Dataset({
        "finger" : finger_coef,
        "finger_error" : xa.DataArray(error[0], coords=finger_coef.coords),
        "offset" : xa.DataArray(finger[1], coords=finger_coef.coords),
        "offset_error" : xa.DataArray(error[1], coords=finger_coef.coords),
        "rmse": xa.DataArray(rmse, coords=finger_coef.coords),
        **all_drivers,
        })

    results.attrs.update({
        "model": model,
        "experiments": experiments,
        "driver": driver,
        **kwargs,
        })

    return results



def load_zos_fingerprints(models=None, driver="tas", force=False, save=True, **kwargs):

    fname = get_datapath(f"fingerprints_zos_{driver}_ar1.nc")

    if not force and Path(fname).exists():
        logger.info(f"Load {fname}...")
        return xa.open_dataset(fname)

    dataset = {}

    if models is None:
        models = list(sorted(set(get_all_models('zos', 'ssp585', database='ar6')).intersection(
            get_all_models(driver, 'ssp585', database=kwargs.get('driver_database')))))

    for model in models:
        try:
            res = calc_zos_fingerprint(model, driver=driver, **kwargs)
            # rmse = get_fingerprint_rmse(res)
            # err = get_fingerprint_error(res)
            dataset[model] = res['finger']
            dataset[model].attrs.update(res.attrs)
            dataset[model+"_RMSE"] = res['rmse']
            dataset[model+"_ERROR"] = res['finger_error']
            for k in res:
                if k.startswith("forcing"):
                    dataset[model+f"_{k.upper()}"] = res[k]
        except Exception as error:
            logger.warning(error)
            logger.warning(f'skip {model}')
            continue

    dataset = xa.Dataset(dataset)

    if save:
        logger.info(f"Save to {fname}...")
        dataset.to_netcdf(fname, encoding={v:{'zlib':True} for v in dataset})

    return dataset


# def load_zos_fingerprint(model, experiments, driver="tas", driver_database="cdc", zos_database="ar6"):

#     fname = f"{FINGERPRINT_DATA}/finger_{model}_{'-'.join(experiments)}_{driver}-{driver_database}_zos-{zos_database}.nc"

#     if os.path.exists(fname):
#         return xa.load_dataset(fname)

#     results = calc_zos_fingerprint(model, experiments, driver, driver_database=driver_database, zos_database=zos_database)
#     os.makedirs(FINGERPRINT_DATA, exists_ok=True)
#     results.write(fname)

#     return results

