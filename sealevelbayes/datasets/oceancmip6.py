# import netCDF4 as nc
import re
import glob
import pickle
import numpy as np
import scipy.signal
import xarray as xa
from pathlib import Path
import tqdm
import concurrent.futures

from sealevelbayes.logs import logger
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.maptools import MaskedGrid, interpolate
from sealevelbayes.datasets.tidegaugeobs import tg, tg_years, psmsl_filelist_rlr as psmsl_filelist
from sealevelbayes.datasets.garnerkopp2022 import get_path as get_garner_path

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


### HERE NEWER CODE WITHOUT MaskedGrid
def load_all_cmip6(models=None, max_workers=None, **kwargs):
    if models is None:
        models = [parse_pimodel(f) for f in list_files()]
    assert len(models) > 0

    if max_workers is None:
        # max_workers = len(models)
        max_workers = max(4, len(models))

    if max_workers > 1:
        logger.info(f"Load {len(models)} with {max_workers} processes...")
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        queues = [pool.submit(load_cmip6, model, **kwargs) for model in models]
        results = [q.result() for q in queues]
        logger.info(f"done.")
    else:
        results = [load_cmip6(model, **kwargs) for model in tqdm.tqdm(models)]

    # do not concatenate here because that may be homogeneous
    assert len(results) == len(models)
    return results


def _sel(ds, x, dim):
    """ Help loading a lon/lat box for a CMIP6 model. The lon wrapping is the reason why we don't use xarray sel directly.
    """
    if x is None:
        return ds

    if type(x) is slice:
        return ds.sel({dim:x})

    x = np.asarray(x)

    mi, ma = x.min(), x.max()

    if dim != 'lon' or (mi > ds[dim].values.min() and ma < ds[dim].values.max()):
        step = np.abs(np.diff(ds[dim].values)).max()
        return ds.sel({dim:slice(mi-step, ma+step)})

    else:
        # lon dimension might need to wrap around 0 line, load it all and let interpolate deal with it
        return ds


def load_cmip6(model, lon, lat, time=None, sel=None, isel=None, experiment='piControl', version='regridded', method_="custom", **kwargs):

    file = get_filename(model, experiment=experiment, version=version)
    if experiment == 'piControl':
        kwargs['decode_times'] = False

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    with xa.open_dataset(file, **kwargs) as ds:
        zos = ds['zos']

        if file.name == "zos_kiost_esm_piControl.nc":
            logger.warning(f"{file.name}: load from time-step 63")
            zos = zos.isel(time=slice(63, None))

        if lon is not None: zos = _sel(zos, lon, 'lon')
        if lat is not None: zos = _sel(zos, lat, 'lat')
        if time is not None: zos = _sel(zos, time, 'time')
        if sel is not None: zos = zos.sel(sel)
        if isel is not None: zos = zos.isel(isel)

        if method_ == 'MaskedGrid':
            if lon.size > 1:
                zos.load()
            assert lon.size == lat.size, 'MaskedGrid method only supported for lon / lat provided as 1-d arrays of coordinates'
            grid = MaskedGrid(zos.lon.values, zos.lat.values, mask=np.isfinite(zos[0].values))
            indices = np.array([grid.nearest_indices(lo, la, context=file) for lo, la in zip(lon, lat)])
            values = np.array([zos[:, i, j].values for i, j in indices])
            return xa.DataArray(values.T,
                dims=('time', 'location'),
                coords={'time': zos.time, 'location': np.arange(lon.size), 'lon': ('location', lon), 'lat': ('location', lat), 'model': model},
                attrs={'file': str(file)})

        elif method_ == 'xarray':
            zos.load()
            # zos = zos.interpolate_na(dim=["lat", "lon"], limit=5, method="nearest")
            # # zos = zos.interpolate_na(dim=["lon"], limit=5, method="nearest")
            # # zos = zos.interpolate_na(dim=["lat"], limit=5, method="nearest")
            res = zos.interp({"lat": lat, "lon": lon}, method="linear")
            res.attrs['model'] = model
            res.attrs['file'] = file
            return res

        zos.load()

        if zos.ndim != 3:
            raise NotImplementedError('only implemented to load time x lat x lon at present -- to load one time slice, you may use a size-one time dimension and squeeze afterward')

    values = interpolate(zos.lon.values, zos.lat.values, zos.values.transpose(1, 2, 0), lon, lat, mask=np.isnan(zos[0].values) | (np.abs(zos[0].values) > 1e10), context=file)

    # typical case: lon.size == lat.size
    if values.ndim == 2:
        return xa.DataArray(values.T,
            dims=('time', 'location'),
            coords={'time': zos.time, 'location': np.arange(lon.size), 'lon': ('location', lon), 'lat': ('location', lat), 'model': model},
            attrs={'file': str(file)})

    # less typical case: lon, lat describe a regular grid
    elif values.ndim == 3:
        return xa.DataArray(values.transpose(2, 0, 1),
            dims=('time', 'lat', 'lon'),
            coords={'time': zos.time, 'lon': lon, 'lat': lat, 'model': model},
            attrs={'file': str(file)})

    else:
        raise NotImplementedError(values.shape)


###

# -----------------------------------------
# Older code below
# -----------------------------------------

# Prepare pi-Control runs at tide-gauge locations
def sample_zos_locations_from_netcdf(f, points, ids=None, names=None, raise_error=False, tol=5, k0=None, plot_on_error=False):
    """Nearest valid data point is loaded
    """
    records = []

    with xa.open_dataset(f, decode_times=False) as ds:

        print("Load zos", f.name, f"({f.stat().st_size*1e-6:.0f} Mb) ...", end=" ")
        zos = ds['zos']
        if f.name == "zos_kiost_esm_piControl.nc":
            logger.warning(f"{f.name}: load from time-step 63")
            zos = zos.isel(time=slice(63, None))
        zos = zos.load()
        print("done")

        mask = np.isnan(zos[0].values)
        grid = MaskedGrid(ds.lon.values, ds.lat.values, mask=~mask)

        for k, (lon, lat) in enumerate(points):

            # pass k0={}; ...(...k0=k0) to restart a failed loop
            if k0 is not None:
                if k <= k0.get(f.name, 0):
                    continue
                k0[f.name] = k

            tagid = f"{ids[k]}: " if ids is not None else ""
            nameid = f"{repr(names[k])}," if names is not None else ""
            context = f"{tagid}{nameid} # {f.name} {k}"

            i, j = grid.nearest_indices(lon, lat, tol=None if (ids is not None) else tol,
                                        raise_error=raise_error, plot_on_error=plot_on_error, context=context)

            mlon, mlat = grid.lon[j], grid.lat[i]
            dist = max(np.abs(lon-mlon), np.abs(lat-mlat))

            records.append({
                "file": f,
                "model": parse_pimodel(f.name),
                "coordinates": (lon, lat),
                "grid": (mlon, mlat),
                "indices": (i, j),
                "distance": dist,
                "values": zos[:, i, j].values,
                "id": ids[k] if ids is not None else None,  # PSMSL ID?
            })

    return records


def sample_zos_locations_from_multiple_files(files, points, **kw):
    records = []
    for i, f in enumerate(files):
        print(f"{i+1}/{len(files)}", end=" ")
        records.extend(sample_zos_locations_from_netcdf(f, points, **kw))
    return records


def list_files(experiment='piControl', version='regridded'):
    # folder = get_datapath(f"cmip6/zos/{version}/{experiment}")
    files = sorted(glob.glob(str(get_filename("*", experiment, version))))
    if len(files) == 0:
        raise ValueError(f"No CMIP6 piControl files found.")
    return files

def get_filename(model, experiment='piControl', version='regridded'):
    if experiment == "piControl":
        return get_datapath(f"cmip6/zos/{version}/{experiment}/zos_{model.replace('-','_').replace(' ','_').lower()}_{experiment}.nc")
    else:
        return get_garner_path(model, experiment, "zos")



RE_PIMODEL = re.compile(r'.*zos_(\w+)_piControl.nc')

def parse_pimodel(file):
    match = RE_PIMODEL.match(str(file))
    if not match:
        return ""
    else:
        return match.groups()[0]
