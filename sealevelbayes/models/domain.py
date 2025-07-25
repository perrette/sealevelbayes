"""This module deals with defining the stations used in runslr.py

It is also used in other modules (e.g. postproc)
"""
import numpy as np
import pandas as pd
import xarray as xa
from itertools import chain

from sealevelbayes.logs import logger
from sealevelbayes.datasets.tidegaugeobs import get_station_ids, psmsl_filelist_all
from sealevelbayes.datasets.satellite import FILENAME as satellite_file
from sealevelbayes.datasets.maptools import interpolate
from sealevelbayes.datasets.oceancmip6 import list_files
from sealevelbayes.runparams import coords_to_ids, id_to_coord

def ids_to_coords(ids):
    return np.array([id_to_coord(id) for id in ids])


def get_stations_from_ids(ids, psmsl_flags={}, coord_flags={}):
    stations = []
    for id in ids:
        if id < 10000:
            stations.append( get_stations_from_psmsl_ids([id], psmsl_flags)[0] )
        else:
            coord = id_to_coord(id)
            stations.append( get_stations_from_coords([coord], coord_flags)[0] )
    return stations


def get_stations_from_psmsl_ids(psmsl_ids, flags={}):

    df = psmsl_filelist_all.set_index('ID').loc[psmsl_ids].reset_index()

    return [{
        "Longitude": r['longitude'],
        "Latitude": r['latitude'],
        'PSMSL IDs': str(r['ID']),
        'Station names': r['station name'], # back compatibility
        'ID': r['ID'], # simplified
        **flags,  # flag
            } for r in df.to_dict('records')]


def get_stations_from_coords(coords, flags={}, fields={}):
    ids = coords_to_ids(coords)
    return [{"ID": id, "Latitude": c[1], "Longitude": c[0], **{k:v[i] for k, v in fields.items()}, **flags} for i, (c, id) in enumerate(zip(coords, ids))]


def get_stations_from_grid(grid, flags={}, gcm_mask=True, satellite_mask=True, bbox=None):

    if grid in ('0.5', '1', '2', '5', '10', '20', '30', '45', '60', '90', '180'):
        step = float(grid)
    elif grid is None:
        return []
    else:
        raise NotImplementedError(grid)
    lat = np.arange(-90+step/2, 90, step)
    lon = np.arange(step/2, 360, step)
    if bbox is not None:
        l, r, b, t = bbox
        lat = lat[(lat >= b) & (lat <= t)]
        lon = lon[(lon >= l) & (lon <= r)]
    lon2, lat2 = np.meshgrid(lon, lat)

    # now pick an ocean mask... use intersection between satellite data and GCM mask (we are only interested in open-ocean measurements)
    domain = np.ones(lon2.shape, dtype=bool)

    # ...satellite
    if satellite_mask:
        with xa.open_dataset(satellite_file) as ds:
            a = ds.isel(time=0)['sla'].load()
            sat_values = interpolate(a.longitude.values, a.latitude.values, a.values, lon, lat)

        sat_domain = np.isfinite(sat_values)
        domain &= sat_domain

    # ...gcms
    if gcm_mask:
        gcm_files = list_files()
        with xa.open_dataset(gcm_files[0], decode_times=False) as ds:
            lon_gcm = ds.lon.values
            lat_gcm = ds.lat.values
        gcm_dummy = np.sum([xa.open_dataset(f, decode_times=False)['zos'].isel(time=64).values for f in gcm_files], axis=0)
        gcm_domain = np.isfinite(interpolate(lon_gcm, lat_gcm, gcm_dummy, lon, lat))
        domain &= gcm_domain

    coords = np.array((lon2[domain], lat2[domain])).T

    logger.info(f'Grid defined: {grid}x{grid} degrees with {coords.shape[0]} ocean points')

    return get_stations_from_coords(coords, flags)


def _get_non_psmsl_gps_stations(dataset, gps_psmsl_stations=None, flags={}):
    from sealevelbayes.datasets.hammond2021 import load_tidegauge_rates, read_midas
    midas_df = read_midas()

    if dataset == "hammond2021-neighboring":
        psmsl_gps = load_tidegauge_rates()
        gps_station_names = sorted(set(gps['name'] for psmsl_station in psmsl_gps for gps in psmsl_station['gps']))
        missing_gps = set(gps_station_names).difference(set(midas_df['ID']))
        logger.warning(f"{len(missing_gps)} missing neighboring GPS stations ({len(gps_station_names)} searched in total): {missing_gps}")
        gps_station_names = [name for name in gps_station_names if name not in missing_gps]

    elif dataset == "hammond2021-all":
        gps_station_names = slice(None)
    else:
        raise ValueError(dataset)

    selected_gps = midas_df.set_index('ID').loc[gps_station_names]
    return get_stations_from_coords(selected_gps[['lon', 'lat']].values, flags=flags, fields={'gps_rate': selected_gps['vz'].values*1000, 'gps_rate_err': selected_gps['vz_err'].values*1000})

def get_non_psmsl_gps_stations(datasets, gps_psmsl_stations=None, flags={}):
    if datasets is None:
        return []
    return list(chain(*[_get_non_psmsl_gps_stations(dataset, gps_psmsl_stations, flags) for dataset in datasets]))


def merge_stations(stations):
    # a few steps are required to make this possible
    # Unifying ID based on coordinates
    # Each locations may be a number of flags
    # (e.g. PSMSL IDs, and more)
    # Before all: each obs constraint must be loaded not on the "station" dimension, but on its own dimension.
    # And adapt the code in "likelihood.py".
    # And post-processing.
    raise NotImplementedError()

