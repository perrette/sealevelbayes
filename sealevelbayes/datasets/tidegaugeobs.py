"""Notebook to assess sampling error of tide-gauge records, as far as trends is concerned.

Use CMIP6 models to estimate the background, low-frequency "noise" in tide-gauge trends (instead of current AR1 noise esimate that represents higher frequency variability within the time period of the tide-gauge record). More specifically, at each tide gauge location, a linear trend would be calculated within a time-window with the same length and same missing values as the actual tide-gauge record. The window can then shifted by one year, so that the trend can be calculated n-t+1 times, where n is the length of the pre-industrial control run, and t is the tide-gauge length. If the pre-industrial control run is detrended, the resulting mean will be zero, and the standard deviation would be indicative of the kind of error we make by comparing our fingerprint-derived, flat trend, with actual tide-gauge records with low-freq natural variability. The operation is of course repeated for each GCM, i.e. 56 times, to get a robust estimate for that standard deviation. The technique also provides covariance between various tide-gauge locations, which reflects the patterns in which water moves around in the world oceans. That can be used later on when we come to estimate various tide-gauges simultaneously.
"""
import json
import tqdm
import numpy as np
import xarray as xa
# import glob

import pandas as pd

from sealevelbayes.logs import logger
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.maptools import MaskedGrid
import sealevelbayes.datasets.frederikse2020 as frederikse2020
from sealevelbayes.datasets.psmsl import is_rlr, load_filelist_rlr, load_filelist_all, load_all_psmsl_rlr
from sealevelbayes.datasets.basins import ThompsonBasins
from sealevelbayes.preproc.tides import nodal_tide_scaled
from sealevelbayes.preproc.linalg import detrend_timeseries
from sealevelbayes.preproc.inversebarometer import load_ib

# DEFAULT_VERSION = "psmsl_rlr_1900_2018"
# DEFAULT_VERSION = "frederikse"
DEFAULT_VERSION = "psmsl_rlr_1900_2018_subset"

hogarth_stations = np.array(json.load(open(get_datapath("Budget_20c/tg/hogarth_stations.json"))))

# load data in the global scope of this module
region_info = frederikse2020.region_info
# lons = region_info["Longitude"]
# lats = region_info["Latitude"]
# lons = np.where(lons < 180, lons, lons-360) # lon as[-180, 180] ??

psmsl_filelist_rlr = load_filelist_rlr()
psmsl_filelist_all = load_filelist_all()
psmsl_rlr = load_all_psmsl_rlr()



def _make_hogart_like(psmsl_rlr, rlr=True):
    """ write psmsl_rls as a dictionary to make it more alike what we worked with before """
    return {
        "id": psmsl_rlr.columns,
        "height": psmsl_rlr.values.T.astype(float),
        "coords": psmsl_filelist_all.set_index('ID').loc[psmsl_rlr.columns][['latitude', 'longitude']].values,
    }


tg_psmsl_rlr_1900_2018_years = np.arange(1900, 2018+1)
tg_psmsl_rlr_1900_2018 = _make_hogart_like(psmsl_rlr.loc[1900:2018])

tg_psmsl_rlr_1900_2018_subset_years = np.arange(1900, 2018+1)
tg_psmsl_rlr_1900_2018_subset = _make_hogart_like(psmsl_rlr[[id for id in psmsl_rlr.columns if id in set(hogarth_stations)]].loc[1900:2018])

tg_psmsl_rlr_1900_2022_years = np.arange(1900, 2022+1)
tg_psmsl_rlr_1900_2022 = _make_hogart_like(psmsl_rlr.loc[1900:2022])


def load_tidegauges_frederikse():
    tg_years = np.arange(1900, 1900+119)
    tg = np.load(get_datapath("Budget_20c/tg/station_data.npy"), allow_pickle=True).item()
    return tg_years, tg


try:
    tg_frederikse_years, tg_frederikse = load_tidegauges_frederikse()
except FileNotFoundError:
    logger.warning("Could not load tide-gauge records from Frederikse et al. 2020. Use equivalent PSMSL data instead.")
    tg_frederikse_years = tg_psmsl_rlr_1900_2018_subset_years
    tg_frederikse = tg_psmsl_rlr_1900_2018_subset


if DEFAULT_VERSION == "frederikse":
    tg = tg_frederikse
    tg_years = tg_frederikse_years

elif DEFAULT_VERSION == "psmsl_rlr_1900_2018":
    tg = tg_psmsl_rlr_1900_2018
    tg_years = tg_psmsl_rlr_1900_2018_years

elif DEFAULT_VERSION == "psmsl_rlr_1900_2018_subset":
    tg = tg_psmsl_rlr_1900_2018_subset
    tg_years = tg_psmsl_rlr_1900_2018_subset_years

elif DEFAULT_VERSION == "psmsl_rlr_1900_2022":
    tg = tg_psmsl_rlr_1900_2022
    tg_years = tg_psmsl_rlr_1900_2022_years

else:
    raise NotImplementedError(DEFAULT_VERSION)


def _get_matching_indices(indices, psmsl_ids):
    idx = np.searchsorted(indices, psmsl_ids)
    assert idx.max() < indices.size
    indexed_ids = indices[idx]
    check = np.array(psmsl_ids) == indexed_ids
    if not np.all(check):
        mismatch = np.where(~check)[0][:10].tolist()
        raise ValueError(f'{(~check).sum()} indices could not be matched: {repr(mismatch)} ...')
    return idx


def load_tidegauge_records(psmsl_ids, remove_meteo=True, remove_nodal=True, classical_formula_for_tides=False, wind_correction=False, version=DEFAULT_VERSION):

    # Note the following equality:
    np.testing.assert_allclose(tg_frederikse['height'], (tg_frederikse['height_corr'] + tg_frederikse['height_meteo']))
    # I checked for a few stations that tg['height'] + tg['height_nodal'] == PSMSL tide-gauges

    if version == "frederikse":
        tg = tg_frederikse
        idx = _get_matching_indices(hogarth_stations, psmsl_ids)
        # tg_values = (tg['height'] + tg['height_nodal'])[idx]
        tg_values = tg['height'][idx]
        tg_years = tg_frederikse_years

    elif version == "psmsl_rlr_1900_2018_subset":
        tg = tg_psmsl_rlr_1900_2018_subset
        idx = _get_matching_indices(tg['id'], psmsl_ids)
        tg_values = tg['height'][idx]
        tg_years = tg_psmsl_rlr_1900_2018_subset_years

    elif version == "psmsl_rlr_1900_2018":
        tg = tg_psmsl_rlr_1900_2018
        idx = _get_matching_indices(tg['id'], psmsl_ids)
        tg_values = tg['height'][idx]
        tg_years = tg_psmsl_rlr_1900_2018_years

    elif version == "psmsl_rlr_1900_2022":
        tg = tg_psmsl_rlr_1900_2022
        idx = _get_matching_indices(tg['id'], psmsl_ids)
        tg_values = tg['height'][idx]
        tg_years = tg_psmsl_rlr_1900_2022_years

        # df = psmsl_rlr.loc[1900:2022].T.loc[psmsl_ids] # to benefit from nodal and meteo corrections, restrict from 1900 to 2018
        # tg_values = df.values.astype(float)
    else:
        raise NotImplementedError(version)

    min_years = np.isfinite(tg_values).sum(axis=1).min()
    logger.info(f"Load tide-gauge records for {version}. Min number of years: {min_years}")

    if (remove_nodal and not classical_formula_for_tides) or (remove_meteo and wind_correction):
        idx_fred = _get_matching_indices(hogarth_stations, psmsl_ids)

        if remove_nodal and not classical_formula_for_tides:
            logger.info(f"...remove nodal tide from Frederikse et al dataset")
            tg_values -= tg_frederikse['height_nodal'][idx_fred]

        if remove_meteo and wind_correction:
            logger.info(f"...remove meteo from Frederikse et al dataset")
            tg_values -= tg_frederikse['height_meteo'][idx_fred]

    if remove_nodal and classical_formula_for_tides:
        logger.info("use classical formula for tides")
        # This is an approximation but can be useful to extend to all psmsl tide-gauges
        lats = tg['coords'][idx, 0]
        height_nodal = nodal_tide_scaled(tg_years[None, :], lats[:, None])
        tg_values -= height_nodal

    if not remove_nodal:
        logger.warning("do not remove the nodal tide")

    if remove_meteo and not wind_correction:
        logger.info(f"...apply inverse barometer correction")
        lats = tg['coords'][idx, 0]
        lons = tg['coords'][idx, 1]
        ib_years, ib_corr = load_ib(lons, lats)
        ib_corr = ib_corr - ib_corr.mean(axis=1)[:, None]
        tg_values -= ib_corr*1e3

    if not remove_meteo:
        logger.warning("do not remove the meteo")

    min_years2 = np.isfinite(tg_values).sum(axis=1).min()
    if min_years2 != min_years:
        logger.warning(f"...min number of years changed after correction: {min_years2} < {min_years}")

    records = pd.DataFrame(tg_values.T, index=tg_years, columns=psmsl_ids)
    return records



stations = region_info.to_dict('records')
psmsl_map = dict(zip(hogarth_stations, np.arange(hogarth_stations.size)))

def decluster_stations(stations=stations, remove_duplicates=True):
    declustered = [{**station, 'PSMSL IDs': psmsl, 'Station names': name,
             'Longitude': tg_frederikse['coords'][psmsl_map[int(psmsl)], 1],
             'Latitude': tg_frederikse['coords'][psmsl_map[int(psmsl)], 0],
            } for station in stations for (psmsl, name) in zip((p.strip() for p in station['PSMSL IDs'].split(',')), (p.strip() for p in station['Station names'].split(';')))]
    if not remove_duplicates:
        return declustered
    # remove duplicate
    return sorted({r["PSMSL IDs"]:r for r in declustered}.values(), key=lambda r: int(r["PSMSL IDs"]))

_indices = hogarth_stations.tolist()

def _valid_years(r):
    return np.isfinite(tg_frederikse['height'][_indices.index(int(r['PSMSL IDs']))]).sum()


def get_stations_declustered(ny=20, **kw):
    return [r for r in decluster_stations(**kw) if _valid_years(r) >= ny and is_rlr(r['PSMSL IDs'])]


def get_station_ids(version=DEFAULT_VERSION, **kw):
    if version == 'frederikse':
        kw.setdefault('included_in_frederikse', True)
        kw.setdefault('mask_from_extended_dataset', True)
        kw.setdefault('metric', True)
        kw.setdefault('flagged', True)
        kw.setdefault('from_year', 1900)
        kw.setdefault('to_year', 2018)
        return _get_station_ids(**kw)

    elif version == 'psmsl_rlr_1900_2018':
        kw.setdefault('included_in_frederikse', False)
        kw.setdefault('mask_from_extended_dataset', False)
        kw.setdefault('metric', False)
        kw.setdefault('flagged', False)
        kw.setdefault('from_year', 1900)
        kw.setdefault('to_year', 2018)
        return _get_station_ids(**kw)

    elif version == 'psmsl_rlr_1900_2018_subset':
        kw.setdefault('included_in_frederikse', True)
        kw.setdefault('mask_from_extended_dataset', False)
        kw.setdefault('metric', False)
        kw.setdefault('flagged', False)
        kw.setdefault('from_year', 1900)
        kw.setdefault('to_year', 2018)
        return _get_station_ids(**kw)

    elif version == 'psmsl_rlr_1900_2022':
        kw.setdefault('included_in_frederikse', False)
        kw.setdefault('mask_from_extended_dataset', False)
        kw.setdefault('metric', False)
        kw.setdefault('flagged', False)
        kw.setdefault('from_year', 1900)
        kw.setdefault('to_year', 2022)
        return _get_station_ids(**kw)

    else:
        raise NotImplementedError(version)


def _get_station_ids(min_years=None, flagged=False, metric=False, included_in_frederikse=False, from_year=None, to_year=None, mask_from_extended_dataset=False):
    """
    min_years: filter stations that have at least 20 years of data, after year filtering `from_year`, `to_year`
    flagged: include flagged data
    metric: include metric records as well as RLR
    included_in_frederikse: only include stations that are present in the dataset provided by Frederikse (useful for nodal correction)
    from_year, to_year: used together with min_years
    mask_from_extended_dataset: if true, use extended dataset by Hogart as provided by Frederikse, when computing the min number of valid data points


    To have stations used by Frederikse et al, use
        `included_in_frederikse=True, flagged=True, metric=True, from_year=1900, to_year=2018, source="frederikse"`
    """
    if included_in_frederikse:
        psmsl_ids = set(hogarth_stations)
    else:
        psmsl_ids = set(psmsl_filelist_all['ID'])

    # not flagged
    if not flagged:
        psmsl_ids = psmsl_ids.intersection(r['ID'] for r in psmsl_filelist_all.to_dict('records') if r['qcflag'].strip() != 'Y')

    if not metric:
        psmsl_ids = psmsl_ids.intersection(psmsl_rlr.columns)

    # more than 20 years of data
    if min_years:
        if mask_from_extended_dataset:
            df = pd.DataFrame(tg_frederikse['height'].T, index=tg_frederikse_years, columns=hogarth_stations)
        else:
            df = psmsl_rlr

        df = df.T.loc[[id for id in psmsl_ids if id in df.columns]].T

        if from_year or to_year:
            idx = slice(from_year, to_year)
            df = df.loc[idx]

        psmsl_ids = set(df.T[np.isfinite(df.values).sum(axis=0) >= min_years].index)

    return sorted(psmsl_ids)


def coords_to_nearest_fred(coords):
    from maptools import search_coords
    inearest, mindist = search_coords(region_info[["Longitude", "Latitude"]].values, coords)
    return region_info.iloc[inearest]


def psmsl_ids_to_basin(psmsl_ids, basins_cls=None):
    """Return basin for any PSMSL ID
    """
    if basins_cls is None:
        basins_cls = ThompsonBasins.load().split_atlantic()

    coords = psmsl_filelist_all.set_index('ID').loc[psmsl_ids][['longitude', 'latitude']].values
    return [basins_cls.get_region_label(lon, lat) for lon, lat in coords]

    # regions = basins_cls.interp_nearest(coords)
    # return [basins_cls.map.get(r) for r in regions]


def _get_model_pi_records():
    """scan directory for CMIP6 records with piControl
    """
    root = get_datapath("cmip6/zos/regridded")
    files = list(root.glob("*piControl.nc"))

    for p in files:
        with xa.open_dataset(p) as ds:
            years = np.array([t.year for t in ds.time.values.astype('datetime64[s]').tolist()])
            breaks = np.sum(np.diff(years) != 1)
    #         if not np.all(np.diff(years) == 1):
            logger.info(f"{'!!' if breaks > 0 else '  '} {p.name} {years.size}")
            yield {
                "name" : p.name,
                "breaks": breaks,
                "size": years.size,
            }


def get_model_data_at_stations(stations, min_years=250):
    dataset = []
    root = get_datapath("cmip6/zos/regridded")

    for record in tqdm.tqdm(_get_model_pi_records()):
        if record['breaks'] > 0 or record['size'] < min_years:
            continue
        with xa.open_dataset(root/record['name']) as ds:

            grid = MaskedGrid(ds['lon'][:], ds['lat'][:], mask=np.isfinite(ds['zos'][0].values))

            for k, station in enumerate(stations):
                i, j = grid.nearest_indices(station['Longitude'], station['Latitude'], tol=10)
                timeseries = ds['zos'].values[:, i, j]*1000  # results in mm

                dataset.append({
                    "name" : record['name'],
                    "station_id": region_info.iloc[k].name,  # legacy
                    "PSMSL IDs": region_info.iloc[k]['PSMSL IDs'],
                    "Station names": region_info.iloc[k]['Station names'],
                    "timeseries": timeseries,
                    "detrended": detrend_timeseries(timeseries),
                })

    return dataset

region_info.index.name = 'station_id' # an ID we made up, with 1000s for ocean basins  // legacy !
stations_by_name = region_info.reset_index().set_index('Station names')

psmsl_filelist_all_by_ID = psmsl_filelist_all.set_index('ID')

def get_station_name(ID):
    """Get station name from PSMSL ID
    """
    try:
        station = psmsl_filelist_all_by_ID.loc[ID]
    except KeyError:
        raise KeyError(f"PSMSL ID {repr(ID)} not found in filelist_all. Available IDs: {psmsl_filelist_all_by_ID.index.tolist()}")

    return station['station name']