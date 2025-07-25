import numpy as np
import xarray as xa
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.maptools import MaskedGrid, _iterate_coords, interpolate

FILENAME = get_datapath("cds/satellite-sea-level-global/satellite_sla_1993_2019.nc")

def open_dataset():
    return xa.open_dataset(FILENAME)

def get_satellite_timeseries(lons, lats, return_coords=False, **kw):
    lons = np.asarray(lons)
    lons = np.where(lons > 180, lons-360, lons)

    with open_dataset() as ds:
        sla = ds['sla'].load().values
        lon1, lat1 = ds['longitude'].values, ds['latitude'].values
        mask = np.isnan(sla).any(axis=0)
        tpa_correction = ds['tpa_correction'].values

        grid = MaskedGrid(lon1, lat1, mask=~mask) # we only consider time-series with full finite values
        # ii = np.searchsorted(ds['latitude'][:], lats)
        # jj = np.searchsorted(ds['longitude'][:], lons)
        years = np.arange(1993, 2019+1)

        # tpa_correction: TOPEX-A instrumental drift correction derived from altimetry and tide gauges global comparisons (WCRP Sea Level Budget Group, 2018)
        # This variable can be added to the gridded SLA to correct for the observed
        # instrumental drift during the lifetime of the TOPEX-A mission (the correction is null after this period).
        # This is a global correction to be added a posteriori (and not before)
        # on the global mean sea level estimate derived from the gridded sea level map.
        # It can be applied at regional or local scale as a best estimate (better than no correction,
        # since the regional variation of the instrumental drift is unknown). See product manual for more details.
        points, results = zip(*[((lon1[j], lat1[i]), sla[:, i, j]+tpa_correction)
            for i,j in _iterate_coords(grid, lons, lats, **kw)])
        if return_coords:
            return years, np.array(results), np.array(points)
        else:
            return years, np.array(results)


def get_satellite_error_map(remove_gia=True):
    """Note the sealeveltrendwithoutgia.nc file was recalculated by us using the rsl.py code from the original dataset (in subrepos/rsl)
    but it can in fact be obtained from the original dataset easily as:

        ds = xa.open_dataset(get_datapath("prandi2021/quality_controlled_data_77853.nc"))
        error = ((ds['trend_ci'].load() / 1.706)**2 - ds['gia_drift']**2)**.5
    """
    filename = "prandi2021/sealeveltrendwithoutgia.nc" if remove_gia else "prandi2021/quality_controlled_data_77853.nc"
    with xa.open_dataset(get_datapath(filename)) as ds:
        # convert 5-95th range to 1-sigma, and remove gia_drift (assumed independent from the rest)
        trend_sd =ds["trend_ci"].values/1.706  # the value comes from how trend_ci is computed in the original dataset
        return ds.longitude.values, ds.latitude.values, trend_sd


def get_satellite_error(lons, lats, **kw):
    lons = np.asarray(lons)
    lons = np.where(lons > 180, lons-360, lons)
    lon, lat, error_map = get_satellite_error_map(**kw)
    domain = np.isfinite(error_map) & (error_map < 1e10)
    return interpolate(lon, lat, error_map.T, lons, lats, mask=~domain.T, wrap_lon=True, context="satellite error")