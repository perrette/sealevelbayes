import itertools
import numpy as np
from pathlib import Path
import tqdm
import xarray as xa
from scipy.interpolate import RegularGridInterpolator

from sealevelbayes.logs import logger
from sealevelbayes.datasets.manager import require_dataset
from sealevelbayes.datasets.naturalearth import add_coast, add_land

# Haversine function to compute geodesic distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Compute pairwise geodesic distances between all points on the sphere (lat, lon)
def compute_geodesic_distances(longitudes, latitudes):
    n = len(latitudes)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j: continue
            distances[j, i] = distances[i, j] = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
    return distances

# def _distance(lons, lats, lon, lat, metric='lonlat'):
def _distance(lons, lats, lon, lat, metric='euclidian'):
    delta_lon = np.abs(np.mod(lons - lon, 360))
    delta_lon = np.where(delta_lon < 180, delta_lon, 360 - delta_lon)
    if metric == 'lonlat':
        coslat = np.cos(np.deg2rad(lat))
        return (delta_lon*coslat)**2 + (lats-lat)**2
    elif metric == 'euclidian':
        return delta_lon**2 + (lats-lat)**2
    elif metric == 'fast':
        return np.abs(delta_lon) + np.abs(lats-lat)
    elif metric == 'max':
        return np.max([np.abs(delta_lon), np.abs(lats-lat)], axis=0)
    else:
        raise NotImplementedError(metric)


def shift_lon(lon, lon0=0):
    lon = np.where(lon < lon0, lon + 360, lon)
    return np.where(lon > lon0 + 360, lon - 360, lon)



def search_coords(coords_array, coords, **kw):
    """ Return nearest neighbour matches of coords array from coords_array
    """
    lons, lats = coords_array.T
    lon, lat = np.array(coords).T

    # 1
    lons = shift_lon(lons, 0)
    lon = shift_lon(lon, 0)
    dist1 = _distance(lons[None], lats[None], lon[:, None], lat[:, None], **kw)

    # 2
    lons = shift_lon(lons, -180)
    lon = shift_lon(lon, -180)
    dist2 = _distance(lons[None], lats[None], lon[:, None], lat[:, None], **kw)

    dist = np.min([dist1, dist2], axis=0)
    inearest = dist.argmin(axis=1)
    mindist = dist[:, inearest]

    return inearest, mindist


class MaskedGrid:

    def __init__(self, lon, lat, mask):
        self.lon = lon
        self.lat = lat
        self.mask = mask

        self.lon0 = -180 if np.any(lon < 0) else 0

        if self.lon.ndim == 2:
            self.lon2, self.lat2 = self.lon, self.lat
        else:
            self.lon2, self.lat2 = np.meshgrid(lon, lat)

        i2, j2 = np.indices((lat.size, lon.size))

        if mask is None:
            self.lons, self.lats = self.lon2.flatten(), self.lat2.flatten()
            self.jjs, self.iis = j2.flatten(), i2.flatten()
        else:
            self.lons = self.lon2[mask]
            self.lats = self.lat2[mask]
            self.iis = i2[mask]
            self.jjs = j2[mask]


    def distance(self, lon, lat, metric='lonlat'):
        return _distance(self.lons, self.lats, lon, lat, metric=metric)

    def _nearest_index(self, lon, lat, tol=None, metric='lonlat', **kw):
        if lon < 0 and self.lon0 == 0:
            lon = lon + 360
        elif lon > 180 and self.lon0 == -180:
            lon = lon - 360
        dist = self.distance(lon, lat, metric=metric)
        k = np.argmin(dist)

        if tol is not None:
            mlon, mlat = self.lons[k], self.lats[k]
            self.check((lon, lat), (mlon, mlat), tol, **kw)

        return k, dist[k]


    def nearest_indices(self, lon, lat, tol=None, **kw):
        k, dist = self._nearest_index(lon, lat, tol=tol, **kw)
        return self.iis[k], self.jjs[k]


    def nearest_indices2(self, lon, lat, tol=None, **kw):
        k, dist = self._nearest_index(lon, lat, tol=tol, **kw)
        return self.iis[k], self.jjs[k], dist


    def check(self, expected_coords, got_coords, tol, raise_error=True, context=None, plot_on_error=False):
        mlon, mlat = got_coords
        lon, lat = expected_coords
        scoremax = _distance(mlon, mlat, lon, lat, metric='max')
        if scoremax > tol:
            msg = f"{scoremax:.1f} > {tol} degree(s) of {lon}, {lat} (model coords: {mlon}, {mlat})"
            if context:
                msg = f"{context} :: {msg}"
            if plot_on_error:
                import matplotlib.pyplot as plt
                # print(f"""{tg['id'][k]}: {repr(tg['name'][k])},  # {f.name[len("zos_"):-len("_piControl.nc")]}""")
                plt.figure()
                plt.pcolormesh(self.lon, self.lat, self.mask)
                # plt.colorbar()
                plt.scatter(lon, lat, label=f"Expected ({lon:.2f}, {lat:.2f})")
                plt.scatter(mlon, mlat, c='r', label=f'Got ({mlon:.2f}, {mlat:.2f})')
                plt.legend()
            if raise_error:
                raise ValueError(msg)
            else:
                logger.warning(msg)


def _iterate_coords(grid, lons, lats, tol=3, context="", **kw):
    for k, (lon, lat) in enumerate(zip(lons, lats)):
        try:
            i, j = grid.nearest_indices(lon, lat, tol=tol, **kw) # no warning if less than 3 degrees away
        except ValueError as error:
            i, j = grid.nearest_indices(lon, lat, tol=50, **kw)
            logger.debug(f"{context}:: {k}: ({lon:.2f}, {lat:.2f}) => ({grid.lon[j]}, {grid.lat[i]})")
        yield i, j


def interpolate(lon, lat, values, lon2, lat2, mask=None, wrap_lon=True, context=""):
    """use linear RegularGridInterpolator for the bulk of values, and fill in mask values with MaskedGrid

    NOTE: here we use mask for missing values !!
    """
    # make sure we're in the proper lon range
    lon0 = 0 if not np.any(lon < 0) else -180
    if np.any(lon2 < lon0):
        lon2 = np.where(lon2 < lon0, lon2 + 360, lon2)
    if np.any(lon2 > lon0 + 360):
        lon2 = np.where(lon2 > lon0 + 360, lon2 - 360, lon2)

    if np.ndim(lon2) < 2 and np.shape(lon2) != np.shape(lat2):
        lon2, lat2 = np.meshgrid(lon2, lat2)

    interp = RegularGridInterpolator((lat, lon), values, method="linear", bounds_error=False)
    res = interp((lat2, lon2))

    # make sure we're handling the lon boundary correctly
    if wrap_lon:
        lon2_w_m = (lon2 < lon.min()) | (lon2 > lon.max())
        if lon2_w_m.any():
            lon_w = np.array([lon[-1], lon[0] + 360])
            values_w = values[:, [-1, 0]]
            lat2_w = lat2[lon2_w_m]
            lon2_w = lon2[lon2_w_m]
            # lon2_w[lon2_w < lon_w.min()] += 360
            lon2_w = np.where(lon2_w < lon_w.min(), lon2_w + 360, lon2_w)
            assert not (lon2_w > lon_w.max()).any()
            interp_w = RegularGridInterpolator((lat, lon_w), values_w, method="linear", bounds_error=True)
            res_w = interp_w((lat2_w, lon2_w))
            res[lon2_w_m] = res_w

    # For masked data with domain less than the full grid (e.g. only ocean points), use MaskedGrid
    if mask is not None:
        # these values contained masked input
        # m = interpolate(lon, lat, mask.astype(float), lon2, lat2) > 0  ## somehow this is not always equal to NaN calculations...
        m = np.isnan(interpolate(lon, lat, np.where(mask, np.nan, 0), lon2, lat2))
        if m.sum() == 0:
            return res # nothing to do
        assert not np.any(np.isnan(m))
        assert not np.any(np.isnan(values[~mask]))
        assert not np.any(np.isnan(res[~m])), 'interpolate has nans outside mask'

        # use MaskedGrid for nearest-neighbor interpolation with lon-wrapping
        grid = MaskedGrid(lon, lat, mask=~mask)  # MaskedGrid use mask for *valid*
        ii, jj = np.array(list(_iterate_coords(grid, lon2[m], lat2[m], context=context))).T
        res[m] = values[ii, jj]

        # if not np.any(np.isnan(values[~mask])):
        assert np.all(np.isfinite(res[m])), "NaNs remain (check MaskedGrid)"  # that's done well
        assert np.all(np.isfinite(res[lon2_w_m])), "NaNs remain (check wrap_lon)"
        assert np.all(np.isfinite(res)), "NaNs remain"

    return res



def global_mean(lon, lat, v):
    if lat.ndim == 2:
        LAT = lat
    else:
        _, LAT = np.meshgrid(lon, lat)

    w = np.cos(np.deg2rad(LAT))

    if v.ndim == 2:
        gismean = np.empty(v.shape)
        m = np.isfinite(v)
        return np.sum(w[m]*v[m])/np.sum(w[m])

    gismean = np.empty(v.shape[0])

    for i in range(gismean.size):
        m = np.isfinite(v[i])
        gismean[i] = np.sum(w[m]*v[i][m])/np.sum(w[m])

    return gismean