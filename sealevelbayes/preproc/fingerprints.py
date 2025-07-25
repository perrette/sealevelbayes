"""The pendant of models for the tide-gauge projections
"""
import os
from pathlib import Path
import numpy as np
import netCDF4 as nc
import xarray as xa
import pandas as pd

from sealevelbayes.logs import logger
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.shared import MAP_FRED
from sealevelbayes.datasets.maptools import MaskedGrid, _iterate_coords, interpolate
from sealevelbayes.datasets.satellite import get_satellite_timeseries
from sealevelbayes.datasets.caron2018 import get_caron2018_ensemble_data, get_caron2018_by_frederikse
from sealevelbayes.datasets.frederikse2020 import open_rgi_fingerprint
from sealevelbayes.preproc.linalg import detrend_timeseries

FINGERPRINTSDIR = get_datapath("savedwork/fingerprints")

def load_oceandyn_coef_ensemble(models=None, driver='zostoga', sel=None, isel=None):
    logger.info(f"Load Ocean Dyn Scaling Coef ensemble...")
    with xa.open_dataset(FINGERPRINTSDIR/f"fingerprints_zos_{driver}.nc") as ds:

        if sel: ds = ds.sel(sel)
        if isel: ds = ds.isel(isel)

        # use all available models
        if models is None:
            models = [v[:-len("_RMSE")] for v in ds if v.endswith("_RMSE")]

        scale = 1000 if driver == "tas" else 1  # in mm/deg for tas (instead of m/deg), and in mm/mm = m/m for zostoga

        logger.info(f"Load Ocean Dyn Scaling Coef ensemble...concatenate...")
        a = ds[models].to_array(dim='model')*scale
        logger.info(f"Load Ocean Dyn Scaling Coef ensemble...concatenate...done")
        return a


def load_oceandyn_coef_ensemble_coords(lons, lats, common_mask=True, **kwargs):

    # load full fingerprint
    a = load_oceandyn_coef_ensemble(**kwargs)

    if common_mask:
        logger.info("oceandyn scaling:: interpolate from all GCMs on a common mask...")
        fprints = interpolate(a['lon'].values, a['lat'].values, a.transpose('lat', 'lon', 'model').values, lons, lats, mask=np.isnan(a).any(dim='model').values, context="oceandyn all")
        logger.info("oceandyn scaling:: interpolate from all GCMs on a common mask...done")

    else:
        logger.info("oceandyn scaling:: interpolate from GCMs on individual masks...")
        fprints = []
        for i, model in enumerate(a.model.values):
            b = a.sel(model=model)
            logger.info(f"oceandyn scaling:: interpolate from GCMs on individual masks...{model} ({i+1}/{len(models)})")
            fprint = interpolate(a['lon'].values, a['lat'].values, b.values, lons, lats, mask=np.isnan(b.values), context=model)
            fprints.append(fprint)
        logger.info("oceandyn scaling:: interpolate from GCMs on individual masks...done")
        fprints = np.array(fprints).T # tide-gauges x model

    return np.asarray(fprints) # tide-gauges x model


def _get_oceandyn_ensemble_reduced_space(driver='zostoga', models=None):
    """
    """
    tag = f"{driver}"
    if models is not None and len(models) != 21:
        tag += f"-{len(models)}-models"
    fname = FINGERPRINTSDIR/f"fingerprints_zos_{tag}_eof.nc"

    if fname.exists():
        # return ReducedSpaceFingerprintGridded.load(fname, sel=sel, isel=isel)
        logger.info(f"Load Ocean Dyn Scaling Coef EOFs {fname}")
        return xa.open_dataset(fname)

    a = load_oceandyn_coef_ensemble(driver=driver, models=models, common_mask=True)

    logger.info(f"Ocean dyn scaling coefs :: Crunch EOFs...")
    eof = ReducedSpaceFingerprintGridded.crunch(a.lon.values, a.lat.values, ['gsl'], [np.array([a[m].values for m in a.model])],
        truncate = len(a.model) - 1)
    logger.info(f"Ocean dyn scaling coefs :: Crunch EOFs... done")

    eof = eof.to_xa()  # to xarray

    if not fname.parent.exists():
        fname.parent.mkdir(exist_ok=True, parents=True)

    # save
    logger.info(f"Write Ocean dyn scaling coefs EOFs to {fname}")
    eof.to_netcdf(fname, encoding = {'Vh': {'zlib': True}})

    return eof


def get_oceandyn_ensemble_reduced_space(lons, lats, truncate=None, **kwargs):
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    assert lons.size == lats.size
    with _get_oceandyn_ensemble_reduced_space(**kwargs) as ds:
        eof = ReducedSpaceFingerprintGridded(ds.lon.values, ds.lat.values, ds.variable.values, ds['mean'].values, ds['S'].values, ds['Vh'].values)
        if truncate is not None:
            logger.info(f"truncate oceandyn EOFs to {truncate} principal components")
            eof = eof.truncate(truncate)
        logger.info("sample oceandyn EOFs...")

        mask = np.isnan(eof.mean[0])
        eof.mean = interpolate(eof.lons, eof.lats, eof.mean.transpose([1, 2, 0]), lons, lats, mask=mask).T
        assert not np.isnan(eof.mean).any()
        # eof.Vh = np.array([[interpolate(eof.lons, eof.lats, x, lons, lats) for x in row] for row in eof.Vh])
        eof.Vh = interpolate(eof.lons, eof.lats, eof.Vh.transpose([2, 3, 0, 1]), lons, lats, mask=mask).transpose([1, 2, 0])
        assert not np.isnan(eof.Vh).any()
        logger.info("sample oceandyn EOFs...done")
        return eof


def get_fingerprint_glacier_region(region_id, lons, lats):
    """here only the RSL fingerprint can be returned
    """
    with open_rgi_fingerprint(region_id) as ds:
        # geoid = ds['geoid'].load()
        rsl = ds['rsl'].load().values
        rad = ds['rad'].load().values
        lon = ds['x'].values
        lat = ds['y'].values

    # the fingerprints are defined everywhere, so use the faster RegularGridInterpolator
    return np.array([interpolate(lon, lat, rsl, lons, lats), interpolate(lon, lat, rad, lons, lats)]).T



def get_fingerprint_ice(source, lons, lats):
    if source.startswith("glacier_region"):
        return get_fingerprint_glacier_region(int(source[len("glacier_region_"):]), lons, lats)

    with nc.Dataset(FINGERPRINTSDIR/f"fingerprints_ice.nc") as ds:
        source_fred = MAP_FRED.get(source, source)
        rsl_rad = np.stack([
            ds[source_fred+"_rsl"][:].filled(np.nan),
            ds[source_fred+"_rad"][:].filled(np.nan),
            ], axis=-1)
        return interpolate(ds['lon'][:], ds['lat'][:], rsl_rad, lons, lats, mask=np.isnan(rsl_rad).any(axis=-1), context=source)


def get_fingerprint_ice_time_dependent(source, lons, lats, w=5):

    filename = FINGERPRINTSDIR/f"fingerprints_ice_time_dependent_{source}_{w}years.nc"
    source_fred = MAP_FRED.get(source, source)

    if not filename.exists():
        from sealevelbayes.preproc.massfingerprints import calc_fred_fingerprints_time_dependent, calc_yearly_fred_fingerprints
        if w == 1:
            logger.info(f"Calculate yearly mass fingerprint")
            fingers = calc_yearly_fred_fingerprints(source_fred)
        else:
            logger.info(f"Calculate mass fingerprint for w = {w} years")
            fingers = calc_fred_fingerprints_time_dependent(source_fred, w=w)

        logger.info(f"...write to netcdf...")
        fingers.to_netcdf(filename)
        logger.info(f"...done")

    with xa.open_dataset(filename) as ds:
        rsl_rad = np.stack([
            ds[source_fred+"_rsl"].transpose("lat", "lon", "time").values,
            ds[source_fred+"_rad"].transpose("lat", "lon", "time").values,
            ], axis=-1)

        # return interpolate(ds['lon'][:], ds['lat'][:], rsl_rad, lons, lats, mask=ds[source_fred+"_rsl"][:].mask.any(axis=0), context=source)
        results = interpolate(ds['lon'].values, ds['lat'].values, rsl_rad, lons, lats, mask=np.isnan(ds[source_fred+"_rsl"].values).any(axis=0), context=source)
        return results.transpose([0, 2, 1]) # loc x time x rsl/rad => loc x rsl/rad x time



def get_gia_by_frederikse(lons, lats):
    with xa.open_dataset(get_caron2018_by_frederikse()) as ds:
        variables = ["rsl_mean", "rsl_sterr", "rad_mean", "rad_sterr", "gsl_mean", "gsl_sterr"]
        arrays = [interpolate(ds['lon'].values, ds['lat'].values, ds[v].values, lons, lats, context=f"GIA::{v}") for v in variables]
        return np.array(arrays).T  # variables as last dimension


def get_gia_ensemble(lons, lats):
    with nc.Dataset(get_caron2018_ensemble_data()) as ds:
        grid = MaskedGrid(ds['lon'][:], ds['lat'][:], mask=all_good)  # here we keep masked grid as it is likely more efficient for 5000 ensemble members (we don't use it anyway)
        coords = list(_iterate_coords(grid, lons, lats, context="GIA"))

        return (
            np.array([ds["vlm"][:, i, j].filled(np.nan) for i,j in coords]),
            np.array([ds["rsl"][:, i, j].filled(np.nan) for i,j in coords]),
            ds["likelihood"][:].filled(0),
            )


class ReducedSpaceFingerprintGridded:
    def __init__(self, lons, lats, variables, mean, S, Vh, U=None):
        self.lons = lons
        self.lats = lats
        self.variables = variables
        self.mean = mean
        self.S = S
        self.Vh = Vh
        self.U = U

    @property
    def sVh(self):
        # the broadcast rules apply on the last dimension
        return (self.Vh.T * self.S).T

    def eval(self, u):
        """ u is supposed to be i.i.d with size(u) == size(S)
        """
        return self.mean + u @ self.sVh

    # def get_cov_ij(self, i, j, r=None):
    #     # for testing
    #     if r is None:
    #         r = self.S.size
    #     sV = self.S[:r, None]*self.Vh[:r, [i, j]]
    #     return sV.T @ sV

    def truncate(self, order):
        return ReducedSpaceFingerprintGridded(self.lons, self.lats, self.variables, self.mean, self.S[:order], self.Vh[:order], self.U[:, :order] if self.U is not None else None)


    @staticmethod
    def _get_truncate_by_variance(S, variance_threshold):
        total_var = (S**2).sum()
        return np.where((S**2).cumsum() > total_var * (1 - variance_threshold))[0][0]

    def to_xa(self):
        nlat = self.lats.size
        nlon = self.lons.size
        nvar = self.variables.size
        samples = self.S.size

        S, Vh, U = self.S, self.Vh, self.U

        coords={"lat": self.lats, "lon": self.lons, "variable": self.variables, "sample": np.arange(samples)}

        if U is not None:
            coords["time"] = np.arange(U.shape[0])

        ds = xa.Dataset(coords=coords)
        ds["mean"] = ("variable", "lat", "lon"), self.mean
        ds["S"] = ("sample",), S
        ds["Vh"] = ("sample", "variable", "lat", "lon"), Vh

        if self.U is not None:
            ds["U"] = ("time", "sample"), U

        return ds

    @classmethod
    def from_xa(cls, ds):
        return cls(ds.lon.values, ds.lat.values, ds.variable.values, ds['mean'].values, ds['S'].values, ds['Vh'].values, ds['U'].values if 'U' in ds else None)

    def save(self, fname):
        ds = self.to_xa()
        ds.to_netcdf(fname, encoding={k: {"zlib": True} for k in ['Vh', 'U'] if k in ds})

    @classmethod
    def load(cls, fname, sel=None, isel=None):
        ds = xa.open_netcdf(fname)
        if sel is not None: ds = ds.sel(sel)
        if isel is not None: ds = ds.isel(isel)
        return cls.from_xa(ds)


    @classmethod
    def crunch(cls, lons, lats, variables, arrays, likelihood=None, truncate=None, variance_threshold=None):
        """
        ...
        truncate: truncate the reduced order space
        variance_threshold: use a criteria on total variance to decide where to truncate,
            e.g. 0.01 means we explain 99% of the total variance
        """

        variables = np.asarray(variables)

        nlon = lons.size
        nlat = lats.size
        nvar = variables.size
        assert nvar == len(arrays)
        assert nvar > 0
        shp = arrays[0].shape
        samples = shp[0]
        points = np.prod(shp[1:])
        assert points == nlat * nlon

        # Reshape the input arrays as samples x (variables*lat*lon)
        X = np.concatenate([a.reshape(samples, points) for a in arrays], axis=1) # (5000, (nvar*nlat*nlon))
        # X = np.concatenate([ds['vlm'].values.reshape(samples, points), ds['rsl'].values.reshape(samples, points)], axis=1) # (5000)

        if likelihood is None:
            likelihood = np.ones(samples)
        else:
            assert likelihood.shape == (samples,)

        w = likelihood / likelihood.sum()
        X_mean = (w[:, None]*X).sum(axis=0)
        # The S.V.D decomposition of p^1/2 X is calculated.
        # That way when we later sample X ~ X_mean + W S V' with W ~ N(0, 1) i.i.d (of size r = 20 or 100, the reduced order)
        # we have the right covariance. Proof:
        # A) The covariance we're aiming to match is defined by (xik xjk pk) where pk is the probability of each sample k
        #    which can be calculated as (p^1/2 X)' @ (p^1/2 X) => covariance using probability as weights (here we note X for X - X_mean)
        # B) Let's define U, S and V the SVD decomposition such as p^1/2 X = U S V' and U'U = I_r and V'V = I_lonlat
        #    we have (xik xjk pk) = (p^1/2 X)' @ (p^1/2 X) = V S^2 V'
        # C) Now with i.i.d samples W (or size r, the reduced space) described as above we have
        #    <W' W> = I_r
        #    < X_mean + W S V', X_mean + W S V' > = < W S V', W S V'> = V S < W, W > S V' = V S^2 V'
        #    (the factorization out of <,> is based on the rule that for x with shape(x) = (n, ...)
        #    and n is the number of samples, the covariance is calculated as 1/n x' x;
        #    so any right-side matrix multiplication can be factored out)
        # This is proof that < X_mean + W S V', X_mean + W S V' > = (xik xjk pk)
        #
        wX = w[:, None]**.5 * (X - X_mean)
        invalid = np.isnan(wX)
        if np.any(invalid):
            logger.warning(f"... EOF :: nan values found, only compute over points where ALL models/samples are valid")
            any_invalid = invalid.any(axis=0)
            wX_with_nans = wX
            wX = wX[:, ~any_invalid]
        else:
            invalid = np.zeros(wX.shape[1], dtype=bool)


        logger.info(f"... EOF :: svd decomposition {wX.shape}...")
        U, S, Vh = np.linalg.svd(wX, full_matrices=False)
        logger.info(f"... EOF :: svd decomposition done.")

        if np.any(invalid):
            logger.debug(f"re-introduce the NaNs in EOFs")
            Vh_nans = np.empty((Vh.shape[0], wX_with_nans.shape[1]))
            Vh_nans[:, ~any_invalid] = Vh
            Vh_nans[:, any_invalid] = np.nan
            Vh = Vh_nans

        if variance_threshold is not None:
            logger.info(f"... EOF :: variance_threshold = {variance_threshold}")
            truncate = cls._get_truncate_by_variance(S, variance_threshold)

        if truncate is not None:
            logger.info(f"... EOF :: truncate = {truncate}")
            S = S[:truncate]
            Vh = Vh[:truncate]

        order = S.size   ## order == samples in the case we're considering, but that can be cut-off to 20 or 100.

        # Reshape back to (samples,), variables, lat, lon
        return cls(lons, lats, variables, X_mean.reshape(nvar, nlat, nlon), S, Vh.reshape(order, nvar, nlat, nlon))


def _get_gia_ensemble_reduced_space(ignore_disk=False, truncate=100):
    """like GIA ensemble but returns arrays for sampling in reduced space via PC analysis (more efficient for large stations)
    """
    fname = get_datapath("savedwork/gia/gia_ensemble_caron2018_eof.nc")
    if not ignore_disk and fname.exists():
        # return ReducedSpaceFingerprintGridded.load(fname, sel=sel, isel=isel)
        logger.info(f"Load GIA EOF {fname}")
        return xa.open_dataset(fname)


    from sealevelbayes.datasets.caron2018 import get_caron2018_ensemble_data
    with xa.open_dataset(get_caron2018_ensemble_data()) as ds:

        variables = ["vlm", "rsl"]

        logger.info(f"GIA :: Crunch EOFs...")
        eof = ReducedSpaceFingerprintGridded.crunch(
            ds.lon.values,
            ds.lat.values,
            variables, [ds[v].values for v in variables], likelihood=ds.likelihood.values, truncate=truncate).to_xa()
        logger.info(f"GIA :: Crunch EOFs... done")

    if not ignore_disk:
        if not fname.parent.exists():
            fname.parent.mkdir(exist_ok=True, parents=True)

        # save
        logger.info(f"Write GIA EOF to {fname}")
        eof.to_netcdf(fname, encoding = {'Vh': {'zlib': True}})

    return eof


def get_gia_ensemble_reduced_space(lons, lats, truncate=None):
    assert lons.size == lats.size
    with _get_gia_ensemble_reduced_space() as ds:
        eof = ReducedSpaceFingerprintGridded(ds.lon.values, ds.lat.values, ds.variable.values, ds['mean'].values, ds['S'].values, ds['Vh'].values)
        if truncate is not None:
            logger.info(f"truncate GIA EOFs to {truncate} principal components")
            eof = eof.truncate(truncate)
        logger.info("sample GIA EOFs...")
        eof.mean = interpolate(eof.lons, eof.lats, eof.mean.transpose([1, 2, 0]), lons, lats).T
        # eof.Vh = np.array([[interpolate(eof.lons, eof.lats, x, lons, lats) for x in row] for row in eof.Vh])
        eof.Vh = interpolate(eof.lons, eof.lats, eof.Vh.transpose([2, 3, 0, 1]), lons, lats).transpose([1, 2, 0])
        logger.info("sample GIA EOFs...done")
        return eof