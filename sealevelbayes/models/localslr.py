"""The pendant of models for the tide-gauge projections
"""
from functools import partial
import numpy as np # type: ignore

import pytensor # type: ignore
import pytensor.tensor as pt # type: ignore
import pymc as pm # type: ignore

from sealevelbayes.models.compat import getmodeldata
from sealevelbayes.logs import logger
from sealevelbayes.preproc.fingerprints import get_fingerprint_ice, load_oceandyn_coef_ensemble_coords, get_oceandyn_ensemble_reduced_space, get_gia_ensemble, get_gia_ensemble_reduced_space, get_fingerprint_ice_time_dependent
from sealevelbayes.models.globalslr import slr_model_global, calculate_global_slr, define_predictors
from sealevelbayes.models.metadata import SOURCES, FIELDS, DIAGS
from sealevelbayes.datasets.maptools import compute_geodesic_distances

# TRACE LEVELS
TRACE_LIGHT = 0
TRACE_MEDIUM = 2
TRACE_HEAVY = 3


def calculate_sterodynamic_slr(X, stations, steric_coef_cov=True, steric_coef_error=True, steric_coef_scale=None,
                               models=None, driver='zostoga',
    oceandyn=True, oceandyn_method=None, reduced_space_coef=False, data=None, trace_level=TRACE_MEDIUM, **kwargs):

    if not oceandyn:
        return X["steric"][..., None] + np.zeros(len(stations))

    # prepare data for steric coefficients
    model = pm.modelcontext(None)
    ns = len(stations)
    lons, lats = np.array([(station["Longitude"], station["Latitude"]) for station in stations]).T

    if models is not None and len(models) == 1:
        steric_coef_error = False

    if data is not None:
        steric_coef_mu = define_from_data(data, "steric_coef_mu")
        steric_coef_sd = define_from_data(data, "steric_coef_sd")

    elif not (steric_coef_error and steric_coef_cov and reduced_space_coef):
        fingers = load_oceandyn_coef_ensemble_coords(lons, lats, models=models, driver=driver)

        assert not np.any(np.isnan(fingers))
        # if np.any(np.isnan(fingers)):
        #     raise ValueError(f'{source} some fingerprint is NaN: {fingers}')

        steric_coef_mu = pm.ConstantData("steric_coef_mu", np.mean(fingers, axis=1), dims="station")
        steric_coef_sd = pm.ConstantData("steric_coef_sd", np.std(fingers, axis=1), dims="station")

    if not steric_coef_error:
        logger.info("Use multi-model mean for sterodynamic coefficients")
        steric_coef = steric_coef_mu

    elif not steric_coef_cov:
        logger.info("Draw uncorrelated samples from sterodynamic coefficients")
        steric_coef = pm.Normal("steric_coef", mu=steric_coef_mu, sigma=steric_coef_sd, dims="station")

    elif reduced_space_coef:

        if data is None:
            logger.info("Draw correlated samples from sterodynamic coefficients (reduced space)")
            eof = get_oceandyn_ensemble_reduced_space(lons, lats, models=models, driver=driver)
            r = eof.S.size
            logger.info(f"...sample in reduced space of dim {r}")

            # for the record, write mean and s.d. as constant data
            steric_coef_mu = pm.ConstantData("steric_coef_mu", eof.mean[0], dims="station")
            steric_coef_sd = pm.ConstantData("steric_coef_sd", ((eof.sVh**2).sum(axis=0)**.5)[0], dims="station")

            # save the matrix to transform from reduced space back to stations space
            if "steric_coef_reduced_space" not in model.coords:
                model.add_coord("steric_coef_reduced_space", np.arange(eof.S.size))

            steric_coef_sVh = pm.ConstantData('steric_coef_sVh', eof.sVh[:, 0, :], dims=('steric_coef_reduced_space', 'station'))

        else:
            # simply load the matrix to transform from reduced space back to stations space
            steric_coef_mu = define_from_data(data, 'steric_coef_mu')
            steric_coef_sVh = define_from_data(data, 'steric_coef_sVh')

        steric_coef_scale_iid = pm.Normal("steric_coef_scale_iid_r", dims=("steric_coef_reduced_space"))
        steric_coef = steric_coef_mu + pt.sum(steric_coef_scale_iid[:, None] * steric_coef_sVh, axis=0)

        if trace_level >= TRACE_MEDIUM:
            # does not include time dimension, so not super heavy unless gridded version
            steric_coef = pm.Deterministic("steric_coef", steric_coef, dims="station")

    else:
        logger.info("Draw correlated samples from sterodynamic coefficients")

        cov = np.cov(fingers)

        if steric_coef_scale is not None:
            logger.info(f"...decrease covariance on steric coefficients with a spatial scale of {steric_coef_scale} km")
            distance_matrix = compute_geodesic_distances(lons, lats)
            cov = cov * np.exp(-distance_matrix / steric_coef_scale)

        epsilon = 1e-6
        # epsilon = (1e-2)**2  # that's (1%)^2 = 1e-4, a quantity that we can safely consider to be "noise"
        cov += np.diag(np.zeros(ns)+epsilon)  # so that cholesky decomposition works
        chol = np.linalg.cholesky(cov)
        pm.ConstantData("steric_coef_chol", chol)
        logger.info("...sample in full space...")
        coefs_raw = pm.Normal("steric_coef_raw", mu=0, sigma=pt.ones(ns))
        steric_coef_demean = chol @ coefs_raw

        steric_coef = pm.Deterministic("steric_coef", steric_coef_demean + steric_coef_mu, dims="station")


    if driver == 'tas':
        tas = X["tas"]
        tasdiff = pt.concatenate([tas[:, 1:2]-tas[:, 0:1], tas[:, 1:]-tas[:, :-1]], axis=1)
        steric = X["steric"][..., None] + tasdiff[..., None]*steric_coef[None, None]
    else:
        steric = X["steric"][..., None]*(1 + steric_coef)[None, None]

    return steric


def getfromdata(data, name, model=None):
    a = data[name]
    if 'station' in a.dims:
        stations = np.asarray(pm.modelcontext(model).coords['station'], dtype=int)
        a = a.sel(station=stations)
    return a

def define_from_data(data, name, model=None):
    model = pm.modelcontext(model)
    if hasattr(model, name):
        return getattr(model, name) # was already defined
    a = getfromdata(data, name, model=model)
    for dim in a.dims:
        if dim not in model.coords:
            model.add_coord(dim, a.coords[dim].values)
    return pm.ConstantData(name, a, dims=a.dims)


def define_all_from_data(data, skip=[], skip_dims=[]):
    model = pm.modelcontext(None)
    for name in data:
        if skip and name in skip: continue
        if skip_dims and any(dim in skip_dims for dim in data[name].dims): continue
        define_from_data(data, name, model=model)



def define_historical_mass_contribution(global_source, name, historical_window, data=None, lons=None, lats=None):
    """ Define RSL and RAD components from Frederikse's datasets
    """

    h = historical_window//2

    model = pm.modelcontext(None)

    if "historical_time" not in model.coords:
        model.add_coord("historical_time", np.arange(1900+h, 2018+1-h))

    logger.info(f"{name}: time-dependent historical fingerprint ({historical_window} years window)")

    if data is not None:
        coef_rsl_historical = define_from_data(data, f"{name}_coef_rsl_historical")
        coef_rad_historical = define_from_data(data, f"{name}_coef_rad_historical")
        nt = int(coef_rsl_historical.shape[0].eval())

    else:
        if (historical_window // 2) * 2 + 1 != historical_window:
            raise ValueError(f"historical window for time-varying fingerprint must be an odd integer number")

        if lons is None: lons = getmodeldata("lons")
        if lats is None: lats = getmodeldata("lats")

        fingers_time = get_fingerprint_ice_time_dependent(name, lons, lats, w=historical_window)
        if np.any(np.isnan(fingers_time)):
            raise ValueError(f'{name} some time-dependent fingerprint is NaN: {fingers_time}')

        nt = fingers_time.shape[2]  # location x {rsl, rad} x time
        nt0 = 2018 - 1900 + 1
        assert nt == nt0 - 2*h # 5-year moving window, so 2 years are shaved at the beginning and towards the end

        coef_rsl_historical_ = np.array([f[0] for f in fingers_time]).T
        coef_rad_historical_ = np.array([f[1] for f in fingers_time]).T
        coef_rsl_historical = pm.Data(f"{name}_coef_rsl_historical", coef_rsl_historical_, dims=("historical_time", "station"), mutable=False)
        coef_rad_historical = pm.Data(f"{name}_coef_rad_historical", coef_rad_historical_, dims=("historical_time", "station"), mutable=False)

    historical_start = [0]*h
    historical = np.arange(nt)
    historical_end = [-1]*h
    n_future = model.coords['year'][-1] - 2019 + 1
    # future = [-1] * (model.coords['year'][-1] - 2019 + 1)
    # ix = np.concatenate([historical_start, historical, historical_end, future])
    ix = np.concatenate([historical_start, historical, historical_end]).astype(int)

    fingers = get_fingerprint_ice(name, lons, lats)
    coef_rsl_future = np.array([f[0] for f in fingers])
    coef_rad_future = np.array([f[1] for f in fingers])
    coef_rsl = np.concatenate([coef_rsl_historical_[ix, :], np.repeat(coef_rsl_future[None, :], n_future, axis=0)], axis=0)
    coef_rad = np.concatenate([coef_rad_historical_[ix, :], np.repeat(coef_rad_future[None, :], n_future, axis=0)], axis=0)

    rsl = global_source[:, :, None] * coef_rsl[None, :, :]
    rad = global_source[:, :, None] * coef_rad[None, :, :]

    return rsl, rad


def define_mass_contribution(global_source, name, historical_mass_fingerprints=False, historical_window=5, data=None, lons=None, lats=None):

    if historical_mass_fingerprints:
        return define_historical_mass_contribution(global_source, name, historical_window=historical_window, data=data)


    logger.info(f"{name}: fixed fingerprint")

    if data is not None:
        coef_rsl = define_from_data(data, f"{name}_coef_rsl")
        coef_rad = define_from_data(data, f"{name}_coef_rad")

    else:

        model = pm.modelcontext(None)
        if lons is None: lons = getmodeldata("lons")
        if lats is None: lats = getmodeldata("lats")

        fingers = get_fingerprint_ice(name, lons, lats)
        if np.any(np.isnan(fingers)):
            raise ValueError(f'{name} some fingerprint is NaN: {fingers}')

        coef_rsl = pm.Data(f"{name}_coef_rsl", np.array([f[0] for f in fingers]), dims="station", mutable=False)
        coef_rad = pm.Data(f"{name}_coef_rad", np.array([f[1] for f in fingers]), dims="station", mutable=False)

    rsl = global_source[..., None] * coef_rsl[None, None]
    rad = global_source[..., None] * coef_rad[None, None]

    return rsl, rad

def format_array(array):
    fmt = lambda x : ", ".join(map(lambda x: f"{x:.2f}", x))
    if np.ndim(array) == 0:
        return f"{array}"
    else:
        return f"{fmt(array.flat[:2])}...{fmt(array.flat[-1:])} -> {array.shape}"

def custom_covariance_function(dists, length_scale, std_error=1):
    if np.ndim(std_error) == 0:
        return std_error**2 * np.exp(-dists / length_scale)
    else:
        return std_error[:, None] * np.exp(-dists / length_scale) * std_error[None, :]

def cholesky(cov, eps=1e-6):
    n = cov.shape[0]
    if hasattr(n, "eval"):
        n = n.eval()
    return np.linalg.cholesky(cov + eps*np.eye(n))

def get_covariance_matrix_from_spatial_scale(lons, lats, vlm_res_spatial_scale, vlm_res_sd=1):
    distance_matrix = compute_geodesic_distances(lons, lats)
    return custom_covariance_function(distance_matrix, vlm_res_spatial_scale, vlm_res_sd)

def get_covariance_matrix_from_spatial_scale_and_autocorrelation(lons, lats, vlm_res_spatial_scale, nt, vlm_res_autocorrel, std_error=1):
    lon2 = np.ones((nt, 1)) * lons[None, :]
    lat2 = np.ones((nt, 1)) * lats[None, :]
    distance_matrix = compute_geodesic_distances(lon2.flat, lat2.flat)
    correl_spatial = custom_covariance_function(distance_matrix, vlm_res_spatial_scale)
    time_index = (np.ones((1, len(lons))) * np.arange(nt)[:, None]).flatten()
    correl_temporal = vlm_res_autocorrel**np.abs(time_index[:, None] - time_index[None, :])
    cov = correl_spatial * correl_temporal
    if np.ndim(std_error) == 0:
        return std_error**2 * cov
    else:
        std_error_broad = (np.ones((nt, 1)) * std_error[None, :]).flatten()
        return std_error_broad[:, None] * cov * std_error_broad[None, :]

def calculate_vlm_residual(model,
                           vlm_res_mode="constant",
                           vlm_res_split_year=2000,
                           vlm_res_autocorrel=None,
                           vlm_res_sd=1, vlm_res_cauchy=False, vlm_res_spatial_scale=None, vlm_res_domain='psmsl'):

    lons = getmodeldata("lons", model)
    lats = getmodeldata("lats", model)
    years = np.array(model.coords['year'])
    n = len(lons)

    if vlm_res_domain == "psmsl":
        logger.info("Only sample VLM residual at PSMSL locations")
        vlm_res_sd = np.where(getmodeldata("psmsl_ids", model) > 0, vlm_res_sd, 1e-12)

    elif vlm_res_domain == "everywhere":
        logger.info("Sample VLM residual everywhere")
        pass
    else:
        raise NotImplementedError(vlm_res_domain)

    # here dummy mode for resampling
    if vlm_res_mode == "dummy":
        vlm_res = pm.Normal("vlm_res", dims=("year", "station"))
        logger.info(f"Prior VLM res mode {vlm_res_mode} {vlm_res.shape.eval()}")
        return vlm_res

    if vlm_res_cauchy:
        assert vlm_res_spatial_scale is None
        assert vlm_res_autocorrel is None
        logger.info(f"Prior VLM res Cauchy(0, {format_array(vlm_res_sd)}) mm/yr")
        vlm_res = pm.Cauchy("vlm_res", 0, vlm_res_sd, dims="station")
        return vlm_res


    if vlm_res_mode == "constant":
        assert vlm_res_autocorrel is None

        if not vlm_res_spatial_scale:
            logger.info(f"Prior VLM res Normal(0, {format_array(vlm_res_sd)}) mm/yr")
            vlm_res = pm.Normal("vlm_res", 0, vlm_res_sd, dims="station")

        else:
            cov = get_covariance_matrix_from_spatial_scale(lons, lats, vlm_res_spatial_scale)
            chol = cholesky(cov)
            vlm_res_iid = pm.Normal("_vlm_res_iid", 0, 1, shape=n)
            vlm_res = pm.Deterministic("vlm_res", pt.dot(chol, vlm_res_iid), dims="station")

        return vlm_res

    assert vlm_res_mode in ("split", "decadal")

    if vlm_res_mode == "split":
        logger.info(f"Prior VLM res mode {vlm_res_mode} ({vlm_res_split_year}) (spatial scale {vlm_res_spatial_scale}, autocorrel {vlm_res_autocorrel})")
        nsplit = 2

    elif vlm_res_mode == "decadal":
        logger.info(f"Prior VLM res mode {vlm_res_mode} (spatial scale {vlm_res_spatial_scale}, autocorrel {vlm_res_autocorrel})")

        # decades = 20 # 1900-1909, 1910-1919, ..., 2010-2019, ..., 2090-2099
        time_block = 10
        nsplit = (years[-1] - years[0] + 1) // time_block
        assert nsplit == 20, 'Decadal mode is only checked for 1900-2099 for now (double check and remove)'

        # vectorize the assignment (that should be the most efficient for higher dimensions)
        year_indices = np.zeros(len(years), dtype=int)
        year_indices[:] = nsplit - 1  # fill with the last index (propagate the last value)
        # fill the past
        for i in range(nsplit):
            year_indices[i*time_block:(i+1)*time_block] = i

        block_matrix = np.zeros((len(years), nsplit))
        for i in range(nsplit):
            block_matrix[year_indices == i, i] = 1
        # example for time_block = 2, 11 years of record -> nsplit = 5
        # array([[1., 0., 0., 0., 0.],
        #        [1., 0., 0., 0., 0.],
        #        [0., 1., 0., 0., 0.],
        #        [0., 1., 0., 0., 0.],
        #        [0., 0., 1., 0., 0.],
        #        [0., 0., 1., 0., 0.],
        #        [0., 0., 0., 1., 0.],
        #        [0., 0., 0., 1., 0.],
        #        [0., 0., 0., 0., 1.],
        #        [0., 0., 0., 0., 1.],
        #        [0., 0., 0., 0., 1.]])


    else:
        raise NotImplementedError(vlm_res_mode)


    if vlm_res_spatial_scale and vlm_res_autocorrel is not None:
        cov_split = get_covariance_matrix_from_spatial_scale_and_autocorrelation(lons, lats, vlm_res_spatial_scale, 2, vlm_res_autocorrel, std_error=vlm_res_sd)
        chol = cholesky(cov_split)
        vlm_res_iid = pm.Normal("_vlm_res_iid", 0, vlm_res_sd, shape=(nsplit, n))  # time, space
        vlm_res_split = pt.dot(chol, vlm_res_iid.reshape([nsplit*n])).reshape((nsplit, n))

        if vlm_res_mode == "split":
            vlm_res = pt.switch(years[:, None] < vlm_res_split_year, vlm_res_split[0], vlm_res_split[1])
        else:
            vlm_res = block_matrix @ vlm_res_split

        logger.info(f"vlm res shape {vlm_res.eval().shape}")
        vlm_res = pm.Deterministic("vlm_res", vlm_res, dims=("year", "station"))


    elif vlm_res_spatial_scale and vlm_res_autocorrel is None:
        cov = get_covariance_matrix_from_spatial_scale(lons, lats, vlm_res_spatial_scale)
        chol = cholesky(cov)

        vlm_res_iid = pm.Normal("_vlm_res_iid", 0, vlm_res_sd, shape=(nsplit, n))

        # spatial correlation
        vlm_res_split = pt.dot(chol, vlm_res_iid.T).T

        if vlm_res_mode == "split":
            vlm_res = pt.switch(years[:, None] < vlm_res_split_year, vlm_res_split[0], vlm_res_split[1])
        else:
            vlm_res = block_matrix @ vlm_res_split

        logger.info(f"vlm res shape {vlm_res.eval().shape}")
        vlm_res = pm.Deterministic("vlm_res", vlm_res, dims=("year", "station"))

    elif not vlm_res_spatial_scale and vlm_res_autocorrel is None:
        logger.info(f"Prior VLM res mode {vlm_res_mode}")
        vlm_res_iid = pm.Normal("_vlm_res_iid", 0, vlm_res_sd, shape=(nsplit, n))
        vlm_res_split = vlm_res_iid

        if vlm_res_mode == "split":
            vlm_res = pt.switch(years[:, None] < vlm_res_split_year, vlm_res_split[0], vlm_res_split[1])
        else:
            vlm_res = block_matrix @ vlm_res_split

        logger.info(f"vlm res shape {vlm_res.eval().shape}")
        vlm_res = pm.Deterministic("vlm_res", vlm_res, dims=("year", "station"))

    else:
        raise NotImplementedError(f"spatial correl {vlm_res_spatial_scale} autocorrel {vlm_res_autocorrel}")


    return vlm_res


def calculate_regional_slr(X, stations,
                           historical_mass_fingerprints=False, historical_window=5,
                           regional_glacier=False, steric_kwargs={},
                           gia_eof=False, gia_eof_num=None, data=None, **vlm_kwargs):
    """
    Parameters
    ----------
    X: dict of predictors, including global SLR data, for one scenario.
        Expected: (tas), steric, ais, gis, glacier, landwater (2-d arrays, dims="experiment", "year")

    Returns
    -------
    RSL, RAD : dict with 3-D arrays (dims="experiment", "year", "location")
    -
    """
    RSL = {}
    RAD = {}

    # broadcasting:
    # predictor: (experiment, year)[..., None] ==> (experiment, year, None)
    # coef: (station)[None, None] ==> (None, None, station)

    RSL['steric'] = calculate_sterodynamic_slr(X, stations, data=data, **steric_kwargs)
    # RAD['steric'] = 0 # we assume no bottom movement from ocean loading here. TODO: quantify this omission
    RAD['steric'] = pt.zeros_like(RSL['steric']) # we assume no bottom movement from ocean loading here. TODO: quantify this omission


    model = pm.modelcontext(None)
    lons = getmodeldata("lons")
    lats = getmodeldata("lats")

    # Mass contribution (ice and landwater)
    if regional_glacier:
        RSL["glacier"], RAD["glacier"] = 0, 0
        for i, r in enumerate(model.coords['glacier_region']):
            rsl, rad = define_mass_contribution(model._glacier_sea_level_rate_final[i], f"glacier_region_{r}", data=data)
            RSL["glacier"], RAD["glacier"] = rsl + RSL["glacier"], rad + RAD["glacier"]


    for source in ["glacier", "gis", "ais", "landwater"]:
        if regional_glacier and source == "glacier":
            continue

        RSL[source], RAD[source] = define_mass_contribution(X[source], source, data=data, lons=lons, lats=lats,
            historical_mass_fingerprints=historical_mass_fingerprints is not None and source in historical_mass_fingerprints,
            historical_window=historical_window)


    # GIA
    n = len(stations)

    model = pm.modelcontext(None)

    if data is not None:
        # NOTE: Here the test is less about data and more about whether it is designed for use in posterior predictive sampling,
        # and thus whether it is present in the trace... Currently we pass the data argument in that case. Should be more specific in the future.

        # Define directly the end variable as it if were a RV, but in reality it will be taken from trace in the posterior predictive sampling
        gia_rad = pm.Flat("gia_rad", dims="station")
        gia_rsl = pm.Flat("gia_rsl", dims="station")


    else:
        if not gia_eof:
            vlm, rsl, likelihood = get_gia_ensemble(lons, lats)  # each of vlm, rsl is location x sample

            # compute the mean, covariance, and cholesky matrix of the combined [vlm, rsl]
            gia_ens_vlm_rsl = np.array([vlm, rsl]).reshape(2 * n, 5000)
            gia_mean = np.sum(gia_ens_vlm_rsl*likelihood[None], axis=1)/np.sum(likelihood)
            gia_cov = np.cov(gia_ens_vlm_rsl, aweights=likelihood, ddof=0)
            regularization_epsilon = np.diag(np.zeros(2*n)+1e-6)
            gia_chol = np.linalg.cholesky(gia_cov + regularization_epsilon)

            ## Only for the record (checks and debugs)
            pm.ConstantData("gia_rad_mean", gia_mean[:n], dims="station")
            pm.ConstantData("gia_rsl_mean", gia_mean[n:], dims="station")
            gia_sd = np.diag(gia_cov)**.5
            pm.ConstantData("gia_rad_sd", gia_sd[:n], dims="station")
            pm.ConstantData("gia_rsl_sd", gia_sd[n:], dims="station")


            gia_station = np.array([f"{v}_{id}" for v in ["vlm", "rsl"] for id in model.coords['station']])
            if "gia_station" not in model.coords:
                model.add_coord("gia_station", gia_station)
            else:
                assert len(model.coords["gia_station"]) == gia_station.size
                assert np.all(model.coords["gia_station"] == gia_station)

            # save as constant data for reference and later re-use
            # model.add_coord("co_gia_station", model.coords['gia_station'])
            # In posterior sampling we don't actually use it, so no need to save it
            # pm.ConstantData("gia_mean_vlm_rsl", gia_mean, dims="gia_station")
            # pm.ConstantData("gia_chol_vlm_rsl", gia_chol, dims=("gia_station", "co_gia_station"))
            # pm.ConstantData("gia_chol_rad_rsl", gia_chol, dims=("gia_station", "co_gia_station"))

            # Now sample from it in a decentered way (supposed to improve convergence compared to sampling from MVNormal directly)
            gia_scale_iid = pm.Normal("gia_scale_iid", dims=("gia_station"))
            gia_rad_rsl = gia_mean + gia_chol @ gia_scale_iid

        else:

            gia_eof = get_gia_ensemble_reduced_space(lons, lats, truncate=gia_eof_num)  # each of vlm, rsl is location x sample

            ## Only for the record (checks and debugs)
            pm.ConstantData("gia_rad_mean", gia_eof.mean[0], dims="station")
            pm.ConstantData("gia_rsl_mean", gia_eof.mean[1], dims="station")
            gia_sd = (gia_eof.sVh**2).sum(axis=0)**.5
            pm.ConstantData("gia_rad_sd", gia_sd[0], dims="station")
            pm.ConstantData("gia_rsl_sd", gia_sd[1], dims="station")
            # pm.ConstantData("gia_chol_rad_rsl", gia_chol, dims=("gia_station", "co_gia_station"))

            # Now sample from it in a decentered way (supposed to improve convergence compared to sampling from MVNormal directly)
            if "gia_reduced_space" not in model.coords:
                model.add_coord("gia_reduced_space", np.arange(gia_eof.S.size))

            gia_scale_iid = pm.Normal("gia_scale_iid_r", dims=("gia_reduced_space"))
            gia_rad_rsl = gia_eof.mean + pt.sum(gia_scale_iid[:, None, None] * gia_eof.sVh, axis=0)
            # print('gia_rad_rsl shape', gia_rad_rsl.shape.eval())
            # logger.warning(f'GIA RAD RSL SHAPE {gia_rad_rsl.shape.eval()}')


        # Now split into VLM and RSL
        gia_rad = pm.Deterministic("gia_rad", gia_rad_rsl[0], dims="station")
        gia_rsl = pm.Deterministic("gia_rsl", gia_rad_rsl[1], dims="station")

    # Use a normal prior for VLM residual, and later constrain it with Hammond et al 2021 GPS observations
    # Frederikse et al's (2020) VLM residual estimates range from -6 to 15 mm / yr, so 10 mm /yr s.d. should cover all cases.
    coords = pm.modelcontext(None).coords
    shp = tuple(len(coords[dim]) for dim in ["experiment", "year", "station"])

    vlm_res = calculate_vlm_residual(model=model, **vlm_kwargs) + pt.zeros(shp)

    vlm_rad = pm.Deterministic("vlm_rad", gia_rad + vlm_res, dims=("experiment", "year", "station"))

    # minus sign to vlm_res because VLM is expressed as RAD
    vlm_rsl = pm.Deterministic("vlm_rsl", gia_rsl - vlm_res, dims=("experiment", "year", "station")) # shape == (None, None, stations) ==> (stations)

    # RSL["vlm"] = vlm_rsl[None, None]
    # RAD["vlm"] = vlm_rad[None, None]

    RSL["vlm"] = vlm_rsl + pt.zeros(shp)
    RAD["vlm"] = vlm_rad + pt.zeros(shp)

    # Sum-up everything
    RSL["total"] = sum(RSL.values())
    RAD["total"] = sum(RAD.values())

    # add other fields for diagnostic
    RSL["vlm_res"] = -vlm_res + pt.zeros(shp)
    RSL["gia"] = gia_rsl[None, None] + pt.zeros(shp)
    RAD["vlm_res"] = vlm_res + pt.zeros(shp)
    RAD["gia"] = gia_rad[None, None] + pt.zeros(shp)

    return RSL, RAD



######
def _make_indices(idx, axis, ndim=None):
    indices = [slice(None) for x in range(ndim or axis)]
    indices[axis] = idx
    return tuple(indices)

def prepare_output_timeseries(x, axis=1, cumul=False, year_frequency=10):
    """Write down every 10 year, and extend to 2100
    """

    thin = _make_indices(slice(None, None, year_frequency), axis, x.ndim)
    last = _make_indices(slice(-1, None, None), axis, x.ndim)

    if cumul:
        last_rate = x[last] # rate
        x = x.cumsum(axis=axis)
        model = pm.modelcontext(None)
        i1, i2  = np.searchsorted(model.coords['year'], [1995, 2014])
        ref = _make_indices(slice(i1, i2+1), axis, x.ndim)
        x -= x[ref].mean(axis=axis)[_make_indices(None, axis, x.ndim)]
        one_step_forward = x[last] + last_rate
    else:
        one_step_forward = x[last]

    if year_frequency == 1:
        return x[thin]
    else:
        return pt.concatenate([x, one_step_forward], axis=axis)[thin]


def make_diagnostic(model=None,
    save_timeseries=True,
    diags=DIAGS,
    fields=FIELDS,
    sources=SOURCES,
    experiments=None,
    year_frequency=10,
    verbose=False,
    ):
    """ output diagnostic
    """
    model = pm.modelcontext(model)
    data = model._save

    if experiments is not None:
        experiment_dim = "experiment_output"
        if "experiment_output" not in model.coords:
            model.add_coord(experiment_dim, experiments)
        i_x = np.array([model.coords['experiment'].index(experiment) for experiment in experiments])
        data = {k: data[k][i_x] for k in data}
    else:
        experiment_dim = "experiment"

    i1, i2, i3, i4  = np.searchsorted(model.coords['year'], [1995, 2014, 2081, 2099])
    i5, i6  = np.searchsorted(model.coords['year'], [1993, 2019])
    i1900, i1990, i2050  = np.searchsorted(model.coords['year'], [1900, 1990, 2050])

    def proj2100(x):
        """Projections by 2100 for each source, each scenario"""
        y = x.cumsum(axis=1)
        one_step_forward = x[:, i4] # rate in 2099
        return (y[:, i4] + one_step_forward - y[:, i1:i2+1].mean(axis=1))

    def projYear(x, i):
        """Projections by 2100 for each source, each scenario"""
        y = x.cumsum(axis=1)
        return (y[:, i] - y[:, i1:i2+1].mean(axis=1))

    def slr_between(x, i1, i2):
        """Projections by 2100 for each source, each scenario"""
        y = x.cumsum(axis=1)
        return y[:, i2] - y[:, i1]

    if save_timeseries and year_frequency > 1:
        year_dim = "year_output"
        if year_dim not in model.coords:
            model.add_coord(year_dim, np.arange(1900, 2100+1, year_frequency))
    else:
        year_dim = "year"

    diag_funcs = {
        "proj2100": (proj2100, ()),
        "rate2100": (lambda x: x[:, i4], ()),
        "proj2050": (lambda x: projYear(x, i2050), ()),
        "rate2050": (lambda x: x[:, i2050], ()),
        "rate2000": (lambda x: x[:, i5:i6+1].mean(axis=1),  ()), # Rate during satellite period
        "past20c": (lambda x: slr_between(x, i1900, i1990), ()), # past 20th century
        "change": (partial(prepare_output_timeseries, cumul=True, year_frequency=year_frequency), (year_dim,)),
        "rate": (partial(prepare_output_timeseries, cumul=False, year_frequency=year_frequency), (year_dim,)),
        }

    if not save_timeseries:
        diags = [diag for diag in diags if year_dim not in diag_funcs[diag][1]]

    var_names = []

    for source in sources:
        for field in fields:
            for diag in diags:
                if field not in data:
                    if verbose: logger.warning(f"{field} not in data. Skip.")
                    continue
                tensor_field = data[field]
                if source not in tensor_field:
                    if verbose: logger.warning(f"{source} not in data. Skip.")
                    continue
                tensor = tensor_field.get(source, pt.zeros(tensor_field['total'].shape))
                if source == "tas":
                    tensor = pt.concatenate([tensor[:, [0]], pt.diff(tensor, axis=1)], axis=1)  # rate format ...
                func, time_dim = diag_funcs[diag]
                station_dim = () if field == "global" else ("station",)
                dims = (experiment_dim,) + time_dim + station_dim
                var_name = f"{diag}_{field}_{source}"
                var_names.append(var_name)
                if hasattr(model, var_name):
                    if verbose: logger.warning(f"!{var_name} already defined, skip")
                    continue
                pm.Deterministic(var_name, func(tensor), dims=dims)

    return var_names

######

def _define_model_constants(stations, model=None, data=None):
    if data is not None:
        define_all_from_data(data, skip_dims=['gia_station'])
        return

    model = pm.modelcontext(model)
    if "station" not in model.coords:
        model.add_coord("station", [r['ID'] for r in stations])

    n = len(stations)
    lons, lats = np.array([(station["Longitude"], station["Latitude"]) for station in stations]).T
    pm.ConstantData("lons", lons, dims="station")
    pm.ConstantData("lats", lats, dims="station")
    # The stations don't have to be PSMSL (can be simple coordinates, but if they are, the ID must correspond PSMSL)
    station_ids = np.array([station["ID"] for station in stations])
    pm.ConstantData("station_ids", station_ids, dims="station")

    # Also document psmsl_ids for back compatibility
    # station_ids = np.array([station["ID"] for station in stations])
    is_psmsl = np.array(["PSMSL IDs" in station for station in stations])
    # assert len(stations) == len(station_ids)
    psmsl_ids = np.where(is_psmsl, station_ids, 0)
    pm.ConstantData("psmsl_ids", psmsl_ids, dims="station")


def _slr_model_tidegauges_definitions(X, stations, slr_kwargs={},
    constraints=None, data=None):
    """
    X: dict of tensors with predictors (tas...) and global SLR terms
    """
    # keep these definitions for later use with full time-series output
    model = pm.modelcontext(None)

    # Save tas to trace !
    if not hasattr(model, "tas"):
        try:
            pm.Deterministic("tas", X["tas"], dims=("experiment", "year"))
        except:
            pm.ConstantData("tas", X["tas"], dims=("experiment", "year"))

    _define_model_constants(stations, model, data)

    RSL, RAD = calculate_regional_slr(X, stations, data=data, **slr_kwargs)

    GSL = {k:RSL.get(k,0)+RAD.get(k,0) for k in RSL}

    model._save = {"global": X, "gsl": GSL, "rad": RAD, "rsl": RSL, "stations": stations}

    if constraints is None:
        constraints = []

    logger.info(f"Full constraints: {', '.join([str(c) for c in constraints if not getattr(c, 'skip_likelihood', None)])}")
    logger.info(f"Diagnostic constraints: {', '.join([str(c) for c in constraints if getattr(c, 'skip_likelihood', None)])}")
    for c in constraints:
        logger.info(f"Apply constraint {str(c)}" + (" (diagnostic only)" if getattr(c, 'skip_likelihood', None) else ""))
        c.apply_model(model)

    return RSL, RAD, GSL


def slr_model_tidegauges(stations, experiments=["ssp126_mu", "ssp585_mu"], diag_kwargs={}, global_slr_kwargs={}, data=None, **kwargs):

    if "glacier_region" in kwargs:
        global_slr_kwargs["glacier_region"] = kwargs["glacier_region"]

    with slr_model_global(experiments=experiments, save_timeseries=False, data=data, **global_slr_kwargs) as model:
        # model.add_coord("station", np.arange(len(stations)))
        model.add_coord("station", [r['ID'] for r in stations])
        X = model._X
        Y = model._Y

        if stations:
            _slr_model_tidegauges_definitions({**X, **Y}, stations, data=data, **kwargs)
        else:
            model._save = {"global": {**X, **Y}, "stations": stations}

        make_diagnostic(model, **diag_kwargs)


    return model


def ravel_trace(trace, var_names):
    """ flatten draw x chain dimensions and return a dict"""

    # if trace from posterior predictive, chain was already merged in
    if hasattr(trace, 'posterior_predictive'):
        return {k: trace.posterior_predictive[k].values for k in var_names}

    return {k:np.concatenate([trace.posterior[k][chain].values for chain in trace.posterior.chain.values], axis=0)
            for k in var_names}


def slr_model_tidegauges_given_global_posterior(stations, global_trace, experiments=None, **kwargs):
    """
    global_trace
    experiments: can only contain experiments already contained in global_trace
    """
    if experiments is None:
        experiments = global_trace.posterior.experiment.values.tolist()

    model = pm.Model(coords={
            "year": np.arange(1900, 2099+1),
            "experiment": experiments,
            "station": np.arange(len(stations)),
            # tidegauge_record dimension is added in the apply_tidegauge_annual_constraints function
            })

    # Define global model in a separate, unused model context, just to retrieve model's parameter names and save them to trace
    with pm.Model(coords={k:model.coords[k] for k in ["year", "experiment"]}) as global_model: # dummy model
        XX = define_predictors(experiments=experiments)
        _ = calculate_global_slr(XX, save_timeseries=False) # only used to define the model parameters, and to retrieve their names
        param_names = [v.name for v in global_model.free_RVs]


    global_sources = ["steric", "glacier", "gis", "ais", "landwater", "total"]

    post = {k: pytensor.shared(v) for k, v in
        ravel_trace(global_trace.sel(experiment=experiments), ["tas"] + global_sources + param_names).items() }

    with model:

        # Index to sample from global trace
        n_global = global_trace.posterior.draw.size * global_trace.posterior.chain.size
        i = pm.Categorical('global_ensemble_i', np.ones(n_global)/n_global)

        XY = {source: post[source][i] for source in post}
        tas = XY["tas"]
        XY["tas2"] = tas**2

        # Add parameter names to trace
        for name in param_names:
            pm.Deterministic(name, post[name][i])

        model._global_RVs = [i]
        model._n_global = n_global

        _slr_model_tidegauges_definitions(XY, stations, **kwargs)

    return model