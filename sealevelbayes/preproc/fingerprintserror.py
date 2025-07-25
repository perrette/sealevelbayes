"""This module aims to quantify the mass fingerprint error in order and factor it in the GPS error model.
"""
from datetime import datetime
from scipy.signal import savgol_filter
import numpy as np
import arviz
import xarray as xa
from sealevelbayes.logs import logger
from sealevelbayes.config import get_runpath
import sealevelbayes.datasets.frederikse2020 as frederikse2020
from sealevelbayes.preproc.massfingerprints import calc_globalmean, _calc_globalmean, _calc_fingerprint
from sealevelbayes.datasets.hammond2021 import get_gps_sampling_period
from sealevelbayes.preproc.linalg import calc_lineartrend_fast
from sealevelbayes.datasets.maptools import interpolate

def calc_annual_model_data(lon, lat, time, data, gmsl=None, smoothing=None, fingerprint_period=(2000, 2018)):
    """Calculate the annual model data for a given variable.

    Parameters:
        lon:
        lat:
        time (np.array): The time array
        data (np.array): time x lat x lon
          array of obs data (e.g. tws_rad by Frederikse et al 2020)
        gmsl (np.array, optional): The global-mean sea level data to use to calculate the fingerprints.
        If not provided, it will be calculated from the data.
        In the case of VLM (rad component) it must be provided because the global mean VLM is zero.
        gmsl_smoothing (int, optional): Smoothing time-scale for gmsl when multiplying by the fingerprint. Defaults to no smoothing
        fingerprint_period (tuple, optional): The period to calculate the fingerprint over. Defaults to (2000, 2018).

    Returns:
        hindcast (time, lat, lon): The annual model data in the form (smoothed) global-mean * fingerprint
        gmsl (time): global-mean sea level
        fingerprint (lat, lon): local fingerprint

    The fingerprints are calculated as a regression of y = local - global_mean(local) versus x = gmsl, where gmsl defaults to global mean of
    local (valid for RSL fingerprints). The result is multiplied by (possibly smoothed) gmsl to get the annual model data.
    """
    data_mean = _calc_globalmean(lon, lat, data)
    if gmsl is None:
        gmsl = data_mean
    # data_no_mean = data - data_mean[..., None, None]

    if fingerprint_period is None:
        # return data, None, None
        _finger = np.diff(data, axis=0) / np.diff(gmsl, axis=0)[:, None, None]
    else:
        time_idx = _get_index(time, fingerprint_period)
        _finger = _calc_fingerprint(gmsl[time_idx], data[time_idx])
    if smoothing:
        gmsl = savgol_filter(gmsl, smoothing, polyorder=1)

    recons = np.cumsum(_finger * np.diff(gmsl)[:, None, None], axis=0)
    recons = np.concatenate([recons[[0]], recons], axis=0)
    recons = recons - recons.mean(axis=0) + data.mean(axis=0)
    return recons, gmsl, _finger
    # return _finger * gmsl[:, None, None], gmsl, _finger


def calc_covariance_matrix(model_data, obs_data, time, time_slices, obs_time_slices=None, period=None, domain=None, psmsl_ids=None):
    """Calculate the covariance matrix of the data.

    Parameters:
        model_data (time, location): model data (annual)
        obs_data (time, location): obs data (annual)
        time (np.array): The time array
        time_slices (list of (start, end) tuple indices): The time slices to calculate the model trend over.
        obs_time_slices (list of (start, end) tuple indices, optional):
            The time slices to calculate the trend over for the obs data. If not provided, it defaults to the model time slices.
        period (tuple, optional): The overall period to calculate the covariance over. Defaults to the whole.

    Returns:
        cov (location, location): The covariance matrix

    Here we determine the minium size of time window from the time_slices and obs_time_slices, and lets this window slides
    through the period of calculation. For each time window, we calculate the linear trends for both model and obs data as if
    that window would represent the most recent period from which GPS measurements were taken. We then calculate the covariance
    matrix of the difference between the model and obs trends.
    """
    if obs_time_slices is None:
        obs_time_slices = time_slices

    if psmsl_ids is None:
        psmsl_ids = np.arange(model_data.shape[1])

    if domain is not None:
        domain = np.asarray(domain)
        assert len(domain) == len(time_slices), "Domain must be the same length as time_slices"
        time_slices = [t for t, d in zip(time_slices, domain) if d]
        obs_time_slices = [t for t, d in zip(obs_time_slices, domain) if d]
        model_data = model_data[:, domain]
        obs_data = obs_data[:, domain]

    _model_window = (min(y1 for (y1, y2) in time_slices), max(y2 for (y1, y2) in time_slices))
    _gps_window = (min(y1 for (y1, y2) in obs_time_slices), max(y2 for (y1, y2) in obs_time_slices))
    window = (min(_model_window[0], _gps_window[0]), max(_model_window[1], _gps_window[1]))
    window = (int(np.ceil(window[0])), int(np.floor(window[1])))

    if period is not None:
        assert period[0] <= window[0], "Overall period must be at least as long as the sampling window"
        assert period[1] >= window[1], "Overall period must be at least as long as the sampling window"
        idx = _get_index(time, period)
        time = time[idx]
        model_data = model_data[idx]
        obs_data = obs_data[idx]

    window_len = window[1] - window[0] + 1
    period_len = time.size
    n_windows = period_len - window_len + 1

    logger.info(f"Generate {n_windows} samples samples...")
    timer = datetime.now()
    trend_errors = np.empty((n_windows, model_data.shape[1]))
    model_trends = np.empty((n_windows, model_data.shape[1]))
    obs_trends = np.empty((n_windows, model_data.shape[1]))
    for i in range(n_windows):
        offset = i + window[1] - time[-1]  # usually 0, but if the gps window does not go all the way to the end of the period, we can also move forward in time
        model_trends[i] = np.array([calc_lineartrend_fast(model_data[_get_index(time+offset, period), k]) for k, period in enumerate(time_slices)])
        obs_trends[i] = np.array([calc_lineartrend_fast(obs_data[_get_index(time+offset, period), k]) for k, period in enumerate(obs_time_slices)])
    trend_errors = model_trends - obs_trends
    logger.info(f"Generate {n_windows} samples samples...done (timer: {datetime.now() - timer})")

    bad_years = np.isnan(trend_errors).all(axis=1)

    if bad_years.any():
        logger.warning(f"Found {bad_years.sum()} bad years in the covariance matrix calculation. Skip them: {', '.join(map(time[bad_years].tolist(), str))}")
        trend_errors = trend_errors[~bad_years]

    bad_years_somewhere = np.isnan(trend_errors).any(axis=1)
    if bad_years_somewhere.any():
        logger.warning(f"Found {bad_years_somewhere.sum()} bad years in the covariance matrix calculation in some places. Skip them: {', '.join(map(time[bad_years_somewhere].tolist(), str))}")
        trend_errors = trend_errors[~bad_years_somewhere]

    assert not np.isnan(trend_errors).any(), "There are still NaNs in the trend errors"

    timer = datetime.now()
    logger.info(f"Compute statistics over {n_windows} samples...")
    mean_error = np.mean(trend_errors, axis=0)
    std_error = np.std(trend_errors, axis=0)
    rms_error = np.sqrt(np.mean(trend_errors**2, axis=0))
    cov_error = np.cov(trend_errors.T)
    cov_error_rms = trend_errors.T @ trend_errors / (n_windows - 1)
    logger.info(f"Compute statistics over {n_windows} samples...done. (timer: {datetime.now() - timer})")

    ds = xa.Dataset(coords={"station": psmsl_ids, "co_station": psmsl_ids, "sample": np.arange(n_windows)})
    for k,v in {
        "mean": mean_error,
        "std": std_error,
        "rms": rms_error,
        "cov": cov_error,
        "cov_rms": cov_error_rms,
        }.items():
        if v.ndim == 2:
            assert np.isfinite(v).all(), f"Found NaNs in {k}"
            v2 = np.empty((psmsl_ids.size, psmsl_ids.size))
            v2.fill(np.nan)
            for j, i in enumerate(np.where(domain)[0]):
                v2[i, domain] = v[j]
            assert np.isfinite(v2[domain][:, domain]).all(), f"Found NaNs in {k}. Domain filling failed"
            assert not np.isnan(v2[domain][:, domain]).any(), f"Found NaNs in {k} after domain filling"
            ds[k] = ("station", "co_station"), v2
            assert not np.isnan(ds[k][domain, domain].values).any(), f"Found NaNs in {k} after passing to dataset"
            assert not np.isnan(ds[k].values).all()
        else:
            v2 = np.empty(psmsl_ids.size)
            v2.fill(np.nan)
            v2[domain] = v
            ds[k] = ("station",), v2

    for k, v in [("model_trends_samples", model_trends), ("obs_trends_samples", obs_trends), ("trend_errors_samples", trend_errors)]:
        v2 = np.empty((n_windows, psmsl_ids.size))
        v2.fill(np.nan)
        v2[:, domain] = v
        ds[k] = ("sample", "station"), v2

    assert not np.isnan(ds["cov"].values).all()
    return ds

def _get_index(time, period):
    if period is None:
        return slice(None)
    return (time >= period[0]) & (time <= period[1])


def load_data(variable="tws", fingerpring_period=(2000, 2018),
                smoothing=None, psmsl_ids=None, field="rad"):
    """Load data for the fingerprint error calculation.
    """
    if psmsl_ids is None:
        with xa.open_dataset(get_runpath("ff49c61/run_5.9.0-py3.11.10-target-0.95")/"trace.nc", group="constant_data") as constant_data:
            psmsl_ids = constant_data.psmsl_ids.values
            lons = constant_data.lons.values
            lats = constant_data.lats.values
    else:
        raise NotImplementedError("Only the default psmsl_ids=None is supported at the moment")

    # with xa.open_dataset(frederikse2020.root / f"{variable}.nc") as ds:
    with xa.open_dataset(frederikse2020.root / f"{variable}.nc") as ds:
        data = ds[f"{variable}_{field}_mean"].values
        time = ds.time.values
        lat = ds.lat.values
        lon = ds.lon.values
        gmsl = calc_globalmean(ds[f"{variable}_rsl_mean"].load())
        idx = np.isfinite(gmsl.values)
        data = data[idx]
        time = time[idx]
        gmsl = gmsl[idx]

    model_data, _, _ = calc_annual_model_data(lon, lat, time, data, gmsl.values, fingerprint_period=fingerpring_period, smoothing=smoothing)

    sampled_model_data = interpolate(lon, lat, model_data.transpose(1,2,0), lons, lats, mask=np.isnan(model_data[0])).T
    sampled_obs_data = interpolate(lon, lat, data.transpose(1,2,0), lons, lats, mask=np.isnan(data).any(axis=0)).T

    ds = xa.Dataset(coords={"time": time, "station": psmsl_ids})
    ds["model"] = ("time", "station"), sampled_model_data
    ds["obs"] = ("time", "station"), sampled_obs_data
    ds["lons"] = ("station",), lons
    ds["lats"] = ("station",), lats

    return ds

def filter_data(data, high_pass, inplace=False, polyorder=1):
    # Apply high-pass filter to model and obs data prior to computing the differences
    if high_pass is None:
        return data
    if not inplace:
        data = data.copy()
    if data.ndim == 1:
        data[:] -= savgol_filter(data, high_pass, polyorder=polyorder)
        return data

    for i in range(data.shape[1]):
        data[:, i] -= savgol_filter(data[:, i], high_pass, polyorder=polyorder)

    return data

def make_fingerprint_error_diag(data,
                                gps_period=(2000, 2018),
                                high_pass=None,
                                ):
    """Main function to calculate the fingerprint error.

    Parameters:
    data (xa.Dataset): The data to use for the fingerprint error calculation (returned by load_data)
    fingerpring_period (tuple, optional): The period to calculate the fingerprint over. Defaults to (2000, 2018).

    - Load the obs data for tws_rad (later we may add AIS_rad, GrIS_rad and perhaps also the rsl counterparts)
    - Calculate the annual model data (calc_annual_model_data)
    - Sample the obs data and the model data at the GPS locations
    - First method: Calculate the linear trends at the GPS locations for both model and obs, and returns the error (calc_linear_trends)
    - Second method: Calculate the covariance matrix of the error (calc_covariance_matrix)
    """

    sampled_model_data = data.model.values.copy()
    sampled_obs_data = data.obs.values.copy()
    time = data.time.values
    psmsl_ids = data.station.values
    periods = get_gps_sampling_period(psmsl_ids)

    # Apply high-pass filter to model and obs data prior to computing the differences
    if high_pass is not None:
        for i in range(sampled_model_data.shape[1]):
            filter_data(sampled_model_data[:, i], high_pass, inplace=True)
            filter_data(sampled_obs_data[:, i], high_pass, inplace=True)

    if gps_period is None:
        model_trends = np.array([calc_lineartrend_fast(sampled_model_data[_get_index(time, period), k]) if period is not None else np.nan for k, period in enumerate(periods)])
    else:
        model_trends = calc_lineartrend_fast(sampled_model_data[_get_index(time, gps_period)])

    obs_trends = np.array([calc_lineartrend_fast(sampled_obs_data[_get_index(time, period), k]) if period is not None else np.nan for k, period in enumerate(periods)])

    # Now compute the cov matrix
    results = calc_covariance_matrix(sampled_model_data, sampled_obs_data, time,
                           time_slices=[gps_period for _ in range(len(periods))] if gps_period is not None else periods, # model time slices
                           obs_time_slices=periods, domain=[p is not None for p in periods], psmsl_ids=psmsl_ids)

    results["model_trends"] = "station", model_trends
    results["obs_trends"] = "station", obs_trends

    return results