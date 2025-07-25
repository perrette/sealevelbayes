import numpy as np
import pandas as pd
import xarray as xa
from statsmodels.api import GLSAR, add_constant
from sealevelbayes.datasets import get_datapath

def estimate_tidegauge_error(data, threshold=0.1):
    """Estimate the variance of a tide gauge dataset.

    Args:
        data (pandas.DataFrame): The tide gauge dataset.

    Returns:
        1-D array (float): The estimated measurement error.

    At the moment load a pre-calculated estimate of the measurement error.
    TODO - make the calculations on-the-fly (check tidegauge_trends.ipynb notebook for the calculations)
    """
    if threshold != 0.1:
        raise NotImplementedError("Only threshold=0.1 is supported at the moment.")

    # Load the tide gauge and satellite error estimate calculated in tidegauge_trends.ipynb notebook (residual from linear trend fitted to the data)
    with xa.open_dataset(get_datapath('savedwork/measurement_error.nc')) as ds:
        measurement_error = ds['tg_measurement_error'].sel(station=data.columns).values

    return measurement_error


def get_tidegauge_to_satellite_residual_ratio(data):
    with xa.open_dataset(get_datapath('savedwork/measurement_error.nc')) as ds:
        return ds['error_ratio'].sel(station=data.columns).values


def _fit_glsar_model(tidegauge_clean):
    return GLSAR(tidegauge_clean.values, exog=add_constant(tidegauge_clean.index.values)).iterative_fit()


def _calculate_trend_and_residual_errors(tidegauge):
    tidegauge_clean = tidegauge.dropna()
    res = _fit_glsar_model(tidegauge_clean)
    return res.params[1], res.cov_params()[1, 1]**.5, (tidegauge_clean - res.fittedvalues).std()


def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def calculate_trend_and_residual_errors(df, attrs={}):
    trends = nans(df.shape[1])
    trend_errors = nans(df.shape[1])
    res_sds = nans(df.shape[1])

    for i in range(df.shape[1]):
        tidegauge = df.iloc[:, i]
        try:
            trend, trend_error, res_sd = _calculate_trend_and_residual_errors(tidegauge)
        except Exception as error:
            print(f"Error for series {i}: {error}")
            continue
        trends[i] = trend
        trend_errors[i] = trend_error
        res_sds[i] = res_sd

    return xa.Dataset({'trends': ('station', trends), 'trend_errors': ('station', trend_errors), 'res_sds': ('station', res_sds)}, coords={'station': df.columns}, attrs=attrs)


def detrend_records(df):
    residuals = nans(df.shape)

    for i in range(df.shape[1]):
        tidegauge = df.iloc[:, i]
        tidegauge_clean = tidegauge.dropna()
        try:
            fit = _fit_glsar_model(tidegauge_clean)
        except Exception as error:
            print(f"Error for series {i}: {error}")
            continue
        residuals[:, i] = pd.Series(fit.resid, index=tidegauge_clean.index).reindex(tidegauge.index).values

    return pd.DataFrame(residuals, index=df.index, columns=df.columns)


def estimate_tg_measurement_error(sat_oceandyn_sd, annual_residual_tg_to_sat_ratio, min_measurement_sd=0.1):
    """
    Estimate the measurement error at tide gauge locations.

    Parameters
    ----------
    ocean_dyn_sd: 1-d array
        The standard deviation of the variance-corrected CMIP6 ensemble at tide-gauge locations, scaled by satellite annual variance
    annual_residual_tg_to_sat_ratio: 1-d array
        Ratio of the annual residual tide gauge variance to the satellite annual variance
    min_measurement_sd: float, 0.1 by default

    Returns
    -------
    measurement_sd: 1-d array

    Background
    ----------
    The purpose of this function is to return a term that can be added to the
    diagonal of the covariance matrix of the tide gauge data, which was derived after scaling
    CMIP6 model with detrended annual satellite variance. So here we first estimate the total variance
    by rescaling the diagonal of that matrix with the ratio of tidegauge to satellite annual variance,
    and we consider any positive difference with open ocean satellite-derived variance (ratio > 1)
    as a "measurement error", i.e. not correlated with the larger-scale ocean dynamics.
    We then calculate which measurement error is required to matches the scaled diagonal under the
    above-stated conditions. We ignore any measurement error that is smaller than the minimum
    measurement error of 0.1 mm/yr, or that would imply negative measurement error (i.e. when the
    tide-gauge variance is smaller than satellite variance.)
    """
    error_ratio = annual_residual_tg_to_sat_ratio
    total_sd = sat_oceandyn_sd * error_ratio
    measurement_sd = np.where(error_ratio > 1, (total_sd**2 - sat_oceandyn_sd**2)**.5, min_measurement_sd)
    measurement_sd[measurement_sd < min_measurement_sd] = min_measurement_sd
    return measurement_sd