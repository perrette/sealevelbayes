"""Models used to fit various sea level components

They are made to work with pytensor.
"""
import numpy as np
import xarray as xa
from sealevelbayes.logs import logger
from sealevelbayes.datasets import get_datapath
from sealevelbayes.datasets.ar6.supp import open_slr_global
import sealevelbayes.datasets.frederikse2020 as frederikse2020
from sealevelbayes.datasets.shared import MAP_FRED_NC, MAP_AR6


sources = ["steric", "glacier", "gis", "ais", "landwater", "vlm"]

def get_fred_samples(source, smooth=False):
    if smooth:
        from scipy.signal import savgol_filter

    with xa.open_dataset(frederikse2020.root / "GMSL_ensembles.nc") as fred:
        tws = fred[MAP_FRED_NC.get(source, source)]
        if smooth:
            tws = xa.DataArray(np.array([savgol_filter(y, 5, 1) for y in tws.values]), coords=fred.coords)
        fred_rates_xa = tws.diff(dim='time')
        fred_rates = fred_rates_xa.values
        # fred_years = fred_rates_xa.time.values
        fred_years = tws.time.values # that's one year longer than diff above, but the diff is extended below
        if smooth:
            fred_rates = np.array([savgol_filter(y, 5, 1) for y in fred_rates])

    fred_rates = np.concatenate([fred_rates[:, :1], fred_rates], axis=1) # set the same rate in 1900 as in 1901, for the sake of simplicity
    return fred_years, fred_rates

def get_fred_mvnormal(source, smooth=False):
    fred_years, fred_rates = get_fred_samples(source, smooth=smooth)

    cov_fred = np.cov(fred_rates.T)
    cov_fred[0, 1:] = cov_fred[1:, 0] = 0 # cancel correlation with first value, otherwise issues with cholesky decomposition

    mu_fred = np.mean(fred_rates, axis=0)

    return fred_years, mu_fred, cov_fred

def _merge_icesheets(source, experiment, years, icesheets):
    years = None
    values = None
    for icesheet in icesheets:
        years_, values_ = get_ar6_samples(source, experiment, years=years, icesheet=icesheet)
        if years is None:
            years = years_
            values = values_
        else:
            assert np.all(years == years_), "Years do not match between icesheets"
            values = np.concatenate([values, values_], axis=0)
    return years, values


def get_ar6_samples(source, experiment, years=None, icesheet="ismipemu"):
    # Now AR6
    ar6source = MAP_AR6.get(source, source  )
    if source in ("ais", "gis"):
        if isinstance(icesheet, (list, tuple)):
            icesheets = icesheet
            return _merge_icesheets(source, experiment, years, icesheets)
        else:
            filename = f"ar6/global/full_sample_components_rates/icesheets-ipccar6-{icesheet}icesheet-{experiment}_{ar6source}_globalsl_rates.nc"
    elif source == "steric":
        filename = f"ar6/global/full_sample_components_rates/{ar6source}-tlm-{ar6source}-{experiment}_globalsl_rates.nc"
    elif source == "glacier":
        filename = f"ar6/global/full_sample_components_rates/{ar6source}-ipccar6-gmipemuglaciers-{experiment}_globalsl_rates.nc"
    else:
        filename = f"ar6/global/full_sample_components_rates/{ar6source}-ssp-{ar6source}-{experiment}_globalsl_rates.nc"

    with xa.load_dataset(get_datapath("zenodo-6382554-garner2021") / filename) as ds:
        future_rates = ds["sea_level_change_rate"].sel(years=slice(None, 2100)).squeeze()

    return future_rates.years.values, future_rates.values


def get_merged_fred_ar6_samples(source, experiment, smooth=None, ar6_kwargs={}):
    """ Model Fred and AR6 distribution as a single multivariate normal distribution

    smooth: like in get_merged_quantiles, if True both gmslr and its rate will be smoothed, prior to computing the covariance
    """
    fred_years, fred_rates = get_fred_samples(source, smooth=smooth)
    assert fred_years.size == fred_rates.shape[1]
    fred_end_year = fred_years[-1]
    assert fred_end_year == 2018, f"Unexpected end year {fred_end_year} for {source}"

    future_years_, future_rates_ = get_ar6_samples(source, experiment, **ar6_kwargs)
    ar6_start_year = future_years_[0]  # this depends on the source
    future_years = np.arange(2019, 2101)

    # interpolate to yearly data
    assert future_years_.size == future_rates_.shape[1]
    # future_rates = np.array([np.interp(future_years, future_years_, future_rates_[i]) for i in range(future_rates_.shape[0])])
        # np.array([np.interp(future_years, future_years_, future_rates_[i]) for i in range(future_rates_.shape[0])])
    # use xarray as they might possibly optimize the interpolation (i didn't check)
    future_rates = xa.DataArray(future_rates_, coords={"year": future_years_}, dims=['sample', 'year']).interp(year=future_years).values

    # random resampling to match fred and AR6 sample size
    # NOTE this breaks any correlation between the two datasets
    rng = np.random.default_rng(0)
    fred_indices = rng.integers(0, fred_rates.shape[0], size=future_rates.shape[0])  # trigger the random number generator
    fred_rates = fred_rates[fred_indices]
    # future_rates[:, 0] = (future_rates[:, 1] + fred_rates[fred_indices, -1])/2

    all_samples = np.concatenate([fred_rates, future_rates], axis=1)
    all_years = np.concatenate([fred_years, future_years])
    assert (np.diff(all_years) == 1).all(), "Non-continuous years in the merged dataset"

    # fill in gaps between Fred and AR6
    missing_years = (all_years > fred_end_year) & (all_years < ar6_start_year)
    if missing_years.any():
        logger.info(f"Filling {all_years[missing_years]} for {source} by interpolating between Fred's {fred_end_year} and AR6's {ar6_start_year} rates")
        all_samples[:, missing_years] = np.array([
            np.interp(all_years[missing_years], all_years[~missing_years], all_samples[i, ~missing_years])
            for i in range(all_samples.shape[0])])

    assert np.all(np.isfinite(all_samples)), f"Non-finite values in future rates for {source}"

    return all_years, all_samples


def get_merged_mvnormal(source, experiment, smooth=None, ar6_kwargs={}):

    all_years, all_samples = get_merged_fred_ar6_samples(source, experiment, smooth=smooth, ar6_kwargs=ar6_kwargs)
    mu_total = all_samples.mean(axis=0)

    cov_total = np.zeros((2100-1900+1, 2100-1900+1))
    cov_total_full = np.cov(all_samples.T)

    # set the intersection to zero (this should be already the case because of the random shuffling)
    n1 = 2018-1900+1
    cov_total[:n1, :n1] = cov_total_full[:n1, :n1]
    cov_total[n1:, n1:] = cov_total_full[n1:, n1:]

    cov_total += np.eye(cov_total.shape[0]) * 1e-6  # regularize

    return mu_total, cov_total
