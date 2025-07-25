# import netCDF4 as nc
import tqdm
import numpy as np
import scipy.signal
import xarray as xa
from pathlib import Path
from itertools import groupby, chain

from sealevelbayes.logs import logger
from sealevelbayes.preproc.linalg import detrend_timeseries, calc_lineartrend_fast
from sealevelbayes.datasets.tidegaugeobs import DEFAULT_VERSION, tg, tg_years
from sealevelbayes.datasets.satellite import get_satellite_timeseries


def _sample_along_sliding_window(values, nt, step=1):
    """
    values: nd-array with time as first dimension
    nt: length of sampling window
    """
    for k in range(0, values.shape[0]-nt+1, step):
        yield values[k:k+nt]


def model_record_to_dataset(model_record):

    ds = xa.Dataset(coords={
        "id": model_record['id'],
        "co_id": model_record['id'],
        "lag": model_record['acf_lags'],
        })

    ds.attrs["model"] = model_record['model']
    if 'picontrol_years' in model_record:
        ds.attrs["picontrol_years"] = model_record['picontrol_years']
    if "models" in model_record:
        ds.attrs["models"] = model_record['models']

    for k, v in model_record.items():
        if k.startswith('cov'):
            ds[k] = (('id', 'co_id'), v)
        elif k.startswith('zos'):
            ds[k] = ('id', v)
        elif k.startswith('acf') and k != "acf_lags":
            ds[k] = (('id','lag'), v)

    return ds


def pc_surrogates_arima(array, order, length=500):
    """ Create surrogate time-series for a time x samples array, based on EOF decomposition
    and ARIMA fitting & simulation of the PCs
    """
    import statsmodels.api as sm
    array = array - array.mean(axis=0)

    # Decomosition in spatial patterns (v) and time-series (or PC; v)
    u, d, v = np.linalg.svd(array)

    # d are the eigenvalues, of length = min(detended.shape) ~ the rank, here 27 (minus the tiny ones)
    # remove really small ones, does not contribute any
    d[d < d.max() / 1000] = 0

    # build the full matrix operation
    r = np.concatenate([np.diag(d), np.zeros((v.shape[0]-d.length, d.size))]).T
    np.testing.assert_allclose(array, u@r@v)

    # for each PC, fit an ARMA model and simulate a time-series of length 500
    pc_simu = np.empty((length, u.shape[0]))  # 500 samples of each of the PCs
    pc_simu.fill(0)

    for i, (uu, dd) in enumerate(zip(u.T, d)):
        if dd == 0:
            break
        model = sm.tsa.arima.ARIMA(uu, order=order)
        res = model.fit()
        pc_simu[:, i] = model.simulate(res.params, length)

    # recompose the full time-series
    return pc_simu@r@v


def create_surrogates_satellite_pc_arma(sat_values, length=500):
    """ Simulate satellite altimetry's principal components (from an EOF decomposition) using an ARMA(1, 1) model
    Here we use the satellite rate. Note that an ARMA(1,1) can serve as a model for the first derivative of an AR(1) model.

    => the variance looks far too large (see )

    sat_values: time x locations in mm
    """
    sat_rate = np.diff(sat_values, axis=0)
    surrogate_rates = pc_surrogates_arima(sat_rate, order=(1, 0, 1), length=length)
    return np.cumsum(surrogate_rates - surrogate_rates.mean(axis=0), axis=0)


def create_surrogates_satellite_pc_cumular1(sat_values, length=500):
    """ Same as create_surrogates_satellite_pc_arma but we conduct EOF on cumulative SLR and fit an AR(1) model to it.

    sat_values: time x locations in mm
    """
    return pc_surrogates_arima(sat_values, order=(1, 0, 0), length=length)


def create_surrogates_satellite_iid(sat_values, size=500, seed=324322):
    """ Create surrogates data only with the covariance matrix, no time structure

    sat_values: time x locations in mm
    """
    covsat = np.cov(sat_values.T)
    return np.random.default_rng(seed=seed).multivariate_normal(np.zeros(sat_values.shape[1]), cov=covsat, size=size)


def create_surrogates_satellite_ar1(sat_values, length=500, seed=97887, warmup=100):
    """ AR(1) based on diagnosed rho and innovation's covariance

    sat_values: time x locations in mm
    """
    rho = np.array([autocorr(x)[1] for x in sat_values.T])
    assert np.all((rho < 1) & (rho > -1))
    innov_ano = sat_values[1:] - sat_values[:-1]*rho[None]
    cov_ar1_innovation = np.cov(innov_ano.T)
    logger.info(f"Satellite surrogates ar1. Mean rho: {rho.mean():.2f}. Mean innov sd: {np.diag(cov_ar1_innovation).mean()**.5:.2f}")

    x = np.random.default_rng(seed=seed).multivariate_normal(np.zeros(sat_values.shape[1]), cov=cov_ar1_innovation, size=length+warmup)
    for i in range(1, length+warmup):
        x[i] += x[i-1]*rho

    return x[warmup:]


def create_surrogates_satellite_ar2(sat_values, length=500, seed=97887, warmup=100):
    """ AR(1) based on diagnosed rho and innovation's covariance

    sat_values: time x locations in mm
    """
    rho, rho2 = np.array([autocorr(x)[1:1+2] for x in sat_values.T]).T
    innov_ano = sat_values[2:] - sat_values[1:-1]*rho[None] - sat_values[:-2]*rho2[None]
    cov_ar2_innovation = np.cov(innov_ano.T)

    x = np.random.default_rng(seed=seed).multivariate_normal(np.zeros(sat_values.shape[1]), cov=cov_ar2_innovation, size=length+warmup)
    for i in range(2, length+warmup):
        x[i] += x[i-1]*rho + x[i-2]*rho2

    return x[warmup:]


def normalize_timeseries(zos, sat_values, rescale_like_satellite=True, detrend=True, keep_mean=False):

    # first make sure that sat_values is detrended !
    sat_values = detrend_timeseries(sat_values, keep_mean=keep_mean)  # detrended, time first

    if detrend:
        logger.debug('...detrend zos')
        zos_detrended = detrend_timeseries(zos, n=2, keep_mean=keep_mean)
    else:
        zos_detrended = zos - zos.mean(axis=0)  # make sure the mean is zero

    logger.debug('...calculate standard deviations')
    sat_std = np.std(sat_values, axis=0)
    zos_std = zos_detrended.std(axis=0)
    # zos_std_27yr = np.mean([vals.std(axis=0) for vals in _sample_along_sliding_window(zos_detrended, sat_values.shape[0])], axis=0)
    logger.debug("calculate zos_std_27yr after detrending each individual 27-yr window ...")
    zos_std_27yr = np.mean([detrend_timeseries(vals, keep_mean=keep_mean).std(axis=0) for vals in _sample_along_sliding_window(zos_detrended, sat_values.shape[0])], axis=0)
    logger.debug("calculate zos_std_27yr after detrending each individual 27-yr window ... done")

    # mean standard deviation over a 27-year period (similar to satellite time-series)
    scale = sat_std / zos_std_27yr

    if rescale_like_satellite:
        logger.debug('...rescale zos to match satellite variance')
        zos_detrended *= scale[None]

    info = {
        "scale": scale,
        "zos_std_unscaled": zos_std, # prior to scaling
        "zos_std_27yr_unscaled": zos_std_27yr, # prior to scaling
        "sat_std": sat_std,
        "zos_std": zos_detrended.std(axis=0), # after scaling
        # "version": version,
    }

    return zos_detrended, info


def autocorr(x):
    '''scipy.signal.correlate : https://stackoverflow.com/a/51168178/2192272

    return the autocorrelation coefficients (starts with 0-lag 1)

    (mean must be zero)
    '''
    return scipy.signal.correlate(x, x, 'full')[x.size-1:]/x.var()/x.size


class Sampler:
    def sample(self, observe):
        raise NotImplementedError()

    def samples(self, observe, size=1000):
        for i in range(size):
            yield self.sample(observe)

    def cov(self, samples=None):
        return np.cov(np.asarray(samples).T)

    def std(self, samples=None):
        return np.std(np.asarray(samples), axis=0)


class SlidingWindowSampler(Sampler):
    def __init__(self, surrogate_data, length=2023-1900+1, step=None):
        self.surrogate_data = surrogate_data  # time x space
        self.length = length
        min_samples = 50  # we keep models that allow for 50 samples
        max_samples = 200  #
        if step is None:
            step = max(1, (surrogate_data.shape[0] - length) // max_samples)
        self.step = step
        min_length = length + min_samples*step
        assert self.surrogate_data.shape[0] >= min_length, f"Surrogate data {self.surrogate_data.shape[0]}. Expected min length {min_length} to yield {min_samples} samples"

    def samples(self, observe):
        for zos_spacetime in _sample_along_sliding_window(self.surrogate_data, self.length, self.step):
            yield observe(zos_spacetime)



class SatelliteIIDSampler(SlidingWindowSampler):
    def __init__(self, sat_values, size=500, seed=324322, **kwargs):
        surrogates = create_surrogates_satellite_iid(detrend_timeseries(sat_values), size=size, seed=seed)
        super().__init__(surrogates, **kwargs)


class SatelliteAR1Sampler(SlidingWindowSampler):
    def __init__(self, sat_values, **kwargs):
        surrogates = create_surrogates_satellite_ar1(detrend_timeseries(sat_values))
        super().__init__(surrogates, **kwargs)


class SatelliteEOFAR1Sampler(SlidingWindowSampler):
    def __init__(self, sat_values, **kwargs):
        surrogates = create_surrogates_satellite_pc_cumular1(detrend_timeseries(sat_values))
        super().__init__(surrogates, **kwargs)


class CMIP6Sampler(SlidingWindowSampler):
    def __init__(self, zos, model=None, rescale_like_satellite=True, sat_values=None, detrend=True, **kwargs):
        zos_detrended, info = normalize_timeseries(zos, sat_values=sat_values, detrend=detrend, rescale_like_satellite=rescale_like_satellite)
        self.model = model
        super().__init__(zos_detrended, **kwargs)


class MeanSampler(Sampler):
    """Use a mean covariance matrix across samplers
    """
    def __init__(self, samplers):
        self.samplers = samplers

    def samples(self, observe):
        return [np.array(list(sampler.samples(observe))) for sampler in tqdm.tqdm(self.samplers)]

    def cov(self, samples):
        return np.mean([sampler.cov(samples_) for sampler, samples_ in zip(self.samplers, samples)], axis=0)

    def std(self, samples):
        return np.mean([sampler.std(samples_)**2 for sampler, samples_ in zip(self.samplers, samples)], axis=0)**.5


def calc_covariance_matrices(zos_detrended, sat_length, tg_masks, metadata={}):

    nlags = 10
    acf_lags = np.arange(nlags)

    # covariance in annual SLR rate
    logger.info('...covariance annual SLR rate')
    zos_rate = np.diff(zos_detrended, axis=0)
    cov_rate_annual = np.cov(zos_rate.T)

    logger.info('...auto-correlation annual SLR rate')
    # Now calculate autocorrelation
    zos_rate_demean = zos_rate - zos_rate.mean(axis=0)
    acf_rate_annual = np.array([autocorr(x)[:nlags] for x in zos_rate_demean.T])

    logger.info('...covariance of AR1 innovation annual SLR rate')
    rho = acf_rate_annual[:, 1] # lag-1 autocorrel
    innov_rate = zos_rate_demean[1:] - zos_rate_demean[:-1]*rho[None]
    cov_ar1_innovation_rate_annual = np.cov(innov_rate.T)

    # covariance in annual SLR anomaly
    logger.info('...covariance annual SLR anomaly')
    cov_ano_annual = np.cov(zos_detrended.T)

    logger.info('...auto-correlation annual SLR anomaly')
    zos_ano_demean = zos_detrended - zos_detrended.mean(axis=0)
    acf_ano_annual = np.array([autocorr(x)[:nlags] for x in zos_ano_demean.T])

    logger.info('...covariance of AR1 innovation annual SLR ano')
    rho = acf_ano_annual[:, 1] # lag-1 autocorrel
    innov_ano = zos_ano_demean[1:] - zos_ano_demean[:-1]*rho[None]
    cov_ar1_innovation_ano_annual = np.cov(innov_ano.T)


    # SAT trend
    logger.info('...covariance SAT trend')
    sat_trends = np.array([calc_lineartrend_fast(vals) for vals in _sample_along_sliding_window(zos_detrended, sat_length)])
    cov_sat_trends = np.cov(sat_trends.T)

    model_record = {
        "picontrol_years": zos_detrended.shape[0],

        "cov_rate_annual": cov_rate_annual,
        "cov_ano_annual": cov_ano_annual,
        "cov_ar1_innovation_rate_annual": cov_ar1_innovation_rate_annual,
        "cov_ar1_innovation_ano_annual": cov_ar1_innovation_ano_annual,
        "cov_sat_trends": cov_sat_trends,

        "acf_lags": acf_lags,
        "acf_rate_annual": acf_rate_annual,
        "acf_ano_annual": acf_ano_annual,
        **metadata,
    }


    return model_record_to_dataset(model_record)