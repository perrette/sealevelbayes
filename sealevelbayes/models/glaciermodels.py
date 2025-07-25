"""Models specifically for mountain glaciers

More definitions and actual main are present in the accompanying notebook:

    notebooks/glacier-regions.ipynb

"""
import itertools
import numpy as np

import pytensor.tensor as pt # type: ignore
import pymc as pm # type: ignore

from sealevelbayes.logs import logger
from sealevelbayes.datasets.glaciers import load_model_data, RCP_SCENARIOS, SSP_SCENARIOS

from sealevelbayes.models.generic import (SourceModel,
                                          diff_bfill,
                                          MAXFLOAT,
                                          GenericConstraint,
                                          save_private_var, getvar,
                                          )

def clip_glacier_volume(x, scale=None):
    return pt.clip(x, 1e-6, MAXFLOAT)
    # """Smooth ramp function to avoid numerical issues with zero values"""
    # # clip_glacier_volume = 0.5 * (x + np.sqrt(x**2 + epsilon**2))
    # # this is a bit more numerically stable
    # # see
    # epsilon = 1e-2
    # return 0.5 * (x + np.sqrt(x**2 + epsilon**2))


def rename(x):
    return f"glacier_{x}" # necessary for multiple regions or integration with the larger SLR models


def define_quadratic_forcing(T, a, aT0, q=0):
    """
    Quadratic Temperature forcing f(T):

    F(T) = q T^2 + a T - aT0
    """
    F = q * T**2 + a * T - aT0

    t = pt.arange(T.shape[-1])
    intT = pt.cumsum(T, axis=-1)
    intT2 = pt.cumsum(T**2, axis=-1)

    if q == 0:
        intF = a * intT - aT0 * t
    else:
        intF = q * intT2 + a * intT - aT0 * t

    return F, intF

def _time_index_by_region(tensor, i1):
    assert np.ndim(i1) in (0, 1), repr(i1)
    if np.size(i1) > 1 and tensor.shape[0].eval() > 1:
        return pt.stack([tensor[i, ..., i1[i]] for i in range(len(i1))])
    elif np.size(i1) > 1:
        return pt.stack([tensor[0, ..., i1[i]] for i in range(len(i1))])
        # return tensor[0, ..., i1].T  # put region back in the first dimension
    else:
        return tensor[..., i1]


def define_glacier_vn(F, V0, V1=1, r1=1, n=0.76, i0=None, intF=None):
    """
    Parameters
    ----------
    F: pytensor
        forcing function with F(2000) ~ 1 and other conditions
    V0: pytensor, optional
        initial volume (mm), or volume at time i0
    V1: pytensor
        volume scaling parameter (mm)
    r1: pytensor
        rate scaling parameter (mm/yr)
    n: float
        volume-scaling exponent in the glacier model
    i0: int, optional
        index of the time corresponding to V0
    intF: pytensor, optional
        integral of the forcing function

    Returns
    -------
    V: pytensor
        volume at each time step (mm)
    rate: pytensor
        minus the rate of volume change at each time step (mm/yr)

    Comments
    --------
    Solution to

        V' = r1 (V/V1)^n F
        V(0) = V0

    where
    - r1 and V1 represent the rate (mm/yr) and volume (mm)
    - and F(T) is a dimensionless function of global-mean temperature

    Note the following special case to match V(2000) = V1 and V'(2000) = r1:
    - F(2000) = 1 ensures that V'(2000) = r1 * (V(2000) / V1)^n
    - int F(T) dt in 2000 = (V1 - V0*(V1/V0)^n) / (m r1) ensures that V(2000) = V1
        (the latter condition is not great, but sampling this non-linear function
        takes times so it is worthwhile to do the math to ensure some basic constraints are met regardless of the value of the free parameters)

    - V0 is the initial volume (it is )

    The integral form is

    `V = max{ 0 , V0^m + m r1 / V1^n \int F(T) dt }^(1/m)`

    where m = 1 - n

    The forcing F and, optionally, its integral form intF, are provided as input.

    """
    if intF is None:
        intF = pt.cumsum(F, axis=-1)

    if i0 is not None:
        check = intF.ndim >= 2
        if check: shape0 = intF.shape.eval()  # noqa: E701
        intF = intF - _time_index_by_region(intF, i0)[..., None]
        if check: shape1 = intF.shape.eval()  # noqa: E701
        if check: assert tuple(shape0) == tuple(shape1), repr((shape0, shape1))  # noqa: E701

    m = 1 - n

    # logger.info(f"glacier model n = {n}")

    # optimize when n == 0 ? (== semi-empirical model Rahmstorf et al. 2007)
    # if False:
    if np.size(n) == 1 and n == 0:
        return V0 + r1 * intF, - r1 * F

    # special case with a different solution
    elif np.size(n) == 1 and n == 1:
        # logV' = r1/V1 F
        # logV = logV1 + r1/V1 \int F
        # V = V0 * exp(r1/V1 \int F)
        # V' = V0 * r1/V1 F exp(r1/V1 \int F)
        V =  V0 * pt.exp(r1/V1 * intF)
        return V, -r1/V1 * F * V

    pseudo_integral = m*r1/V1**n * intF

    # Vinside = pt.clip(V0**m + pseudo_integral, 1e-6, MAXFLOAT)

    Vinside = clip_glacier_volume(V0**m + pseudo_integral, V1)

    volume = Vinside**(1/m)
    rate = -r1/V1**n * Vinside**(n/m) * F  # theoretical calculation for Vinside > 0

    return volume, rate


## Uncharted glaciers from Parkes and Marzeion 2018

def get_uncharted_glacier_timeseries(model_years, slr_1901_2015, mass_2015, time_scale=None):
    """
    assume rate = r0 exp( - t / tau)
    with t = 0 in 1901 and 1 in 2015
    and R = \int_0^t = r0 * tau (1 - exp(-t / tau)) * (2015 - 1901)
    with R(1) = slr_1901_2015
    and  R(inf) = slr_1901_2015 + mass_2015

    """
    delta_t = 2015 - 1901
    t = (model_years - 1901) / delta_t
    slr_inf = slr_1901_2015 + mass_2015  # total contribution in the future (assuming full disappearance)
    delta = slr_1901_2015 / slr_inf
    if time_scale is None:
        tau = - 1 / np.log(1 - delta)
    else:
        tau = time_scale / delta_t
    r0 = slr_inf / (tau * delta_t)
    return r0 * np.exp(- t / tau)


def get_uncharted_glacier_global(model):
    """Include uncharted glaciers according to Parkes and Marzeion 2018
    """
    model_years = np.array(model.coords['year'])
    rate_low = get_uncharted_glacier_timeseries(model_years, 16.7, 2.1)
    rate_high = get_uncharted_glacier_timeseries(model_years, 48, 2.4)
    _uncharted_glacier_uniform = pm.Uniform("_uncharted_glacier_uniform", 0, 1)
    return _uncharted_glacier_uniform * rate_high + (1-_uncharted_glacier_uniform) * rate_low


def get_uncharted_glacier_distribution(model, regions=None, method=None, exclude_antarctica=False):
    """If not all regions are present the sum will be less than 1
    """
    logger.info("distribute uncharted glaciers according to early 20th century volume")
    if regions is None:
        regions = np.asarray(model.coords['glacier_region'])
    if method is None:
        method = "mm21-slr20" # back-compat

    all_regions = np.arange(1, 20).tolist()

    if method == "mm21-v1900":
        ds = load_model_data(all_regions)
        V2000 = ds['V2000'].values
        slr20 = ds['slr20'].values
        V0 = V2000 + slr20
        fractions = V0 / V0.sum()

    elif method == "mm21-slr20":
        ds = load_model_data(all_regions)
        slr20 = ds['slr20'].values
        fractions = slr20 / slr20.sum()

    elif method.startswith("rgi"):
        from sealevelbayes.datasets.rgi import load_glaciers_metadata, check_fraction_small_glaciers_per_region
        rgi = load_glaciers_metadata(version="7.0", regions=all_regions)
        if method == "rgi":
            thres = np.inf

        elif method == "rgi-small-2":
            thres = 2

        elif method == "rgi-small-2-10":
            thres = (2, 10)

        else:
            raise NotImplementedError(method)

        fractions = check_fraction_small_glaciers_per_region(rgi, thres=thres, regions=all_regions) # type: ignore

    else:
        raise NotImplementedError(method)

    fractions = np.asarray(fractions)
    if exclude_antarctica:
        fractions[all_regions.index(19)] = 0
        fractions = fractions / fractions.sum()
    return fractions[np.isin(all_regions, regions)]


class GlacierModelRegion(SourceModel):
    """This class defines a glacier model for a single region, and is designed to be used the same way other contributions are used.
    """

    def __init__(self, region, param_names, params_dist=None, i0=0, n=0.76, r1=1, V1=1, noise_on_forcing=False, label="", **kw):
        if not label:
            label = "glacier_" + str(region)
        super().__init__(label, param_names, params_dist=params_dist, **kw)
        self.i0 = i0
        self.n = n
        self.r1 = r1 # in units of SLR contribution
        self.V1 = V1
        self.noise_on_forcing = noise_on_forcing
        self.region = region
        if self.noise_on_forcing:
            self.add_noise = False
            assert self.rate_obs is True, "forcing noise only works with rate_obs=True"

    def calc_tensor(self, T, V0, a, aT0, **kw):
        _, F = super().calc_tensor(T, a=a, aT0=aT0, **kw)

        if self.noise_on_forcing:
            _, rate_noise, _, _ = super()._generate_noise(F, F, **self.noise_kw)  # tuned to match past rate's s.d.
            # forcing_noise = rate_noise * self.r1 / self.V1**self.n  # --> it will be later scaled reversely
            forcing_noise = rate_noise / self.r1 #
            F = F + forcing_noise

        # r1 needed in units of glacier mass balance
        volume, rate = define_glacier_vn(F=F, V0=V0, i0=self.i0, n=self.n, r1=-self.r1, V1=self.V1)
        return -volume, rate  # here return the negative volume to match the sea-level rise convention

    def _generate_noise(self, slr, rate, clip_noise=True, **kwargs):
        slr_noise, rate_noise, slr, rate = super()._generate_noise(slr, rate, **kwargs)

        save_private_var(f"{self.label}_slr_before_clipping", slr)
        save_private_var(f"{self.label}_rate_before_clipping", rate)

        if clip_noise:
            # clip the volume and recompute rate to avoid negative values : volume = -slr
            # do that *after* applying the constraints, otherwise the solver does not converge
            slr = -clip_glacier_volume(-slr, self.V1)
            rate = diff_bfill(slr)

        return slr_noise, rate_noise, slr, rate


SSP_TO_RCP_MAP = dict(itertools.chain(zip(SSP_SCENARIOS, RCP_SCENARIOS), zip([x+"_mu" for x in SSP_SCENARIOS], RCP_SCENARIOS)))

class Glacier2100Constraint(GenericConstraint):
    def __init__(self, obs, obs_sd, region, experiment, dummy=False, on_trend=False, no_clip=False, scale=None):
        self.region = region
        self.experiment = experiment
        self.dummy = dummy
        self.no_clip = no_clip
        self.on_trend = on_trend
        self.diag = "proj2100"
        self.obs = obs
        self.obs_sd = obs_sd
        self.scale = scale
        self.label = f"glacier_{region}_{self.diag}_{experiment}"

        if self.scale is not None:
            self.label += "_scaled"
            self.obs = self.obs / self.scale
            self.obs_sd = self.obs_sd / self.scale

    @classmethod
    def from_glacier_data(cls, region, experiment, glacier_data=None, scaled=False, **kwargs):
        obs_experiment = SSP_TO_RCP_MAP.get(experiment, experiment)
        if glacier_data is None:
            glacier_data = load_model_data([region], experiment=[obs_experiment])
        obs = glacier_data["slr21"].sel(experiment=obs_experiment, region=region).item()
        obs_sd = glacier_data["slr21_sd"].sel(experiment=obs_experiment, region=region).item()
        if scaled:
            scale = glacier_data["V2000"].sel(region=region).item() # normalizing factor
        else:
            scale = None
        return cls(obs=obs, obs_sd=obs_sd, region=region, experiment=experiment, scale=scale, **kwargs)

    def apply_model(self, model=None):
        model = pm.modelcontext(model)
        all_experiments = list(model.coords['experiment'])
        ix = all_experiments.index(self.experiment)

        if self.on_trend:
            volume_experiments = -getvar(f"glacier_{self.region}_slr_trend")
        else:
            if self.no_clip:
                volume_experiments = -getvar(f"glacier_{self.region}_slr_before_clipping")
            else:
                volume_experiments = -getvar(f"glacier_{self.region}_slr")

        shape = volume_experiments.shape.eval()
        assert volume_experiments.ndim == 2, f"Expected 2D volume tensor, got {volume_experiments.ndim}D in glacier volume"
        assert len(all_experiments) == shape[0], f"Expected {len(all_experiments)} experiments, got {shape[0]} in glacier volume"

        V = volume_experiments[ix]

        pm.ConstantData(f"{self.label}_obs_mu", self.obs)
        pm.ConstantData(f"{self.label}_obs_sd", self.obs_sd)

        # model_proj = pm.Deterministic(label), (V[:, ix, 2005-1900] - V[:, ix, 2099-1900]), dims=(rename('region'), rename('experiment_constraint')))
        model_proj = pm.Deterministic(self.label, V[2005-1900] - V[2099-1900])

        if self.scale is not None:
            model_proj = model_proj / self.scale
            pm.ConstantData(f"{self.label}_scale", self.scale)

        if not self.dummy:
            pm.Normal(f"{self.label}_obs", model_proj, self.obs_sd, observed=self.obs)