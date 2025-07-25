import json
import os

import numpy as np
from scipy.stats import norm, lognorm
import pymc as pm # type: ignore
import pytensor.tensor as pt # type: ignore
from pytensor.ifelse import ifelse # type: ignore

from sealevelbayes.logs import logger
import sealevelbayes.datasets.ar6
from sealevelbayes.datasets.ar6.tables import (
    ar6_table_9_8_medium_confidence, ar6_table_9_8_quantiles,
    ar6_table_9_5, ar6_table_9_5_quantiles )
from sealevelbayes.preproc.stats import fit_dist_to_quantiles
from sealevelbayes.models.generic import MINFLOAT, MAXFLOAT, ScaledCov, FixedMean, FixedCov, getvar


def observe_slr2100(rates):
    "observations to 2100"
    slr = pt.cumsum(rates, axis=0)
    return (slr[199] + rates[199])- pt.mean(slr[95:114+1], axis=0) # last year missing, we extrapolate

def _getyearval(target, years, tensor):
    if target is None:
        return 0 # useful when year_ref is None

    if np.size(target) == 1:
        val = tensor[..., np.searchsorted(years, target)]
    else:
        y1, y2 = target
        i1, i2 = np.searchsorted(years, [y1, y2])
        val = tensor[..., i1:i2+1].mean(axis=-1)

    return val

class Constraint:
    def __init__(self, dist_name, dist_args, dummy=False, experiment=None, prefixed=True,
                 source=None, diag=None, label=None, region=None,
                 input_vars=None, on_trend=False):
        self.args = dist_args
        self.dist_name = dist_name
        self.dummy = dummy
        self.experiment = experiment
        self.region = region
        self.prefixed = prefixed
        self.source = source
        self.diag = diag
        assert source is not None, "source must be provided"
        assert diag is not None, "diag must be provided"
        if not label:
            self.label = f"{source}_{diag}"
            if experiment:
                self.label += f"_{experiment}"
            if region:
                self.label += f"_{region}"
        else:
            self.label = label

        self.input_vars = input_vars
        self.on_trend = on_trend

    @property
    def model(self):
        return pm.modelcontext(None)

    @property
    def years(self):
        return np.asarray(self.model.coords['year'])

    @property
    def experiments(self):
        return list(self.model.coords['experiment'])

    def _select_year(self, tensor, year, years=None):
        return _getyearval(year, self.years if years is None else years, tensor)

    def _select_experiment(self, tensor, experiment, experiments=None):
        if experiment is None:
            return tensor

        try:
            ix = list(self.experiments if experiments is None else experiments).index(experiment)
        except ValueError:
            raise ValueError(f"{self.label}: experiment {experiment} not found in the model")

        return tensor[ix]

    def rename(self, name):
        if self.prefixed:
            name = f"{self.label}_{name}"
        return name

    def observe(self, tensor):
        model = self.model
        dist_name = self.dist_name
        args = self.args
        label = self.label

        # save (unobserved) model prediction to trace
        if not hasattr(model, label):
            pm.Deterministic(label, tensor)

        if self.dummy:
            return

        # now define the constraint
        name = label+"_obs"

        if dist_name == "norm":
            dist_name = "Normal"

        if dist_name == "Normal":
            mean, sigma = args
            return getattr(pm, dist_name)(name, tensor, sigma, observed=mean)

        # that's a bit of a special case where the likelihood distribution is hard to define in Bayesian sense (p(obs | model))
        # so we just define a potential
        # scipy convention (to add "loc")
        elif dist_name == "lognorm":
            name = label+"_potential"
            s, loc, scale = args  # scipy lognorm convention
            log_lik = lambda x: pm.logp(pm.Lognormal.dist(mu=np.log(scale), sigma=s), pt.clip(x-loc, MINFLOAT, MAXFLOAT))
            pm.Potential(name, log_lik(tensor))

        # or pymc LogNormal convention
        elif dist_name == "LogNormal":
            name = label+"_potential"
            mu, sigma = args  # scipy lognorm convention
            log_lik = lambda x: pm.logp(pm.Lognormal.dist(mu=mu, sigma=sigma), pt.clip(x, MINFLOAT, MAXFLOAT))
            pm.Potential(name, log_lik(tensor))

        elif dist_name == "custom_ais":
            mu, sigma, s, loc, scale = args
            name = label+"_potential"
            logger.info(f"Use custom_ais distribution for {label}")

            pm.Potential(name, ifelse(
                # pt.le(q, loc),
                pt.le(tensor, mu),
                pm.logp(pm.Normal.dist(mu, sigma), tensor),
                # pm.logp(pm.Normal.dist(observed_values[1], observed_values[2]-observed_values[1]), q),
                pm.logp(pm.Lognormal.dist(mu=np.log(scale), sigma=s), tensor-loc),
                ))

        else:
            raise NotImplementedError(dist_name)

    def __call__(self, slr, rate):
        """Call the constraint on (slr, rate) ==> this method is deprecated, use apply_model() instead"""
        if slr is not None: slr = self._select_experiment(slr, experiment=self.experiment)
        if rate is not None: rate = self._select_experiment(rate, experiment=self.experiment)
        tensor = self.get_observable(slr, rate)
        return self.observe(tensor)

    def get_observable_from_model(self, model=None):
        """Get the variable directly from the model object (either from private variable or defined variables)

        This is a little more general than calling the constraint on (slr, rate)
        """
        model = pm.modelcontext(model)

        if self.input_vars is None:
            if self.on_trend:
                input_vars = [ f"{self.source}_slr_trend", f"{self.source}_rate_trend" ]
            else:
                input_vars = [ f"{self.source}_slr", f"{self.source}_rate" ]
        else:
            input_vars = self.input_vars

        assert len(input_vars) == 2, f"{self.label}: input_vars must be a list of two variables (slr, rate), got {input_vars}"

        tensors = []
        for i, name in enumerate(input_vars):
            tensor_multiexp = getvar(name, model=model)
            if self.experiment is None:
                tensor = tensor_multiexp[0] # first exp
            else:
                tensor = self._select_experiment(tensor_multiexp, experiment=self.experiment)
            tensors.append(tensor)
        slr, rate = tensors

        return self.get_observable(slr, rate)

    def apply_model(self, model=None):
        tensor = self.get_observable_from_model(model)
        return self.observe(tensor)


class ScalarConstraint(Constraint):
    pass

class SLRConstraint(ScalarConstraint):

    def __init__(self, dist_name, args, year_ref, year, **kw):
        super().__init__(dist_name, args, **kw)
        self.year_ref = year_ref
        self.year = year

    def get_observable(self, slr, rate, years=None):
        assert slr.size.eval() > 1, (self.label, slr.eval().shape)
        # print(self.label, slr.eval().shape)
        return self._select_year(slr, self.year, years=years) - self._select_year(slr, self.year_ref, years=years)

class RateConstraint(ScalarConstraint):

    def __init__(self, dist_name, args, year, **kw):
        super().__init__(dist_name, args, **kw)
        self.year = year

    def get_observable(self, slr, rate, years=None):
        return self._select_year(rate, self.year, years=years)


class TimeSeriesConstraint(Constraint):
    def __init__(self, label, obs, obs_sd=None, obs_cov=None, obs_years=None, rate_obs=True, scale=None, input_vars=None, **kw):
        super().__init__(None, None, label=label, **kw)
        self.obs = obs
        self.obs_sd = obs_sd
        self.obs_cov = obs_cov
        self.obs_years = obs_years
        self.rate_obs = rate_obs
        self.scale = scale
        if input_vars is None:
            raise ValueError("input_vars must be provided for TimeSeriesConstraint")
        self.input_vars = input_vars

    def get_observable_from_model(self, model=None):
        return self.get_observable(*[getvar(name, model=model) for name in self.input_vars])

    def get_observable(self, slr, rate):
        prediction = rate if self.rate_obs else slr

        if self.obs_years is not None:
            iy = np.searchsorted(self.years, self.obs_years)
            prediction = prediction[..., iy]

        if prediction.ndim == 2:
            shape = prediction.shape.eval()
            assert shape[0] == len(self.experiments), (self.label, shape, self.experiments)
            ix = self.experiments.index(self.experiment)
            prediction = prediction[ix]

        return prediction


    def _cast_sigma_quad(self, sigma, dim):
        if dim == 0:
            assert sigma.ndim == 0, (self.label, sigma.shape)
            return sigma**2

        if dim == 1:
            assert sigma.ndim <= 1, (self.label, sigma.shape)
            return sigma**2

        if dim == 2 and sigma.ndim == 0:
            return pt.eye(len(self.years)) * sigma**2

        if dim == 2 and sigma.ndim == 1:
            return pt.diag(sigma**2)

        if dim == 2 and sigma.ndim == 2:
            return sigma

        raise ValueError(f"Unknown covariance shape {sigma.shape} for {self.label}")


    def get_sigma(self, sigma=None):

        if callable(self.scale):
            self.scale = self.scale()

        # pm.ConstantData(self.rename("data"), self.obs)
        if self.obs_cov is not None:
            logger.info(f"Using covariance matrix for {self.label}")
            sigma_obs = pm.ConstantData(self.rename("data_cov"), self.obs_cov)
            if self.scale is not None:
                logger.info(f"Using scale for {self.label}")
                sigma_obs = sigma_obs * self.scale**2

        elif self.obs_sd is not None:
            logger.info(f"Using s.d. matrix for {self.label}")
            assert self.obs_sd is not None
            sigma_obs = pm.ConstantData(self.rename("data_sd"), self.obs_sd)
            if self.scale is not None:
                logger.info(f"Using scale for {self.label}")
                sigma_obs = sigma_obs * self.scale

        else:
            raise ValueError(f"No obs error provided for {self.label}")

        if sigma is None:
            sigma = sigma_obs

        else:
            dim = max([sigma.ndim, sigma_obs.ndim])
            sigma = self._cast_sigma_quad(sigma, dim) + self._cast_sigma_quad(sigma_obs, dim)
            if dim < 2:
                sigma = pt.sqrt(sigma)

        return sigma


    def observe(self, tensor, sigma=None, **kw):
        tensor = pm.Deterministic(self.label, tensor)
        name = self.label + "_obs"

        sigma = self.get_sigma(sigma)

        if sigma.ndim < 2:
            return pm.Normal(name, tensor, sigma, observed=self.obs)

        elif sigma.ndim == 2:
            return pm.MvNormal(name, tensor, cov=sigma, observed=self.obs, **kw)

        else:
            raise ValueError(f"Unknown covariance shape {sigma.shape} for {self.label}")


class TimeSeriesConstraintGP(TimeSeriesConstraint):

    def __init__(self, label, obs, obs_sd=None, obs_cov=None, obs_years=None, rate_obs=True,
            cov_func="Matern12", cov_params=None, ls_dist="ConstantData", ls_dist_params=(5.,),
            sigma_dist="HalfNormal", sigma_dist_params=(1.,), gp_marginal=True, **kw):
        """
        """
        super().__init__(label, obs, obs_sd=obs_sd, obs_cov=obs_cov, obs_years=obs_years, rate_obs=rate_obs, **kw)
        assert np.isfinite(obs).all(), ("NaNs: not implemented:", label, obs)
        self.gp_marginal = gp_marginal
        self.cov_func = cov_func
        self.cov_params = cov_params
        self.ls_dist = ls_dist
        self.ls_dist_params = ls_dist_params
        self.sigma_dist = sigma_dist
        self.sigma_dist_params = sigma_dist_params

    def observe(self, prediction):

        sigma = getattr(pm, self.sigma_dist)(self.rename("sigma"), *self.sigma_dist_params)

        cov_func = get_gp_cov_func(self.cov_func, self.cov_params,
                                   ls_dist=self.ls_dist, ls_dist_params=self.ls_dist_params, ls_label=self.rename("ls"), sigma=sigma)


        if self.obs_years is None:
            years = self.years
        else:
            years = self.obs_years

        X = years[:, None] - 1950

        if not self.gp_marginal:
        # if True:
            sigma = cov_func(X) if callable(cov_func) else cov_func
            return super().observe(prediction, sigma=sigma)


        mean_func = FixedMean(prediction)
        marginal = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)

        if self.obs_cov is not None:
            obs_cov = FixedCov(pm.ConstantData(self.rename("obs_cov"), self.obs_cov))
            return marginal.marginal_likelihood(self.rename("obs"), X=X, y=self.obs, sigma=obs_cov)

        else:
            self.obs_sd = pm.ConstantData(self.rename("obs_sd"), self.obs_sd)
            return marginal.marginal_likelihood(self.rename("obs"), X=X, y=self.obs, sigma=self.obs_sd)

def get_gp_cov_func(cov_name, cov_params=None, ls_dist=None, ls_dist_params=None, ls_label=None, sigma=1):
    if cov_params is None:
        assert ls_dist is not None
        ls = getattr(pm, ls_dist)(ls_label, *ls_dist_params)
        cov_params = (ls,)

    kernel = getattr(pm.gp.cov, cov_name)

    if issubclass(kernel, pm.gp.cov.Covariance):
        cov_func = kernel(1, *cov_params) * sigma**2

    else:
        assert len(cov_params) == 1
        assert issubclass(kernel, pm.gp.cov.WhiteNoise), kernel
        cov_func = kernel(cov_params[0])
        cov_func = ScaledCov(cov_func, sigma) # just multiplying triggers and error w.r.t input_dim

    return cov_func




def get_ar6_constraint_table_95(source, type, period, diag, dummy=False):

    year1, year2 = period
    period_tag = f"{year1}-{year2}"
    table_source = {"total": "gmsl", "antarctica": "ais", "greenland": "gis"}.get(source, source)

    if type == "slr":
        # dist_slr20 = fit_dist(ar6_table_9_5_quantiles, np.array(ar6_table_9_5[table_source]["Δ (mm)"]["1901-1990"]), dist=norm)
        dist = fit_dist_to_quantiles(ar6_table_9_5[table_source]["Δ (mm)"][period_tag], ar6_table_9_5_quantiles, dist_name="norm")
        return SLRConstraint(dist.dist.name, dist.args, year1, year2, source=source, diag=diag, dummy=dummy)

    elif type == "rate":
        dist = fit_dist_to_quantiles(ar6_table_9_5[table_source]["mm/yr"][period_tag], ar6_table_9_5_quantiles, dist_name="norm")
        return RateConstraint(dist.dist.name, dist.args, (year1, year2), source=source, diag=diag, dummy=dummy)

    else:
        raise ValueError(type)

def get_ar6_constraint_table_95_by_alias(alias, source, diag=None, dummy=False):
    alias = {
        "slr20": "slr_1901-1990",
        "rate2000": "rate_1993-2018",
        }.get(alias, alias)
    try:
        type, period_tag = alias.split("_")
    except ValueError as error:
        raise ValueError(f"{alias}: {error}")
    period = map(int, period_tag.split("-"))
    return get_ar6_constraint_table_95(source, type, period, diag or alias, dummy=dummy)


all_sources=["steric", "glacier", "ais", "gis", "total"]


def get_20c_constraints_ar6_table_9_5(sources=["steric", "glacier", "ais", "gis"], rate2000=True, slr20=True, dummy=False):
    """ Y: dict of time-series tensors
    """
    constraints = []

    for source in sources:

        if slr20:
            constraints.append(get_ar6_constraint_table_95(source, "slr", (1901, 1990), "slr20", dummy=dummy))

        if rate2000:
            constraints.append(get_ar6_constraint_table_95(source, "rate", (1993, 2018), "rate2000", dummy=dummy))

    return constraints

EXPERIMENTS_MAP = {"ssp126": "ssp126_mu", "ssp585": "ssp585_mu"}

def get_21c_constraints_ar6_table_9_8(
    experiments_map=EXPERIMENTS_MAP,
    sources=["steric", "glacier", "ais", "gis"],
    experiments=['ssp126', 'ssp585'],
    default_dist=norm, dist_cls={'ais':lognorm, 'total':lognorm}, dummy=False, include_peripheral_glaciers=False, on_trend=False):
    # default_dist=norm, dist_cls={'ais':"custom_ais", 'total':lognorm}, dummy=False):
    """ AR6 constraints for future projections
    """
    constraints = []

    for source in sources:
        dist = dist_cls.get(source, default_dist)

        # The custom_ais distribution is a mixture of a normal and a lognormal distribution,
        # with the lognormal distribution being used for the upper tail of the distribution
        # This was found to yield better convergence in minimal models, but it is not clear if it is necessary here
        # ref: https://discourse.pymc.io/t/lognormal-constraint-poor-convergence/15901/5
        # It is included as an experimental option
        if dist == "custom_ais":
            dist_name = dist
            dist = lognorm
        else:
            dist_name = dist.name

        for x in experiments:
            qvalues = np.array(ar6_table_9_8_medium_confidence[x][source])*1e3
            # https://gitlab.pik-potsdam.de/dcm-impacts/slr-tidegauges-future/-/issues/76#note_53387
            peripheral_glaciers = {
                "gis": {"ssp126": (10, 10), "ssp585": (19, 11)},
                "ais": {"ssp126": (80, 56), "ssp585": (160, 86)},
            }
            if include_peripheral_glaciers and source in ("gis", "ais"):
                mean, scale = peripheral_glaciers[source][x]
                logger.info(f"Include peripheral glaciers in future constraint: {source} {x}: add {mean} +- {scale} to existing {qvalues[1]}")
                qvalues += mean # all quantiles shifted
                # range enlarged
                med = qvalues[ar6_table_9_8_quantiles.index(.5)]
                lower = med - qvalues[ar6_table_9_8_quantiles.index(.17)]
                upper = qvalues[ar6_table_9_8_quantiles.index(.83)] - med
                lower_enhanced = (lower**2 + scale**2)**.5
                upper_enhanced = (upper**2 + scale**2)**.5
                qvalues[ar6_table_9_8_quantiles.index(.17)] = med - lower_enhanced
                qvalues[ar6_table_9_8_quantiles.index(.83)] = med + upper_enhanced

            dist_slr21 = fit_dist_to_quantiles(qvalues, ar6_table_9_8_quantiles, dist_name=dist.name)
            if dist_name == "custom_ais":
                constraint = SLRConstraint(dist_name, (qvalues[1], qvalues[1]-qvalues[0]) + dist_slr21.args, (1995, 2014), 2099, experiment=experiments_map.get(x, x), source=source, diag="proj2100", dummy=dummy, on_trend=on_trend)
            else:
                constraint = SLRConstraint(dist_name, dist_slr21.args, (1995, 2014), 2099, experiment=experiments_map.get(x, x), source=source, diag="proj2100", dummy=dummy, on_trend=on_trend)
            constraints.append(constraint)

    return constraints


def get_21c_constraints_low_confidence(source,
                                       experiments_map=EXPERIMENTS_MAP,
                                       experiments = ['ssp585'],
                                       **kwargs):

    fits = json.load(open(os.path.join(sealevelbayes.datasets.ar6.__path__[0], "pb_fits_by_matthias.json")))
    constraints = []
    for x in experiments:
        fit = fits[f"{source.upper()}_{x}_pb_2e"]
        dist_name = fit["distribution"]
        args = fit["s"], fit["loc"], fit["scale"]
        constraint = SLRConstraint(dist_name, args, (1995, 2014), 2099, experiment=experiments_map.get(x, x), source=source, diag="proj2100", **kwargs)
        constraints.append(constraint)
    return constraints