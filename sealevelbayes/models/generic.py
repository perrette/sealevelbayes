"""Models specifically for mountain glaciers

More definitions and actual main are present in the accompanying notebook:

    notebooks/glacier-regions.ipynb

"""
import numpy as np
from statsmodels.tsa.stattools import acf

import pandas as pd
import xarray as xa

import pytensor.tensor as pt
import pymc as pm

from sealevelbayes.logs import logger
from sealevelbayes.preproc.linalg import detrend_timeseries

MAXFLOAT = 1e34  # to avoid warning in the volume calculation
MINFLOAT = 1e-9  #

def _reshape(a, dims, target_dims):
    assert set(dims).issubset(target_dims), ("dims must be subset of target_dims", dims, target_dims)
    # make sure the dimensions are in the right order, otherwise the broadcast will fail
    assert (np.diff([target_dims.index(dim) for dim in dims]) > 0).all(), ("bad order", dims, target_dims)
    idx = tuple(slice(None) if dim in dims else None for dim in target_dims)
    # print(dims, target_dims, idx, _shape(a), _shape(a[idx]), shape)
    return a[idx]

def _eval(a):
    return a.eval() if hasattr(a, "eval") else a

def _shape(a):
    return tuple(_eval(a.shape))

def _regularize_cov(cov, epsilon=1e-6):
    return cov + epsilon * np.eye(cov.shape[0])

def detrend_timeseries_2d(rate):
    detrended = rate.copy()

    # rate = filter_timeseries(rate, frequency=frequency, lowpass=False, **filter_kw)
    for i in range(rate.shape[1]):
        if isinstance(rate, pd.DataFrame):
            values = detrended.values[:, i]
        else:
            values = detrended[:, i]
        mi = np.isfinite(values)
        values[mi] = detrend_timeseries(values[mi])

    return detrended


def diff_bfill(array, value=None):
    """calculate the difference and back-fill the first value
    """
    module = np if isinstance(array, np.ndarray) else pt
    diff = module.diff(array, axis=-1)
    first_value = diff[..., [0]]
    if value is not None:
        first_value = pt.zeros_like(first_value) + value
    return module.concatenate([first_value, diff], axis=-1)

def define_dist(name, dist_name, *args, **kwargs):
    # just get back an unnamed tensor if the name is not provided, that can be useful to call "eval" and get a default value
    if not name:
        cls_ = getattr(pm, dist_name)
        if hasattr(cls_, "dist"):
            return cls_.dist(*args, **kwargs)
        # Data, Deterministic
        elif len(args) == 1:
            return args[0]
        else:
            raise ValueError(f"Cannot find an unnamed version of {dist_name}: {args}")

    # transform the distribution to avoid any funnel effect
    if dist_name == "Normal" and "observed" not in kwargs:
        dist_normed = pm.Normal("__"+name+"_unit", **kwargs)
        m, s = args
        return pm.Deterministic(name, m + s * dist_normed, **kwargs)

    return getattr(pm, dist_name)(name, *args, **kwargs)

def getparam(dist_kw, name, dist_name, *args, **kwargs):
    if name in (dist_kw or {}):
        dist_name, args = dist_kw[name]
    return define_dist(name, dist_name, *args, **kwargs)


class FixedMean(pm.gp.mean.Mean):
    """A mean function that ignore its input and return the model trend
    """

    def __init__(self, model_trend):
        pm.gp.mean.Mean.__init__(self)
        self.model_trend = model_trend

    def __call__(self, X):
        return self.model_trend

class FixedCov(pm.gp.cov.Covariance):
    def __init__(self, cov):
        super(FixedCov, self).__init__(1, None)
        self.cov = cov

    def diag(self, X):
        return pt.diag(self.cov)

    def full(self, X, Xs=None):
        return self.cov

class ScaledCov(pm.gp.cov.Covariance):
    def __init__(self, covf, scale):
        super().__init__(1, None)
        self.covf = covf
        self.scale = scale

    def diag(self, X):
        return self.covf.diag(X) * self.scale**2

    def full(self, X, Xs=None):
        return self.scale**2 * self.covf.full(X, Xs)


class SourceModel:

    def __init__(self, label, param_names, params_dist=None, p0=None,
                 scales=None,
                 locs=None,
                 transform=None,
                 params=None,
                 years=None,
                 years_ref=None,
                 log_params=None, interpolated_params=None,
                 add_noise=False,
                 noise_kw={},
                 ref=None,
                 filepath=None,
                 rate_obs=False,
                 constraints=None,
                 **data_kw):

        self.label = label
        self.param_names = param_names
        self.p0 = p0
        self.params_dist = params_dist
        self._params = params # if already pre-allocated (not tested)
        self._years = years
        self.years_ref = years_ref
        self._prefix = (label + "_") if label is not None else None
        self.interpolated_params = interpolated_params
        self.log_params = log_params
        self.add_noise = add_noise
        self.noise_kw = noise_kw
        self.rate_obs = rate_obs
        self.ref = ref
        self.scales = scales
        self.locs = locs
        self.transform = transform
        self.filepath = filepath

        # deprecated
        deprecated_params = ["add_residuals", "constraints_on_trend", "frequentist"]
        for k in deprecated_params:
            if k in data_kw:
                logger.warning(f"{k} is deprecated, will be ignored.")
                del data_kw[k]

        experimental_params = ["regions"]
        for k in experimental_params:
            if k in data_kw:
                raise NotImplementedError(f"{k} is not implemented yet, will be ignored.")

        # experimental
        self.regions = regions = None
        self.nr = 1 if regions is None else len(regions)

        self.constraints = constraints or []
        self.set_data(**data_kw)


    def set_data(self, data=None, obs=None, obs_sd=None, obs_cov=None, sample_dim=None, obs_key=None):

        if not obs_key:
            obs_key = self.label

        self.obs_key = obs_key

        if data is None:
            data = {}

        if obs is not None:
            data[obs_key] = obs

        if obs_sd is not None:
            data[obs_key + "_sd"] = obs_sd

        if obs_cov is not None:
            data[obs_key + "_cov"] = obs_cov

        if self.add_noise:
            assert len(data) > 0, "No data provided"

        self.data = xa.Dataset(data)

        self.sample_dim = sample_dim

    @property
    def obs(self):
        if self.sample_dim is not None:
            return self.data[self.obs_key].mean(self.sample_dim)
        else:
            return self.data[self.obs_key]

    @property
    def obs_sd(self):
        if self.sample_dim is not None:
            return self.data[self.obs_key].std(self.sample_dim)
        else:
            return self.data[self.obs_key + "_sd"]

    @property
    def obs_cov(self):
        if self.sample_dim is not None:
            axis = self.data[self.obs_key].dims.index(self.sample_dim)
            cov = np.cov(self.data[self.obs_key].values, rowvar=axis > 0)
            cov += np.eye(cov.shape[0]) * 1e-6  # regularize
            return cov
        else:
            return None

    @property
    def years(self):
        if self._years is not None:
            return self._years
        model = pm.modelcontext(None)
        return np.asarray(model.coords['year'])

    @property
    def ndim(self):
        if self.regions is None:
            return 0
        else:
            return 1

    def rename(self, name):
        if callable(self._prefix):
            return self._prefix(name)
        elif self._prefix is None:
            return name
        return self._prefix + name

    def define_data(self, name, value, **kw):
        key = self.rename(name)
        model = pm.modelcontext(None)
        if not hasattr(model, key): # type: ignore
            nans = np.isnan(np.asarray(value))
            if nans.any():
                logger.warning(f"{self.label}: replacing {nans.sum()} NaNs in {name} with -9999 (total of {len(value)} values)")
                value[nans] = -9999
            return pm.ConstantData(key, value, **kw)
        else:
            return getattr(model, key) # type: ignore

    def define_deterministic(self, name, value, **kw):
        return pm.Deterministic(self.rename(name), value, **kw)

    def _getparams(self):
        """return the parameters as a dictionary
        """
        if self.params_dist is None:
            # we should
            raise ValueError("No parameters distribution defined. Please define params_dist or params")

        params = []
        for i, (name, param_dist) in enumerate(zip(self.param_names, self.params_dist)):
            if len(param_dist) == 2:
                dist_name, args = param_dist
                kwargs = {}
            else:
                dist_name, args, kwargs = param_dist

            p = define_dist(self.rename(name), dist_name, *args, **kwargs)
            if self.scales is not None:
                p = p * self.scales[i]
            if self.locs is not None:
                p = p + self.locs[i]
            params.append(p)

        if self.transform is not None:
            params = self.transform(params)

        self._params = params

        return params

    @property
    def params(self):
        if self._params is None:
            self._params = self._getparams()
        return self._params

    def getparam(self, name):
        i = self.param_names.index(name)
        return self.params[i]

    def _broadcast_param(self, a, ndim):
        "broadcast 1-D params (variable) to 2-D variable x time"
        # scalar dim
        if not hasattr(a, "ndim"):
            return a
        # scalar dim or already broadcasted
        if a.ndim == 0 or a.ndim == ndim:
            return a

        assert _shape(a) == (self.nr,), f"Expected {(self.nr,)}, got {_shape(a)}"
        target_dims = ["v", "x", "t"] if ndim == 3 else ["v", "t"]  # {variable x} experiment x time
        return _reshape(a, ["v"], target_dims)


    def calc_tensor(self, T, a=None, aT0=None, b=None, q=None, T2=None, Tdiff=None):
        assert aT0 is not None
        if a is not None:
            a = self._broadcast_param(a, T.ndim)
        aT0 = self._broadcast_param(aT0, T.ndim)
        if b is not None: b = self._broadcast_param(b, T.ndim)
        if q is not None: q = self._broadcast_param(q, T.ndim)

        if self.ndim == 0:
            assert T.ndim <= 2, _shape(T)  # experiment x time
        elif self.ndim == 1:
            assert T.ndim <= 3, f"Expected 2 or 3 dimensions for target (temperature) dimension, got {T.ndim}"
            if T.ndim == 2:
                T = T[None]

        F = -aT0 + np.zeros(T.shape.eval())

        if a is not None:
            F = F + a * T

        if b is not None:
            if Tdiff is None:
                Tdiff = diff_bfill(T)
            F += b * Tdiff

        if q is not None:
            if T2 is None:
                T2 = T**2
            F += q * T2

        assert F.ndim == T.ndim

        rate = F
        slr = pt.cumsum(rate, axis=-1)

        if self.years_ref is not None:
            i1, i2 = np.searchsorted(self.years, self.years_ref)
            slr = slr - slr[..., i1:i2+1].mean(axis=-1)[..., None]

        return slr, rate

    def _generate_noise(self, slr, rate,
                        marginalize=False, apply_constraints=False,
                        obs_autocorrel=True,
                        gp_cov="Matern12", gp_cov_params=None, ls_dist="ConstantData", ls_dist_params=(5.,), ls_dist_auto=False,
                        intercept=False, sigma_dist="HalfNormal", sigma_dist_params=(1.,), scale=None, **deprecated_kw):
        """Does a job similar to DataNoise.predict_ar_noise, but for a single time-series (no cross-correlation).
        Here the data autocovariance matrix can be accounted for.

        marginalize: if True, a marginalized GP process is used, otherwise a latent GP process is used
        obs_autocorrel: if True, the obs autocovariance matrix is used in the likelihood

        apply_constraints: controls whether the annual observations are used as constraints on the noise in this function. If False, the constraints must be applied later on,
            such as via the "constraints=" parameter. Generally we want to eventually apply the constraints explicitly via the constraints and constraints_on_trend parameters,
            but at the tine of writing, this is only implemented for glaciers. For all other sources, apply_constraints should be True in this function.
        """
        assert slr.ndim == 2
        assert rate.ndim == 2

        # depracted: sample_vectorized, ar_order, vectorized -> marginalize,
        for k in deprecated_kw:
            logger.warning(f"{k} is deprecated, will be ignored.")

        if self.obs is None:
            raise ValueError("No observation provided: cannot add noise")

        assert self.obs.year.values[0] == self.years[0], "First year of the observation must match the first year of the model"
        obs = self.obs.values
        obs_error = self.obs_sd.values
        obs_mask = np.searchsorted(self.years, self.obs.year.values)

        if not self.rate_obs:
            logger.info(f"Generating noise for {self.label} on cumulative sea level")
            slr = slr[0] # first experiment only
            # slr_offset = pm.Normal(self.rename("slr_offset"), self.obs.values[0], self.obs_sd.values[0])
            # trend = slr - slr[obs_mask].mean() + self.obs.values.mean()
            # offset = self.obs.values[0] - slr[obs_mask][0]
            if intercept:
                if self.years_ref:
                    # model slr is zero at the reference years
                    y1, y2 = self.years_ref
                    offset = pm.Normal(self.rename("slr_offset"), self.obs.loc[y1:y2].mean(), self.obs_sd.loc[y1:y2].mean())
                else:
                    # model slr starts at zero (well the year after zero, strictly speaking)
                    offset = pm.Normal(self.rename("slr_offset"), obs[0], obs_error[0])
            else:
                offset = obs.mean() - slr[obs_mask].mean()
            trend = slr + offset

        else:
            logger.info(f"Generating noise for {self.label} on rate of sea level")
            trend = rate[0]

        sigma = getattr(pm, sigma_dist)(self.rename("ar_sigma"), *sigma_dist_params)

        if scale is not None:
            sigma = sigma * scale

        trend_obs = trend[obs_mask]

        sigmatag = f"{sigma_dist}({', '.join(map(str, sigma_dist_params))})"
        if gp_cov_params is None:
            if ls_dist_auto:
                # determine the length scale automatically from the data
                rho = (self.obs.to_series() - self.obs.to_series().mean()).autocorr()
                ls_dist_params = (1/rho, )
                logger.info(f"{self.label} :: automatically determined the length scale from the data: {ls_dist_params}")
            ls = getattr(pm, ls_dist)(self.rename("noise_ls"), *ls_dist_params)
            gp_cov_params = (ls,)
            gp_cov_params_tag = f"{ls_dist}({', '.join(map(str, ls_dist_params))})"
        else:
            gp_cov_params_tag = ', '.join(map(str, gp_cov_params))

        kernel = getattr(pm.gp.cov, gp_cov)
        if issubclass(kernel, pm.gp.cov.Covariance):
            cov_func = kernel(1, *gp_cov_params)
            if marginalize:
                cov_func = cov_func * sigma**2
        else:
            assert len(gp_cov_params) == 1
            assert issubclass(kernel, pm.gp.cov.WhiteNoise), kernel
            cov_func = kernel(gp_cov_params[0])
            if marginalize:
                cov_func = ScaledCov(cov_func, sigma) # just multiplying triggers and error w.r.t input_dim

        # mean_func = pm.gp.mean.Constant(0.) + (lambda x: trend_obs)

        # Dont sample the noise
        if marginalize:
            logger.info(f"{self.label} :: assume a marginalized GP process with cov {sigmatag} * {gp_cov}(*{gp_cov_params_tag}) (obs_autocorrel={obs_autocorrel})")
            mean_func = FixedMean(trend_obs)
            marginal = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)

            if obs_autocorrel:
                obs_cov = FixedCov(pm.ConstantData(self.rename("obs_cov"), self.obs_cov))
                marginal.marginal_likelihood(self.rename("obs"), X=self.obs.year.values[:, None] - 1950, y=obs, sigma=obs_cov)

            else:
                obs_error = pm.ConstantData(self.rename("obs_sd"), obs_error)
                marginal.marginal_likelihood(self.rename("obs"), X=self.obs.year.values[:, None] - 1950, y=obs, sigma=obs_error)

            marginal = pm.gp.Marginal(cov_func=cov_func)

            noise = pt.zeros_like(trend) # here there is no noise, only the mean

            return self._return_noise(noise, slr, rate)


        else:
            logger.info(f"{self.label} :: sample noise from a latent GP process with cov {sigmatag} * {gp_cov}(*{gp_cov_params_tag}) (obs_autocorrel={obs_autocorrel})")

            # gp = pm.gp.Latent(cov_func=cov_func * sigma**2)
            latent = pm.gp.Latent(cov_func=cov_func)
            noise = latent.prior(self.rename("noise"), X=self.years[:, None] - 1950) * sigma

            if apply_constraints:
                if obs_autocorrel:
                    pm.MvNormal(self.rename("obs"), trend_obs + noise[obs_mask], cov=self.obs_cov, observed=obs)
                else:
                    pm.Normal(self.rename("obs"), trend_obs + noise[obs_mask], sigma=obs_error, observed=obs)

            return self._return_noise(noise, slr, rate)


    def _return_noise(self, noise, slr, rate):
        if self.rate_obs:
            rate_noise = noise
            slr_noise = pt.cumsum(noise, axis=-1)
        else:
            slr_noise = noise
            rate_noise = diff_bfill(noise)

        return slr_noise, rate_noise, slr + slr_noise, rate + rate_noise


    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def define_model(self, T, from_trace=None):

        if from_trace is not None:
            raise DeprecationWarning("from_trace is deprecated")

        params = { name: self.getparam(name) for name in self.param_names }

        for k in self.data:
            self.define_data(k, self.data[k])

        slr, rate = self.calc_tensor(T, **params)

        save_private_var(f"{self.label}_slr_trend", slr)
        save_private_var(f"{self.label}_rate_trend", rate)

        assert slr.ndim == 2, _shape(slr)
        assert rate.ndim == 2, _shape(rate)

        if self.add_noise:
            _, _, slr, rate = self._generate_noise(slr, rate, **self.noise_kw)
            assert slr.ndim == 2, _shape(slr)
            assert rate.ndim == 2, _shape(rate)

        save_private_var(f"{self.label}_slr", slr)
        save_private_var(f"{self.label}_rate", rate)

        assert slr.ndim == 2, _shape(slr)
        assert rate.ndim == 2, _shape(rate)

        # apply the constraints, if they are defined here (usually not unless the class is standalone)
        for c in self.constraints:
            logger.info(f"apply constraint: {c}")
            c.apply_model()

        return slr, rate


class DeferredDist:
    """deferring until a model context is available"""
    def __init__(self, name, dist_name, args, **kwargs):
        self.name = name
        self.dist_name = dist_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model=None):
        model = pm.modelcontext(model)
        if hasattr(model, self.name):
            return getattr(model, self.name)

        else:
            return getattr(pm, self.dist_name)(self.name, *self.args, **self.kwargs)

    def __multiply__(self, other):
        return lambda model=None: self() * other

    def __pow__(self, power):
        return lambda model=None: self() ** power


class ScaledObsSd:
    def __init__(self, obs_sd, scale):
        self.obs_sd = obs_sd
        self.scale = scale

    def __call__(self, model=None):

        if callable(self.scale):
            self.scale = self.scale(model)

        return self.obs_sd * self.scale


def save_private_var(name, value, model=None):
    model = pm.modelcontext(model)
    if not hasattr(model, "_private_named_vars"):
        model._private_named_vars = {}
    model._private_named_vars[name] = value


def getvar(name, model=None):
    """Get a variable from the model's private named variables or its attributes"""
    model = pm.modelcontext(model)

    if hasattr(model, "_private_named_vars") and name in model._private_named_vars:
        return model._private_named_vars[name]

    elif hasattr(model, name):
        return getattr(model, name)

    all_names = set.union(set(model.named_vars), set(getattr(model, '_private_named_vars', {})))
    raise ValueError(f"Variable {name} not found in model. Available variables: {', '.join(sorted(all_names))}")


class GenericConstraint:
    def getvar(self, name, model=None):
        """Get a variable from the model's private named variables or its attributes"""
        return getvar(name, model)

    def __call__(self, slr, rate, model=None):
        """This is a back-compatible port to old, per component interface.
        """
        return self.apply_on_model(model)

    def apply_on_model(self, model=None):
        """The constraint is applied on the pymc model. Necessary variables must be saved appropriately.
        """
        raise NotImplementedError("This method should be implemented in subclasses")