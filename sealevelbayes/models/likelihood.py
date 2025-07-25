## here attempt to write a class-based form, for more flexibility then
import numpy as np
import xarray as xa

import pytensor # type: ignore
import pytensor.tensor as pt # type: ignore
import pymc as pm # type: ignore

from sealevelbayes.logs import logger
from sealevelbayes.models.compat import getmodeldata
from sealevelbayes.datasets import get_datapath
from sealevelbayes.datasets.satellite import get_satellite_timeseries
from sealevelbayes.datasets.tidegaugeobs import load_tidegauge_records
from sealevelbayes.datasets.hammond2021 import load_tidegauge_rates as load_gps_rate_at_tidegauges, update_rates as update_gps_rate_at_tidegages
from sealevelbayes.preproc.linalg import calc_lineartrend_fast, detrend_timeseries


def pytensor_lintrend(series, mask=None, A=None):
    # like calc_linear_trend but applies to (py)tensor objects
    n = series.shape[0].eval()
    if A is None:
        A = np.array([np.arange(n) - n//2, np.ones(n)]).T
    if mask is not None:
        imask = np.where(~mask)[0]
        A = A[imask]
        series = series[imask]
    P = np.linalg.pinv(A.T@A)
    x = (P@A.T) @ series
    return x[0]


######## Constraints

MISSING = 1e7

def load_satellite_data(model, observed_mask, load_all=True):
    """Load observation data related to satellite altimeter and tidegauges. Will be saved to the trace.
    """
    if hasattr(model, "satellite_observed"):
        logger.info('satellite data already loaded')
        # assert ((getmodeldata("satellite_observed", model) > 0) == observed_mask).all()
        assert np.isfinite(getmodeldata("satellite_mm", model)[:, observed_mask]).all()
        return

    # Load satellite data everywhere, regardless of the required observed mask ==> that will make post-processing and validation easier.
    if load_all:
        lons = getmodeldata("lons", model)
        lats = getmodeldata("lats", model)
    else:
        lons = getmodeldata("lons", model)[observed_mask]
        lats = getmodeldata("lats", model)[observed_mask]

    logger.info('Load satellite altimeter data...')
    years, values_ = get_satellite_timeseries(lons, lats)
    values_ *= 1000  # in mm
    detrended_, coefs_ = detrend_timeseries(values_.T, n=1, return_coefs=True);  # Remove linear trend

    if load_all:
        values = values_
        trend = coefs_[0]
        detrended = detrended_

    else:
        # now put these data in shape so that it matches the 'station' dimension
        fullsize = getmodeldata("lons", model).size;

        values = np.empty((fullsize, years.size));
        values.fill(MISSING);
        values[observed_mask] = values_;

        detrended = np.empty((years.size, fullsize));
        detrended.fill(MISSING);
        detrended[:, observed_mask] = detrended_;

        trend = np.empty(fullsize);
        trend.fill(MISSING);
        trend[observed_mask] = coefs_[0];

    with model: model.add_coord('satellite_year', years);
    with model: model.add_coord('satellite_year_diff', years[1:]);

    with model: pm.ConstantData("satellite_observed", observed_mask, dims=("station"));
    with model: pm.ConstantData("satellite_mm", values.T, dims=("satellite_year", "station"));
    with model: pm.ConstantData("satellite_detrended", detrended, dims=("satellite_year", "station"));
    with model: pm.ConstantData("satellite_trend", trend, dims=("station"));

    logger.info('...satellite altimeter data loaded')


def load_tidegauge_data(model, observed_mask, year_min=None, year_max=None, records=None, **kw):

    if hasattr(model, "tidegauge_observed"):
        logger.info('tidegauge data already loaded')
        assert ((getmodeldata("tidegauge_observed", model) > 0) == observed_mask).all()
        return

    psmsl_ids = getmodeldata("psmsl_ids", model)[observed_mask]

    assert (psmsl_ids > 0).all()

    if psmsl_ids[psmsl_ids > 0].size == 0:
        return

    # NOTE: for now use the mask attribute of tidegauge to observe tide-gauge data where they are present
    # Probably better, later, to use a smaller, "tidegauge" dimension
    if records is None:
        records = load_tidegauge_records(psmsl_ids, **kw)
    records = records.reindex(getmodeldata("station_ids", model), axis=1)
    if year_min:
        records = records.loc[year_min:]
    if year_max:
        records = records.loc[:year_max]

    model.add_coord("tidegauge_year", records.index)
    pm.ConstantData("tidegauge_ids", records.columns, dims=("station"))
    pm.ConstantData("tidegauge_observed", observed_mask, dims=("station"))
    # pm.ConstantData("tidegauge_start", records.index[np.argmax(np.isfinite(records.values), axis=0)], dims=("station"))
    mask = np.isnan(records.values)
    pm.ConstantData("tidegauge_mask", mask, dims=("tidegauge_year", "station"))
    pm.ConstantData("tidegauge_mm", np.ma.array(records.values, mask=mask).filled(MISSING), dims=("tidegauge_year", "station"))


######## Constraints

def get_tidegauge_masked_array(model, year_start=None, year_end=None):

    if not hasattr(model, 'tidegauge_mm'):
        raise ValueError('no tidegauge data loaded -- use load_tidegauge_data function first')

    # seems like version 5.5 does not store masked arrays, and does not store bool (converted to float)
    tg_obs = np.ma.array(getmodeldata("tidegauge_mm", model), mask=getmodeldata("tidegauge_mask", model).copy() > 0.5)
    tg_years = np.asarray(model.coords['tidegauge_year'])
    if year_start is not None:
        tg_obs.mask[tg_years < year_start] = True
    if year_end is not None:
        tg_obs.mask[tg_years > year_end] = True

    model_years = np.asarray(model.coords['year'])
    assert model_years[0] == tg_years[0], 'first year of tide-gauge data must match first year of model -> need fix'

    return tg_obs


def apply_gps_constraints(rad, stations, observed_mask=None, dummy=False,
    interpolation_method="roughness", dist="normal", model=None, gps_distance_tol=.1, quality=None, scale_error=1,
    gps_rates=None, gps_rate_errors=None):

    model = pm.modelcontext(model)
    n = len(model.coords['station'])

    psmsl_ids = getmodeldata("psmsl_ids", model)
    available_domain = psmsl_ids > 0

    if observed_mask is None:
        observed_mask = available_domain

    # if (psmsl_ids[observed_mask] == 0).any():
        # raise NotImplementedError('GPS constraint for lon/lat points not implemented')

    tgrates = load_gps_rate_at_tidegauges()
    tgrates_by_ID = {r['ID']:r for r in tgrates}

    if False:
        logger.info("Original Hammond et al 2021 rates")
    else:
        logger.info(f"Update Hammond et al 2021 rates: {interpolation_method}")
        update_gps_rate_at_tidegages(tgrates, method=interpolation_method, gauge_dist_tol=gps_distance_tol)  # recalculate the error trend, use centered and filtered GPS stations

    vus = np.empty(n)
    svus = np.empty(n)
    colocated = np.zeros(n, dtype=bool)
    quality_ok = np.ones(n, dtype=bool)
    # only used in mvnormal mode:
    svus_local = np.empty(n)
    svus_correlated = np.empty(n)

    MISSING_SVU = 100   # very large error bars to effectively remove any constraint here

    def _set_missing(k):
        vus[k] = 0
        svus[k] = MISSING_SVU
        svus_local[k] = MISSING_SVU
        svus_correlated[k] = 0  # we don't need correlated error here
        observed_mask[k] = False
        quality_ok[k] = False

    for k, station in enumerate(stations):

        if not observed_mask[k]:
            _set_missing(k)
            continue

        # GPS rates were provided as input
        if gps_rates is not None and gps_rates[k] is not None:
            vus[k] = gps_rates[k]
            if gps_rate_errors is None or gps_rate_errors[k] is None:
                raise ValueError(f"gps_rate_errors must be provided if gps_rates is provided: {k}: {station}")
            svus[k] = gps_rate_errors[k]
            colocated[k] = True # just to avoid all filterings
            quality_ok[k] = True
            continue

        # select all GPS rates in that area
        if not available_domain[k]: # define also outside the observed mask, for diagnostic purpose
            _set_missing(k)
            continue

        # At the moment we support only one GPS record per station. That might change in the future.
        ID = psmsl_ids[k]

        try:
            r = tgrates_by_ID[ID]
        except KeyError:
            logger.warning(f"missing VLM record for {ID}. Skip.")
            _set_missing(k)
            continue

        colocated[k] = r.get('colocated', False)
        vus[k] = r['vu']
        svus[k] = r['svu']

        if quality is not None:
            if r['quality'].lower() not in quality:
                _set_missing(k)

        # if dist == 'mvnormal':
        #     svus_local[k] = r['svu_local']
        #     svus_correlated[k] = r['svu_smooth']

    if quality is not None:
        logger.info(f"Quality filter: {quality} => {quality_ok.sum()} GPS stations out of {quality_ok.size} are used.")

    if interpolation_method == "no-interp":
        observed_mask[~colocated] = False
        _set_missing(~colocated)
        logger.info(f"No interpolation: only {colocated.sum()} colocated GPS stations out of {colocated.size} are used.")

    if dist == "mvnormal":
        gpserrorpath = get_datapath(f'savedwork/gps_error_corr_{interpolation_method}.nc')
        with xa.open_dataset(gpserrorpath) as ds:
            ids = np.array([r['ID'] for r in stations])
            gps_error_cov = ds['gps_error_cov'].reindex(id=ids, co_id=ids).load().values

        # break the correlation for missing or colocated locations
        missing = np.isnan(np.diag(gps_error_cov))
        sd = np.diag(gps_error_cov)**.5
        decorrelate = missing | colocated | ~observed_mask | ~quality_ok
        sd[decorrelate] = svus[decorrelate]
        gps_error_cov[decorrelate, :] = 0
        gps_error_cov[:, decorrelate] = 0

        # here we re-use the COV instead of SVU, because we did a proper MC simulation and that might differ from svu
        # here consistency between diagonal and off-diagonal elements is key
        svus = sd
        np.fill_diagonal(gps_error_cov, svus**2)
        gps_error_cov *= scale_error

    logger.info(f"GPS constraint: scale error by {scale_error}")
    svus *= scale_error

    # we're working at the station level (average of neighboring tide-gauges)
    # use the mean rate over 2000-2020 (90% of GPS time-series start in 2002)
    modelled_vlm = pt.mean(rad[2000-1900:2020-1900+1], axis=0)

    if not hasattr(model, "gps"):
        modelled_vlm = pm.Deterministic("gps", modelled_vlm, dims="station")
        pm.ConstantData("gps_mu", vus, dims="station")
        pm.ConstantData("gps_sd", svus, dims="station")
        pm.ConstantData("gps_mask", observed_mask + 0., dims="station")
        if dist == "mvnormal":
            if "co_station" not in model.coords:
                model.add_coord("co_station", np.arange(n))
            pm.ConstantData("gps_error_cov", gps_error_cov, dims=("station", "co_station"))
    else:
        # This is used in cases where GPS is added a posterior for posterior predictive sampling
        # after a run where it did not actually enter in the constraints
        logger.warning("gps already defined")
        modelled_vlm = model.gps

    assert not (vus == 0).all()

    # legacy fix for previous versions:
    if not hasattr(model, "gps_mu"):
        pm.ConstantData("gps_mu", vus, dims="station")
        pm.ConstantData("gps_sd", svus, dims="station")
        pm.ConstantData("gps_mask", observed_mask + 0., dims="station")


    # Missing data handled by a masked array
    if not dummy:
        logger.info(f"GPS distribution: {dist}")

        svus[~observed_mask] = MISSING_SVU

        if dist == "normal":
            pm.Normal("gps_obs", mu=modelled_vlm, sigma=svus, observed=vus, dims="station")

        elif dist == "mvnormal":
            # raise NotImplementedError(dist)
            # with xa.open_dataset(get_datapath("savedwork/gps_error_corr.nc")) as ds:
            #     ids = np.array([r['ID'] for r in stations])
            #     gps_error_corr = ds['gps_error_corr'].reindex(id=ids, co_id=ids, fill_value=0).load().values
            tiny_regularization = (0.01)**2  # residual "measurement" error of 0.01 mm/yr to regularize the matrix
            # svus_correlated[~observed_mask] = 0
            # svus_local[~observed_mask] = MISSING_SVU
            # cov = svus_correlated[:, None]*gps_error_corr*svus_correlated[None, :] + np.diag(svus_local**2 + tiny_regularization)
            chol = np.linalg.cholesky(gps_error_cov + np.diag(np.zeros_like(svus) + tiny_regularization))
            pm.MvNormal("gps_obs", mu=modelled_vlm, chol=chol, observed=vus, dims="station")

        elif dist == "cauchy":
            pm.Cauchy("gps_obs", alpha=modelled_vlm, beta=svus, observed=vus, dims="station")

        elif dist == "mixed":
            col = pm.Normal("gps_obs_colocated", mu=modelled_vlm[colocated], sigma=svus[colocated], observed=vus[colocated])
            not_col = pm.Cauchy("gps_obs_smooth", alpha=modelled_vlm[~colocated], beta=svus[~colocated], observed=vus[~colocated])
            # create merged field for easier postprocessing:
            class Index:
                def __init__(self):
                    self.count = 0
                def get(self):
                    c = self.count
                    self.count += 1
                    return c
            col_index = Index()
            not_col_index = Index()
            pm.Deterministic("gps_obs", pt.stack([col[col_index.get()] if colocated[i] else not_col[not_col_index.get()] for i in range(colocated.size)]), dims="station")

        else:
            raise NotImplementedError()
    else:
        logger.info("Do not apply GPS error")



class Constraint:
    def apply(self, model):
        logger.warning(DeprecationWarning("Constraint.apply is deprecated, use apply_model instead"))
        return self.apply_model(model)

    def apply_model(self, model):
        raise NotImplementedError("apply_model must be implemented in the subclass")


class GPSConstraint(Constraint):
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def apply_model(self, model):
        apply_gps_constraints(model._save["rad"]["total"][0], model._save["stations"], **vars(self))


class StationConstraint(Constraint):
    def __init__(self, name, observed_mask, dim="station", skip_likelihood=False,
                 save_to_trace=True, oceandyn_surrogate_sampler=None, scale_cov=None,
                 scale_oceandyn_samples=None,
                 oceandyn_cov=None, measurement_error=0.1, cov=None, data=None):
        """
        measurement_error : will be added to the diagonal of the covariance matrix (by default 0.1, as a way to regularize the otherwise likely non-invertible cov matrix)
        """
        self.name = name
        self.data = data
        self._dim = dim
        self.skip_likelihood = skip_likelihood
        self.save_to_trace = save_to_trace
        self.scale_cov = scale_cov
        self.observed_mask = observed_mask
        if observed_mask is None or observed_mask.all():
            self.station_mask = None
        else:
            self.station_mask = ~observed_mask
        self.oceandyn_cov = oceandyn_cov
        self.cov = cov
        self.measurement_error = measurement_error
        if observed_mask is not None and isinstance(self.measurement_error, np.ndarray):
            assert self.measurement_error.size == observed_mask.sum(), 'measurement_error must match the observed values (obseverd_mask == True)'
        self.oceandyn_surrogate_sampler = oceandyn_surrogate_sampler
        self.scale_oceandyn_samples = scale_oceandyn_samples
        self._observe_numpy_f = None

    def _compile_observe_function(self):
        x = pt.dmatrix('x')
        y = self.observe_model(x)
        self._observe_numpy_f = pytensor.function([x], y)

    def _observe_numpy(self, array):
        if self._observe_numpy_f is None:
            self._compile_observe_function()
        return self._observe_numpy_f(array)

    @property
    def dim(self):
        return self._dim if self.station_mask is None else (self.name + "_" + self._dim)

    def define_coords(self, model):

        if self.dim not in model.coords:
            model.add_coord(self.dim, self.get_coord_value(model))
        if "co_" + self.dim not in model.coords:
            model.add_coord("co_" + self.dim, self.get_coord_value(model))

    def get_coord_value(self, model):
        values = np.asarray(model.coords[self._dim])
        if self.station_mask is not None:
            values = values[~self.station_mask]
        return values

    def size(self, model):
        return len(self.get_coord_value(model))
        # raise NotImplementedError(self.dim)

    # def _make_np_array_full_dim(self, array):
    #     full = np.empty(self.observed_mask.size)
    #     full.fill(MISSING)  # nans not supported in ConstantData
    #     full[observed_mask] = array
    #     return full

    # def _make_tensor_full_dim(self, array):
    #     # full = np.empty(self.observed_mask.size)
    #     t = pt.zeros(shape=array.shape) + np.nan
    #     t = pt.set_subtensor(t[observed_mask], array)
    #     return t

    def register(self, model, modelarray=None, obsarray=None, sd=None, cov=None):

        self.define_coords(model)
        model = pm.modelcontext(model)

        with model:


            # register to trace (Constant Data does not cost much)
            if obsarray is not None and not hasattr(model, f"{self.name}_mu"):
                pm.ConstantData(f"{self.name}_mu", obsarray, dims=self.dim)
            if cov is not None and isinstance(cov, np.ndarray) and not hasattr(model, f"{self.name}_cov"):
                pm.ConstantData(f"{self.name}_cov", cov, dims=(self.dim, "co_"+self.dim))
            if sd is None and cov is not None and isinstance(cov, np.ndarray):
                    sd = np.diag(cov)**.5
            if sd is not None and not hasattr(model, f"{self.name}_sd"):
                pm.ConstantData(f"{self.name}_sd", sd, dims=self.dim)

            if modelarray is not None and self.save_to_trace and not hasattr(model, self.name):
                logger.info(f"Define model variable {self.name}")
                pm.Deterministic(self.name, modelarray, dims=self.dim)

    def apply_model(self, model):
        obsarray = self.get_obs_array(model)
        if obsarray.size == 0:
            return
        cov = self.get_covariance_matrix(model) if not self.skip_likelihood else None
        modelarray = self.get_model_array(model)
        self.register(model, modelarray=modelarray, obsarray=obsarray, cov=cov)
        if self.skip_likelihood:
            return

        # That's to define the scale_cov definition outside model conctext (do that in a function)
        if self.scale_cov is not None:
            if callable(self.scale_cov):
                self.scale_cov = self.scale_cov(model)

        if isinstance(cov, np.ndarray):
            chol = np.linalg.cholesky(cov)
            # the cholesky factor can be scaled
            if self.scale_cov is not None:
                chol *= (self.scale_cov)**.5
            logger.info(f"Define observation {self.name}_obs")
            like = pm.MvNormal(f"{self.name}_obs", mu=modelarray, chol=chol, observed=obsarray, dims=self.dim)
        else:
            if self.scale_cov is not None:
                cov *= self.scale_cov
            logger.info(f"Define observation {self.name}_obs")
            like = pm.MvNormal(f"{self.name}_obs", mu=modelarray, cov=cov, observed=obsarray, dims=self.dim)
        return like

    def get_obs_array(self, model):
        raise NotImplementedError()

    def get_model_observable(self, model):
        """Typically this returns the model's time x space field for the relevant variable
        """
        raise NotImplementedError()

    def observe_model(self, observable):
        """Typically this compute trends or else from time x space field as returned by get_model_observable

        The separate is useful to oberve a surrogate time-series and compute covariance matrices
        """
        raise NotImplementedError()

    def observe_surrogate(self, surrogate):
        raise NotImplementedError()

    def get_model_array(self, model):
        return self.observe_model(self.get_model_observable(model))

    def sample_oceandyn_errors(self, model):
        self.get_obs_array(model) # initialize stuff

        if self.oceandyn_surrogate_sampler is None:
            raise ValueError(f"{self}:: Must provide oceandyn_surrogate_sampler or oceandyn_cov")

        logger.info(f"{self}: compute ocean covariance matrix...")
        ## WARNING: the MixedConstraint relies on this step and extends the observe_surrogate method
        samples = self.oceandyn_surrogate_sampler.samples(self.observe_surrogate)

        if type(samples).__name__ == "generator":
            samples = list(samples)

        # This is used to adjust the samples with T.G. to Sat variance**.5 ratio
        if self.scale_oceandyn_samples is not None:

            logger.info("Scale oceandyn samples")
            # Composite sampler such as MeanSampler
            if hasattr(self.oceandyn_surrogate_sampler, "samplers"):
                samples = [modelsamples * self.scale_oceandyn_samples for modelsamples in samples]

            else:
                samples = np.asarray(samples)
                samples *= self.scale_oceandyn_samples

        return samples

    def _crunch_oceandyn_covariances(self, model):
        if self.oceandyn_cov is not None:
            return

        samples = self.sample_oceandyn_errors(model)
        self.oceandyn_cov = self.oceandyn_surrogate_sampler.cov(samples)
        self.oceandyn_std = self.oceandyn_surrogate_sampler.std(samples)
        logger.info(f"{self}: compute ocean covariance matrix...done")

    def _crunch_measurement_error(self, model):
        pass

    def _crunch_all(self, model):
        model = pm.modelcontext(model)
        self._crunch_oceandyn_covariances(model)
        self._crunch_measurement_error(model)

    def get_covariance_matrix(self, model):
        if self.cov is not None: return self.cov  # in case it is provided as input
        self._crunch_all(model)
        covdyn = self.oceandyn_cov
        return covdyn + np.diag(np.zeros(covdyn.shape[0]) + self.measurement_error**2)

    def get_standard_deviation(self, model=None):
        self._crunch_all(model)
        return (self.oceandyn_std**2 + self.measurement_error**2)**.5

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"

    def __str__(self):
        return f"{type(self).__name__}::{self.name}"



class TideGaugeConstraint(StationConstraint):
    def __init__(self, name, observed_mask, year_start=None, year_end=None, **kwargs):
        self.year_start = year_start
        self.year_end = year_end
        super().__init__(name, observed_mask, **kwargs)



class TideGaugeTrend(TideGaugeConstraint):
    def __init__(self, name, observed_mask, model_mean_rate_instead_of_lintrend=False, **kwargs):
        # Here we use the linear trends
        # This is a stronger constraint than mean rate because it assumes a perfect vertical datum adjustment of the tidegauge record across gaps
        logger.info("Tidegauge constraints based on full TG trend")
        self.model_mean_rate_instead_of_lintrend = model_mean_rate_instead_of_lintrend
        super().__init__(name, observed_mask, **kwargs)

    def get_obs_array(self, model):
        # Load corresponding obs constraints
        load_tidegauge_data(model, self.observed_mask, self.year_start, self.year_end, records=self.data)
        tg_obs = get_tidegauge_masked_array(model, year_start=self.year_start, year_end=self.year_end)
        self.obs_mask = tg_obs.mask
        if self.obs_mask.all(axis=0).any():
            self.station_mask = self.obs_mask.all(axis=0)
        return self.observe_surrogate(tg_obs.data)

    def observe_surrogate(self, tg_obs):
        # Observational trend from tidegauges
        # return np.array([calc_lineartrend_fast(vals, mask=mask) for vals, mask in zip(tg_obs.T, self.obs_mask.T) if (~mask).any()])
        tg_obs = np.ma.array(tg_obs[:self.obs_mask.shape[0]], mask=self.obs_mask)
        if self.station_mask is not None:
            tg_obs = tg_obs[:, ~self.station_mask]
        return np.array([calc_lineartrend_fast(tg_obs.data[:, i], mask=tg_obs.mask[:, i]) for i in range(tg_obs.shape[1])])

    def get_model_observable(self, model):
        if self.model_mean_rate_instead_of_lintrend:
            return model._save["rsl"]["total"][0]
        else:
            return model._save["rsl"]["total"][0].cumsum(axis=0)  # we operate on the integral form

    def observe_model(self, observable):
        observable = observable[:self.obs_mask.shape[0]]
        if self.station_mask is not None:
            obs_mask = self.obs_mask[:, ~self.station_mask]
            observable = observable[:, np.where(~self.station_mask)[0]]
        else:
            obs_mask = self.obs_mask

        if self.model_mean_rate_instead_of_lintrend:
            # Instead of a calculating the linear trend, it is more efficient to use the mean rate
            # Since the model is smooth, that should not be too far off (though the acceleration does introduce a bias, probably smaller with mean rate)
            # here we use the full TG period from first to last, since even single points are accounted for in the T.G. trend
            stackable = [observable[np.where(~obs_mask[:, i])[0], i].mean() for i in range(obs_mask.shape[1])]
        else:
            stackable = [pytensor_lintrend(observable[:, i], mask=obs_mask[:, i]) for i in range(obs_mask.shape[1])]

        if not stackable: return []
        return pt.stack(stackable)



class SatelliteTrend(TideGaugeTrend):
    def __init__(self, name, **kwargs):
        # Here we use the linear trends
        # This is a stronger constraint than mean rate because it assumes a perfect vertical datum adjustment of the tidegauge record across gaps
        logger.info("Tidegauge constraints based on satellite trend")
        super().__init__(name, **kwargs)


    def get_obs_array(self, model):
        # Load corresponding obs constraints
        load_satellite_data(model, self.observed_mask)
        i1 = model.coords["satellite_year"][0] - model.coords['year'][0]
        i2 = model.coords["satellite_year"][-1] - model.coords['year'][0]
        self.idx = slice(i1, i2+1)
        return calc_lineartrend_fast(getmodeldata("satellite_mm", model)[:, self.observed_mask])

    def observe_surrogate(self, data):
        # Observational trend from tidegauges
        return calc_lineartrend_fast(data[self.idx, np.where(self.observed_mask)[0]])
        # return np.array([calc_lineartrend_fast(data[:, i], mask=self.mask[:, i]) for i in range(n) if not self.mask[:, i].all()])

    def get_model_observable(self, model):
        if self.model_mean_rate_instead_of_lintrend:
            return model._save["gsl"]["total"][0]
        else:
            return model._save["gsl"]["total"][0].cumsum(axis=0)  # we operate on the integral form

    def observe_model(self, observable):
        imask = np.where(self.observed_mask)[0]
        if self.model_mean_rate_instead_of_lintrend:
            return observable[self.idx, imask].mean(axis=0)
        else:
            return pytensor_lintrend(observable[self.idx, imask])


class MixedConstraint(StationConstraint):
    def __init__(self, name, constraints, cov=None, dim='station_mixed', save_to_trace=False, measurement_error=None, **kwargs):
        self.constraints = constraints
        self.cov = cov
        super().__init__(name, dim=dim, observed_mask=None, save_to_trace=save_to_trace,
                         measurement_error=measurement_error, **kwargs)

    def apply_model(self, model):
        like = super().apply_model(model)

        # Get the standard deviations of the individual constraints to save to trace (post-proc)
        sd = self.get_standard_deviation(model)
        start = 0
        for k, c in enumerate(self.constraints):
            obs = c.get_obs_array(model)
            idx = slice(start, start + obs.size)
            # we already defined obs in _get_obs_array_registered. Check:
            np.testing.assert_allclose(obs, getmodeldata(c.name + "_mu", model))
            c.register(model, sd=sd[idx])
            pm.Deterministic(f"{c.name}_obs", like[idx], dims=c.dim) # for sample prior / posterior predictive checks
            start += obs.size

    def _crunch_measurement_error(self, model):
        """Here we fetch the measurement error from all constraints separately
        """
        if self.measurement_error is not None: return
        self.measurement_error = np.empty(self.size(model))
        start = 0
        for c in self.constraints:
            c._crunch_measurement_error(model)
            size = c.size(model)
            self.measurement_error[start:start+size] = c.measurement_error
            start += size

    # def _crunch_oceandyn_covariances(self, model):
    #     """we don't need to do anything here because it all happens in observe_surrogate,
    #     on which the covariance function is called.
    #     """
    #     super()._crunch_oceandyn_covariances(model)

    def _get_obs_array_registered(self, model, c):
        obsarray = c.get_obs_array(model)
        # sd = c.get_standard_deviation(model)
        c.register(model, obsarray=obsarray)
        return obsarray

    def _get_model_array_registered(self, model, c):
        modelarray = c.get_model_array(model)
        c.register(model, modelarray=modelarray)
        return modelarray

    def _get_constraint_size(self, model):
        return [c.get_obs_array(model).size for c in self.constraints]

    def get_obs_array(self, model):
        return np.concatenate([self._get_obs_array_registered(model, c) for c in self.constraints])

    def get_model_array(self, model):
        return pt.concatenate([self._get_model_array_registered(model, c) for c in self.constraints])

    def observe_model(self, observable):
        return pt.concatenate([c.observe_model(observable) for c in self.constraints])

    def observe_surrogate(self, surrogate):
        return np.concatenate([c.observe_surrogate(surrogate) for c in self.constraints])

    def sample_oceandyn_errors(self, model):
        # return np.concatenate([c.sample_oceandyn_errors(model) for c in self.constraints])

        # This is a special case where things are done for each model separately,
        # so samples is not an array to concatenate, but a list of size-incompatible,
        # model-specific sample draws
        if hasattr(self.oceandyn_surrogate_sampler, "samplers"):
            # this was written to combine satellite and TG observations, which share the same oceandyn sampler
            assert all(hasattr(c, "oceandyn_surrogate_sampler") for c in self.constraints), "each constraint must have an oceandyn sampler"
            assert all(hasattr(c.oceandyn_surrogate_sampler, "samplers") for c in self.constraints), "each constraint's oceandyn sampler much be a composite class"
            assert all(len(c.oceandyn_surrogate_sampler.samplers) == len(self.constraints[0].oceandyn_surrogate_sampler.samplers) for c in self.constraints), "each constraint's composite oceandyn sampler much have the same number of models"

            # list(over constraints) of list(over models)
            samples_constraint_first = [c.sample_oceandyn_errors(model) for c in self.constraints]

            # built a list(over models) of concatenated constraints
            samples_models_first = []
            n_models = len(samples_constraint_first[0])
            for i in range(n_models):
                # concatenate on columns (axis=1) because each model samples have shape (pi-control samples, station)
                samples_models_first.append(np.concatenate([s[i] for s in samples_constraint_first], axis=1))

            return samples_models_first

        else:
            return np.concatenate([c.sample_oceandyn_errors(model) for c in self.constraints])

    def get_coord_value(self, model):
        return [f"{c.name}_{id}" for c in self.constraints for id in c.get_coord_value(model)]

    def __str__(self):
        return f"Mixed({self.name}, {', '.join(str(c) for c in self.constraints)})"