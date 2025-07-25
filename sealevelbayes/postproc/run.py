#!/usr/bin/env python3
from pathlib import Path
import copy, os
import numpy as np # type: ignore
import json
import xarray as xa # type: ignore
import pymc as pm # type: ignore
import arviz # type: ignore
import cloudpickle  # type: ignore

from sealevelbayes.config import get_runpath
from sealevelbayes.logs import logger

from sealevelbayes.models.compat import getmodeldata
from sealevelbayes.models.localslr import slr_model_tidegauges, slr_model_tidegauges_given_global_posterior, make_diagnostic
from sealevelbayes.models.localslr import SOURCES, DIAGS, FIELDS

from sealevelbayes.runslr import get_parser, get_model_kwargs, get_stations, get_local_constraints, get_gps_constraints
from sealevelbayes.runslr import get_model, parse_args
from sealevelbayes.runslr import get_oceandynsampler, ignore_unused_kwargs


def get_model_from_command_args(path, args=(), **kwargs):
    parser = get_parser()
    o = parser.parse_args(args)    # command line arguments
    vars(o).update(kwargs)         # update with key-word arguments (optional)
    return get_model(o)[0]


def get_model_from_folder(dirname, args=(), **kwargs):
    pickled = cloudpickle.load(open(Path(dirname)/"config.cpk", 'rb'))
    return get_model_from_pickled(pickled, args, **kwargs)


def get_model_from_pickled(pickled, args=(), model_kwargs={}, **kwargs):
    if not args and not kwargs:
        return pickled["model"]
    parser = get_parser()
    options = vars(pickled['options'])

    _check_back_compatibility(options)

    o = parser.parse_args(args)               # initialize with defaults arguments (to load an earlier version with less variables)
    vars(o).update(options)  # update with pickled arguments
    vars(o).update(kwargs)                    # update with key-word arguments
    return get_model(o, model_kwargs=model_kwargs)[0]


def _check_back_compatibility(options):
    """useful when renaming parameters or adding new ones or removing parameters
    """
    pass


RENAME_POSTERIOR_MAPPING = {
    **{f"{source}_gp_noise": f"{source}_noise" for source in ["gis", "ais", "steric"]},
    **{f"{source}_gp_noise_rotated_": f"{source}_noise_rotated_" for source in ["gis", "ais", "steric"]},
    # "_antarctica_iid": "_fred_ais_iid",
}


class ExperimentTrace:
    def __init__(self, trace, options, model=None):
        self.trace = trace
        if self.trace is not None:
            rename_ = {k:v for k,v in RENAME_POSTERIOR_MAPPING.items() if k in self.trace.posterior}
            if len(rename_) > 0:
                logger.warning(f"renaming {len(rename_)} variables in trace: {rename_}")
                self.trace = self.trace.rename(rename_, groups=["posterior"])
        # self.options = options
        self.model = model

        _check_back_compatibility(options)

        o = parse_args([])    # command line arguments
        vars(o).update(options)        # update with key-word arguments
        self.o = o

    @classmethod
    def load(cls, cirun):
        runfolder = get_runpath(cirun)
        try:
            pickled = cloudpickle.load(open(runfolder/"config.cpk", "rb"))
            options = dict(vars(pickled['options']))
            model = pickled['model']
        except Exception as error:
            logger.warning("could not load pickled config")

            if (runfolder/"options.json").exists():
                logger.info("load options.json")
                options = json.load(open(runfolder/"options.json"))
            else:
                options = {}

            model = None

        # folder-renaming safe !
        options["cirun"] = cirun
        options["dirname"] = runfolder

        trace = arviz.from_netcdf(runfolder/"trace.nc")
        return cls(trace, options, model=model)

    @property
    def cirun(self):
        return self.o.cirun

    def get_data(self, station_ids=None, tas=None):
        if station_ids is None:
            return self.trace.constant_data

        if station_ids is not None:
            try:
                return self.trace.constant_data.sel(station=station_ids)
            except:
                # empty dataset => force reload
                return xa.Dataset(coords={k:v if k != 'station' else station_ids for k,v in self.trace.constant_data.items()})

        else:
            global_data = [v for v in self.trace.constant_data if 'station' not in self.trace.constant_data[v].dims]
            return self.trace.constant_data[global_data]


    def get_model(self, sources=['total'], diags=['change'], fields=['rsl'], experiments=None, station_ids=None, psmsl_ids=None, year_frequency=1, save_timeseries=True, tas=None, no_data=False, coordinates=None, isimip_model=None, update_runslr_params=None, **kwargs):

        o = copy.deepcopy(self.o)

        if update_runslr_params is not None:
            for k in update_runslr_params:
                if k not in vars(o):
                    raise ValueError(f"update_runslr_params: {k} not in options")
            vars(o).update(update_runslr_params)

        trace = self.trace

        # no need the constraints for resampling (free RVs obtained from trace)
        o.skip_all_constraints = True

        if coordinates is not None:
            o.coords = coordinates

        if psmsl_ids is not None: o.psmsl_ids = psmsl_ids

        if station_ids is not None:
            o.station_ids = station_ids

        if experiments is not None: o.experiments = np.asarray(experiments).tolist()
        if isimip_model is not None: o.isimip_model = isimip_model

        kwargs = get_model_kwargs(o)

        if sources is not None: kwargs['diag_kwargs']["sources"] = sources
        if diags is not None: kwargs['diag_kwargs']["diags"] = diags
        if fields is not None: kwargs['diag_kwargs']["fields"] = fields
        kwargs['diag_kwargs']["year_frequency"] = year_frequency
        kwargs['diag_kwargs']["save_timeseries"] = save_timeseries

        station_ids = np.array([station['ID'] for station in kwargs['stations']])
        data = self.get_data(station_ids=station_ids)

        if tas is not None:
            if np.ndim(tas) == 2:
                data['tas'] = tas if isinstance(tas, xa.DataArray) else (("experiment", "year"), tas)
            else:
                kwargs['global_slr_kwargs']["resample"] = True

        if len(data) == 1:
            assert 'tas' in data
            raise NotImplementedError('need to update tidegaugemodels to handle the case of partially filled data')

        elif len(data) == 0:
            # right now the model only sees fully defined data or None
            data = None

        # resampling settings
        kwargs['slr_kwargs'].setdefault('steric_kwargs', {})
        # in the reduced space coef case, we resample in the reduce dimension, so covariances are not an issue
        # kwargs['slr_kwargs']['steric_kwargs']['steric_coef_cov'] = False
        kwargs['slr_kwargs']['steric_kwargs']['steric_coef_cov'] = kwargs['slr_kwargs']['steric_kwargs'].get('reduced_space_coef')
        kwargs['slr_kwargs']['vlm_res_mode'] = "dummy"  # yearly vlm_res taken directly (otherwise _vlm_res_iid causes issues or other)

        print("kwargs", kwargs["slr_kwargs"].get("vlm_res_mode"))
        return slr_model_tidegauges(data=None if no_data else data, **kwargs)


    def resample_prior_global(self, sources=['total'], diags=['change'], var_names=None, extend_var_names=[], experiments=None, tas=None, no_data=True, samples=4000, sample_kwargs={}, **kwargs):

        fields = ['global']
        model = self.get_model(sources=sources, diags=diags, fields=fields, experiments=experiments, psmsl_ids=[], tas=tas, no_data=no_data, **kwargs)

        with model:
            if var_names is not None:
                diag_var_names = var_names
            else:
                diag_var_names = make_diagnostic(diags=diags, fields=fields, sources=sources)
                if not diag_var_names:
                    make_diagnostic(diags=diags, fields=fields, sources=sources, verbose=True)
                    raise ValueError("No value sample (likely error in sources / diags / fields)")
            if extend_var_names:
                diag_var_names.extend(extend_var_names)
            trace_proj = pm.sample_prior_predictive(var_names=diag_var_names, samples=samples, **sample_kwargs)

        return trace_proj


    def resample_prior(self, sources=['total'], diags=['change'], fields=['rsl'], var_names=None, extend_var_names=[], experiments=None,
                       psmsl_ids=None, tas=None, no_data=True, samples=4000, sample_kwargs={}, **kwargs):

        if var_names is None and (fields == ['global'] or sources == ['tas']):
            return self.resample_prior_global(sources=sources, diags=diags, experiments=experiments, tas=tas,
                                              no_data=no_data, var_names=var_names, extend_var_names=extend_var_names, samples=samples, sample_kwargs=sample_kwargs, **kwargs)

        model = self.get_model(sources=sources, diags=diags, fields=fields, experiments=experiments, psmsl_ids=psmsl_ids, tas=tas, no_data=no_data, **kwargs)
        # psmsl_ids = np.asarray(model.coords['station'])

        with model:
            if var_names is not None:
                diag_var_names = var_names
            else:
                diag_var_names = make_diagnostic(diags=diags, fields=fields, sources=sources)  # only to collect var names
                if not diag_var_names:
                    make_diagnostic(diags=diags, fields=fields, sources=sources, verbose=True)
                    raise ValueError("No value sample (likely error in sources / diags / fields)")
            if extend_var_names:
                diag_var_names.extend(extend_var_names)
            trace_proj = pm.sample_prior_predictive(var_names=diag_var_names, samples=samples, **sample_kwargs)

        return trace_proj



    def resample_posterior_global(self, sources=['total'], diags=['change'], experiments=None, tas=None, var_names=None, extend_var_names=[], sample_kwargs={}, trace=None, **kwargs):

        fields = ['global']
        model = self.get_model(sources=sources, diags=diags, fields=fields, experiments=experiments, psmsl_ids=[], tas=tas, **kwargs)

        if trace is None:
            trace = self.trace

        free_var_names = [v.name for v in model.free_RVs]
        if "tas" in free_var_names:
            trace.posterior[f"tas"] = trace.posterior[f"tas_factor"].dims + ("experiment", "year"), np.asarray(tas)

        trace_free_RVs = trace.posterior[free_var_names].load()


        with model:
            if var_names is not None:
                diag_var_names = var_names
            else:
                diag_var_names = make_diagnostic(diags=diags, fields=fields, sources=sources)
                if not diag_var_names:
                    make_diagnostic(diags=diags, fields=fields, sources=sources, verbose=True)
                    raise ValueError("No value sample (likely error in sources / diags / fields)")
            if extend_var_names:
                diag_var_names.extend(extend_var_names)
            return pm.sample_posterior_predictive(trace_free_RVs, var_names=diag_var_names, **sample_kwargs)


    def get_free_RVs_trace(self, model):
        """return trace for posterior predictive sampling

        model: model returned by the get_model method
        It accomodates for new stations, provided the model is defined appropriately.
        """

        free_var_names = [v.name for v in model.free_RVs]

        station_ids = np.asarray(model.coords['station'])
        existing_ids = self.trace.posterior.station.values
        station_ids_exist = np.array([ID in set(existing_ids) for ID in station_ids])
        old_station_idx = station_ids[station_ids_exist]
        if old_station_idx.size == 0:
            old_station_idx = slice(0,0)  # empty slice
        trace_free_RVs = self.trace.posterior[free_var_names].sel(station=old_station_idx).load()

        # if some new IDs are required (e.g. via coordinates=), need to define vlm_res at these locations.
        # for now I just set it to zero
        if not station_ids_exist.all():
            new_stations = station_ids[~station_ids_exist]
            card_new = new_stations.size
            logger.warning(f"{card_new} new stations are added in posterior sampling")

            missing_coords = {k:(v if k != 'station' else new_stations) for k,v in trace_free_RVs.coords.items()}
            trace_free_RVs_coords = xa.Dataset(coords={k:(v if k != 'station' else station_ids) for k,v in trace_free_RVs.coords.items()})

            for k,v in trace_free_RVs.items():

                if "station" not in v.dims:
                    trace_free_RVs_coords[k] = v
                    continue

                if k == 'vlm_res':
                    missing_values = np.zeros([s if d != 'station' else card_new for d,s in zip(v.dims, v.shape)])
                    missing_coords = {d: missing_coords[d] for d in v.dims}
                    missing = xa.DataArray(missing_values, dims=v.dims, coords=missing_coords)
                    if v.size == 0:
                        assert (station_ids == new_stations).all()
                        updated = missing
                    else:
                        updated = xa.concat([v, missing], dim='station').sel(station=station_ids)  # concat and re-order
                    trace_free_RVs_coords[k] = updated

                else:
                    raise NotImplementedError(f"cannot expand to new coordinates: {k} : {v.dims}")

            trace_free_RVs = trace_free_RVs_coords

        else:
            logger.info(f"Resample existing stations")

        # # fix for gia_scale_iid variable (first dim was not named as station in a previous version)
        # if "gia_scale_iid_dim_0" in trace_free_RVs.dims:
        #     trace_ids = self.trace.constant_data.psmsl_ids.values.tolist()
        #     i_stations = [trace_ids.index(id) for id in psmsl_ids]
        #     trace_free_RVs = trace_free_RVs.isel(gia_scale_iid_dim_0=i_stations)

        # dummy mode for VLM res resampling -> need to expand to year
        if "vlm_res" in trace_free_RVs and "year" not in trace_free_RVs["vlm_res"].dims:
            trace_free_RVs["vlm_res"] = trace_free_RVs["vlm_res"].expand_dims(year=trace_free_RVs["year"]).transpose(..., "year", "station")

        return trace_free_RVs


    def resample_posterior(self, sources=['total'], diags=['change'], fields=['rsl'], var_names=None,
                           extend_var_names=[], experiments=None, psmsl_ids=None, tas=None, coordinates=None, sample_kwargs={}, add_constraints=[], **kwargs):

        if sources is None: sources = SOURCES

        if fields == ['global'] or sources == ['tas']:
            return self.resample_posterior_global(sources=sources, diags=diags, experiments=experiments, tas=tas,
                                                  var_names=var_names, extend_var_names=extend_var_names, sample_kwargs=sample_kwargs, **kwargs)

        if var_names is not None:
            # make sure everything is defined in the model
            sources = SOURCES
            fields = FIELDS
            diags = DIAGS

        model = self.get_model(sources=sources, diags=diags, fields=fields, experiments=experiments, psmsl_ids=psmsl_ids, tas=tas, coordinates=coordinates, **kwargs)

        if add_constraints:
            with model:
                for c in add_constraints:
                    if not hasattr(c, "apply"):
                        raise ValueError(f"add_constraints: {c} is not a constraint")
                    c.apply_model(model)
                    extend_var_names = extend_var_names + [c.name, c.name + "_obs"]

        trace_free_RVs = self.get_free_RVs_trace(model)

        with model:
            if var_names is None:
                var_names = make_diagnostic(diags=diags, fields=fields, sources=sources)  # only to collect var names
            if not var_names:
                make_diagnostic(diags=diags, fields=fields, sources=sources, verbose=True)
                raise ValueError("No value sample (likely error in sources / diags / fields)")
            if extend_var_names:
                var_names.extend(extend_var_names)
            trace_proj = pm.sample_posterior_predictive(trace_free_RVs, var_names=var_names, **sample_kwargs)

        return arviz.InferenceData(posterior=trace_proj.posterior_predictive, constant_data=trace_proj.constant_data, sample_stats=self.trace.sample_stats)


    def add_missing_local_constraints(self, model=None, observations=["satellite", "tidegauge", "gps"]):
        """If not other experiment can be found which already defines those, this is an alternative.
        """
        if model is None:
            model = self.model
        missing_obs = [obs for obs in observations if obs+"_obs" not in model.named_vars]

        if not missing_obs:
            return

        logger.warning(f"add_missing_local_constraints: {missing_obs} not found in the model. Adding them.")

        o2 = copy.deepcopy(self.o)
        o2.skip_constraints = [o for o in observations if o not in missing_obs]
        stations = get_stations(o2)
        constraints = get_local_constraints(o2, stations)

        with model:
            for c in constraints:
                c.apply_model(model)

    def sample_posterior_predictive(self, var_names=None, model=None, **kwargs):
        if model is None:
            model = self.model
        with model:
            # extend_inferencedata=True does not make much sense when we return posterior_predictive
            # ... and it may also be causing a bug (at least in pymc==5.9)
            # return pm.sample_posterior_predictive(self.trace, var_names=var_names, extend_inferencedata=True).posterior_predictive
            return pm.sample_posterior_predictive(self.trace, var_names=var_names, **kwargs).posterior_predictive


    def sample_prior_predictive(self, var_names=None, **kwargs):
        with self.model:
            return pm.sample_prior_predictive(self.trace, var_names=var_names, extend_inferencedata=True, **kwargs).prior_predictive


    def resample_array(self, source='total', diag='change', field='rsl', **kwargs):
        post = self.resample_posterior(sources=[source], diags=[diag], fields=[field], **kwargs).posterior
        return arviz.extract(post)[f"{diag}_{field}_{source}"]


    def resample_sources(self, sources, diag='change', field='rsl', **kwargs):
        post = self.resample_posterior(sources=sources, diags=[diag], fields=[field], **kwargs).posterior
        return arviz.extract(post).rename_vars({f"{diag}_{field}_{source}": source for source in sources})


    def resample_fields(self, fields, source='total', diag='change', **kwargs):
        post = self.resample_posterior(sources=[source], diags=[diag], fields=fields, **kwargs).posterior
        return arviz.extract(post).rename_vars({f"{diag}_{field}_{source}": field for field in fields})


    def add_gps_constraint(self):
        """useful to do posterior predictive sampling
        """
        if not hasattr(self, 'o_orig'):
            self.o_orig = copy.deepcopy(self.o)

        # update the options for functions that define the model on the fly
        if "gps" in self.o.skip_constraints:
            self.o.skip_constraints.remove("gps")

        # ... and update the model itself for functions that use the model as it
        if not hasattr(self.model, "gps_mu"):
            stations = get_stations(self.o)
            [gps_constraint] = get_gps_constraints(self.o, stations)

            with self.model:
                gps_constraint.apply_model(self.model)

        # ... now update constant_data
        if "gps_mu" not in self.trace.constant_data:
            self.trace.constant_data["gps_mu"] = "station", getmodeldata("gps_mu", self.model)

        if "gps_sd" not in self.trace.constant_data:
            self.trace.constant_data["gps_sd"] = "station", getmodeldata("gps_sd", self.model)


    ## HACK TO GET CONSISTENT INFO AT TG LOCATIONS
    def sample_posterior_predictive_at_tidegauges(self, var_names=None, psmsl_ids=None):
        """This function sets satellite_mask argument to "psmsl", to resample from there
        """
        import copy
        o = copy.copy(self.o)
        o.grid = None
        o.satellite_mask = ["psmsl"]
        if psmsl_ids is not None:
            o.psmsl_ids = psmsl_ids

        psmsl_ids = o.psmsl_ids or self.trace.constant_data.psmsl_ids.values[self.trace.constant_data.psmsl_ids.values > 0]

        station_dims = ["tidegauge_station", "satellite_station", "gps_station", "satellite_psmsl_station"]

        def _index_on_psmsl(ds):
            return xa.Dataset({ k.replace("satellite_psmsl", "satellite"): v.rename({d:"station" for d in station_dims if d in v.dims}).sel(station=psmsl_ids)
                for k, v in ds.items() })


        if all(hasattr(self.model, name) for name in var_names if not name.startswith("satellite")):
            # the below cases only deal with satellite

            # no need to redefine the model (and load oceandyn surrogates...) when ....
            # ... satellite mask is defined everywhere
            # ... no satellite is required in resampling
            if 'psmsl' in self.o.satellite_mask or (var_names is not None and not any('satellite' in name for name in var_names)):
                logger.info("sample_posterior_predictive_at_tidegauges: simply re-index satellite")
                return _index_on_psmsl(self.sample_posterior_predictive(var_names=var_names))

            # ... diagnostic satellite constraint satellite_psmsl is defined: request that instead
            elif 'satellite_psmsl' in self.trace.posterior:
                var_names2 = [v.replace("satellite", "satellite_psmsl") if not v.startswith("satellite_psmsl") else v for v in var_names]
                logger.info("sample_posterior_predictive_at_tidegauges: simply re-index satellite_psmsl")
                return _index_on_psmsl(self.sample_posterior_predictive(var_names=var_names2))

        # otherwise need to redefine the model
        logger.info("sample_posterior_predictive_at_tidegauges: redefine the model")
        model_psmsl, stations_psmsl = get_model(o)
        post_psmsl = self.trace.posterior.sel(station=np.array(model_psmsl.coords['station']))

        with model_psmsl:
            trace_psmsl = pm.sample_posterior_predictive(post_psmsl, var_names=var_names)


        return _index_on_psmsl(trace_psmsl.posterior_predictive)

    def get_oceandynsampler(self):
        lons = self.trace.constant_data.lons.values
        lats = self.trace.constant_data.lats.values
        odyn = ignore_unused_kwargs(get_oceandynsampler)(lons, lats, **vars(self.o))


def find_global_standalone_run(cirun=None, runfolder=None):

    if "global-slr-only" in cirun:
        return None

    parts = cirun.split(os.path.sep)
    if parts[-1].startswith("run_"):
        parts[-1] = "run_global-slr-only_" + parts[-1][4:]
        candidate = os.path.sep.join(parts)
        if os.path.exists(get_runpath(candidate)):
            print("CIRUNGLOBAL detected automatically:", candidate)
            return candidate

    return None
