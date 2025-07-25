from pathlib import Path
import cloudpickle
import numpy as np
import xarray as xa
import pymc as pm

from sealevelbayes.logs import logger 
from sealevelbayes.models.compat import getmodeldata
from sealevelbayes.datasets.satellite import get_satellite_timeseries
from sealevelbayes.datasets.oceancmip6 import load_all_cmip6
from sealevelbayes.preproc.oceancovariance import MeanSampler, CMIP6Sampler
from sealevelbayes.models.likelihood import SatelliteTrend, TideGaugeTrend, GPSConstraint
from sealevelbayes.models.localslr import make_diagnostic

# It takes time to generate the constraints, but does not take up too much space, so keep them in a store
# so it can be shared across model versions. Only index by the number of grid points, in case several
# versions are used in // in a script.

CONSTRAINTS_STORE = {}   

class ModelCoords:
    def __init__(self, model):
        # self.model = model
        self.station_ids = station_ids = np.asarray(model.coords['station'])
        self.psmsl_domain = psmsl_domain = station_ids < 10000
        self.coast_domain = coast_domain = station_ids > 10000
        self.psmsl_ids = station_ids[psmsl_domain]
        self.coast_ids = station_ids[coast_domain]
        self.lons = getmodeldata("lons", model)
        self.lats = getmodeldata("lats", model)
        self.coords = np.array([self.lons, self.lats]).T
        self.size = len(self.station_ids)


def define_coastal_obs(m):

    sat_years, sat_values_t = get_satellite_timeseries(m.lons, m.lats)
    sat_values_t *= 1000 # meter to millimeter
    sat_values = sat_values_t.T

    model_data = load_all_cmip6(lon=m.lons, lat=m.lats, max_workers=5)

    minlength = 2023-1900 + 60  # about 60 samples min to estimate the cov matrix (to catch cycles ~ 60 years)
    oceandynsampler = MeanSampler([CMIP6Sampler(zos.values, model=zos.model, sat_values=sat_values, rescale_like_satellite=True) for zos in model_data if zos.shape[0] >= minlength])    
    
    satellite_mask = np.ones(m.size, dtype=bool)
    tidegauge_mask = m.psmsl_domain
    sat_kwargs = dict(observed_mask=satellite_mask, oceandyn_surrogate_sampler=oceandynsampler, measurement_error=0.1)
    tg_kwargs = dict(observed_mask=tidegauge_mask, oceandyn_surrogate_sampler=oceandynsampler, measurement_error=0.1)
    trend_kwargs = dict(model_mean_rate_instead_of_lintrend=False)
    
    sat = SatelliteTrend("satellite", **sat_kwargs, **trend_kwargs)
    # tg = TideGaugeTrend("tidegauge", **tg_kwargs, **trend_kwargs)
    # tg_pre = TideGaugeTrend("tidegauge_1900_1990", year_end=1990, **tg_kwargs, **trend_kwargs)
    tg_post = TideGaugeTrend("tidegauge_1990_2018", year_start=1990, **tg_kwargs, **trend_kwargs)
    
    gpscons = GPSConstraint(observed_mask=m.psmsl_domain)

    return [sat, tg_post, gpscons]
    
def get_coastal_model(tr, continuous_coast, add_constraints=False, isimip_model=None, **kwargs):
    
    folder = Path(tr.o.dirname)/"postproc"
    folder.mkdir(exist_ok=True, parents=True)
    if isimip_model is not None:
        file = folder / f"model_coastline-{len(continuous_coast)}-{isimip_model}.pkl"
    else:
        file = folder / f"model_coastline-{len(continuous_coast)}.pkl"
    if add_constraints:
        STORE_KEY = len(continuous_coast)
    else:
        STORE_KEY = None
    
    if file.exists():
        # return cloudpickle.load(open(file, "rb"))['model']
        print(f"load model {file}")
        pkl = cloudpickle.load(open(file, "rb"))
        model = pkl['model']

        # Check if constraints were added to the model (they should be saved along with it -- thats how we check)
        constraints = pkl.get("constraints",[]) # define constraints in globals

        # Save for future use
        if constraints and STORE_KEY not in CONSTRAINTS_STORE:
            CONSTRAINTS_STORE[STORE_KEY] = constraints

        # all is done already
        if add_constraints and constraints:
            return model

    else:
        # NOTE: this returns a model whose coordinate is 1) original PSMSL + 2) continuous_coast
        print(f"crunch model")

        model = tr.get_model(coordinates=continuous_coast, isimip_model=isimip_model, **kwargs)
        constraints = []

    # return the simple coastline model (without the addition of constraints)
    if not add_constraints:
        # save a partial version
        if not file.exists():
            cloudpickle.dump({"model": model}, open(file, "wb"))
        return model


    # If add_constraints is False: STORE_KEY=None is always present and matches []
    if STORE_KEY in CONSTRAINTS_STORE:
        logger.info("Re-use constraints from global")
        constraints = CONSTRAINTS_STORE[STORE_KEY]

    else:
        logger.info("Define constraints")
        m = ModelCoords(model)        
        CONSTRAINTS_STORE[STORE_KEY] = constraints = define_coastal_obs(m) # here the model is only passed for coordinate info

    # actually add the constraints to the model
    if constraints:
        with model:
            for c in constraints:
                c.apply(model)
        
    # save
    cloudpickle.dump({"model": model, "constraints": constraints}, open(file, "wb"))
    
    return model


def sample_posterior(tr, diag, model=None, trace=None, coordinates=None, isimip_model=None, **kwargs):
    cirun = tr.o.cirun
    # file = CIDIR / f'{cirun}/postproc/{diag}_total_coastlines.nc'
    # USE case: default trace driven with isimip temperature
    if isimip_model is not None:
        file = Path(tr.o.dirname) / f'postproc/{diag}-{isimip_model}_total_coastlines.nc'
    else:
        file = Path(tr.o.dirname) / f'postproc/{diag}_total_coastlines.nc'

    if file.exists():    
    # if False and file.exists():    
        print("Load", file)
        return xa.open_dataset(file)
        # return xa.open_dataset(file).load()

    print("Sample posterior", cirun, diag, end=" ")    
    if model is None:
        assert coordinates is not None
        model = get_coastal_model(tr, coordinates, isimip_model=isimip_model, **kwargs)
        if isimip_model is not None:
            assert list(model.coords["experiment"]) == kwargs["experiments"]

    if trace is None:
        trace = tr.get_free_RVs_trace(model)
        
    with model:    
        file.parent.mkdir(exist_ok=True)                
        var_names = make_diagnostic(diags=[diag], sources=['total', 'steric'], fields=['rsl', 'gsl', 'rad', 'global'])
        print(var_names)
        # var_names = ["rate2000_rsl_total", "rate2000_gsl_total", "satellite_obs", "tidegauge_1990_2018_obs"]
        # var_names = ["rate2000_rsl_total", "rate2000_rad_total", "rate2000_gsl_total"]
        # print(var_names)    
        post_grid = pm.sample_posterior_predictive(trace, var_names=var_names).posterior_predictive
        
    print("Write to", file)
    post_grid.to_netcdf(file)
    return post_grid