"""Models used to fit various sea level components

They are made to work with pytensor.
"""
import numpy as np
import pymc as pm # type: ignore
import xarray as xa
from sealevelbayes.logs import logger
from sealevelbayes.models.compat import getmodeldata
from sealevelbayes.models.globalconstraints import observe_slr2100, SLRConstraint
from sealevelbayes.datasets.isimip import ISIMIP_EXPERIMENTS, load_tuning_data_zostoga, load_forcing_data_tas


def define_isimip_data(isimip_model, isimip_tas_noise=False, isimip_tas_no_obs=False):
    pymc_model = pm.modelcontext(None)


    if "isimip_experiment" not in pymc_model.coords:
        pymc_model.add_coord("isimip_experiment", ISIMIP_EXPERIMENTS)

    # zostoga
    if hasattr(pymc_model, 'isimip_zostoga'):
        logger.info('isimip_zostoga already defined')
    elif isimip_model == "GFDL-ESM4":
        logger.info('GFDL-ESM4 does not provide zostoga output -- do not load anything')
    else:
        zostoga = xa.concat([load_tuning_data_zostoga(isimip_model, experiment) for experiment in ISIMIP_EXPERIMENTS], dim='isimip_experiment')
        zostoga = zostoga.sel(year=slice(1900, 2099)) # legacy compatibility...
        meters_to_mm = 1000
        pm.ConstantData('isimip_zostoga', zostoga.values*meters_to_mm, dims=zostoga.dims)

    # tas
    if hasattr(pymc_model, 'tas'):
        logger.info('tas already defined')
    else:
        tas = xa.concat([load_forcing_data_tas(isimip_model, experiment,
            smooth=not isimip_tas_noise, mergeobs=not isimip_tas_no_obs,
            ) for experiment in ISIMIP_EXPERIMENTS], dim='isimip_experiment')
        tas -= tas.sel(year=slice(1995, 2014)).mean('year')
        tas = tas.sel(year=slice(1900, 2099)) # legacy compatibility... (forcing)
        pm.ConstantData('isimip_tas', tas.values, dims=tas.dims)


def apply_21c_isimip_steric_constraints(Y, experiments=['isimip_ssp126', 'isimip_ssp585'], sigma=10):
    """
    experiments: name of experiments on which to apply the constraints (must be the names used in the model)
    """

    def _get_zostoga(pymc_model):
        "get the data from the model attribute, and assign it prefixed experiment names"
        zostoga = np.array(getmodeldata("isimip_zostoga", pymc_model))
        years = list(pymc_model.coords['year'])
        all_experiments = list(pymc_model.coords['isimip_experiment'])
        all_experiments_prefixed = ["isimip_"+x for x in all_experiments]
        return xa.DataArray(zostoga, coords={'experiment':all_experiments_prefixed, 'year': years}, dims=['experiment', 'year'])

    pymc_model = pm.modelcontext(None)
    zostoga = _get_zostoga(pymc_model)

    for x in experiments:
        y = zostoga.sel(experiment=x).loc[1900:2099]
        y = (y - y.loc[1995:2014].mean()).values
        trend = y[-1] - y[-2]
        obs = y[-1] + trend  # 2099 -> 2100
        proj = observe_slr2100(Y[x]['steric'])
        name = f"steric_{x}_slr21"
        pm.ConstantData("obs_"+name+"_mu", obs)
        pm.ConstantData("obs_"+name+"_sd", sigma)
        pm.Deterministic(name, proj)
        pm.Normal('obs_'+name, proj, sigma, observed=obs)


class ISIMIPStericConstraint(SLRConstraint):
    """use the ISIMIP model as target for future steric expansion (or provided alternative if not available)
    """
    def __init__(self, experiment):
        self.source = "steric"
        self.experiment = experiment
        self.diag = "proj2100"

    # def __call__(self, slr, rate, prefix="", years=None, experiments=None):

    def get_observable(self, slr, rate):
        return rate

    def observe(self, rate):
        model = pm.modelcontext(None)
        assert hasattr(model, "isimip_zostoga")
        logger.info("Apply ISIMIP constraints for steric contribution")
        Ydict = { self.experiment: { self.source: self._select_experiment(rate, experiment=self.experiment) }}
        return apply_21c_isimip_steric_constraints(Ydict, experiments=[self.experiment])