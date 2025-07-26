"""Models used to fit various sea level components

They are made to work with pytensor.
"""
import numpy as np
from scipy.stats import norm
import pytensor.tensor as pt # type: ignore
import pymc as pm # type: ignore

from sealevelbayes.logs import logger
from sealevelbayes.datasets.ar6.tables import (
    ar6_table_9_5, ar6_table_9_5_quantiles )
from sealevelbayes.datasets.climate import load_temperature, SSP_EXPERIMENTS, get_ssp_experiment_data, IMP_EXPERIMENTS, get_imp_experiment_data
from sealevelbayes.preproc.gmsldatasets import get_merged_mvnormal, get_fred_mvnormal
from sealevelbayes.preproc.stats import fit_dist_to_quantiles
from sealevelbayes.models.glaciermodels import get_uncharted_glacier_global, get_uncharted_glacier_distribution, GlacierModelRegion, load_model_data as load_glacier_data
from sealevelbayes.models.generic import SourceModel, save_private_var

# experimental isimip mode
from sealevelbayes.datasets.isimip import ISIMIP_EXPERIMENTS
from sealevelbayes.models.isimip import define_isimip_data

sources = ["steric", "glacier", "gis", "ais", "landwater", "vlm"]


def get_past_rate_fred(source, transition_year):
    logger.info(f"Sample {source.capitalize()} past rate from a multivariate normal distribution according to Frederikse's until {transition_year}")
    years, mu, cov = get_fred_mvnormal(source)
    assert pm.modelcontext(None).coords['year'][0] == 1900
    nt = transition_year-1900+1
    chol = np.linalg.cholesky(cov[:nt, :nt])
    nexp = len(pm.modelcontext(None).coords['experiment'])
    zeros = pt.zeros((nexp, nt))
    iid = pm.Normal(f"_fred_{source}_iid", size=nt)
    return (chol @ iid + mu[:nt])[None, :] + zeros


def get_past_rate_ar6(source, transition_year):
    logger.info(f"Sample {source.capitalize()} past rate from AR6 tabe 9.5's 1901-1990 and 1993-2018 rates until {transition_year}")
    dist_slr20 = fit_dist_to_quantiles(ar6_table_9_5[source]["Δ (mm)"]["1901-1990"], ar6_table_9_5_quantiles, dist_name="norm")
    duration = 1990-1901+1
    past_rate = pm.Normal("antarctica_past_trend", dist_slr20.mean()/duration, dist_slr20.std()/duration)
    dist_rate2000 = fit_dist_to_quantiles(ar6_table_9_5[source]["mm/yr"]["1993-2018"], ar6_table_9_5_quantiles, dist_name="norm")
    present_rate = pm.Normal("antarctica_present_rate", dist_rate2000.mean(), dist_rate2000.std())
    rate1d = pt.concatenate([past_rate + pt.zeros(1990-1900+1), present_rate + pt.zeros(2018-1990+1)])
    nt = transition_year-1900+1
    nexp = len(pm.modelcontext(None).coords['experiment'])
    zeros = pt.zeros((nexp, nt))
    return rate1d[None, :nt] + zeros

def get_steric_model(bterm=True,
        prior_dist_aT0="Normal", prior_dist_params_aT0=(-1.3, 1),
        prior_dist_a="Exponential", prior_dist_params_a=(1.,),
        prior_dist_b="ConstantData", prior_dist_params_b=(0.,),
                     **kw):
    n = 3 if bterm else 2

    rate2000 = fit_dist_to_quantiles(ar6_table_9_5["steric"]["mm/yr"]["1993-2018"], ar6_table_9_5_quantiles, dist_name="norm")

    if prior_dist_params_aT0 is None or not len(prior_dist_params_aT0):
        prior_dist_params_aT0 = (-rate2000.mean(), rate2000.std())


    return SourceModel("steric",
                        ["a", "aT0", "b"][:n],
                        # [("Uniform", (0, 2)), ("Uniform", (-2, 0)), ("Uniform", (-25, 60))],
                        # [("Normal", (1, 1)), ("Normal", (-1, 1)), ("Normal", (0, 20))], # <- check if that helps the convergence
                        # [("Uniform", (0, 2)), ("Uniform", (-2, 0)), ("Normal", (0, 20))][:n],
                        [(prior_dist_a, prior_dist_params_a), (prior_dist_aT0, prior_dist_params_aT0), (prior_dist_b, prior_dist_params_b)][:n],
                        p0=[0.85, -0.11*0.85, 20][:n], **kw)

def define_steric(X, from_trace=None,
        static_steric=False, **kw):

    if static_steric:
        logger.info("Sample steric as a multivariate normal distribution")
        mus = []
        covs = []
        for x in ["ssp126", "ssp245", "ssp370", "ssp585"]:
            mu_, cov_ = get_merged_mvnormal("steric", x)
            mus.append(mu_)
            covs.append(cov_)
        mu = np.mean(mus, axis=0)
        cov = np.mean(covs, axis=0)
        nt = len(pm.modelcontext(None).coords['year'])
        assert pm.modelcontext(None).coords['year'][0] == 1900, 'static steric assumes start year is 1900'
        chol = np.linalg.cholesky(cov[:nt, :nt])
        zeros = pt.zeros(X['tas'].shape)
        iid = pm.Normal("_steric_iid", dims=("year",))
        rate = (chol @ iid + mu[:nt])[None, :] + zeros
        return pt.cumsum(rate, axis=-1), rate

    source = get_steric_model(**kw)

    return source.define_model(X["tas"], from_trace=from_trace)


def get_regional_glacier_models(regions,
            prior_dist_aT0="Normal", prior_dist_params_aT0=(-1, 2),
            prior_dist_a="Exponential", prior_dist_params_a=(.3,),
            prior_dist_V0="Normal", prior_dist_params_V0=(0, 1),
            dataset_kw={}, # old v1 param
            data=None, # SourceModel param
            add_noise=False,
            dimensionless=True,
            n=0.76,
            noise_kw={},
            # cap_volume=False, # ignore
            **kw):

    # get the scaling dataset
    glacier_data = load_glacier_data(regions, **dataset_kw)
    # getobs = lambda key : data[key].values


    # from sealevelbayes.models.glaciermodels import draw_past_glacier_samples
    # rng = np.random.default_rng(97023)
    # mm21 = -draw_past_glacier_samples(rng=rng, include_zemp=False)


    for region in regions:
        V1 = glacier_data["V2000"].sel(region=region).item()
        V1_sd = glacier_data["V2000_sd"].sel(region=region).item()
        r1 = glacier_data["rate2000"].sel(region=region).item()
        if data is not None:
            data_region = data.sel(region=region)
        else:
            data_region = None

        noise_region_kw = noise_kw.copy()
        if add_noise:
            assert data_region is not None
            noise_region_kw["scale"] = data_region.median("sample").std(dim="year").item()

        # optimization for the normal dist:
        if prior_dist_V0 == "Normal":
            mu, sigma = prior_dist_params_V0
            mu = mu + V1
            sigma = sigma * V1_sd
            locs = None
            scales = None
            prior_dist_V0_specs = ["TruncatedNormal", [mu, sigma], {"lower": 1e-6} ] # [mu, sigma, tau, lower, upper]

        else:
            locs = [V1, 0, 0]
            scales = [V1_sd, 1, 1]
            prior_dist_V0_specs = [prior_dist_V0, prior_dist_params_V0]

        yield GlacierModelRegion(region,
                            ["V0", "a", "aT0"],
                            [ prior_dist_V0_specs, (prior_dist_a, prior_dist_params_a), (prior_dist_aT0, prior_dist_params_aT0) ],
                            locs=locs,
                            scales=scales,
                            r1=r1 if dimensionless else 1, # ~rate2000 scaling for the a*T - aT0 term * r1/V1^n
                            V1=V1 if dimensionless else 1, # ~V2000 scaling for the a*T - aT0 term
                            i0=2000-1900,
                            n=n,
                            data={f"obs": data_region},
                            obs_key="obs",
                            noise_kw=noise_region_kw,
                            add_noise=add_noise,
                            **kw)

def define_glacier(X, regional_glacier=True,
    regional_glacier_experiment=None, filepath=None, static_glacier=False,
    glacier_trend_prior="ar6", from_trace=None,
    uncharted_glaciers=False, uncharted_glaciers_distribution=None,
    uncharted_glaciers_exclude_antarctica=False,
    exclude_icesheets=False, regions=None, **kw):

    model = pm.modelcontext(None)

    if static_glacier:
        assert not exclude_icesheets
        logger.info("Sample glacier as a multivariate normal distribution")
        mus = []
        covs = []
        for x in ["ssp126", "ssp245", "ssp370", "ssp585"]:
            mu_, cov_ = get_merged_mvnormal("glacier", x)
            mus.append(mu_)
            covs.append(cov_)
        mu = np.mean(mus, axis=0)
        cov = np.mean(covs, axis=0)
        nt = len(pm.modelcontext(None).coords['year'])
        assert pm.modelcontext(None).coords['year'][0] == 1900, 'static glacier assumes start year is 1900'
        chol = np.linalg.cholesky(cov[:nt, :nt])
        zeros = pt.zeros(X['tas'].shape)
        iid = pm.Normal("_glacier_iid", dims=("year",))
        rate = (chol @ iid + mu[:nt])[None, :] + zeros

        if uncharted_glaciers:
            logger.info("include uncharted glaciers")
            uncharted_glaciers_rate_global = get_uncharted_glacier_global(pm.modelcontext(None))
            rate = rate + uncharted_glaciers_rate_global[None, :]

        return pt.cumsum(rate, axis=-1), rate

    if not regional_glacier:
        raise NotImplementedError("Only regional glacier model is supported")

    if regions is None:
        regions = np.arange(1, 20)

    if exclude_icesheets:
        regions = [reg for reg in regions if reg not in [5, 19]] # exclude greenland (5) and antarctica (19) regions

    if regional_glacier_experiment:
        raise DeprecationWarning("The regional_glacier_experiment parameter is deprecated")

    sources = get_regional_glacier_models(regions=regions, **kw)

    # aggregate glacier regions for back-compatibility
    _glacier_sea_level_rates = []
    _glacier_volumes = []
    for glacier_model in sources:
        slr, rate = glacier_model.define_model(X["tas"], from_trace=from_trace)
        _glacier_sea_level_rates.append(rate)
        _glacier_volumes.append(-slr)

    model._glacier_volumes = _glacier_volumes # keep track of the list, to avoid the intermediate stack step when constraining
    model._glacier_sea_level_rates = _glacier_sea_level_rates # keep track of the list, to avoid the intermediate stack step when constraining

    if "glacier_region" not in pm.modelcontext(None).coords:
        pm.modelcontext(None).add_coords({"glacier_region": regions})
    pm.Deterministic("glacier_sea_level_rate", pt.stack(_glacier_sea_level_rates, axis=0), dims=("glacier_region", "experiment", "year"))
    pm.Deterministic("glacier_volume", pt.stack(_glacier_volumes, axis=0), dims=("glacier_region", "experiment", "year"))


    if uncharted_glaciers:
        logger.info("include uncharted glaciers")
        uncharted_glaciers_rate_global = get_uncharted_glacier_global(model)
        distribution_uncharted = get_uncharted_glacier_distribution(model,
                                                                    method=uncharted_glaciers_distribution,
                                                                    exclude_antarctica=uncharted_glaciers_exclude_antarctica)
        uncharted_glaciers_rate_regional = uncharted_glaciers_rate_global[None, :] * distribution_uncharted[:, None]

        pm.Deterministic("uncharted_glaciers_rate_regional", uncharted_glaciers_rate_regional, dims=("glacier_region", "year"))
        model._glacier_sea_level_rate_final = model.glacier_sea_level_rate + uncharted_glaciers_rate_regional[:, None, :]

    else:
        model._glacier_sea_level_rate_final = model.glacier_sea_level_rate

    if exclude_icesheets:
        model._glacier_sea_level_rate_final_partial = model._glacier_sea_level_rate_final
        assert model._glacier_sea_level_rate_final_partial.shape[0].eval() == 19 - 2, model._glacier_sea_level_rate_final_partial.shape.eval()
        zeros = pt.zeros_like(model._glacier_sea_level_rate_final_partial[:1])
        model._glacier_sea_level_rate_final = pt.concatenate([model._glacier_sea_level_rate_final_partial[:4], zeros, model._glacier_sea_level_rate_final_partial[4:19-2], zeros], axis=0)

    total_rate = pt.sum(model._glacier_sea_level_rate_final, axis=0)
    return pt.cumsum(total_rate, axis=-1), total_rate


def get_greenland_model(linear_icesheet=False,
        prior_dist_aT0="Normal", prior_dist_params_aT0=(-.43, 1.),
        prior_dist_a="Normal", prior_dist_params_a=(0, 1),
        prior_dist_q="Normal", prior_dist_params_q=(0, 1), **kw):

    n = 2 if linear_icesheet else 3

    rate2000 = fit_dist_to_quantiles(ar6_table_9_5["gis"]["mm/yr"]["1993-2018"], ar6_table_9_5_quantiles, dist_name="norm")

    if prior_dist_params_aT0 is None or not len(prior_dist_params_aT0):
        prior_dist_params_aT0 = (-rate2000.mean(), rate2000.std())

    return SourceModel("gis",
                        ["a", "aT0", "q"][:n],
                        [(prior_dist_a, prior_dist_params_a), (prior_dist_aT0, prior_dist_params_aT0), (prior_dist_q, prior_dist_params_q)][:n], #  with posterior
                        p0=[.3, -0.33*0.3*1.5, 0.1][:n], **kw)

def define_greeland(X, linear_icesheet=False, from_trace=None, static_greenland=False, **kw):

    if static_greenland:
        logger.info("Sample greenland as a multivariate normal distribution")
        mus = []
        covs = []
        for x in ["ssp126", "ssp245", "ssp370", "ssp585"]:
            mu_, cov_ = get_merged_mvnormal("gis", x)
            mus.append(mu_)
            covs.append(cov_)
        mu = np.mean(mus, axis=0)
        cov = np.mean(covs, axis=0)
        nt = len(pm.modelcontext(None).coords['year'])
        assert pm.modelcontext(None).coords['year'][0] == 1900, 'static greenland assumes start year is 1900'
        chol = np.linalg.cholesky(cov[:nt, :nt])
        zeros = pt.zeros(X['tas'].shape)
        iid = pm.Normal("_greenland_iid", dims=("year",))
        rate = (chol @ iid + mu[:nt])[None, :] + zeros
        return pt.cumsum(rate, axis=-1), rate


    source = get_greenland_model(linear_icesheet=linear_icesheet, **kw)

    return source.define_model(X["tas"], from_trace=from_trace)

def get_antarctica_model(linear_icesheet=False, linear_response=False, straight_icesheet=False,
                         prior_dist_q="Normal", prior_dist_params_q=(0, 2),
                         prior_dist_a="Normal", prior_dist_params_a=(0, 2),
                         prior_dist_aT0="Normal", prior_dist_params_aT0=(-.25, 1),
                         cls=SourceModel,
                         **kw):

    if linear_response:
        raise NotImplementedError("linear_response is not implemented yet")

    if straight_icesheet:
        n = 1

    elif linear_icesheet:
        n = 2
        kw.setdefault("log_params", ["a"])
    else:
        n = 3
        kw.setdefault("log_params", ["a"])
        kw.setdefault("log_params", ["q"])

    rate2000 = fit_dist_to_quantiles(ar6_table_9_5["ais"]["mm/yr"]["1993-2018"], ar6_table_9_5_quantiles, dist_name="norm")

    if prior_dist_params_aT0 is None or not len(prior_dist_params_aT0):
        prior_dist_params_aT0 = (-rate2000.mean(), rate2000.std())


    return cls(label="ais",
                        param_names=["aT0", "a", "q"][:n],
                        params_dist=[(prior_dist_aT0, prior_dist_params_aT0), (prior_dist_a, prior_dist_params_a), (prior_dist_q, prior_dist_params_q)][:n], #  with posterior
                        # 0.16, 0.25, 0.33
                        p0=[-0.33*0.3*1.5, .3, 0.1][:n], **kw)


def define_antarctica(X, linear_icesheet=False, from_trace=None, static_antarctica=False,
                      ar6_kwargs={},
                      **kw):

    if static_antarctica:
        logger.info("Sample Antarctica as a multivariate normal distribution")
        mus = []
        covs = []
        for x in ["ssp126", "ssp245", "ssp370", "ssp585"]:
            try:
                mu_, cov_ = get_merged_mvnormal("ais", x, ar6_kwargs=ar6_kwargs)
            except FileNotFoundError as error:
                logger.warning(f"{x} :: File not found :: {error}")
                continue
            mus.append(mu_)
            covs.append(cov_)
        mu = np.mean(mus, axis=0)
        cov = np.mean(covs, axis=0)
        nt = len(pm.modelcontext(None).coords['year'])
        assert pm.modelcontext(None).coords['year'][0] == 1900, 'static antarctica assumes start year is 1900'
        chol = np.linalg.cholesky(cov[:nt, :nt])
        zeros = pt.zeros(X['tas'].shape)
        iid = pm.Normal("_antarctica_iid", dims=("year",))
        rate = (chol @ iid + mu[:nt])[None, :] + zeros
        return pt.cumsum(rate, axis=-1), rate

    else:
        source = get_antarctica_model(linear_icesheet=linear_icesheet, **kw)

    return source.define_model(X["tas"], from_trace=from_trace)


def define_landwater(X, from_trace=None):
    logger.info("Sample landwater as a multivariate normal distribution")
    mu, cov = get_merged_mvnormal("landwater", "ssp585") # 585 is middle of the road whereas 370 is larger
    nt = len(pm.modelcontext(None).coords['year'])
    assert pm.modelcontext(None).coords['year'][0] == 1900, 'landwater assumes start year is 1900'
    chol = np.linalg.cholesky(cov[:nt, :nt])
    zeros = pt.zeros(X['tas'].shape)
    iid = pm.Normal("_landwater_iid", dims=("year",))
    rate = (chol @ iid + mu[:nt])[None, :] + zeros
    return pt.cumsum(rate, axis=-1), rate


def calculate_global_slr(X, save_timeseries=True, data=None,
    sources=None, from_trace=None, landwater_kwargs={}, steric_kwargs={},
    glacier_kwargs={}, antarctica_kwargs={}, greenland_kwargs={}):
    """
    """
    if sources is None:
        sources = ["steric", "glacier", "gis", "ais", "landwater"]

    Y = {}
    Yslr = {}

    if "steric" in sources:
        Yslr["steric"], Y["steric"] = define_steric(X, from_trace=from_trace, **steric_kwargs)

    if "glacier" in sources:
        Yslr["glacier"], Y["glacier"] = define_glacier(X, from_trace=from_trace, **glacier_kwargs)

    if "gis" in sources:
        Yslr["gis"], Y["gis"] = define_greeland(X, from_trace=from_trace, **greenland_kwargs)

    if "ais" in sources:
        Yslr["ais"], Y["ais"] = define_antarctica(X, from_trace=from_trace, **antarctica_kwargs)

    if "landwater" in sources:
        Yslr["landwater"], Y["landwater"] = define_landwater(X, from_trace=from_trace, **landwater_kwargs)

    # if len(sources) > 1:
    Y["total"] = sum(Y.values())
    Yslr["total"] = sum(Yslr.values())

    # Save to model
    dims = "experiment", "year"
    if save_timeseries:
        for v in Y:
            pm.Deterministic(v, Y[v], dims=dims)

    # Save data to model (private variables --> won't show up in trace)
    for name in Yslr:
        save_private_var(f"{name}_slr", Yslr[name])
    for name in Y:
        save_private_var(f"{name}_rate", Y[name])

    return Y


def define_tas(experiments=None, experiments_data={}, model=None):
    """return a dict of year, tas, tas2 and tasdiff in vectorized form

    experiments: list of experiments to run. By default all experiment in models.DEFAULT_EXPERIMENTS
    are available. That includes:
    - ssp126_mu
    - ssp185_mu
    - ssp126
    - ssp185
    - IMPs
    experiments_data: dict of experiments, optional. See climate.get_ssp_experiment_data for specifications.
    model: pyMC model, optional (will be retrieved from context)
    """
    # Load merged historical temperature data to provide a common history
    tas_df = load_temperature()
    tas_pi = tas_df['ssp585'].loc[1995:2014].mean() - tas_df['ssp585'].loc[1855:1900].mean()
    tas_df = tas_df.loc[1900:2099] # only defined up to 2099
    tas_df -= tas_df.loc[1995:2014].mean()

    tas_hist = tas_df['ssp585'].loc[:2014].values

    if experiments is None:
        experiments = DEFAULT_EXPERIMENTS

    # Load additional experiment data
    # load_ssp = [x for x in experiments if x not in experiments_data and x in SSP_EXPERIMENTS+SSP_EXPERIMENTS_mu]
    load_ssp = list(set(SSP_EXPERIMENTS).intersection(x[:-3] if x.endswith("_mu") else x for x in experiments))
    experiments_data.update(get_ssp_experiment_data(load_ssp))
    load_imp = list(set(IMP_EXPERIMENTS).intersection(x[:-3] if x.endswith("_mu") else x for x in experiments))
    experiments_data.update(get_imp_experiment_data(load_imp))

    # Define the _mu version of the experiments
    for x in experiments:
        if x.endswith("_mu") and x[:-3] in experiments_data:
            experiments_data.setdefault(x, {
                "median": experiments_data[x[:-3]]['median'],
                "years": experiments_data[x[:-3]]['years'],
            })

    # Also define the step sampler for the experiments
    # The reason why we use the GlobalTemperatureSamplingStep was originally related to using SSP constraints,
    # to keep the full range of temperature. Now that we apply the 21st constraint to the median scenario,
    # it should not matter any more, since there is no correlation between tas_factor and other temperature-related
    # random variables and the constaints. However, using a custom sampler does speed up calculations (42s )
    model = pm.modelcontext(model=model)
    model._stepper = []

    tas_dict = {}

    # Here we allow temperature to vary according to IPCC AR6 projections
    # We introduce a new, normally distributed parameter tas_factor that controls how tas varies within that range
    s = pm.Normal("tas_factor", 0, 1)
    model._stepper.append((s, lambda rng: rng.normal()))

    # Define in the model
    for x in experiments:
        if x not in experiments_data:
            continue

        data = experiments_data[x]

        if "years" not in data:
            # assert data["years"][0] == 2015, "experiment data must be from 2015 to 2099"
            # assert data["years"][-1] == 2099, "experiment data must be from 2015 to 2099"
            assert np.size(data["median"]) == 2099-2015+1, "experiment data must be from 2015 to 2099 unless year is provided"
            year0 = 2015
            year1 = 2099
        else:
            year0 = data["years"][0]
            year1 = data["years"][-1]

        merge_year = 2015
        i = merge_year - 1900
        i0 = merge_year - year0
        imax = 2099 + 1 - year0

        def merge(x):
            " merge in 2015 with an offset "
            if merge_year > year0:
                prev = x[i0-1]
            else:
                prev = x[i0] - (x[i0+1]-x[i0])
            offset = tas_hist[i-1] - prev
            return np.concatenate([tas_hist[:i], x[i0:imax]+offset])

        tas_mu = pm.ConstantData(f"tas_{x}", merge(data["median"]), dims=("year"))
        assert ("p95" in data and "p05" in data) or ("p95" not in data and "p05" not in data)

        # Uncertainty info present ?
        if "p95" in data:
            tas_hi = pm.ConstantData(f"tas_{x}_95th", merge(data["p95"]), dims=("year"))
            tas_lo = pm.ConstantData(f"tas_{x}_5th", merge(data["p05"]), dims=("year"))
            # s == 0 => tas = tas_mu
            # s == 1.64 => tas = tas_hi
            # s == -1.64 => tas = tas_lo
            sd = pt.switch(pt.gt(s, 0), tas_hi-tas_mu, tas_mu-tas_lo)/norm.ppf(0.95)  # 1.64 is the factor to pass from 90% to 1-sigma range
            tas_dict[x] = tas_mu + s*sd

        # Otherwise only median temperature is requested
        else:
            tas_dict[x] = tas_mu

    # isimip_mode
    for x in experiments:
        if x.startswith("isimip_"):
            assert "isimip_experiment" in model.coords, "activate isimip_mode to load/define isimip experment data"
            ix = model.coords["isimip_experiment"].index(x[len("isimip_"):])
            tas_dict[x] = model.isimip_tas[ix]

    # Now stack tas
    return pm.Deterministic("tas", pt.stack([tas_dict[x] for x in experiments]), dims=("experiment", "year"))


def define_predictors(experiments=None, experiments_data={}, data=None, resample=False, isimip_mode=False, **kwargs):
    """here tas can be provided as a 2-D pytensor array with dimensions experiment x year
    """
    # Here provide as a dummy random variable for resampling from trace
    if resample:
        logger.info("dummy Normal tas to be resampled")
        tas = pm.Normal("tas", dims=("experiment", "year"))

    # We can also pass it as (deterministic) ND-array, provided the dimensions match
    elif data is not None and "tas" in data:
        logger.info("deterministic tas taken from data")
        tas = pm.ConstantData("tas", np.asarray(data["tas"]), dims=("experiment", "year"))

    else:
        logger.info("tas defined internally")
        tas = define_tas(experiments=experiments, experiments_data=experiments_data, **kwargs)

    tas2 = tas**2
    tasdiff = pt.concatenate([tas[:, 1:2]-tas[:, 0:1], tas[:, 1:]-tas[:, :-1]], axis=1)

    return {
        "tas": tas,
        "tas2": tas2,
        "tasdiff": tasdiff,
    }


SSP_EXPERIMENTS_mu = [x+"_mu" for x in SSP_EXPERIMENTS]
IMP_EXPERIMENTS_mu = [x+"_mu" for x in IMP_EXPERIMENTS]
ISIMIP_EXPERIMENTS_PREFIXED = ["isimip_" + x for x in ISIMIP_EXPERIMENTS]

AVAILABLE_EXPERIMENTS = [
            # "ssp126_mu", "ssp585_mu",  # for constraining
            SSP_EXPERIMENTS_mu +
            IMP_EXPERIMENTS_mu +
            SSP_EXPERIMENTS +
            IMP_EXPERIMENTS +
            ISIMIP_EXPERIMENTS_PREFIXED
            # "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    ]


DEFAULT_EXPERIMENTS = [
    "ssp126_mu", "ssp585_mu",  # for constraining
    # ["ssp126", "ssp585"] +
    # ['SP', 'GS', 'CurPol'] +
    # ['SP_mu', 'GS_mu', 'CurPol_mu'] +
    # ["ssp126_mu", "ssp585_mu"]
]

# The order of experiments is such that the past obs scenario will be DEFAULT_ISIMIP_EXPERIMENTS[0] = DEFAULT_EXPERIMENTS[0] = "ssp126_mu"
# This implies the past constraint will happen with the same GMT pathway regardless if whether --isimip-mode is active or not
# If that behaviour is not
DEFAULT_ISIMIP_EXPERIMENTS = DEFAULT_EXPERIMENTS + ["isimip_ssp126", "isimip_ssp585"]


def slr_model_global(experiments=None, years=np.arange(1900, 2099+1),
    data=None,
    model=None,
    resample=False,
    sources=None,
    save_timeseries=True,
    isimip_mode=False, isimip_model=None, isimip_steric_sigma=10, isimip_tas_noise=False, isimip_tas_no_obs=False,
    constraints=[], **kwargs):
    """
    experiments: experiments to include
    years: normally 1900 to 2099
    data : trace.constant_data that may be used instead of re-loading
    resample: if True, define some variables ("tas" in particular) as dummy pm.Normal to use from trace.posterior instead (via sample_posterior_predictive)
    model: pymc.Model instance, to be provided in case tas is already defined outside this scope.
    save_timeseries: ...
    constraints : SLRConstraint instances to apply
    """
    if model is None:
        if experiments is None:
            experiments = DEFAULT_ISIMIP_EXPERIMENTS if isimip_mode else DEFAULT_EXPERIMENTS
        model = pm.Model(coords={"year": years, "experiment": experiments })

    experiments = list(model.coords['experiment'])

    with model:
        if isimip_model:
            define_isimip_data(isimip_model, isimip_tas_noise=isimip_tas_noise, isimip_tas_no_obs=isimip_tas_no_obs)

        X = define_predictors(experiments=experiments, data=data, resample=resample, isimip_mode=isimip_mode)
        Y = calculate_global_slr(X, save_timeseries=save_timeseries, data=data, sources=sources, **kwargs)

        # for re-use in slr_model_tidegauges_coupled
        model._X = X
        model._Y = Y

        for c in constraints:
            logger.info(f"apply constraint: {c}")
            logger.info(c.label)
            c.apply_model(model)

    return model