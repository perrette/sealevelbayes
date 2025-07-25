#!/usr/bin/env python3
import subprocess as sp
from pathlib import Path
import os
import sys
import shutil
import numpy as np # type: ignore
import json
import xarray as xa # type: ignore
import datetime

import pymc as pm # type: ignore
import arviz # type: ignore
import cloudpickle  # type: ignore # to save the model

import sealevelbayes
from sealevelbayes.config import get_runpath, setup_logger, get_webpath, get_version, print_experiment_info
from sealevelbayes.logs import logger, logging
from sealevelbayes.datasets.satellite import get_satellite_timeseries, get_satellite_error
from sealevelbayes.datasets.oceancmip6 import load_all_cmip6
from sealevelbayes.datasets.tidegaugeobs import load_tidegauge_records
from sealevelbayes.datasets.ar6.misc import load_temperature
from sealevelbayes.datasets.glaciers import load_zemp, gt_to_mm_sle, MM21_FORCING, open_mm21, get_rate2000_ar6, draw_past_glacier_samples
from sealevelbayes.datasets.frederikse2020 import sample_greenland_dataset
from sealevelbayes.datasets.basins import ThompsonBasins

from sealevelbayes.models.domain import get_station_ids
from sealevelbayes.models.domain import get_stations_from_psmsl_ids, get_stations_from_coords, get_stations_from_grid, get_stations_from_ids
from sealevelbayes.models.domain import get_non_psmsl_gps_stations
from sealevelbayes.models.globalslr import slr_model_global, DEFAULT_EXPERIMENTS, DEFAULT_ISIMIP_EXPERIMENTS
from sealevelbayes.models.globalconstraints import get_21c_constraints_ar6_table_9_8, get_20c_constraints_ar6_table_9_5, get_ar6_constraint_table_95_by_alias, get_21c_constraints_low_confidence
from sealevelbayes.models.isimip import ISIMIPStericConstraint
from sealevelbayes.models.localslr import slr_model_tidegauges
from sealevelbayes.models.glaciermodels import (get_uncharted_glacier_timeseries,
                                                get_uncharted_glacier_distribution, load_model_data as load_glacier_data, Glacier2100Constraint)
from sealevelbayes.models.globalconstraints import TimeSeriesConstraint, TimeSeriesConstraintGP, RateConstraint
from sealevelbayes.models.likelihood import GPSConstraint
from sealevelbayes.models.likelihood import SatelliteTrend, TideGaugeTrend
from sealevelbayes.models.likelihood import MixedConstraint
from sealevelbayes.models.generic import DeferredDist
from sealevelbayes.preproc.oceancovariance import CMIP6Sampler, MeanSampler, SatelliteAR1Sampler, SatelliteEOFAR1Sampler
from sealevelbayes.preproc.tidegaugeerror import estimate_tidegauge_error, get_tidegauge_to_satellite_residual_ratio

from sealevelbayes.runparams import (_source_short, _source_long, _get_param, get_watermark, get_parser, parse_args, get_runid)

BASINS = ThompsonBasins.load().split_atlantic()

def get_glacier_constraints(o):
    # get glacier constraints for 2100

    constraints = []

    rng = np.random.default_rng(o.random_seed + 232891 if o.random_seed else None)

    for name in o.glacier_constraints:

        if name == "proj2100":
            glacier_data = load_glacier_data(o.glacier_regions, constraints_kwargs=dict(v2000_source=o.glacier_volume_source))

            for region in o.glacier_regions:
                for experiment_ in o.glacier_future_constraint_experiments:
                    experiment = experiment_ + "_mu" if not experiment_.endswith("_mu") else experiment_
                    assert not o.glacier_normalize_future
                    c = Glacier2100Constraint.from_glacier_data(region, experiment, glacier_data=glacier_data,
                                                                no_clip=o.debug_glacier_future_constraint_no_clip,
                                                                on_trend=o.glacier_future_constraint_on_trend,
                                                                scaled=o.debug_glacier_future_constraint_scaled,
                                                                dummy="glacier" in o.skip_future_constraints or "glacier" in o.skip_constraints,
                                                                )
                    c.source = "glacier" # it does not really matter where it is called
                    c.diag = "proj2100"

                    constraints.append(c)

        elif name == "ar6-present":
            for region in o.glacier_regions:
                if region in [13, 14, 15]:
                    continue # merged in table 9.SM.2
                obs, obs_sd = get_rate2000_ar6(region)
                c = RateConstraint("Normal", [obs, obs_sd], (2000, 2019), source="glacier", region=region, diag=name, experiment="ssp585_mu")
                constraints.append(c)

        elif name in ("mm21", "mm21+zemp19", "mm21-2000"):
            include_zemp = name == "mm21+zemp19"
            samples = -draw_past_glacier_samples(rng=rng, regress_on_mm21=o.glacier_regress_on_mm21,
                                                include_zemp=include_zemp,
                                                mm21_forcing=o.glacier_mm21_forcing,
                                                include_mm21_error=o.glacier_include_mm21_error,
                                                mm21_drop_20CRV3_17=o.glacier_mm21_drop_20CRV3_17,
                                                ).rename({"Sample": "sample", "Time": "year", "Region": "region"})

            for region in o.glacier_regions:
                if region == 19 and not include_zemp:
                    continue
                data = samples.sel(region=region)

                if name == "mm21-2000":
                    data = data.sel(year=slice(None, 2000))

                mean = data.mean("sample").values
                sd = data.std("sample").values
                axis = data.dims.index("sample")
                cov = np.cov(data.values, rowvar=axis > 0)
                cov += np.eye(cov.shape[0]) * 1e-6  # regularize

                if getattr(o, "glacier_noise_obs_autocorrel"):
                    sd = None
                else:
                    cov = None

                c = TimeSeriesConstraint(f"glacier_{region}_{name}", obs=mean, obs_cov=cov, obs_sd=sd,
                                         obs_years=data.year.values, source="glacier", region=region, diag="mm21", experiment="ssp585_mu",
                                         input_vars=[f"glacier_{region}_slr_before_clipping", f"glacier_{region}_rate_before_clipping"])

                constraints.append(c)

        elif name == "mm21-indiv":
            with open_mm21() as ds:
                ds = ds[["Mass change", "Mass change uncertainty"]].load()

            # scale = DeferredDist("glacier_mm21_sigma_scale", "Exponential", [1])
            for region in o.glacier_regions:

                if region == 19:
                    continue

                if o.glacier_mm21_indiv_weighting == "scale_region":
                    scale = DeferredDist(f"glacier_mm21_sigma_scale_{region}", "Exponential", [1]) # one scale per region

                for forcing in o.glacier_mm21_forcing:
                    df = ds.sel(Forcing=forcing, Region=region).to_pandas().dropna(axis=0)
                    obs = df["Mass change"].values * gt_to_mm_sle * (-1)
                    obs_sd = df["Mass change uncertainty"].values * gt_to_mm_sle
                    obs_years = df.index.values

                    if o.glacier_mm21_indiv_weighting in ("none", "scale_region"):
                        c = TimeSeriesConstraint(f"glacier_{region}_{name}_{forcing}", obs=obs, obs_sd=obs_sd,
                                                scale=1 if o.glacier_mm21_indiv_weighting == "none" else scale,
                                                obs_years=obs_years,
                                                source="glacier", region=region, diag=name, experiment="ssp585_mu",
                                                input_vars=[f"glacier_{region}_slr_before_clipping", f"glacier_{region}_rate_before_clipping"])

                    elif o.glacier_mm21_indiv_weighting == "add_region":
                        # obs_sd_func = ScaledObsSd(obs_sd, name=f"glacier_mm21_sigma_scale_{region}_{forcing}")
                        c = TimeSeriesConstraintGP(f"glacier_{region}_{name}_{forcing}",
                                                 obs=obs, obs_sd=obs_sd, obs_years=obs_years,
                                                 cov_func="WhiteNoise", cov_params=[1],
                                                 sigma_dist="Exponential", sigma_dist_params=[obs.std().item()],
                                                 source="glacier", region=region, diag=name, experiment="ssp585_mu",
                                                 input_vars=[f"glacier_{region}_slr_before_clipping", f"glacier_{region}_rate_before_clipping"])

                    else:
                        raise ValueError(f"Unknown glacier_mm21_indiv_weighting {o.glacier_mm21_indiv_weighting}")

                    constraints.append(c)


        elif name == "zemp19":
            for region in o.glacier_regions:
                zemp = load_zemp(region)
                mean = -zemp["INT_Gt"].values * gt_to_mm_sle
                sd = zemp["sig_Total_Gt"].values * gt_to_mm_sle
                years = zemp["Year"].values
                c = TimeSeriesConstraint(f"glacier_{region}_{name}", obs=mean, obs_sd=sd,
                                         obs_years=years, source="glacier", region=region, diag="mm21", experiment="ssp585_mu",
                                         input_vars=[f"glacier_{region}_slr_before_clipping", f"glacier_{region}_rate_before_clipping"])

                constraints.append(c)


        else:
            raise ValueError(f"Unknown glacier constraint {name}")

    return constraints


def get_global_constraints(o):

    constraints = []

    # for low-conf constraint only use ssp585
    if o.antarctica_future_constraint == "ar6-low-confidence":
        o.antarctica_future_constraint_experiments = ["ssp585"]
    if o.greenland_future_constraint == "ar6-low-confidence":
        o.greenland_future_constraint_experiments = ["ssp585"]

    if o.skip_all_constraints:
        return constraints

    if o.static_antarctica:
        global_c_ais = []
    else:
        global_c_ais = ["slr20"] + o.antarctica_future_constraint_experiments

    if o.static_greenland:
        global_c_gis = []
    else:
        global_c_gis = ["slr20"] + o.greenland_future_constraint_experiments

    if o.static_steric:
        global_c_steric = []
    else:
        global_c_steric = ["slr20", "rate2000"] + o.steric_future_constraint_experiments

# for source in ["steric", "glacier", "ais", "gis"]:
    global_c = {
        "steric": global_c_steric,
        "glacier": [], # dealt with separately in v1, and further below in v2
        "ais": global_c_ais,
        "gis": global_c_gis,
        "landwater": [],
    }

    all_future_experiments = ["ssp126", "ssp585"]

    for source in ["steric", "glacier", "ais", "gis", "landwater"]:
        if source not in o.sources:
            continue
        if source in o.skip_constraints:
            logger.info(f"skip {source} constraint")
            global_c[source] = []
        if source in o.skip_past_constraints:
            logger.info(f"skip past {source} constraints")
            global_c[source] = [c for c in global_c[source] if c not in ["slr20", "rate2000"]]
        if source in o.skip_future_constraints:
            logger.info(f"skip future {source} constraints")
            global_c[source] = [c for c in global_c[source] if c not in all_future_experiments]

        # add custom constraints
        long_name = _source_long(source)
        custom_constraints = getattr(o, f"{long_name}_constraints", None)
        if custom_constraints is not None and source != "glacier": # (glacier constraints are handled separately)
            global_c[source].extend([c for c in custom_constraints if c not in global_c[source]])

        # SLR20
        constraints.extend(get_20c_constraints_ar6_table_9_5(sources=[source], slr20=True, rate2000=False,
                                                                dummy="slr20" not in global_c[source]))

        # rate2000
        constraints.extend(get_20c_constraints_ar6_table_9_5(sources=[source], slr20=False, rate2000=True,
                                                             dummy="rate2000" not in global_c[source]))

        # other constraints
        constraints.extend(get_ar6_constraint_table_95_by_alias(name, source) for name in global_c[source] if name not in ["slr20", "rate2000"] and not name.startswith("ssp"))


        # future constraints
        # ... global glacier is applied on the total (which may include noise -- it's a dummy constraint anyway)
        on_trend = source != "glacier" and getattr(o, f"{source}_future_constraint_on_trend", False)

        # In this mode, use the ISIMIP model as target for future steric expansion (and AR6 table 9.8 for other sources)
        if source == "steric" and o.isimip_mode and o.isimip_model != "GFDL-ESM4":
            constraints.extend([ISIMIPStericConstraint(experiment=x) for x in o.steric_future_constraint_experiments])

        else:
            for x in all_future_experiments:
                if ((source == "ais" and o.antarctica_future_constraint == "ar6-low-confidence") or
                    (source == "gis" and o.greenland_future_constraint == "ar6-low-confidence")):
                    constraints.extend(get_21c_constraints_low_confidence(source, dummy=x not in global_c[source], experiments=[x], on_trend=on_trend)),
                else:
                    constraints.extend(get_21c_constraints_ar6_table_9_8(sources=[source], experiments=[x],
                                                                        dummy=x not in global_c[source],
                                                                        include_peripheral_glaciers=o.glacier_exclude_icesheets, on_trend=on_trend))

    constraints.extend(get_glacier_constraints(o))

    return constraints


def load_timeseries_data(o, rng=None):
    # load frederikse data to use as constraint or to generate noise
    from sealevelbayes.datasets.frederikse2020 import root as fred_folder

    if rng is None:
        rng = np.random.default_rng(o.random_seed + 23291 if o.random_seed else None)

    glaciers_resampled_dataset = -draw_past_glacier_samples(rng=rng,
                                                           mm21_forcing=o.glacier_mm21_forcing,
                                                           regress_on_mm21=o.glacier_regress_on_mm21,
                                                           include_zemp=o.glacier_include_zemp,
                                                           include_mm21_error=o.glacier_include_mm21_error,
                                                           mm21_drop_20CRV3_17=o.glacier_mm21_drop_20CRV3_17,
                                                           )

    if o.add_uncharted_glaciers:
        uncharted_rate_low = get_uncharted_glacier_timeseries(glaciers_resampled_dataset.Time.values, 16.7, 2.1)
        uncharted_rate_high = get_uncharted_glacier_timeseries(glaciers_resampled_dataset.Time.values, 48, 2.4)
        share = rng.uniform(0, 1, size=glaciers_resampled_dataset.Sample.size)
        uncharted_rate = share[None, :] * uncharted_rate_high[:, None] + (1-share[None, :]) * uncharted_rate_low[:, None]

        uncharted_rate_mean = (uncharted_rate_low + uncharted_rate_high) / 2
        uncharted_dist = get_uncharted_glacier_distribution(None, np.arange(1, 19+1),
                                                            method=o.glacier_uncharted_distribution,
                                                            exclude_antarctica=o.glacier_uncharted_exclude_antarctica)
    else:
        uncharted_rate_mean = 0
        uncharted_dist = np.ones(19) / 19

    with xa.open_dataset(fred_folder / "GMSL_ensembles.nc") as ds:
        mapping = {"GrIS":"gis", "AIS": "ais", "Steric": "steric", "Glaciers": "glacier", "TWS": "landwater"}
        ensemble = ds[list(mapping.keys())].rename(mapping).rename({"time":"year", "likelihood":"sample"})
        data = ensemble - ensemble.mean("year") # remove the mean prior to computing std
        # data = ensemble.rename({"likelihood":"sample"})
        # data = ensemble.mean(dim="likelihood")
        # # obs error specified in mm / yr because it applies on the AR(1) innovation, which happens on a yearly basis
        # obs_sd = ensemble.diff("year").std(dim="likelihood").rename({k: k+"_sd" for k in ensemble.keys()})
        # # obs_sd = obs_sd.fillna(-999) # the first value is not used anyway
        # data.update(obs_sd)

    data_rate = data.diff("year").reindex(year=data.year.values, method="bfill")

    regions = glaciers_resampled_dataset.Region.values
    if o.glacier_exclude_icesheets:
        regions = [r for r in regions if r not in [5, 19]]

    data_rate["glacier"] = ("sample", "year"), (
        glaciers_resampled_dataset.sel(Region=regions).sum("Region").transpose("Sample", "Time").values
        + uncharted_rate.T * uncharted_dist.sum()  # if glacier-exclude-antarctica the sum of dist is < 1
        )

    data["glacier"] = data_rate["glacier"].cumsum("year")
    data["glacier"] = data["glacier"] - data["glacier"].mean("year")

    if o.greenland_exclude_fred_peripheral_glaciers:
        logger.info("Resample Frederikse GIS source data to exclude peripheral glaciers")

        # Resample from the original data sources used by Frederikse
        if o.greenland_exclude_fred_peripheral_glaciers_method == "resampled":
            data_rate["gis"] = ("sample", "year"), sample_greenland_dataset(samples=data_rate.sample.size,
                                                                            datasets=o.greenland_exclude_fred_peripheral_glaciers_datasets,
                                                                            years=data_rate.year.values).values.T


        # just take the mean glacier contribution off Fred and to the GIS
        elif o.greenland_exclude_fred_peripheral_glaciers_method == "offset":
            fraction_of_greenland_to_total_uncharted = uncharted_dist[5-1]
            offset = glaciers_resampled_dataset.sel(Region=5).mean("Sample").values + uncharted_rate_mean * fraction_of_greenland_to_total_uncharted
            data_rate["gis"] = data_rate["gis"] - offset[None, :]

        else:
            raise ValueError(f"Unknown method {o.greenland_exclude_fred_peripheral_glaciers_method}")

        data["gis"] = data_rate["gis"].cumsum("year")
        data["gis"] = data["gis"] - data["gis"].mean("year")

    if o.antarctica_exclude_fred_peripheral_glaciers:
        logger.info("Resample Frederikse AIS source data to exclude peripheral glaciers")

        # Resample from the original data sources used by Frederikse
        if o.antarctica_exclude_fred_peripheral_glaciers_method == "resampled":
            raise NotImplementedError("Resampling AIS data not implemented yet")
            # data_rate["ais"] = ("sample", "year"), sample_antarctica_dataset(samples=data_rate.sample.size,
            #                                                                 datasets=o.antarctica_exclude_fred_peripheral_glaciers_datasets,
            #                                                                 years=data_rate.year.values).values.T

        # just take the mean glacier contribution off Fred and to the ais
        elif o.antarctica_exclude_fred_peripheral_glaciers_method == "offset":
            aa_glaciers = glaciers_resampled_dataset.sel(Region=19).mean("Sample").values
            aa_glaciers[data.year.values < 2003] = 0  # Fred dataset does not include the peripheral glaciers before 2003, so no risk of double counting there
            # In Fred there does not seem to be any uncharted glaciers in Antarctica so no need to correct for that
            data_rate["ais"] = data_rate["ais"] - aa_glaciers[None, :]

        else:
            raise ValueError(f"Unknown method {o.antarctica_exclude_fred_peripheral_glaciers_method}")

        data["ais"] = data_rate["ais"].cumsum("year")
        data["ais"] = data["ais"] - data["ais"].mean("year")

    data["total"] = data["gis"] + data["ais"] + data["steric"] + data["glacier"] + data["landwater"]
    data_rate["total"] = data_rate["gis"] + data_rate["ais"] + data_rate["steric"] + data_rate["glacier"] + data_rate["landwater"]

    # add some jitter on the first time-step so that the cov matrix is positive definite
    for k in data.keys():
        data_rate[k].loc[:, 1900] += rng.normal(0, 0.01*data_rate[k].loc[:, 1900].std("sample").values,
                                                size=data.coords["sample"].size)

    # see on a per-contribution basis whether the cumulative SLR or the rate is used
    for k in data_rate:
        name = {"gis": "greenland", "ais": "antarctica"}.get(k, k)
        if o.noise_on_rate or getattr(o, f"{name}_noise_on_rate", False):
            data[k] = data_rate[k]


    # also load GSAT
    tas_df = load_temperature()['ssp585']
    tas_df -= tas_df.loc[1995:2014].mean()
    tas_df = tas_df.reindex(data.year.values)
    tas_df.index.name = 'year'
    data['gsat'] = tas_df

    return data, glaciers_resampled_dataset

def get_global_slr_kwargs(o, standalone=False):

    global_slr_kwargs = dict( sources = o.sources )

    data, glaciers_resampled_dataset = load_timeseries_data(o)

    if o.static_glacier:
        o.regional_glacier = False

    constraints = get_global_constraints(o)

    for source in ["greenland", "antarctica", "steric", "glacier"]:
        key = f"{source}_kwargs"
        options = global_slr_kwargs.setdefault(key, {})
        options["add_noise"] = getattr(o, f"add_{source}_noise")
        if source == "glacier":
            options["noise_on_forcing"] = o.glacier_noise_on_forcing
        options["data"] = glaciers_resampled_dataset.rename({"Sample": "sample", "Time": "year", "Region": "region"}) if source == "glacier" else data[[_source_short(source)]]
        options["sample_dim"] = "sample"
        options["noise_kw"] = {
            "apply_constraints": o.glacier_noise_apply_constraints if source == "glacier" else True,
            "obs_autocorrel": o.noise_obs_autocorrel or getattr(o, f"{source}_noise_obs_autocorrel", False),
            "intercept": o.noise_intercept,
            "sigma_dist": getattr(o, f"{source}_noise_sigma_dist") or o.noise_sigma_dist,
            "sigma_dist_params": _get_param(o, "noise_sigma_dist_params", source),
            "gp_cov": _get_param(o, "noise_gp_cov", source),
            "gp_cov_params": _get_param(o, "noise_gp_cov_params", source),
            "ls_dist": _get_param(o, "noise_ls_dist", source),
            "ls_dist_auto": _get_param(o, "noise_ls_dist_auto", source),
            "ls_dist_params": _get_param(o, "noise_ls_dist_params", source),
            }
        if source == "glacier":
            options["noise_kw"]["clip_noise"] = o.glacier_noise_clip_volume
        options["rate_obs"] = o.noise_on_rate or getattr(o, f"{source}_noise_on_rate", False)

        for c in list(constraints):
            if c.source != _source_short(source):
                continue

            if c.diag in ["slr20", "rate2000"]:
                if options["add_noise"]:
                    logger.info(f"Noise constraint for {source}: skip {c.label}")
                    c.dummy = True

        # pass parameters for the prior distribution
        group = f"{source}_kwargs"
        group_kw = global_slr_kwargs.setdefault(group, {})
        for p in ["prior_dist_q", "prior_dist_params_q",
                  "prior_dist_a", "prior_dist_params_a",
                  "prior_dist_aT0", "prior_dist_params_aT0",
                  "prior_dist_V0", "prior_dist_params_V0",
                  "prior_dist_b", "prior_dist_params_b"]:
            if hasattr(o, f"{source}_{p}"):
                group_kw[p] = getattr(o, f"{source}_{p}")


    # assert global_slr_kwargs["steric_kwargs"]["noise_kw"]["gp"]
    # assert global_slr_kwargs["steric_kwargs"]["noise_kw"]["gp_ls"] == 10

    global_slr_kwargs["greenland_kwargs"]["linear_icesheet"] = o.linear_icesheet
    global_slr_kwargs["antarctica_kwargs"]["linear_icesheet"] = o.linear_icesheet
    global_slr_kwargs["antarctica_kwargs"]["static_antarctica"] = o.static_antarctica
    global_slr_kwargs["antarctica_kwargs"]["ar6_kwargs"] = {"icesheet": o.antarctica_ar6_method}
    global_slr_kwargs["greenland_kwargs"]["static_greenland"] = o.static_greenland
    global_slr_kwargs["steric_kwargs"]["static_steric"] = o.static_steric

    glacier_kwargs = global_slr_kwargs.setdefault("glacier_kwargs", {})
    glacier_kwargs["static_glacier"] = o.static_glacier
    glacier_kwargs["regional_glacier"] = o.regional_glacier
    glacier_kwargs["uncharted_glaciers"] = o.add_uncharted_glaciers
    glacier_kwargs["uncharted_glaciers_distribution"] = o.glacier_uncharted_distribution
    glacier_kwargs["uncharted_glaciers_exclude_antarctica"] = o.glacier_uncharted_exclude_antarctica
    glacier_kwargs["exclude_icesheets"] = o.glacier_exclude_icesheets
    glacier_kwargs["n"] = o.glacier_exponent
    glacier_kwargs["dimensionless"] = o.glacier_dimensionless
    glacier_kwargs["regions"] = o.glacier_regions
    glacier_kwargs.setdefault("dataset_kw", {})["constraints_kwargs"] = {"v2000_source": o.glacier_volume_source}


    global_slr_kwargs["isimip_mode"] = o.isimip_mode
    global_slr_kwargs["isimip_model"] = o.isimip_model
    global_slr_kwargs["isimip_steric_sigma"] = o.isimip_steric_sigma
    global_slr_kwargs["isimip_tas_noise"] = o.isimip_tas_noise
    global_slr_kwargs["isimip_tas_no_obs"] = o.isimip_tas_no_obs

    # In the two-step mode, remove global constraints unless we're in the first, standalone step
    if o.global_slr_method != "estimate_coupled" and not standalone:
        constraints = []
        global_trace = arviz.from_netcdf(o.standalone_global_trace)
        # global_slr_kwargs["data"] = global_trace.constant_data
        global_slr_kwargs["from_trace"] = global_trace

    global_slr_kwargs["constraints"] = constraints

    return global_slr_kwargs


def prepare_standalone_global_model(o):

    if o.standalone_global_trace is not None:
        assert Path(o.standalone_global_trace).exists(), f"{o.standalone_global_trace} does not exist"

    dirname = Path(o.dirname)
    dirname.mkdir(exist_ok=True, parents=True)

    if o.standalone_global_trace is None:
        o.standalone_global_trace = str(dirname / "trace_standalone_global.nc")

    if Path(o.standalone_global_trace).exists():
        logger.info(f"Standalone global model trace already exists: {o.standalone_global_trace}")
        return None

    global_slr_kwargs = get_global_slr_kwargs(o, standalone=True)
    global_model = slr_model_global(experiments=o.experiments, **global_slr_kwargs)

    with global_model:
        trace_standalone_global = pm.sample(random_seed=o.random_seed + 345 if o.random_seed else None, nuts_sampler=o.nuts_sampler)

    print("Save standalone global trace to netCDF", o.standalone_global_trace)
    trace_standalone_global.to_netcdf(o.standalone_global_trace)
    print("global trace saved")

    o.global_slr_method = "load_existing"

    return global_model


def in_bbox(bbox, coord):
    l, r, b, t = bbox
    lon, lat = coord
    lon_mod = np.mod(lon, 360)

    in_latbox = lat >= b and lat <= t

    if not in_latbox:
        return False

    return (
        (lon >= l and lon <= r)
        or
        (lon_mod >= l and lon_mod <= r)
        or
        (lon_mod + 360 >= l and lon_mod + 360 <= r)
        )


def update_stations_constraints_flags(o, stations):

    # update stations' flags
    if 'psmsl' in o.gps_mask:
        for s in stations:
            if s.get('psmsl'):
                s['gps'] = True

    # update stations' flags
    if 'psmsl' in o.satellite_mask:
        for s in stations:
            if s.get('psmsl'):
                s['satellite'] = True

    if 'grid' in o.satellite_mask:
        if o.grid is None:
            raise ValueError('need to provide the --grid parameter with --satellite-mask grid')
        for s in stations:
            if s.get('grid'):
                s['satellite'] = True

    if 'psmsl-no-bbox' in o.satellite_mask:
        if o.grid_bbox is None:
            raise ValueError('psmsl-no-bbox only applies when --grid-bbox are provided')

        for s in stations:
            if s.get('psmsl') and not in_bbox(o.grid_bbox, (s['Longitude'], s['Latitude'])):
                s['satellite'] = True

    if 'coords' in o.satellite_mask:
        if o.coords is None:
            raise ValueError('need to provide the --grid parameter with --satellite-mask coords')
        for s in stations:
            if s.get('coords'):
                s['satellite'] = True


    # By default use tide-gauge constraint for all PSMSL sides
    psmsl_sites = []
    for s in stations:
        if s.get('psmsl'):
            s['tidegauge'] = True
            psmsl_sites.append(s)


    # Determine specific locations and constraints what to leave out
    if o.leave_out_tidegauge_ordinal:
        for i in o.leave_out_tidegauge_ordinal:
            s = stations[i]
            for obs in o.leave_out_obs_type:
                s[obs] = False

    if o.leave_out_tidegauge:
        for s in psmsl_sites:
            if s['ID'] in o.leave_out_tidegauge:
                for obs in o.leave_out_obs_type:
                    s[obs] = False

    if o.leave_out_tidegauge_fraction:
        rng = np.random.default_rng(o.random_seed)
        N = len(psmsl_sites)
        n = int(o.leave_out_tidegauge_fraction * N)
        leave_out_tidegauges = rng.integers(N, size=n)
        for i in leave_out_tidegauges:
            for obs in o.leave_out_obs_type:
                psmsl_sites[i][obs] = False

    if o.leave_out_tidegauge_basin_id:
        for s in psmsl_sites:
            if BASINS.get_region(s['Longitude'], s['Latitude']) in o.leave_out_tidegauge_basin_id:
                for obs in o.leave_out_obs_type:
                    s[obs] = False


def get_gps_constraints(o, stations):
    gps_mask = np.array([r.get('gps', False) for r in stations])
    gps_rates = [r.get('gps_rate', None) for r in stations]
    gps_rate_errors = [r.get('gps_rate_err', None) for r in stations]
    return [ GPSConstraint(observed_mask=gps_mask, dummy="gps" in o.skip_constraints, interpolation_method=o.gps_formal_error,
        dist=o.gps_dist, gps_distance_tol=o.gps_distance_tol, quality=o.gps_quality, scale_error=o.gps_scale_error,
        gps_rates=gps_rates, gps_rate_errors=gps_rate_errors) ]

def ignore_unused_kwargs(f):
    """
    Decorator to ignore unused kwargs in a function
    """

    def wrapper(*args, **kwargs):
        # Remove unused kwargs
        for key in list(kwargs.keys()):
            if key not in f.__code__.co_varnames:
                del kwargs[key]
        return f(*args, **kwargs)

    return wrapper

def get_oceandynsampler(lons, lats, sat_values=None, covariance_source="cmip6", cpus=None, rescale_cmip6=True, cmip6_interp_method=None, models=None):

    if sat_values is None:
        _, sat_values_t = get_satellite_timeseries(lons, lats)
        sat_values_t *= 1000 # meter to millimeter
        sat_values = sat_values_t.T

    if covariance_source == "cmip6":
        # model_data = load_all_cmip6(lon=lons, lat=lats, method_=o.cmip6_interp_method, models=[o.isimip_model] if o.isimip_mode else None)
        model_data = load_all_cmip6(lon=lons, lat=lats, method_=cmip6_interp_method, max_workers=cpus, models=models) # keep all models in all cases for the COV matrix
        assert len(model_data) > 0
        minlength = 2023-1900 + 60  # about 60 samples min to estimate the cov matrix (to catch cycles ~ 60 years)
        oceandynsampler = MeanSampler([CMIP6Sampler(zos.values, model=zos.model, sat_values=sat_values, rescale_like_satellite=rescale_cmip6) for zos in model_data if zos.shape[0] >= minlength])
        if len(oceandynsampler.samplers) == 0:
            raise ValueError(f"Oceandyn sampler failed to be initialized. Model data: {len(model_data)}. zos time lengths: {[zos.shape[0] for zos in model_data]}")

    elif covariance_source == "satellite":
        oceandynsampler = SatelliteAR1Sampler(sat_values=sat_values)
    elif covariance_source == "satellite_eof_ar1":
        oceandynsampler = SatelliteEOFAR1Sampler(sat_values=sat_values)
    else:
        raise NotImplementedError(covariance_source)

    return oceandynsampler


def get_local_constraints(o, stations):

    if o.skip_all_constraints or len(stations) == 0:
        return []

    # if len(o.coords) > 0:
    #     raise NotImplementedError('cannot yet specify constraints for a subset of points -- must be handled in postprocessing (resample_posterior)')

    update_stations_constraints_flags(o, stations)

    station_ids = np.array([r['ID'] for r in stations])
    coords = np.array([[r['Longitude'], r['Latitude']] for r in stations])
    lons, lats = coords.T

    # tidegauge and satellite flags set in `update_stations_constraints_flags`
    psmsl_mask = np.array([r.get('psmsl', False) for r in stations])  # used to identify psmsl sites
    tidegauge_mask = np.array([r.get('tidegauge', False) for r in stations])  # used to identify where to apply constraints
    satellite_mask = np.array([r.get('satellite', False) for r in stations])

    gps_constraints = get_gps_constraints(o, stations)
    # gps_mask = np.array([r.get('gps', False) for r in stations])
    gps_mask = gps_constraints[0].observed_mask

    psmsl_ids = station_ids[psmsl_mask]

    logger.info(f"Constraints mask ({len(stations)} locations in total): {len(psmsl_ids)} tide-gauges, {np.sum(satellite_mask)} satellite locations, {np.sum(gps_mask)} GPS locations")

    constraints = []
    constraints.extend(gps_constraints)

    # Prepare covariance
    # Prepare the "sampler" for constraints' covariance matrices
    _, sat_values_t = get_satellite_timeseries(lons, lats)
    sat_values_t *= 1000 # meter to millimeter
    sat_values = sat_values_t.T

    oceandynsampler = get_oceandynsampler(lons, lats, sat_values=sat_values,
                                          covariance_source=o.covariance_source,
                                          cpus=o.cpus,
                                          rescale_cmip6=o.rescale_cmip6,
                                          cmip6_interp_method=o.cmip6_interp_method)


    tidegaugedata = load_tidegauge_records(
        psmsl_ids,
        version=o.psmsl_label,
        remove_meteo=o.remove_meteo,
        wind_correction=o.wind_correction,
        classical_formula_for_tides=o.classical_formula_for_tides)


    # various ways of using the tide-gauge variance:

    # 1. Fix a in #46 : anything bigger than oceandyn goes as uncorrelated measurement error (and anything smaller is ignored)
    if o.estimate_tidegauge_measurement_error:
        assert not o.scale_tidegauge_oceandyn, "--estimate-tidegauge-measurement-error and --scale-tidegauge-oceandyn are mutually exclusive"
        tidegauge_measurement_error = estimate_tidegauge_error(tidegaugedata, o.tidegauge_measurement_error)[tidegauge_mask]
        scale_oceandyn_samples = None

    # 2. Fix b in #46 : full correlation, and scaling down is also OK
    elif o.scale_tidegauge_oceandyn:
        tidegauge_measurement_error = o.tidegauge_measurement_error
        scale_oceandyn_samples = get_tidegauge_to_satellite_residual_ratio(tidegaugedata)[tidegauge_mask]

    else:
        tidegauge_measurement_error = o.tidegauge_measurement_error
        scale_oceandyn_samples = None

    if o.satellite_measurement_error_method == "prandi2021":
        satellite_measurement_error = get_satellite_error(lons, lats)[satellite_mask]
        # Some values in Hudson Bays are close to zero: let's take the safe side here.
        satellite_measurement_error[satellite_measurement_error < 0.1] = 0.1

    elif o.satellite_measurement_error_method == "constant":
        satellite_measurement_error = o.satellite_measurement_error
        # satellite_measurement_error = o.satellite_measurement_error
    else:
        raise NotImplementedError(o.satellite_measurement_error_method)

    sat_kwargs = dict(observed_mask=satellite_mask, oceandyn_surrogate_sampler=oceandynsampler, measurement_error=satellite_measurement_error)
    tg_kwargs = dict(observed_mask=tidegauge_mask, oceandyn_surrogate_sampler=oceandynsampler, data=tidegaugedata,
                    measurement_error=tidegauge_measurement_error, scale_oceandyn_samples=scale_oceandyn_samples)
    trend_kwargs = dict(model_mean_rate_instead_of_lintrend=o.model_mean_rate_instead_of_lintrend)

    if o.method == "trend":
        sat = SatelliteTrend("satellite", **sat_kwargs, **trend_kwargs)
        tg = TideGaugeTrend("tidegauge", **tg_kwargs, **trend_kwargs)
        tg_pre = TideGaugeTrend("tidegauge_1900_1990", year_end=1990, **tg_kwargs, **trend_kwargs)
        tg_post = TideGaugeTrend("tidegauge_1990_2018", year_start=1990, **tg_kwargs, **trend_kwargs)

    else:
        raise ValueError(f"Unknown method {o.method} for local constraints")


    if o.mask_pre_1990 or o.mask_post_1990 or o.split_tidegauges:
        sub_constraints = [tg, tg_pre, tg_post, sat]
    else:
        sub_constraints = [tg, sat] # don't bother with pre and post bits otherwise

    # HACK: add a diagnostic-only constraint of satellite at tide gauges, in case it is not provided
    if "psmsl" not in o.satellite_mask:
        import copy
        sat_psmsl = copy.copy(sat)
        sat_psmsl.name = "satellite_psmsl"
        sat_psmsl.skip_likelihood = True
        sat_psmsl.observed_mask = psmsl_mask
        sat_psmsl.station_mask = ~psmsl_mask
        sub_constraints.append(sat_psmsl)

    if o.mask_pre_1990:
        o.split_tidegauges = True
        tg_pre.skip_likelihood = True

    if o.mask_post_1990:
        o.split_tidegauges = True
        tg_post.skip_likelihood = True

    if o.split_tidegauges:
        tg.skip_likelihood = True
    else:
        tg_pre.skip_likelihood = True
        tg_post.skip_likelihood = True

    if "tidegauge" in o.skip_constraints or "satellite" in o.skip_constraints:
        o.mixed_covariance = False

    if "tidegauge" in o.skip_constraints:
        tg.skip_likelihood = True
        tg_post.skip_likelihood = True
        tg_pre.skip_likelihood = True

    if "satellite" in o.skip_constraints:
        sat.skip_likelihood = True

    if o.mixed_covariance:
        mixed = lambda c : not c.skip_likelihood and c.name not in o.independent_constraints # not correlated to satellite
        mixed_skipped = [c for c in sub_constraints if not mixed(c)]
        sub_constraints = [c for c in sub_constraints if mixed(c)]

        constraints.append(
            MixedConstraint("mixed", constraints=sub_constraints, oceandyn_surrogate_sampler=oceandynsampler)
        )
        constraints.extend(mixed_skipped)

    else:
        constraints.extend(sub_constraints)

    return constraints


def update_options(o):

    if o.global_slr_only:
        o.number = 0
        o.skip_constraints = ['tidegauge', 'satellite', 'gps']
        o.oceandyn = False # to avoid loading anything

    if o.experiments is None:
        if o.isimip_mode:
            o.experiments = DEFAULT_ISIMIP_EXPERIMENTS
        else:
            o.experiments = DEFAULT_EXPERIMENTS

    if o.glacier_mm21_drop_20CRV3:
        o.glacier_mm21_forcing = [f for f in MM21_FORCING if f != "20CRV3"]

    for name in o.add_constraints:
        if name in o.skip_constraints:
            o.skip_constraints.remove(name)
        else:
            logger.warning(f'--add-constraints {name} :  not found in skipped list. Nothing to do.')

    if o.psmsl_ids is None:
        psmsl_ids = get_station_ids(min_years=o.min_years, included_in_frederikse=o.frederikse_only, version=o.psmsl_label, flagged=o.flagged)

        if o.number is not None:
            psmsl_ids = psmsl_ids[:o.number]

        o.psmsl_ids = np.asarray(psmsl_ids).tolist()

    if o.isimip_mode:
        if not o.isimip_model:
            raise ValueError(f"Please provide one ISIMIP model in ISIMIP mode.")
        if not o.isimip_all_scaling_patterns:
            if o.oceandyn_models is None:
            # assert o.oceandyn_models is None or o.oceandyn_models == [o.isimip_model], 'cannot provide --oceandyn-models in ISIMIP mode'
                logger.info(f"set oceandyn_models to {o.isimip_model}")
                o.oceandyn_models = [o.isimip_model]
            else:
                logger.warning(f"--oceandyn-models already set. Do not change despite ISIMIP mode")
        else:
            logger.info(f"Keep all scaling patterns despite ISIMIP mode")

    if o.leave_out_tidegauge_basin:
        inv_map = {v:k for k,v in BASINS.map.items()}
        o.leave_out_tidegauge_basin_id = [inv_map[id] for id in o.leave_out_tidegauge_basin]

    if o.static_glacier:
        o.regional_glacier = False
        o.add_uncharted_glaciers = False  # static glacier already adds uncharted glaciers since it draws on Frederikse et al


def get_stations(o):
    # manually indicated stations
    if getattr(o, "station_ids") is not None:
        logger.warning("station selection is overriden by station_ids")
        stations = get_stations_from_ids(o.station_ids, psmsl_flags={"psmsl": True}, coord_flags={"grid": True, "coords": True} )
    else:
        stations = get_stations_from_psmsl_ids(o.psmsl_ids, {'psmsl': True}) + get_stations_from_coords(o.coords, {'coords': True}) + get_stations_from_grid(o.grid, {'grid': True}, bbox=o.grid_bbox) + get_non_psmsl_gps_stations(o.include_gps_stations, o.psmsl_ids, {'gps': True})
    return stations


def get_model_kwargs(o):

    update_options(o)

    # grid, satellite_mask.., get_station_from_grid
    stations = get_stations(o)

    # if o.merge_stations:
    #     stations = merge_stations(stations)

    constraints = get_local_constraints(o, stations)

    global_slr_kwargs = get_global_slr_kwargs(o)

    kwargs = dict(
            experiments=o.experiments,
            stations=stations,
            global_slr_kwargs=global_slr_kwargs,
            slr_kwargs = dict(
                vlm_res_mode=o.vlm_res_mode,
                vlm_res_split_year=o.vlm_res_split_year,
                vlm_res_autocorrel=o.vlm_res_autocorrel,
                vlm_res_sd=o.vlm_res_sd,
                vlm_res_cauchy=not o.vlm_res_normal,
                vlm_res_domain=o.vlm_res_domain,
                vlm_res_spatial_scale=o.vlm_res_spatial_scale,
                historical_mass_fingerprints=o.historical_mass_fingerprints,
                historical_window=o.historical_window,
                steric_kwargs={
                    "steric_coef_cov": o.steric_coef_cov,
                    "oceandyn": o.oceandyn,
                    "steric_coef_error": not o.mean_oceandyn,
                    "reduced_space_coef": o.steric_coef_eof,
                    "steric_coef_scale": o.steric_coef_scale,
                    "models": o.oceandyn_models,
                    },
                gia_eof=o.gia_eof,
                gia_eof_num=o.gia_eof_num,
                regional_glacier=o.regional_glacier,
            ),
             diag_kwargs={
                 "save_timeseries": o.save_timeseries,
             },
             constraints=constraints,
        )

    return kwargs


def get_model(o, model_kwargs={}):

    kwargs = get_model_kwargs(o)
    kwargs = {**kwargs, **model_kwargs}

    # print("global SLR model", o.global_slr_method)
    # print("tidegauge version", o.psmsl_label)
    # print("remove meteo", o.remove_meteo)
    # print("wind correction", o.wind_correction)
    # print("only Frederikse's stations?", o.frederikse_only)
    # print(len(kwargs["stations"]), "stations")
    # print("tidegauge and satellite method", o.method)
    # print("covariance method", o.covariance_method)
    # print("covariance source", o.covariance_source)

    model = slr_model_tidegauges(**kwargs)

    return model, kwargs

def run_quiet(*args, log_level=logging.WARNING, **kwargs):
    from sealevelbayes.runutils import CaptureOutput
    with CaptureOutput():
        return run(*args, **kwargs)

def run(cmd=[], no_disk=False, **options):

    parser = get_parser()

    # group.add_argument_group("Multiple Experiments")
    # group.add_parser("--repeat-leave-tidegauge-out", action='store_true', help='apply --leave-out-tidegauge for all tide-gauge')
    # group.add_parser("--repeat-leave-tidegauge-fraction-out", type=int, help='apply --leave-out-tidegauge-fraction a number of times')

    if cmd and type(cmd) is str:
        cmd = cmd.strip().split()

    o = parse_args(cmd, parser, options)

    print("============")
    print(f"sealevelbayes-run {' '.join(cmd or sys.argv[1:])}")
    print("============")


    setup_logger(o)

    o.cirun = get_runid(parser, **vars(o))

    if not o.dirname:
        dirname = get_runpath(o.cirun)
        o.dirname = str(dirname)

    if no_disk:
        o.cirun = None
        o.dirname = None
        o.resume = False

    else:
        print_experiment_info(o.cirun)
        logger.info(f"Simulation will be saved in {o.dirname}")
        filepath_options = Path(o.dirname) / "options.json"
        filepath_model = Path(o.dirname) / "config.cpk"
        filepath_trace = Path(o.dirname) / "trace.nc"
        filepath_watermark = Path(o.dirname) / "watermark.json"

        watermark_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "commit": get_version(), # will try to use git
                "watermark": get_watermark(),
                }

    if o.resume and filepath_model.exists():

        # simulate standalone global SLR in case it's missing
        if o.add_global_slr_only:
            update_options(o)
            prepare_standalone_global_model(o)

        logger.info("Load pickled model")
        pickled = cloudpickle.load(open(filepath_model, 'rb'))
        model = pickled['model']
        o2 = pickled['options']
        # update options
        for k, v in vars(o2).items():
            if k in ['resume', 'cirun', 'no_sampling', 'json', 'figs', 'cpus', 'chains', 'add_global_slr_only']:
                continue
            setattr(o, k, v)
        options = dict(vars(o))

    else:

        update_options(o)

        if o.global_slr_method != "estimate_coupled" or o.add_global_slr_only:
            global_model = prepare_standalone_global_model(o)
        else:
            global_model = None

        # In the two-step approach, restrict the fitting step to only one experiment for tuning
        # (other experiments can be obtained later with ExperimentTrace's resampling)
        if o.global_slr_method == "two-step":
            keep_experiments = o.experiments
            o.experiments = ['ssp585_mu']  # only need one exp for the fit

        model, model_kwargs = get_model(o)

        if no_disk and o.no_sampling:
            return model

        if o.global_slr_method == "two-step":
            o.experiments = keep_experiments

        stations = model_kwargs['stations']

        if not no_disk:
            dirname.mkdir(exist_ok=True, parents=True)

            # Save options to runslr (for later re-use)
            options = dict(vars(o))
            logger.info(f"Save config to json {filepath_options}")
            json.dump(options, open(filepath_options, "w"))

            logger.info(f"Save model to cloudpickle {filepath_model}")
            cloudpickle.dump({"model": model, "global_model": global_model, "stations": stations, "options": o}, open(filepath_model, "wb"))

            # Save model kwargs (for debug purposes only, since some objects like arrays and Constraints class cannot be serialized, so the info is incomplete)
            filepath = dirname / "options-debug.json"
            logger.info(f"Save simplified model kwargs to json {filepath}")
            json.dump(model_kwargs, open(filepath, "w"), default=repr) # the default param is a fallback for non-serializable object (e.g. Constraint)
            json.dump(watermark_data, open(filepath_watermark, "w"))

            if os.path.exists(filepath_trace):
                logger.warning(f"Trace file {filepath_trace} already exists. Remove it to re-run the sampling.")
                # shutil.rmtree(filepath_trace)
                shutil.move(filepath_trace, filepath_trace.with_suffix(".old"))

    if o.no_sampling:
        logger.info("--no-sampling :: Exit before sampling.")
        from sealevelbayes.postproc.run import ExperimentTrace
        return ExperimentTrace(None, vars(o), model)


    for k, v in vars(o).items():
        logger.info(f"{k}: {v}")

    for v in model.unobserved_RVs:
        logger.info(str(v))

    for v in model.observed_RVs:
        logger.info(str(v))

    if not (o.resume and filepath_trace.exists()):
        logger.info(f"Proceed with sampling (cpus={o.cpus}, chains={o.chains}, draws={o.draws}, tune={o.tune}).")

        with model:
            trace_k = pm.sample(chains=o.chains, random_seed=o.random_seed + 829 if o.random_seed else None,
                                tune=o.tune, draws=o.draws,
                                cores=o.cpus, nuts_sampler=o.nuts_sampler, **(dict(target_accept=o.target_accept) if o.target_accept else {}))

        if no_disk:
            from sealevelbayes.postproc.run import ExperimentTrace
            return ExperimentTrace(trace_k, vars(o), model)

        logger.info(f"Save trace to netCDF {filepath_trace}")
        trace_k.to_netcdf(filepath_trace)
        logger.info("trace saved")

    else:
        trace_k = arviz.from_netcdf(filepath_trace)

    REPODIR = Path(sealevelbayes.__path__[0]).parent

    if o.figs:
        # logger.warning("--figs option currently not implemented")
        logger.info("Crunch figures")
        new_env = dict(os.environ);
        new_env['CIRUN'] = o.cirun
        notebookdir = get_webpath(o.cirun) / "notebooks"
        notebookdir.mkdir(parents=True, exist_ok=True)
        json.dump(watermark_data, open(notebookdir.parent / "watermark.json", "w"))
        json.dump(options, open(notebookdir.parent / filepath_options.name, "w"))

        logger.info(f"figure folder: {get_webpath(o.cirun)/'figures'}")
        try:
            # sp.check_call(f"jupyter execute --inplace  --allow-errors ./notebooks/figures-global.ipynb", shell=True, env=new_env, cwd=REPODIR)
            # sp.check_call(f"jupyter execute --inplace  --allow-errors ./notebooks/figures-local.ipynb", shell=True, env=new_env, cwd=REPODIR) # fixed now?
            sp.check_call(f"jupyter nbconvert --to html --execute --allow-errors ./notebooks/figures-global.ipynb --output {notebookdir}/figures-global.html", shell=True, env=new_env, cwd=REPODIR)
            if not o.global_slr_only:
                sp.check_call(f"jupyter nbconvert --to html --execute --allow-errors ./notebooks/figures-local.ipynb --output {notebookdir}/figures-local.html", shell=True, env=new_env, cwd=REPODIR)

        except ValueError as error:
            logger.info(error)
            logger.warning("failed to crunch figs")
    if o.json:
        logger.warning("--json option currently not implemented")
        from sealevelbayes.postproc.web import main
        main([o.cirun])
        # print("Crunch json files")
        # sp.check_call(f"{PYTHONEXE} ./scripts/resample_trace_to_json_files.py {cirun} --batch 100", shell=True, cwd=REPODIR)


    print("Finished successfully. Exit.")

    from sealevelbayes.postproc.run import ExperimentTrace
    return ExperimentTrace(trace_k, vars(o), model)


def main():
    return run(None)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()