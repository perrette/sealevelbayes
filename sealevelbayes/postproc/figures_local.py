import os
from pathlib import Path
import json, os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xa
import arviz
from arviz.stats.density_utils import kde as arviz_kde

import sealevelbayes
from sealevelbayes.config import CONFIG
from sealevelbayes.config import get_runpath, get_webpath
from sealevelbayes.postproc.serialize import trace_to_json, serialize_trace, split_stations, obsjs_from_trace
from sealevelbayes.logs import logger
from sealevelbayes.postproc.sensitivityfigs import sample_pp as _sample_pp, get_kde_plot_data as _get_kde_plot_data
from sealevelbayes.postproc.figures import get_model_with_full_local_constraints, SELECT_EXPERIMENTS, SELECT_EXPERIMENTS_mu

def get_figdir(cirun):
    return get_webpath(cirun) / 'figures'

def sample_pp(tr, var_names=None, return_trace=False, model=None):
    if model is not None:
        alt_model = model
    else:
        alt_model = get_model_with_full_local_constraints(tr)
    return _sample_pp(alt_model, tr.cirun, var_names=var_names, return_trace=return_trace)


def prepare_barplot_json(tr, constant_data=None, load=True, save=True):

    figdir = get_figdir(tr.cirun)
    jsname = figdir / "tidegauges_barplot.json"

    if load and jsname.exists():
        print("load", jsname)
        stations_js_bar = json.load(open(jsname))

    else:
        pptrace = sample_pp(tr, return_trace=True)
        ppp = arviz.extract(pptrace.posterior_predictive)
        # ppp = arviz.extract(pp.posterior_predictive if hasattr(pp, "posterior_predictive") else pp)
        if constant_data is None:
            # constant_data = pptrace.constant_data
            constant_data = tr.trace.constant_data

        # make sure all obs data are present in trace.constant_data
        ser = serialize_trace(tr.trace.sel(experiment=['ssp126_mu']), diags=['rate2000'],
                              posterior_predictive=ppp, constant_data=constant_data)
        stations_js_bar, _ = split_stations(ser)
        # stations_js_bar, _ = trace_to_json(tr.trace.sel(experiment=['ssp126_mu']), diags=['rate2000'])

        if save:
            json.dump(stations_js_bar, open(jsname, 'w'))

    return stations_js_bar


def get_kde_plot_data(tr):
    alt_model = get_model_with_full_local_constraints(tr, tr.cirun)
    return _get_kde_plot_data(alt_model, tr.cirun)


def get_obs_match_stats(recs, label=None, verbose=False):
    if verbose: print("---")
    if verbose: print(label, len(recs))
    if verbose: print("---")
    allgood_mask = None
    for obs in ["tidegauge", "gps", "satellite", "all"]:
        if obs != "all":
            obs_values = np.array([o["obs"] for r in recs for o in r["obs"]["obs"] if o["name"] == obs])
            assert len(obs_values) == len(recs), f"{len(obs_values)} != {len(recs)}"
            lower, upper = np.array([(m["lower"], m["upper"]) for r in recs for m in r["posterior_predictive"] if m["name"] == obs + '_obs']).T
            assert len(lower) == len(recs), f"{len(lower)} != {len(recs)}"
            assert len(upper) == len(recs), f"{len(upper)} != {len(recs)}"
            good_mask = (obs_values > lower) & (obs_values < upper)
            if allgood_mask is None:
                allgood_mask = good_mask
            else:
                allgood_mask = good_mask & allgood_mask
        else:
            good_mask = allgood_mask
        good = good_mask.sum()
        f_good = good / len(recs)
        f_bad = 1 - f_good
        if verbose: print(obs, f"good:  {f_good:.2f} ({good})", f"bad: {f_bad:.2f} ({len(recs) - good})")
        yield obs, f_good

def print_stats(recs, label):
    list(get_obs_match_stats(recs, label, verbose=True))


def get_trace_proj(tr, experiments=SELECT_EXPERIMENTS+SELECT_EXPERIMENTS_mu+["ssp126_mu", "ssp585_mu"],
                    sources=["total"],
                    diags=['proj2100', 'rate2100', 'rate2000'],
                    fields=['rsl', 'rad', 'gsl', 'global'],
                    suffix="", load=True, save=True, **kw):

    postprocfolder = get_runpath(tr.cirun) / 'postproc'
    trace_proj_file = postprocfolder / f"trace_proj{suffix}.nc"
    trace_proj = None

    if load and trace_proj_file.exists():
        # trace_proj = xa.open_dataset(trace_proj_file)
        trace_proj = arviz.from_netcdf(trace_proj_file)

    else:
        trace_proj = tr.resample_posterior( fields=fields,
                                                    diags=diags,
                                                    sources=sources,
    #                                                 experiments=DEFAULT_EXPERIMENTS)
                                                    experiments=experiments, **kw)

        if save: trace_proj.to_netcdf(trace_proj_file)

    return trace_proj


def get_trace_proj_trend(tr, suffix="_trend", **kw):
    update_runslr_params = {
                    "add_greenland_noise": False,
                    "add_steric_noise": False,
                    "add_antarctica_noise": False,
                    "add_glacier_noise": False
                    }
    return get_trace_proj(tr, update_runslr_params=update_runslr_params, suffix=suffix, **kw)
