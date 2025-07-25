from pathlib import Path
import json
import os
from itertools import groupby, cycle
import tqdm
import pickle
import numpy as np
from scipy import stats
import xarray as xa
import arviz
from arviz.stats.density_utils import kde as arviz_kde
# reload(postproc)
import pymc as pm

import matplotlib.pyplot as plt

from sealevelbayes.config import CONFIG, get_runpath, get_webpath
from sealevelbayes.logs import logger
from sealevelbayes.datasets.tidegaugeobs import psmsl_ids_to_basin
from sealevelbayes.datasets.ar6.supp import open_slr_global, open_slr_regional
from sealevelbayes.postproc.run import (
    ExperimentTrace, get_model, get_local_constraints, get_model_kwargs,
    slr_model_tidegauges, get_stations
)
from sealevelbayes.postproc.figures import BarPlot, sort_basins, plot_obs_from_dataset, argsort_stations_by_trace
from sealevelbayes.postproc.colors import basincolors
from sealevelbayes.postproc.serialize import trace_to_json, serialize_trace, split_stations
# from sealevelbayes.models.localslr import SOURCES
# reload(figures)
from sealevelbayes.postproc.figures import make_proj_figure, SELECT_EXPERIMENTS, SELECT_EXPERIMENTS_mu
from sealevelbayes.models.globalslr import SSP_EXPERIMENTS, SSP_EXPERIMENTS_mu
from sealevelbayes.postproc.featured import get_featured_locations
from sealevelbayes.postproc.figures import xcolors, xlabels
# from sealevelbayes.models.globalslr import DEFAULT_EXPERIMENTS


def sample_pp(model, cirun, var_names=None, trace=None, return_trace=False):

    runfolder = get_runpath(cirun)
    fname0 = runfolder / "posterior_predictive.nc"
    if var_names is None:
        var_names = ["gps_obs", "tidegauge_obs", "satellite_obs"]

    if set(var_names) == set(["gps_obs", "tidegauge_obs", "satellite_obs"]):
        fname = fname0
    else:
        fname = runfolder / f"posterior_predictive_{'_'.join(var_names)}.nc"

    if fname0.exists():
        fname = fname0

    if fname.exists():
        print("Load", fname)
        ds = xa.open_dataset(fname)
        if ds.variables or ds.dims:
            if return_trace:
                raise ValueError("return_trace=True but the file is not a trace -- delete and run this function again")
            pp = ds
            pptrace = ds  # for the ".close()" method

        else:
            ds.close()
            pptrace = arviz.from_netcdf(fname)
            # if not hasattr(pptrace, "posterior_predictive") and hasattr(pptrace, "posterior"):
            #     logger.warning("posterior_predictive group not found. Load posterior group instead.")
            #     pp = pptrace.posterior
            # else:
            pp = pptrace.posterior_predictive

        check_all = True
        for v in var_names:
            if v not in pp:
                print(f"Missing {v} in {fname}")
                check_all = False

        if check_all:
            return pptrace if return_trace else pp

        print(f"Some variable not found in {fname}. Crunch again.")
        pptrace.close()
    else:
        print(f"{fname} not found, crunch")

    if trace is None:
        trace = arviz.from_netcdf(runfolder/"trace.nc")

    if model is None:
        import cloudpickle
        model = cloudpickle.load(open(runfolder/"config.cpk", "rb"))["model"]

    assert hasattr(model, "tidegauge_obs"), "tidegauge_obs not found in model. Try ExperimentTrace.add_missing_local_constraints()"
    assert hasattr(model, "satellite_obs"), "satellite_obs not found in model. Try ExperimentTrace.add_missing_local_constraints()"
    assert hasattr(model, "gps_obs"), "gps_obs not found in model. Try ExperimentTrace.add_missing_local_constraints()"

    print("sample_posterior_predictive", cirun, var_names)

    with model:
        # if not hasattr(tr0.model, "gps_obs"):
        #     tr0.add_gps_constraint()
        pptrace = pm.sample_posterior_predictive(trace, var_names=var_names)
#     pp = tr.sample_posterior_predictive(var_names=var_names, redefine_model=True, skip_all_constraints=False)

    print("Save to", fname)
    pptrace.to_netcdf(fname)
    return pptrace if return_trace else pptrace.posterior_predictive

def _get_kde_raw(yens):
    if yens.ndim == 1: yens = yens[:, None]
    return [arviz_kde(yens[:, i]) for i in tqdm.tqdm(list(range(0, yens.shape[1])))]

def _get_kde_raw_plot_data(model0, cirun=None, trace=None, constant_data=None, var_names=None, **kw):
    figdir = get_webpath(cirun)/'figures'

    if var_names is not None:
        fname = figdir / f"kde_plot_data_raw_{'_'.join(var_names)}.pickle"
    else:
        fname = figdir / "kde_plot_data_raw.pickle"

    if fname.exists():
        print("Load", fname)
        return pickle.load(open(fname, "rb"))

    else:
        runfolder = get_runpath(cirun)
        trace = arviz.from_netcdf(runfolder/"trace.nc")
        pp = sample_pp(model0, cirun, var_names=var_names, trace=trace, **kw)

        kde_raw = {}

        if constant_data is None:
            constant_data = trace.constant_data

        for name in ["tidegauge", "gps", "satellite"]:
            if var_names and name+"_obs" not in var_names: continue
            print("compute KDEs for", name)
            post = arviz.extract(trace.posterior[[name]])[name].values
            post_pred = arviz.extract(pp[[name+'_obs']])[name+'_obs'].values
            if name + "_mu" not in constant_data:
                logger.warning(f"Missing {name}_mu in constant_data")
                obs = None
            else:
                obs = constant_data[name+'_mu'].values

            kde_raw[name] = {
                "posterior": _get_kde_raw(post),
                "posterior_predictive": _get_kde_raw(post_pred),
                "obs": _get_kde_raw(obs) if obs is not None else None,
            }

        print("Save to", fname)
        pickle.dump(kde_raw, open(fname, "wb"))

    return kde_raw

def _interp_kdes(bins, kdes_raw):
    kdes = np.empty((len(kdes_raw), len(bins)))
    for i, (grid, density) in enumerate(kdes_raw):
        kdes[i] = np.interp(bins, grid, density)
    return kdes.squeeze()


def get_kde_plot_data(model0, cirun, **kw):

    kde_raw = _get_kde_raw_plot_data(model0, cirun, **kw)

    plot_data = {}
    bins = np.arange(-20, 20, .01)

    for name in kde_raw:
        plot_data[name] = {
            k : _interp_kdes(bins, kde_raw[name][k]) for k in kde_raw[name]
        }
        plot_data[name]['bins'] = bins

    return plot_data

def makefig_constraints_validation_kde_distributions_pp(tr0, DEFAULT, skip_constraint):
    # DEFAULT = sensitivity_experiments['loose GPS']
    # tr0 = ExperimentTrace.load(CIDIR / DEFAULT)

    # stations = get_stations(tr0.trace.constant_data.psmsl_ids.values)
    stations = get_stations(tr0.o)
    local_constraints = get_local_constraints(tr0.o, stations)

    trace0 = tr0.trace
    plot_data = {}

    for name in ["default"] + list(skip_constraint):
    #     print("Load plot data for", name)
        if name == "default":
            plot_data['default'] = get_kde_plot_data(tr0.model, DEFAULT)
        elif name == "all":
            plot_data['all'] = get_kde_plot_data(tr0.model, skip_constraint["all"])
        else:
            plot_data[name] = get_kde_plot_data(tr0.model, skip_constraint[name], var_names=[name+'_obs'])
    #         plot_data[name] = get_kde_plot_data(skip_constraint[name])


    f, axes = plt.subplots(3, 1, figsize=(8, 10))

    for letter, name, ax in zip("ABC", [k for k in skip_constraint if k != "default"], axes):

        edges = np.arange(-20, 20, .01)

        def axhist(bins, kdes, label, histtype=None, alpha=.3, zorder=0, linestyle=None, **kw):
            y, lo, hi = np.percentile(kdes, [50, 5, 95], axis=0)
            l, = ax.plot(bins, y, linewidth=2, label=label, zorder=zorder, linestyle=linestyle, **kw)
            ax.fill_between(bins, lo, hi, alpha=alpha, zorder=zorder-1, **kw)
            return l

        l = axhist(plot_data["default"][name]['bins'], plot_data["default"][name]['posterior_predictive'], label=f"default")
    #     if name == "gps":
        axhist(plot_data["default"][name]['bins'], plot_data["default"][name]['posterior'], label=f"default (posterior)", alpha=0, linestyle="--", color=l.get_color())
        l2 = axhist(plot_data[name][name]['bins'], plot_data[name][name]['posterior_predictive'], label=f"no {name} constraint")
        l3 = axhist(plot_data["all"][name]['bins'], plot_data["all"][name]['posterior_predictive'], label=f"no constraints")

    #         axhist(plot_data[name][name]['bins'], plot_data[name][name]['posterior'], label=f"no {name} constraint (posterior)", alpha=0, linestyle="--", color=l2.get_color())
    #         axhist(plot_data["all"][name]['bins'], plot_data["all"][name]['posterior'], label=f"no constraints (posterior)", alpha=0, linestyle="--", color=l3.get_color())

    #     ax.plot(plot_data[name][name]['bins'], plot_data[name][name]['obs'], label=f"{name} obs", color="k", lw=2)
        ax.plot(plot_data["default"][name]['bins'], plot_data["default"][name]['obs'], label=f"{name} obs", color="k", lw=2)

        leg = ax.legend(title="($\mathbf{"+letter+"}$) "+name, loc="upper left")
        leg._legend_box.align = "left"

        model_names = {"gps": "VLM", "satellite": "GSL", "tidegauge": "RSL"}

        ax.set_xlabel(f"({letter}) {model_names.get(name)} and {name} (mm/yr)")
        ax.set_ylabel("Locations density")

        xlims = {
            "tidegauge": [-7, 10],
            "gps": [-7, 7],
            "satellite": [0, 7],
        }
        ax.set_xlim(xlims.get(name))

    plt.tight_layout()

    return f, axes
    # f.savefig(figdir/"tidegauge_constraints_validation_posterior_predictive.png", dpi=300)

# def _get_data_barplot_skip_one_constraint(tr0, skip_constraint):
#     results = {}
#     for obs in ["tidegauge", "gps", "satellite"]:
#         cirun = skip_constraint[obs]
#         posterior_predictive = sample_pp(tr0.model, cirun, var_names=[obs+"_obs"], return_trace=False)
#         ser = serialize_trace(None, diags=['rate2000'], fields=[{"tidegauge": "rsl", "gps": "rad", "satellite": "gsl"}[obs]],
#                               posterior_predictive=arviz.extract(posterior_predictive),
#                               constant_data=tr0.trace.constant_data)
#         stations_js_bar, _ = split_stations(ser)
#         results[obs] = stations_js_bar
#     return results
#         # station_js = trace_to_json(tracepp)

# def get_data_barplot_skip_one_constraint(tr0, DEFAULT, skip_constraint, load=True, save=True):
#     fname = get_webpath(DEFAULT)/"figures"/"tidegauges_barplot_skip_one_constraint.json"
#     if load and fname.exists():
#         print("Load", fname)
#         return json.load(open(fname))
#     data = _get_data_barplot_skip_one_constraint(tr0, skip_constraint)
#     if save:
#         print("Save to", fname)
#         def fallback(obj):
#             if isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             if isinstance(obj, np.generic):
#                 return obj.item()
#             raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))
#         json.dump(data, open(fname, "w"), default=fallback)
#     return data

def get_data_barplot_skip_one_constraint(tr0, skip_constraint):
    posterior_predictive = {}
    posterior = {}
    for obs in ["tidegauge", "gps", "satellite"]:
        cirun = skip_constraint[obs]
        pp = sample_pp(tr0.model, cirun, var_names=[obs+"_obs"], return_trace=False)
        posterior_predictive[obs+"_obs"] = pp[obs+"_obs"]
        with xa.open_dataset(get_runpath(cirun) / "trace.nc", group="posterior") as posterior_:
            posterior[obs] = posterior_[obs]
    return arviz.InferenceData(posterior=xa.Dataset(posterior),
                               posterior_predictive=xa.Dataset(posterior_predictive),
                               constant_data=tr0.trace.constant_data)

# def makefig_barplot_skip_one_constraint(tr0, DEFAULT, skip_constraint):
def makefig_barplot_skip_one_constraint(idata, idata0=0, **kw):
    f, axes = plot_obs_from_dataset(idata, **kw)
    for ax in axes.flat:
        ax.set_ylim([-8, 12])
    f.suptitle("Validation: model posterior (predictive) without the obs, versus the obs")

    if idata0 is not None:
        i = argsort_stations_by_trace(idata0)
        idata0 = idata0.isel(station=i)
        for i, (ax, obs) in enumerate(zip(axes.flat, ["tidegauge", "gps", "satellite"])):
            sample_dims = ["chain", "draw"]
            # ax.plot(idata0["posterior"][obs].median(dim=sample_dims).values, ".", color="tab:orange", lw=1, markersize=2, label="full model (median)")
            ax.plot(idata0["posterior"][obs].median(dim=sample_dims).values, color="tab:orange", lw=.5, markersize=2, label="full model (median)")
            ax.legend()

    f.tight_layout()
    return f, axes





    # f, axes = plt.subplots(3, 1, figsize=(8, 10))
    # for i, (ax, obs) in enumerate(zip(axes, ["tidegauge", "gps", "satellite"])):
    #     bp = BarPlot(data[obs])
    #     bp.plot_obs(obs, label=None, ax=ax, panel="ABC"[i], legend_kw={})
    #     bp.set_axis_layout(ax)
    # f.tight_layout()
    # _get_data_barplot_skip_one_constraint_io

# def makefig_constraints_validation_kde_distributions_pp_v2(trace0, skip_constraint):
#     import tqdm
#     import json
#     import pickle
#     import numpy as np
#     from scipy import stats
#     import xarray as xa
#     import arviz
#     from arviz.stats.density_utils import kde as arviz_kde
#     import pymc as pm
#     from sealevelbayes.runslr import ExperimentTrace, get_model, get_local_constraints, get_model_kwargs, slr_model_tidegauges
#     from sealevelbayes.postproc.figures import BarPlot

#     # DEFAULT = sensitivity_experiments['loose GPS']
#     tr0 = ExperimentTrace.load(CIDIR / DEFAULT)

#     show_experiments = ["default", ""]

#     def sample_pp(cirun, var_names=None, dataset_backend=False, trace=None):

#         runfolder = CIDIR / cirun
#         fname0 = runfolder / "posterior_predictive.nc"
#         if var_names is None:
#             fname = fname0
#         else:
#             fname = runfolder / f"posterior_predictive_{'_'.join(var_names)}.nc"

#         if fname0.exists():
#             fname = fname0

#         if fname.exists():
#             print("Load", fname)
#             if dataset_backend:
#                 return xa.open_dataset(fname)
#             else:
#                 return arviz.from_netcdf(fname).posterior_predictive

#         print(fname,"not found, crunch")

#         tr = ExperimentTrace.load(cirun)

#         if not hasattr(tr0.trace, "posterior_predictive") and not hasattr(tr0.model, "tidegauge_obs") and not hasattr(tr0.model, "satellite_obs") and hasattr(tr0.model, "mixed_obs"):
#             n = tr0.model.psmsl_ids.data.size
#             if tr0.model.mixed_obs.shape.eval()[0] == n*2:
#                 print("Disentangle mixed obs into tidegauge_obs and satellite_obs")

#                 with tr0.model:
#                     pm.Deterministic("tidegauge_obs", tr0.model.mixed_obs[:n], dims="station")
#                     pm.Deterministic("satellite_obs", tr0.model.mixed_obs[n:], dims="station")
#             else:
#                 print("Unusual case of mixed obs, needs manual fix to disentangle.")

#         else:
#             print("Tidegauge and satellite obs already defined, nothing to do.")

#         if var_names is None:
#             var_names=["gps_obs", "tidegauge_obs", "satellite_obs"]

#         print("sample_posterior_predictive", cirun, var_names)

#         with tr0.model:
#             pp = pm.sample_posterior_predictive(tr.trace, var_names=var_names)
#     #     pp = tr.sample_posterior_predictive(var_names=var_names, redefine_model=True, skip_all_constraints=False)

#         print("Save to", fname)
#         pp.to_netcdf(fname)
#         return pp.posterior_predictive


#     def _get_kde_raw(yens):
#         if yens.ndim == 1: yens = yens[:, None]
#         return [arviz_kde(yens[:, i]) for i in tqdm.tqdm(list(range(0, yens.shape[1])))]


#     def _interp_kdes(bins, kdes_raw):
#         kdes = np.empty((len(kdes_raw), len(bins)))
#         for i, (grid, density) in enumerate(kdes_raw):
#             kdes[i] = np.interp(bins, grid, density)
#         return kdes.squeeze()


#     def _get_kde_raw_plot_data(cirun, var_names=None, **kw):

#         figdir = WEBDIR/cirun/'figures'

#         if var_names is not None:
#             fname = figdir / f"kde_plot_data_raw_{'_'.join(var_names)}.pickle"
#         else:
#             fname = figdir / "kde_plot_data_raw.pickle"

#         if fname.exists():
#             print("Load", fname)
#             return pickle.load(open(fname, "rb"))

#         else:

#             runfolder = CIDIR / cirun
#             trace = arviz.from_netcdf(runfolder/"trace.nc")
#             pp = sample_pp(cirun, var_names=var_names, trace=trace, **kw)

#             kde_raw = {}

#             for name in ["tidegauge", "gps", "satellite"]:
#                 if var_names and name+"_obs" not in var_names: continue
#                 print("compute KDEs for", name)
#                 post = arviz.extract(trace.posterior[[name]])[name].values
#                 post_pred = arviz.extract(pp[[name+'_obs']])[name+'_obs'].values
#                 obs = trace0.constant_data[name+'_mu'].values

#                 kde_raw[name] = {
#                     "posterior": _get_kde_raw(post),
#                     "posterior_predictive": _get_kde_raw(post_pred),
#                     "obs": _get_kde_raw(obs),
#                 }

#             print("Save to", fname)
#             pickle.dump(kde_raw, open(fname, "wb"))

#         return kde_raw


#     def get_kde_plot_data(cirun, **kw):

#         kde_raw = _get_kde_raw_plot_data(cirun, **kw)

#         plot_data = {}
#         bins = np.arange(-20, 20, .01)

#         for name in kde_raw:
#             plot_data[name] = {
#                 k : _interp_kdes(bins, kde_raw[name][k]) for k in kde_raw[name]
#             }
#             plot_data[name]['bins'] = bins

#         return plot_data

#     trace0 = tr0.trace
#     plot_data = {}

#     for name in ["default"] + list(skip_constraint):
#     #     print("Load plot data for", name)
#         if name == "default":
#             plot_data['default'] = get_kde_plot_data(DEFAULT, dataset_backend=True)
#         elif name == "all":
#             plot_data['all'] = get_kde_plot_data(skip_constraint["all"])
#         else:
#             plot_data[name] = get_kde_plot_data(skip_constraint[name], var_names=[name+'_obs'])
#     #         plot_data[name] = get_kde_plot_data(skip_constraint[name])


#     f, axes = plt.subplots(3, 1, figsize=(8, 10))

#     for letter, name, ax in zip("ABC", skip_constraint, axes):

#         edges = np.arange(-20, 20, .01)

#         def axhist(bins, kdes, label, histtype=None, alpha=.3, zorder=0, linestyle=None, **kw):
#             y, lo, hi = np.percentile(kdes, [50, 5, 95], axis=0)
#             l, = ax.plot(bins, y, linewidth=2, label=label, zorder=zorder, linestyle=linestyle, **kw)
#             ax.fill_between(bins, lo, hi, alpha=alpha, zorder=zorder-1, **kw)
#             return l

#         l = axhist(plot_data["default"][name]['bins'], plot_data["default"][name]['posterior_predictive'], label=f"default")
#     #     if name == "gps":
#         axhist(plot_data["default"][name]['bins'], plot_data["default"][name]['posterior'], label=f"default (posterior)", alpha=0, linestyle="--", color=l.get_color())
#         l2 = axhist(plot_data[name][name]['bins'], plot_data[name][name]['posterior_predictive'], label=f"no {name} constraint")
#         l3 = axhist(plot_data["all"][name]['bins'], plot_data["all"][name]['posterior_predictive'], label=f"no constraints")

#     #         axhist(plot_data[name][name]['bins'], plot_data[name][name]['posterior'], label=f"no {name} constraint (posterior)", alpha=0, linestyle="--", color=l2.get_color())
#     #         axhist(plot_data["all"][name]['bins'], plot_data["all"][name]['posterior'], label=f"no constraints (posterior)", alpha=0, linestyle="--", color=l3.get_color())

#     #     ax.plot(plot_data[name][name]['bins'], plot_data[name][name]['obs'], label=f"{name} obs", color="k", lw=2)
#         ax.plot(plot_data["default"][name]['bins'], plot_data["default"][name]['obs'], label=f"{name} obs", color="k", lw=2)

#         leg = ax.legend(title="($\mathbf{"+letter+"}$) "+name, loc="upper left")
#         leg._legend_box.align = "left"

#         model_names = {"gps": "VLM", "satellite": "GSL", "tidegauge": "RSL"}

#         ax.set_xlabel(f"({letter}) {model_names.get(name)} and {name} (mm/yr)")
#         ax.set_ylabel("Locations density")

#         xlims = {
#             "tidegauge": [-7, 10],
#             "gps": [-7, 7],
#             "satellite": [0, 7],
#         }
#         ax.set_xlim(xlims.get(name))

#     plt.tight_layout()

#     f.savefig(figdir/"tidegauge_constraints_validation_posterior_predictive_v2.png", dpi=300)

def makefig_tidegauge_constraints_validation_matrix(trace0, skip_constraint):

    CIDIR = Path(CONFIG['rundir'])

    # f, axes = plt.subplots(3, 3, sharey="row", sharex="row", figsize=(9.75, 9.75))
    f, axes = plt.subplots(1, 3, sharey="row", sharex="row", figsize=(9.75, 3.5), squeeze=False)

    # trace0 = arviz.from_netcdf(CIDIR / DEFAULT / "trace.nc")
    post0 = arviz.extract(trace0.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")
    traceX = arviz.from_netcdf(CIDIR / skip_constraint["all"] / "trace.nc")
    postX = arviz.extract(traceX.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")

    # traceX = arviz.from_netcdf(CIDIR / skip_constraint["all"] / "trace.nc")
    # postX = arviz.extract(traceX.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")

    # for name, axes_ in zip(skip_constraint, axes):
    for name, axes_ in zip(['tidegauge'], axes):
        trace = arviz.from_netcdf(CIDIR / skip_constraint[name] / "trace.nc")
        post = arviz.extract(trace.posterior[[name]]).median(dim="sample")
        psmsl_ids = trace.constant_data.psmsl_ids
        basins = psmsl_ids_to_basin(psmsl_ids)
        c = [basincolors[b] for b in basins]

        y0 = post0[name].values
        yX = postX[name].values
        y = post[name].values
        Y = trace0.constant_data[name+'_mu'].values

        def plot_ax(ax, x, y, ylabel="", title=""):


            for basin, group in groupby(sorted(zip(basins, y, x), key=lambda t:sort_basins.index(t[0])), key=lambda t:t[0]):
        #         YY = np.array([x for _, x in group])
                YY, OO = np.array([[x,o] for _, x, o in group]).T
                r2 = np.corrcoef(YY, OO)[0,1]**2
                ax.plot(OO, YY, '.', color=basincolors[basin], label=f"{basin} (r²={r2:.2f})", markersize=2)

            r2 = np.corrcoef(x, y)[0, 1]**2
    #         ax.plot(x, y, '.', color="black", label=f"All tide gauges (r²={r2:.2f})")

            ax.set_xlabel(f"{name} (mm / yr)")
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_xlim())
            ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', lw=.5)
            ax.legend(fontsize="xx-small", loc="lower right", title=f"All ocean basins (r²={r2:.2f})", title_fontsize='x-small')
    #         ax.legend(fontsize="xx-small", loc="lower right")
            ax.set_title(f"{title}")
            ax.set_aspect(1)


        model_names = {"gps": "VLM", "satellite": "GSL", "tidegauge": "RSL"}
        plot_ax(axes_[0], Y, y0, title="(A) default", ylabel=f"posterior {model_names.get(name)} (mm / yr)")
        plot_ax(axes_[1], Y, y, title=f"(B) no {name} constraint")
        plot_ax(axes_[2], Y, yX, title="(C) no local constraints")

    plt.tight_layout()
    return f, axes





# def make_unused_figures(tr0):

#     import json
#     import numpy as np
#     import arviz
#     from sealevelbayes.runslr import ExperimentTrace
#     from sealevelbayes.postproc.figures import BarPlot


#     f, axes = plt.subplots(1, 3, figsize=(9.75, 3.5))

#     # stations_js0 = json.load(open(WEBDIR/"default/figures"/"tidegauges_barplot.json"))
#     # stations_jsX = json.load(open(WEBDIR/skip_constraint["all"]/"figures"/"tidegauges_barplot.json"))
#     trace0 = arviz.from_netcdf(CIDIR / DEFAULT / "trace.nc")
#     post0 = arviz.extract(trace0.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")

#     traceX = arviz.from_netcdf(CIDIR / skip_constraint["all"] / "trace.nc")
#     postX = arviz.extract(traceX.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")
#     # json.load(open(WEBDIR/skip_constraint["all"]/"figures"/"tidegauges_barplot.json"))

#     for letter, name, ax in zip("ABC", skip_constraint, axes):
#         cirun = skip_constraint[name]
#         trace = arviz.from_netcdf(CIDIR / skip_constraint[name] / "trace.nc")
#         post = arviz.extract(trace.posterior[[name]]).median(dim="sample")

#         y0 = post0[name].values
#         yX = postX[name].values
#         y = post[name].values
#         Y = trace0.constant_data[name+'_mu'].values

#         ax.plot(y0, yX, '.', label=f"no local constraints", color="gray", alpha=0.3)
#         ax.plot(y0, y, '.', label=f"no {name}\nconstraint")
#         ax.plot(y0, Y, 'x', label="observations", alpha=.6, markersize=3, markeredgewidth=.5)
#         ax.set_xlabel("default model (mm / yr)")
#         ax.set_ylabel(f"test experiment or observations (mm / yr)")

#         ax.set_xlim(ax.get_xlim())
#         ax.set_ylim(ax.get_xlim())
#         ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', label="default")
#     #     if name == "tidegauge":
#         ax.legend(fontsize="x-small", loc="lower right")
#         ax.set_aspect(1)
#         ax.set_title(f"({letter}) {name}")
#     plt.tight_layout()
#     #     ax.set_xlim(xmin=0, xmax=bp.x.size)
#     f.savefig(figdir/"tidegauge_constraints_validation_vs.png", dpi=300)


#     import json
#     from itertools import groupby
#     import numpy as np
#     import arviz
#     from sealevelbayes.runslr import ExperimentTrace
#     from sealevelbayes.datasets.tidegaugeobs import psmsl_ids_to_basin
#     from sealevelbayes.postproc.colors import basincolors
#     from sealevelbayes.postproc.figures import BarPlot, sort_basins

#     f, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9.75, 6))

#     trace0 = arviz.from_netcdf(CIDIR / DEFAULT / "trace.nc")
#     post0 = arviz.extract(trace0.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")

#     traceX = arviz.from_netcdf(CIDIR / skip_constraint["all"] / "trace.nc")
#     postX = arviz.extract(traceX.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")

#     name = "tidegauge"

#     cirun = skip_constraint[name]
#     trace = arviz.from_netcdf(CIDIR / skip_constraint[name] / "trace.nc")
#     post = arviz.extract(trace.posterior[[name]]).median(dim="sample")
#     psmsl_ids = trace.constant_data.psmsl_ids
#     basins = psmsl_ids_to_basin(psmsl_ids)

#     y0 = post0[name].values
#     yX = postX[name].values
#     y = post[name].values
#     Y = trace0.constant_data[name+'_mu'].values

#     bins = np.arange(-15, 15, .5)

#     for i, (ax, yy, label, letter) in enumerate(zip(axes.ravel(),
#                             [Y, y0, y, yX],
#                             ["Tide-gauges observations",
#                             "Default experiment",
#                             "Only GPS and satellite",
#                             "No local constraint"], "ABCd")):

#         bottom = 0
#         for basin, group in groupby(sorted(zip(basins, yy, Y), key=lambda t:sort_basins.index(t[0])), key=lambda t:t[0]):
#     #         YY = np.array([x for _, x in group])
#             YY, OO = np.array([[x,o] for _, x, o in group]).T
#             r2 = np.corrcoef(YY, OO)[0,1]**2
#         #     ax.plot(xx, yy, '.', c=basincolors[basin], label=basin, markersize=2)
#             res = ax.hist(YY, bottom=bottom, color=basincolors[basin], label=f"{basin} (r²={r2:.2f})" if i!=0 else basin, bins=bins)
#             bottom += res[0]

#         r2 = np.corrcoef(Y, yy)[0,1]**2
#         suffix = f" (r²={r2:.2f})" if i !=0 else ""
#         ax.hist(Y, color="black", bins=bins, histtype='step', label='All observations'+suffix)
#         ax.set_title(f"({letter}) {label}")
#         ax.set_xlabel("Relative sea level (mm/yr)")
#         if np.mod(i, 2) == 0:
#             ax.set_ylabel("number of locations")
#         ax.set_xlim(bins[[0, -1]])
#         ax.legend(fontsize="x-small", loc="upper left")


#     # axes[0,0].legend(fontsize="x-small", loc="upper left")

#     plt.tight_layout()
#     f.savefig(figdir/"tidegauge_constraints_no_tidegauge.png", dpi=300)


#     import json
#     import numpy as np
#     import arviz
#     from scipy import stats
#     from sealevelbayes.runslr import ExperimentTrace
#     from sealevelbayes.postproc.figures import BarPlot

#     f, axes = plt.subplots(3, 1, figsize=(8, 10))

#     # stations_js0 = json.load(open(WEBDIR/"default/figures"/"tidegauges_barplot.json"))
#     # stations_jsX = json.load(open(WEBDIR/skip_constraint["all"]/"figures"/"tidegauges_barplot.json"))
#     trace0 = arviz.from_netcdf(CIDIR / DEFAULT / "trace.nc")
#     post0 = arviz.extract(trace0.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")

#     traceX = arviz.from_netcdf(CIDIR / skip_constraint["all"] / "trace.nc")
#     postX = arviz.extract(traceX.posterior[["tidegauge", "gps", "satellite"]]).median(dim="sample")
#     # json.load(open(WEBDIR/skip_constraint["all"]/"figures"/"tidegauges_barplot.json"))

#     for name, ax in zip(skip_constraint, axes):
#         cirun = skip_constraint[name]
#         trace = arviz.from_netcdf(CIDIR / skip_constraint[name] / "trace.nc")
#         post = arviz.extract(trace.posterior[[name]]).median(dim="sample")

#         y0 = post0[name].values
#         yX = postX[name].values
#         y = post[name].values
#         Y = trace0.constant_data[name+'_mu'].values

#         edges = np.arange(-20, 20, .01)

#         def axhist(y, label, bins, histtype=None, **kw):
#     #         ax.hist(y, label=label, bins=bins, **kw)
#             kde = stats.gaussian_kde(y)
#             if histtype is None:
#                 ax.fill_between(bins, bins-bins, kde(bins), label=label, **kw)
#             else:
#                 ax.plot(bins, kde(bins), label=label, **kw)

#         kw = dict()
#         axhist(Y, label="obs", color="k", alpha=0.3, bins=edges, **kw)
#         r2 = np.corrcoef(Y, y0)[0, 1]**2
#         axhist(y0, label=f"default (r²={r2:.2f})", histtype="step", bins=edges, **kw)
#         r2 = np.corrcoef(Y, y)[0, 1]**2
#         axhist(y, label=f"no {name} constraint (r²={r2:.2f})", histtype="step", bins=edges, **kw)
#         r2 = np.corrcoef(Y, yX)[0, 1]**2
#         axhist(yX, label=f"no constraints (r²={r2:.2f})", histtype="step", bins=edges, **kw)
#         ax.legend()
#         ax.set_xlabel(f"{model_names.get(name)} and {name} (mm/yr)")
#         ax.set_ylabel("Number of locations")

#         xlims = {
#             "tidegauge": [-7, 10],
#             "gps": [-7, 7],
#             "satellite": [0, 7],
#         }
#         ax.set_xlim(xlims.get(name))


#     plt.tight_layout()
#     plt.savefig(figdir/"tidegauge_constraints_validation_kde.png", dpi=300)



# def extract_proj(trace, varname):
# #     post = arviz.extract(trace_proj.posterior[varname].sel(experiment=SELECT_EXPERIMENTS_mu))[varname]
#     post = arviz.extract(trace.posterior[varname].sel(experiment=SELECT_EXPERIMENTS_mu+SSP_EXPERIMENTS_mu))[varname]

#     ds = post.to_dataset('experiment')
#     ds = ds.rename_vars({x: x[:-3] for x in ds})
#     for x in SELECT_EXPERIMENTS_mu:
#         ds[f"CurPol_minus_{x[:-3]}"] = post.sel(experiment='CurPol_mu') - post.sel(experiment=x)

#     ds = ds.quantile([.5, .05, .95], dim='sample')
#     return ds


fields = ["rsl", "global"]
diags = ['proj2050', 'proj2100', 'rate2100', 'rate2000', 'rate2050']


def crunch_one(cirun):

    rundir = get_runpath(cirun)
    ncname = rundir/"postproc/trace_sensitivity.nc"
    if not rundir.exists():
        logger.warning(f"Directory {rundir} does not exist. Skip")
        return
    tracepath  = rundir/"trace.nc"
    if not tracepath.exists():
        logger.warning(f"Trace file {tracepath} does not exist. Skip")
        return

    ncname.parent.mkdir(exist_ok=True)

    if ncname.exists():
        print(ncname, "already present, continue")

    else:
        print(ncname, "not found, crunch")
        try:
            tr = ExperimentTrace.load(cirun)
        except Exception as e:
            print("FAILED TO LOAD", cirun, e)
            return

        trace_proj = tr.resample_posterior( fields=fields,
                                                    diags=diags,
                                                    sources=['total'],
    #                                                 experiments=DEFAULT_EXPERIMENTS)
                                                    experiments=SELECT_EXPERIMENTS+SELECT_EXPERIMENTS_mu+SSP_EXPERIMENTS+SSP_EXPERIMENTS_mu)
        print("Write to", ncname)
        trace_proj.to_netcdf(ncname)


def load_ar6_global(percentiles=[50, 5, 95], diag="proj2100", experiment='ssp585'):
    if diag.startswith("rate"):
        quantity = "rates"
        name = "sea_level_change_rate"
    else:
        quantity = "values"
        name = "sea_level_change"
    year = int(diag[-4:])
    ds = open_slr_global("total", experiment, quantity=quantity)
    if year == 2000: year = ds[name].years[0].item()
    return ds[name].sel(quantiles=np.array(percentiles)/100, years=year).squeeze()


def load_ar6_local(psmsl_ids, percentiles=[50, 5, 95], diag="proj2100", experiment='ssp585'):
    if diag.startswith("rate"):
        quantity = "rates"
        name = "sea_level_change_rate"
    else:
        quantity = "values"
        name = "sea_level_change"
    year = int(diag[-4:])
    ds = open_slr_regional("total", experiment, quantity=quantity)
    if year == 2000: year = ds[name].years[0].item()
    return ds[name].sel(years=year, quantiles=np.array(percentiles)/100, locations=[id for id in psmsl_ids if id in ds.locations.values]).reindex(locations=psmsl_ids)



import pandas as pd
from sealevelbayes.postproc.figures import boxplot_custom, xlabels

DIAG = "proj2050"
DIAG = "proj2100"

def load_cirun_df(cirun, field="rsl", diag=DIAG):

    WEBDIR = Path(CONFIG['webdir'])

    figdir = WEBDIR/cirun/"figures"

    fname = figdir/f"table_{diag}_{field}.csv"
    df = pd.read_csv(fname)

    if field == "global":
        return df

    quantiles = df.iloc[0][1:]
    df = df.iloc[1:]
    df.set_index("variable", inplace=True)
    df.index.name = "psmsl_id"
    df.columns = [f'{c.split(".")[0]} {float(q)*100:.0f}th' for c, q in zip(df.columns, quantiles.values)]
    return df


def load_array(cirun, experiment, diag=DIAG, field="global", diff_experiment=None):

    # concatenate exp
    if type(cirun) is list:
        # hack: the "_unobserved" suffix indicates the need to mask unobserved obs
        runids, observed = zip(*((runid if not runid.endswith("_unobserved") else runid[:-11], runid.endswith("_unobserved")) for runid in cirun))
        all_arrays = [load_array(runid, experiment, diag=diag, field=field).assign_coords(subexperiment=runid) for runid in runids]
        # load the list of IDs used to constrain that exp and set to NaNs the locations in the list
        if "station" in all_arrays[0].dims:
            for k, (runid, unobsonly) in enumerate(zip(runids, observed)):
                if unobsonly:
                    ids = json.load(open(get_runpath(runid)/"postproc"/ "psmsl_ids.json"))["psmsl_ids"]
                    all_arrays[k].loc[dict(station=ids)] = np.nan
        concated = xa.concat(all_arrays, dim="subexperiment")
        return concated

    CIDIR = Path(CONFIG['rundir'])
    ncname = CIDIR/cirun/"postproc/trace_sensitivity.nc"
    vname = f"{diag}_{field}_total"
    a = arviz.extract(arviz.from_netcdf(ncname).posterior[[vname]].sel(experiment=experiment))[vname]

    if diff_experiment is not None:
        b = arviz.extract(arviz.from_netcdf(ncname).posterior[[vname]].sel(experiment=diff_experiment))[vname]
        z = a - b
    else:
        z = a

    return z


def _add_shading(ax, labels, lab1, lab2, color, label=None):
    i = labels.index(lab1)
    j = labels.index(lab2)
    x = list(range(1, len(labels)+1))
    ylim = ax.get_ylim()
    ax.fill_between([x[i]-.5, x[j]+.5], [ylim[0]]*2, [ylim[-1]]*2, alpha=0.1, label=label, color=color)


def add_shading(ax, labels, group_labels_specs):
#     i = ["global" in x for x in all_experiments].index(True)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
    indices = [labels.index(lab) for lab,_,_ in group_labels_specs]
    for i, (lab, _, colors) in enumerate(group_labels_specs):
        lab1 = labels[indices[i]]
        lab2 = labels[indices[i+1] - 1] if i < len(indices)-1 else lab1
        sublabels = [labels[k] for k in range(indices[i], (indices[i+1] if i < len(indices)-1 else indices[i]+1))]
        for c, sublab in zip(cycle(colors), sublabels):
            _add_shading(ax, labels, sublab, sublab, c)

def add_shading_labels(ax, labels, group_labels_specs):
    y1, y2 = ax.get_ylim()
    for name, label, _ in group_labels_specs:
        x = labels.index(name) + 1 - 0.5 + 0.1
        y = y2-(y2-y1)*0.15
        ax.text(x, y, label)


def plot_sens(psmsl_ids, all_experiments, values=None, global_values=None, relative=False, normed=False, normedrange=False,
              ar6_as_reference=False, ref=None, group_labels_specs=[],
              experiment="ssp585", field="rsl", diag=DIAG, diff_experiment=None, showmedian=True, showrange=True, ax=None,
              figsize=(8, 8), conf_intervals=[50, 90], test_interval=66, showar6=True, idx=None, title_prefix="", title_suffix="",
              showglobalnullhyptothesis=True, showshading=True, grid=True, color=None, globalcolor=None, globallabel=None,
              xoffset=0, width=0.25, rotation=30, horizontalalignment="right", fontsize='small', xtickkwargs={}, renamelabels={}):

    if diag in ("proj2100", "rate2100"):
        year = 2100

    elif diag in ("proj2050", "rate2050"):
        year = 2050

    elif diag in ("rate2000"):
        year = "1993-2018"

    else:
        raise NotImplementedError(diag)

    diagtype = diag[:4]
    if diagtype == "proj":
        scale = 1/10
        units = 'cm'
    else:
        scale = 1
        units = 'mm/yr'


    c1, c2 = conf_intervals
    c11, c22 = (100-c1)/2, (100-c2)/2
    all_pcts = [50, c11, 100-c11, c22, 100-c22]
    test_pct = (100-test_interval)/2

    if values is None:
        values = xa.concat([load_array(all_experiments[x], experiment=experiment, diff_experiment=diff_experiment, field=field, diag=diag) for x in tqdm.tqdm(all_experiments)],
                           dim="mainexperiment")
        # values = values.reshape(*values.shape[:-2], np.prod(values.shape[-2:]))

    if global_values is None:
        global_values = xa.concat([
            load_array(all_experiments[x], experiment, diff_experiment=diff_experiment, diag=diag)
                for x in tqdm.tqdm(all_experiments)],
            dim="mainexperiment")
        # global_values = global_values.reshape(*global_values.shape[:-2], np.prod(global_values.shape[-2:]))
#     global_pct = np.percentile(global_values, all_pcts, axis=0)

    values_ = values
    global_values_ = global_values

    values = values*scale
    global_values = global_values*scale

    if idx is not None:
        psmsl_ids_ = np.array(psmsl_ids)[idx].tolist()
        values = values.sel(station=psmsl_ids_)
    else:
        psmsl_ids_ = np.array(psmsl_ids)
        # psmsl_ids_ = np.array(psmsl_ids)[idx].tolist()


    labels = list(all_experiments)
    # medians = np.nanmedian(values, axis=-1)
    # rng = np.diff(np.nanpercentile(values, [test_pct, 100-test_pct], axis=-1), axis=0).squeeze()
    # global_median = np.nanmedian(global_values, axis=-1)
    # global_lo, global_hi = np.nanpercentile(global_values, [test_pct, 100-test_pct], axis=-1).squeeze()
    medians = values.median("sample").median("subexperiment").values
    rng = np.diff(values.quantile([test_pct/100, 1-test_pct/100], dim="sample").median("subexperiment").values, axis=0).squeeze()
    global_median = global_values.median("sample").median("subexperiment").values
    # global_lo, global_hi = np.nanpercentile(global_values, [test_pct, 100-test_pct], axis=-1).squeeze()
    global_lo, global_hi = global_values.quantile([test_pct/100, 1-test_pct/100], dim="sample").median("subexperiment").values.squeeze()
    global_rng = global_hi - global_lo

    if diff_experiment is not None:
        showar6 = False

    # append AR6
    if showar6:
        labels.append("IPCC AR6")
        try:
            ar6_mid_g, ar6_lo_g, ar6_hi_g = load_ar6_global(percentiles=[50, test_pct, 100-test_pct], diag=diag, experiment=experiment)*scale
        except Exception as error:
            logger.warning("FAILED TO LOAD AR6 GLOBAL SLR")
            logger.warning(str(error))
            ar6_mid_g, ar6_lo_g, ar6_hi_g = np.nan, np.nan, np.nan

        ar6_rng_g = ar6_hi_g - ar6_lo_g

        try:
            ar6_mid, ar6_lo, ar6_hi = load_ar6_local(psmsl_ids_, percentiles=[50, test_pct, 100-test_pct], diag=diag, experiment=experiment)*scale
            ar6_rng = ar6_hi - ar6_lo

        except Exception as error:
            logger.warning("FAILED TO LOAD AR6 LOCAL SLR")
            logger.warning(str(error))
            ar6_mid = medians + np.nan
            ar6_rng = rng + np.nan

        global_median = np.concatenate([global_median, [np.asarray(ar6_mid_g)]])
        global_rng = np.concatenate([global_rng, [ar6_rng_g]])
        medians = np.concatenate([medians, [np.asarray(ar6_mid)]])
        rng = np.concatenate([rng, [ar6_rng]])

    if showglobalnullhyptothesis:
        labels.append("Global-mean")

        global_median = np.concatenate([global_median, [global_median[0]]])
        global_rng = np.concatenate([global_rng, [global_rng[0]]])
        medians = np.concatenate([medians, [np.zeros_like(medians[0])+global_median[0]]])
        rng = np.concatenate([rng, [np.zeros_like(rng[0])+global_rng[0]]])

    def _format_axes(ax, showlabels=True):
        if grid: ax.grid()
        ax.set_xticks(ticks=list(range(1, 1+len(labels))),
                      labels=["$\mathbf{"+lab+"}$" if i == iref else lab for i, lab in enumerate(labels)],
                      rotation=rotation, horizontalalignment=horizontalalignment, fontsize=fontsize, **xtickkwargs)
        if showshading:
            add_shading(ax, labels, group_labels_specs)
            if showlabels: add_shading_labels(ax, labels, group_labels_specs)

        if renamelabels:
            ax.set_xticks(ticks=list(range(1, 1+len(labels))),
                        labels=["$\mathbf{"+lab+"}$" if i == iref else lab for i, lab in enumerate([renamelabels.get(lab, lab) for lab in labels])],
                        rotation=rotation, horizontalalignment=horizontalalignment, fontsize=fontsize, **xtickkwargs)

    if ax is None:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=figsize)
        ax0 = None
    else:
        ax1 = ax
        ax2 = ax
        ax0 = ax

    if ar6_as_reference:
        assert showar6
        global_median0 = ar6_mid_g.values
        global_lo0 = ar6_hi_g.values
        global_hi0 = ar6_lo_g.values
        global_rng0 = ar6_rng_g.values
        median0 = ar6_mid.values
        rng0 = ar6_rng.values
        iref = labels.index("IPCC AR6")
    else:
        iref = labels.index(ref or labels[0])
        global_median0 = global_median[iref]
        global_lo0 = global_lo[iref]
        global_hi0 = global_hi[iref]
        global_rng0 = global_rng[iref]
        median0 = medians[iref]
        rng0 = rng[iref]

    if relative:
        medians -= median0
        if normedrange:
            rng /= global_median[:, None]
            global_rng /= global_median

        else:
            # rng /= rng0
            # global_rng /= global_rng0
            rng -= rng0
            global_rng -= global_rng0

        global_median -= global_median0

        # if ax is None: ax2.set_ylim(ymax=2.5)


    elif normed:
        medians = (medians - median0) / global_median0 * 100
        rng = (rng - rng0) / global_median0 * 100
        global_median = (global_median - global_median0) / global_median0 * 100
        global_rng = (global_rng - global_rng0) / global_median0 * 100

        if ax is None: ax2.set_ylim([-100, 150])

    else:
        pass
        # if ax is None: ax2.set_ylim(ymax=200)

    if normed:
        units = f"% of {global_median0:.0f} cm"
    elif diag.startswith("rate"):
        units = "mm/yr"
    else:
        units = "cm"

    prefix = "$\Delta$ " if relative else ""
    exp_label = xlabels.get(experiment, experiment)

    boxplotargs = dict(xoffset=xoffset, w=width, color=color)

    ## Median values
    if showmedian:
        ax = ax1
        pcts = (np.nanpercentile if showar6 else np.percentile)(medians, all_pcts, axis=1)
        boxplot_custom(ax, pcts[0], pcts[1:3], pcts[3:5], **boxplotargs)

        # add global values as indicator
    #     global_pcts = np.percentile(global_median, all_pcts, axis=1)
    #     boxplot_custom(ax, global_median[0], global_pcts[1:3], global_pcts[3:5], color="tab:red", w=.1)
        ax.plot(1+np.arange(global_median.size) + xoffset, global_median, '.', color=globalcolor or color, label=globallabel)

        if relative or normed:
            ax.axhline(pcts[0,iref], lw=1, ls="--", color="black")

        ax.set_ylabel(f"{prefix}RSLR {year} ({units})")
        if diff_experiment is not None:
            exp_label += " minus "+xlabels.get(diff_experiment, diff_experiment)

        ax.set_title(f"{title_prefix}Sensitivity of median {exp_label} projection to {year}{title_suffix} ({global_median0:.0f} cm)")
        # plt.ylim(ymin=0)
        _format_axes(ax)

        ax.set_xlim([1-0.6, len(labels)+0.6 + xoffset])

    ## Range
    if showrange:
        ax = ax2
        pcts = (np.nanpercentile if showar6 else np.percentile)(rng, all_pcts, axis=1)

        boxplot_custom(ax, pcts[0], pcts[1:3], pcts[3:5], **boxplotargs)

        # also add global range
        ax.plot(1+np.arange(global_rng.size)+xoffset, global_rng, '.', color=globalcolor or color, label=globallabel)

        if not normedrange:
            ax.axhline(pcts[0,iref], lw=1, ls="--", color="black")

        if normed:
            suffix = f" ({units})"
        elif relative:
            if normedrange:
                suffix = " ratio to global mean median"
            else:
                # suffix = " ratio to default"
                suffix = " difference to default"
        else:
            suffix = f" ({units})"
    #     suffix = " ratio" if relative else f" ({units})"
        ax.set_ylabel(f"{prefix}RSLR {year} range{suffix}")
        ax.set_title(f"Sensitivity of {exp_label} projection 90% range to {year} ({global_lo0:.0f} cm to {global_hi0:.0f} cm)")

        _format_axes(ax, showlabels=False)

        ax.set_xlim([1-0.6, len(labels)+0.6+xoffset])

    if ax0 is None:
        plt.tight_layout()

    return {"values": values_, "global_values": global_values_}


def plot_sens_big(reordered_experiments, diags, relative=False, group_labels_specs=None, renamelabels=None, diaglabels={}, expdata=None, psmsl_ids=None):
    f, axes = plt.subplots(2, len(diags), sharex=False, sharey=False, figsize=(10, 9), squeeze=False)
    axes1, axes2 = axes

    if expdata is None:
        expdata = {}

    for ax in list(axes1)+list(axes2):
        ax.grid()
    # relative = False
    # width = 0.7
    # group_offset = len(show_exps) + 2
    # letters = 'ABCdef'
    # count = -1
    for i, diag in enumerate(diags):

# def make_sens_fig(relative, diags, expdata={}):
        ax1 = axes1[i]
        ax2 = axes2[i]

        diagtype = diag[:4]
        year = diag[4:]

        if diagtype == "rate":
            if relative:
                # ax2.set_ylim(0, 5)
                pass
            else:
                ax1.set_ylim(-0.5, 30)
                ax2.set_ylim(0, 40)
        elif diagtype == 'proj':
            if relative:
                # ax2.set_ylim(0, 1.5)
                if diag == 'proj2050':
                    # ax1.set_ylim(0, 40)
                    # ax2.set_ylim(-3.5, 17)
                    pass

                else:
                    # ax1.set_ylim(-30, 15)
                    # ax2.set_ylim(-15, 30)
                    ax1.set_ylim(-45, 20)
                    pass
            else:
                if diag == 'proj2050':
                    # ax1.set_ylim(0, 40)
                    # ax2.set_ylim(0, 40)
                    pass
                else:
                    # ax1.set_ylim(0, 120)
                    # ax2.set_ylim(0, 100)
                    # ax2.set_ylim(0, 100)

                    pass

        # diagtype = diags[0][:4]

        # for experiment in ["ssp585", "ssp126"]:
        for experiment in ["ssp585"]:
        # for experiment in ["ssp370"]:
            if diag == 'rate2000' and experiment == 'ssp126':
                continue

            width = 0.4
            # xoffset = 2*width if diag.endswith(("2050", "2100"))
            xoffset = 0

            # if year == '2000':
            #     xoffset = 0
            # else:
            #     xoffset = -width/2 if experiment == 'ssp126' else width/2

            if year == "2000":
                color = "black"
            else:
                color = xcolors.get(experiment)
            # color = {"ssp126": "gray"}.get(experiment, experiment)

        # for experiment in ["ssp585"]:
            # diag = "proj2100"
            # experiment = 'ssp585'

            test_interval = 90
            kwargs = dict(
                test_interval=test_interval,
                relative=relative,
                globallabel=xlabels.get(experiment, experiment) if year != "2000" else None,
                experiment=experiment, diag=diag,
                normedrange=False,
                # ref="default", showshading=False,
                # ref="default",
                ref="tidegauge + satellite + GPS",
                group_labels_specs=group_labels_specs,
                # shading_specs=shading_specs,
                showshading=True,
                showglobalnullhyptothesis=False,
                rotation=90, fontsize='small',
                xoffset=xoffset,
                grid=False,
                # width=width/2,
                width=width,
                color=color,
                renamelabels=renamelabels,
                # alpha=colors.get(experiment, experiment),
                **expdata.get((diag, experiment), {})
            )
            expdata[(diag, experiment)] = plot_sens(psmsl_ids, reordered_experiments, ax=ax1, showmedian=True, showrange=False, **kwargs);
            expdata[(diag, experiment)] = plot_sens(psmsl_ids, reordered_experiments, ax=ax2, showmedian=False, showrange=True, **kwargs);
            # proj26 = plot_sens(all_experiments, relative=True, **proj26);
            # plt.gca().set_ylim(0, 3.5)

            if i == 0:
                if diagtype == 'rate':
                    if relative:
                        ax1.set_ylabel("SLR rate (mm/yr)\ndifference to default")
                        ax2.set_ylabel(f"{test_interval}% range (mm/yr)\ndifference to default")
                        # ax2.set_ylabel(f"Ratio of local {test_interval}% range\nto global 90% range")
                    else:
                        # ax1.set_ylabel("SLR rate (mm/yr)")
                        # ax2.set_ylabel("Local 90% range (mm/yr)")
                        ax1.set_ylabel("Median (mm/yr)")
                        ax2.set_ylabel(f"{test_interval}% range (mm/yr)")
                else:
                    if relative:
                        ax1.set_ylabel("SLR (cm)\ndifference to default")
                        ax2.set_ylabel(f"{test_interval}% range (cm)\ndifference to default")
                        # ax2.set_ylabel(f"Ratio of local {test_interval}% range\nto global 90% range")
                    else:
                        ax1.set_ylabel("Median (cm)")
                        ax2.set_ylabel(f"{test_interval}% range (cm)")

            else:
                ax1.set_ylabel("")
                ax2.set_ylabel("")

            # ax2.set_ylim(0, 2)
            ax1.set_xticklabels(["" for _ in ax1.get_xticks()])
            # ax1.set_title("Sensitivity of modeleld SLR rate to constraints (present-day rate, 2050 and 2100)")
            ax1.set_title(diaglabels.get(diag, diag), fontweight='bold')
            ax2.set_title("")
    # ax2.set_ylabel(ymax=5)


    # ax1.set_xlim(1 - width * 1.5, len(show_exps) + 3*2.1*width)
    # ax1.set_ylim(1 - width * 1.5, len(show_exps) + 3*2.1*width)
        tx, ty = 0.02, 0.98
        tkwargs = dict(horizontalalignment="left", verticalalignment="top", fontweight='bold')

        if year == '2050':
            # ax1.legend(ncol=1, fontsize='small', loc='lower left' if relative else 'upper left')
            ax1.legend(ncol=1, fontsize='small', loc='lower left' if relative else 'upper right')

            ax1.text(tx, ty, f"(A)", transform=ax1.transAxes, **tkwargs)
            ax2.text(tx, ty, f"(B)", transform=ax2.transAxes, **tkwargs)
        else:
            ax1.text(tx, ty, f"(A)", transform=ax1.transAxes, **tkwargs)
            ax2.text(tx, ty, f"(B)", transform=ax2.transAxes, **tkwargs)

    # for ax2 in axes2:
    # save ticks and labels (same for all)

    ticks, labels = ax2.get_xticks(), ax2.get_xticklabels()
    labels = [lab._text for lab in labels]

    # shorten for tighter layout (cos(30) = 1/2)
    for ax2 in axes2:
        ax2.set_xticks(ticks, [lab[:15] for lab in labels], rotation=90)

    plt.tight_layout(rect=[0,0,1,1])

    for ax2 in axes2:
        ax2.set_xticks(ticks, labels, rotation=30)

    return expdata


from sealevelbayes.postproc.figures import resample_as_json, plot_locations, plot_locations_experiments, xt_plot_locations
from sealevelbayes.datasets.maptools import compute_geodesic_distances

# sources = ["vlm_res", 'gia', "landwater", "ais", "gis", "glacier", "steric", "total"]

def make_validation_fig(tr, post, sensposts, ids, getvar=None, sources=["total"], labels=None, experiment="CurPol"):

    if getvar is None:
        getvar = lambda x: x.change_rsl_total

    allids = tr.trace.posterior.station.values.tolist()
    distmat = compute_geodesic_distances(tr.trace.constant_data.lons.values, tr.trace.constant_data.lats.values)

    post = getvar(post)

    f, axes = plot_locations_experiments(post, labels=labels, experiments=[experiment], sources=sources)
    for j, ax in enumerate(axes.flat):
        if j >= len(ids):
            continue
        ax.axhline(y=0, color="black", lw=.5, ls="-")
        ax.axvline(x=2005, color="black", lw=.5, ls="-")
        id = ids[j]
        id_k = allids.index(id)
        collect = {}
        for i, (exp_ids, senspost) in enumerate(sensposts):
            v = getvar(senspost).sel(experiment=experiment, station=id).median(["draw", "chain"])/10
            if id not in exp_ids:
                id_ks = [allids.index(idd) for idd in exp_ids]
                dists = distmat[id_k, id_ks]
                nearest_loc = dists.min()
                collect.setdefault(nearest_loc, []).append(v)
                l1, = ax.plot(senspost.year, v, lw=.5, ls='--', color="black", zorder=100)

                # ax.fill_between(senspost.year, *senspost.change_rsl_total.sel(experiment=experiment, station=id).quantile([.05, .95], ["draw", "chain"])/10, color=l1.get_color(), alpha=.2)
            else:
                l2, = ax.plot(senspost.year, v, lw=.5, color="darkgray")
        l1.set_label("50% Experiment\nLocation excluded")
        l2.set_label("50% Experiment\nLocation included")

        ax.plot([], [], color="darkgray", label="Distance to nearest\nobservation")
        handles, labels_ = ax.get_legend_handles_labels()

        from matplotlib.transforms import blended_transform_factory
        texttransform = blended_transform_factory(ax.transData, ax.transAxes)
        # last = None
        # minh = 15
        for tt, (nearestdist, vv) in enumerate(sorted(collect.items(), key=lambda x: x[1][0][0])):
            for v in vv:
                ax.annotate(f"{int(nearestdist)} km", xy=(1900, v[0].item()),
                            # xytext=(1901, 0.07*tt + 0.02),
                            xytext=(1901 + tt*40, 0.02),
                            ha="left", fontsize=6, arrowprops=dict(facecolor='gray', edgecolor="gray", arrowstyle='-'),
                            textcoords=texttransform, xycoords='data', color="gray", label="Distance to nearest observation" if tt==0 else "")

    ax.legend(handles, labels_, frameon=False)

    return f, axes


def get_sensdata(xname, save=True, load=True, experiments=["CurPol"], fields=None,
                 diags=["change"],
                 sources=["steric", "glacier", "gis", "ais", "landwater", "vlm", "gia", "vlm_res", "total"], filename=None, featured_ids=None):

    if fields is None:
        fields = ['rsl', 'global']
        fieldstag = ""
    else:
        fieldstag = "_" + "-".join(fields) + "_v2"

    if featured_ids is None:
        featured_ids, _ = get_featured_locations()

    xpath = get_runpath(xname) / "postproc" / (filename or f"sens_featured_{'-'.join(experiments)}_{len(sources)}sources{fieldstag}.nc")
    ids_path = get_runpath(xname) / "postproc" / "psmsl_ids.json"

    tr_ = None

    # if load and os.path.exists(xpath) and os.path.exists(ids_path):
    #     post = xa.open_dataset(xpath)
    #     ids = json.load(open(ids_path, "r"))["psmsl_ids"]
    #     return ids, post

    if load and os.path.exists(ids_path):
        ids = json.load(open(ids_path, "r"))["psmsl_ids"]
    else:
        tr_ = ExperimentTrace.load(xname)
        ids = [int(name[len("tidegauge_"):]) for name in tr_.trace.observed_data.mixed_obs.station_mixed.values.tolist() if name.startswith("tidegauge_")]
        if save:
            json.dump({"psmsl_ids": ids}, open(ids_path, "w"))

    if load and os.path.exists(xpath):
        post = xa.open_dataset(xpath)
    else:
        if tr_ is None:
            tr_ = ExperimentTrace.load(xname)
        print(xname)
        post = tr_.resample_posterior(diags=diags, experiments=experiments, fields=fields, sources=sources, psmsl_ids=featured_ids).posterior
        if save:
            post.to_netcdf(xpath)

    return ids, post.sel(station=featured_ids)



