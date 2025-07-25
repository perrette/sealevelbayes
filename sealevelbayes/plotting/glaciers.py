import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xa
import pandas as pd
mpl.rcParams['hatch.linewidth'] = 0.5  # affects everything (imported from a notebook...) -- would be cleaner in a local context but OK for now

from sealevelbayes.logs import logger
from sealevelbayes.datasets.glaciers import load_zemp, get_volume_ranges, load_glacier_datasets, get_sle_mm21, gt_to_mm_sle, load_merged_timeseries
import sealevelbayes.datasets.frederikse2020 as frederikse2020
from sealevelbayes.datasets.ar6.supp import load_global as load_ar6_global
from sealevelbayes.datasets.glaciers import get_gmip_glacier_regions, RCP_SCENARIOS, M20_SCENARIOS, SSP_SCENARIOS
from sealevelbayes.postproc.colors import xcolors

EXPERIMENTS_TO_RCP = dict(zip(SSP_SCENARIOS+[s + "_mu" for s in SSP_SCENARIOS], RCP_SCENARIOS*2))
# RCP_SCENARIOS = RCP_SCENARIOS + SSP_SCENARIOS + [s + "_mu" for s in SSP_SCENARIOS]
# M20_SCENARIOS = M20_SCENARIOS*3

gmip_regions = get_gmip_glacier_regions()
gmip_regions_dict = get_gmip_glacier_regions(as_dict=True)

ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

def setup_axes(n=None, ni=5, nj=4, figsize=(12, 10), sharex=True, ylabel="Sea level rise (mm)", regions=None, region_names=None, **kw):

    if regions is None:
        regions = list(range(1, 19+1))

    if region_names is None:
        region_names = [gmip_regions_dict.get(r) for r in regions]

    if n is None:
        n = len(regions)

    f, axes = plt.subplots(ni, nj, figsize=figsize, sharex=sharex, squeeze=False, **kw)

    for i, ax in enumerate(axes.flat):
        region_id = regions[i] if i < n else None
        # if i == 19:
        #     ax.text(0.05, 0.9, f"World", transform=ax.transAxes)
        if region_id is None:
            ax.text(0.05, 0.9, f"Total", transform=ax.transAxes)
        else:
            ax.text(0.05, 0.9, f"{region_id}. {region_names[i]}", transform=ax.transAxes)
        if ylabel and i%nj == 0:
            ax.set_ylabel(ylabel)
        ax.grid()
        ax.axhline(0, color='k', lw=0.5, ls='--')

    return f, axes

def plot_glacier_samples(data, axes=None, filled=False, axes_kwargs={}, sample_dim='Forcing', time_dim='Time', region_dim='Region', **kw):

    nr = data[region_dim].size
    years = data[time_dim].values

    if axes is None:
        _, axes = setup_axes(nr, **axes_kwargs)

    for i in range(nr + 1):
        ax = axes.flat[i]
        if i == data.Region.size:
            datasel = data.sum(region_dim)
        else:
            datasel = data.isel({region_dim:i})
        if filled:
            # ax.fill_between(years, datasel.min(dim=sample_dim), datasel.max(dim=sample_dim), **kw)
            mu, lo, hi = datasel.quantile([.5, .05, .95], dim=sample_dim)
            # mu = datasel.mean(sample_dim) # more robust
            ax.fill_between(years, lo, hi, **kw)
            ax.plot(years, mu, color=kw.get('color', kw.get('c')), lw=0.5)
        else:
            ax.plot(years, datasel.values.T, **kw)

    return axes


def ref(yens, dim='year'):
    return yens - yens.sel({dim:slice(1995, 2014)}).mean(dim)

def add_fred_data(ax, ds=None):
    if ds is None:
        ds = xa.load_dataset(frederikse2020.root/"GMSL_ensembles.nc")

    dat = ref(ds["Glaciers"], "time").quantile([.5, .05, .95], dim="likelihood")
    ax.plot(dat.time, dat[0], c="gray", label='Frederikse')
    ax.plot(dat.time, dat[1:].T, c="gray", ls="--", lw=1)


def add_ipcc_data(ax, hatch='...'):
    ar6_ssp585 = load_ar6_global("ssp585").loc[:2100]
    ar6_ssp585_lo = load_ar6_global("ssp585", 0.05).loc[:2100]
    ar6_ssp585_hi = load_ar6_global("ssp585", 0.95).loc[:2100]
    # ar6_ssp370 = load_ar6_global("ssp370").loc[:2100]
    # ar6_ssp245 = load_ar6_global("ssp245").loc[:2100]
    ar6_ssp126 = load_ar6_global("ssp126").loc[:2100]
    ar6_ssp126_lo = load_ar6_global("ssp126", 0.05).loc[:2100]
    ar6_ssp126_hi = load_ar6_global("ssp126", 0.95).loc[:2100]

    ax.plot(ar6_ssp585.index, ar6_ssp585['glaciers'], '.', c="tab:red", label='AR6 SSP5-8.5')
    ax.fill_between(ar6_ssp585.index, ar6_ssp585_lo['glaciers'], ar6_ssp585_hi['glaciers'],
                     hatch=hatch, edgecolor='tab:red', facecolor="none", linewidth=0.5, linestyle=":")
    ax.plot(ar6_ssp126.index, ar6_ssp126['glaciers'], '.', c="tab:green", label='AR6 SSP1-2.6')
    ax.fill_between(ar6_ssp126.index, ar6_ssp126_lo['glaciers'], ar6_ssp126_hi['glaciers'],
                     hatch=hatch, edgecolor='tab:green', facecolor="none", linewidth=0.5, linestyle=":")


def add_volume_ranges(ax, region_number, w=3, color="black"):
        # ipcc
    ranges = get_volume_ranges(region_number)

    for k, (source, (sle, sle_sd)) in enumerate(ranges.items()):
        l, = ax.plot(2000 + k*w, sle, '.', lw=1, color=color, label=source)
        ax.bar(2000 + k*w, 2*sle_sd*1.64, bottom=sle - sle_sd*1.64, width=0.8*w, color=l.get_color(), alpha=0.3)
    #     ax.errorbar(2000, 2*sle_sd*1.64/gt_to_mm_sle, bottom=(sle - sle_sd*1.64)/gt_to_mm_sle, width=5, color='black', alpha=0.3)

    return ranges


def plot_glaciers_data(axes=None, regions=None, rate=False, slr=False, gt=False, experiments=None, hatch="...", linestyle="-", linewidth=None, color=None,
                       past_gmip_show_individual_forcing=False,
                       past_gmip_show_ensemble=True,
                       show_zemp=True,
                       show_past_gmip=True,
                       show_future_gmip=True,
                         **data_kwargs):
    print("regions", regions)
    if regions is None:
        regions = np.arange(1, 19+1)

    if experiments is None:
        experiments = ["rcp26", "rcp85"]

    # map to RCP
    experiments = [EXPERIMENTS_TO_RCP.get(e, e) for e in experiments]

    variable, transform, ylabel = {
        # (False, True):["mass", lambda x: x.sel(Time=[2005]) - x, "Sea level rise (mm)"],
        (False, True):["mass", lambda x: x.sel(Time=2005) - x, "Sea level rise (mm)"],
        (True, True):["rate", lambda x: -x, "Sea level rise (mm/yr)"],
        (False, False):["mass", None, "Glacier volume (mm sle)"],
        (True, False):["rate", None, "Glacier melt rate (mm sle/yr)"],
        }.get((rate, slr))

    if transform is None: transform = lambda x: x

    # all_regions = np.arange(1, 19+1)
    all_regions = regions

    if axes is None:
        print("setup axes", all_regions)
        nn = np.sqrt(len(all_regions))
        nj = int(nn)
        ni = int(np.ceil(len(all_regions)/nj))
        f, axes = setup_axes(ni=ni, nj=nj, figsize=(max([12*ni/5, 8]), max([10*nj/4, 6])), sharex=True, ylabel=ylabel, regions=all_regions)
        # f, axes = plt.subplots(5, 4, sharex=True, figsize=(9.75, 9))

    if gt:
        transform = lambda x: transform(x) / gt_to_mm_sle

    data_kwargs.setdefault("normalize_future", True)
    data_kwargs.setdefault("fill_nans", False)
    data_kwargs.update(dict(stack_sample_dims=False, return_gsat=False))

    show_past_gmip_ = show_past_gmip and past_gmip_show_ensemble
    show_indiv_gmip = show_past_gmip and past_gmip_show_individual_forcing

    if show_past_gmip_ or show_future_gmip:
        timeseries_gmip = load_merged_timeseries(past_source="mm21", future_source="m20", **data_kwargs).sel(Region=regions)
        timeseries_gmip = transform(timeseries_gmip[variable])
        timeseries_gmip_past = timeseries_gmip.sel(Time=slice(None, 2018)).isel(Scenario=0, Glacier_Model=0, Climate_Model=0)
        timeseries_gmip_future = timeseries_gmip.sel(Time=slice(2018, 2100)).isel(Forcing=0)

    if show_zemp:
        data_kwargs.setdefault("zemp_kwargs", {}).setdefault("resample", True)
        data_kwargs.setdefault("zemp_kwargs", {}).setdefault("deterministic", True)
        timeseries_zemp = load_merged_timeseries(past_source="zemp2019", future_source=None, **data_kwargs).sel(Region=regions)
        timeseries_zemp = transform(timeseries_zemp[variable])

    if show_indiv_gmip:
        from sealevelbayes.datasets.glaciers import load_mm21_constraints
        mm21 = load_mm21_constraints(rate=rate, Forcing=data_kwargs.get("mm21_forcing"), slr_eq=False) * gt_to_mm_sle
        mm21 = transform(mm21)

    kw = dict(linestyle=linestyle, linewidth=linewidth, color=color)

    def _plot(ax, data, region, sample_dim, **kw):
        if region is None or region == 20:
            data = data.sum("Region", skipna=False)
        else:
            data = data.sel(Region=region)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = data.mean(sample_dim)
            sd = data.std(sample_dim)
        l, = ax.plot(data.Time, mean, **kw)
        ax.fill_between(data.Time, mean - 1.64*sd, mean + 1.64*sd, edgecolor=l.get_color(), facecolor="none", linewidth=0.5, linestyle=":", hatch=hatch)

    for i, ax in enumerate(axes.flat):
        region = all_regions[i] if i < len(all_regions) else None
        if region is not None and region not in regions:
            ax.remove()
            continue

        if region != 19:

            if show_past_gmip_:
                _plot(ax, timeseries_gmip_past, region, "Forcing", **kw)

            if show_indiv_gmip:
                for forc in mm21.Forcing.values:
                    mm21_ = mm21.sel(Forcing=forc, Region=region)
                    obs = mm21_["obs"]
                    obs_sd = mm21_["obs_sd"]
                    l, = ax.plot(mm21.Time, obs, label=f"{forc}", color="tab:blue", lw=1, ls='--')
                    ax.fill_between(mm21.Time, obs - 1.64*obs_sd, obs + 1.64*obs_sd, edgecolor=l.get_color(), alpha=0.3)

        if show_future_gmip:
            for experiment in (experiments):
                scenario = dict(zip(RCP_SCENARIOS, M20_SCENARIOS))[experiment]
                _plot(ax, timeseries_gmip_future.sel(Scenario=scenario), region, ["Glacier_Model", "Climate_Model"],
                    **{**kw, "color":xcolors.get(experiment)})

        if show_zemp:
            _plot(ax, timeseries_zemp, region, "Forcing", **{**kw, "color":"black"})

        # glacier volume?
        if not rate and not slr:
            mi, ma = ax.get_ylim()
            ax.set_ylim(0, ma)

    f.tight_layout()

    return f, axes


def plot_glacier_volume(V, trace=None, ax=None, slr=False, experiments=['rcp26', 'rcp85'], linestyle='-'):
    if ax is None:
        f, ax = plt.subplots(1,1)

    if trace is not None:
        div = trace.sample_stats.diverging.sum().item()
    else:
        div = 0

    offset = 0.03
    vstep = 0.06
    if div > 0:
        ax.text(0.03, offset, f"divergences: {div}", transform=ax.transAxes, horizontalalignment="left", verticalalignment="bottom", fontsize='x-small')
        offset += vstep

    if trace is not None:
        if "n" in trace.constant_data:
            n = trace.constant_data["n"].item()
            ax.text(0.03, offset, f"n={n:.2f}", transform=ax.transAxes, horizontalalignment="left", verticalalignment="bottom", fontsize='x-small')
        else:
            n = trace.posterior["n"].mean().item()
            n_sd = trace.posterior["n"].std().item()
            ax.text(0.03, offset, f"n={n:.2f}±{n_sd:.2f}", transform=ax.transAxes, horizontalalignment="left", verticalalignment="bottom", fontsize='x-small')

    if slr:
        y = V.sel(year=slice(1995, 2014)).mean('year') - V
    else:
        y = V

    mid, lo, hi = y.quantile([.5, .05, .95], dim=["chain", "draw"])

    for x in experiments:
        l, = ax.plot(V.year, mid.sel(experiment=x), color=xcolors.get(x), linestyle=linestyle)
        ax.plot(V.year, mid.sel(experiment=x), color="black", linestyle=linestyle, linewidth=0.5)
        ax.fill_between(V.year, lo.sel(experiment=x), hi.sel(experiment=x), color=l.get_color(), alpha=0.3)


def plot_glacier_slr(V_total, ax=None):
#     med, lo, hi = (V_total.sel(year=2005) - V_total).quantile([0.5, 0.05, 0.95], dim='sample')
    candidates = ['sample', 'chain', 'draw']
    med, lo, hi = (V_total.sel(year=2005) - V_total).quantile([0.5, 0.05, 0.95], dim=[d for d in candidates if d in V_total.coords])

    if ax is None:
        f, ax = plt.subplots(1,1)

    for x in V_total.experiment.values:
        l, = ax.plot(V_total.year, med.sel(experiment=x), c=xcolors.get(x), label=x)
        ax.fill_between(V_total.year, lo.sel(experiment=x), hi.sel(experiment=x), color=l.get_color(), alpha=0.3)

    add_fred_data(ax)
    add_ipcc_data(ax, hatch='')

    plot_glaciers_data([20], slr=True, rate=False, include_mass_uncertainty=False, hatch='...', axes=[ax], linestyle='--')

    ax.legend(loc='upper left')
#     ax.grid()

    ax.set_ylabel("Sea level equivalent (mm)")
    ax.set_xlabel("Year")
    ax.set_title("Sum of all glaciers")


# f, axes = plot_glaciers_data(gt=False)


### Experimental figs for issue

def plot_lines(ax, x, arr, c):
    a = arr.values.reshape(arr.shape[0]*arr.shape[1], -1)
    return ax.plot(x, a, lw=0.5, c='k', alpha=0.01);

def plot_quantiles(ax, x, arr, c):
    mid, lo, hi = arr.quantile([.5, .05, .95], dim=('chain', 'draw')).values
    l, = ax.plot(x, mid, c=c);
    h = ax.fill_between(x, lo, hi, color=c, alpha=0.3);
    return l, h


def plot_rate_vs_cumul(obs, prior, posterior, plotf=plot_quantiles, title=""):
    """
    obs: pd.DataFrame of observations (sea-level rate)
    prior: xa.DataArray of prior samples (trace.prior.glacier_sea_level_rate)
    posterior: xa.DataArray of prior samples (trace.prior.glacier_sea_level_rate)
    plotf: plotting funciton (plot_quantiles or plot_lines)
    """
    n = prior.shape[-1]

    f, axes = plt.subplots(n, 2, sharex=True, figsize=(10, 2*n))
    dim = prior.dims[-2]
    plotf_ = lambda ax, arr, c: plotf(ax, obs.index, arr, c)

    for i in range(n):

        ax1, ax2 = axes[i]

        plotf_(ax1, prior[..., i], 'k')
        plotf_(ax1, posterior[..., i], 'tab:blue')
        ax1.plot(obs.index, obs.iloc[:, i], lw=1, c='r');

        plotf_(ax2, prior[..., i].cumsum(axis=-1), 'k')
        plotf_(ax2, posterior[..., i].cumsum(axis=-1), 'tab:blue')
        ax2.plot(obs.index, obs.iloc[:, i].cumsum(), lw=1, c='r');

        ax1.text(0.05, 0.9, f"{i+1}. {gmip_regions['IceName'].iloc[i]}", transform=ax1.transAxes)

    axes[0, 0].set_title("rate of SLR")
    axes[0, 1].set_title("cumulative SLR")
    axes[1, 0].set_ylabel("mm/yr")
    axes[1, 1].set_ylabel("mm")
    if title:
        plt.suptitle(title)
    plt.tight_layout()


def make_fig(dataset, rate=False, slr=True, experiments=None, regions=None, title="", **data_kwargs):

    if regions is None:
        regions = dataset.glacier_region.values.tolist()
    if len(regions) > 1:
        all_regions = regions + [None]
    else:
        all_regions = regions

    f, axes = plot_glaciers_data(
        linestyle='--', experiments=experiments,
        regions=regions,
        # include_mass_uncertainty=True,
        slr=slr, rate=rate, **data_kwargs)

    if (rate, slr) == (True, True):
        data = dataset["glacier_sea_level_rate"]

    elif (rate, slr) == (True, False):
        data = -dataset["glacier_sea_level_rate"]

    elif (rate, slr) == (False, True):
        data = dataset["glacier_volume"].sel(year=2005) - dataset["glacier_volume"]

    elif (rate, slr) == (False, False):
        data = dataset["glacier_volume"]

    else:
        raise ValueError(f"invalid combination of rate and slr: {rate}, {slr}")

    for i, region in enumerate(all_regions):

        if region is not None and region not in regions:
            continue
        ax = axes.flatten()[i]

        if region is not None:
            V = data.sel(glacier_region=region)
        else:
            V = data.sel(glacier_region=regions).sum('glacier_region')

        # plot_glacier_volume(V.rename({"glacier_experiment": "experiment"}), ax=ax, slr=True, experiments=sub_experiments, linestyle='-')
        plot_glacier_volume(V, ax=ax, slr=False, linestyle='-', experiments=experiments)

    if title:
        f.suptitle(title)

    f.tight_layout()

    return f, axes


from sealevelbayes.datasets.glaciers import load_zemp_with_forcing_dimension

def plot_resampled(samples, cumul=False):

    # ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    obsmm21 = load_merged_timeseries(past_source="mm21", future_source=None, zemp_kwargs={"resample": True})
    # obszemp = load_merged_timeseries(past_source="zemp2019", future_source=None, zemp_kwargs={"resample": True, "sample_size": 5000})["mass"]
    obszemp_rate = load_zemp_with_forcing_dimension(**{"resample": True, "sample_size": 1000})

    if cumul:
        obszemp = obszemp_rate.cumsum("Time")
        obszemp = obszemp.sel(Time=2005) - obszemp
        slr = samples.cumsum("Time")
        slr = slr.sel(Time=2005) - slr
        obsmm21 = obsmm21["mass"]
        obsmm21 = obsmm21.sel(Time=2005) - obsmm21
        title = "Cumulative glacier melt since 2005 [ mm ]"
    else:
        obszemp = obszemp_rate
        slr = samples
        obsmm21 = obsmm21["rate"]
        title = "Glacier meltrate [ mm/yr ]"


    axes = plot_glacier_samples(slr, sample_dim="Sample", filled=True, alpha=0.3, label="Resampled")

    # ADD GIMP
    # mass_1_18 = gmip_past["Mass"].assign_coords({"Time": np.arange(1901, 2018+1)})*gt_to_mm_sle
    for i in range(20):
    # for i, region in enumerate(gmip_past.Region.values):
        if i == 18:
            continue

        if i == 19:
            region = "World"
            slr = obsmm21.sum("Region")
        else:
            slr = obsmm21.isel(Region=i)

        ax = axes.flat[i]
        # slr = mass_1_18.sel(Region=region)
        mu, lo, hi = slr.quantile([.5, 0.05, 0.95], dim="Forcing")
        l, = ax.plot(slr.Time, mu, color="darkgreen", label="GMIP")
        ax.fill_between(slr.Time, lo, hi, color=l.get_color(), alpha=0.3)

    # ADD ZEMP
    for i in range(20):
        region = i+1
        # zemp_samples = _load_zemp_with_forcing_dimension(region, random_seed=25, resample=True, sample_size=5000).cumsum("Time")
        # slr = zemp_samples.cumsum("Time")
        if region == 20:
            slr = obszemp.sum("Region")
        else:
            slr = obszemp.sel(Region=region)
        ax = axes.flat[i]
        mu, lo, hi = slr.quantile([.5, 0.05, 0.95], dim="Forcing")
        l, = ax.plot(slr.Time, mu, color="darkred", label="Zemp")
        ax.fill_between(slr.Time, lo, hi, color=l.get_color(), alpha=0.3)

    for ax in axes.flat:
        ax.legend(loc="lower right", fontsize="small")

    f = axes.flat[0].figure
    f.suptitle(title)
    f.tight_layout()

    return f, axes


def plot_cov(samples, title=None, cumul=False):
    if cumul:
        samples = samples.cumsum("Time")
        samples = samples.sel(Time=2005) - samples
        title="Error Covariance Matrix of the cumulate glacier melt since 2005 [ mm^2 ]"
    else:
        title="Error Covariance Matrix of the glacier meltrate [ (mm/yr)^2 ]"

    f, axes = setup_axes(19, ni=5, nj=4, figsize=(12, 10), sharex=True, ylabel="Year", sharey=True)
    for i, ax in enumerate(axes.flat):
        if i == 19:
            dat = samples.sum("Region")
        else:
            dat = samples.sel(Region=i+1)
        # ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="k", linestyle="--", alpha=0.5)
        years = dat.Time.values
        ax.set_xlim(years[0], years[-1])
        ax.set_ylim(years[0], years[-1])
        cov = np.cov(dat.values.T)
        mag = np.abs(cov).max()
        vmin, vmax = -mag, mag
        im = ax.pcolormesh(years, years, cov, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        # im = ax.pcolormesh(years, years, cov)
        plt.colorbar(im, ax=ax)
        # ax.legend([], [], title=f"Region {i+1}")
    if title:
        f.suptitle(title)
    f.tight_layout()
    return f, axes



from sealevelbayes.plotting.glaciers import plot_glaciers_data, plot_glacier_volume, make_fig
from sealevelbayes.postproc.run import ExperimentTrace, get_webpath

def make_glacier_figs(tr : ExperimentTrace, dataset=None, group="posterior", label="",
                      diags=["slr", "rate", "volume"],
                      save=True, figdir=None, experiments=None, **kwargs):
    if save:
        if figdir is None:
            figdir = get_webpath(tr.o.cirun) / "figures"
        figdir.mkdir(exist_ok=True, parents=True)

    if dataset is None:
        dataset = tr.trace[group]

    if not label:
        label = group

    if experiments is None:
        experiments = dataset.experiment.values.tolist()

    print(f"Make glacier ({label}) figures and save to", figdir)

    for diag in diags:
        if diag == "slr":
            f, axes = make_fig(dataset, rate=False, slr=True, experiments=experiments,
                            title=f"Glaciers ({label})", **kwargs)
            if save: f.savefig(figdir/f'glaciers_{label}.png', dpi=300)

        elif diag == "rate":
            f, axes = make_fig(dataset, rate=True, slr=True, experiments=experiments,
                            title=f"Glaciers rate ({label})", **kwargs)
            if save: f.savefig(figdir/f'glaciers_{label}_rate.png', dpi=300)

        elif diag == "volume":
            f, axes = make_fig(dataset, rate=False, slr=False, experiments=experiments,
                            title=f"Glaciers volume ({label})", **kwargs)
            if save: f.savefig(figdir/f'glaciers_{label}_volume.png', dpi=300)

        else:
            raise ValueError(f"Unknown glacier diag: {diag}. Valid options are: slr, rate, volume")

    return dataset