import itertools
import tqdm
from itertools import groupby, product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xa
import arviz
from scipy.stats import norm

from sealevelbayes.datasets.config import logger
import sealevelbayes.datasets.frederikse2020 as frederikse2020
import sealevelbayes.datasets.ar6.supp
import sealevelbayes.datasets.ar6 as ar6
from sealevelbayes.datasets.shared import MAP_FRED, MAP_AR6
from sealevelbayes.datasets.tidegaugeobs import get_stations_from_psmsl_ids
from sealevelbayes.models.localslr import SOURCES, FIELDS, DIAGS


def scattermatrix(post, pnames, like=None, prior=None, **kwargs):
    plt.rcParams['axes.labelsize'] = 'medium'
    axes = pd.plotting.scatter_matrix(pd.DataFrame({k.replace("_","\n"):post[k]
                                             for k in pnames }
                                           ), diagonal="kde", **kwargs);

    if like or prior:
        for i in range(axes.shape[0]):
            ax = axes[i,i]
            ylim = ax.get_ylim()
            name = ax.get_xlabel().replace('\n','_')

            x = np.linspace(*list(ax.get_xlim())+[100])

            if like is not None and like.get(name) is not None:
                ax.plot(x, like[name].pdf(x), color="tab:red", linestyle="--")
            if prior is not None and prior.get(name) is not None:
                ax.plot(x, prior[name].pdf(x), color="tab:green", linestyle="--")

    plt.rcParams['axes.labelsize'] = plt.rcParamsDefault["axes.labelsize"]


def align(years, series):
    return series - series[(years>=1995) & ((years<=2014))].mean(axis=0)


def plot_ensemble(years, ensemble, conf=0.9, ax=None, color=None, label=None, alpha=0.5, **kwargs):

    if ax is None:
        ax = plt.gca()


    data = align(years, np.cumsum(ensemble.T, axis=0))

    if data.ndim > 1:
        pct = 100*np.array([0.5, (1-conf)/2, 1-(1-conf)/2])
        med, lo, hi = np.percentile(data, pct, axis=1)
    else:
        med = lo = hi = data

    l, = ax.plot(years, med, color=color, label=label, **kwargs);
    patch = ax.fill_between(years, lo, hi, color=l.get_color(), alpha=alpha);

    return {"line":l, "patch":patch, "data":data, "median":med, "lo":lo, "hi": hi}


def plot_source(post, source, conf=0.9, ax=None):

    if ax is None: ax = plt.gca()

    scale = 0.1

    years = np.arange(1900, 2100)

    # idx = years >= 1995 # no need to overload the historical period
    plot_ensemble(years, post[f'ssp126_{source}']*scale, color='tab:green', label="SSP1-26", ax=ax, conf=conf)
    plot_ensemble(years, post[f'ssp585_{source}']*scale, color='tab:orange', label="SSP5-85", ax=ax, conf=conf)

    # Overlay observations
    ax.plot(fred_df.index,
        align(fred_df.index, fred_df[MAP_FRED.get(source, source)].values) * scale,
        label="Frederikse", color='k')

    (ar6_26[MAP_AR6.get(source, source)]*scale).loc[:2100].plot(color='k', ls='--', label="", ax=ax)
    (ar6_85[MAP_AR6.get(source, source)]*scale).loc[:2100].plot(color='k', ls='--', label='AR6', ax=ax)

    ax.plot([1900, 2100], [0, 0], c='gray', lw=1)

    ax.set_xlabel('Year')
    ax.set_ylabel(f'{source} (cm)')
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(years[[0, -1]])
    ax.plot(years[[0, -1]], [0, 0], c='gray', lw=1)


class LocalFit:
    def __init__(self, trace, prior, stations, experiment=None):
        if isinstance(trace, xa.Dataset):
            posterior = trace
        else:
            posterior = trace.posterior
        if "chain" in posterior.dims:
            posterior = posterior.rename_dims({"draw": "old_draw"}).stack(draw = ("chain", "old_draw"))

        # back-compatibility...
        posterior = _transpose(posterior, ['station', 'year'], append=True)

        post = {k:posterior[k].values for k in posterior}

        self.years = posterior.year.values
        self.post = post
        self.prior = prior
        # self.gmsl = gmsl
        self.stations = stations
        self.ar6_proj = {}
        self.experiment = experiment


    def _get_ar6(self, ID, experiment, source):
        key = ID, experiment, source

        if key in self.ar6_proj:
            return self.ar6_proj[key]

        source_ar6 = MAP_AR6.get(source, source)

        try:
            series = ar6.supp.load_location(ID, experiment, sources=[source_ar6])[source_ar6].loc[:2100]
        except Exception as error:
            # print(">>>", error)
            print("!! Failed to load AR6 data", key)
            series = None

        self.ar6_proj[key] = series
        return series


    def plot(self, source, idx=0, conf=0.9, ax=None, title=None, legend=True, **kwargs):

        ax = ax or plt.gca()
        # idx = [station.name for station in stations].index(station_index)
        station = self.stations[idx]
        ax.set_title(title or f"{station['Station names']} {source}")


        scale = 0.1
        if source != "vlm":
            if self.prior is not None: plot_ensemble(self.years, self.post[source+"_global"]*scale, color='k', label='global', alpha=0, lw=1, ax=ax)
            if self.prior is not None: plot_ensemble(self.years, self.prior[source+"_rsl"][:, idx]*scale, color='tab:green', lw=2, label='prior', conf=conf, ax=ax, **kwargs)
            ens = plot_ensemble(self.years, self.post[source+"_rsl"][:, idx]*scale, color='tab:orange', lw=2, label='posterior', conf=conf, ax=ax, **kwargs)
        else:
            # plot_ensemble(self.years, self.prior.get(source+"_global", np.zeros(shp))*scale, color='k', label='global', alpha=0, lw=1, ax=ax)
            if self.prior is not None:
                shp = self.prior['total_global'].shape
                plot_ensemble(self.years, self.prior[source+"_rsl"][:, idx, None]*np.ones(shp)*scale, color='tab:green', lw=2, label='prior', conf=conf, ax=ax, **kwargs)
            shp2 = self.post['total_global'].shape
            ens = plot_ensemble(self.years, self.post[source+"_rsl"][:, idx,None]*np.ones(shp2)*scale, color='tab:orange', lw=2, label='posterior', conf=conf, ax=ax, **kwargs)

        if self.experiment:
            for ID in station["PSMSL IDs"].split(','):
                series = self._get_ar6(int(ID), self.experiment, source)
                if series is None: continue
                ax.plot(series.index, series*scale, marker='x', linestyle='', color="tab:red")

        # Add observations
        if source == "total":

            # tide gauges
            guide = pd.Series(ens['median'], index=self.years)

            from sealevelbayes.models.localslr import tg, tg_years, get_satellite_timeseries

            IDs = [int(ID) for ID in station["PSMSL IDs"].split(',')]
            info = f"s ({len(IDs)})" if len(IDs) > 1 else ""

            for i, ID in enumerate(IDs):
                # tide gauges
                k = tg['id'].tolist().index(int(ID))

                slr = tg['height'][k]*scale
                off = (slr - guide.reindex(tg_years)).dropna().mean()
                l, = ax.plot(tg_years, slr-off, c='tab:blue', label=f'tide-gauge{info} ("height")' if i == 0 else "")

                slr = tg['height_corr'][k]*scale
                off = (slr - guide.reindex(tg_years)).dropna().mean()
                l, = ax.plot(tg_years, slr-off, c='tab:blue', linestyle=":", label=f'tide-gauge{info} ("height_corr")' if i == 0 else "")

            # satellite
            # is defined w.r.t geoid, and here we want to convert to RSL
            guide_rad = pd.Series(np.median(self.post[source+"_rad"][:, idx]*scale, axis=0), index=self.years)

            sat_years, timeseries = get_satellite_timeseries([station["Longitude"]], [station["Latitude"]])

            slr = timeseries[0]*1000*scale
            off = (slr - guide.reindex(sat_years)).dropna().mean()
            l, = ax.plot(sat_years, slr-off, c='darkblue', label=f'satellite geoid', linestyle=":")

            slr = timeseries[0]*1000*scale - guide_rad.reindex(sat_years)
            off = (slr - guide.reindex(sat_years)).dropna().mean()
            l, = ax.plot(sat_years, slr-off, c='darkblue', label=f'satellite rsl')

#                 ar6_df = ar6.load_location(ID, "ssp585")

        ylims = ax.get_ylim()
        ax.vlines(2018, *ylims, color='k', linestyle=":")
        ax.set_ylim(*ylims)
        ax.set_xlim([self.years[0], self.years[-1]])
        ax.grid()
        ax.set_ylabel("Sea level (cm)")
        if legend: ax.legend(fontsize='small', loc='upper left')


    def plot_all(self, indices=None, sources=None, num=None, figsize=None, **kwargs):
        if sources is None:
            sources = ["steric", "glacier", "gis", "ais", "landwater", "total"]

        if indices is None:
            indices = np.arange(len(self.stations))

        if type(indices) is int:
            indices = [indices]

        if figsize is None:
            figsize = (2*len(sources), len(indices)*1.5)

        print(figsize)

        fig, axes = plt.subplots(len(indices), len(sources), sharex=True, sharey=False, num=num, figsize=figsize, clear=True)
        # axes_ = axes.flatten()

        for i, idx in enumerate(indices):
            for s, source in enumerate(sources):
                ax = axes[i, s]
                self.plot(source, idx, legend=False, ax=ax, title=" ", **kwargs)
                if not ax.is_first_col():
                    ax.set_ylabel('')
                else:
                    nl = "\n"
                    ax.set_ylabel(f'{self.stations[idx]["Station names"].replace(";",nl)}\nSea level (cm)')
                # if source == "total":
                #     ax.set_title(self.stations[idx]["Station names"])
                # else:
                if ax.is_first_row():
                    ax.set_title(source)
                else:
                    ax.set_title("")

                # ax.set_ylim([-20, 120])

        # ax.legend(fontsize='xx-small', loc='upper left')
        # plt.suptitle(self.stations[idx]["Station names"])
        plt.tight_layout(pad=0,h_pad=-1)



# def plot_cov_likelihood():
#     "simple check"

#     names = []
#     dataset = []
#     for source in MAP_FRED:
#         slr20_ensemble, rate2000_ensemble = _get_fred_ensembles(source)
#         dataset.extend([slr20_ensemble, rate2000_ensemble])
#         names.extend([source+"_slr20", source+"_rate2000"])

#     plt.figure()

#     plt.pcolormesh(np.arange(13), np.arange(13), np.corrcoef(dataset), shading='flat', edgecolor='white')
#     ax = plt.gca()
#     ax.set_xticks(np.arange(12)+0.5, [name for name in names], rotation=20, fontsize='x-small', horizontalalignment='right');
#     ax.set_yticks(np.arange(12)+0.5, [name for name in names], fontsize='x-small', horizontalalignment='right');
#     plt.colorbar()
#     plt.title('correlation in the observations')