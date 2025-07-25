#!/usr/bin/env python

import itertools
from pathlib import Path
import json, os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import xarray as xa
import pandas as pd
from scipy.signal import savgol_filter
import xarray as xa


# import sys
# sys.path.append('notebooks')

from sealevelbayes.config import logger, get_runpath, get_webpath
import sealevelbayes.datasets.frederikse2020 as frederikse2020
from sealevelbayes.datasets.shared import MAP_AR6, MAP_FRED_NC
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.ar6.supp import AR6_GLOBAL, load_ar6_medium_confidence, get_ar6_global_numbers
from sealevelbayes.models.metadata import SOURCES
from sealevelbayes.postproc.serialize import interpolate_js, get_model_quantiles
from sealevelbayes.postproc.colors import xcolors, xlabels, sourcelabels

# from sensitivity import sensitivity_experiments

root = Path(__file__).parents[2]

# SELECT_EXPERIMENTS = ['SP', 'GS', 'CurPol', 'ssp126', 'ssp585']
SELECT_EXPERIMENTS = ['SP', 'GS', 'CurPol']
SELECT_EXPERIMENTS_mu = [x+"_mu" for x in SELECT_EXPERIMENTS]

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWX")

for x in list(xcolors):
    if x+"_mu" not in xcolors and x in xcolors:
        xcolors[x+"_mu"] = xcolors[x]

for x in SELECT_EXPERIMENTS + list(xlabels):
    if x+"_mu" not in xlabels:
        if x in xlabels:
            xlabels[x+"_mu"] = xlabels[x] + "*"
        else:
            xlabels[x+"_mu"] = x + "*"


# from models import observe_slr20, observe_rate, observe_slr2100

# CIRUN = "9d02309"
# trace = arviz.from_netcdf(f"../../ci/runs/{CIRUN}/slr-global.trace.nc")

def load_ar6_records(rate=False):
    records = []
    for x in ['ssp126', 'ssp585']:
        for s in MAP_AR6:
            if s == 'vlm': continue
            s2 = MAP_AR6.get(s, s)
            res = load_ar6_medium_confidence(s2, x, [.5, .05, .95], rate=rate, from_samples=rate).sel(years=slice(None, 2100)).squeeze()
            # res = load_ar6_medium_confidence(s2, x, [.5, .05, .95], rate=rate, from_samples=False).sel(years=slice(None, 2100)).squeeze()
            records.append({
                'source': s,
                'experiment': x,
                'median': res.values[0].tolist(),
                'lower': res.values[1].tolist(),
                'upper': res.values[2].tolist(),
                'years': res.years.values.tolist(),
            })
    return {
        'confidence': 'medium_confidence',
        'records': records,
    }


def load_sat():
    sat = xa.load_dataset(get_datapath("cds/satellite-sea-level-global/satellite_sla_1993_2019_globalmean.nc"))
    sat_years = [d.year for d in sat.time.values.astype('datetime64[Y]').tolist()]
    sat_values = (sat['sla'] + sat['tpa_correction']).values.squeeze()
    sat_values = (sat_values - sat_values[1995-1993:2014+1-1993].mean())*1000 # in mm
    return sat_years, sat_values


def load_fredjs(rate=False):
    # MAP_FRED_NC
    gmsl = xa.open_dataset(frederikse2020.root/"GMSL_ensembles.nc")
    if rate:
        aligned = gmsl.diff('time')
    else:
        aligned = gmsl - gmsl.sel(time=slice(1995, 2014)).mean(dim='time')
    fredjs = {}
    fredjs['years'] = aligned.time.values
    fredjs['records'] = []
    for s in list(MAP_FRED_NC) + ["Sum"]:

        if s == "Sum":
            data = aligned["Barystatic"] + aligned["Steric"]
        else:
            data = aligned[MAP_FRED_NC.get(s, s)]
        lo, mid, hi = data.quantile([0.05, 0.5, 0.95], dim='likelihood')
        fredjs['records'].append({
            'source': s,
            'year': gmsl.time.values,
            'lower': lo.values,
            'median': mid.values,
            'upper': hi.values,
        })

    # Also add the sum of contributions
    return fredjs


def load_observations(rate=False, options=None):
    if options is None:
        return load_fredjs(rate)

    import argparse
    from sealevelbayes.runslr import load_timeseries_data
    o = argparse.Namespace(**options)

    o.noise_on_rate = False # load the cumsum
    o.greenland_noise_on_rate = False
    o.antarctica_noise_on_rate = False
    o.steric_noise_on_rate = False
    o.glacier_noise_on_rate = False

    gmsl, _ = load_timeseries_data(o)

    if rate:
        aligned = gmsl.diff('year')
    else:
        aligned = gmsl - gmsl.sel(year=slice(1995, 2014)).mean(dim='year')

    fredjs = {}
    fredjs['years'] = aligned.year.values
    fredjs['records'] = []
    for s in ["ais", "gis", "steric", "landwater", "glacier", "total"]:
        data = aligned[s]
        lo, mid, hi = data.quantile([0.05, 0.5, 0.95], dim='sample')
        fredjs['records'].append({
            'source': s,
            'year': gmsl.year.values,
            'lower': lo.values,
            'median': mid.values,
            'upper': hi.values,
        })

    return fredjs


DANGENDORF = get_datapath("dangendorf2019")
CHURCH = get_datapath("church_white_gmsl_2011_up")

def read_dangendorf2019():
    dg = pd.read_csv(DANGENDORF/"41558_2019_531_MOESM2_ESM.txt", sep="\s+", skiprows=1)
    dg.columns = ["Time", "GMSL (mm)", "GMSL uncertainty (mm)"]
    return dg

def read_church2011():
    path = CHURCH / "CSIRO_Recons_gmsl_yr_2015.csv"
    return pd.read_csv(path)

# from sealevelbayes.datasets.glaciers import load_glacier_datasets, gt_to_mm_sle, M20_SCENARIOS, SSP_SCENARIOS
# _, _, _, _, _, _, _, gmip_future = load_glacier_datasets()
# m20_scenario_map = dict(zip(SSP_SCENARIOS, M20_SCENARIOS))

class DataCache:
    def __init__(self):
        self._cache = {}

    def get(self, name, func, *args, **kwargs):
        if name not in self._cache:
            self._cache[name] = func(*args, **kwargs)
        return self._cache[name]


def add_gmsl_obs(axes, sources, rate=False, options=None, ar6=True, tuningdata=True,
                 fred_gmsl=False, fred_sources=False, satellite=True, marzeion=False,
                 records=[],
                 dangendorf=True, scale=lambda y: y, colors=xcolors, cache=None, ar6width=5, hpad=2, hspace=2):

    if cache is None:
        cache = DataCache()

    if tuningdata and options is not None:
        # fred = pd.read_csv(root/'web/obs/frederikse2020_global.csv')
        # ar6 = pd.read_csv(root/'web/obs/ar6_global.csv')
        obsjs = cache.get(f"tuningdata-{rate}", load_observations, rate=rate, options=options)

        for r in obsjs['records'][:-2]+obsjs['records'][-2:][::-1]:
            source2 = r['source'].replace('Sum', 'total')
            ax = axes.ravel()[sources.index(source2)]

            suffix = ''
            color = 'black'
            hatch = '...'
            label = 'Observations'
            facecolor = 'None'
            alpha = 0.5

            if source2 == 'total':
                label += " (Total)"


            ax.plot(obsjs['years'], scale(r['median']), color=color, linewidth=1)
            ax.fill_between(obsjs['years'], scale(r['lower']), scale(r['upper']),
                        color=color, facecolor=facecolor, alpha=alpha, hatch=hatch, label=label)

    if fred_sources or fred_gmsl:
        fredjs = load_fredjs(rate=rate)
        fredjs = cache.get(f"fredjs-{rate}", load_fredjs, rate=rate)

        for r in fredjs['records'][:-2]+fredjs['records'][-2:][::-1]:
            source2 = r['source'].replace('Sum', 'total')
            ax = axes.ravel()[sources.index(source2)]

            if r["source"] == "total":
                if not fred_gmsl:
                    continue

            else:
                if not fred_sources:
                    continue

            if source2 == 'total':
                color = {'Sum': 'black'}.get(r['source'], 'gray')
                hatch = {'Sum': '...'}.get(r['source'], 'None')
                facecolor = {'Sum': 'None'}.get(r['source'], color)
                alpha = {'Sum': .5}.get(r['source'], .2)
                label = 'Frederikse (ref 13) (' + {'Sum': 'Total'}.get(r['source'], 'GMSL') + ')'
                # hatch = '...' i
            else:
                suffix = ''
                color = 'black'
                hatch = '...'
                label = 'Frederikse (ref 13)'
                facecolor = 'None'
                alpha = 0.5

            ax.plot(fredjs['years'], scale(r['median']), color=color, linewidth=1)
            ax.fill_between(fredjs['years'], scale(r['lower']), scale(r['upper']),
                        color=color, facecolor=facecolor, alpha=alpha, hatch=hatch, label=label)

    # ar6
    if ar6 and not rate:
        # from sealevelbayes.datasets.ar6.tables import ar6_table_9_8_medium_confidence
        for x in "ssp126", "ssp585":
            # for s in ar6_table_9_8_medium_confidence[x]:
                # if s not in sources:
                #     continue
            color = colors.get(x)

            def plot_bar(ax, xx, quantiles, alpha=.2, hatch=None, **kw):
                mid, lo, hi, lo2, hi2 = quantiles
                patch = ax.fill_between(xx, [scale(lo)]*2, [scale(hi)]*2, color=color, alpha=alpha, hatch=hatch)
                ax.fill_between(xx, [scale(lo2)]*2, [scale(hi2)]*2, color=color, alpha=alpha, hatch=hatch)
                ax.plot(xx, [scale(mid)]*2, color="black", lw=.5, **kw)
                return patch

            for s in sources:
                qlevs = [.5, .05, .95, .167, .833]
                ax = axes.ravel()[sources.index(s)]
                xlims = ax.get_xlim()
                texttransform = blended_transform_factory(ax.transData, ax.transAxes)
                fontsize = 8
                # fontfamily = "serif"
                fontfamily = None
                fontweight = 300
                texty = .02
                textpad = 1

                xstart = 2100 + hspace

                if records:
                    qvals = get_model_quantiles(records, experiment=x, source=s, diag="rate2100" if rate else "proj2100", field="global", quantiles=qlevs)
                    # xx = (xlims[1] - 2*ar6width-hpad, xlims[1]-ar6width-hpad)
                    # xx = [2100+hpad, 2100+hpad+ar6width]
                    # xx2 = [xx[0]-hpad-ar6width, xx[0]-hpad]
                    xx = [xstart, xstart + ar6width]
                    # plot_bar(ax, xx, qvals, label='This study 2100' if x == "ssp585" else None)
                    plot_bar(ax, xx, qvals, label=None)
                    # ax.text(xstart + fontsize, .05, "This study", ha='right', va='bottom', fontsize=fontsize, transform=texttransform, rotation=90)

                    if x == "ssp126":
                        # ax.text(2100-textpad, texty, "This study", ha='right', va='bottom', fontsize=fontsize, transform=texttransform, weight=fontweight, fontfamily=fontfamily)
                        ax.annotate('This study', xy=(xstart+ar6width/2, scale(min(qvals))), xytext=(2100-textpad, texty),
                        arrowprops=dict(facecolor='black', arrowstyle='-'),
                        # arrowprops=dict(facecolor='black', arrowstyle='-', connectionstyle='arc3,rad=0.2'),
                        textcoords=texttransform, xycoords='data',
                        ha='right', va='bottom', fontsize=fontsize, weight=fontweight, fontfamily=fontfamily)


                    # textythis = .02 + .1
                    xstart += ar6width + hpad
                    # xline = xstart-hpad/2
                    # ax.text(xline-textpad, textythis, "This study", ha='right', va='bottom', fontsize=fontsize, transform=texttransform, weight=fontweight, fontfamily=fontfamily)
                    # ax.vlines([xline], textythis, 1, color="k", lw=.5, ls="--", transform=texttransform)

                # xx = (xlims[1] - ar6width, xlims[1])
                if not rate:
                    qvals = get_ar6_global_numbers(x, s, years=2100, quantiles=qlevs, rate=rate).squeeze("locations").values
                    # if rate:
                    #     qvals *= 10
                    # xmid = (xx[0] + xx[1])/2

                    xx = [xstart, xstart + ar6width]
                    # plot_bar(ax, xx, qvals, label='IPCC AR6 2100' if x == "ssp585" else None)
                    patch = plot_bar(ax, xx, qvals, label=None)

                    if x == "ssp126":

                        # ax.text(.999, .95, "AR6", ha='right', va='top', fontsize=8, transform=ax.transAxes)
                        # ax.text(.999, .01, "AR6", ha='right', va='bottom', fontsize=8, transform=ax.transAxes)
                        # ax.plot(xx, [scale(mid)]*2, color=color, label='IPCC AR6 2100' if x == "ssp585" else None, lw=.5)
                        # ax.text(xstart+fontsize, .05, "AR6", ha='right', va='bottom', fontsize=fontsize, transform=texttransform, rotation=90)

                        # ax.text(2100+textpad, texty, "AR6", ha='left', va='bottom', fontsize=fontsize, transform=texttransform, weight=fontweight, fontfamily=fontfamily)
                        # ax.plot([xstart+ar6width/2, xstart+ar6width/2], [scale(mid)]*2, color="black", lw=.5)
                        # text_position = texttransform.transform((2100+textpad, texty))
                        # print(text_position)

                        ax.annotate('AR6', xy=(xstart+ar6width/2, scale(min(qvals))), xytext=(2100+textpad, texty),
                            arrowprops=dict(facecolor='black', arrowstyle='-'),
                            # arrowprops=dict(facecolor='black', arrowstyle='-', connectionstyle='arc3,rad=0.2'),
                            textcoords=texttransform, xycoords='data',
                            ha='left', va='bottom', fontsize=fontsize, weight=fontweight, fontfamily=fontfamily)

                    if s == "ais" and x == "ssp585":
                        ylim = ax.get_ylim()
                        ax.set_ylim(ylim[0]-5, ylim[1])

                ax.set_xlim(xlims[0], xx[-1]+hspace)
                ax.axvline(2100, color="k", lw=.5, ls="-")


                # ax.plot([2015, 2100], [scale(mid), scale(mid)], color=color, linewidth=1, linestyle='None', marker='.')


                # ax.fill_between([2015, 2100], scale(res[0]), scale(res[2]),
                #             color=color, facecolor='None', alpha=0.5, hatch='...', label='AR6' if x == "ssp585" else None)
        # ar6js = load_ar6_records(rate=rate)
        # ar6js = cache.get(f"ar6js-{rate}", load_ar6_records, rate=rate)
        # for r in ar6js['records']:
        #     ax = axes.ravel()[sources.index(r['source'])]
        #     color = colors.get(r["experiment"])
        #     ax.plot(r['years'], scale(r['median']), color=color, linewidth=1, linestyle='None', marker='.')
        #     ax.fill_between(r['years'], scale(r['lower']), scale(r['upper']),
        #                 color=color, facecolor='None', alpha=0.5, hatch='...', label='AR6' if r['experiment'].startswith('ssp585') else None)
        pass


    if satellite:
        sat_years, sat_values = load_sat()
        # satellite
        ax = axes.ravel()[sources.index('total')]
        satcolor = 'cyan'
        if rate:
            ax.plot(sat_years[1:], scale(np.diff(sat_values)), label="Satellite altimetry", color=satcolor, zorder=100)
        else:
            ax.plot(sat_years, scale(sat_values), label="Satellite altimetry", color=satcolor, zorder=100)

    # glaciers
    # malles and marzeion 2021
    if marzeion:
        marz = pd.read_csv(root/'web/obs/marzeionmalles2021-figure6-digitized.csv')
        if rate:
            marz = marz.set_index('year').diff().reset_index()
            for k in ['upper', 'lower', 'median']:
                marz[k] = savgol_filter(marz[k], 5, 1)
        ax = axes.ravel()[sources.index('glacier')]
        col = 'tab:orange'
        # ax.fill_between(marz['year'], scale(marz['lower']), scale(marz['upper']), color='tab:blue', alpha=0.3)
        ax.fill_between(marz['year'], scale(marz['lower']), scale(marz['upper']),
                        color=col, facecolor='None', alpha=0.5, hatch='|||', label='Malles and Marzeion 2021')

    if dangendorf:
        # Other global reconstructions
        ax = axes.flat[-1]
        dg = read_dangendorf2019().set_index('Time')/10
        if rate:
            dg = dg.diff()
            dg['GMSL uncertainty (mm)'] = np.nan
        else:
            dg['GMSL (mm)'] -= dg['GMSL (mm)'].loc[1995:2014.9999].mean()

        l, = ax.plot(dg.index, dg['GMSL (mm)'], label="Dangendorf (ref 12)", color=colors.get('dangendorf2019'), lw=1)
        ax.plot(dg.index, dg['GMSL (mm)'] + dg['GMSL uncertainty (mm)'], color=l.get_color(), ls=':', lw=.5)
        ax.plot(dg.index, dg['GMSL (mm)'] - dg['GMSL uncertainty (mm)'], color=l.get_color(), ls=':', lw=.5)



def make_gmsl_timeseries(js, historical=False, experiments=['ssp126','ssp585'], axes=None, setup_axes=True, alpha=.2, colors=None,
                         labels=None, index="experiment", rate=False, legend=True, shade_kwargs={}, line_kw={}, add_obs=True,
                         gridspecs=None, letters=LETTERS,
                         **obs_kwargs):

    js = interpolate_js(js)

    if colors is None:
        colors = xcolors.copy()
        colors["historical"] = "steelblue"

    if labels is None:
        labels = xlabels.copy()

    if historical:
        ymin, ymax = 1900, 2018
    else:
        ymin, ymax = 1900, 2100


    def fix(y, ymin=ymin, ymax=ymax, experiment=None):
        y = np.asarray(y)
        m = np.ones_like(y, dtype=bool)
        if experiment == 'historical':
            ymax = 2018
        # elif index == "experiment": # does not apply for "by chain"
            # ymin = 2018
        if ymin: m &= (y >= ymin)
        if ymax: m &= (y <= ymax)
        i, j = np.where(m)[0][[0, -1]]
        return slice(i, j+1)

    # records specs
    sources = ['steric', 'glacier', 'gis', 'ais', 'landwater', 'total']
    recs = [r for r in js['records'] if (not experiments or r['experiment'] in experiments) and r['diag'] == ('rate' if rate else 'change') and r['source'] in sources]
    recs2100 = [r for r in js['records'] if (not experiments or r['experiment'] in experiments) and r['diag'] == ('rate2100' if rate else 'proj2100') and r['source'] in sources]
    # recs2050 = [r for r in js['records'] if (not experiments or r['experiment'] in experiments) and r['diag'] == ('rate2050' if rate else 'proj2050') and r['source'] in sources]

    experiments = list(sorted(set(r['experiment'] for r in recs)))
    # also add "historical" experiment with a different color coding (blue)
    if index == "experiment" or historical: # does not apply for "by chain"
        recs.extend([{**r, **{"experiment":"historical"}} for r in recs if r['experiment'] == experiments[0]])

    # axes layout specs
    if axes is None:
        f, axes = plt.subplots(3, 2, figsize=(8, 6))

    if rate:
        scale = lambda y: np.asarray(y)
    else:
        scale = lambda y: np.asarray(y)*1e-1

    def plot_model(r, zorder=100, label="", color=None, alpha=alpha, shade_kwargs={}, **kwargs):
        ix = fix(js['years'], experiment=r['experiment'])
        l, = ax.plot(js['years'][ix], scale(r['median'][ix]),
            **{**dict(label=label or labels.get(r[index], r[index]),
            color=color or colors.get(r[index]), zorder=zorder), **kwargs})
        ax.fill_between(js['years'][ix], scale(r['lower'][ix]), scale(r['upper'][ix]),
                **{**dict(color=l.get_color(), alpha=alpha, zorder=zorder-1), **shade_kwargs})
        ilower2 = r["quantile_levels"].index(0.167)
        iupper2 = r["quantile_levels"].index(0.833)
        lower2 = r["quantile_values"][ilower2]
        upper2 = r["quantile_values"][iupper2]
        ax.fill_between(js['years'][ix], scale(lower2[ix]), scale(upper2[ix]),
                **{**dict(color=l.get_color(), alpha=alpha, zorder=zorder-1), **shade_kwargs})
        return l,

    # model data
    for i, r in enumerate(recs):
        if historical and r["experiment"] != "historical":
            continue
        # ix = fix(js['years'], ymin=ymin if r['experiment'] == experiments[0] else 2016)
        ax = axes.ravel()[sources.index(r['source'])]
        l, = plot_model(r, alpha=alpha, shade_kwargs=shade_kwargs, **line_kw)
        ax.set_xlim([ymin, ymax])

    if historical:
        obs_kwargs['ar6'] = False

    if add_obs:
        add_gmsl_obs(axes, sources, rate=rate, colors=colors, scale=scale, records=recs2100, **obs_kwargs)

    # axes specification
    if not setup_axes:
        return plt.gcf(), axes

    for i, ax in enumerate(axes.ravel()):
        # ax.grid()
        ax.axvline(2005, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        if legend:
            # l = ax.legend(title=f"{letters[i]}. {sourcelabels.get(sources[i])}", fontsize='xx-small', loc='upper left' if sources[i] not in ('landwater', 'glacier') else None)
            l = ax.legend(title=f"({letters[i]}) {sourcelabels.get(sources[i])}", fontsize='xx-small', loc='upper left' if sources[i] not in ('landwater', 'glacier') else None,
                          title_fontproperties={'weight': 'bold', 'size': 9} )
            # l = ax.legend(title=sourcelabels.get(sources[i]), fontsize='xx-small', loc='upper left' if sources[i] not in ('landwater', 'glacier') else None)
            l._legend_box.align = "left"

        if gridspecs is None:
            axspec = ax.get_subplotspec()
        else:
            axspec = gridspecs[i]

        if axspec is not None:
            if not axspec.is_last_row():
                ax.set_xticklabels([])
            if axspec.is_last_col():
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
            if axspec.is_first_col() or axspec.is_last_col():
                if rate:
                    ax.set_ylabel('SLR (mm/yr)')
                else:
                    ax.set_ylabel('SLR (cm)')
            if axspec.is_last_row():
                ax.set_xlabel('Year')

        if ymin == 1900 and ymax == 2100:
            less_x_ticks = np.arange(1900, 2100+50, 50)
            ax.set_xticks(less_x_ticks)

        ax.tick_params(axis='y', labelsize='small')
        ax.tick_params(axis='x', labelsize='small')


    return plt.gcf(), axes


def make_gmsl_timeseries2(js, **kwargs):

    # axes layout specs
    f, axes = plt.subplots(6, 2, figsize=(8, 12))

    make_gmsl_timeseries(js, axes=axes[:, 0], rate=True, legend=False, **kwargs)
    make_gmsl_timeseries(js, axes=axes[:, 1], rate=False, **kwargs)

    f.tight_layout(w_pad=0)

    return f, axes


def make_gmsl_timeseries3(js, rate=False, **kwargs):

    # axes layout specs
    f, axes = plt.subplots(6, 2, figsize=(8, 12))

    kwargshist = kwargs.copy()
    kwargshist["ar6"] = False
    make_gmsl_timeseries(js, axes=axes[:, 0], rate=rate, fred_gmsl=True, historical=True, letters=LETTERS[::2], **kwargshist)
    make_gmsl_timeseries(js, axes=axes[:, 1], rate=rate, tuningdata=False, fred_gmsl=False, letters=LETTERS[1::2], fred_sources=False, dangendorf=False, satellite=False, **kwargs)

    f.tight_layout(w_pad=0)

    return f, axes

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def make_gmsl_timeseries4(js, legend=False, **kwargs):

    f, mainaxes = plt.subplots(3, 2, figsize=(8, 6))
    make_gmsl_timeseries(js, axes=mainaxes, tuningdata=False, fred_gmsl=False, fred_sources=False, dangendorf=False, satellite=False, **kwargs)
    f.tight_layout(w_pad=0)


    subaxes = []
    maindatawidth = 2100 + kwargs.get('ar6width', 0) - 1900
    insertdatawidth = 2018 - 1900
    xfrac = insertdatawidth / maindatawidth
    width = f"{xfrac*100}%"

    for i, ax in enumerate(mainaxes.ravel()):
        ax_inset = inset_axes(ax, width=width, height="60%", loc='upper left', borderpad=0)
        subaxes.append(ax_inset)

    kwargs.setdefault("legend", False)

    axes = np.array([subaxes, mainaxes.ravel().tolist()]).T

    make_gmsl_timeseries(js, axes=axes[:, 0], ar6=False, fred_gmsl=True, historical=True, setup_axes=True, **kwargs)

    # for i, ax in enumerate(subaxes):
    sources = kwargs.get("sources", ['steric', 'glacier', 'gis', 'ais', 'landwater', 'total'])

    for i, ax in enumerate(subaxes):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.tick_params(labelbottom=False)
        ax.xaxis.set_visible(False)
        source = sources[i]
        if source == "landwater":
            ax.set_visible(False)

    return f, axes





def _fmt(q):
    return "{1} ({0} to {2})".format(*q)

def fmt(q, d=None):
    if q is None: return "-"
    q = np.asarray(q)
    if d is None: return _fmt(q)
    return _fmt(q.round(d).astype(int) if d == 0 else q.round(d))


def get_gmsl_dataset(tr,
                     experiments=["ssp126", "ssp585", "ssp126_mu", "ssp585_mu"]+SELECT_EXPERIMENTS+SELECT_EXPERIMENTS_mu,
                     diags=['change', 'rate', 'proj2100', 'rate2100'],
                     sources=SOURCES,
                     extended_var_names=[], trace=None,
                     group="posterior_predictive", filename=None, suffix="", sample_obs=False, save=True, load=True, **kw):
    # from sealevelbayes.models.globalslr import DEFAULT_EXPERIMENTS

    if trace is None:
        trace = tr.trace

    if sample_obs:
        observed_data = trace.observed_data
        extended_var_names = extended_var_names + [name for name in [f"obs_{source}" for source in sources] if name in observed_data]

    if save or load:
        if filename is None:
            if group.startswith("posterior"):
                filename = f"posterior_global{suffix}.nc"
            elif group.startswith("prior"):
                filename = f"prior_global{suffix}.nc"
            else:
                raise ValueError(f"Unknown group: {group}")
        runfolder = get_runpath(tr.o.cirun)
        filepath = runfolder / "postproc" / filename

    if load and filepath.exists():
        logger.info(f"Load from {filepath}")
        posterior_global = xa.open_dataset(filepath)

    else:
        if group.startswith("posterior"):
            resample = tr.resample_posterior_global
        elif group.startswith("prior"):
            resample = tr.resample_prior_global
        else:
            raise ValueError(f"Unknown group: {group}")

        trace_global = resample(sources=sources,
                                diags=diags,
                                experiments=experiments,
                                extend_var_names=extended_var_names, trace=trace, **kw)

        # ok_chains = divergences_per_chain < 30
        # print(f"OK chains: {ok_chains.sum()} / {ok_chains.size} ")

        # posterior_global = trace_global.posterior_predictive.isel(chain=ok_chains)
        posterior_global = getattr(trace_global, group)

        if save:
            filepath.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"Save to {filepath}")
            posterior_global.to_netcdf(filepath)

    # add the 2100 projections for global-only sources
    for proj in ["proj", "rate"]:
        diag = f"{proj}2100"
        field = "global"
        for source in ["steric", "glacier", "gis", "ais", "landwater", "total"]:
            k = f"{diag}_{field}_{source}"
            ts_varname = f"change_{field}_{source}" if proj == "proj" else f"rate_{field}_{source}"
            if k not in posterior_global and ts_varname in posterior_global:
                logger.warning(f"Adding {k} from {ts_varname} to posterior global")
                ts_var = posterior_global[ts_varname]
                one_year_ahead_tendency = ts_var.sel(year=2099) - ts_var.sel(year=2098)
                posterior_global[k] = ts_var.sel(year=2099) + one_year_ahead_tendency

    return posterior_global


def get_gmsl_trend_dataset(tr, suffix="_trend", **kw):
    update_runslr_params = {
                    "add_greenland_noise": False,
                    "add_steric_noise": False,
                    "add_antarctica_noise": False,
                    "add_glacier_noise": False
                    }
    return get_gmsl_dataset(tr, update_runslr_params=update_runslr_params, suffix=suffix, **kw)


# def get_gmsl_datasets(tr, experiments=["ssp126", "ssp585", "ssp126_mu", "ssp585_mu"], trace_standalone=None):
#     datasets = {}
#     datasets["posterior"] = get_gmsl_dataset(tr, experiments=experiments)
#     datasets["posterior_trend"] = get_gmsl_trend_dataset(tr, experiments=experiments)
#     datasets["prior"] = get_gmsl_trend_dataset(tr, experiments=experiments, group="prior")
#     datasets["prior_trend"] = get_gmsl_trend_dataset(tr, experiments=experiments, group="prior")

#     # also prepare global-constraints-only posterior
#     if trace_standalone is not None:
#         datasets["posterior_standalone"] = get_gmsl_trend_dataset(tr, experiments=experiments, trace=trace_standalone)

#     return datasets

def convert_gmsl_dataset_to_js(dataset, nanvar="rate_global_glacier"):
    import arviz
    from sealevelbayes.postproc.serialize import trace_to_json
    import numpy as np
    dataset = arviz.extract(dataset)
    try:
        nans = np.isnan(dataset[nanvar].isel(experiment=0, year=0).values)
        print(nans.sum(), "Nan values found")
        dataset = dataset.isel(sample=np.where(~nans)[0])
    except:
        pass

    _, global_js = trace_to_json(None, posterior=dataset)
    return global_js


def get_table_numbers(post, post_global=None, prior=None):
    ar6tables = json.load(open(root/'web/obs/ar6tables.json'))

    recs = []

    def _scale_cm(x):
        return np.asarray(x)*.1  # mm to cm

    fred_ds = xa.open_dataset(frederikse2020.root/"GMSL_ensembles.nc")

    for s_ in ar6tables['table_9_5']:
        s = {'gmsl':'total', 'sum': 'total'}.get(s_, s_)
        d = ar6tables['table_9_5'][s_]

        # if s == "landwater":
            # continue

        source = s_
        if s_ == "sum": source += '*'
        if s_ == "gmsl": source += '*'
        # if s_ == "landwater": source += '*'

        if s_ == "sum":
            y = fred_ds["Barystatic"] + fred_ds["Steric"]
        else:
            y = fred_ds[MAP_FRED_NC.get(s, s)]
    #     mean = lambda y, y1, y2: y.sel(time=slice(y1, y2)).mean(dim='time')

        recs.append({
            "source": source,
            "name": "Δ 1901-1990 (cm)",
            # "model": post[f"{s}_slr20"].quantile([0.05, 0.5, 0.95], dim="draw").values,
            "model": _scale_cm(post[f"rate_global_{s}"].sel(experiment="ssp585_mu", year=slice(1901, 1990)).sum(dim='year').quantile([0.05, 0.5, 0.95], dim="sample").values),
            "model_global": _scale_cm(post_global[f"rate_global_{s}"].sel(experiment="ssp585_mu", year=slice(1901, 1990)).sum(dim='year').quantile([0.05, 0.5, 0.95], dim="sample").values) if post_global is not None else None,
            "model_prior": _scale_cm(prior[f"rate_global_{s}"].sel(experiment="ssp585_mu", year=slice(1901, 1990)).sum(dim='year').quantile([0.05, 0.5, 0.95], dim="sample").values) if prior is not None else None,
            "ar6": _scale_cm(d["Δ (mm)"]["1901-1990"]),
            "fred": _scale_cm((y.sel(time=1990)-y.sel(time=1901)).quantile([0.05, 0.5, 0.95], dim="likelihood").values),
        })

        recs.append({
            "source": source,
            "name": "1993-2018 (mm/yr)",
            # "model": post[f"{s}_rate2000"].quantile([0.05, 0.5, 0.95], dim="draw").values,
            "model": post[f"rate_global_{s}"].sel(experiment="ssp585_mu", year=slice(1993, 2018)).mean(dim='year').quantile([0.05, 0.5, 0.95], dim="sample").values,
            "model_global": post_global[f"rate_global_{s}"].sel(experiment="ssp585_mu", year=slice(1993, 2018)).mean(dim='year').quantile([0.05, 0.5, 0.95], dim="sample").values if post_global is not None else None,
            "model_prior": prior[f"rate_global_{s}"].sel(experiment="ssp585_mu", year=slice(1993, 2018)).mean(dim='year').quantile([0.05, 0.5, 0.95], dim="sample").values if prior is not None else None,
            "ar6": d["mm/yr"]["1993-2018"],
            "fred": y.diff(dim="time").sel(time=slice(1993, 2018)).mean(dim="time").quantile([0.05, 0.5, 0.95], dim="likelihood").values,
        })


    fred_ds.close()


    # Now add projections
    for x in ar6tables['table_9_8']:
        for s_ in ar6tables['table_9_8'][x]:
            s = {}.get(s_, s_)

            # if s in ("rate", "total"):
            if s in ("rate"):
                continue

            source = {"total":"sum"}.get(s, s)

            if s_ == "total": source += '*'
            if s_ == "gmsl": source += '*'
            # if s_ == "landwater": source += '*'

            xlab = {"ssp126": "SSP1-2.6", "ssp585": "SSP5-8.5"}
            xlab2 = {"ssp126_mu": "SSP1-2.6 (median)", "ssp585_mu": "SSP5-8.5 (median)", "ssp126": "SSP1-2.6*", "ssp585": "SSP5-8.5*"}

            # for suffix in ["_mu", ""]:
            x_mu = x + '_mu'

            recs.append({
                "source": source,
                "name": f"{xlab.get(x, x)} Δ 2100 (cm)",
                # "label": f"{xlab.get(x2, x2)} Δ 2100 (mm)",
                "ar6": _scale_cm(np.array(ar6tables['table_9_8'][x][s_])*1000),
                # "model": post[f"{s}_{x}_slr21"].quantile([0.17, 0.5, 0.83], dim="draw").values,
                "model": _scale_cm(observe_slr2100(post[f"rate_global_{s}"].sel(experiment=x_mu).T).quantile([0.17, 0.5, 0.83], dim="sample").values),
                "model_full": _scale_cm(observe_slr2100(post[f"rate_global_{s}"].sel(experiment=x).T).quantile([0.17, 0.5, 0.83], dim="sample").values),
                "model_global": _scale_cm(observe_slr2100(post_global[f"rate_global_{s}"].sel(experiment=x_mu).T).quantile([0.17, 0.5, 0.83], dim="sample").values) if post_global is not None else None,
                "model_prior": _scale_cm(observe_slr2100(prior[f"rate_global_{s}"].sel(experiment=x_mu).T).quantile([0.17, 0.5, 0.83], dim="sample").values) if prior is not None else None,
                # "model": post[f"proj2100_global_{s}"].sel(experiment=x_mu).T.quantile([0.17, 0.5, 0.83], dim="sample").values,
                # "model_full": post[f"proj2100_global_{s}"].sel(experiment=x).T.quantile([0.17, 0.5, 0.83], dim="sample").values,
             })

    return recs


def observe_slr2100(rates):
    "observations to 2100"
    slr = rates.cumsum(dim='year')
    return (slr.sel(year=2099) + rates.sel(year=2099)) - slr.sel(year=slice(1995, 2014)).mean(dim='year') # last year missing, we extrapolate


def make_table(recs):
    # re-arrange for nicer table formatting

    sources = ['steric', 'glacier', 'gis', 'ais', 'landwater', 'landwater*', 'sum*', 'sum', 'gmsl*', 'gmsl']

    def rearrange(recs, label=" "):
        recs2 = []
        key = lambda r: (sources.index(r['source']), r['source'])
        for s, group in itertools.groupby(sorted(recs, key=key), key=key):
            g = list(group)
            for ref in ["model", "model_full", "model_global", "model_prior", "ar6", "fred"]:
                recs2.append({
                    "source": sourcelabels.get(s[1], s[1].capitalize()),
                    label: ref,
                    **{r["name"]: fmt(r.get(ref, None), d) for r, d in zip(g, [0, 2, 0, 0])}
                })
        return recs2

    recs2 = rearrange(recs)

    return pd.DataFrame(recs2).set_index(['source', ' ']).fillna('-').rename({"model": "this study", "fred": "Frederikse*", "ar6":"AR6"})


TABLE_FIG_NAMES = ['Δ 1901-1990 (cm)', '1993-2018 (mm/yr)',
                      'SSP1-2.6 Δ 2100 (cm)', 'SSP5-8.5 Δ 2100 (cm)']

def make_table_fig(recs):

    # x, y = zip(pd.DataFrame(recs2))
    f, axes = plt.subplots(2, 2, sharex=False)
    colors = {
        'model': 'tab:blue',
        'model_full': 'lightsteelblue',
        'model_global': 'skyblue',
        'model_prior': 'powderblue',
        'ar6': 'black',
        'fred': 'gray'
    }

    legendlabels = {
        'model': 'This study',
        'model_full': 'This study (with temperature uncertainty)',
        'model_global': 'This study (no local constraints)',
        'model_prior': 'This study (prior)',
        'ar6': 'AR6',
        'fred': 'Frederikse et al (2020)*'
    }

    names = TABLE_FIG_NAMES

    # sources = []
    sources = ['steric', 'glacier', 'gis', 'ais', 'landwater*', 'landwater', 'sum*', 'sum', 'gmsl*', 'gmsl']

    key = lambda r: r['name']
    key_sort = lambda r: (names.index(r['name']), sources.index(r['source']))

    def ebar(q):
        lo, mi, hi = q
        return {
            'y': mi,
            'yerr': [[mi-lo], [hi-mi]],
    #         'elinewidth': 2,
            'capsize': 8*dx,
            'marker': '.',
        }

    # labels = [r['source'] for r in recs if r['name'] == 'Δ 1901-1990 (mm)' ]

    maxlabelslen = 0

    for j, (ax, (name, group)) in enumerate(zip(axes.ravel(), itertools.groupby(sorted(recs, key=key_sort), key))):
        labels = []
        for i, r in enumerate(group):
    #         ax.bar(i*s, r['model'], width=0.35*s, color=colors.get('model'), label='this study' if i == 0 else None)
    #         ax.bar((i+0.4)*s, r['ar6'], width=0.35*s, color=colors.get('ar6'), label='AR6' if i == 0 else None)
            nbars = ('model_full' in r) + ('model_global' in r) + ('model_prior' in r) + 1 + ('fred' in r)
            dx = .6/nbars
            k = -1

            if r.get('model_full') is not None:
                # k+=1  # don't shift the rest
                ax.errorbar(i+k*dx, **ebar(r['model_full']), color=colors.get('model_full'), label=legendlabels.get('model_full') if i == 0 else None)

            k+=1
            ax.errorbar(i+k*dx, **ebar(r['model']), color=colors.get('model'), label=legendlabels.get('model') if i == 0 else None)

            if r.get('model_global') is not None:
                k+=1
                ax.errorbar(i+k*dx, **ebar(r['model_global']), color=colors.get('model_global'), label=legendlabels.get('model_global') if i == 0 else None)

            if r.get('model_prior') is not None:
                k+=1
                ax.errorbar(i+k*dx, **ebar(r['model_prior']), color=colors.get('model_prior'), label=legendlabels.get('model_prior') if i == 0 else None)

            k+=1
            ax.errorbar((i+k*dx), **ebar(r['ar6']), color=colors.get('ar6'), label=legendlabels.get('ar6') if i == 0 else None)

            if 'fred' in r:
                k += 1
                ax.errorbar((i+k*dx), **ebar(r['fred']), color=colors.get('fred'), label=legendlabels.get('fred') if i == 0 else None)

            labels.append(sourcelabels.get(r['source'], r['source'].capitalize()))

            # if name == 'Δ 1901-1990 (mm)' and r["source"] in ["glacier"]:
            #     labels[-1] += '*'

        maxlabelslen = max(maxlabelslen, len(labels))

    #     ax.legend(title=name,fontsize='xx-small')
        if j == 0 or j == 2: ax.legend(fontsize='xx-small', loc="upper left")
        ax.set_title(f'({list("abcd")[j]}) {name}', fontsize='small', fontweight="bold")
        ax.set_ylabel(name.split('(')[1].split(')')[0], fontsize='small')

        if ax.get_subplotspec().is_last_col():
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        # while len(labels) < 7:
        #     labels.append('')

        ax.grid()
        ax.set_xticks(np.arange(len(labels))+0.2, labels, fontsize='small', rotation=45, horizontalalignment='right')
        ax.set_xlim(-0.5, maxlabelslen)
        # ax.set_xticklabels(labels, fontsize='small')
        # ax.tick_params(axis='x', labelrotation = 45)
        ax.tick_params(axis='y', labelsize='small')

    # save y-lim for SSP1-2.6
    lo, _ = axes.ravel()[2].get_ylim()
    _, hi = axes.ravel()[3].get_ylim()
    axes.ravel()[2].set_ylim([lo, hi])
    axes.ravel()[3].set_ylim([lo, hi])

    plt.tight_layout()

    return f, axes


def make_table_fig_sensitivity(records, colors=None, show_ar6=False,
    sources=['steric', 'glacier', 'gis', 'ais', 'landwater', 'sum*'], sourcelabels={}):

    recs = [{label: r for label, r in zip(records, rs)} for rs in zip(*records.values())]
    # recs0 = next(iter(records.values()))
    keys = list(records)

    if colors is None:
        colors = {}
    for i, c in enumerate([k for k in keys+(['AR6'] if show_ar6 else []) if k not in colors]):
        colors[c] = plt.cm.tab10(i)

    # x, y = zip(pd.DataFrame(recs2))
    f, axes = plt.subplots(2, 2, sharex=False)


    names = ['Δ 1901-1990 (mm)', '1993-2018 (mm/yr)',
                      'SSP1-2.6 Δ 2100 (mm)', 'SSP5-8.5 Δ 2100 (mm)']

    recs = [r for r in recs if r[keys[0]]['source'] in sources]

    key = lambda r: r[keys[0]]['name']
    key_sort = lambda r: (names.index(r[keys[0]]['name']), sources.index(r[keys[0]]['source']))

    def ebar(q):
        lo, mi, hi = q
        return {
            'y': mi,
            'yerr': [[mi-lo], [hi-mi]],
    #         'elinewidth': 2,
            'capsize': 4,
            'marker': '.',
        }

    # labels = [r['source'] for r in recs if r['name'] == 'Δ 1901-1990 (mm)' ]

    for j, (ax, (name, group)) in enumerate(zip(axes.ravel(), itertools.groupby(sorted(recs, key=key_sort), key))):
        labels = []
        for i, r in enumerate(group):
    #         ax.bar(i*s, r['model'], width=0.35*s, color=colors.get('model'), label='this study' if i == 0 else None)
    #         ax.bar((i+0.4)*s, r['ar6'], width=0.35*s, color=colors.get('ar6'), label='AR6' if i == 0 else None)
            dx = 1/(len(colors)+1)
            for j, k in enumerate(keys):
                ax.errorbar(i+j*dx, **ebar(r[k]['model_full'] if 'model_full' in r[k] else r[k]['model']), color=colors.get(k), label=k if i == 0 else None)
            if show_ar6:
                ax.errorbar(i+(j+1)*dx, **ebar(r[k]['ar6']), color=colors.get('AR6'), label='AR6' if i == 0 else None)
    #         ax.bar((i+0.4)*s, r['ar6'], width=0.35*s, color=colors.get('ar6'), label='AR6' if i == 0 else None)

            labels.append(source.labels.get(r[k]['source'], r[k]['source']))

    #     ax.legend(title=name,fontsize='xx-small')
        if j == 0 or j == 2: ax.legend(fontsize='xx-small', loc="upper left")
        ax.set_title(name, fontsize='small')
        ax.set_ylabel(name.split('(')[1].split(')')[0], fontsize='small')

        if ax.get_subplotspec().is_last_col():
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        ax.grid()
        ax.set_xticks(np.arange(len(labels))+0.2)
        ax.set_xticklabels(labels, fontsize='small')
        ax.tick_params(axis='x', labelrotation = 45)
        ax.tick_params(axis='y', labelsize='small')

    # save y-lim for SSP1-2.6
    lo, _ = axes.ravel()[2].get_ylim()
    _, hi = axes.ravel()[3].get_ylim()
    axes.ravel()[2].set_ylim([lo, hi])
    axes.ravel()[3].set_ylim([lo, hi])

    plt.tight_layout()

    return f