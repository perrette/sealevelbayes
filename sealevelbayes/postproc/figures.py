#!/usr/bin/env python
from functools import partial
import cloudpickle
import itertools
import copy
import json
import os
from pathlib import Path
import argparse
import subprocess
import tqdm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import xarray as xa
import arviz

# import sys
# sys.path.append('notebooks')

from sealevelbayes.config import logger, get_runpath, get_webpath
from sealevelbayes.datasets.tidegaugeobs import tg_years, tg, region_info, psmsl_ids_to_basin, get_station_name
from sealevelbayes.datasets.satellite import get_satellite_timeseries
from sealevelbayes.datasets.ar6.supp import get_ar6_numbers

import sealevelbayes.datasets.ar6 as ar6
from sealevelbayes.datasets.climate import load_temperature
from sealevelbayes.postproc.serialize import trace_to_json, interpolate_js, resample_as_json, get_model_quantiles
from sealevelbayes.postproc.colors import xcolors, xlabels, sourcecolors, sourcelabels, sourcelabelslocal, basincolors, legend_title_font
from sealevelbayes.postproc.gmslfigure import (make_gmsl_timeseries,
                                               get_table_numbers, make_table_fig, make_table, )

from sealevelbayes.datasets.ar6.tables import ar6_table_9_8_medium_confidence

# from sensitivity import sensitivity_experiments

root = Path(__file__).parents[2]

#########################
# Figures at tide-gauges
#########################

palmer_map = {
#     "NY-ALESUND": "NY-ALESUND", # instead of "Barentsburg" => sacrified in favour of Alaskan glaciers
    "Reykjavik": "REYKJAVIK",
    "Oslo": "OSLO",
    "Cuxhaven": "CUXHAVEN 2",
#     "Newlyn": "NEWLYN", # sacrificed for Grand Isle
    # "Venice": "VENEZIA (S.STEFANO); VENEZIA (ARSENALE); VENEZIA (PUNTA DELLA SALUTE); VENEZIA II", # instead of Stanley II
    "Venice": "VENEZIA (S.STEFANO)", # instead of Stanley II
    # "NewYork": "NEW YORK (THE BATTERY); BERGEN POINT, STATEN IS.",
    "NewYork": "NEW YORK (THE BATTERY)",
    "Grand Isle": "GRAND ISLE",
    "SimonsBay": "SIMONS BAY",
#     "Palermo": "BUENOS AIRES; PALERMO", # same as SimonsBay, with better tide-gauge records, but no other stations in Africa
    "Skagway": "SKAGWAY",
    # "SanFrancisco": "SAN FRANCISCO; ALAMEDA (NAVAL AIR STATION)",
    "SanFrancisco": "SAN FRANCISCO",
    "Lima": "CALLAO 2",
    "ABURATSUBO".capitalize(): "ABURATSUBO", # instead of Mera
    "HALDIA".capitalize(): "HALDIA", # instead of DiamondHarbour
    "PagoPago": "PAGO PAGO",
    "PortLouis": "PORT LOUIS II",
    # "Auckland": "AUCKLAND II; AUCKLAND-WAITEMATA HARBOUR",
    "Auckland": "AUCKLAND II",
}


def get_coordinates_from_psmsl_ids(psmsl_ids):
    from sealevelbayes.datasets.tidegaugeobs import psmsl_filelist_all
    return psmsl_filelist_all.set_index("ID")[["longitude", "latitude"]].loc[psmsl_ids].values.T

def plot_location_shares(ax, years, records, scale=1):
    sumpos = 0
    sumneg = 0

#     for j, source in enumerate(["steric", "glacier", "gis", "ais", "landwater", "total"]):
    for r in records:
        source = r['source']
        med = np.array(r['mean'])*scale
        lo = np.array(r['lower'])*scale
        hi = np.array(r['upper'])*scale

        if source == "total":
#             ax.fill_between(years, lo, hi, color="black", alpha=.2)
            ax.plot(years, med, color=sourcecolors.get(source))
            ix = slice(9, None, 10)
            # ix = slice(None)
            # ax.errorbar(years[ix], med[ix], yerr=np.array([med-lo, hi-med])[:, ix], color=sourcecolors.get(source), label=sourcelabels.get(source, source))
            # ax.plot(years, np.array([lo, hi]).T, color=sourcecolors.get(source), linestyle=":")
            hlw = mpl.rcParams['hatch.linewidth']
            lw = 0.5
            mpl.rcParams['hatch.linewidth'] = lw
            ax.fill_between(years, lo, hi, color=sourcecolors.get(source), hatch='|||', facecolor="none", linewidth=lw, label="Total")
            # mpl.rcParams['hatch.linewidth'] = hlw  # set back the default
        else:
            # positive contributions
            sumpos_prev = sumpos
            sumneg_prev = sumneg

            sumpos = sumpos + np.where(med > 0, med, 0)
            sumneg = sumneg + np.where(med < 0, med, 0)

            ax.fill_between(years, sumpos_prev, sumpos, color=sourcecolors.get(source), edgecolor="none", label=sourcelabels.get(source, source))
            ax.fill_between(years, sumneg_prev, sumneg, color=sourcecolors.get(source), edgecolor="none")


def plot_locations(stations_js, xlim=[1900, 2100], ylim=[-100, 150], squeeze_labels=True, experiment=None, labels={}, ni=4, nj=4, figsize=(9,7), global_js=None, field="rsl", axes=None, add_ipcc=False):

    # featured = {v:k for k,v in palmer_map.items()}
    if axes is None:
        f, axes = plt.subplots(ni, nj, sharex=True, sharey=True, figsize=figsize)
        f.subplots_adjust(hspace=0.05, wspace=0.05, left=0.1, right=0.98, top=0.98, bottom=0.08)
    else:
        axes = np.asarray(axes)

    scale = 1/10
#     mi, ma = -500, 1100
#     mi, ma = -1000, 1500
    mi, ma = ylim
    y1, y2 = xlim

    sat_years, sat_values_all = get_satellite_timeseries([js['station']['Longitude'] for js in stations_js], [js['station']['Latitude'] for js in stations_js])

    if experiment is None:
        experiment = stations_js[0]["experiments"][0]

    for i, js in tqdm.tqdm(enumerate(stations_js)):

        js = interpolate_js(js)

        ax = axes.flatten()[i]

        station = js['station']

        years = np.array(js['years'])
        sources = ["vlm_res", 'gia', "landwater", "ais", "gis", "glacier", "steric", "total"]
        records = list(sorted([r for r in js['records']
                if r['experiment'] == experiment
                and r['field'] == (field if 'PSMSL IDs' in js["station"] else 'global')
                and r['source'] in sources
                and r['diag'] == 'change' ],
            key=lambda r: sources.index(r['source'])))
        assert len(records) <= len(sources)

        if 'mean' not in records[0]:
            logger.warning('mean not found, use median instead')
            records = copy.deepcopy(records)
            for r in records:
                r['mean'] = r['median']

        plot_location_shares(ax, years, records, scale=scale)

        ax.set_xlim(y1, y2)
        ax.set_ylim(mi, ma)

        ax.axvline(2005, color="black", linestyle="--", linewidth=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

        ax.grid()

        # add tide-gauges
        total_line = {r['source']:r for r in records}['total']

        interpolated_tg = np.interp(tg_years, years, total_line['mean'])*scale
        interpolated_sat = np.interp(sat_years, years, total_line['mean'])*scale

        legend_elements = []

        if field == "rsl":
            for id_k, ID in enumerate(station["PSMSL IDs"].split(",") if "PSMSL IDs" in station else []):

                # obs from actual tide gauges
                idx = tg['id'].tolist().index(int(ID))
        #         tg_values = tg['height_corr'][idx]
                tg_values = tg['height'][idx]*scale


                offset = np.nanmean(tg_values - interpolated_tg)

                l, = ax.plot(tg_years, tg_values - offset, 'k', linewidth=1, label='Tide-gauges' if id_k == 0 else None, zorder=999)

        # Also plot satellite minus gps
        # sat_years, [sat_values] = get_satellite_timeseries([js['station']['Longitude']], [js['station']['Latitude']])
        sat_values = sat_values_all[i]*1000*scale  # in mm
        offset = np.nanmean(sat_values - interpolated_sat)
        # offset = np.mean(sat_values) - np.array(total_line['mean'])[sat_years[0]-years[0]:sat_years[0]-years[0]+len(sat_years)].mean()
        gps = js['obs']['obs'][[o['name'] for o in js['obs']['obs']].index('gps')]

        if field == "rsl":
            ax.plot(sat_years, sat_values - offset - (sat_years-sat_years[sat_years.size//2])*gps['obs']*scale, color='darkblue', label='Sat. altimetry minus GPS', zorder=999, linewidth=1)
            ax.plot(sat_years, sat_values - offset - (sat_years-sat_years[sat_years.size//2])*gps['obs']*scale, color='white', zorder=999, linewidth=0.5)

        elif field == "gsl":
            ax.plot(sat_years, sat_values - offset, color='darkblue', label='Sat. altimetry', zorder=999, linewidth=1)
            ax.plot(sat_years, sat_values - offset, color='white', zorder=999, linewidth=0.5)
        elif field == "rad":
            gps_years = np.arange(2000, 2020+1)
            ax.plot(gps_years, (gps_years-2005)*gps['obs']/10, color='darkblue', label='GPS', zorder=999, linewidth=1)
            ax.plot(gps_years, (gps_years-2005)*gps['obs']/10, color='white', zorder=999, linewidth=0.5)

        # IPCC projections
        if add_ipcc and field == "rsl" and xlim[-1] >= 2099 and experiment.startswith("ssp"):
            ipcc_color = 'tab:orange'
            try:
                for id_k, ID in enumerate(station["PSMSL IDs"].split(",") if "PSMSL IDs" in station else []):
                    try:
                        series = ar6.supp.load_location(int(ID), experiment, sources=['total'])['total'].loc[:2100]
                        lo = ar6.supp.load_location(int(ID), experiment, sources=['total'], quantile=0.95)['total'].loc[:2100]
                        hi = ar6.supp.load_location(int(ID), experiment, sources=['total'], quantile=0.05)['total'].loc[:2100]
                    except KeyError as error:
                        print("!! Failed to load AR6 projection for tide-gauge",ID,experiment,'total',":",str(error))
                        continue
                    ax.plot(series.index, series.values*scale, color=ipcc_color, label='IPCC AR6' if id_k == 0 else None,
                        zorder=999, markeredgecolor='black', linestyle='--', linewidth=1)
                    for y in [lo, hi]:
                        ax.plot(series.index, y.values*scale, color=ipcc_color, label=None,
                            zorder=999, markeredgecolor='black', linestyle=':', linewidth=1)

                    # ax.plot(series.index[-1], series.values[-1]*scale, color=ipcc_color, label='IPCC AR6' if id_k == 0 else None,
                    #     zorder=999, marker='*', markeredgecolor='black', clip_on=False, linestyle='--')
                        # zorder=999, marker='*', markeredgecolor='black', clip_on=False if series.values[-1] >= mi and series.values[-1] <= ma else True, linestyle='none')
            except Exception as error:
                logger.warning(str(error))

        nm = station['Station names']
        # ax.set_title(labels.get(nm, nm.capitalize()), fontsize='small')
        label = labels.get(nm, nm.capitalize()) if isinstance(labels, dict) else labels[i]
        ax.text(0.05, 0.95, label,
                fontsize="small",
                # fontproperties=legend_title_font,
            transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', bbox=dict(color="white", alpha=0.5, boxstyle="round"))
    #     ax.set_title(featured.get(nm, nm.capitalize()), pad=-5)

        if i == 13:
            ax.set_xlabel("Years")

        # overlapping years
        if squeeze_labels and i >= 12:
            for j, tick in enumerate(ax.xaxis.get_majorticklabels()):
                if j == len(ax.xaxis.get_majorticklabels())-1:
                    tick.set_horizontalalignment("right")
                if j == 0:
                    tick.set_horizontalalignment("left")

        # overlapping y labels
        if squeeze_labels and np.mod(i, 4) == 0:
            for j, tick in enumerate(ax.yaxis.get_majorticklabels()):
                if j == len(ax.yaxis.get_majorticklabels())-1:
                    tick.set_verticalalignment("top")
                if j == 0:
                    tick.set_verticalalignment("bottom")

        if i == 0:
            leg = ax.legend(loc="upper left", fontsize="xx-small", ncol=1)
            h, hlab = ax.get_legend_handles_labels()
            leg.remove()

        if i == 4:
            if field == "rad":
                ax.set_ylabel("Vertical land motion (cm)")
            elif field == "gsl":
                ax.set_ylabel("Geocentric sea level anomaly (cm)")
            else:
                ax.set_ylabel("Sea level anomaly (cm)")

        ax.tick_params(axis='x', labelsize='small')
        ax.tick_params(axis='y', labelsize='small')

    ax = axes.ravel()[-1]

    # re-order legend labels to have total immediately following other sources
    sourcelabels_ = [sourcelabels.get(s,s) for s in sources]
    h, hlab = zip(*[(hh, lab) for hh, lab in zip(h, hlab) if lab in sourcelabels_] + [(hh, lab) for hh, lab in zip(h, hlab) if lab not in sourcelabels_])
    ax.legend(h, hlab, fontsize='x-small')
    ax.set_axis_off()

    return f, axes


def plot_location_experiments(ax, js, xlim=[1900, 2100], ylim=[-100, 150], experiments=["C1", "C6"],
                               global_js=None, sources = ["total"], ar6_experiments=[], field='rsl', add_ipcc=False,
                               ensemble=False, quantiles=True, trace=None,
                               sat_years=None, sat_values=None, add_satellite=False, add_tidegauge=True, subplot_index=None,):

    if isinstance(js, dict):
        js = interpolate_js(js)
        station = js['station']
        years = np.array(js['years'])
    else:
        station = {"ID": js.station.item(), "PSMSL IDs": str(js.station.item())}
        years = np.array(js.year)

    for x in experiments:
        for source in sources:
            # mid, lo, hi = js_to_array(js, field=field, source=source, experiment=x)/10
            mid, lo, hi = get_model_quantiles(js, x, source, field, quantiles=[.5, .05, .95], diag="change")/10

        if ensemble:
            if trace is None:
                raise ValueError("trace must be provided when ensemble=True")
            if isinstance(trace, xa.DataArray):
                posterior = trace
            else:
                posterior = (trace if isinstance(trace, xa.Dataset) else trace.posterior)["change_rsl_total"]
            array = posterior.sel(experiment=x, station=station['ID']).stack(sample=("chain", "draw")).transpose("year", "sample")/10
            ax.plot(array.year.values, array.isel(sample=np.arange(500)).values,
                    color=xcolors.get(x), alpha=0.01, linewidth=0.5)
            ax.plot([], [], label=xlabels.get(x, x), color=xcolors.get(x))

            if quantiles:
                ax.plot(years, mid, color=xcolors.get(x), lw=1, clip_on=True)
                ax.plot(years, np.array([lo, hi]).T, color=xcolors.get(x), ls='--', lw=1, clip_on=True)
                ax.plot(years, mid, color=xcolors.get(x), lw=.5, clip_on=False, zorder=200)
                ax.plot(years, np.array([lo, hi]).T, color=xcolors.get(x), ls='--', lw=.5, clip_on=False, zorder=200)

        else:
            ax.plot(years, mid, label=xlabels.get(x, x), color=xcolors.get(x))
            ax.fill_between(years, lo, hi, color=xcolors.get(x), alpha=0.3)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.axvline(2005, color="black", linestyle="--", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.grid()

    if add_tidegauge and field == "rsl":
        # add tide-gauges
        interpolated_tg = np.interp(tg_years, years, mid)

        for id_k, ID in enumerate(station["PSMSL IDs"].split(",") if "PSMSL IDs" in station else []):
            # obs from actual tide gauges
            idx = tg['id'].tolist().index(int(ID))
            tg_values = tg['height'][idx]/10
            offset = np.nanmean(tg_values - interpolated_tg)
            l, = ax.plot(tg_years, tg_values - offset, 'k', linewidth=1, label='Tide-gauges' if id_k == 0 else None, zorder=999)

    # Also plot satellite minus gps
    # sat_years, [sat_values] = get_satellite_timeseries([js['station']['Longitude']], [js['station']['Latitude']])
    if add_satellite:
        if sat_years is None or sat_values is None:
            sat_years, sat_values_all = get_satellite_timeseries([js['station']['Longitude']], [js['station']['Latitude']])
            sat_values = sat_values_all[0] * 1000 # standard mm units

        sat_values /= 10  # in cm

        interpolated_sat = np.interp(sat_years, years, mid.values)

        # offset = np.mean(sat_values) - np.array(total_line['mean'])[sat_years[0]-years[0]:sat_years[0]-years[0]+len(sat_years)].mean()
        gps = js['obs']['obs'][[o['name'] for o in js['obs']['obs']].index('gps')]

        if field == "rsl":
            offset = np.nanmean(sat_values - interpolated_sat)
            ax.plot(sat_years, sat_values - offset - (sat_years-sat_years[sat_years.size//2])*gps['obs']/10, color='darkblue', label='Sat. altimetry minus GPS', zorder=999, linewidth=1)
            ax.plot(sat_years, sat_values - offset - (sat_years-sat_years[sat_years.size//2])*gps['obs']/10, color='white', zorder=999, linewidth=0.5)

        elif field == "rad":
            gps_years = np.arange(2000, 2020+1)
            ax.plot(gps_years, (gps_years-2005)*gps['obs']/10, color='darkblue', label='GPS', zorder=999, linewidth=1)
            ax.plot(gps_years, (gps_years-2005)*gps['obs']/10, color='white', zorder=999, linewidth=0.5)


    # IPCC projections
    if add_ipcc and field == "rsl" and xlim[-1] >= 2099:
        for x in ar6_experiments:
            with ar6.supp.open_slr_regional("total", x)["sea_level_change"] as ar6ds:
                for id_k, ID in enumerate(station["PSMSL IDs"].split(",") if "PSMSL IDs" in station else []):
                    try:
                        series = ar6ds.loc[:, :, int(ID)]
                    except KeyError as error:
                        print("!! Failed to load AR6 projection for tide-gauge",ID,x,'total',":",str(error))
                        continue
                    lo, mid, hi = series.loc[[.05, .5, .95], 2100].values / 10
                    ax.plot([2100, 2100], [lo, hi], color=xcolors.get(x), lw=2, label=f"AR6 {xlabels.get(x, x)}", clip_on=False)
                    ax.plot([2095, 2105], [mid, mid], color=xcolors.get(x), lw=2, clip_on=False)

    if field == "rad":
        ax.set_ylabel("Vertical land motion (cm)")
    else:
        ax.set_ylabel("Sea level anomaly (cm)")

    ax.tick_params(axis='x', labelsize='small')
    ax.tick_params(axis='y', labelsize='small')


    # re-order legend labels to have total immediately following other sources
    sourcelabels_ = [sourcelabels.get(s,s) for s in sources]
    h, hlab = ax.get_legend_handles_labels()
    h, hlab = zip(*[(hh, lab) for hh, lab in zip(h, hlab) if lab in sourcelabels_] + [(hh, lab) for hh, lab in zip(h, hlab) if lab not in sourcelabels_])
    ax.legend(h, hlab, fontsize='x-small', loc="upper left")

    ax.set_xlabel("Years")

    return


def plot_location_proj2100_ar6(ax, js=None, xlim=None, ylim=None, experiments=["ssp126", "ssp185"],
                               global_js=None, sources = ["steric", "glacier", "gis", "ais", "landwater", "vlm", "total"], ar6_experiments=None,
                               field='rsl', add_ipcc=True, stacked=False, subplot_index=None, show_experiment_text=False,
                               quantiles=True, posterior=None, legend_title=None, show_likely_range=True, show_ar6_table_98=False, diag="proj2100"):

    station = (js or {}).get("station", {})
    texttransform = blended_transform_factory(ax.transData, ax.transAxes)

    if js is None:
        assert posterior is not None, "Please provide either `js` (json-serialized) or a valid trace posterior object."


    # one station may have more than one PSMSL ID attached (legacy -- not the case anymore)
    IDs = [int(ID) for ID in station["PSMSL IDs"].split(",")] if "PSMSL IDs" in station else [None]

    if ar6_experiments is None:
        ar6_experiments = experiments

    counter = 0
    xcounter = 0
    for i, (x, x2) in enumerate(zip(experiments, ar6_experiments)):
        ID = IDs[0]
        show_likely_range_ = show_likely_range if show_likely_range is not None else ID is None

        if len(experiments) > 1:
            logger.info(f"{xlabels.get(x, x)} for {i+1}/{len(experiments)}")
        if stacked:
            raise NotImplementedError("stacked=True not implemented yet")

        else:
            for j, source in enumerate(sources):
                # logger.info(f"{source} for {j}/{len(sources)}")

                color = sourcecolors.get(source)

                # our model bars
                qlevs = [.5, .05, .95, .167, .833]
                mid, lo, hi, lo2, hi2 = map(lambda x: x/10, get_model_quantiles(js, x, source, field, diag, quantiles=qlevs))

                ax.bar(counter, mid, yerr=[[mid-lo], [hi-mid]], color=color, label=sourcelabels.get(source, source) if i == 0 else None, width=0.8, error_kw=dict(elinewidth=1))
                if show_likely_range_: ax.bar(counter, mid, yerr=[[mid-lo2], [hi2-mid]], color=color, label=None, width=0.8, error_kw=dict(elinewidth=2))
                counter += 1

                # AR6 bars
                if add_ipcc:
                    for (mid, lo, hi, lo2, hi2) in get_ar6_numbers(x, source, IDs, years=2100, quantiles=qlevs, rate=diag.startswith("rate")).transpose("locations", ...).values/10:
                        ax.bar(counter, mid, yerr=[[mid-lo], [hi-mid]], color=color, label=f"IPCC AR6 5-95th" if i == 0 and j == len(sources)-1 else None, width=0.8, hatch='//', edgecolor="black", linewidth=0.5, error_kw=dict(elinewidth=1))
                        if show_likely_range_: ax.bar(counter, mid, yerr=[[mid-lo2], [hi2-mid]], color=color, label=None, width=0.8, hatch='//', edgecolor="black", linewidth=0.5, error_kw=dict(elinewidth=2))
                        # if ID is None and show_ar6_table_98:
                        # AR6 table
                        if show_ar6_table_98 and not ID:
                            ax.plot([counter]*3, np.array(ar6_table_9_8_medium_confidence[x][source])*100, 'o', label="AR6 Table 9.8" if i == 0 and j == len(sources) -1 else None, color=color, markersize=3, zorder=1000)
                        counter += 1


            if i == len(experiments) - 1 and j == len(sources) - 1 and show_likely_range_:
                ax.plot([], [], color="black", label="Range 17-83th", linewidth=1)
                ax.plot([], [], color="black", label="Range 5-95th", linewidth=2)

            if show_experiment_text:
                vpad = 0.05 # axis units
                hpad = 1  # data units
                if i == 1 and len(experiments) == 2:
                    ax.text(xcounter + hpad, 1 - vpad, xlabels.get(x, x), fontsize='small', verticalalignment='top',
                            horizontalalignment='left', transform=texttransform)
                    xcounter = counter

                else:

                    ax.text(counter - hpad, 1 - vpad, xlabels.get(x, x), fontsize='small', verticalalignment='top',
                            horizontalalignment='right', transform=texttransform)

            if i + 1 < len(experiments):
                ax.axvline(counter, color="black", linestyle="--", linewidth=0.5)

            xcounter = counter
            counter += 1

    # remove x ticks
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    # ax.text(xc-0.01, 0.95, )
    # ax.axvline(xc, color="black", linestyle="--", linewidth=0.5)


    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # add y-axis grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    if field == "rad":
        ax.set_ylabel("Vertical land motion (cm)")
    else:
        ax.set_ylabel("Sea level anomaly (cm)")

    # ax.tick_params(axis='x', labelsize='small')
    # ax.tick_params(axis='y', labelsize='small')

    # re-order legend labels to have total immediately following other sources
    # sourcelabels_ = [sourcelabels.get(s,s) for s in sources]
    # h, hlab = ax.get_legend_handles_labels()
    # h, hlab = zip(*[(hh, lab) for hh, lab in zip(h, hlab) if lab in sourcelabels_] + [(hh, lab) for hh, lab in zip(h, hlab) if lab not in sourcelabels_])
    # ax.legend(h, hlab, fontsize='x-small', loc="upper left")
    ax.legend(loc="upper left", fontsize='x-small', title=legend_title or " | ".join([xlabels.get(x,x) for x in experiments]))
    ax.axhline(0, color='k', lw=0.5, ls='-')

    return

def plot_global_proj2100_ar6(global_js, experiments=["ssp126", "ssp585"], sources = ['ais', 'gis', 'glacier', 'landwater', 'steric', 'total'], **kwargs):
    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_location_proj2100_ar6(ax, global_js, experiments=experiments, sources=sources, field='global', show_experiment_text=True, legend_title="Global SLR")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def plot_locations_handler(caller, stations_js, squeeze_labels=True, labels={}, ni=4, nj=4, figsize=(9,7), axes=None, sharex=True, sharey=True, move_legend=True, **kw):

    # featured = {v:k for k,v in palmer_map.items()}
    if axes is None:
        f, axes = plt.subplots(ni, nj, sharex=sharex, sharey=sharey, figsize=figsize)
        f.subplots_adjust(hspace=0.05, wspace=0.05, left=0.1, right=0.98, top=0.98, bottom=0.08)
    else:
        axes = np.asarray(axes)

    if isinstance(stations_js, list):
        station_data_list = stations_js
    else:
        data_ = stations_js
        stations_js = [{"station": {"ID": station, "PSMSL IDs":str(station), "Station names": get_station_name(station)}} for station in data_.station.values]
        station_data_list = [data_.sel(station=station) for station in data_.station.values]

    for i, (data, js) in enumerate(zip(station_data_list, stations_js)):

        ax = axes.flatten()[i]
        spec = ax.get_subplotspec()
        caller(ax, data, subplot_index=i, **kw)
        station = js['station']

        nm = station['Station names']

        logger.info(f"{nm} for {i+1}/{len(stations_js)}")

        # ax.set_title(labels.get(nm, nm.capitalize()), fontsize='small')
        label = labels.get(nm, nm.capitalize()) if isinstance(labels, dict) else labels[i]
        ax.text(0.05, 0.95, label,
            fontsize='small',
            # fontproperties=legend_title_font,
            transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', bbox=dict(color="white", alpha=0.5, boxstyle="round"))
    #     ax.set_title(featured.get(nm, nm.capitalize()), pad=-5)

        if i != 13:
            # ax.set_xlabel("Years")
            ax.set_xlabel("")

    #     if ax.is_last_row:
        if squeeze_labels and i >= 12:
            for j, tick in enumerate(ax.xaxis.get_majorticklabels()):
                if j == len(ax.xaxis.get_majorticklabels())-1:
                    tick.set_horizontalalignment("right")
                if j == 0:
                    tick.set_horizontalalignment("left")


        # overlapping y labels
        if squeeze_labels and np.mod(i, 4) == 0:
            for j, tick in enumerate(ax.yaxis.get_majorticklabels()):
                if j == len(ax.yaxis.get_majorticklabels())-1:
                    tick.set_verticalalignment("top")
                if j == 0:
                    tick.set_verticalalignment("bottom")

        # if i == 0:
        #     leg = ax.get_legend()
        #     h, hlab = ax.get_legend_handles_labels()
        #     leg.remove()

        if i != 4:
            ax.set_ylabel("")

        # remove the legend -> we'll have that in the last, empty subplot
        if move_legend:
            leg = ax.get_legend()
            h, hlab = ax.get_legend_handles_labels()
            if leg is not None:
                legend_title = leg.get_title().get_text()
                leg.remove()
            else:
                legend_title = None

    # use the last axis for the legend
    if move_legend:
        ax = axes.ravel()[-1]
        ax.set_axis_off()
        ax.legend(h, hlab, fontsize='x-small', title=legend_title, frameon=False)

    return f, axes



plot_locations_experiments = partial(plot_locations_handler, plot_location_experiments)

def plot_locations_proj2100_ar6(stations_js, **kwargs):
    kwargs.setdefault('sharey', False)
    f, axes = plot_locations_handler(plot_location_proj2100_ar6, stations_js, **kwargs)

    for ax in axes.flatten():
        spec = ax.get_subplotspec()

        if not spec.is_first_col() and not spec.is_last_col():
            ax.set_yticklabels("")
            # ax.set_yticks([])
            ax.tick_params(axis='y', direction='in')

        elif spec.is_last_col():
            # ticks on the right side
            ax.yaxis.tick_right()

    f.tight_layout()

    return f, axes



def plot_locations_experiments_ensemble(trace, stations_js, **kwargs):
    kwargs.setdefault('quantiles', False)
    return plot_locations_experiments(stations_js, ensemble=True, trace=trace, **kwargs)


def _plotbar(ax, x, values, color, label=None, show_likely_range=True, width=0.8, edgecolor="black", linewidth=0.5, bottom=None, errors=True, zorder=None, **kw):
    """plot error bars based on spatially distributed values for plot_2050_vs_2100_uncertainty_ar6
    """
    # values = values ** 2
    common_kw = dict(color=color, width=width, edgecolor=edgecolor, linewidth=linewidth, **kw)
    mid, lo, hi, lo2, hi2 = np.percentile(values, [50, 5, 95, 16.7, 83.3], axis=-1)
    errorbarkw = dict(fmt='none', ecolor='black')
    # if bottom is not None:
    #     mid_b, lo_b, hi_b, lo2_b, hi2_b = np.percentile(bottom, [50, 5, 95, 16.7, 83.3], axis=-1)
    #     bottom = mid_b
    # else:
    #     bottom = 0

    ax.bar(x, mid, bottom=bottom, label=label, zorder=zorder, **common_kw)
    if errors:
        ax.errorbar(x, mid, yerr=[[mid-lo], [hi-mid]], elinewidth=1, capsize=5, **errorbarkw)
        # ax.errorbar(x, mid, yerr=[mid-lo, hi-mid], elinewidth=1, **errorbarkw)
        if show_likely_range:
            ax.errorbar(x, mid, yerr=[[mid-lo2], [hi2-mid]], elinewidth=5, capsize=0, **errorbarkw)


def getdata(posterior, diag, field, source, experiment=None, variable=None):
    if variable is None:
        variable = f"{diag}_{field}_{source}"
    if variable not in posterior:
        diagtype = diag[:4]
        try:
            year = int(diag[4:])
        except ValueError:
            print(f"Invalid diag: {diag}")
            raise
        if diagtype == "proj":
            variable = f"change_{field}_{source}"
        elif diagtype == "rate":
            variable = f"rate_{field}_{source}"
        else:
            raise ValueError(f"Invalid diag: {diag}")
        data = posterior[variable].sel(year=year)
    else:
        data = posterior[variable]

    if experiment is not None:
        data = data.sel(experiment=experiment)
    return data


def plot_2050_vs_2100_uncertainty_ar6(ax, posterior, xlim=None, ylim=None, experiments=["ssp126", "ssp185"],
                               source="total", diags=["proj2050", "proj2100"],
                               field='rsl',
                               legend_title=None, show_likely_range=True, qlevs=None, uncertaintylevel="likely",
                               oceandyn_diags=None):

    texttransform = blended_transform_factory(ax.transData, ax.transAxes)

    if qlevs is None:
        if uncertaintylevel == "likely":
            qlevs = [.167, .833]

        elif uncertaintylevel == "very_likely":
            qlevs = [.05, .95]

        else:
            raise ValueError("uncertaintylevel must be either 'likely' or 'very_likely'")

    ticks = []

    counter = 0
    for i, diag in enumerate(diags):

        year = int(diag[4:8])  # year proj2100 or proj2050
        rate = diag.startswith("rate")

        # for j, source in enumerate(sources):
        for j, experiment in enumerate(experiments):
            # logger.info(f"{source} for {j}/{len(sources)}")

            color = xcolors.get(experiment)

            # our model bars
            # qlevs = [.5, .05, .95, .167, .833]
            # likely range

            v = getdata(posterior, diag, field, source, experiment) / 10

            if "chain" in posterior.dims:
                dim=["chain", "draw"]
            elif "sample" in posterior.dims:
                dim=["sample"]
            else:
                dim=None

            # if oceandyn_diags is not None:
            #     odyndiag = oceandyn_diags[diag].squeeze() / 10
            #     v = v + odyndiag

            if dim is not None:
                los, his = v.quantile(qlevs, dim=dim).values
            else:
                los, his = v.sel(quantile=qlevs).values
            error = his - los
            _plotbar(ax, counter, error, color=color, label=f"This study {xlabels.get(experiment, experiment)}" if i == 0 else None,
                     errors=oceandyn_diags is None)

            ticks.append(counter+0.5)

            if oceandyn_diags is not None:
                odyndiag = oceandyn_diags[diag].squeeze() / 10
                v = v + odyndiag
                los2, his2 = v.quantile(qlevs, dim=dim).values
                error2 = his2 - los2
                # print(error2.shape)
                # _plotbar(ax, counter, error2 - error, bottom=error, color="tab:blue", label=f"Ocean Dyn." if i == 0 else None)
                _plotbar(ax, counter, error2, color="tab:blue", label=f"This Study incl. Ocean Dyn." if i == 0 else None, alpha=0.5, hatch='.....', edgecolor="black", linewidth=0.5, zorder=0)

            counter += 1

            # AR6 bars
            IDs = posterior['station'].values
            los, his = get_ar6_numbers(experiment, source, IDs, years=year, quantiles=qlevs, rate=rate).transpose("locations", ...).values.T/10
            error = his - los
            _plotbar(ax, counter, error, color=color, label=f"IPCC AR6 {xlabels.get(experiment, experiment)}" if i == 0 and j == 0 else None, hatch='////')
            counter += 1


        if i == len(experiments) - 1 and j == len(experiments) - 1:
            if show_likely_range: ax.plot([], [], color="black", label="Range 17-83th", linewidth=1)
            ax.plot([], [], color="black", label="Range 5-95th", linewidth=2)

        # show diags
        vpad = 0.05 # axis units
        hpad = 1  # data units
        # label = xlabels.get(experiment, experiment)
        label = year
        # if i == 1 and len(experiments) == 2:
        #     ax.text(xcounter + hpad, 1 - vpad, label, fontsize='small', verticalalignment='top',
        #             horizontalalignment='left', transform=texttransform)
        #     xcounter = counter

        # else:

        #     ax.text(counter - hpad, 1 - vpad, label, fontsize='small', verticalalignment='top',
        #             horizontalalignment='right', transform=texttransform)

        if i + 1 < len(experiments):
            ax.axvline(counter, color="black", linestyle="--", linewidth=0.5)

        xcounter = counter
        counter += 1

    # remove x ticks
    # ax.set_xticks([])
    # ax.set_xticklabels([])
    years = [int(diag[4:8]) for diag in diags]
    ax.set_xticks(ticks)
    ax.set_xticklabels(years)

    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    # ax.text(xc-0.01, 0.95, )
    # ax.axvline(xc, color="black", linestyle="--", linewidth=0.5)


    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # add y-axis grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    if rate:
        ax.set_ylabel(f"Local {(qlevs[1] - qlevs[0])*100:.0f}% range of sea level rate (mm/yr)")
    else:
        ax.set_ylabel(f"Local {(qlevs[1] - qlevs[0])*100:.0f}% range of sea level change (cm)")

    # ax.tick_params(axis='x', labelsize='small')
    # ax.tick_params(axis='y', labelsize='small')

    # re-order legend labels to have total immediately following other sources
    # sourcelabels_ = [sourcelabels.get(s,s) for s in sources]
    # h, hlab = ax.get_legend_handles_labels()
    # h, hlab = zip(*[(hh, lab) for hh, lab in zip(h, hlab) if lab in sourcelabels_] + [(hh, lab) for hh, lab in zip(h, hlab) if lab not in sourcelabels_])
    # ax.legend(h, hlab, fontsize='x-small', loc="upper left")
    ax.legend(loc="upper left", fontsize='x-small', title=legend_title or " | ".join([xlabels.get(x,x) for x in experiments]))
    ax.axhline(0, color='k', lw=0.5, ls='-')

    return

def get_proj(x, year, rate=False):
    if rate:
        x = np.diff(x, axis=0)
    return x[year-1995]-x[1995-1995:2014-1995+1].mean(axis=0)

def get_oceandyn_diags(odyn, years, rng=None, shp=(4, 1000, 1), rate=False, proj=True):

    if rng is None:
        rng = np.random.default_rng(2135)

    oceandyn = {}
    for year in years:
        if proj: oceandyn[f"proj{year}"] = rng.normal(scale=odyn.std(odyn.samples(partial(get_proj, year=year))), size=shp + (578,))
        if rate: oceandyn[f"rate{year}"] = rng.normal(scale=odyn.std(odyn.samples(partial(get_proj, year=year, rate=True))), size=shp + (578,))

    return oceandyn


def xt_plot_locations(tr, psmsl_ids, experiment="ssp370", field="rsl",
                      sources=["vlm_res", 'gia', "landwater", "ais", "gis", "glacier", "steric", "total"],
                      split_resampling=False, stations_js=None, **kw):
    """Wrapper of plot_location_shape that takes ExperimentTrace as input
    """
    if stations_js is None:
        stations_js = resample_as_json(tr, psmsl_ids, [experiment], [field], sources=sources, split_resampling=split_resampling)
    return stations_js, plot_locations(stations_js, experiment=experiment, field=field, **kw)


def xt_plot_location(tr, psmsl_id, ax=None, **kw):
    """Wrapper of plot_location_shape that takes ExperimentTrace as input
    """
    if ax is None: ax = plt.gca()

    return xt_plot_locations(tr, [psmsl_id], axes=[ax], **kw)


def xt_plot_locations_experiments(tr, psmsl_ids, experiments, field="rsl", source="total", split_resampling=False, stations_js=None, **kw):
    """Wrapper of plot_location_shape that takes ExperimentTrace as input
    """
    if stations_js is None:
        stations_js = resample_as_json(tr, psmsl_ids, experiments, [field], [source], split_resampling=split_resampling)
    return plot_locations_experiments(stations_js, experiments=experiments, field=field, sources=[source], **kw)


def xt_plot_location_experments(tr, psmsl_id, experiments, ax=None, **kw):
    """Wrapper of plot_location_shape that takes ExperimentTrace as input
    """
    if ax is None: ax = plt.gca()
    return xt_plot_locations_experiments(tr, [psmsl_id], experiments, axes=[ax], **kw)


sort_basins = [
 # 'Subpolar North Atlantic',
 # 'Subpolar North Atl. East',
 # 'Subpolar North Atl. West',
 'Northeast Atlantic',
 'Mediterranean',
 'Northwest Atlantic',
 # 'Subtropical North Atlantic',
 'South Atlantic',
 'East Pacific',
 'Northwest Pacific',
 'Indian Ocean - South Pacific',
]


def add_coastline_to_json(stations_js):
    for r, basin in zip(stations_js, psmsl_ids_to_basin([r['station']['ID'] for r in stations_js])):
        r['station']['coastline'] = basin

def sort_stations(stations, sort_key=None, lat_getter=lambda r : r['Latitude'], coast_getter=lambda r : r['coastline'], coast_order=sort_basins):
    # Sort the stations by basin and latitude
    if sort_key is None: sort_key = lambda r: (coast_order.index(coast_getter(r)), -lat_getter(r))
    return sorted(stations, key=sort_key)

def argsort_stations(stations, sort_key=None, lat_getter=lambda r : r['Latitude'], coast_getter=lambda r : r['coastline'], coast_order=sort_basins):
    if sort_key is None: sort_key = lambda r: (coast_order.index(coast_getter(r)), -lat_getter(r))
    return np.array([i for i,x in sort_stations(enumerate(stations), sort_key=lambda r: sort_key(r[1]))])

def argsort_stations_by_id(ids, lons=None, lats=None):
    if lons is None:
        lons, lats = get_coordinates_from_psmsl_ids(ids)
    stations_js_bar = [{"station": {"ID": ID, "Longitude": lon, "Latitude": lat}} for ID, lon, lat in zip(ids, lons, lats)]
    add_coastline_to_json(stations_js_bar)
    bar = BarPlot(stations_js_bar)
    sorted_ids = [s['station']['ID'] for s in bar.stations_js]
    argsort_ids = np.searchsorted(ids, sorted_ids)
    return argsort_ids

def argsort_stations_by_trace(trace):
    data = trace if isinstance(trace, xa.Dataset) else trace.constant_data
    lons = data.lons.values
    lats = data.lats.values
    psmsl_ids = data.station.values
    return argsort_stations_by_id(psmsl_ids, lons, lats)


def _get_obs_legend_title(name, panel=""):
    return {
        "tidegauge": f"{panel}Tide gauges 1900-2018*",
        "satellite": f"{panel}Satellite 1993-2019",
        "gps": f"{panel}GPS 2000-2020",
    }.get(name, name)

def _plot_obs_from_records(stations_js, name, panel="", label=None, x=None, **kw):
    if x is None:
        x = np.arange(len(stations_js))
    if label is None:
        label = _get_obs_legend_title(name, panel)

    getobs = lambda js: js['obs']['obs'][[o['name'] for o in js['obs']['obs']].index(name)]
    # getpp = lambda js: [r for r in js['posterior_predictive'] if r["name"] == name][0]
    getpp = lambda js: [r for r in js['posterior_predictive'] if r["name"] == f"{name}_obs"][0]

    if "posterior_predictive" in stations_js[0]:
        y = y_pp = np.array([getpp(js)['median'] for js in stations_js])
        ylo = ylo_pp = np.array([o['lower'] for js in stations_js for o in [getpp(js)]])
        yhi = yhi_pp = np.array([o['upper'] for js in stations_js for o in [getpp(js)]])
        post_pred = (y, ylo, yhi)
    else:
        post_pred = None

    y = np.array([getobs(js)['median'] for js in stations_js])
    ylo = np.array([o['lower'] for js in stations_js for o in [getobs(js)]])
    yhi = np.array([o['upper'] for js in stations_js for o in [getobs(js)]])
    post = (y, ylo, yhi)

    obs = np.array([getobs(js)['obs'] for js in stations_js])

    return _plot_obs_from_arrays(x, post, post_pred, obs, label=label, **kw)

def _show_bad(ax, x, Y, ylo_pp, yhi_pp, scale_x, **kw):
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
    low_bias = yhi_pp < Y
    hi_bias = ylo_pp > Y
    bad = hi_bias | low_bias
    for i, ix in enumerate(np.where(bad)[0]):
        xx = x[ix]
        if hi_bias[ix]:
            l = ax.vlines(xx, Y[ix], ylo_pp[ix], alpha=.4, **kw)
        else:
            l = ax.vlines(xx, Y[ix], yhi_pp[ix], alpha=.4, **kw)
        if i == 0: l.set_label('Obs out of 90% PP range')

    ax.plot(x[bad], Y[bad], 'o', markeredgecolor=kw.get("color"), markerfacecolor="none", markersize=scale_x, markeredgewidth=0.25)


def _plot_obs_from_arrays(x, post, post_pred=None, obs=None, label=None,
                          post_label="Model posterior", obs_label="Obs",
                          post_pred_label="Model posterior predictive", units="mm/yr",
                          legend_kw={}, ax=None, show_bad=True):
    if ax is None: ax = plt.gca()

    thres = 100

    if x.size < thres:
        scale_x = thres/x.size
        scale_y = 1
    else:
        scale_x = 1
        scale_y = 1

    ax.set_ylabel(units)
    ax.plot(x, [0 for _ in x], 'k-', linewidth=0.5)

    # Model
    if post_pred is not None:
        y, ylo, yhi = y_pp, ylo_pp, yhi_pp = post_pred
        ax.errorbar(x, y, label=post_pred_label, color='gray', yerr=[y-ylo, yhi-y], alpha=0.4, fmt='none', linewidth=scale_x)

    # ax.errorbar(x, y, label='Model posterior', color='steelblue', yerr=[y-ylo, yhi-y], alpha=0.6, fmt='.', markersize=1*scale_x, linewidth=scale_x)
    if post is not None:
        y, ylo, yhi = post
        ax.errorbar(x, y, label=post_label, color='steelblue', yerr=[y-ylo, yhi-y], alpha=0.6, fmt='none', linewidth=scale_x)
        ax.plot(x, y, '.', color='steelblue', markersize=.8*scale_x)

    # Obs
    if obs is not None:
        Y = obs
        ax.plot(x, Y, '.', label=obs_label, color='black', markersize=.8*scale_x)

    # Hightlight when the 90% error ranges do not intersect
    if show_bad:
        _show_bad(ax, x, Y, ylo_pp, yhi_pp, scale_x, linewidth=scale_x, color="tab:red")

    h, labs = ax.get_legend_handles_labels()

    if post_pred is not None and post is not None and obs is not None:
        hlabs = list(zip(h, labs))
        h, labs = zip(*[hlabs[i] for i in ([0, 3, 2, 1] if show_bad else [0, 2, 1])])

    l = ax.legend(h, labs, fontsize='x-small', title=label, **legend_kw)
    l._legend_box.align = "left"


def plot_obs_from_dataset(idata, names=["tidegauge", "gps", "satellite"],
                          sort_stations=True, axes=None, labels=None, featured_locations=None, **kw):

    psmsl_ids = next(iter(idata.values())).station.values

    if sort_stations:
        if "constant_data" in idata:
            argsort_ids = argsort_stations_by_trace(idata["constant_data"])
        else:
            argsort_ids = argsort_stations_by_id(psmsl_ids)
        psmsl_ids = psmsl_ids[argsort_ids]
        # idata = idata.sel(station=psmsl_ids)
        idata = idata.isel(station=argsort_ids)

    x = np.arange(psmsl_ids.size)

    if axes is None:
        n = len(names)
        f, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 2.5*n), squeeze=False)
        f.subplots_adjust(hspace=0.05, wspace=0.05, left=0.1, right=0.98, top=0.98, bottom=0.08)
    else:
        f = axes[0].figure

    def _get_quantiles(post, quantiles=[0.5, 0.05, 0.95]):
        sample_dims = [k for k in post.dims if k in ["draw", "chain", "sample"]]
        return post.quantile(quantiles, dim=sample_dims).values

    for i, name in enumerate(names):
        post = _get_quantiles(idata["posterior"][name]) if "posterior" in idata else None
        post_pred = _get_quantiles(idata["posterior_predictive"][name + "_obs"]) if "posterior_predictive" in idata and name + "_obs" in idata['posterior_predictive'] else None
        obs = idata["constant_data"][name + "_mu"].values if "constant_data" in idata else None
        ax = axes.flat[i]
        if labels is None:
            label = _get_obs_legend_title(name, "abcdef"[i])
        else:
            label = labels[i]

        _plot_obs_from_arrays(x, post, post_pred, obs, label=label, ax=ax, **kw)

    bp = BarPlot.from_psmsl_ids(psmsl_ids, featured_stations=featured_locations)
    for i, ax in enumerate(axes.flat):
        bp.set_axis_layout(ax, featured=True)

    return f, axes


class BarPlot:

    def __init__(self, stations_js, global_js=None, featured_stations=None,
            coord_getter=lambda r: (r['station']['Longitude'], r['station']['Latitude']),
            coast_getter=lambda r: r['station']['coastline'],
            coast_order=sort_basins, basins_cls=None):

        self.stations_js = stations_js
        self.global_js = global_js

        self.featured_stations = featured_stations or {}
        self.featured_color = "black"
        self.lat_getter = lambda r: coord_getter(r)[1]

        self.coast_getter = coast_getter
        self.coast_order = coast_order

        n = len(stations_js)
        self.scale = 1/10
        self.x = np.arange(n)

        if 'coastline' not in self.stations_js[0]['station']:
            add_coastline_to_json(stations_js)

        self._sort()

    @classmethod
    def from_psmsl_ids(cls, psmsl_ids, lons=None, lats=None, **kw):
        if lons is None:
            lons, lats = get_coordinates_from_psmsl_ids(psmsl_ids)
        stations_js = [{"station": {"ID": ID, "Longitude": lon, "Latitude": lat}} for ID, lon, lat in zip(psmsl_ids, lons, lats)]
        return cls(stations_js, **kw)

    def _sort(self):
        self.stations_js = sort_stations(self.stations_js, lat_getter=self.lat_getter, coast_getter=self.coast_getter, coast_order=self.coast_order)
        self.basins, self.bcounts = zip(*[(basin, len(list(group))) for basin, group in itertools.groupby(self.stations_js, key=self.coast_getter)])
        self.counts = np.cumsum(self.bcounts)


    def set_axis_layout(self, ax, featured=False, featured_labels="first row"):
        for c in self.counts[:-1]:
            ax.axvline(c-0.5, color='gray', linewidth=0.5)

        ax.set_xlim(self.x[0]-0.5, xmax=self.x[-1]+0.5)

        # y-grid
        # ax.yaxis.grid()

        # bad = (ylo > Yhi) | (yhi < Ylo)
        # ax.bar(x[bad], height=ylim[1]-ylim[0], bottom=ylim[0], color='tab:red', alpha=0.1, width=0.8)
        xx = []
        if featured:
            for i, js in enumerate(self.stations_js):
                label =  self.featured_stations.get(js["station"].get("Station names"))
                if label is None: continue
                xx.append(i)
                # l = ax.axvline(i, color='tab:red', linewidth=0.5, linestyle=':')
                # l = ax.axvline(i, color=self.featured_color, alpha=0.1, linewidth=1.5)
                # l = ax.axvline(i, color=self.featured_color, alpha=0.3, linestyle='--', linewidth=0.5)
                l = ax.axvline(i, color=self.featured_color, alpha=0.1, linestyle='-', linewidth=0.5)
                # l = ax.axvline(i, 0, 0.01, color=self.featured_color, linestyle='-')
                # l = ax.arrow(xx, 0, 0, 0.01*(ylim[1]-ylim[0]), color=self.featured_color, linewidth=0.5, linestyle='-')
                # if i == 0: l.set_label('mismatch')
                lo, hi = ax.get_ylim()
                # bbox = dict(edgecolor=self.featured_color, facecolor="white", boxstyle="round", linewidth=0.5)
                # bbox = dict(edgecolor='none', facecolor="white", boxstyle="round", linewidth=0.5)
                r, g, b, a = mpl.colors.to_rgba(self.featured_color)
                # bbox = dict(edgecolor=(r, g, b, 0.1), facecolor="white", boxstyle="round", linewidth=1.5)
                bbox = dict(edgecolor=(r, g, b, 0.1), facecolor="white", boxstyle="rarrow", linewidth=.5)
                # bbox = None
                if (featured_labels == "first row" and ax.get_subplotspec().is_first_row()):
                    line_height = 1
                    offset = line_height * 1.3
                    if len(xx) == 2:
                        offset = 0
                    elif len(xx) == 7:
                        offset = 0
                    elif len(xx) == 8:
                        offset *= 2
                    elif len(xx) == 10:
                        offset = 0
                    elif len(xx) == 14:
                        offset = 0
                    ax.text(i-4, lo+offset, label, color=self.featured_color, fontsize="xx-small", bbox=bbox, horizontalalignment="right", verticalalignment="bottom")
                elif featured_labels == "top":
                    bbox = dict(edgecolor=self.featured_color, facecolor="white", boxstyle="round", linewidth=0.5, alpha=0.5)
                    ax.text(i, hi-(hi-lo)*0.025, label.split('.')[0], color=self.featured_color, fontsize="xx-small", bbox=bbox, horizontalalignment="center", verticalalignment="top")
                elif featured_labels == "bottom":
                    bbox = dict(edgecolor=self.featured_color, facecolor="white", boxstyle="round", linewidth=0.5, alpha=0.5)
                    ax.text(i, lo +(hi-lo)*0.025, label.split('.')[0], color=self.featured_color, fontsize="xx-small", bbox=bbox, horizontalalignment="center", verticalalignment="bottom")
                elif featured_labels == "ticks":
                    bbox = dict(edgecolor=self.featured_color, facecolor="white", boxstyle="round", linewidth=0.5, alpha=0.5)
                    ax.text(i, lo - (hi-lo)*0.025, label.split('.')[0], color=self.featured_color, fontsize="xx-small", bbox=bbox, horizontalalignment="center", verticalalignment="top")
                    ax.set_xticks([], [])


        # xbad = x[bad]
            # l = ax.arrow(xbad, np.zeros_like(xbad)+ylim[0], np.zeros_like(xbad)+0, np.zeros_like(xbad)+(ylim[1]-ylim[0])*0.01, color=self.featured_color, linewidth=0.5)
            # ax.set_xticks([], ['' for x in bad])

        # ax.set_xticks(xx, xx, color="tab:red", fontsize=4, rotation=90)

        # ax.xaxis.set_tick_params(length=0, direction="in")



    def plot_obs_old(self, name, label=None, ax=None, panel="", legend_kw={}):

        if ax is None: ax = plt.gca()
        x = self.x
        stations_js = self.stations_js
        if label is None:
            label = {
                "tidegauge": f"{panel}Tide gauges",
                "satellite": f"{panel}Satellite 1993-2019",
                "gps": f"{panel}GPS 2000-2020",
            }.get(name, name)

        ax.set_ylabel('mm/yr')
        ax.plot(x, [0 for _ in x], 'k-', linewidth=0.5)

        getobs = lambda js: js['obs']['obs'][[o['name'] for o in js['obs']['obs']].index(name)]

        # Model
        y = np.array([getobs(js)['median'] for js in stations_js])
        ylo = np.array([o['lower'] for js in stations_js for o in [getobs(js)]])
        yhi = np.array([o['upper'] for js in stations_js for o in [getobs(js)]])

        thres = 100
        if x.size < thres:
            scale_x = thres/x.size
            scale_y = 1
        else:
            scale_x = 1
            scale_y = 1

        # ax.errorbar(x, y, label='Model', color='steelblue', yerr=[y-ylo, yhi-y], ecolor='steelblue', alpha=0.5)
        ax.errorbar(x, y, label='Model', color='steelblue', yerr=[y-ylo, yhi-y], alpha=0.5, fmt='.', markersize=1*scale_x, linewidth=2*scale_x)
        # ax.bar(x, bottom=y-ylo, height=yhi-ylo, width=0.8, label='Model', color='steelblue', alpha=0.5, capsize=1)


        # Obs
        Y = np.array([getobs(js)['obs'] for js in stations_js])
        Ylo = np.array([(getobs(js)['obs_lower']) for js in stations_js])
        Yhi = np.array([(getobs(js)['obs_upper']) for js in stations_js])

        ax.errorbar(x, Y, fmt='.', label='Obs', color='black', yerr=[Y-Ylo, Yhi-Y],
                    markersize=1*scale_x, linewidth=0.25*scale_x, capsize=1*scale_x, capthick=0.5*scale_y)

        # ax.bar(x, bottom=Y-Ylo, height=Yhi-Ylo, width=0.25, label='Obs', color='black', capsize=1)


        # ax.plot(x, y, label='Obs', color='black', linewidth=0.5)
        # ax.plot(x, ylo, color='black', linewidth=0.5, linestyle='--')
        # ax.plot(x, yhi, color='black', linewidth=0.5, linestyle='--')

        # Hightlight when the 90% error ranges do not intersect
        if True:
            ylim = ax.get_ylim()
            ax.set_ylim(ylim)
            bad = (ylo > Yhi) | (yhi < Ylo)
            # ax.bar(x[bad], height=ylim[1]-ylim[0], bottom=ylim[0], color='tab:red', alpha=0.1, width=0.8)
            for i, xx in enumerate(x[bad]):
                # l = ax.axvline(xx, color='tab:red', linewidth=0.5, linestyle=':')
                l = ax.axvline(xx, color='tab:red', alpha=0.1)
                l = ax.axvline(xx, 0, 0.01, color='tab:red', linestyle='-')
                # l = ax.arrow(xx, 0, 0, 0.01*(ylim[1]-ylim[0]), color='tab:red', linewidth=0.5, linestyle='-')
                if i == 0: l.set_label('mismatch')

            # ax.set_xticks(x[bad], [self.stations_js[i]["station"]["station_id"] for i in x[bad]],
            #     color="tab:red", fontsize=4, rotation=90)
            # ax.xaxis.set_tick_params(length=0, direction="in")


        l = ax.legend(fontsize='x-small', title=label, **legend_kw)
        l._legend_box.align = "left"


    def plot_obs(self, name, *args, **kwargs):
        return _plot_obs_from_records(self.stations_js, name, x=self.x, **kwargs)

    def plot_proj(self, experiments=None, diag="proj2100", panel="", **kwargs):
        # rate = kwargs.pop("rate", False) or diag == "rate2100"
        rate = diag.startswith("rate")
        legend_kw = dict(
            fontsize='x-small',
            title=f'{panel}Rate in 2100' if rate else f'{panel}Projections',
            title_fontproperties=legend_title_font,
            )
        legend_kw.update(kwargs.pop("legend_kw", {}))
        kwargs["legend_kw"] = legend_kw

        if rate:
            return self.plot_diag(experiments=experiments, diag=diag, label="RSLR (mm/yr)", scale=10, **kwargs)
        else:
            return self.plot_diag(experiments=experiments, diag=diag, label="RSLR (cm)", scale=1, **kwargs)


    def plot_diag(self, diag, source="total", field="rsl", experiments=None, ax=None, legend_kw={}, uncertainty=True, scale=None, label=None, labels=None, color=None):

        if ax is None: ax = plt.gca()
        x = self.x
        scale = self.scale*scale

        getv = lambda r : (r['median']*scale,
            (r['median'] - r['lower'])*scale,
            (r['upper'] - r['median'])*scale)

        ax.plot(x, [0 for _ in x], 'k-', linewidth=0.5)

        if experiments is None:
            experiments = self.stations_js[0]["experiments"]

        records = [r for js in self.stations_js for r in js['records'] if r['diag'] == diag and r['source'] == source and r['field'] == field]

        for experiment, label in zip(experiments, (labels or [xlabels.get(experiment, experiment) for experiment in experiments])):
            y, ylo, yhi = zip(*(getv(r) for r in records if r['experiment'] == experiment))
            if uncertainty:
                ax.errorbar(x, y, color=color or xcolors.get(experiment), ecolor=color or mpl.colors.to_rgba(xcolors.get(experiment))[:3]+(0.2,), yerr=[ylo, yhi],
                    # alpha=0.2,
                    fmt='.', markersize=1, label=label)
            else:
                ax.plot(x, y, '.', color=color or xcolors.get(experiment), markersize=1, label=label)


            # add AR6 values?
            if False:
                ar6_x = []
                ar6_y = []
                with ar6.supp.open_slr_regional('total', experiment) as ds:
                    for i, js in enumerate(self.stations_js):
                        station = js['station']
                        for id_k, ID in enumerate(station["PSMSL IDs"].split(",") if "PSMSL IDs" in station else []):
                            try:
                                value = ds['sea_level_change'].loc[0.5, 2100, int(ID)].values / 10 # mm to cm
                                ar6_x.append(i)
                                ar6_y.append(value)
                            except KeyError as error:
                                # print(error)
                                # raise
                                continue
                ax.plot(ar6_x, ar6_y, color=color or xcolors.get(experiment), label='IPCC AR6' if experiment == experiments[0] else None,
                    marker='o', linestyle='none', markeredgewidth=0.25, markerfacecolor='none', markersize=2)


            if self.global_js:
                y = [getv(r)[0] for r in self.global_js['records'] if r['source'] == 'total' and r['experiment'] == experiment and r['diag'] == diag]
                # ylo = np.array([(r['lower'][-1])*scale for r in self.global_js['records'] if r['source'] == 'total' and r['experiment'] == experiment])
                # yhi = np.array([(r['upper'][-1])*scale for r in self.global_js['records'] if r['source'] == 'total' and r['experiment'] == experiment])
                assert len(y) == 1, len(y)
                ax.plot(x, [y[0] for _ in x], label='GMSL' if experiment == 'ssp585' else None, color=color or xcolors.get(experiment), linestyle='-', linewidth=0.5)
                # ax.plot(x, [ylo[0] for _ in x], color=xcolors.get(experiment_), linestyle=':', linewidth=0.5)
                # ax.plot(x, [yhi[0] for _ in x], color=xcolors.get(experiment_), linestyle=':', linewidth=0.5)

        if legend_kw:
            l = ax.legend(**legend_kw)
            l._legend_box.align = "left"


        if label:
            ax.set_ylabel(label)


    def _plot_share(self, experiment, name, choices, labels, colors, filter, t=lambda r, v: v, ax=None, historical_period=(1993, 2018), legend_kw={}, gmsl=None):
        if ax is None: ax = plt.gca()
        x = self.x
        scale = self.scale


        if experiment == "historical":
            filter2 = lambda r: (filter(r) and r['experiment'] == self.stations_js[0]['records'][0]['experiment'] and r["diag"] == "rate2000")
            get = lambda r: t(r, r['median'])
            ax.set_ylabel('RSLR (mm/yr)')
        else:
            filter2 = lambda r: (filter(r) and r['experiment'] == experiment and r["diag"] == "proj2100")
            ax.set_ylabel('RSLR (cm)')
            get = lambda r: t(r, r['median']*scale)

        if "title" in legend_kw:
            xlabels = globals()['xlabels'].copy()
            xlabels['historical'] = "{0}-{1}".format(*historical_period)
            legend_kw["title"] += f" ({xlabels.get(experiment)})"

        # Contribution by source
        # ylims = (-290, 240)

        yplus_prev = np.array([0. for _ in x])
        yminus_prev = np.array([0. for _ in x])

        for source in choices:
            filtered = [r for js in self.stations_js for r in js['records'] if r[name] == source and filter2(r)]
            assert len(filtered) == self.x.size, repr((name, source, experiment, [{k:r[k] for k in [name, "experiment", "diag"]} for r in self.stations_js[0]["records"][:10]]))
            yplus = np.array([get(r) if get(r) >= 0 else 0 for r in filtered])
            yminus = np.array([get(r) if get(r) < 0 else 0 for r in filtered])

            ax.bar(x, yplus, bottom=yplus_prev, label=labels.get(source, source), color=colors.get(source))
            ax.bar(x, yminus, bottom=yminus_prev, color=colors.get(source))

            yplus_prev += yplus
            yminus_prev += yminus

        # overlay total as black
        y = np.array([get(r) for js in self.stations_js for r in js['records'] if r["source"] == 'total' and r["field"] == "rsl" and filter2(r)])
        # ax.plot(x, y, '-', label=sourcelabels.get('total'), markersize=1, color='black', linewidth=0.5)
        ax.plot(x, y, '.', label=sourcelabels.get('total'), markersize=1, color='black', linewidth=0.5)

        # add GMSL
        if gmsl:
            y = [get(r) for r in gmsl['records'] if r['source'] == 'total' and filter2(r)]
            assert len(y) == 1, len(y)
            ax.plot(x, [y[0] for _ in x], label='GMSL', color='black', linestyle='--', linewidth=0.5)

        l = ax.legend(fontsize='x-small', **legend_kw)
        l._legend_box.align = "left"
        # ax.text(0.005, 0.98, xlabels.get(experiment, experiment), transform=ax.transAxes, va='top')



    def plot_source(self, experiment, ax=None, legend_kw={}, panel="", **kw):

        legend_kw = legend_kw.copy()
        legend_kw.setdefault("title", f'{panel}Contributions per source')
        legend_kw.setdefault("title_fontproperties", legend_title_font)
        legend_kw.setdefault("ncol", 3)

        return self._plot_share(experiment, "source", ['glacier', 'gis', 'ais', 'landwater', 'steric', 'gia', 'vlm_res'],
            sourcelabelslocal, sourcecolors, lambda r: r['field'] == 'rsl',
            ax=ax, legend_kw=legend_kw, **kw)


    def plot_field(self, experiment, ax=None, legend_kw={}, panel="", **kw):

        legend_kw = legend_kw.copy()
        legend_kw.setdefault("title", f'{panel}Contributions per VLM and geocentric sea level')
        legend_kw.setdefault("title_fontproperties", legend_title_font)
        # legend_kw.setdefault("ncol", 2)

        # return self._plot_share(experiment, "field", ['rad', 'gsl'],
        return self._plot_share(experiment, "field", ['gsl', 'rad'],
            {
                "gsl": "geocentric sea level",
                "rad": "VLM (land subsidence > 0)",
            },
            {
                "gsl": mpl.colors.to_rgba('royalblue')[:3]+(0.2,),
                "rad": mpl.colors.to_rgba('tab:brown')[:3]+(0.7,),
            },
            lambda r: r['source'] == 'total',
            t=lambda r, v: -v if r['field'] == 'rad' else v,
            ax=ax, legend_kw=legend_kw, **kw)


    def plot_latitude(self, ax=None, panel="", legend_kw={}):

        if ax is None: ax = plt.gca()

        ylims = -79.99, 89.99 # remove uppermost label 80 to decrease paddin
        ax.set_ylim(ylims)

        ax.set_ylabel('Latitude (°N)')
        # ax.bar(x, y, color='black', alpha=1, width=0.95)

        for basin, group in itertools.groupby(enumerate(self.stations_js), key=lambda js: js[1]['station']['coastline']):
            indices, stations_js = zip(*group)
            x = np.array(indices)
            y = [js['station']['Latitude'] for js in stations_js]
            ax.fill_between(x, 0, y, color=basincolors.get(basin, 'black'), alpha=0.4)

        # add featured stations
        x = self.x
        y = [js['station']['Latitude'] for js in self.stations_js]
        for i, js in enumerate(self.stations_js):
            label = self.featured_stations.get(js["station"]["Station names"])
            if label is None: continue
            ax.bar(i, y[i], width=0.9, color=self.featured_color)

            bbox = dict(edgecolor=self.featured_color, facecolor="white", boxstyle="round", linewidth=0.5, alpha=0.5)
            # bbox = None
            ax.text(i, y[i]+5 if y[i] > 0 else 5, label.split('.')[0], color=self.featured_color, fontsize="xx-small", bbox=bbox, horizontalalignment="center")

        if panel:
            legend_kw['loc'] = "lower left"
            legend_kw['title'] = f"{panel}Ocean basins"
            legend_kw['frameon'] = False
            legend_kw['title_fontproperties'] = legend_title_font
            ax.legend([],[], **legend_kw)

        self.set_basins_labels(ax)


    def set_basins_labels(self, ax, colored=False, offset_fraction=20/160, offset_from_bottom=0, show_counts=True, show_featured=False):

        ylims = ax.get_ylim()

        # Basin labels
        dx = -2.5
        dy = ylims[1]-ylims[0]
        y = ylims[0]+0.01*dy

        basin_label = {
            "Subpolar North Atl. West": "Subpolar North\nAtl. West",
            "Subtropical North Atlantic": "Subtropical\nNorth Atlantic",
            # "Northwest Pacific": "Northwest\nPacific",
            # "South Atlantic": "South\nAtlantic",
        }

        # label_offset = {"Subtropical North Atlantic": 30}
        # label_offset = {"Subtropical North Atlantic": 20/160*dy}
        label_offset = {
            "Mediterranean": offset_fraction*dy,
            "Northwest Atlantic": offset_fraction*dy,
            "East Pacific": offset_fraction*dy,
        }

        for i, basin in enumerate(self.basins):
            kw = dict(color=basincolors[basin]) if colored else {}
            ax.text(self.counts[i]+dx, offset_from_bottom*dy + y + label_offset.get(basin, 0), basin_label.get(basin, basin),
                    fontdict={'ha':'right', 'fontsize':'x-small', 'va': 'bottom'}, **kw)

        # Station count for each basin is to be shown
        if show_counts:
            ax.set_xticks(np.array(self.counts)-0.5, self.bcounts, fontsize='x-small')
        else:
            ax.set_xticks([], [])

        if show_featured:
            pass

        ax.set_xlabel('Coastline locations per ocean basin')



    def plot_all(self, plots, gridscale=None, shares=['field', 'source'], obs_names=['tidegauge', 'gps', 'satellite'], legend_kw={}, scale_fig_width=True, figwidth=None, old_obs=False):

        specs = {
            'obs': len(obs_names),
            'ssp585': len(shares),
            'ssp126': len(shares),
            'historical': len(shares),
        }

        plots = [{"type": p} if type(p) is str else p.copy() for p in plots]

        plot_sizes = []
        for p in plots:
            size = p.get('size', 1 if p['type'] != 'latitude' else 0.5)
            for i in range(specs.get(p['type'], 1)):
                plot_sizes.append(size)

        if figwidth is None:
            if scale_fig_width:
                fixed = .75
                figwidth = ((9.75-fixed)*self.x.size/200 + fixed)
                figwidth = max(5, figwidth)
                figwidth = min(9.75, figwidth)

            else:
                figwidth = 9.75


        f = plt.figure(figsize=(figwidth, 2.5*sum(plot_sizes)))

        if gridscale is None:
            gridscale = int(np.prod([1/s for s in set(plot_sizes) if s < 1]))
        axes = [plt.subplot2grid((int(sum(plot_sizes)*gridscale), 1), (i*gridscale, 0), rowspan=int(s*gridscale)) for i, s in enumerate(plot_sizes)]

        for ax in axes:
            # Remove x-ticks by default
            ax.set_xticks([], [])


        # Median 2100 projection, and VLM in transparency
        for k, v in dict(borderaxespad=0.1, frameon=True, edgecolor="None", fancybox=False, loc='upper left').items():
            legend_kw.setdefault(k, v)

        legend_kw.setdefault("title_fontproperties", legend_title_font)

        axes_iter = (ax for ax in axes)
        axes_names = []
        def nextaxis(name):
            axes_names.append(name)
            return next(axes_iter)

        panels = list("abcdefghijklmnopqrstuvwxyz".upper())
        ipanel = -1

        for kw in plots:

            plot_type = kw.pop('type')
            size = kw.pop('size', None)

            lkw = kw.pop("legend_kw", {})
            for k,v in legend_kw.items():
                lkw.setdefault(k, v)

            ylim = {
                "ssp585": (-120, 200) if not kw.get('rate') else (-10, 45),
                "ssp126": (-100, 120) if not kw.get('rate') else (-8, 12),
                "historical": (-20, 30) if not kw.get('rate', True) else (-8, 12),
                "obs": (-8, 12),
            }

            if plot_type == 'obs':
                for name in obs_names:
                    ax = nextaxis('obs')
                    ipanel += 1
                    panel = f"({panels[ipanel]}) "
                    ax.set_ylim(ylim.get('obs'))
                    (self.plot_obs_old if old_obs else self.plot_obs)(name, ax=ax, legend_kw=lkw, panel=panel, **kw)

            if plot_type == 'proj':
                ax = nextaxis('proj')
                ipanel += 1
                panel = f"({panels[ipanel]}) "
                ax.set_ylim(ylim.get('ssp585'))
                self.plot_proj(kw.pop('experiments', ['ssp126', 'ssp585']), ax=ax, legend_kw=lkw, panel=panel, **kw)

            if plot_type in ['historical', 'ssp126', 'ssp585']:
                for k in shares:
                    ax = nextaxis(f'{k}-{plot_type}')
                    ax.set_ylim(ylim.get(plot_type))
                    ipanel += 1
                    panel = f"({panels[ipanel]}) "
                    getattr(self, f"plot_{k}")(plot_type, ax=ax, legend_kw=lkw, panel=panel, **kw)

            if plot_type == 'latitude':
                ax = nextaxis('latitude')
                ipanel += 1
                panel = f"({panels[ipanel]}) "
                self.plot_latitude(ax, panel=panel, legend_kw=lkw)

        for i, ax in enumerate(axes):
            self.set_axis_layout(ax, featured=i!=len(axes)-1)

        # Tick only for last subplot where station count for each basin is to be shown
        ax.set_xticks(np.array(self.counts)-0.5, self.bcounts, fontsize='x-small')

        plt.tight_layout(h_pad=0)

        return f


def make_barplot(stations_js, global_js=None, plots=['obs', 'historical', {'type': 'latitude', 'size': 0.5}], featured_stations=None, **kw):
    return BarPlot(stations_js, global_js, featured_stations=featured_stations).plot_all(plots, **kw)



class BarPlotSensitivity(BarPlot):

    def __init__(self, stations_dict, colors=None, symbols=None):
        k1 = lambda r : sort_basins.index(r['station']['coastline'])
        # k1 = lambda r : sort_basins.index(r['station']['sub-basin'])
        k1_sort = lambda r : (k1(r), -r['station']['Latitude'])

        # Add station_id field
        # if any("station_id" not in js["station"] for js in self.stations_js):
        station_ID_by_name = region_info.reset_index().set_index("Station names")
        for k, v in stations_dict.items():
            for js in v:
                if "station_id" not in js["station"]:
                    js["station"]["station_id"] = station_ID_by_name.loc[js["station"]["Station names"]]["station_id"]

        #
        all_ids = set(js["station"]["station_id"] for v in stations_dict.values() for js in v)
        n = len(all_ids)
        if not all(len(v) == n for v in stations_dict.values()):
            lengths = {k:len(v) for k,v in stations_dict.items()}
            missing = {k: all_ids.difference(set(js['station']['station_id'] for js in v)) for k,v in stations_dict.items() if lengths[k] < n}
            all_missing = set.union(*missing.values())
            logger.warning(f"stations length (out of {n} in total): {lengths}")
            logger.warning(f"missing stations: {missing}")
            logger.warning(f"Remove missing stations from all: {all_missing}")
            stations_dict = {k : [js for js in stations_js if js['station']['station_id'] not in all_missing] for k, stations_js in stations_dict.items()}
            n -= len(all_missing)

        assert all(len(v) == n for v in stations_dict.values())
            # raise ValueError(f"Stations must be homogeneous")
        # assert all(len(v) == n for v in stations_dict.values()), f"Station length must be homogeneous. Expected len {n}. Got: {k:len(v) for k,v in stations_dict.items()}"

        # Sort the stations by basin and latitude
        # assert len(set(len(v) for v in stations_dict.values())) == 1, f"Station length must be homogeneous, got: {k:len(v) for k,v in stations_dict.items()}"
        self.stations_dict = {k: list(sorted(v, key=k1_sort)) for k, v in stations_dict.items()}
        self.stations_js = next(iter(self.stations_dict.values())) # take first station for some diagnostics

        # self.global_js = global_js
        self.x = np.arange(n)
        self.scale = 1/10

        if colors is None:
            colors = {k: plt.cm.tab10(i) for i, k in enumerate(self.stations_dict)}
        self.colors = colors
        self.symbols = symbols or {}


        basins, bcounts = zip(*[(basin, len(list(group))) for basin, group in itertools.groupby(self.stations_js, key=lambda js: js['station']['coastline'])])
        counts = np.cumsum(bcounts)

        self.basins = basins
        self.counts = counts
        self.bcounts = bcounts


    def plot_obs(self, name, label=None, ax=None, panel="", legend_kw={}):

        if ax is None: ax = plt.gca()
        x = self.x
        stations_js = self.stations_js
        if label is None:
            label = {
                "tidegauge": f"{panel}Tide gauge",
                "satellite": f"{panel}Satellite 1993-2019",
                "gps": f"{panel}GPS 2000-2020",
            }.get(name, name)

        ax.set_ylabel('mm/yr')
        ax.plot(x, [0 for _ in x], 'k-', linewidth=0.5)

        getobs = lambda js: js['obs']['obs'][[o['name'] for o in js['obs']['obs']].index(name)]

        # Model
        ranges = []

        for k, stations_js in self.stations_dict.items():
            y = np.array([getobs(js)['median'] for js in stations_js])
            ax.errorbar(x, y, fmt=self.symbols.get(k, '.'), label=k, color=self.colors.get(k), markersize=1, zorder=20 if k == 'default' else None)

            ranges.append(y)

        y = np.mean(ranges, axis=0)
        yerr = y - np.min(ranges, axis=0), np.max(ranges, axis=0) - y

        color = "black"
        ax.errorbar(x, y, color='none', ecolor=mpl.colors.to_rgba(color)[:3]+(0.2,), yerr=yerr, label='Range of median values', zorder=-1)

        # Obs
        Y = np.array([getobs(js)['obs'] for js in self.stations_js])
        Ylo = np.array([(getobs(js)['obs_lower']) for js in self.stations_js])
        Yhi = np.array([(getobs(js)['obs_upper']) for js in self.stations_js])

        ax.errorbar(x, Y, fmt='.', label='Obs', color='black', yerr=[Y-Ylo, Yhi-Y],
                    markersize=1, linewidth=0.25, capsize=1, capthick=0.5)

        l = ax.legend(fontsize='x-small', title=label, **legend_kw)
        l._legend_box.align = "left"


    def plot_proj(self, experiment, ax=None, legend_kw={}, uncertainty=False, rate=False, source='total', field='rsl', range_=True, panel=""):

        assert type(experiment) is str or (type(experiment) is list and len(experiment) == 1)
        if type(experiment) is list:
            experiment = experiment[0]

        if ax is None: ax = plt.gca()
        x = self.x
        stations_js = self.stations_js
        scale = self.scale

        if rate:
            ax.set_ylabel('RSLR (mm/yr)')
        else:
            ax.set_ylabel('RSLR (cm)')

        get = lambda a : a[-1]

        getv = lambda r : (get(r['median'])*scale,
            (get(r['median']) - get(r['lower']))*scale,
            (get(r['upper']) - get(r['median']))*scale)


        ax.plot(x, [0 for _ in x], 'k-', linewidth=0.5)

        ranges = []

        for i, (k, stations_js) in enumerate(self.stations_dict.items()):
            y, ylo, yhi = zip(*(getv(r) for js in stations_js for r in js['records'] if r['source'] == source and r['field'] == field and r['experiment'] == experiment and r['diag'] == ('rate' if rate else 'change')))

            # ax.errorbar(x, y, color=xcolors.get(experiment), ecolor=mpl.colors.to_rgba(xcolors.get(experiment))[:3]+(0.2,), yerr=[ylo, yhi],
            ax.errorbar(x, y, color=self.colors.get(k), ecolor=mpl.colors.to_rgba(self.colors.get(k))[:3]+(0.2,),
                yerr=[ylo, yhi] if uncertainty else None,
                fmt=self.symbols.get(k, '.'), markersize=1, label=k, zorder=20 if k == 'default' else None)

            ranges.append(y)

        if range_:
            y = np.mean(ranges, axis=0)
            yerr = y - np.min(ranges, axis=0), np.max(ranges, axis=0) - y

            # color = 'tab:blue'
            color = 'black'
            # ax.errorbar(x, y, color='none', ecolor="black", linewidth=0.5, yerr=yerr, label='Median update', zorder=-1)
            ax.errorbar(x, y, color='none', ecolor=mpl.colors.to_rgba(color)[:3]+(0.2,), yerr=yerr, label='Range of median values', zorder=-1)


        if rate:
            l = ax.legend(fontsize='x-small', title=f'{panel}Rate in 2100 ({xlabels.get(experiment)}, {sourcelabelslocal.get(source)})', **legend_kw)
        else:
            l = ax.legend(fontsize='x-small', title=f'{panel}Projections to 2100 ({xlabels.get(experiment)}, {sourcelabelslocal.get(source)})', **legend_kw)

        l._legend_box.align = "left"


    def plot_all(self, plots, *args, **kwargs):
        super().plot_all(plots, *args, **kwargs)

def make_barplot_sensitivity(stations_dict, plots=['obs', {'type': 'proj', 'experiments': 'ssp126'}, {'type': 'proj', 'experiments': 'ssp585'}, 'latitude'], colors=None, symbols=None, custom_legend=True, **kw):
    BarPlotSensitivity(stations_dict, colors=colors, symbols=symbols).plot_all(plots, **kw)

    # Better legend (will break if different plots is passed...)
    if not custom_legend:
        return

    for i, ax in enumerate(plt.gcf().axes):
            legend = ax.get_legend()
            handles, labels = ax.get_legend_handles_labels()
            if legend is None:
                continue
            kw = dict(fontsize='x-small', title=str(legend.get_title().get_text()),
                          borderaxespad=0.1, frameon=True, edgecolor="None", fancybox=False, loc='upper left')
            if i in [0, 2, 3, 4]:
                legend.remove()
                l = ax.legend([], **kw)
                l._legend_box.align = "left"
            if i == 1:
                kw2 = kw.copy()
                kw2['frameon'] = True
                l = ax.legend(ncol=3, **kw2)
                l._legend_box.align = "left"
            # if i in (3, 4):
            #     ax.legend(*zip(*((h,l) for h,l in zip(handles, labels) if l == 'Range of median values') ), **kw)
            #     l._legend_box.align = "left"



# def make_proj_figure(trace, stations_js, experiments=None, extra_experiments=None):
def make_proj_figure(global_js=None, stations_js=None, experiments=None, extra_experiments=None, diff_experiments=None, diag="proj2100",
    underlay=None, overlay=None, underlay_label="", overlay_label="", featured_stations=None):
    if stations_js and global_js:
        f = plt.figure(figsize=(9.75, 6))
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        axes = [ax1, ax2]
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        all_axes = [ax1, ax2, ax3]

    elif stations_js:
        f = plt.figure(figsize=(9.75, 4))
        ax3 = plt.subplot2grid((2, 1), (0, 0), colspan=2, rowspan=2)
        axes = []
        all_axes = [ax3]

    elif global_js:
        f = plt.figure(figsize=(9.75, 3))
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (0, 1))
        axes = []
        all_axes = [ax1, ax2]

    else:
        raise ValueError("no stations_js or global_js provided")
    # f, axes = plt.subplots(, 2, figsize=(9.75, 4))

    tas_df = load_temperature()
    tas_pi = tas_df['ssp585'].loc[1995:2014].mean() - tas_df['ssp585'].loc[1855:1900].mean()

    # post = arviz.extract(trace.posterior)

    if experiments is None:
        experiments = [f"C{i+1}" for i in range(8)]

    if extra_experiments is None:
        extra_experiments = ['ssp126', 'ssp585']


    if diff_experiments:
        diff = True
    else:
        diff_experiments = []

    rate = diag.startswith("rate")

    if global_js:
        for i, experiment in enumerate(experiments+extra_experiments+diff_experiments):
            # postx = post.sel(experiment=experiment)
            cat = experiment

            ax = ax1
            # q = .5, .05, .95
            x = np.array(global_js['years'])
            recs = [r for r in global_js['records'] if r['source'] == 'tas' and r['diag'] == 'change' and r['experiment'] == experiment]
            assert len(recs) == 1, repr(recs)
            r = recs[0]

            mid, lo, hi = ((tas_pi if experiment not in diff_experiments else 0) + np.array(r[k]) for k in ['median', 'lower', 'upper'])

            # mid, lo, hi = (postx.tas+tas_pi).quantile(q, dim="sample").values
            l, = ax.plot(x, mid, label=xlabels.get(cat, cat), color=xcolors.get(cat), linestyle='--' if experiment not in experiments else '-')

            if experiment in experiments:
                ax.fill_between(x, lo, hi, color=l.get_color(), alpha=0.2)

            if experiment in diff_experiments:
                ax.plot(x, lo, color=xcolors.get(cat), linestyle=':')
                ax.plot(x, hi, color=xcolors.get(cat), linestyle=':')

            ax = ax2
            recs = [r for r in global_js['records'] if r['source'] == 'total' and r['diag'] == ('rate' if rate else 'change') and r['experiment'] == experiment]
            assert len(recs) == 1, repr(recs)
            r = recs[0]
            scale = 1 if rate else 1/10
            mid, lo, hi = (np.array(r[k])*scale for k in ['median', 'lower', 'upper'])
            # slr = postx.total.cumsum('year')/10
            # slr -= slr.sel(year=slice(1995, 2014)).mean('year')
            # mid, lo, hi = slr.quantile(q, dim="sample").values
            ax.plot(x, mid, color=l.get_color(), label="median" if experiment == experiments[-1] else None, linestyle='--' if experiment not in experiments else '-')
            if experiment not in extra_experiments:
                ax.fill_between(x, lo, hi, color=l.get_color(), alpha=0.2, label="90% range" if experiment == experiments[-1] else None)
        #     # add tide-gauge data
        #     ax = ax3
        #     x = np.arange(len(dats))
        #     ax.errorbar(x, [r for js in dats for r in js if r['experiment'] == cat and r['source'] == 'total' and r['field'] == 'rsl'], color=xcolors.get(cat))


        ax = ax1
        ax.legend(fontsize='x-small', loc='upper left')
        ax.set_ylabel('Warming (C)')

        ax = ax2
        ax.set_ylabel('SLR (mm/yr)' if rate else 'SLR (cm)')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.axhline(0, linewidth=0.5, color='black')
        ax.legend(fontsize='x-small', loc='upper left')

        for ax in axes:
            ax.grid()
            ax.set_xlim(1900, 2100)

        ax1.set_xticks([1900, 1950, 2000, 2050, 2100])
        ax1.set_xlabel('Years')
        ax1.set_title('(A) Global mean temperature')
        ax2.set_xticks([1950, 2000, 2050, 2100])
        ax2.set_xlabel('Years')
        if rate:
            ax2.set_title('(B) Global mean rate of sea-level rise')
        else:
            ax2.set_title('(B) Global mean sea-level rise')

    if stations_js:
        barplot = BarPlot(stations_js, featured_stations=featured_stations)

        if underlay:
            barplot.plot_proj(experiments[:1], ax=ax3, uncertainty=False, diag=underlay, color="black", labels=[underlay_label or underlay])

        barplot.plot_proj(experiments, ax=ax3, uncertainty=True, diag=diag)


        if diff_experiments:
            barplot.plot_proj(diff_experiments, ax=ax3, uncertainty=True, diag=diag)

        for experiment in extra_experiments:
            recs = [r for js in barplot.stations_js for r in js['records'] if r['experiment'] == experiment and r['field'] == 'rsl' and r['source'] == 'total' and r['diag'] == diag]
            ax3.plot(barplot.x, [r['median']*scale for r in recs], '.', color=xcolors.get(experiment), markersize=1, label=xlabels.get(experiment, experiment))

        if overlay:
            barplot.plot_proj(experiments[:1], ax=ax3, uncertainty=False, diag=overlay, color="black", labels=[overlay_label or overlay])

        ax3.get_legend().remove()
        if rate:
            ax3.set_ylim(-10, 25)
        else:
            ax3.set_ylim(-100, 200)
        barplot.set_axis_layout(ax3, featured=featured_stations is not None, featured_labels="bottom")
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        # ax3.set_xticks([], [])
        # ax3.set_xticks(barplot.counts, [f"{lab} ({c})" for lab, c in zip(barplot.basins, barplot.bcounts)], horizontalalignment='right')
        # ax3.set_xticks(np.array(barplot.counts)-0.5, barplot.counts, fontsize='small')
        # ax3.set_xticks(np.array(barplot.counts)-0.5, [])
        barplot.set_basins_labels(ax3, colored=True, offset_fraction=1/20, offset_from_bottom=1.5/20, show_counts=False)
        ax3.set_title('(C) Local, relative sea-level rise projections')
        # ax3.set_xlabel('Coastline locations per ocean basin')
        # ax3.xaxis.set_label_coords(.5, -.1)
        ax3.legend(loc="upper left", fontsize="x-small")
        if rate:
            ax3.set_ylabel('RSLR (mm/yr)')
        else:
            ax3.set_ylabel('RSLR (cm)')

    plt.tight_layout()

    return f, all_axes


def get_repo_data(ref='HEAD'):
    try:
        return {
            'short': subprocess.check_output(f'git rev-parse --short {ref}', shell=True).decode('utf8').strip(),
            # 'branch': subprocess.check_output(f'git rev-parse --abbrev-ref HEAD', shell=True),
            'date': subprocess.check_output(f'git log -1 {ref} --date=format:"%Y/%m/%d %T" --format="%ad"', shell=True).decode('utf8').strip(),
        }
    except:
        return {}


def _boxplot_custom(ax, x, median_y, box_y, whisker_y, color, w=.25, fill=False, median_color=None):
    # Draw the box
    box_left = x - w
    box_right = x + w
    box_lower, box_upper = box_y
    ax.add_patch(plt.Rectangle((box_left, box_lower), box_right - box_left, box_upper - box_lower, fill=fill, color=color, zorder=2))

    # Draw the whiskers
    for y in whisker_y:
        ax.plot([x - w/2, x + w/2], [y, y], color=color, linewidth=1)

    # Draw the lines connecting boxes and whiskers
    for ys in zip(box_y, whisker_y):
        ax.plot([x, x], ys, color=color, linestyle='-', linewidth=1)

    # Draw the median
    ax.plot([x-w, x+w], [median_y]*2, color=median_color or "black", linewidth=1)


def boxplot_custom(ax, median_y, box_y, whisker_y, colors=None, color="black", median_color=None, xoffset=0, **kwargs):
    if median_color is None:
        median_color = color
    if colors is None:
        colors = [color or "tab:blue"]*len(median_y)
    for i, (m, b, wis, c) in enumerate(zip(median_y, np.array(box_y).T, np.array(whisker_y).T, colors)):
        _boxplot_custom(ax, i + 1 + xoffset, m, b, wis, c, median_color=median_color, **kwargs)


def get_model_with_full_local_constraints(tr, cirun=None):
    # Make sure we use a model definition that actually includes the obs, so we can do posterior predictive sampling.

    # ... case of time-series observations in the model
    tidegauge_timeseries_vars = [v.name for v in tr.model.observed_RVs if v.name.startswith("obs_tidegauge_")]

    # Compute Tide-gauge trends from 1993 to 2018 based on annual variables
    if not hasattr(tr.model, "tidegauge_obs") and tidegauge_timeseries_vars:
        import pytensor.tensor as pt
        import pymc as pm
        from sealevelbayes.models.likelihood import pytensor_lintrend
        lintrends = []
        print("Calculating tide-gauge trends from annual observed variables, for PP sampling")
        for v in tqdm.tqdm(tidegauge_timeseries_vars):
            # calculate the linear trend over 1993-2018?
            id = v[len("obs_tidegauge_"):]
            years = np.array(tr.model.coords[f"tidegauge_year_{id}"])
            A = np.array([years - years[years.size//2], np.ones(years.size)]).T
            exclude = (years < 1993) | (years > 2018)
            lintrends.append(pytensor_lintrend(tr.model[v], A=A, mask=exclude))

        lintrend = pt.stack(lintrends)
        with tr.model:
            pm.Deterministic("tidegauge_obs", lintrend, dims="station")

    # if any(obs in tr.o.skip_constraints for obs in ["gps", "tidegauge", "satellite"]) or "gps-quality" in tr.o.skip_constraints:
    if any(not hasattr(tr.model, f"{obs}_obs") for obs in ["gps", "tidegauge", "satellite"]):

        missing = [obs for obs in ["gps", "tidegauge", "satellite"] if obs in tr.o.skip_constraints]
        alt_cirun = cirun or tr.o.cirun

        if missing:
            pattern = "_skip-" + "-".join([obs for obs in tr.o.skip_constraints if obs in ["gps", "tidegauge", "satellite"]])
            pattern2 = ""
            alt_cirun = str(alt_cirun).replace(pattern, pattern2)

        if "gps-quality" in tr.o.skip_constraints:
            patterns = "_gps-good", "_gps-good-medium", "_gps-medium-good", "_gps-medium"
            pattern2 = ""
            for patt in patterns:
                alt_cirun = str(alt_cirun).replace(patt, pattern2)

        alt_path = get_runpath(alt_cirun)/"config.cpk"

        if alt_path.exists():
            logger.warning(f"Use alternative model that contains the local constraints: {alt_path}")
            alt_model = cloudpickle.load(open(alt_path, "rb"))["model"]

        else:
            logger.warning("Add the mssing constraints")
            tr.add_missing_local_constraints()
            alt_model = tr.model

    else:
        alt_model = None

    return alt_model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("--tidegauge-figures", action='store_true')
    parser.add_argument("--gmsl", action='store_true')
    parser.add_argument("--all", action='store_true')
    # parser.add_argument("--www")
    # parser.add_argument("--experiment", nargs='+',
    #     help='experiment folder will be run_folder/experiment. Default to all experiments.')
    parser.add_argument("--cross-experiment", action='store_true', help=argparse.SUPPRESS)

    o = parser.parse_args()

    # main = Path(o.run_folder)

    # if o.experiment is None:
    #     o.experiment = []
    #     for x in sensitivity_experiments:
    #         name = x.get("name", "default")
    #         if (main/name).exists():
    #             o.experiment.append(name)

    def do_one_experiment(experiment):

        run_folder = get_runpath(experiment)
        wwwfolder = get_webpath(experiment)
        figdir = wwwfolder / "figures"
        jsdatadir = wwwfolder / "data"

        figdir.mkdir(exist_ok=True, parents=True)

        logger.info(f"figures: {experiment}...")

        out = figdir
        # out = main / experiment if experiment else main

        # figdir = out / 'figures'
        # if not figdir.exists():
        #     os.makedirs(figdir)

        if o.all:
            o.gmsl = True
            o.tidegauge_figures = True

        def savefig(fig, name, dpi=150):
            fig.savefig(figdir/(name+'.png'), dpi=dpi)

        logger.info("figures: load global trace and json file")

        try:
            trace0 = arviz.from_netcdf(out/"slr-global.trace.nc")
        except Exception as error:
            logger.warning(error)
            logger.warning(f"Skip {experiment}")
            return

        trace1 = trace0.sel(experiment=['ssp126', 'ssp585', 'ssp126_mu', 'ssp585_mu'])
        trace = trace1.sel(experiment=['ssp126', 'ssp585'])
        stations_js, global_js = trace_to_json(trace)

        if o.gmsl:
            logger.info("figures: gmsl_timeseries_historical")
            # js = json.load(open(out/"www/data/slr-global.json"))

            fig = make_gmsl_timeseries(global_js, historical=True)
            savefig(fig, 'gmsl_timeseries_historical')

            logger.info("figures: gmsl_timeseries_future")
            fig = make_gmsl_timeseries(global_js)
            savefig(fig, 'gmsl_timeseries_future')

            logger.info("figures: gmsl_table")
            post = arviz.extract(trace1.posterior)
            recs = get_table_numbers(post)
            fig = make_table_fig(recs)
            savefig(fig, 'gmsl_table')

            logger.info("figures: gmsl_table.html")
            df = make_table(recs)
            # df.to_csv(figdir/'gmsl_table.csv')
            gmsl_table = df.to_html()
            open(figdir/'gmsl_table.html', 'w').write(gmsl_table)


        if o.tidegauge_figures:
            # Load all stations
            logger.info("figures: load tidegauge stations")
            fname = lambda i: out/f"www/data/slr-station-{region_info.iloc[i].name}.json"
            stations_js = [json.load(open(fname(i))) for i in range(len(region_info)) if fname(i).exists()]

            featured_stations = [js for js in stations_js if js['station']['Station names'] in palmer_map.values()]
            featured_labels = {name:label for label,name in palmer_map.items()}

            if len(stations_js) == 0:
                logger.warning(f"No local station found.")
                logger.warning(f"Skip {experiment}")
                return

            if len(stations_js) < len(region_info): logger.warning(f"{len(region_info)-len(stations_js)} json files missing (out of {len(region_info)} stations)")

            logger.info("figures: tidegauges_barplot")
            fig = make_barplot(stations_js, global_js, featured_stations=featured_labels)
            savefig(fig, 'tidegauges_barplot', dpi=300)

            logger.info("figures: tidegauges_barplot_proj")
            fig = make_proj_figure(trace0, stations_js)
            savefig(fig, 'tidegauges_barplot_proj', dpi=300)

            logger.info("figures: featured_locations_historical")
            fig = plot_locations(featured_stations, [1900, 2018], [-500, 600], labels=featured_labels, squeeze_labels=False)
            savefig(fig, 'featured_locations_historical')

            logger.info("figures: featured_locations_ssp585")
            fig = plot_locations(featured_stations, experiment='ssp585', labels=featured_labels)
            savefig(fig, 'featured_locations_ssp585')

            logger.info("figures: featured_locations_ssp126")
            fig = plot_locations(featured_stations, experiment='ssp126', labels=featured_labels)
            savefig(fig, 'featured_locations_ssp126')
        else:
            stations_js = []


        if o.www:
            logger.info("figures: report.html")
            www = Path(o.www)/experiment
            dest = www/'figures'
            os.makedirs(figdir, exist_ok=True)
            subprocess.check_call(f'rsync -ra {figdir}/ {dest}/', shell=True)

            import jinja2
            templateLoader = jinja2.FileSystemLoader(searchpath="./web")
            templateEnv = jinja2.Environment(loader=templateLoader)

            open(f"{www}/report.html", "w", encoding="utf-8").write(
                templateEnv.get_template("report.html").render(
                    runfolder_git=get_repo_data(main.name) or main.name,
                    code_git=get_repo_data(),
                    )
                )

            subprocess.check_call(f'rsync -ra web/assets {www}/', shell=True)

        logger.info(f"figures: {experiment} done.")

        return js, stations_js


    stations_dict = {}

    for experiment in o.experiment:
        try:
            js, stations_js = do_one_experiment(experiment)
        except Exception as error:
            logger.warning(error)
            logger.warning(f'figure: {experiment} failed')
            raise
            continue

        if o.cross_experiments:
            stations_js
            stations_dict[experiment] = {
                "global": js,
                "local": stations_js,
            }
        else:
            del js


    if o.cross_experiments:
        pass


    logger.info("figures: done.")

if __name__ == "__main__":
    main()