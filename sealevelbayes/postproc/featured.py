import json
import copy
import os
from itertools import product, chain
import numpy as np

import matplotlib.pyplot as plt
import pymc as pm
import arviz

from sealevelbayes.config import get_runpath, get_webpath, logger
from sealevelbayes.datasets.basins import ThompsonBasins
from sealevelbayes.datasets.frederikse2020 import region_info
from sealevelbayes.datasets.naturalearth import add_land, add_coast
from sealevelbayes.datasets.hammond2021 import load_tidegauge_rates
from sealevelbayes.datasets.tidegaugeobs import psmsl_filelist_all
from sealevelbayes.models.localslr import SOURCES
from sealevelbayes.models.globalslr import SSP_EXPERIMENTS, SSP_EXPERIMENTS_mu
from sealevelbayes.postproc.serialize import trace_to_json
from sealevelbayes.postproc.serialize import resample_as_json
from sealevelbayes.postproc.colors import basincolors
from sealevelbayes.postproc.figures import psmsl_ids_to_basin
from sealevelbayes.postproc.figures import SELECT_EXPERIMENTS, SELECT_EXPERIMENTS_mu
from sealevelbayes.postproc.figures import plot_locations_experiments, SELECT_EXPERIMENTS_mu as select_experiments
from sealevelbayes.postproc.figures import plot_locations
from sealevelbayes.postproc.run import ExperimentTrace

palmer_map = {
#     "NY-ALESUND": "NY-ALESUND", # instead of "Barentsburg" => sacrified in favour of Alaskan glaciers
    "Reykjavik (Iceland)": "REYKJAVIK",
    "Oslo (Norway)": "OSLO",
    "Cuxhaven (Germany)": "CUXHAVEN 2",
#     "Newlyn": "NEWLYN", # sacrificed for Grand Isle
#     "Venice": "VENEZIA (S.STEFANO); VENEZIA (ARSENALE); VENEZIA (PUNTA DELLA SALUTE); VENEZIA II", # instead of Stanley II
#     "Venice": "VENEZIA (S.STEFANO)", # instead of Stanley II
    "Venice (Italy)": "VENEZIA (PUNTA DELLA SALUTE)", # instead of Stanley II
#     "Venice": "VENEZIA (ARSENALE)", # instead of Stanley II
#     "NewYork": "NEW YORK (THE BATTERY); BERGEN POINT, STATEN IS.",
    "New York (USA)": "NEW YORK (THE BATTERY)",
    "Grand Isle (USA)": "GRAND ISLE",
#     "SimonsBay": "SIMONS BAY",
#     "Table Bay": "TABLE BAY HARBOUR",  # instead of Simons Bay (not flagged)
    "Port Elizabeth (South Africa)": "PORT ELIZABETH",  # instead of Simons Bay (not flagged)
#     "Palermo": "BUENOS AIRES; PALERMO", # same as SimonsBay, with better tide-gauge records, but no other stations in Africa
    "Skagway (Alaska)": "SKAGWAY",
#     "SanFrancisco": "SAN FRANCISCO; ALAMEDA (NAVAL AIR STATION)",
    "San Francisco (USA)": "SAN FRANCISCO",
#     "Lima": "CALLAO 2",
    "Iquique II (Chile)": "IQUIQUE II",
    "Aburatsubo (Japan)": "ABURATSUBO", # instead of Mera
    "Haldia (India)": "HALDIA", # instead of DiamondHarbour
    "Pago pago (Samoa, USA)": "PAGO PAGO",
    "Port Louis (Mauritius)": "PORT LOUIS II",
#     "Auckland": "AUCKLAND II; AUCKLAND-WAITEMATA HARBOUR",
    "Auckland (New Zealand)": "AUCKLAND II",
}

# featured = {v:i+1 for i, (k,v) in enumerate(palmer_map.items())}
featured = {v:f'{i+1}. {k}' for i, (k,v) in enumerate(palmer_map.items())}

def get_featured_locations(psmsl_ids=None, labels=None):
    """Return a list of PSMSL ids and station names for the featured locations.
    """
    # stations_db_id = psmsl_filelist_all.set_index('ID')
    if psmsl_ids is None:
        stations_db = psmsl_filelist_all.set_index('station name')
        psmsl_ids = [stations_db.loc[name]['ID'] for name in featured]
        labels = list(featured.values())

    else:
        stations_db = psmsl_filelist_all.set_index('ID')
        if labels is None:
            labels = [f'{i+1}. {stations_db.loc[ID]["name"]}' for i,ID in enumerate(psmsl_ids)]

    return psmsl_ids, labels

def print_check_nameS(b):

    for r in region_info.to_dict("records"):
        thompson_label = b.get_region_label(r["Longitude"], r["Latitude"])
        fred_label = r["coastline"]
        if thompson_label != fred_label:
            print(r["PSMSL IDs"].split(",")[0], r["Station names"].split(";")[0], fred_label, "!=", thompson_label, r["Longitude"], r["Latitude"])


def plot_map(lons, lats, psmsl_ids, b=None, featured=featured):

    if b is None:
        b = ThompsonBasins.load().split_atlantic()

    aspect = 1.5
    plt.figure(figsize=(9.5, 9.5*(90+60)/360*aspect))
    plt.gca().set_aspect(aspect)
    plt.xlim(-30, 330)
    plt.ylim(-60, 90)

    def adjust_lon(lon):
        if lon > 330:
            return lon - 360
        if lon < 330 - 360:
            return lon + 360
        return lon


    # b.plot(shift_lon=-30, zorder=-1)

    # add_land(zorder=-1)
    # add_land(shift_lon=360, zorder=-1)

    # add_coast()
    # add_coast(shift_lon=360)

    # stations_by_name = {r['station']['Station names']: r for r in stations_js}

    bbox = dict(color="white", alpha=0.5, boxstyle="round")

    custom_kw = {
        "REYKJAVIK": {"x":-10, "y": 0, "horizontalalignment": "left", "bbox": bbox},
        "OSLO": {"y" : -5, "horizontalalignment": "left"},
        "CUXHAVEN 2": {"y" : -6, "horizontalalignment": "left", "bbox": bbox},
        "VENEZIA (PUNTA DELLA SALUTE)": {"y" : -6, "horizontalalignment": "left", "bbox": bbox},
        "NEW YORK (THE BATTERY)": {"x":10, "y": 5, "bbox": bbox},
        "GRAND ISLE": {"x": 22, "y": -10, "bbox": bbox},
        "PORT ELIZABETH": {"y": -10},
        "SKAGWAY": {},
        "SAN FRANCISCO": {"x": 10, "horizontalalignment": "right"},
        "IQUIQUE II": {},
        "ABURATSUBO": {"x": -5},
        "HALDIA": {},
        "PAGO PAGO": {},
        "PORT LOUIS II": {},
        "AUCKLAND II": {},
    }


    all_gps_rates = load_tidegauge_rates()
    # all_gps_rates_by_id = {r['ID']: r for r in all_gps_rates}
    # gps_stations = {r['name']:r for id in psmsl_ids for r in all_gps_rates_by_id[id]['gps'] if id in all_gps_rates_by_id}
    gps_stations = list({r['name']:r for tg in all_gps_rates for r in tg['gps'] if tg['ID'] in psmsl_ids}.values())

    # basins = psmsl_ids_to_basin(psmsl_ids, basins_cls=b)
    # [(b.get_region_label(lon, lat), basin) for lon, lat, basin in zip(lons, lats, basins)]

    for lon, lat, id in zip(lons, lats, psmsl_ids):
        basin = b.get_region_label(lon, lat)
        plt.plot(adjust_lon(lon), lat, '.', color=basincolors[basin])

    # add GPS stations
    for r in gps_stations:
        lon = r['lon']
        lat = r['lat']
        # basin = b.get_region_label(lon, lat)
        # plt.plot(adjust_lon(lon), lat, 'x', color=basincolors[basin])
        plt.plot(adjust_lon(lon), lat, 'kx', markeredgewidth=0.5, markersize=3)

    for lon, lat, id in zip(lons, lats, psmsl_ids):
        basin = b.get_region_label(lon, lat)
        plt.plot(adjust_lon(lon), lat, '.', markeredgewidth=0.5, markerfacecolor='none', markeredgecolor=basincolors[basin])

    stations_db = psmsl_filelist_all.set_index('station name')

    for name, label in featured.items():

        if name not in stations_db.index:
            print(name, label, "not existing !")
            continue

        d = stations_db.loc[name]
    #     label = featured.get(d["station"]["Station names"])
    #     if not label: continue
        lon, lat = d["longitude"], d["latitude"]
        lon = adjust_lon(lon)
        plt.scatter(lon, lat, color="black", facecolor="none", zorder=2)

        kw = custom_kw.get(name, {}).copy()

        text_kwargs = dict(x=lon+5+kw.pop("x", 0), y=lat+3.5+kw.pop("y", 0), s=label.split("(")[0].strip(),
                bbox=dict(color="white", alpha=0, boxstyle="round"), horizontalalignment="center")
        text_kwargs.update(kw)
    #     print(text_kwargs)

    #     text_kwargs["fontweight"] = "bold"
        plt.text(**text_kwargs)


    h, cb = b.plot(shift_lon=-30, zorder=-1, alpha=0.2)

    add_land(zorder=-1)
    add_land(shift_lon=360, zorder=-1)

    # plt.title("Tide-gauge locations and ocean basins")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")


def main():

    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cirun", default=os.getenv("CIRUN"))
    parser.add_argument("--map", action="store_true")
    parser.add_argument("--psmsl-ids", type=int, nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("-x", "--experiments", nargs="+", default=["ssp585", "ssp370", "ssp126"])
    parser.add_argument("-X", "--experiment-groups", nargs="+", action="append",
        default=[["ssp585", "ssp370", "ssp126"], ['SP_mu', 'GS_mu', 'CurPol_mu'], ['SP', 'GS', 'CurPol']])
    parser.add_argument("--fields", nargs="+", default=["rsl"])

    o = parser.parse_args()

    CIRUN = o.cirun
    CIDIR = get_runpath()
    WEBDIR = get_webpath("")
    runfolder = CIDIR/CIRUN
    wwwfolder = WEBDIR/CIRUN
    figdir = wwwfolder / 'figures'
    postprocfolder = runfolder / 'postproc'

    os.makedirs(runfolder, exist_ok=True)
    os.makedirs(postprocfolder, exist_ok=True)
    os.makedirs(wwwfolder, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    tr = ExperimentTrace.load(o.cirun)
    trace = tr.trace

    featured_ids, labels = get_featured_locations(psmsl_ids=o.psmsl_ids, labels=o.labels)


 #   b = ThompsonBasins.load().split_atlantic()

    if o.map:
        all_ids = trace.constant_data.psmsl_ids.values
        lons = trace.constant_data.lons.values
        lats = trace.constant_data.lats.values

        plot_map(lons, lats, all_ids)
        plt.savefig(figdir/'map_featured_locations.png', dpi=300)

    sources = ["vlm_res", 'gia', "landwater", "ais", "gis", "glacier", "steric", "total"]

    field = "rsl"
    for experiment in o.experiments:
        stations_js = resample_as_json(tr, psmsl_ids=featured_ids, experiments=[experiment], fields=[field], sources=sources)

        plot_locations(stations_js, labels=labels)
        plt.savefig(figdir/f"locations_{field}_{experiment}.png", dpi=300)

        if experiment == "ssp370":
            plot_locations(stations_js, [1900, 2018], [-50, 60],
                    labels=labels, squeeze_labels=False)
            plt.savefig(figdir/f"locations_{field}_historical.png", dpi=300)

    # Now "total" proj
    all_experiments = sorted(set(list(chain(*o.experiment_groups))))
    stations_js = resample_as_json(tr, psmsl_ids=featured_ids, experiments=all_experiments, fields=o.fields, sources=["total"])

    for field, group in product(o.fields, o.experiment_groups):
        plot_locations_experiments(stations_js, experiments=group, field=field, labels=labels);
        plt.savefig(figdir/f'featured_locations_{field}_{"-".join(group)}.png', dpi=300)


    # **Grand Isle**: The [PSMSL page for this station](https://psmsl.org/data/obtaining/stations/526.php) notes that:
    # > For a discussion of submergence near to Eugene Island and Grand Isle/Bayou Rigaud, see Emery and Aubrey (Sea Levels, Land Levels and Tide Gauges), 1991, Springer-Verlag, page 40.
    #
    # Strong subsidence values for the Grand Isle location are corroborated by visual analysis of tidegauge, satellite and GPS measurements in the [SLR tool](http://www.pik-potsdam.de/~perrette/slr-tidegauges/744c3e7/?station_id=2017&gmsl=false&source=total&field=rsl&experiment=ssp585&version=default&prior=false)

    # In[ ]:


    # from sealevelbayes.postproc.figures import plot_locations
    # plot_locations(featured_stations_js),
    #               [1900, 2100], [-1000, 1500], labels=featured);


    # In[ ]:


    # plt.savefig(figdir/'featured_locations_ssp585.png', dpi=300)


if __name__ == "__main__":
    main()