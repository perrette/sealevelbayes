import os
import sys
import math
import tqdm
import json
from config import logger
sys.path.append("notebooks")
import ar6.supp
from models import MAP_FRED, MAP_AR6
from tidegaugeobs import  tg_psmsl_rlr_1900_2018 as tg, tg_psmsl_rlr_1900_2018_years as tg_years, get_stations_from_psmsl_ids, psmsl_filelist_rlr
from satellite import get_satellite_timeseries


def get_stations(psmsl_ids=None):

    if psmsl_ids is None:
        psmsl_ids = psmsl_filelist_rlr['ID'].values.tolist()
    stations = get_stations_from_psmsl_ids(psmsl_ids)

    return [{
            'coords': [float(station['Longitude']), float(station['Latitude'])],
            'name': station['Station names'],
            'PSMSL_IDs': [station_id],
            'station_id': station_id,
        } for station_id, station in zip(psmsl_ids, stations)]


import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--input-stations", "-i", help="stations.json file")
parser.add_argument("--psmsl-ids", nargs="*", type=int)
o = parser.parse_args()

# if o.input_stations:
    # stations = json.load(open(o.input_stations))["stations"]
# else:
stations = get_stations(o.psmsl_ids)

obsdir = "web/obs"


# add tide-gauge and satellite data
for station in tqdm.tqdm(stations):
    station_id = station["station_id"]

    fobs = f"{obsdir}/obs-{station_id}.json"
    if not os.path.exists(fobs):
        obs = {}

        # satellite
        lon, lat = station["coords"]
        sat_years, timeseries = get_satellite_timeseries([lon], [lat])
        timeseries = timeseries.squeeze()
        obs["satellite"] = {
            "years": sat_years.tolist(),
            "values": (timeseries - timeseries[1995-sat_years[0]:2014+1-sat_years[0]].mean()).tolist(),
        }

        # tide-gauges
        obs["tidegauges"] = []
        for ID in station["PSMSL_IDs"]:
            try:
                k = tg['id'].tolist().index(ID)
            except:
                logger.warning(f"Failed to find station: {ID}")
                continue
            obs["tidegauges"].append({
                "ID": ID,
                "years": tg_years.tolist(),
                "values": [v if not math.isnan(v) else None for v in tg['height'][k].tolist()],
            })

        # IPCC AR6
        obs["ar6"] = []
        for ID in station["PSMSL_IDs"]:
            for source in ["total", "steric","glacier", "ais", "gis", "landwater", "vlm"]:
                for experiment in ["ssp585", "ssp126"]:
                    source_ar6 = MAP_AR6.get(source, source)
                    try:
                        series = ar6.supp.load_location(ID, experiment, sources=[source_ar6])[source_ar6].loc[:2100]
                    except KeyError as error:
                        print("!! Failed to load AR6 projection for tide-gauge",ID,experiment,source_ar6,":",str(error))
                        continue
                    obs["ar6"].append({
                        "ID": ID,
                        "years": series.index.values.tolist(),
                        "values": series.values.tolist(),
                        "source": source,
                        "experiment": experiment,
                    })

        json.dump(obs, open(fobs, "w"))

