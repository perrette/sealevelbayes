"""local sea-level projections from AR6 supplement

Downloaded from: https://zenodo.org/record/6382554

Garner, G. G., Hermans, T., Kopp, R. E., Slangen, A. B. A., Edwards, T. L., Levermann, A., Nowicki, S., Palmer, M. D., Smith, C., Fox-Kemper, B., Hewitt, H. T., Xiao, C., Aðalgeirsdóttir, G., Drijfhout, S. S., Golledge, N. R., Hemer, M., Krinner, G., Mix, A., Notz, D., … Pearson, B. (2021). IPCC AR6 Sea Level Projections (Version 20210809) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6382554
"""
from pathlib import Path
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xa
import yaml

from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.logs import logger
from sealevelbayes.datasets.shared import MAP_AR6

AR6_REGIONAL = get_datapath("zenodo-6382554-garner2021/ar6-regional-confidence/regional/confidence_output_files")
AR6_GLOBAL = get_datapath("zenodo-6382554-garner2021/ar6/global")
LOCATION_LIST = get_datapath("zenodo-6382554-garner2021/location_list.lst")

sources = [
    "oceandynamics",
    "glaciers",
    "GIS",
    "AIS",
    "landwaterstorage",
    "verticallandmotion",
    "total",
]

scenarios = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
    "tlim1.5win0.25",
    "tlim2.0win0.25",
    "tlim3.0win0.25",
    "tlim4.0win0.25",
    "tlim5.0win0.25",
]

quantities = [
    "rates",
    "values",
]

confidences = [
    "medium",
    "low",
]


def _validate(source, scenario, quantity, confidence):
    assert source in sources, f"available sources: {sources}"
    assert scenario in scenarios, f"available scenarios: {scenarios}"
    assert quantity in quantities, f"available quantities: {quantities}"
    assert confidence in confidences, f"available confidences: {confidences}"

def open_slr_global(source, scenario, quantity="values", confidence="medium"):
    _validate(source, scenario, quantity, confidence)
    return xa.open_dataset(Path(AR6_GLOBAL) / f"confidence_output_files/{confidence}_confidence/{scenario}/{source}_{scenario}_{confidence}_confidence_{quantity}.nc")

def open_slr_regional(source, scenario, quantity="values", confidence="medium"):
    _validate(source, scenario, quantity, confidence)
    return xa.open_dataset(Path(AR6_REGIONAL) / f"{confidence}_confidence/{scenario}/{source}_{scenario}_{confidence}_confidence_{quantity}.nc")

def get_ar6_global_numbers(experiment, source, years=None, quantiles=[.05, .5, .95], rate=False):
    # quantity, variable = ("rates", "sea_level_change_rate") if False else ("values", "sea_level_change") # no rates file available
    quantity, variable = ("rates", "sea_level_change_rate") if rate else ("values", "sea_level_change") # no rates file available
    source2 = MAP_AR6.get(source, source)
    with open_slr_global(source2, experiment, quantity=quantity)[variable] as ar6ds:
        vals = ar6ds.loc[quantiles, years]
        # if rate:
        #     vals = vals.diff("years") / vals.years.diff("years")
        return vals


def get_ar6_regional_numbers(experiment, source, IDs, years=None, quantiles=[.05, .5, .95], rate=False):
    quantity, variable = ("rates", "sea_level_change_rate") if rate else ("values", "sea_level_change")
    # quantity, variable = ("rates", "sea_level_change_rate") if False else ("values", "sea_level_change")
    source2 = MAP_AR6.get(source, source)
    with open_slr_regional(source2, experiment, quantity=quantity)[variable] as ar6ds:
        vals = ar6ds.loc[quantiles, years, IDs]
        # if rate:
        #     vals = vals.diff("years") / vals.years.diff("years")
        return vals

def get_ar6_numbers(experiment, source, IDs=None, years=None, quantiles=[.05, .5, .95], rate=False):
    if IDs is None or (isinstance(IDs, int) and IDs == -1) or (isinstance(IDs, list) and (IDs == [None] or IDs == [-1])):
        return get_ar6_global_numbers(experiment, source, years=years, quantiles=quantiles, rate=rate)
    else:
        return get_ar6_regional_numbers(experiment, source, IDs, years=years, quantiles=quantiles, rate=rate)


def load_location(ID, scenario, quantile=0.5, **kwargs):
    """Load a dataframe of sources at one location, for one scenario
    """
    # print("load ID", ID)
    data = {}
    variable = "sea_level_change_rate" if kwargs.get("quantity", "values") == "rates" else "sea_level_change"
    for source in kwargs.pop("sources", sources):
        with (open_slr_global if ID == -1 else open_slr_regional)(source, scenario, **kwargs) as ds:
            data[source] = ds[variable].loc[quantile, :, ID].to_pandas()
    return pd.DataFrame(data).loc[2020:2150]

def load_global(scenario, quantile=0.5, **kwargs):
    srcs = [s for s in kwargs.pop("sources", sources) if s != "verticallandmotion"]
    return load_location(-1, scenario, quantile=quantile, sources=srcs, **kwargs)


def get_locations(tide_gauges_only=False): #ID=None, lon=None, lat=None):
    """Load a dataframe of locations
    """
    # if ID is None and lon is None
    # First come the tige-gauge locations, then the grid with lat values 90, 89, ..., -90 (181) and lon values -180, -179,...,179 (360)
    if tide_gauges_only:
        n = 1030 # 66190 - 181*360 == 1030
    else:
        n = None
    with open_slr_regional("oceandynamics", "ssp585") as ds:
        return pd.DataFrame({"lon":ds.lon[:n].values, "lat": ds.lat[:n].values, "ID": ds.locations[:n].values})


def search_tidegauge(lon, lat, dist=50e3, include_grid=False):
    locs = get_locations(tide_gauges_only=not include_grid)
    distances = np.sqrt(np.deg2rad(locs.lon.values - lon)**2 * np.cos(np.deg2rad(lat))**2 + np.deg2rad(locs.lat.values - lat)**2)*6371e3
    locs["distance (km)"] = distances*1e-3
    return locs.iloc[distances<dist].set_index("distance (km)").sort_index()


def get_grid_location(lon, lat):
    # first determine which location is the lon/lat point
    lats = np.arange(90, -90-1, -1)
    # lons = np.arange(-180, 180)
    lons = np.arange(0, 360) # it is ordered as 0, 1, ..., 180, -179, ..., -1
    ilat = np.searchsorted(lats, lat, sorter=np.arange(lats.size)[::-1])
    ilon = np.searchsorted(lons, lon if lon >=0 else lon+360)

    i = 1030 + ilat*360 + ilon

    locs = get_locations()
    location = locs.iloc[i]

    assert location["lat"] == lats[ilat]
    assert np.mod(location["lon"], 360) == np.mod(lons[ilon], 360)

    ID = location["ID"]
    return ID

def load_grid(lon, lat, scenario, **kwargs):
    """here we look for grid points which are not part of the PSMSL tide gauges locations (as it seems those have a special treatment, e.g. see Venice)
    """
    ID = get_grid_location(lon,lat)

    return load_location(ID, scenario, **kwargs)


def load_ar6_full_samples(source, scenario, rate=False, workflow=None, source2=None):
    """This function attempts to load the full ensemble from the AR6 "full samples" dataset

    It is not straightforward because the dataset is not organized per source, but per workflow,
    and the way workflows are combined to produce the AR6 medium confidence values is not direct.

    In particular, for the AIS different workflows are combined to produce the AR6 medium confidence values,
    and it is done differently for the median and for the range.

    To accomodate for this while retaining a versatile function, we load only one workflow at a time, and return all samples.

    See load_ar6_medium_confidence for a function that loads the AR6 medium confidence values or combines the workflows to produce them.
    """
    if rate:
        suffix = "_rates"
        variable = "sea_level_change_rate"
    else:
        suffix = ""
        variable = "sea_level_change"

    if source == "AIS":
        workflows = ["ismipemu", "FittedISMIP", "bamber", "larmip"]
        if workflow is None:
            workflow = "larmip"
        else:
            assert workflow in workflows, f"available workflows for AIS: {workflows}"
        if source2 is None:
            source2 = "TOT" if workflow == "larmip" else "AIS"

        file = AR6_GLOBAL/f'full_sample_components{suffix}/icesheets-ipccar6-{workflow}icesheet-{scenario}_{source2}_globalsl{suffix}.nc'

    elif source == "GIS":
        workflows = ["ismipemu", "FittedISMIP", "bamber"]
        if workflow is None:
            # workflow = "FittedISMIP"
            workflow = "ismipemu"
        else:
            assert workflow in workflows, f"available workflows for GIS: {workflows}"
        if workflow == "FittedISMIP":
            file = AR6_GLOBAL/f'full_sample_components{suffix}/icesheets-{workflow}-icesheets-{scenario}_{source}_globalsl{suffix}.nc'
        else:
            file = AR6_GLOBAL/f'full_sample_components{suffix}/icesheets-ipccar6-{workflow}icesheet-{scenario}_{source}_globalsl{suffix}.nc'

    elif source == "glaciers":
        file = AR6_GLOBAL/f'full_sample_components{suffix}/glaciers-ipccar6-gmipemuglaciers-{scenario}_globalsl{suffix}.nc'

    elif source == "landwaterstorage":
        file = AR6_GLOBAL/f'full_sample_components{suffix}/landwaterstorage-ssp-landwaterstorage-{scenario}_globalsl{suffix}.nc'

    elif source == "oceandynamics":
        file = AR6_GLOBAL/f'full_sample_components{suffix}/oceandynamics-tlm-oceandynamics-{scenario}_globalsl{suffix}.nc'

    elif source == "total":
        # This underestimates the uncertainty from the medium confidence values, for some reason -> only OK to estimate teh rate I guess
        return ( load_ar6_full_samples("AIS", scenario, rate=rate, workflow=workflow)
                +  load_ar6_full_samples("GIS", scenario, rate=rate)
                +  load_ar6_full_samples("oceandynamics", scenario, rate=rate)
                +  load_ar6_full_samples("glaciers", scenario, rate=rate)
                +  load_ar6_full_samples("landwaterstorage", scenario, rate=rate)
                )

    else:
        ar6_sources = ["AIS", "GIS", "glaciers", "landwaterstorage", "oceandynamics"]
        raise ValueError(f"source {source} not supported. Available sources: {ar6_sources}")


    with xa.open_dataset(file) as ds:
        return ds[variable].load()


def load_ar6_medium_confidence(source, scenario, quantiles, rate=False, from_samples=False):

    if rate:
        if not from_samples:
            logger.warning("rate not provided in medium confidence files. Take the diff. You may also try from samples = True to load from components")

    if not from_samples:
        if rate:
            res = load_ar6_medium_confidence(source, scenario, quantiles, rate=False, from_samples=False)
            res2 = res.diff("years")/res.years.diff("years")
            return xa.concat([res.isel(years=0) / (res.years.values[0] - 2005), res2], dim="years")

        with open_slr_global(source, scenario, quantity="rates" if rate else "values", confidence="medium") as ds:
            return ds["sea_level_change_rate" if rate else "sea_level_change"].sel(quantiles=quantiles).rename(quantiles="quantile")

    if source in ("AIS", "Total"):
        qvalues = load_ar6_full_samples(source, scenario, rate=rate, workflow="larmip").quantile(quantiles, dim="samples")
        qvalues2 = load_ar6_full_samples(source, scenario, rate=rate, workflow="ismipemu").quantile(quantiles, dim="samples").reindex(years=qvalues.years)

        # NOTE below from trial and error. It works well for q < .3 but is (increasingly) underestimated in [.3, .5]
        median = qvalues.sel(quantile=.5)*.5 + qvalues2.sel(quantile=.5)*.5
        lower = qvalues.sel(quantile=quantiles) - qvalues.sel(quantile=.5) + median
        return xa.concat([
            lower.sel(quantile=q) if q <= .5 else qvalues.sel(quantile=q) for q in quantiles], dim="quantile")

    else:
        return load_ar6_full_samples(source, scenario, rate=rate).quantile(quantiles, dim="samples")
