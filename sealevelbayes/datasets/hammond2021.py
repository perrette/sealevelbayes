""" http://geodesy.unr.edu/vlm.php
"""
import os
from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd

from sealevelbayes.logs import logger
from sealevelbayes.datasets.manager import get_datapath

datapath = get_datapath("VLM")

def load_tidegauge_rates():

    lines = open(datapath/"GPS_VLM_at_TideGauges_LongTable.txt").read().splitlines()
    quality = load_quality_flags()

    # also load the stations
    stations_df = read_midas()
    stations_df.set_index("ID", inplace=True)

    tgs = []
    labels = ["lon", "lat", "vu", "svu", "eta", "zeta", "nu"]
    gpslabels = ["vu", "vu_filtered", "svu", "delta", "eta", "nu", "weight", "name"]

    for i, l in enumerate(lines):
        if l.strip() == "":
            tgs.append(record)
            del record

        elif l.startswith('>>'):
            record = {
                "ID": int(l.split()[1]),
                "name": " ".join(l.split()[2:]),
            }
            record["quality"] = quality.loc[record["ID"]].quality

        elif l.startswith('##'):
            data = l.split()[1:]
            for k,v in zip(labels, data):
                record[k] = float(v)
            record['gps'] = []

        else:
            data = l.strip().split()
            assert len(data) == len(gpslabels)
            gps = {}
            name = data[-1]

            _fields = ["lon", "lat", "year_start", "year_end", "length"]
            _get_info = lambda s: {f: s[f] for f in _fields}

            try:
                gps.update(_get_info(stations_df.loc[name]))

            except KeyError:
                if '-' not in name:
                    logger.warning(f"GPS station {name}: not found for {record['ID']}:{record['name']}")
                    gps.update({f:np.nan for f in _fields})

                else:
                    # print("composed station", name)
                    infos = []
                    for nn in name.split('-'):
                        try:
                            item = stations_df.loc[nn]
                        except KeyError:
                            logger.warning(f"GPS station {nn}: not found for {record['ID']}:{record['name']}")
                            continue
                        infos.append(_get_info(item))

                    if len(infos) == 0:
                        gps.update({f:np.nan for f in _fields})

                    else:
                        gps["lon"] = np.mean([info['lon'] for info in infos])
                        gps["lat"] = np.mean([info['lat'] for info in infos])
                        gps["year_end"] = np.max([info['year_end'] for info in infos])
                        gps["year_start"] = np.min([info['year_start'] for info in infos])
                        gps["length"] = np.sum([info['length'] for info in infos])

            for k,v in zip(gpslabels, data):
                gps[k] = float(v) if k != "name" else v

            record["gps"].append(gps)

    # also add year start and end
    for record in tgs:
        record["year_start"], record["year_end"] = _get_gps_period_at_psmsl(record)

    return tgs


def _parse_quality_flags():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(open(datapath/"table.html"), "html.parser")

    # Below all from Chat GTP !
    # Refine and focus on the table structure to extract only key information

    # Collecting tide gauge information more directly by narrowing the scope
    tide_gauges = []

    # Each tide gauge entry starts with a row containing the link to the tide gauge, followed by VLM quality
    rows = soup.find_all("tr")
    for row in rows:
        # Identify rows containing Tide Gauge details
        link = row.find("a", href=True)
        if link and "Tide Gauge" in link.text:
            # Extract Tide Gauge ID
            tide_gauge_id = link.text.split(":")[1].split()[0].strip()
            if tide_gauge_id == '"Number"':
                continue

            # Look for VLM quality in the same row or subsequent rows
            quality = row.find_next("h3", style=True)
            if quality and "VLM quality is" in quality.text:
                vlm_quality = quality.text.split("VLM quality is")[-1].strip()
                tide_gauges.append((int(tide_gauge_id), vlm_quality))
    return tide_gauges

def load_quality_flags():
    fn = datapath / "table.csv"
    if not fn.exists():
        logger.info("Parse quality flags from HTML")
        tide_gauges = _parse_quality_flags()
        logger.info("Save quality flags to CSV")
        pd.DataFrame(tide_gauges, columns=["ID", "quality"]).to_csv(fn, index=False)
    logger.info(f"Load quality flags from {fn}")
    return pd.read_csv(fn).set_index("ID")


def open_interpolated():
    return xa.open_dataset(datapath / "VLM_Global_Imaged.nc")


def _get_gps_period_at_psmsl(record):
    # add start and end years
    year_starts = np.array([gps["year_start"] for gps in record["gps"]])
    year_ends = np.array([gps["year_end"] for gps in record["gps"]])
    weights = np.array([gps["weight"] for gps in record["gps"]])
    year_start = weighted_median(year_starts, weights)
    year_end = weighted_median(year_ends, weights)
    return year_start, year_end


def update_rates(rates, method="roughness-no-filtering", interpolate_median=True, gauge_dist_tol=.1, verbose=False, copy=False):
# def update_rates(rates, method="original_but_nearest", interpolate_median=True, gauge_dist_tol=.1, verbose=False, copy=False):

    if copy:
        import copy
        rates = [copy.deepcopy(r) for r in rates]

    for rate in rates:

        # make a backup
        if "vu_orig" not in rate:
            rate["vu_orig"] = rate["vu"]
            rate["svu_orig"] = rate["svu"]

        # Reset the numbers to Hammond et al 2021, regardless of possible colocation
        if method == "reset":
            rate["vu"] = rate["vu_orig"]
            rate["svu"] = rate["svu_orig"]
            continue

        weights = np.array([r["weight"] for r in rate["gps"]])
        assert np.abs(weights.sum() - 1) < 0.003, repr((rate, weights.sum()))
        # assert np.abs((weights**2).sum() - 1) < 0.95, repr((rate, (weights**2).sum()))
        rate["svu_median"] = weighted_median([r["svu"]**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5

        if any(r["delta"] <= gauge_dist_tol for r in rate["gps"]):
            if verbose:
                print(rate["ID"], rate["name"], len([r for r in rate["gps"] if r["delta"] <= gauge_dist_tol]), "stations less than", gauge_dist_tol,"km from tide-gauge. Take closest.")
            i = np.argmin([r["delta"] for r in rate["gps"]])
            rate["vu"] = rate["gps"][i]["vu"]
            rate["svu"] = rate["gps"][i]["svu"]
            rate["svu_smooth"] = rate["gps"][i]["svu"]
            rate["svu_local"] = 0
            rate["colocated"] = True
            continue

        else:
            rate["colocated"] = False

        if method in ["original", "original_but_nearest"]:
            rate["vu"] = rate["vu_orig"]
            rate["svu"] = rate["svu_orig"]

        elif method in ["original-fixed"]:
            # rate["vu"] = rate["vu_orig"]
            rate["vu"] = weighted_median([r["vu"] for r in rate["gps"]], weights, interpolate=interpolate_median)
            rate["svu"] = np.sum(np.array([r["svu"]**2 for r in rate["gps"]])*weights)**.5

        # Use the median of stations rate minus filtered rate, which should be representative of "geophysical error"
        elif method in ["roughness"]:
            rate["vu"] = rate["vu_orig"]
            rate["svu"] = weighted_median([r['svu']**2 + (r["vu_filtered"]-r["vu"])**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5
            rate["svu_smooth"] = weighted_median([r['svu']**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5
            rate["svu_local"] = weighted_median([(r["vu_filtered"]-r["vu"])**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5

        # Use the median of stations rate minus filtered rate, which should be representative of "geophysical error"
        elif method in ["roughness-no-filtering"]:
            rate["vu"] = weighted_median([r['vu'] for r in rate["gps"]], weights, interpolate=interpolate_median)
            rate["svu"] = weighted_median([r['svu']**2 + (r["vu_filtered"]-r["vu"])**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5
            rate["svu_smooth"] = weighted_median([r['svu']**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5
            rate["svu_local"] = weighted_median([(r["vu_filtered"]-r["vu"])**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5

        # Same with weighted mean
        elif method in ["roughness-wmean"]:
            rate["vu"] = rate["vu_orig"]
            rate["svu"] = (np.array([r['svu']**2 + (r["vu_filtered"]-r["vu"])**2 for r in rate["gps"]])*weights).sum()**.5

        elif method == "check":
            rate["vu"] = weighted_median([r["vu_filtered"] for r in rate["gps"]], weights, interpolate=False)
            rate["svu"] = np.sum((np.array([r["svu"] for r in rate["gps"]])*weights)**2)**.5

        elif method == "wmean":
            # Here we use weighted mean (unfiltered) based on the weights calculated by Hammond
            # ...we only use unfiltered because filtered already use the median operator
            rate["vu"] = np.sum(np.array([r["vu"] for r in rate["gps"]])*weights)
            rate["svu"] = np.sum(np.array([r["svu"]**2 for r in rate["gps"]])*weights)**.5

        elif method == "wmedian":
            # ... image of error, assuming a true station will have a time error comparable to the error of surrounding stations
            rate["svu"] = weighted_median([r["svu"]**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5
            rate["vu"] = weighted_median([r["vu"] for r in rate["gps"]], weights, interpolate=interpolate_median)

        elif method == "wmedian+scatter":
            # same as wmedian but including scatter error
            rate["vu"] = weighted_median([r["vu"] for r in rate["gps"]], weights, interpolate=interpolate_median)
            svu_wmedian = weighted_median([r["svu"]**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5
            scatter = weighted_median([np.abs(r["vu"]-rate["vu"]) for r in rate["gps"]], weights, interpolate=interpolate_median)
            rate["svu"] = (svu_wmedian**2 + scatter**2)**.5

        elif method == "wmedian_filtered":
            # ... image of error, assuming a true station will have a time error comparable to the error of surrounding stations
            rate["svu"] = weighted_median([r["svu"]**2 for r in rate["gps"]], weights, interpolate=interpolate_median)**.5
            rate["vu"] = weighted_median([r["vu_filtered"] for r in rate["gps"]], weights, interpolate=interpolate_median)

        elif method == "nearest":
            # ... image of error, assuming a true station will have a time error comparable to the error of surrounding stations
            i = np.argmin([r["delta"] for r in rate["gps"]])
            rate["svu"] = rate["gps"][i]["svu"]
            rate["vu"] = rate["gps"][i]["vu"]

        elif method == "no-interp":
            # if we make it here it means we have no valid stations
            rate["svu"] = np.inf
            rate["vu"] = np.nan

        else:
            raise NotImplementedError(method)

    # return results (useful only if copy is True)
    return rates


def weighted_quantiles(values, weights, quantiles=0.5, interpolate=False):
    values = np.asarray(values)
    weights = np.asarray(weights)
    i = np.argsort(values)
    sorted_weights = weights[i]
    sorted_values = values[i]
    Sn = np.cumsum(sorted_weights)

    if interpolate:
        Pn = (Sn - sorted_weights/2 ) / Sn[-1]
        return np.interp(quantiles, Pn, sorted_values)
    else:
        return sorted_values[np.searchsorted(Sn, np.asarray(quantiles) * Sn[-1])]


def weighted_mean(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)
    return np.sum(values*weights)/np.sum(weights)


def weighted_median(values, weights, **kw):
    return weighted_quantiles(values, weights, quantiles=0.5, **kw)

def mad(values):
    values = np.array(values)
    return np.median(np.abs(values-np.median(values)))

def weighted_mad(values, weights, **kw):
    values = np.array(values)
    weights = np.array(weights)
    return weighted_median(np.abs(values - weighted_median(values, weights, **kw)), weights, **kw)




def fetch_gps_station(path):
    import requests
    URL = "http://geodesy.unr.edu/"+str(path)
    print("Download", URL)
    response = requests.get(URL)
    return response.content.decode()

def _strip__(name):
    while name.startswith("_"):
        name = name[1:]
    return name


def load_gps_station(name):
    relpath = Path(f"gps_timeseries/tenv3/IGS14/{name}.tenv3")
    path = datapath/relpath

    if not path.exists():
        response_content = fetch_gps_station(relpath)
        os.makedirs(path.parent, exist_ok=True)
        open(path, "w").write(response_content)

    df = pd.read_csv(path, sep=r'\s+')
    df.columns = [_strip__(nm) for nm in df.columns]
    return df

import datetime
from itertools import groupby

_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

def _get_dates(df):
    for yymmmdd, decimal_year in zip(df['YYMMMDD'], df['yyyy.yyyy']):
        year = int(yymmmdd[:2])
        # assert year >= 0 and year <= 23, year
        if year < 50:
            year = 2000 + year
        else:
            year = 1900 + year
        assert year == int(decimal_year), (yyymmmdd, year, decimal_year)
        yield datetime.date(year, _months.index(yymmmdd[2:5])+1, int(yymmmdd[5:7]))


def monthly_gps(df, field='up(m)', pad_with_nans=True, func=np.median):

    dates = []
    values = []

    for m, group in groupby(zip(_get_dates(df), df[field]), key=lambda d:d[0].month):
        group = list(group)
        date = datetime.date(group[0][0].year, m, 15)
        if len(dates) > 0:
            while (date - dates[-1]).days > 30+15: # more than a month has passed !
                missing_date_approx = dates[-1] + datetime.timedelta(days=30)
                missing_date = datetime.date(missing_date_approx.year, missing_date_approx.month, 15)  # make sure we move month by month
                dates.append(missing_date)
                values.append(np.nan)
        dates.append(date)
        values.append(func([v[1] for v in group]))

    return np.array(dates), np.array(values)


def _filter_monthly(values, outliers_to_mad=5, inplace=False):
    if not inplace:
        values = values.copy()
    rates = np.diff(values)
    median = np.nanmedian(rates)
    dev = np.abs(rates - median)
    mad = np.nanmedian(dev)
    outliers = dev > outliers_to_mad*mad
    print(outliers.sum(), 'outliers')
    for i in np.where(outliers)[0]:
        values[i+1:] -= rates[i]  # simply cancel the jump
        # values[i] = np.nan # create a NaN value

    return values


def _yearly_gps(dates, values, min_valid_months=6):
    years, rates = _yearly_gps_rates(dates, values, min_valid_months)
    return years, _cumul_gps(rates)


def _yearly_gps_rates(dates, values, min_valid_months=6):
    previous_year = None
    rates = []
    years = []
    for y, group in groupby(zip(dates, values), key=lambda d:d[0].year):
        if previous_year is None:
            months, start = zip(*[(d.month, v) for d, v in group])
            previous_year = np.array([np.nan]*(months[0]-1) + list(start))  # pad with nans
            continue

        months, current = zip(*[(d.month, v) for d, v in group])
        if len(months) < 12:
            assert y == dates[-1].year, months   # last year
            current = list(current) + [np.nan]*(12 - months[-1])  # pad with nans
        current_year = np.array(current)

        rate = current_year - previous_year
        valid = np.isfinite(rate)

        rates.append(np.median(rate[valid]) if valid.sum() >= min_valid_months else np.nan) # median of month to month difference if more than 6 valid months
        years.append(y)

        previous_year = current_year

    return np.array(years), np.array(rates)


def _cumul_gps(rates):
    vals = np.nancumsum(rates)
    vals[np.isnan(rates)] = np.nan
    return vals



def read_midas(file=datapath / "midas/midas.IGS14.txt.2023"):
    """This is the GPS rate at individual source stations, prior to imaging.

    Reference:
    ---------
    Blewit et al (2016), JGR
        doi: 10.1002/2015JB012552
        url: https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015JB012552

    Source:
    ------

    File downloaded at http://geodesy.unr.edu/velocities/midas.IGS14.txt

    README: http://geodesy.unr.edu/velocities/midas.readme.txt

    column 1 - 4 character station ID
    column 2 - MIDAS version label
    column 3 - time series first epoch, in decimal year format.
    column 4 - time series last epoch, in decimal year format (See http://geodesy.unr.edu/NGLStationPages/decyr.txt for translation to YYMMMDD format).
    column 5 - time series duration (years).
    column 6 - number of epochs of data, used or not
    column 7 - number of epochs of good data, i.e. used in at least one velocity sample
    column 8 - number of velocity sample pairs used to estimate midas velocity
    column 9-11 - east, north, up mode velocities (m/yr)
    column 12-14 - east, north, up mode velocity uncertainties (m/yr)
    column 15-17 - east, north, up offset at at first epoch (m)
    column 18-20 - east, north, up fraction of outliers
    colums 21-23 - east, north, up standard deviation velocity pairs
    column 24 - number of steps assumed, determined from our steps database
    column 25-27 - latitude (degrees), longitude (degrees) and height (m) of station.
    """
    # stations_df = pd.read_csv(datapath/"midas.IGS14.txt", sep=r"\s+", header=None)
    # # df.columns = ["name", 1, ]
    # columns = list(stations_df.columns)
    # columns[2] = "year_start"
    # columns[3] = "year_end"
    # columns[4] = "length"
    # columns[24] = "lat"
    # columns[25] = "lon"
    # stations_df.columns = columns
    # stations_df.set_index(0, inplace=True)

    midas = pd.read_csv(file, sep=r"\s+", header=None)
    midas.columns = [
    "ID",
    "MIDAS version",
    "year_start",
    "year_end",
    "length",
    "#epochs of data",
    "#epochs of good data",
    "#velocity sample pairs",
    "vx", "vy", "vz",
    "vx_err", "vy_err", "vz_err",
    "offset_x", "offset_y", "offset_z",
    "outliers_fraction_x", "outliers_fraction_y", "outliers_fraction_z",
    "vx_pair_sd", "vy_pair_sd", "vz_pair_sd",
    "steps",
    "lat", "lon", "height",
    ]
    # midas["lon"] += 360
    ix = midas['lon'].values <= -180
    midas['lon'].values[ix] = midas['lon'].values[ix] + 360

    return midas


def get_gps_sampling_period(psmsl_ids):
    """Get the GPS sampling period for a given list of PSMSL IDs.

    Parameters:
        psmsl_ids (list): A list of PSMSL IDs

    Returns:
        periods (list of (year_start, year_end)): The GPS sampling period

    We interpolate (weighted median) the GPS mid-year and start-to-end length of the GPS record
    onto the PSMSL locations.
    """
    records = load_tidegauge_rates()
    periods = {r['ID']: (r['year_start'], r['year_end']) for r in records}
    return [periods.get(psmsl_id) for psmsl_id in psmsl_ids]