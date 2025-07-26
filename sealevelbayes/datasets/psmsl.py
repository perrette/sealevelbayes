"""Load PSMSL data
"""
import numpy as np
import pandas as pd
from sealevelbayes.config import logger
from sealevelbayes.datasets.manager import get_datapath, require_dataset_by_name
from sealevelbayes.datasets.catalogue import download_met_monthly, download_rlr_monthly, download_rlr_annual

psmslroot = get_datapath('psmsl')
annual = psmslroot / 'rlr_annual'
rlr_monthly = psmslroot / 'rlr_monthly'
metric_monthly = psmslroot / 'met_monthly'


def _load_rlr_df(path):
    df = pd.read_csv(path, sep=';', na_values=-99999, header=None)
    df.columns = ['year', 'value', 'flag', 'missing days']
    df.set_index('year', inplace=True)
    df['flag'] = df['flag'] == 'Y'
    df['value'] -= 7000
    return df

def _load_rlr(path, name=None, remove_flagged=False):
    df = _load_rlr_df(path)
    if remove_flagged:
        df['value'][df['flag']] = np.nan
    s = df['value']
    s.name = name
    return s

def is_rlr(id):
    return (annual / f"data/{id}.rlrdata").exists()


def load_filelist_rlr():
    require_dataset_by_name("psmsl/rlr_annual")
    df = pd.read_csv(annual / "filelist.txt", sep=";", header=None, skipinitialspace=True)
    df.columns = ["ID", "latitude", "longitude", "station name", "coastline code", "station code", "qcflag"]
    df['station name'] = df['station name'].str.strip()
    return df

def load_filelist_all():
    require_dataset_by_name("psmsl/met_monthly")
    df = pd.read_csv(metric_monthly / "filelist.txt", sep=";", header=None, skipinitialspace=True)
    df.columns = ["ID", "latitude", "longitude", "station name", "coastline code", "station code", "qcflag"]
    df['station name'] = df['station name'].str.strip()
    return df

def load_filelist_rlr_html():
    """from the website https://psmsl.org/data/obtaining/index.php using pandas' read_html"""
    return pd.read_csv(psmslroot / f"station_list_psmsl_20230620.csv", skipinitialspace=True)

def load_rlr(id, **kw):
    path = annual / f"data/{id}.rlrdata"
    return _load_rlr(path, name=id, **kw)


# def fetch_rlr(id, **kw):
#     url = f"https://psmsl.org/data/obtaining/rlr.annual.data/{id}.rlrdata"
#     with urllib.request.urlopen(url) as response:
#        urlData = response.read()
#     return _load_rlr(io.StringIO(urlData.decode()), name=id)



def load_all_psmsl_rlr():

    filepath = get_datapath('psmsl_timeseries_full.csv')
    if filepath.exists():
        logger.debug(f'Read from {filepath}')
        df = pd.read_csv(filepath, skipinitialspace=True).set_index('year')
        df.columns = df.columns.astype(int)
        return df

    df = load_filelist_rlr()

    timeseries = []

    for r in df.to_dict('records'):
        try:
            series = load_rlr(r['ID'])
        except Exception as error:
            logger.error(str(error))
            logger.warning(f"error loading PSMSL RLR: {r['ID']}")
            continue

        timeseries.append(series)

    timeseries = pd.DataFrame(timeseries).T

    logger.info(f'Write to {filepath}')
    timeseries.to_csv(filepath)

    return timeseries