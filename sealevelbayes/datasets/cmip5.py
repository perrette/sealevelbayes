"""CMIP5 Tas files from the KNMI Climate Explorer
"""
import re
import pandas as pd
from sealevelbayes.datasets import get_datapath

regex = re.compile(r"global_tas_Amon_(?P<model>[\w-]+)_(?P<experiment>\w+?)_(?P<ensemble_member>\w+?)\.dat")
# regex_strict = re.compile(r"global_tas_Amon_(?P<model>[\w-]+)_(?P<experiment>\w+?)_(?P<ensemble_member>r\d+i\d+p\d+)\.dat")

def scan_cmip5_tglobal_files():
    all_files = sorted(get_datapath("climexp.knmi.nl/CMIP5/Tglobal").glob("global_tas_Amon_*.dat"))
    return [{**regex.match(f.name).groupdict(), **{"filepath" : str(f)}} for f in all_files]

def read_rcps_as_dataframe(scanned=None):

    if scanned is None:
        scanned = scan_cmip5_tglobal_files()

    records = []
    columns = []
    for r in scanned:
        if r['experiment'] == 'historicalNat':
            continue
        df = pd.read_csv(r['filepath'], sep='\s+', header=None, index_col=0, comment='#',
                        names=['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        records.append(df.mean(axis=1) - 273.15)
        columns.append((r['model'], r['experiment'], r['ensemble_member']))

    return pd.DataFrame(records, columns=df.index, index=pd.MultiIndex.from_tuples(columns, names=['model', 'experiment', 'ensemble_member']))

def write_cmip5_tglobal_df(df=None):
    if df is None:
        df = read_rcps_as_dataframe()
    df.to_csv(get_datapath('savedwork/cmip5_tglobal.csv'))

def read_cmip5_tglobal_df():
    df = pd.read_csv(get_datapath('savedwork/cmip5_tglobal.csv'), index_col=["model", "experiment", "ensemble_member"]).T
    df.index = df.index.astype(int)
    return df
    # return pd.read_csv(get_datapath('savedwork/cmip5_tglobal.csv')).T.set_index('year')