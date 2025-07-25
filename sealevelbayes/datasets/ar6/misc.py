import pandas as pd
from pathlib import Path
from sealevelbayes.datasets.manager import get_datapath

def load_temperature():
    """Load median temperature scenarios from AR6 (ssp19,ssp26,ssp45,ssp70,ssp85) merged with historical scenarios.

    See data folder: AR6_Projections/SurfaceTemperatureGlobal for sources.
    (Data obtained from AR6 supplementary material and online repos.)

    Past data is obtained from SPM1_1850-2020_obs.csv (kind of smooth) and tas_global_Historical.csv (normally smooth but starting only in 1950)
    The merged time-series start in 1855.

    Recipe: notebooks/topics/temperature-hist-rcp-ssp-ar5-ar6.ipynb
    """
    return pd.read_csv(get_datapath("savedwork/tas_ar6_merged.csv")).set_index("year")