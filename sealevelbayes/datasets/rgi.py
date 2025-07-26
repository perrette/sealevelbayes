"""Randolf Glacier Inventory dataset module.
"""
import glob
import os
import pandas as pd

from sealevelbayes.datasets.manager import require_dataset

rgi_regions = {
    1: "Alaska",
    2: "Western Canada and USA",
    3: "Arctic Canada North",
    4: "Arctic Canada South",
    5: "Greenland Periphery",
    6: "Iceland",
    7: "Svalbard",
    8: "Scandinavia",
    9: "Russian Arctic",
    10: "North Asia",
    11: "Central Europe",
    12: "Caucasus and Middle East",
    13: "Central Asia",
    14: "South Asia West",
    15: "South Asia East",
    16: "Low Latitudes",
    17: "Southern Andes",
    18: "New Zealand",
    19: "Antarctic and Subantarctic"
}

def load_glaciers_metadata(regions=None, version="7.0", rename=True):

    kwargs = {}
    rename = {}
    if version == "7.0":
        datapath = require_dataset("nsidc0770_rgi_v7/global_files/RGI2000-v7.0-G-global")
        files = {int(os.path.basename(file)[15:15+2]): file for file in sorted(glob.glob(f"{datapath}/RGI2000-v7.0-G-*/RGI2000-v7.0-G-*-attributes.csv"))}
    elif version == "5.0":
        datapath = require_dataset("nsidc0770_rgi_v5/nsidc0770_00.rgi50.attribs")
        files = {int(os.path.basename(file)[:2]): file for file in sorted(glob.glob(f"{datapath}/*.csv"))}
        kwargs = {"encoding":'latin1'}
        rename = {"Area": "area_km2", "O1Region": "o1region"}
    else:
        raise ValueError(f"Version {version} not supported")

    # ! ls -lh {datapath}/RGI2000-v7.0-G-01_alaska/
    dfs = []
    for region, file in files.items():
        if regions is not None and region not in regions:
            continue
        # print(file)
        df = pd.read_csv(file, **kwargs)
        dfs.append(df)

    return pd.concat(dfs).rename(columns=rename)


def check_fraction_small_glaciers_per_region(rgi, thres = 2, exponent = 1, regions=None):
    """Check the fraction of small glaciers in each region.

    Parameters
    ----------
    rgi : pd.DataFrame
        The RGI DataFrame
    thres : float or tuple
        The threshold for small glaciers. If a tuple, the threshold is in the form (min, max).
    exponent : float, optional
        The exponent for the area calculation (e.g. to convert to a volume)
    """
    if regions is None:
        regions = sorted(rgi["o1region"].unique())
    if isinstance(thres, tuple):
        small_glaciers = rgi[(rgi["area_km2"] > thres[0]) & (rgi["area_km2"] < thres[1])]
    else:
        small_glaciers = rgi[rgi["area_km2"] < thres]
    # exponent = 1 # surface to volume exponent
    total_area_small = (small_glaciers["area_km2"]**exponent).sum()
    fractions = []
    # print(f"Fraction of small glaciers < {thres} km2")
    for region in regions:
        regional_area_small = (small_glaciers[small_glaciers["o1region"] == region]["area_km2"]**exponent).sum()
        fraction = regional_area_small/total_area_small
        fractions.append(fraction)
        # print(f'{region} : {fraction*100:.1f}%')
    return fractions
